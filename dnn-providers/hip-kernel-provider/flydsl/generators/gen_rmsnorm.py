# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Compile the checked-in RMSNorm code objects for one architecture.

    python -m generators.gen_rmsnorm --arch gfx1151

Writes ``kernels/<arch>/rmsnorm/<instance>.hsaco`` for every row of
``_instances.rmsnorm_instances()``, plus that op's ``manifest.json`` and a
refreshed per-arch ``SOURCE.md``.

Compilation needs no matching GPU. Under ``COMPILE_ONLY=1`` the launcher traces
and lowers but never dispatches, so a gfx942 object builds on an RDNA laptop --
which is what makes the CDNA work in ``PORTING.md`` a *validation* handoff
rather than a generation one.

**Every object is verified against its own AMDGPU metadata before it is
written.** The increment-1 gate established that FlyDSL emits no argument
*names* in that note (HIP ``extern "C"`` kernels do not; the AITER ASM objects
do), so ``requireSignatureMatch`` at load time can catch a kind/size/offset
change but cannot catch an operand *permutation* -- two pointers of the same
size swapped look identical to it. The compensation is here, at the point where
we still know what we asked for: assert the full ``(kind, size, offset)``
sequence against the layout recorded below, and carry the semantic names
forward into the descriptor metadata so the permutation question is answerable
by reading the manifest. Emitting ``.name`` from FlyDSL is the real fix and is
upstream's to make; this keeps it off the critical path.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from . import _flydsl_env as env
from ._extract_hsaco import hsaco_from_dump
from ._instances import RMSNORM_SMALL_N_THRESHOLD, Instance, instances_for
from ._manifest import refresh_source_md, sha256, write_op_manifest

OP = "rmsnorm"

# The traced tensor extents. Both dimensions lower to `?` in the kernel's layout
# (`!fly.layout<(?,?):(?{i64},1)>`), so these shapes are not baked into the
# object -- they exist only to give the tracer something with a dtype and a rank.
# `N` still has to be plausible for the instance: the *builder* reads the `N`
# argument, not the tensor, but a mismatch would make a debug IR dump misleading.
_TRACE_M = 4
_GENERIC_TRACE_N = 4096

# The kernarg layout every instance of this family must have. Four (pointer,
# descriptor) pairs for `rmsnorm_kernel(Input, Gamma, Rstd, Output)`; the 16/4
# descriptor sizes are 2D tensors vs 1D ones. Recorded from the gate run and
# byte-identical across both objects it examined.
_KERNARG_SEGMENT_SIZE = 80
_EXPECTED_SIGNATURE = (
    ("global_buffer", 8, 0),
    ("by_value", 16, 8),
    ("global_buffer", 8, 24),
    ("by_value", 4, 32),
    ("global_buffer", 8, 40),
    ("by_value", 4, 48),
    ("global_buffer", 8, 56),
    ("by_value", 16, 64),
)
# What the slots above mean, in order. `Rstd` aliases `Gamma` whenever
# `store_rstd=False` -- upstream passes Gamma twice to fill the unused slot -- so
# slots 4/5 are live kernargs pointing at the weight tensor, not at an output.
_SEMANTIC_ARG_NAMES = (
    "Input.ptr",
    "Input.desc",
    "Gamma.ptr",
    "Gamma.desc",
    "Rstd.ptr",
    "Rstd.desc",
    "Output.ptr",
    "Output.desc",
)

# The launch geometry the native dispatch handler has to reproduce. One block per
# row, `BLOCK_THREADS` wide. Recorded per instance so the handler reads it rather
# than hardcoding a formula that silently stops matching when a second family
# (e.g. upstream's large-M/small-N builder) is added.
_GRID_RULE = "one_block_per_row"


class GeneratorError(RuntimeError):
    """An instance could not be built, or the object built is not the one asked for."""


def _hkp_pack_module():
    """Import ``hkp_pack.kernel_signature`` from the sibling packaging tree.

    We reach for two of its helpers rather than re-deriving them: ``amdgcn_object``
    unwraps the clang offload bundle by *selecting on the ``-amdgcn-`` triple*
    instead of by position, and ``_metadata_document`` walks the ELF's SHT_NOTE
    sections for ``NT_AMDGPU_METADATA``. A second copy of either in this
    directory would be a second thing to keep correct, and the packer this
    generator feeds parses the very same note.
    """
    packaging = env.PROVIDER_DIR.parent / "descriptor-packaging" / "python"
    if not (packaging / "hkp_pack").is_dir():
        raise GeneratorError(f"hkp_pack not found under {packaging}")
    if str(packaging) not in sys.path:
        sys.path.insert(0, str(packaging))
    from hkp_pack import kernel_signature as module  # noqa: PLC0415

    return module


def _describe(blob: bytes, where: str) -> dict:
    """The AMDGPU metadata facts we verify and record, for one code object."""
    hkp = _hkp_pack_module()
    # `_metadata_document` is private to hkp_pack, but `kernel_signature()` -- its
    # public entry point -- returns only the argument list, and we also need the
    # kernarg segment size and the target triple to verify the object at all.
    document = hkp._metadata_document(hkp.amdgcn_object(blob, where), where)

    kernels = document.get("amdhsa.kernels") or []
    if len(kernels) != 1:
        names = ", ".join(str(k.get(".name")) for k in kernels)
        raise GeneratorError(
            f"{where}: expected exactly one kernel in the object, found "
            f"{len(kernels)} [{names}]. The packer keys on (toc_key, symbol), so a "
            "multi-kernel object needs a TOC entry per symbol, not one per file."
        )
    kernel = kernels[0]

    args = [
        arg
        for arg in (kernel.get(".args") or [])
        if not str(arg.get(".value_kind", "")).startswith(hkp._HIDDEN_KIND_PREFIX)
    ]
    return {
        "symbol": kernel.get(".name"),
        "kernarg_segment_size": kernel.get(".kernarg_segment_size"),
        "targets": document.get("amdhsa.target"),
        "signature": tuple(
            (arg[".value_kind"], arg[".size"], arg[".offset"]) for arg in args
        ),
        "named_args": sum(1 for arg in args if arg.get(".name") is not None),
    }


def _target_arch(targets, where: str) -> str:
    """The gfx name out of ``amdhsa.target``, ignoring feature suffixes.

    Two spellings of the same triple are in circulation and both have to parse:
    the AITER ASM objects carry an *empty* environment field,
    ``amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-``, and FlyDSL emits it as
    ``unknown``, ``amdgcn-amd-amdhsa-unknown-gfx1151``. Splitting on ``--`` reads
    only the first. So: drop the ``:feature`` suffixes, then take the last
    ``-``-separated field. The suffixes come off *first* because a feature can
    itself end in a dash (``xnack-``), which would otherwise swallow the arch.

    We compare the arch alone -- feature flags describe how the object was built,
    while the filename claims which device it runs on.
    """
    if isinstance(targets, (list, tuple)):
        if len(targets) != 1:
            raise GeneratorError(f"{where}: expected one target, got {targets!r}")
        target = targets[0]
    else:
        target = targets
    if not isinstance(target, str):
        raise GeneratorError(f"{where}: unreadable amdhsa.target {target!r}")
    arch = target.split(":", 1)[0].rsplit("-", 1)[-1]
    if not arch.startswith("gfx"):
        raise GeneratorError(
            f"{where}: amdhsa.target {target!r} does not end in a gfx name"
        )
    return arch


def _verify(described: dict, instance: Instance, arch: str, where: str) -> None:
    """Fail before writing, rather than shipping a mislabelled object."""
    built_for = _target_arch(described["targets"], where)
    if built_for != arch:
        raise GeneratorError(
            f"{where}: object targets {built_for!r} but is being filed under "
            f"{arch!r}. FlyDSL reads ARCH from the environment at each compile, so "
            "this means the env moved under the run -- not that the file is "
            "misnamed. Regenerate one arch per invocation."
        )

    if described["signature"] != _EXPECTED_SIGNATURE:
        raise GeneratorError(
            f"{where}: kernarg layout for {instance.name!r} is\n"
            f"  {described['signature']}\nbut this family is declared as\n"
            f"  {_EXPECTED_SIGNATURE}\n"
            "Either the kernel's parameter list changed, or this instance builds a "
            "different kernel than the rest of the family. The second is the one to "
            "check first: build_rmsnorm_module dispatches N <= "
            f"{RMSNORM_SMALL_N_THRESHOLD} to a separate builder with its own ABI."
        )
    if described["kernarg_segment_size"] != _KERNARG_SEGMENT_SIZE:
        raise GeneratorError(
            f"{where}: kernarg segment is {described['kernarg_segment_size']} bytes, "
            f"expected {_KERNARG_SEGMENT_SIZE}"
        )


def _assert_threshold_mirror(kernel_module) -> None:
    """The instance table mirrors SMALL_N_THRESHOLD; confirm the mirror is true.

    `_instances.py` must stay importable without FlyDSL installed, so it cannot
    read the constant from the kernel source directly. This is where the two are
    compared -- a drift here silently changes *which kernel* a specialized N
    builds, which the signature check would catch, but with a far less useful
    message than naming the constant.
    """
    upstream = getattr(kernel_module, "SMALL_N_THRESHOLD", None)
    if upstream != RMSNORM_SMALL_N_THRESHOLD:
        raise GeneratorError(
            f"kernels_src SMALL_N_THRESHOLD is {upstream!r} but _instances.py "
            f"mirrors {RMSNORM_SMALL_N_THRESHOLD!r}. Update the mirror and re-check "
            "RMSNORM_SPECIALIZED_N: values at or below the threshold build the "
            "large-M/small-N kernel, which has a different ABI and grid."
        )


def _build_one(kernel_module, torch, instance: Instance, dump_dir: Path) -> bytes:
    """Compile one instance and return its code object."""
    import_n = instance.knobs["N"]
    trace_n = _GENERIC_TRACE_N if import_n is None else import_n
    # `f32` has no instance in the table today, but the kernel supports it and
    # this map was the only thing that raised on it (NOTES §18.6). Kept so that
    # widening `DTYPES` is a one-line change here rather than two in two files.
    torch_dtype = {
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "f32": torch.float32,
    }[instance.dtype]

    env.set_dump_dir(dump_dir)
    launch = kernel_module.build_rmsnorm_module(
        import_n,
        instance.dtype,
        BLOCK_THREADS=instance.knobs["block_threads"],
    )

    # CPU tensors on purpose: COMPILE_ONLY never dispatches, and requiring a live
    # HIP context here would mean a machine with a working GPU is needed to build
    # objects for a machine we do not have.
    x = torch.zeros((_TRACE_M, trace_n), dtype=torch_dtype)
    gamma = torch.zeros((trace_n,), dtype=torch_dtype)
    y = torch.zeros((_TRACE_M, trace_n), dtype=torch_dtype)
    launch(x, gamma, y, _TRACE_M)

    blob, _ = hsaco_from_dump(dump_dir)
    return blob


def generate(arch: str, out_root: Path, keep_ir: Path | None = None) -> Path:
    env.assert_flydsl_version()
    env.prepare(arch)

    import torch  # noqa: PLC0415  (after prepare(): the env must be set first)

    from kernels.norm import rmsnorm_kernel  # noqa: PLC0415

    _assert_threshold_mirror(rmsnorm_kernel)

    arch_dir = out_root / arch
    op_dir = arch_dir / OP
    op_dir.mkdir(parents=True, exist_ok=True)

    instances = instances_for(OP)
    records: list[dict] = []

    for index, instance in enumerate(instances, start=1):
        print(f"[{index}/{len(instances)}] {instance.name}", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"flydsl-{instance.name}-") as tmp:
            dump_dir = Path(keep_ir) / instance.name if keep_ir else Path(tmp)
            try:
                blob = _build_one(rmsnorm_kernel, torch, instance, dump_dir)
            except Exception as exc:
                raise GeneratorError(f"{instance.name}: {exc}") from exc

        filename = f"{instance.name}.hsaco"
        where = f"{arch}/{OP}/{filename}"
        described = _describe(blob, where)
        _verify(described, instance, arch, where)

        (op_dir / filename).write_bytes(blob)

        record = instance.to_record()
        record.update(
            {
                "file": filename,
                "symbol": described["symbol"],
                "bytes": len(blob),
                "sha256": sha256(blob),
                "kernarg_segment_size": described["kernarg_segment_size"],
                "block": [instance.knobs["block_threads"], 1, 1],
                "grid_rule": _GRID_RULE,
                # The gate's finding, recorded per object rather than asserted once:
                # if a future FlyDSL starts emitting names, this stops being 0 and
                # the packer can hand real names to requireSignatureMatch.
                "named_args": described["named_args"],
                "args": [
                    {"name": name, "kind": kind, "size": size, "offset": offset}
                    for name, (kind, size, offset) in zip(
                        _SEMANTIC_ARG_NAMES, described["signature"]
                    )
                ],
            }
        )
        records.append(record)

    write_op_manifest(arch_dir, arch, OP, env.provenance(), records)
    source_md = refresh_source_md(arch_dir, arch)
    print(f"wrote {len(records)} object(s) to {op_dir}")
    print(f"refreshed {source_md}")
    return op_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arch",
        required=True,
        help="GPU target, e.g. gfx1151. One arch per invocation: FlyDSL reads ARCH "
        "from the environment at each compile, so a multi-arch run could file an "
        "object under the wrong name.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=env.KERNELS_OUT_DIR,
        help=f"kernels/ root to write under (default: {env.KERNELS_OUT_DIR})",
    )
    parser.add_argument(
        "--keep-ir",
        type=Path,
        default=None,
        help="keep each instance's IR dump under this directory (debugging)",
    )
    args = parser.parse_args(argv)

    try:
        generate(args.arch, args.out_dir, args.keep_ir)
    except (GeneratorError, env.FlydslEnvError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
