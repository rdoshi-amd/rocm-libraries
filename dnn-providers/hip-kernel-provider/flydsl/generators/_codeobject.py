# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Reading and verifying a compiled code object, shared by every op's generator.

Every generator compiles an instance, then checks the object it got back against
what it asked for -- arch, kernarg layout, kernarg segment size -- *before*
writing it. The checks differ per op only in the expected values, so the reading
and the arch parse live here once.
"""

from __future__ import annotations

import sys

from . import _flydsl_env as env


class GeneratorError(RuntimeError):
    """An instance could not be built, or the object built is not the one asked for."""


def hkp_pack_module():
    """Import ``hkp_pack.kernel_signature`` from the sibling packaging tree.

    We reach for two of its helpers rather than re-deriving them: ``amdgcn_object``
    unwraps the clang offload bundle by *selecting on the ``-amdgcn-`` triple*
    instead of by position, and ``_metadata_document`` walks the ELF's SHT_NOTE
    sections for ``NT_AMDGPU_METADATA``. A second copy of either in this
    directory would be a second thing to keep correct, and the packer these
    generators feed parses the very same note.
    """
    packaging = env.PROVIDER_DIR.parent / "descriptor-packaging" / "python"
    if not (packaging / "hkp_pack").is_dir():
        raise GeneratorError(f"hkp_pack not found under {packaging}")
    if str(packaging) not in sys.path:
        sys.path.insert(0, str(packaging))
    from hkp_pack import kernel_signature as module  # noqa: PLC0415

    return module


def describe(blob: bytes, where: str) -> dict:
    """The AMDGPU metadata facts we verify and record, for one code object."""
    hkp = hkp_pack_module()
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
        "group_segment_fixed_size": kernel.get(".group_segment_fixed_size"),
        "vgpr_count": kernel.get(".vgpr_count"),
        "vgpr_spill_count": kernel.get(".vgpr_spill_count"),
    }


def target_arch(targets, where: str) -> str:
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


def verify_arch(described: dict, arch: str, where: str) -> None:
    """Fail before writing an object filed under an arch it was not built for."""
    built_for = target_arch(described["targets"], where)
    if built_for != arch:
        raise GeneratorError(
            f"{where}: object targets {built_for!r} but is being filed under "
            f"{arch!r}. FlyDSL reads ARCH from the environment at each compile, so "
            "this means the env moved under the run -- not that the file is "
            "misnamed. Regenerate one arch per invocation."
        )


def arg_records(names: tuple[str, ...], signature: tuple) -> list[dict]:
    """The manifest's per-argument records: semantic name beside the real layout."""
    return [
        {"name": name, "kind": kind, "size": size, "offset": offset}
        for name, (kind, size, offset) in zip(names, signature)
    ]
