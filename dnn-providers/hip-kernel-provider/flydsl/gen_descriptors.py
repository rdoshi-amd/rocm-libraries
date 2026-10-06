# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Generate (or verify) the ingestor descriptors for one arch's FlyDSL kernels.

Usage:
    python gen_descriptors.py --kernel-dir <flydsl/kernels> \\
        --descriptor-dir <flydsl/descriptors> --arch gfx1151
    python gen_descriptors.py ... --check     # verify, write nothing

**Generated, not authored, and checked in.** Every field below is derived from
``kernels/<arch>/<op>/manifest.json`` and from the code objects it names, so a
descriptor cannot disagree with the archive it describes. The output is then
committed alongside the ``.hsaco`` files, because the build must stay "copy the
objects, pack them, stage the descriptors" -- generation is a regeneration step
a human runs when the kernels change, not a build step.

**We author the shipped form, not the authored form.** The upstream pack
pipeline takes ``kind: "hip"`` descriptors naming a source file and rewrites
them into ``kind: "kpack"`` descriptors naming an archive entry
(``pipeline.py:1040-1100``). Our objects are pre-built and our CMake stages
descriptors verbatim with no rewrite, so there is no rewrite pass to produce
the shipped form for us: this script emits it directly, matching that function
field for field.

**The signature is read out of the object, never typed.** ``kernel_signature``
parses the AMDGPU msgpack metadata note out of the code object itself; its own
docstring is explicit that "a hand-authored arity restates the same assumption
that drifted". Ours would have drifted on its first line -- ``manifest.json``
records eight *named* arguments, and the objects FlyDSL emits carry no argument
names at all (``named_args: 0``; clang omits ``.args[].name`` for HIP
``extern "C" __global__``).

``--check`` deliberately does **not** re-derive signatures, so it needs no
``msgpack`` and is cheap enough to run on every build. It cannot miss a
signature drift regardless: a signature can only change if the object changed,
which changes the SHA256 that ``--check`` does compare.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "descriptor-packaging" / "python"))

from hkp_pack.descriptors import HkpPackError, load_flat_input  # noqa: E402

from pack import KernelEntry, collect, kpack_name  # noqa: E402

# Descriptor schema version, as the loader spells it: UKD_VERSION_MAJOR = 1,
# UKD_VERSION_MINOR = 0 (DescriptorLoader.hpp:288-289).
DESCRIPTOR_VERSION = "1.0"

# The directory the packer writes the archive into, relative to the shard root.
# It must agree with the staging layout in flydsl/CMakeLists.txt: our
# descriptors sit flat at that root, so the `kernel_source.library` string is
# this plus the filename with no climb. The loader resolves that string against
# the descriptor file's own directory and rejects a result outside the
# descriptor tree, so a climb would not merely be ugly -- it would not load.
KPACK_DIR_NAME = "kpack"

# `metadataValueFromJson` (DescriptorLoader.hpp:523) accepts bool/int/float/
# string/int-list and *fails on null*, so "this kernel reads N at runtime"
# cannot be spelled the way the manifest spells it. 0 is the sentinel: no shape
# is ever 0 rows, and the matcher reads it as "generic, accepts any N".
GENERIC_N = 0

# Manifest knobs carried as null when not baked in. Only these may take the
# `GENERIC_N` sentinel; a null anywhere else is a generator bug, not a tier.
SHAPE_KNOBS = {"N"}

# Fields the graph fixes rather than the tuner choosing. They belong in the
# metadata schema -- the matcher needs every one of them to pick a variant --
# but not in the engine's `knobs` list, which names what autotune may vary.
# Offering the tuner a dtype or a row length it cannot legally change would
# have it search a space where all but one point fails to match.
GRAPH_DETERMINED_FIELDS = {"N", "dtype"}

# uuid5 from a name, so ids are stable across regenerations and adding a
# thirteenth instance cannot disturb the twelve already shipped. The namespace
# is itself derived from a URL rather than being a magic constant.
_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/ROCm/rocm-libraries/dnn-providers/hip-kernel-provider/flydsl",
)


def _uuid(name: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, name))


def _metadata_type(value: Any) -> str:
    """The KMD type token for a manifest knob value.

    A null knob means "not baked" and is carried as the `GENERIC_N` int
    sentinel, so it types as an int like its baked siblings -- one field cannot
    change type between instances.
    """
    if value is None or isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    raise HkpPackError(f"knob value {value!r} has no metadata type")


def _metadata_value(key: str, value: Any) -> Any:
    if value is None:
        if key not in SHAPE_KNOBS:
            raise HkpPackError(
                f"knob '{key}' is null but is not a shape knob; only a shape "
                "knob has a 'not baked' sentinel"
            )
        return GENERIC_N
    return value


class OpDescriptors:
    """The one-per-op descriptors: schema, engine, heuristic, dispatch, matcher.

    Shared by every instance of the op and identical in every arch's shard. The
    ids are deliberately *not* arch-scoped: a machine installs only its own
    shard, so each shard must carry its own copy, and two shards declaring the
    same engine with the same id and the same bytes is how the catalog settles
    them into one entry rather than two rival engines.
    """

    def __init__(self, op: str, fields: dict[str, str]) -> None:
        self.op = op
        self.stem = f"flydsl_{op}"
        self.symbols = f"hipkernel.{self.stem}"
        self.fields = fields

        self.kmd_id = _uuid(f"kmd/{op}")
        self.uhd_id = _uuid(f"uhd/{op}")
        self.udd_id = _uuid(f"udd/{op}")
        self.ued_id = _uuid(f"ued/{op}")
        self.umd_id = _uuid(f"umd/{op}/kernel")

    def documents(self) -> dict[str, dict[str, Any]]:
        kmd_fields = []
        for name, type_token in self.fields.items():
            field: dict[str, Any] = {"name": name, "type": type_token}
            # Only the tunable field carries a default. The rest are required:
            # `coerceKernelMetadata` (DescriptorLoader.hpp:1291) drops the whole
            # pack when a field with no default is missing, which is the loud
            # failure we want if an instance ever ships without its dtype or N.
            if name == "block_threads":
                field["default_value"] = 256
            kmd_fields.append(field)

        return {
            f"{self.stem}.kmd.json": {
                "version": DESCRIPTOR_VERSION,
                "id": self.kmd_id,
                "name": f"flydsl {self.op} variant fields",
                "fields": kmd_fields,
            },
            f"{self.stem}.uhd.json": {
                "version": DESCRIPTOR_VERSION,
                "id": self.uhd_id,
                "name": f"flydsl {self.op} selector",
                "kind": "native",
                "payload": f"{self.symbols}.score",
            },
            f"{self.stem}.udd.json": {
                "version": DESCRIPTOR_VERSION,
                "id": self.udd_id,
                "name": f"flydsl {self.op} dispatch",
                "dispatch_symbol": f"{self.symbols}.dispatch",
            },
            f"{self.stem}.ued.json": {
                "version": DESCRIPTOR_VERSION,
                "id": self.ued_id,
                "name": f"hipkernel:{self.stem}",
                "graph_match": {"native": f"{self.symbols}.graph_match"},
                "heuristic": self.uhd_id,
                "metadata": self.kmd_id,
                "knobs": [f for f in self.fields if f not in GRAPH_DETERMINED_FIELDS],
            },
            f"{self.stem}_kernel_match.umd.json": {
                "version": DESCRIPTOR_VERSION,
                "id": self.umd_id,
                "name": f"flydsl {self.op}: kernel variant fits the graph",
                "scope": "kernel",
                "match_symbol": f"{self.symbols}.kernel_match",
            },
        }

    def provenance(self) -> dict[str, Any]:
        """What this pack's metadata is used for, stated honestly.

        Nothing is compiled in from a binding: the objects arrive pre-built from
        FlyDSL's own generator, so every field exists only to pick which
        pre-built object fits. `metadata_fields` empty and all three fields
        matcher-only is the exact shape of that claim. Derived from the same
        field list the KMD is built from, so the two cannot drift apart.
        """
        return {
            "specialization_contract": {
                "schema_version": 1,
                "consumers": [
                    {
                        "engine_id": self.ued_id,
                        "kmd_id": self.kmd_id,
                        "metadata_fields": [],
                        "matcher_only_fields": list(self.fields),
                        "bindings": {},
                        "vocabulary": {},
                    }
                ],
            }
        }


def _fields_for_op(entries: list[KernelEntry]) -> dict[str, str]:
    """The metadata schema for an op, derived from its instances' knobs.

    Every instance must declare the same knob set: a schema is one shape, and an
    instance that carries a knob its siblings do not is a generator bug that
    would otherwise surface as a silently unselectable kernel.
    """
    fields: dict[str, str] = {}
    first: dict[str, Any] | None = None
    for entry in entries:
        knobs = entry.record["knobs"]
        if first is None:
            first = knobs
            fields["dtype"] = "string"
            for key, value in knobs.items():
                fields[key] = _metadata_type(value)
        elif set(knobs) != set(first):
            raise HkpPackError(
                f"{entry.record['name']} declares knobs {sorted(knobs)} but its "
                f"siblings declare {sorted(first)}; one op is one schema"
            )
    return fields


def _kernel_documents(
    entries: list[KernelEntry], op: OpDescriptors, arch: str, *, derive_signature: bool
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """One UKD per code object, plus the id list the KDP names them by."""
    library = f"{KPACK_DIR_NAME}/{kpack_name(arch)}"
    provenance = op.provenance()

    documents: dict[str, dict[str, Any]] = {}
    ids: list[str] = []

    for entry in entries:
        record = entry.record
        ukd_id = _uuid(f"ukd/{arch}/{entry.toc_key}")
        ids.append(ukd_id)

        signature: list[dict[str, Any]] = []
        if derive_signature:
            from hkp_pack.kernel_signature import kernel_signature

            signature = kernel_signature(entry.data, record["symbol"], str(entry.path))

        metadata = {"dtype": record["dtype"]}
        for key, value in record["knobs"].items():
            metadata[key] = _metadata_value(key, value)

        documents[f"{record['name']}.ukd.json"] = {
            "version": DESCRIPTOR_VERSION,
            "id": ukd_id,
            "name": record["name"],
            "arch": [arch],
            "kernel_source": {
                "kind": "kpack",
                "library": library,
                "toc_key": entry.toc_key,
                "symbol": record["symbol"],
                "sha256": record["sha256"],
                "signature": signature,
            },
            "metadata": metadata,
            "priority": record["priority"],
            "provenance": provenance,
        }

    return documents, ids


def build_documents(
    entries: list[KernelEntry], arch: str, *, derive_signature: bool
) -> dict[str, dict[str, Any]]:
    """Every descriptor for `arch`, keyed by the filename it is written as."""
    by_op: dict[str, list[KernelEntry]] = {}
    for entry in entries:
        by_op.setdefault(entry.record["op"], []).append(entry)

    documents: dict[str, dict[str, Any]] = {}
    for op_name, op_entries in sorted(by_op.items()):
        op = OpDescriptors(op_name, _fields_for_op(op_entries))
        documents.update(op.documents())

        kernels, ids = _kernel_documents(
            op_entries, op, arch, derive_signature=derive_signature
        )
        documents.update(kernels)

        documents[f"{op.stem}.kdp.json"] = {
            "version": DESCRIPTOR_VERSION,
            "id": _uuid(f"kdp/{op_name}/{arch}"),
            "name": f"flydsl {op_name} kernels ({arch})",
            "arch": [arch],
            "matchers": [op.umd_id],
            "engine": op.ued_id,
            "dispatch": op.udd_id,
            "kernelDescriptors": ids,
        }

    return documents


def _serialize(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2) + "\n"


def _write(out_dir: Path, documents: dict[str, dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, document in sorted(documents.items()):
        (out_dir / name).write_text(_serialize(document), encoding="utf-8")

    # A descriptor whose kernel no longer exists resolves to a TOC key the
    # archive does not contain, so leaving one behind fails the load rather than
    # the regeneration. Regeneration owns the directory.
    for stale in sorted(out_dir.glob("*.json")):
        if stale.name not in documents:
            print(f"  removing stale descriptor: {stale.name}")
            stale.unlink()


def _check(out_dir: Path, entries: list[KernelEntry], arch: str) -> list[str]:
    """Cross-check the checked-in descriptors against the objects they claim.

    Signature-free by design (see the module docstring), so this runs on every
    build without `msgpack`.
    """
    problems: list[str] = []
    library = f"{KPACK_DIR_NAME}/{kpack_name(arch)}"

    expected = {entry.toc_key: entry for entry in entries}
    found: dict[str, Path] = {}

    for path in sorted(out_dir.glob("*.ukd.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        source = document.get("kernel_source", {})
        toc_key = source.get("toc_key")

        if toc_key not in expected:
            problems.append(f"{path.name}: toc_key '{toc_key}' names no packed object")
            continue
        if toc_key in found:
            problems.append(
                f"{path.name}: toc_key '{toc_key}' already claimed by {found[toc_key].name}"
            )
            continue
        found[toc_key] = path

        entry = expected[toc_key]
        if source.get("sha256") != entry.record["sha256"]:
            problems.append(
                f"{path.name}: sha256 does not match {entry.path.name}; regenerate"
            )
        if source.get("symbol") != entry.record["symbol"]:
            problems.append(f"{path.name}: symbol does not match the manifest")
        if source.get("library") != library:
            problems.append(
                f"{path.name}: library '{source.get('library')}' is not '{library}'"
            )
        if not source.get("signature"):
            problems.append(f"{path.name}: empty signature")

    for toc_key in sorted(set(expected) - set(found)):
        problems.append(f"packed object '{toc_key}' has no descriptor")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify ingestor descriptors for FlyDSL kernels"
    )
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        required=True,
        help="Root containing <arch>/<op>/ subdirectories (flydsl/kernels)",
    )
    parser.add_argument(
        "--descriptor-dir",
        type=Path,
        required=True,
        help="Root to write <arch>/ descriptors into (flydsl/descriptors)",
    )
    parser.add_argument(
        "--arch", type=str, required=True, help="GPU architecture (e.g. gfx1151)"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the checked-in descriptors and exit; write nothing",
    )
    args = parser.parse_args()

    arch_dir: Path = args.kernel_dir / args.arch
    if not arch_dir.is_dir():
        print(f"Error: architecture directory not found: {arch_dir}", file=sys.stderr)
        return 1

    out_dir: Path = args.descriptor_dir / args.arch

    try:
        entries = collect(arch_dir, args.arch)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not args.check:
        try:
            documents = build_documents(entries, args.arch, derive_signature=True)
        except ImportError as exc:
            print(
                f"Error: failed to import msgpack: {exc}\n"
                "Signature derivation reads the AMDGPU metadata note out of each "
                "code object; install msgpack to regenerate descriptors.",
                file=sys.stderr,
            )
            return 1
        except HkpPackError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        print(f"Writing {len(documents)} descriptor(s) for {args.arch}...")
        _write(out_dir, documents)

    if not out_dir.is_dir():
        print(f"Error: descriptor directory not found: {out_dir}", file=sys.stderr)
        return 1

    problems = _check(out_dir, entries, args.arch)
    if problems:
        for problem in problems:
            print(f"  DESCRIPTOR FAIL: {problem}", file=sys.stderr)
        print(
            f"Error: {len(problems)} descriptor(s) disagree with the packed "
            f"objects under {arch_dir}",
            file=sys.stderr,
        )
        return 1

    # The authoring validator: shapes, enums, reserved names, and -- the reason
    # it is worth running rather than trusting the generator -- that every id a
    # KDP references actually resolves. One arch directory at a time, because
    # one shard is one arch and the per-op descriptors are shared verbatim
    # between shards, which a whole-tree load would read as duplicate ids.
    try:
        load_flat_input(out_dir, log=lambda *a, **k: None)
    except HkpPackError as exc:
        print(f"Error: descriptor validation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Validated {len(entries)} kernel descriptor(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
