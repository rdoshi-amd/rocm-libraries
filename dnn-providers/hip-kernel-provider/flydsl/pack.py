# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pack the checked-in FlyDSL ``.hsaco`` objects for one arch into a ``.kpack``.

Usage:
    python pack.py --kernel-dir <flydsl/kernels> --arch gfx1151 --out-dir <dir>

Requires ``rocm_kpack`` on PYTHONPATH (consumed from source, not pip install);
see ``src/engines/asm_sdpa_engine/asm/asm_kernels/pack.py``, which this mirrors.

**The manifest is the input, not the directory listing.** The ASM packer globs
``*.co`` because its objects arrive as an opaque vendored drop with no index.
Ours arrive with ``kernels/<arch>/<op>/manifest.json``, written by the generator
that produced them, and every consumer downstream -- the UKD ``kernel_source``
block, ``SOURCE.md``, the CI drift check -- keys off that file. Packing from the
manifest instead of from a glob buys three properties a glob cannot:

* a stray or half-written object in the tree cannot silently enter the archive;
* an object *missing* from the tree is an error here rather than a load failure
  on a machine that has no way to diagnose it;
* the SHA256 the descriptor will claim is verified against the bytes actually
  packed, at pack time, so descriptor and archive cannot disagree.

The TOC key is ``<op>/<file>`` -- the object's path relative to the arch
directory, which is the same convention the ASM packer derives with
``relative_to(arch_dir)``. It is the key the UKD ``kernel_source.toc_key`` must
name, so the two are generated from one string in one place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

GROUP_NAME = "hip_kernel_provider_flydsl"
MANIFEST_NAME = "manifest.json"


class KernelEntry(NamedTuple):
    """One code object, as both the packer and the descriptor generator see it.

    `record` is the manifest entry verbatim, carried so that a second consumer
    does not have to re-read and re-validate the manifest to learn an object's
    tier, symbol or baked N. `toc_key` and `record` travelling together is the
    point: the archive key and the descriptor that names it are then two views
    of one object rather than two strings that have to be kept equal by hand.
    """

    toc_key: str
    path: Path
    data: bytes
    record: dict[str, Any]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def kpack_name(arch: str) -> str:
    """The archive filename for `arch`, as both the packer and CMake spell it."""
    return f"{GROUP_NAME}_{arch}.kpack"


def collect(arch_dir: Path, arch: str) -> list[KernelEntry]:
    """Every `KernelEntry` named by the manifests under `arch_dir`.

    Raises on a manifest that disagrees with its own location, on a named object
    that is absent, and on one whose bytes do not hash to the recorded SHA256.
    """
    manifests = sorted(arch_dir.glob(f"*/{MANIFEST_NAME}"))
    if not manifests:
        raise FileNotFoundError(
            f"no */{MANIFEST_NAME} under {arch_dir}; run the generators for this "
            "arch first (see flydsl/REGEN.md)"
        )

    entries: list[KernelEntry] = []
    seen: dict[str, Path] = {}

    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        op = manifest["op"]

        # The manifest records the arch it was built for; the directory records
        # the arch it is shipped as. A disagreement means an object is about to
        # be packed into an archive for hardware it was not compiled for.
        if manifest.get("arch") != arch:
            raise ValueError(
                f"{manifest_path}: manifest records arch '{manifest.get('arch')}' "
                f"but sits under '{arch}/'"
            )
        if manifest_path.parent.name != op:
            raise ValueError(
                f"{manifest_path}: manifest records op '{op}' but sits in "
                f"'{manifest_path.parent.name}/'"
            )

        for record in manifest["instances"]:
            path = manifest_path.parent / record["file"]
            if not path.is_file():
                raise FileNotFoundError(
                    f"{manifest_path} names '{record['file']}' but {path} does not exist"
                )
            data = path.read_bytes()
            actual = _sha256(data)
            if actual != record["sha256"]:
                raise ValueError(
                    f"{path}: SHA256 {actual} does not match the {record['sha256']} "
                    f"recorded in {manifest_path}. Regenerate rather than repacking: "
                    "the descriptors claim the recorded hash."
                )
            toc_key = f"{op}/{record['file']}"
            if toc_key in seen:
                raise ValueError(
                    f"duplicate TOC key '{toc_key}' from {seen[toc_key]} and {path}"
                )
            seen[toc_key] = path
            entries.append(KernelEntry(toc_key, path, data, record))

    return entries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pack checked-in FlyDSL .hsaco objects into a .kpack archive"
    )
    parser.add_argument(
        "--kernel-dir",
        type=Path,
        required=True,
        help="Root directory containing <arch>/<op>/ subdirectories (flydsl/kernels)",
    )
    parser.add_argument(
        "--arch", type=str, required=True, help="GPU architecture (e.g. gfx1151)"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Output directory for the .kpack"
    )
    args = parser.parse_args()

    arch_dir: Path = args.kernel_dir / args.arch
    if not arch_dir.is_dir():
        print(f"Error: architecture directory not found: {arch_dir}", file=sys.stderr)
        return 1

    try:
        entries = collect(arch_dir, args.arch)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        from rocm_kpack.compression import ZstdCompressor
        from rocm_kpack.kpack import PackedKernelArchive
    except ImportError as exc:
        print(
            f"Error: failed to import rocm_kpack: {exc}\n"
            "Ensure PYTHONPATH includes the rocm_kpack Python source directory.",
            file=sys.stderr,
        )
        return 1

    print(f"Packing {len(entries)} FlyDSL code object(s) for {args.arch}...")

    archive = PackedKernelArchive(
        group_name=GROUP_NAME,
        gfx_arch_family=args.arch,
        gfx_arches=[args.arch],
        compressor=ZstdCompressor(compression_level=3),
    )

    for entry in entries:
        archive.add_kernel(
            archive.prepare_kernel(
                relative_path=entry.toc_key, gfx_arch=args.arch, hsaco_data=entry.data
            )
        )

    archive.finalize_archive()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    kpack_path = args.out_dir / kpack_name(args.arch)
    archive.write(kpack_path)

    print(f"Written: {kpack_path} ({kpack_path.stat().st_size} bytes)")

    # Round-trip: the archive is what ships, so it -- not the input tree -- is
    # what has to hash correctly.
    written = PackedKernelArchive.read(kpack_path)
    failures = 0
    for entry in entries:
        packed = written.get_kernel(entry.toc_key, args.arch)
        if packed is None:
            print(
                f"  VERIFY FAIL: key '{entry.toc_key}' not found in archive",
                file=sys.stderr,
            )
            failures += 1
            continue
        if _sha256(packed) != _sha256(entry.data):
            print(
                f"  VERIFY FAIL: key '{entry.toc_key}' SHA256 mismatch against "
                f"{entry.path}",
                file=sys.stderr,
            )
            failures += 1

    if failures:
        print(
            f"Error: {failures}/{len(entries)} round-trip verification failures",
            file=sys.stderr,
        )
        return 1

    print(f"Round-trip verification passed for all {len(entries)} kernels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
