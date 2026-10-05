#!/usr/bin/env python3
"""Derive a minimal descriptor root for a packaging probe.

Copies every file under `--from` to `--out`, except that each KDP shipping for
`--arch` has its inline `kernelDescriptors` reduced to one UKD per compile group
among the UKDs that themselves ship for `--arch`:

    rocke                                     (kind, builder)
    hip                                       (kind, source, build)

The pick within a group is the first by sorted UKD `name`. Packing the derived
root therefore exercises every compile path once instead of every variant.
KDPs that do not ship for `--arch` are copied unchanged; the packer prunes them.

`expect.json` is written beside `--out` (`<out parent>/expect.json`): a list of
{"kdp": <KDP path relative to the root>, "name": <UKD name>, "kind": <authored
kernel_source.kind>}, one entry per kept UKD, for hkp_probe_assert.py.

Files are written only when their content differs and files absent from the
source are removed, so deriving twice leaves every mtime of the derived root
alone and the pack stamp stays fresh across a reconfigure.

Arch rules are the packer's (`hkp_pack.descriptors.arch_matches` and
`_arch_subset_ok`): an empty or absent `arch` list is a wildcard.

Exit code 0 on success. Exit code 2, with a `hkp_probe_derive: ...` message on
stderr, when the root cannot be derived: a KDP shipping for `--arch` references
a standalone UKD (`kernelDescriptors` entry that is not an object), a kept UKD
has a kind other than rocke or hip, a UKD is malformed, or the source is unreadable.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import sys
from pathlib import Path

_PREFIX = "hkp_probe_derive:"

# The kinds the packer compiles to kpack output, and the kernel_source fields that
# define a compile group within a KDP. A new producer needs one row here AND one row
# in hkp_probe_assert.py's _PROVENANCE_RULES; any other kind is rejected.
_GROUP_FIELDS = {
    "rocke": ("builder",),
    "hip": ("source", "build"),
}


class DeriveError(Exception):
    pass


def arch_matches(doc, arch):
    """Port of hkp_pack.descriptors.arch_matches: empty or absent list = wildcard."""
    archs = doc.get("arch")
    if not archs:
        return True
    return arch in archs


def arch_subset_ok(ukd_arch, kdp_arch):
    """Port of hkp_pack.descriptors._arch_subset_ok: empty on either side = wildcard."""
    if not ukd_arch or not kdp_arch:
        return True
    return set(ukd_arch) <= set(kdp_arch)


def _group_key(ukd, where):
    source = ukd.get("kernel_source")
    kind = source.get("kind") if isinstance(source, dict) else None
    if kind not in _GROUP_FIELDS:
        raise DeriveError(
            f"{where} kind {kind!r} has no probe support: add its compile-group "
            "fields here and its provenance rules in hkp_probe_assert.py, once "
            "the packer produces kpack output for it"
        )
    return (kind,) + tuple(
        json.dumps(source.get(f), sort_keys=True) for f in _GROUP_FIELDS[kind]
    )


def _derive_kdp(kdp, rel, arch):
    """Return the kept UKDs of a KDP shipping for `arch`, in authored order."""
    entries = kdp.get("kernelDescriptors")
    if not isinstance(entries, list):
        raise DeriveError(f"{rel} 'kernelDescriptors' is not a list")
    candidates = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise DeriveError(
                f"{rel} references a standalone UKD ({entry!r}): standalone UKD "
                "references unsupported by probe derive"
            )
        where = f"{rel}: UKD '{entry.get('id', '?')}'"
        if not isinstance(entry.get("name"), str):
            raise DeriveError(f"{where} has no string 'name'")
        if not arch_subset_ok(entry.get("arch") or [], kdp.get("arch") or []):
            raise DeriveError(
                f"{where} arch {entry.get('arch')} is not a subset of the KDP "
                f"arch {kdp.get('arch')}"
            )
        if arch_matches(entry, arch):
            candidates.append((_group_key(entry, where), entry))

    picks = {}
    for key, entry in sorted(candidates, key=lambda c: c[1]["name"]):
        picks.setdefault(key, entry)
    kept = [e for e in entries if any(e is p for p in picks.values())]
    names = [e["name"] for e in kept]
    if len(set(names)) != len(names):
        raise DeriveError(
            f"{rel} kept UKDs from different compile groups share a name, so "
            f"expect.json could not tell them apart: {sorted(names)}"
        )
    return kept


def _plan(src, arch):
    """Return (files, expect): files maps relative posix path -> source Path or bytes."""
    files = {}
    for p in sorted(src.rglob("*")):
        if p.is_file():
            files[p.relative_to(src).as_posix()] = p

    expect = []
    for rel in sorted(files):
        if not rel.endswith(".kdp.json"):
            continue
        # The packer skips dot-prefixed paths; so does derive.
        if any(part.startswith(".") for part in rel.split("/")):
            continue
        try:
            kdp = json.loads(files[rel].read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DeriveError(f"{rel} is unreadable: {type(exc).__name__}: {exc}")
        if not isinstance(kdp, dict):
            raise DeriveError(f"{rel} is not a JSON object")
        if not arch_matches(kdp, arch):
            continue
        kept = _derive_kdp(kdp, rel, arch)
        if not kept:
            # Every UKD filters out for this arch: the packer drops the whole KDP.
            continue
        kdp["kernelDescriptors"] = kept
        files[rel] = (json.dumps(kdp, indent=2) + "\n").encode("utf-8")
        expect.extend(
            {"kdp": rel, "name": u["name"], "kind": u["kernel_source"]["kind"]}
            for u in kept
        )
    return files, expect


def _same(dest, content):
    if not dest.is_file() or dest.is_symlink():
        return False
    if isinstance(content, bytes):
        return dest.read_bytes() == content
    return filecmp.cmp(content, dest, shallow=False)


def _write(dest, content):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_dir() and not dest.is_symlink():
        shutil.rmtree(dest)
    elif dest.is_symlink():
        dest.unlink()
    if isinstance(content, bytes):
        dest.write_bytes(content)
    else:
        shutil.copyfile(content, dest)


def _sync(out, files):
    """Make `out` hold exactly `files`, touching only what differs."""
    out.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        dest = out / rel
        if not _same(dest, content):
            _write(dest, content)
    wanted = set(files)
    # Deepest first so emptied directories can go after their files.
    for p in sorted(out.rglob("*"), key=lambda q: len(q.parts), reverse=True):
        rel = p.relative_to(out).as_posix()
        if p.is_symlink() or p.is_file():
            if rel not in wanted:
                p.unlink()
        elif p.is_dir() and not any(w.startswith(rel + "/") for w in wanted):
            shutil.rmtree(p)


def derive_root(src: Path, arch: str, out: Path) -> str | None:
    """Write the derived root to `out`; return an error message, or None on success."""
    if not src.is_dir():
        return f"{_PREFIX} {src} is not a directory"
    try:
        files, expect = _plan(src, arch)
    except DeriveError as exc:
        return f"{_PREFIX} {exc}"

    _sync(out, files)
    expect_text = json.dumps(expect, indent=2) + "\n"
    expect_path = out.parent / "expect.json"
    if (
        not expect_path.is_file()
        or expect_path.read_text(encoding="utf-8") != expect_text
    ):
        expect_path.write_text(expect_text, encoding="utf-8")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="src", type=Path, required=True)
    parser.add_argument(
        "--arch", required=True, help="architecture to probe, e.g. gfx950"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    error = derive_root(args.src, args.arch, args.out)
    if error is not None:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
