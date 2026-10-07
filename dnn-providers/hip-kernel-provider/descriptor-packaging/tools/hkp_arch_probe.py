"""Does a KDP ship for any of the selected arches? Prints `TRUE` or `FALSE`.

    python3 tools/hkp_arch_probe.py --generic-targets-json <table.json> \\
        --arches "gfx1100;gfx1151" --kdp <path/to/x.kdp.json>

Only FALSE is authoritative: a malformed KDP or unknown generic prints TRUE so the
packer reports it. Non-zero exit only when the table cannot be read.
"""

import argparse
import json
import sys
from pathlib import Path

# Same shadowing guard as hkp_pack.py.
_PKG_ROOT = str(Path(__file__).resolve().parent.parent / "python")
while _PKG_ROOT in sys.path:
    sys.path.remove(_PKG_ROOT)
sys.path.insert(0, _PKG_ROOT)

from hkp_pack.errors import HkpPackError  # noqa: E402
from hkp_pack.generic_targets import (  # noqa: E402
    GenericTargets,
    admits_target,
    is_generic_shaped,
)


def _covers_any(kdp_path, arches, table):
    """False only when the KDP's well-formed arch list reaches none of @arches."""
    try:
        entries = json.loads(Path(kdp_path).read_text(encoding="utf-8"))["arch"]
        if not isinstance(entries, list) or not entries:
            return True
        # An unknown generic would read as absence; the packer must reach it to report it.
        if any(is_generic_shaped(e) and not table.has(e) for e in entries):
            return True
        return any(admits_target(entries, arch, table) for arch in arches)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return True


def main(argv=None):
    p = argparse.ArgumentParser(prog="hkp_arch_probe", description=__doc__)
    p.add_argument("--generic-targets-json", required=True)
    p.add_argument("--arches", required=True, help="';'- or ','-separated arch list")
    p.add_argument("--kdp", required=True)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        table = GenericTargets.load(args.generic_targets_json)
    except HkpPackError as exc:
        print(f"hkp_arch_probe: error: {exc}", file=sys.stderr)
        return 1
    arches = [a for a in args.arches.replace(",", ";").split(";") if a]
    print("TRUE" if _covers_any(args.kdp, arches, table) else "FALSE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
