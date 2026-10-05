"""Does a KDP ship for any of the selected arches? Prints `TRUE` or `FALSE`.

    python3 tools/hkp_arch_probe.py --generic-targets-json <table.json> \\
        --arches "gfx1100;gfx1151" --kdp <path/to/x.kdp.json>

The CMake configure step asks this to decide whether a root is dormant, so it
must give the packer's own answer: it applies `generic_targets.admits_target`
(an empty `arch` is a wildcard; an entry admits a target it names or a table
generic containing it) to each selected arch.

Only FALSE is authoritative. A KDP that does not parse, or whose `arch` is not
an array of strings, prints TRUE so the packer reports what is wrong with it.
Exits 0 whenever it answered; non-zero only when the table cannot be read.
"""

import argparse
import json
import sys
from pathlib import Path

# Same shadowing hazard hkp_pack.py's own tool guards against.
_PKG_ROOT = str(Path(__file__).resolve().parent.parent / "python")
while _PKG_ROOT in sys.path:
    sys.path.remove(_PKG_ROOT)
sys.path.insert(0, _PKG_ROOT)

from hkp_pack.errors import HkpPackError  # noqa: E402
from hkp_pack.generic_targets import GenericTargets, admits_target  # noqa: E402


def _covers_any(kdp_path, arches, table):
    try:
        doc = json.loads(Path(kdp_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    if not isinstance(doc, dict):
        return True
    if "arch" not in doc:
        return True
    entries = doc["arch"]
    if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
        return True
    if not entries:
        return True
    return any(admits_target(entries, arch, table) for arch in arches)


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
