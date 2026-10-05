"""`tools/hkp_arch_probe.py`: the packer's arch rule, as the CMake configure asks it."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import GENERIC_TARGETS_JSON

pytestmark = pytest.mark.quick

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "hkp_arch_probe.py"


def _probe(tmp_path, arch, arches, raw=None):
    kdp = tmp_path / "x.kdp.json"
    kdp.write_text(raw if raw is not None else json.dumps({"arch": arch}))
    proc = subprocess.run(
        [
            sys.executable,
            str(_TOOL),
            "--generic-targets-json",
            str(GENERIC_TARGETS_JSON),
            "--arches",
            arches,
            "--kdp",
            str(kdp),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_probe_true_when_a_generic_entry_contains_a_selected_target(tmp_path):
    assert _probe(tmp_path, ["gfx11-generic"], "gfx942;gfx1151") == "TRUE"


def test_probe_false_when_no_entry_reaches_any_selected_target(tmp_path):
    assert _probe(tmp_path, ["gfx11-generic", "gfx950"], "gfx942;gfx1250") == "FALSE"
    assert _probe(tmp_path, ["gfx12-generic"], "gfx1100") == "FALSE"


def test_probe_true_for_empty_arch(tmp_path):
    assert _probe(tmp_path, [], "gfx942") == "TRUE"


@pytest.mark.parametrize(
    "raw",
    ['{"arch": "gfx942"}', "{not json", '{"arch": [1]}', "[]", '{"name": "x"}'],
    ids=["string", "unparseable", "non_string_entry", "not_an_object", "absent"],
)
def test_probe_true_for_an_unparseable_or_non_array_arch(tmp_path, raw):
    assert _probe(tmp_path, None, "gfx1100", raw=raw) == "TRUE"
