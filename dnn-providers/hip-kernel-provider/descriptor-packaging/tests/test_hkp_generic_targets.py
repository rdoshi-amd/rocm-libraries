"""The Python tier algebra over the shared generic target table.

The golden vectors in plugin_sdk/tests/data/arch_tier_vectors.json also drive the C++
oracle, so a disagreement between the two implementations fails one side. This test
locates the table and the vectors itself rather than through conftest.py.
"""

import json
from pathlib import Path

import pytest

from hkp_pack import generic_targets
from hkp_pack.errors import HkpPackError
from hkp_pack.generic_targets import GenericTargets

pytestmark = pytest.mark.quick

_SDK_DIR = Path(__file__).resolve().parents[4] / "projects" / "hipdnn" / "plugin_sdk"
_TABLE = _SDK_DIR / "data" / "gpu_generic_targets.json"
_VECTORS = _SDK_DIR / "tests" / "data" / "arch_tier_vectors.json"

_TIER_NAMES = {
    generic_targets.TIER_EXPLICIT: "explicit",
    generic_targets.TIER_GENERIC: "generic",
    generic_targets.TIER_UNRESTRICTED: "unrestricted",
    None: "none",
}


@pytest.fixture(scope="module")
def table():
    return GenericTargets.load(_TABLE)


def test_default_table_path_is_the_in_repo_table():
    assert generic_targets.DEFAULT_TABLE_PATH == _TABLE
    assert _TABLE.is_file()


def test_loads_the_table_members_in_document_order(table):
    assert table.path == _TABLE
    assert sorted(table.names()) == ["gfx11-generic", "gfx12-generic"]
    assert table.members("gfx11-generic") == (
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",
    )
    assert table.members("gfx12-generic") == ("gfx1200", "gfx1201")
    assert table.has("gfx11-generic")


@pytest.mark.parametrize(
    "doc, message",
    [
        ({"schemaVersion": 2, "generics": {}}, "schemaVersion"),
        ({"schemaVersion": True, "generics": {}}, "schemaVersion"),
        ({"generics": {}}, "schemaVersion"),
        ({"schemaVersion": 1}, "'generics' must be an object"),
        ({"schemaVersion": 1, "generics": []}, "'generics' must be an object"),
        ({"schemaVersion": 1, "generics": {"gfx11-generic": []}}, "non-empty array"),
        (
            {"schemaVersion": 1, "generics": {"gfx11-generic": "gfx1100"}},
            "non-empty array",
        ),
        (
            {"schemaVersion": 1, "generics": {"gfx11-generic": [1100]}},
            "non-empty string",
        ),
        (
            {"schemaVersion": 1, "generics": {"gfx11-generic": ["gfx1100", "gfx1100"]}},
            "more than once",
        ),
        (
            {"schemaVersion": 1, "generics": {"gfx1100": ["gfx1100"]}},
            "must end in '-generic'",
        ),
        ([], "must be a JSON object"),
    ],
)
def test_rejects_a_malformed_table(tmp_path, doc, message):
    path = tmp_path / "table.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(HkpPackError, match=message):
        GenericTargets.load(path)


def test_rejects_an_unreadable_or_non_json_table(tmp_path):
    with pytest.raises(HkpPackError, match="cannot read"):
        GenericTargets.load(tmp_path / "absent.json")
    path = tmp_path / "table.json"
    path.write_text("{not json")
    with pytest.raises(HkpPackError, match="not valid JSON"):
        GenericTargets.load(path)


def _vectors():
    return json.loads(_VECTORS.read_text())


def test_matches_the_golden_tier_vectors(table):
    vectors = _vectors()
    failures = []
    for case in vectors["tier"]:
        got = _TIER_NAMES[
            generic_targets.list_tier(case["arch"], case["device"], table)
        ]
        if got != case["expect"]:
            failures.append(("tier", case, got))
    for case in vectors["overlap"]:
        got = generic_targets.overlaps(case["a"], case["b"], table)
        if got != case["expect"]:
            failures.append(("overlap", case, got))
    for case in vectors["covers"]:
        got = generic_targets.covers(case["outer"], case["inner"], table)
        if got != case["expect"]:
            failures.append(("covers", case, got))
    for case in vectors["compete"]:
        got = generic_targets.compete(case["a"], case["b"], table)
        if got != case["expect"]:
            failures.append(("compete", case, got))
    assert not failures
    assert all(vectors[k] for k in ("tier", "overlap", "covers", "compete"))


def test_unknown_generic_is_not_a_member(table):
    assert not table.has("gfx9-4-generic")
    assert generic_targets.is_generic_shaped("gfx9-4-generic")
    assert generic_targets.entry_tier("gfx9-4-generic", "gfx942", table) is None
    assert generic_targets.entry_tier("gfx9-4-generic", "gfx9-4-generic", table) is None
    assert generic_targets.expand(["gfx9-4-generic"], table) == frozenset()
    assert not generic_targets.admits_target(["gfx9-4-generic"], "gfx942", table)


def test_expand_of_an_empty_list_is_unrestricted(table):
    assert generic_targets.expand([], table) is None
    assert generic_targets.expand(["gfx11-generic", "gfx942"], table) == frozenset(
        [*table.members("gfx11-generic"), "gfx942"]
    )


def test_admits_target_is_empty_literal_or_containing_generic(table):
    assert generic_targets.admits_target([], "gfx1151", table)
    assert generic_targets.admits_target(["gfx1151"], "gfx1151", table)
    assert generic_targets.admits_target(["gfx11-generic"], "gfx1151", table)
    assert not generic_targets.admits_target(["gfx11-generic"], "gfx1154", table)
    assert not generic_targets.admits_target(["gfx942"], "gfx1151", table)
