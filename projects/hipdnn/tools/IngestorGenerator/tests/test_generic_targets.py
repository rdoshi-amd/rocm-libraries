# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The codegen mirror of the tier algebra, held equal to the C++ loader and to
``hkp_pack.generic_targets`` by the shared golden vectors. Fails (never skips) outside
the monorepo: the vectors and the table are located by repo-relative path."""

import json
from pathlib import Path

from codegen import generic_targets

_SDK_DIR = Path(__file__).resolve().parents[3] / "plugin_sdk"
_VECTORS = _SDK_DIR / "tests" / "data" / "arch_tier_vectors.json"
_TIER_NAMES = {
    generic_targets.TIER_EXPLICIT: "explicit",
    generic_targets.TIER_GENERIC: "generic",
    generic_targets.TIER_UNRESTRICTED: "unrestricted",
    None: "none",
}


def test_matches_the_shared_golden_vectors():
    table = generic_targets.GenericTargets.load(generic_targets.DEFAULT_TABLE_PATH)
    assert (
        generic_targets.DEFAULT_TABLE_PATH
        == _SDK_DIR / "data" / "gpu_generic_targets.json"
    )
    vectors = json.loads(_VECTORS.read_text())
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
