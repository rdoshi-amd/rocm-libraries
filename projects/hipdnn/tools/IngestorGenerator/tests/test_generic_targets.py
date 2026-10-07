# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The codegen mirror of the tier algebra, over a small in-memory table so the cases
pin the algebra rather than LLVM-owned membership data."""

from pathlib import Path

import pytest

from codegen import generic_targets as gt

_TABLE = gt.GenericTargets(
    Path("table.json"),
    {
        "gfx11-generic": ("gfx1100", "gfx1151"),
        "gfx12-generic": ("gfx1200", "gfx1201"),
    },
)
_G11 = "gfx11-generic"
_G12 = "gfx12-generic"
_UNKNOWN = "gfx9-4-generic"

_UNRESTRICTED = gt.TIER_UNRESTRICTED
_EXPLICIT = gt.TIER_EXPLICIT
_GENERIC = gt.TIER_GENERIC


@pytest.mark.parametrize(
    ("arch", "device", "expect"),
    [
        ([], "gfx1151", _UNRESTRICTED),
        ([], "gfx942:sramecc+:xnack-", _UNRESTRICTED),
        (["gfx1151"], "gfx1151", _EXPLICIT),
        (["gfx942"], "gfx942:sramecc+:xnack-", _EXPLICIT),
        ([_G11], "gfx1100", _GENERIC),
        ([_G11], "gfx1151:sramecc+", _GENERIC),
        # Explicit outranks a generic that also admits the device.
        (["gfx1151", _G11], "gfx1151", _EXPLICIT),
        (["gfx1151", _G11], "gfx1100", _GENERIC),
        (["gfx942", _G11], "gfx1100", _GENERIC),
        # Non-matches.
        ([_G11], "gfx1154", None),
        ([_G12], "gfx1100", None),
        (["gfx942"], "gfx950", None),
        # An id is a whole-name (or feature-suffix) match, never a prefix match.
        (["gfx94"], "gfx942", None),
        (["gfx1250"], "gfx1250-strict", None),
        (["gfx1250-strict"], "gfx1250-strict", _EXPLICIT),
        # A generic-shaped name outside the table matches nothing, itself included.
        ([_UNKNOWN], "gfx942", None),
        ([_UNKNOWN], _UNKNOWN, None),
        ([_UNKNOWN, "gfx942"], "gfx942", _EXPLICIT),
    ],
)
def test_list_tier(arch, device, expect):
    assert gt.list_tier(arch, device, _TABLE) == expect


@pytest.mark.parametrize(
    ("outer", "inner", "expect"),
    [
        ([], ["gfx942"], True),
        (["gfx942"], [], True),
        (["gfx942", "gfx950"], ["gfx942"], True),
        (["gfx942"], ["gfx942", "gfx950"], False),
        ([_G11], ["gfx1151"], True),
        ([_G11], ["gfx1100", "gfx1151"], True),
        ([_G11], ["gfx1154"], False),
        (["gfx1151"], [_G11], False),
        (["gfx1100", "gfx1151"], [_G11], True),
        ([_G11], [_G11], True),
        ([_G11], [_G12], False),
        ([_UNKNOWN], [_UNKNOWN], True),
        ([_UNKNOWN], ["gfx942"], False),
    ],
)
def test_covers(outer, inner, expect):
    assert gt.covers(outer, inner, _TABLE) is expect


@pytest.mark.parametrize(
    ("a", "b", "expect"),
    [
        ([], [], True),
        ([], ["gfx942"], False),
        ([_G11], [], False),
        (["gfx942"], ["gfx942"], True),
        (["gfx942"], ["gfx950"], False),
        # A generic and a member it contains never tie: the member outranks it.
        ([_G11], ["gfx1151"], False),
        ([_G11], ["gfx1151", "gfx1100"], False),
        ([_G11], [_G11], True),
        ([_G11], [_G12], False),
        # Tied on the generic's other member even though gfx1151 differs in tier.
        (["gfx1151", _G11], [_G11], True),
        ([_G11], [_G11, "gfx942"], True),
        # An unknown generic admits nothing, so it ties with nothing.
        ([_UNKNOWN], [_UNKNOWN], False),
        ([_UNKNOWN], [], False),
    ],
)
def test_compete(a, b, expect):
    assert gt.compete(a, b, _TABLE) is expect
    assert gt.compete(b, a, _TABLE) is expect


def test_the_shipped_table_loads_and_lists_gfx11_generic_members():
    table = gt.default_table()
    assert table.has(_G11)
    assert "gfx1151" in table.members(_G11)
    assert gt.list_tier([_G11], "gfx1151", table) == _GENERIC
