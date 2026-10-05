# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The ``custom`` correctness configuration of ``tests/sdpa/bwd_kernel_cases.py``.

CPU only: parsing of ``ROCKE_BWD_CUSTOM_KNOBS``, rule matching per head size
and ``stage_vec``, configuration selection, and the plan a custom policy
resolves (the swept candidate's main kernel for matching requests, the arch
default for the others).
"""

import dataclasses
import json

import pytest

from benchmarks.common.attention_bwd_rocke_arm import RockeConfig, config_policy
from kernels.common.attention_bwd_plan import attn_bwd_plan, attn_bwd_stage_vec

from .sdpa.bwd_kernel_cases import (
    CONFIGS,
    CUSTOM_CONFIG,
    CUSTOM_KNOBS_ENV,
    config_knobs,
    config_policy as case_policy,
    custom_knobs_for,
    custom_rules,
    request_from_case,
    selected_cases,
    selected_configs,
)

ARCH = "gfx950"
# A swept geometry of the d128 primary slice (not the arch default).
D128_KNOBS = {
    "waves": 8,
    "block_m": 32,
    "block_n": 128,
    "block_k4": 32,
    "warp_grid_g4": [2, 4],
}
RULES = [{"head_size": 128, "stage_vec": 8, "knobs": D128_KNOBS}]


def _cases(d, *, narrow=False):
    out = []
    for c in selected_cases(tier="smoke"):
        if c.d != d:
            continue
        req = request_from_case(c)
        if narrow:
            req = dataclasses.replace(req, tensor_alignment=8)
        if (attn_bwd_stage_vec(req) == 1) == narrow:
            out.append((c, req))
    assert out, (d, narrow)
    return out


def test_rules_parse_lists_and_matches():
    rules = custom_rules(json.dumps(RULES + [{"knobs": {"g_split": 2}}]))
    assert rules[0] == (
        {"head_size": 128, "stage_vec": 8},
        {**D128_KNOBS, "warp_grid_g4": (2, 4)},
    )
    assert rules[1] == ({}, {"g_split": 2})
    assert custom_knobs_for(rules, 128, 8)["warp_grid_g4"] == (2, 4)
    assert custom_knobs_for(rules, 128, 1) == {"g_split": 2}  # first match wins
    assert custom_knobs_for(rules[:1], 64, 8) == {}
    assert custom_knobs_for(rules[:1], 128, 1) == {}


def test_rules_unset_or_empty_is_none(monkeypatch):
    monkeypatch.delenv(CUSTOM_KNOBS_ENV, raising=False)
    assert custom_rules() is None
    assert custom_rules("  ") is None
    monkeypatch.setenv(CUSTOM_KNOBS_ENV, json.dumps(RULES))
    assert custom_rules()[0][0] == {"head_size": 128, "stage_vec": 8}


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        "[]",
        json.dumps({"knobs": {}}),
        json.dumps([{"head_size": 128}]),
        json.dumps([{"knobs": {"head_size": 64}}]),  # a request field
        json.dumps([{"knobs": {"block_q": 64}}]),  # not a knob
        json.dumps([{"dtype": "bf16", "knobs": {}}]),  # not a match key
    ],
)
def test_malformed_rules_raise(raw):
    with pytest.raises(ValueError):
        custom_rules(raw)


def test_custom_is_selected_only_by_name(monkeypatch):
    monkeypatch.delenv(CUSTOM_KNOBS_ENV, raising=False)
    assert CUSTOM_CONFIG not in CONFIGS
    assert CUSTOM_CONFIG not in selected_configs(ARCH, "both")
    with pytest.raises(ValueError):
        selected_configs(ARCH, "custom")
    with pytest.raises(ValueError):
        case_policy(CUSTOM_CONFIG, ARCH)
    with pytest.raises(ValueError):
        config_knobs(CUSTOM_CONFIG, ARCH)
    monkeypatch.setenv(CUSTOM_KNOBS_ENV, json.dumps(RULES))
    assert CUSTOM_CONFIG not in selected_configs(ARCH, "both")
    assert selected_configs(ARCH, "custom") == ["custom"]
    assert selected_configs(ARCH, "default,custom,custom") == ["default", "custom"]


def test_custom_policy_plans_the_candidate_where_it_matches(monkeypatch):
    monkeypatch.setenv(CUSTOM_KNOBS_ENV, json.dumps(RULES))
    pol = case_policy(CUSTOM_CONFIG, ARCH)
    default = case_policy("default", ARCH)
    cand = config_policy(RockeConfig("c", custom_rules()[0][1]))
    for _, req in _cases(128):
        spec = attn_bwd_plan(req, ARCH, policy=pol).specs["main"]
        assert spec == attn_bwd_plan(req, ARCH, policy=cand).specs["main"]
        assert spec != attn_bwd_plan(req, ARCH, policy=default).specs["main"]
        assert (spec.waves, spec.block_m, spec.block_n) == (8, 32, 128)
        assert spec.warp_grid_g4 == (2, 4)
    # other head sizes and the narrow-access class keep the arch default
    for _, req in _cases(64) + _cases(128, narrow=True):
        spec = attn_bwd_plan(req, ARCH, policy=pol).specs["main"]
        assert spec == attn_bwd_plan(req, ARCH, policy=default).specs["main"]


def test_custom_policy_applies_overrides_on_top(monkeypatch):
    monkeypatch.setenv(CUSTOM_KNOBS_ENV, json.dumps(RULES))
    pol = case_policy(CUSTOM_CONFIG, ARCH, {"edge_tiles": True})
    for _, req in _cases(128)[:3]:
        spec = attn_bwd_plan(req, ARCH, policy=pol).specs["main"]
        assert spec.edge_tiles and spec.block_m == 32
    for _, req in _cases(64)[:3]:
        spec = attn_bwd_plan(req, ARCH, policy=pol).specs["main"]
        assert spec.edge_tiles
