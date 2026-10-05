# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Negative controls of the general attention backward (device).

Each control breaks one input of the pipeline and the comparison against the
oracle must then fail; the unbroken run of the same case must pass, so a
control that "fails" because the case itself is broken is caught too.

* LSE fed in log2 units instead of the natural log;
* LSE offset by 0.05;
* the ``bottom_right`` kernel argument flipped on an ``s_q != s_kv`` case;
* the main kernel built without the dead-row term ``lse2 < +inf`` of the dS
  select (test-only body switch ``sentinel=False``): it passes on consistent
  inputs, and gives NaN where a dead band-live row (``-inf`` LSE, NaN O)
  meets an ungated Dsum, which the shipped body turns into exact zeros; the
  same ungated Dsum on dead rows inside interior tiles (``edge_tiles = True``,
  whose steps keep only the dead-row term of the select) stays exact zeros;
* direct mode launched with ``G = 1`` on a GQA / MQA case (the on-chip
  group sum of dK / dV is missing);
* prep skipped (the workspace keeps its NaN poison);
* the dQ convert multiplier applying the scale a second time;
* atomic dK / dV (``h_k != h_v``): prep's kv-zero role switched off (the dK /
  dV workspace keeps its NaN poison), or the dV convert skipped (dV keeps the
  output sentinel);
* padding: the main kernel launched with ``has_len = 0`` (it ignores the
  per-batch SEQ_LEN counts, while prep and the converts still honour them);
* THD: the main kernel told that an int64 offset table is int32 (prep and
  the converts read it at its width).

Configurations follow ``ROCKE_BWD_CONFIG`` (default: every distinct one of the
arch). Skipped unless the visible device is gfx942, gfx950, gfx1151 or
gfx1201.
"""

from __future__ import annotations

import numpy as np
import pytest

from ._attention_bwd_harness import (
    ARCH,
    DEVICE_ARCHS,
    ThdStorage,
    _replace_launch,
    dead_row_inputs,
    dead_rows,
    main_built_with,
    run_case,
    ungated_dsum_hook,
)
from .sdpa.bwd_cases import compare_bwd, get_bwd_case
from .sdpa.bwd_kernel_cases import SUPPLEMENTARY_CASES, selected_configs

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in DEVICE_ARCHS,
        reason=f"needs gfx942/gfx950/gfx1151/gfx1201; device is {ARCH}",
    ),
]

CASES = ("bwd_seqlen_33x72_none", "bwd_dtype_bf16_d128_mha")
LOG2E = np.float32(1.4426950408889634)


def _configs():
    return selected_configs(ARCH) if ARCH in DEVICE_ARCHS else ["none"]


def _check(run, *, expect_ok):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    if expect_ok:
        assert not probs and not run.integrity, (probs, run.integrity)
    else:
        assert probs, "the negative control was not detected"


@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("case_id", CASES)
def test_lse_in_log2_units_is_detected(case_id, config):
    case = get_bwd_case(case_id)
    _check(run_case(case, arch=ARCH, config=config), expect_ok=True)
    run = run_case(
        case,
        arch=ARCH,
        config=config,
        mutate_inputs=lambda lse: (lse * LOG2E).astype(np.float32),
    )
    _check(run, expect_ok=False)


@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("case_id", CASES)
def test_lse_offset_is_detected(case_id, config):
    case = get_bwd_case(case_id)
    _check(run_case(case, arch=ARCH, config=config), expect_ok=True)
    run = run_case(
        case,
        arch=ARCH,
        config=config,
        mutate_inputs=lambda lse: (lse + np.float32(0.05)).astype(np.float32),
    )
    _check(run, expect_ok=False)


# s_q != s_kv, so top-left and bottom-right differ.
BR_CASES = ("bwd_seqlen_33x72_br", "bwd_causal_br_q_lt_kv")


@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("case_id", BR_CASES)
def test_flipped_bottom_right_is_detected(case_id, config):
    case = get_bwd_case(case_id)
    _check(run_case(case, arch=ARCH, config=config), expect_ok=True)

    def flip(plan):
        br = plan.launches[1].scalars["bottom_right"]
        assert br == 1
        return _replace_launch(plan, "main", scalars={"bottom_right": 1 - br})

    _check(run_case(case, arch=ARCH, config=config, mutate_plan=flip), expect_ok=False)


# ---------------------------------------------------------------------------
# the dead-row term of the dS select (body switch sentinel=False)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", _configs())
def test_dropped_dead_row_select_gives_nan(config):
    # Consistent inputs (fully masked rows with -inf LSE): the switched body
    # passes, because prep's row_live gate already zeroes their Dsum.
    case = get_bwd_case("bwd_lse_fully_masked_rows_neg_inf")
    with main_built_with(sentinel=False) as kcache:
        _check(run_case(case, arch=ARCH, config=config, cache=kcache), expect_ok=True)
    # -inf LSE on band-live rows with a NaN O and an ungated Dsum: the shipped
    # body keeps every gradient finite (dQ of those rows exactly 0); without
    # the lse2 < +inf term of the dS select the NaN reaches the gradients.
    case = get_bwd_case("bwd_seqlen_33x72_br")
    rows = dead_rows(case)
    inputs = dead_row_inputs(case, rows, [-np.inf], np.nan)
    hook = ungated_dsum_hook(rows)
    run = run_case(case, arch=ARCH, config=config, inputs=inputs, after_prep=hook)
    _check(run, expect_ok=True)
    assert np.all(run.dq[rows] == 0.0)
    with main_built_with(sentinel=False) as kcache:
        bad = run_case(
            case, arch=ARCH, config=config, inputs=inputs, after_prep=hook,
            cache=kcache,
        )  # fmt: skip
    assert not np.isfinite(bad.dq).all() or not np.isfinite(bad.dk).all()


# Spans long enough that dead rows (rows 0, len / 2 and len - 1) sit in interior
# q tiles of interior kv tiles under every configuration's tile (kN0 up to 256):
# an unmasked 1000 x 1000 d64 case and a top-left causal d128 band (interior
# below the diagonal).
INTERIOR_DEAD_CASES = ("bwd_seqlen_1000x1000_none", "bwd_dtype_bf16_d128_causal_tl")


@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("case_id", INTERIOR_DEAD_CASES)
def test_interior_tiles_keep_the_dead_row_select(case_id, config):
    """With ``edge_tiles = True`` the interior steps drop the per-cell keep
    and keep only the dead-row term of the dS select. A dead row (``-inf``
    LSE, NaN O) whose Dsum is left ungated (NaN) must still give exact-zero
    dS there: every gradient finite, dQ of those rows exactly 0, the rest
    equal to the oracle without them. Dropping the interior select makes dS
    ``0 * (dP - NaN)``, NaN in dK."""
    case = get_bwd_case(case_id)
    rows = dead_rows(case)
    inputs = dead_row_inputs(case, rows, [-np.inf], np.nan)
    run = run_case(
        case, arch=ARCH, config=config, inputs=inputs,
        after_prep=ungated_dsum_hook(rows), overrides={"edge_tiles": True},
    )  # fmt: skip
    assert run.plan.specs["main"].edge_tiles
    for name in ("dq", "dk", "dv"):
        assert np.isfinite(getattr(run, name)).all(), f"non-finite {name}"
    assert np.all(run.dq[rows] == 0.0)
    _check(run, expect_ok=True)


# ---------------------------------------------------------------------------
# grouped heads: direct mode launched without the group sum
# ---------------------------------------------------------------------------

# A GQA group on the unpacked loop, and a GQA decode group (s_q = 1) whose
# query heads the plan packs onto one q tile.
GROUP_CASES = ("bwd_gqa_gqa4_causal_tl", "bwd_decode_skv17_br")


@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("case_id", GROUP_CASES)
def test_missing_group_sum_is_detected(case_id, config):
    case = get_bwd_case(case_id)
    assert case.h_q > case.h_k == case.h_v
    _check(run_case(case, arch=ARCH, config=config), expect_ok=True)

    def ungrouped(plan):
        assert plan.specs["main"].dkv_mode == "direct"
        assert plan.launches[1].scalars["G"] == case.h_q // case.h_k
        if case.s_q == 1:
            assert plan.runtime["pack_heads"] > 1
        return _replace_launch(plan, "main", scalars={"G": 1})

    run = run_case(case, arch=ARCH, config=config, mutate_plan=ungrouped)
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    # dK / dV keep one query head of the group instead of the group sum
    assert any(p.startswith(("dk ", "dv ")) for p in probs), probs


@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("case_id", CASES)
def test_skipped_prep_is_detected(case_id, config):
    case = get_bwd_case(case_id)
    run = run_case(
        case,
        arch=ARCH,
        config=config,
        mutate_plan=lambda plan: _replace_launch(plan, "prep", drop=True),
    )
    _check(run, expect_ok=False)


@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("case_id", CASES)
def test_convert_scale_applied_twice_is_detected(case_id, config):
    case = get_bwd_case(case_id)

    def twice(plan):
        mult = plan.launches[2].scalars["mult"] * plan.runtime["scale"]
        return _replace_launch(plan, "convert", index=0, scalars={"mult": mult})

    run = run_case(case, arch=ARCH, config=config, mutate_plan=twice)
    _check(run, expect_ok=False)


# ---------------------------------------------------------------------------
# atomic dK / dV: the workspace must be zeroed by prep and converted
# ---------------------------------------------------------------------------

ATOMIC_CASE = "bwd_kern_hkv_842_tl_fp16_d64"


def _atomic_case():
    return next(c for c in SUPPLEMENTARY_CASES if c.id == ATOMIC_CASE)


@pytest.mark.parametrize("config", _configs())
def test_skipped_kv_zero_is_detected(config):
    case = _atomic_case()
    _check(run_case(case, arch=ARCH, config=config), expect_ok=True)

    def no_kv_zero(plan):
        assert plan.specs["main"].dkv_mode == "atomic"
        assert plan.launches[0].scalars["zero_kv"] == 1
        return _replace_launch(plan, "prep", scalars={"zero_kv": 0})

    run = run_case(case, arch=ARCH, config=config, mutate_plan=no_kv_zero)
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert any(p.startswith(("non-finite", "dk ", "dv ")) for p in probs), probs


@pytest.mark.parametrize("config", _configs())
def test_skipped_dv_convert_is_detected(config):
    case = _atomic_case()

    def no_dv_convert(plan):
        assert [s.stage for s in plan.launches].count("convert") == 3
        return _replace_launch(plan, "convert", index=2, drop=True)

    run = run_case(case, arch=ARCH, config=config, mutate_plan=no_dv_convert)
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert any(p.startswith("dv ") for p in probs), probs
    assert not any(p.startswith(("dq ", "dk ")) for p in probs), probs


# ---------------------------------------------------------------------------
# padding: the main kernel must read the per-batch lengths
# ---------------------------------------------------------------------------

# Unmasked padded cases, so the padding keys are inside the band: a kernel
# that ignores len_kv lets them into dQ and stores their dK / dV rows. (The
# padding query rows are dead either way: prep gives them the +inf lse2
# sentinel, so ignoring len_q alone is masked by the stats contract.)
PAD_CASES = ("bwd_padding_q_lt_kv_none", "bwd_padding_kv_only_shorter")


@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("case_id", PAD_CASES)
def test_ignored_seq_len_is_detected(case_id, config):
    case = get_bwd_case(case_id)
    assert case.length_mode == "padded"
    _check(run_case(case, arch=ARCH, config=config), expect_ok=True)

    def no_len(plan):
        assert plan.launches[1].scalars["has_len"] == 1
        return _replace_launch(plan, "main", scalars={"has_len": 0})

    run = run_case(case, arch=ARCH, config=config, mutate_plan=no_len)
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    # the padding rows join the sums (dK / dV of valid rows) or get values
    assert probs or run.integrity, "ignored SEQ_LEN counts were not detected"


# ---------------------------------------------------------------------------
# THD: the main kernel must decode the offset table at its width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", _configs())
def test_wrong_offset_width_is_detected(config):
    """An int64 offset table decoded as int32 by the main kernel only (prep
    and the converts keep the right width): the sequences of the main kernel
    move, and the comparison fails."""
    case = get_bwd_case("bwd_ragged_tails_br")
    storage = ThdStorage(offset_dtype="int64")
    _check(run_case(case, arch=ARCH, config=config, thd=storage), expect_ok=True)

    def as_int32(plan):
        assert plan.launches[1].scalars["off64"] == 1
        return _replace_launch(plan, "main", scalars={"off64": 0})

    run = run_case(case, arch=ARCH, config=config, thd=storage, mutate_plan=as_int32)
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert probs or run.integrity, "a misread offset table was not detected"
