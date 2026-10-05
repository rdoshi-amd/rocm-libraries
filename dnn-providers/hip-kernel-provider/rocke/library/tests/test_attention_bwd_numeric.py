# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device numerics of the layout-general attention backward.

Every selected case runs ``prep -> main -> convert`` through the host plan and
the run function, then compares dQ, dK and dV with the float64 oracle on the
stored inputs (:func:`tests.sdpa.bwd_cases.compare_bwd`: global and per-row
bounds, no NaN / Inf, fully masked rows of dQ zero) and checks the integrity
facts of the harness (untouched stride gaps and canaries, an intact workspace
canary, exact-zero batched padding rows).

* ``test_table_case``: the case table plus the supplementary cases;
* ``test_edge_tiles_agree``: the same cases with ``edge_tiles = True``
  (interior tiles skip the per-cell selects) pass, and in direct mode give
  dK / dV bit for bit equal to ``edge_tiles = False`` (atomic dK / dV are
  summed by fp32 atomics in a run-dependent order);
* ``test_neg_inf_lse_on_live_rows`` / ``test_nan_lse_rows`` /
  ``test_dead_rows_with_poisoned_output``: band-live rows inside ``len_q``
  with a ``-inf`` or NaN LSE (and, for the last, a NaN or ``+inf`` O on those
  rows) are dead: no NaN / Inf anywhere, dQ of those rows exactly 0, dK / dV
  equal to the oracle with those rows removed;
* ``test_head_packing``: short query spans of GQA groups; the plan packs the
  group's heads onto the M rows of one q tile when ``G * s_q <= kM0``;
* ``test_hk_ne_hv``: the table's ``h_k != h_v`` cases (exhaustive tier) and
  the THD ``h_k != h_v`` cells run in atomic mode (five launches, dK / dV
  workspace terms);
* ``test_atomic_run_to_run``: two runs of an atomic-mode case agree within
  the tolerance (not bit for bit);
* ``test_runtime_variants_agree``: ``g_split``, the reverse load-balance
  order, the scale placements and an explicit AGPR reservation pass, and in
  direct mode the order and AGPR variants give dK / dV bit for bit equal to
  the plain run (mirror pairing is not built; the plan refuses it);
* ``test_zero_length_sequences``: padded batches with ``len_q = 0`` and / or
  ``len_kv = 0`` (in the middle and last batch, both zero, all but one
  empty): exact-zero gradients of the empty batches, dQ exactly 0 for a
  batch without keys, intact stride gaps and canaries;
* ``test_seq_len_element_stride``: SEQ_LEN counts read at an element stride
  with decoy counts between them;
* ``test_lse_strides_are_honoured``: a non-packed LSE (token-major dense
  rows; head-major THD rows, as ``[T_q, H_q, 1]`` and in the batched shape)
  with NaN gaps is read at the request's strides; the same buffer read at the
  packed default fails the oracle (negative control);
* ``test_ragged_offset_forms``: THD offset tables as int32 / int64 element
  offsets, token offsets and element offsets in units of a multiplier, on a
  storage with NaN tokens around the sequences and padded tokens;
* ``test_ragged_sequence_counts``: THD with SEQ_LEN counts shorter than the
  offset spans (gap tokens between the sequences, never read or written);
* ``test_zero_length_segments``: THD sequences of length 0, including the
  last q and kv sequences, on the compact layout and a NaN-padded storage;
* ``test_ragged_storage_forms`` / ``test_ragged_storage_fuzz``: non-compact
  THD storages (tokens before the first sequence, SEQ_LEN counts shorter
  than the allocated segments, tails, padded tokens, an empty last segment)
  without a caller ``max_total`` (workspace rows by sequence slot) and with
  the storage token count as the bound (rows by storage token), on a direct,
  an atomic and a head-packed case; the fuzz test draws seeded storages;
* ``test_max_total_guard``: understated ``max_total_q`` / ``max_total_kv``:
  memory-safe (canaries intact), exact-zero dQ past the bound, correct rows
  that depend only on tokens inside the bound;
* ``test_token_major_workspace``: the token-major workspace layout passes and
  gives direct-mode dK / dV equal to the head-major run bit for bit;
* ``test_unaligned_strides``: the narrow-access instances (``stage_vec = 1``)
  on BHSD and BSHD views with odd S strides and H strides that are not
  multiples of 8, and on THD storages with an odd token stride, every tensor
  base 2, 4 or 8 bytes off a 16-byte boundary (stride gaps and storage
  outside the sequences hold NaN inputs);
* ``test_narrow_twin_agrees``: a ``stage_vec = 8`` run and its
  ``stage_vec = 1`` twin (same tile, same storage) give dK / dV bit for bit
  equal in direct mode.

Selection (environment):

* ``ROCKE_BWD_TIER``: ``smoke`` (default) | ``full`` | ``exhaustive``;
* ``ROCKE_BWD_STAGE``: run cases with ``stage_of(case) <=`` this value
  (default: every built stage);
* ``ROCKE_BWD_CONFIG``: ``debug`` | ``default`` | ``multi`` | ``both``
  (default; every distinct configuration of the arch) | ``custom`` (knobs
  per head size and ``stage_vec`` from ``ROCKE_BWD_CUSTOM_KNOBS``, see
  ``tests/sdpa/bwd_kernel_cases.py``);
* ``ROCKE_BWD_SHARD``: ``i/n`` (stable hash of the case id).

Skipped unless the visible device is gfx942, gfx950, gfx1151 or gfx1201.
"""

from __future__ import annotations

import dataclasses
import math
import os

import numpy as np
import pytest

from ._attention_bwd_harness import (
    ARCH,
    DEVICE_ARCHS,
    ThdStorage,
    dead_row_inputs,
    dead_rows,
    narrow_inputs,
    plan_case,
    run_case,
)
from .sdpa.bwd_cases import SdpaBwdReference, compare_bwd, get_bwd_case
from .sdpa.bwd_kernel_cases import (
    HEAD_PACK_CASES,
    HK_NE_HV_TABLE_CASES,
    NARROW_CASES,
    NARROW_TWIN_CASES,
    SUPPLEMENTARY_CASES,
    THD_HK_NE_HV_CASES,
    THD_STORAGE_CASES,
    THD_STORAGE_FORMS,
    THD_ZERO_LENGTH_CASES,
    ZERO_LENGTH_CASES,
    selected_cases,
    resolve_storage_form,
    selected_configs,
    shard_of,
    thd_fuzz_count,
    thd_storage_fuzz,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH not in DEVICE_ARCHS,
        reason=f"needs gfx942/gfx950/gfx1151/gfx1201; device is {ARCH}",
    ),
]

# dK / dV of the edge_tiles = False runs, reused by the edge-tile agreement test.
_RUNS: dict = {}


def _params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    return [
        pytest.param(case, cfg, id=f"{case.id}-{cfg}")
        for case in selected_cases()
        for cfg in selected_configs(ARCH)
    ]


def _configs():
    return selected_configs(ARCH) if ARCH in DEVICE_ARCHS else ["none"]


def _passes(run):
    probs = compare_bwd(run.case, run.dq, run.dk, run.dv, run.ref)
    assert not probs, probs
    assert not run.integrity, run.integrity


@pytest.mark.parametrize("case,config", _params())
def test_table_case(case, config):
    run = run_case(case, arch=ARCH, config=config)
    _passes(run)
    _RUNS[(case.id, config)] = (run.dk, run.dv)


@pytest.mark.parametrize("case,config", _params())
def test_edge_tiles_agree(case, config):
    key = (case.id, config)
    if key not in _RUNS:
        run = run_case(case, arch=ARCH, config=config)
        _passes(run)
        _RUNS[key] = (run.dk, run.dv)
    run = run_case(case, arch=ARCH, config=config, overrides={"edge_tiles": True})
    assert run.plan.specs["main"].edge_tiles
    _passes(run)
    if run.plan.specs["main"].dkv_mode != "direct":
        return
    dk, dv = _RUNS[key]
    assert np.array_equal(run.dk, dk), "dK differs across edge_tiles"
    assert np.array_equal(run.dv, dv), "dV differs across edge_tiles"


# ---------------------------------------------------------------------------
# dead rows: -inf / NaN LSE on band-live rows, poisoned O
# ---------------------------------------------------------------------------

# Cases whose every row inside len_q attends at least one key (band-live).
DEAD_ROW_CASES = ("bwd_seqlen_33x72_br", "bwd_dtype_bf16_d128_causal_tl")


def _dead_rows_case(case_id, config, lse_values, o_value=None):
    case = get_bwd_case(case_id)
    rows = dead_rows(case)
    inputs = dead_row_inputs(case, rows, lse_values, o_value)
    run = run_case(case, arch=ARCH, config=config, inputs=inputs)
    for name in ("dq", "dk", "dv"):
        assert np.isfinite(getattr(run, name)).all(), f"non-finite {name}"
    assert np.all(run.dq[rows] == 0.0), "dQ of dead rows is not exactly 0"
    _passes(run)
    return run


@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("case_id", DEAD_ROW_CASES)
def test_neg_inf_lse_on_live_rows(case_id, config):
    _dead_rows_case(case_id, config, [-np.inf])


@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("case_id", DEAD_ROW_CASES)
def test_nan_lse_rows(case_id, config):
    _dead_rows_case(case_id, config, [np.nan])


@pytest.mark.parametrize("o_value", [np.nan, np.inf], ids=["nan_o", "inf_o"])
@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("case_id", DEAD_ROW_CASES)
def test_dead_rows_with_poisoned_output(case_id, config, o_value):
    _dead_rows_case(case_id, config, [np.nan, -np.inf], o_value)


# ---------------------------------------------------------------------------
# head packing
# ---------------------------------------------------------------------------


def _pack_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [
        pytest.param(case, cfg, id=f"{case.id}-{cfg}")
        for case in HEAD_PACK_CASES
        if shard_of(case.id, n) == i
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case,config", _pack_params())
def test_head_packing(case, config):
    group = case.h_q // case.h_k
    plan = plan_case(case, arch=ARCH, config=config)
    spec = plan.specs["main"]
    packs = group * case.s_q <= spec.block_m
    assert spec.head_pack == packs
    assert plan.runtime["pack_heads"] == (group if packs else 0)
    run = run_case(case, arch=ARCH, config=config)
    _passes(run)


# ---------------------------------------------------------------------------
# h_k != h_v, atomic dK / dV, runtime knobs
# ---------------------------------------------------------------------------


def _round_robin(keys):
    """The keys of this ``ROCKE_BWD_SHARD`` (sorted, dealt round robin), so a
    small set lands in every shard when it has at least one key per shard."""
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [k for j, k in enumerate(sorted(keys)) if j % n == i]


def _hk_ne_hv_params():
    # The table's two cases (batched) and the THD cells, dealt over the shards.
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    return [
        pytest.param(cid, cfg, id=f"{cid}-{cfg}")
        for cid in _round_robin(HK_NE_HV_TABLE_CASES + THD_HK_NE_HV_CASES)
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case_id,config", _hk_ne_hv_params())
def test_hk_ne_hv(case_id, config):
    case = _case(case_id)
    assert case.h_k != case.h_v
    plan = plan_case(case, arch=ARCH, config=config)
    assert plan.specs["main"].dkv_mode == "atomic"
    assert [s.stage for s in plan.launches] == ["prep", "main"] + ["convert"] * 3
    main = plan.launches[1]
    assert main.pointers["dK"] is None and main.pointers["dV"] is None
    assert {"WS_DK", "WS_DV"} <= set(plan.workspace_layout)
    _passes(run_case(case, arch=ARCH, config=config))


# (case id, runtime / spec overrides): an h_k != h_v case and a GQA group
# split over two CTAs (atomic dK / dV).
RUN_TO_RUN = (
    ("bwd_kern_hkv_842_tl_fp16_d64", {}),
    ("bwd_gqa_gqa4_causal_tl", {"g_split": 2}),
)


def _run_to_run_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [
        pytest.param(cid, ov, cfg, id=f"{cid}-{cfg}")
        for cid, ov in RUN_TO_RUN
        if shard_of(cid, n) == i
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case_id,overrides,config", _run_to_run_params())
def test_atomic_run_to_run(case_id, overrides, config):
    case = _case(case_id)
    first = run_case(case, arch=ARCH, config=config, overrides=overrides)
    assert first.plan.specs["main"].dkv_mode == "atomic"
    _passes(first)
    second = run_case(case, arch=ARCH, config=config, overrides=overrides)
    _passes(second)
    # the second run measured against the first: within the tolerance
    ref = SdpaBwdReference(
        dq=first.dq, dk=first.dk, dv=first.dv, delta=first.ref.delta,
        valid_q=first.ref.valid_q, valid_kv=first.ref.valid_kv,
        dead_q=first.ref.dead_q,
    )  # fmt: skip
    probs = compare_bwd(case, second.dq, second.dk, second.dv, ref)
    assert not probs, probs


def _case(case_id):
    """A table case or a supplementary kernel case by id."""
    for c in SUPPLEMENTARY_CASES:
        if c.id == case_id:
            return c
    return get_bwd_case(case_id)


# Cases of the runtime-variant test: an MHA band (direct), a GQA group
# (direct; g_split = 2 makes it atomic) and an h_k != h_v case (atomic).
VARIANT_CASES = (
    "bwd_seqlen_33x72_br",
    "bwd_gqa_gqa4_causal_tl",
    "bwd_kern_hkv_842_tl_fp16_d64",
)
# name -> (overrides, request fields, bit for bit in direct mode)
RUNTIME_VARIANTS = {
    "g_split2": ({"g_split": 2}, None, False),
    "reverse": ({"lb_order": "reverse"}, None, True),
    "scale_convert": ({"scale_placement": "convert"}, None, False),
    "scale_dq_before_atomic": ({"scale_placement": "dq_before_atomic"}, None, False),
    "agpr128": ({"agpr_alloc": (128, 128)}, None, True),
}


def _variant_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    out = []
    for cid in VARIANT_CASES:
        if shard_of(cid, n) != i:
            continue
        case = _case(cid)
        group = math.gcd(case.h_q // case.h_k, case.h_q // case.h_v)
        for name in RUNTIME_VARIANTS:
            if name == "g_split2" and group % 2:
                continue
            if name == "agpr128" and ARCH not in ("gfx942", "gfx950"):
                continue  # AGPRs exist on the MFMA targets only
            for cfg in selected_configs(ARCH):
                out.append(pytest.param(cid, name, cfg, id=f"{cid}-{name}-{cfg}"))
    return out


@pytest.mark.parametrize("case_id,variant,config", _variant_params())
def test_runtime_variants_agree(case_id, variant, config):
    case = _case(case_id)
    key = (case_id, config, "plain")
    if key not in _RUNS:
        base = run_case(case, arch=ARCH, config=config)
        _passes(base)
        _RUNS[key] = (base.plan.specs["main"].dkv_mode, base.dk, base.dv)
    mode, dk, dv = _RUNS[key]
    overrides, request, bitwise = RUNTIME_VARIANTS[variant]
    run = run_case(case, arch=ARCH, config=config, overrides=overrides, request=request)
    for k, v in overrides.items():
        if k in run.plan.runtime:
            assert run.plan.runtime[k] == v, (k, run.plan.runtime[k])
        else:
            assert getattr(run.plan.specs["main"], k) == v
    _passes(run)
    if bitwise and mode == "direct":
        assert run.plan.specs["main"].dkv_mode == "direct"
        assert np.array_equal(run.dk, dk), f"dK differs across {variant}"
        assert np.array_equal(run.dv, dv), f"dV differs across {variant}"


# ---------------------------------------------------------------------------
# padding: zero-length batches, the SEQ_LEN element stride
# ---------------------------------------------------------------------------


def _zero_length_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    return [
        pytest.param(case, cfg, id=f"{case.id}-{cfg}")
        for case in ZERO_LENGTH_CASES
        if shard_of(case.id, n) == i
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case,config", _zero_length_params())
def test_zero_length_sequences(case, config):
    """Padded batches with ``len_q = 0`` / ``len_kv = 0``: every gradient row of
    an empty batch is an exact zero, a batch without keys has dQ exactly 0 on
    its live query rows, the other batches pass the oracle, and no byte outside
    the gradient views or past the workspace changes."""
    lq, lk = case.lens()
    assert 0 in lq + lk
    plan = plan_case(case, arch=ARCH, config=config)
    assert plan.launches[1].scalars["has_len"] == 1
    run = run_case(case, arch=ARCH, config=config)
    _passes(run)
    for bi, (nq, nk) in enumerate(zip(lq, lk)):
        if nq == 0:
            assert np.all(run.dq[bi] == 0.0), f"dQ of empty batch {bi}"
        if nk == 0:
            assert np.all(run.dk[bi] == 0.0), f"dK of empty batch {bi}"
            assert np.all(run.dv[bi] == 0.0), f"dV of empty batch {bi}"
            assert np.all(run.dq[bi] == 0.0), f"dQ of batch {bi} without keys"


# Each count followed by (stride - 1) decoys: valid lengths of other batches,
# so a kernel reading SEQ at the wrong element changes the result.
SEQ_STRIDE = 3


def _strided_lengths(lengths):
    out = {}
    for role, counts in lengths.items():
        a = np.zeros(len(counts) * SEQ_STRIDE, np.int32)
        a[::SEQ_STRIDE] = counts
        for j in range(1, SEQ_STRIDE):
            a[j::SEQ_STRIDE] = np.roll(counts, j)
        out[role] = a
    return out


def _seq_stride_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    i, n = (int(x) for x in os.environ.get("ROCKE_BWD_SHARD", "0/1").split("/"))
    cases = (ZERO_LENGTH_CASES[0], _case("bwd_padding_q_lt_kv_br"))
    return [
        pytest.param(case, cfg, id=f"{case.id}-{cfg}")
        for case in cases
        if shard_of(case.id, n) == i
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("case,config", _seq_stride_params())
def test_seq_len_element_stride(case, config):
    """SEQ_LEN tensors read at an element stride (``len_stride``) by every
    kernel of the plan: the decoys between the counts are never read."""
    run = run_case(
        case,
        arch=ARCH,
        config=config,
        request={"seq_len_stride": SEQ_STRIDE},
        mutate_lengths=_strided_lengths,
    )
    assert run.plan.launches[1].scalars["len_stride"] == SEQ_STRIDE
    _passes(run)


# ---------------------------------------------------------------------------
# LSE strides: a non-packed LSE is read at the request's strides
# ---------------------------------------------------------------------------

# form -> case id. ``dense_bsh``: token-major [B, S, H] rows with one gap
# element per token and a gap between batches; ``thd_th1``: [T_q, H_q, 1]
# head-major with a gap after every head; ``thd_bhs1``: the same storage
# declared in the batched shape, whose (deliberately odd) B stride is unused.
LSE_FORMS = {
    "dense_bsh": "bwd_seqlen_33x72_br",
    "thd_th1": "bwd_ragged_gqa4_br",
    "thd_bhs1": "bwd_ragged_gqa4_br",
}
_LSE_PAD = 64  # NaN elements after the last LSE element


def _lse_layout(case, form):
    """``(request fields, (l_b, l_h, l_t), scatter)`` of an LSE form;
    ``scatter(flat packed lse)`` returns the NaN-gapped device buffer."""
    h = case.h_q
    if form == "dense_bsh":
        b, s = case.b, case.s_q
        l_t = h + 1
        l_b = s * l_t + 5
        fields = dict(lse_dims=(b, h, s, 1), lse_strides=(l_b, 1, l_t, 1))
        want = (l_b, 1, l_t)
        idx = (
            np.arange(b)[:, None, None] * l_b
            + np.arange(h)[None, :, None]
            + np.arange(s)[None, None, :] * l_t
        )
        shape = (b, h, s)
    else:
        t = int(sum(case.lens()[0]))  # the compact THD storage: one row per token
        l_h = t + 3
        if form == "thd_th1":
            fields = dict(lse_dims=(t, h, 1), lse_strides=(1, l_h, 1))
        else:
            fields = dict(
                lse_dims=(case.b, h, case.s_q, 1), lse_strides=(77, l_h, 1, 1)
            )
        want = (0, l_h, 1)
        idx = np.arange(t)[:, None] + np.arange(h)[None, :] * l_h
        shape = (t, h)

    def scatter(lse):
        buf = np.full(int(idx.max()) + 1 + _LSE_PAD, np.nan, np.float32)
        buf[idx] = np.reshape(lse, shape)
        return buf

    return fields, want, scatter


def _lse_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    return [
        pytest.param(form, cfg, id=f"{form}-{cfg}")
        for form in _round_robin(LSE_FORMS)
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("form,config", _lse_params())
def test_lse_strides_are_honoured(form, config):
    """A non-packed LSE (NaN in every gap) read at the request's strides
    passes the oracle; the same buffer read at the packed default fails it
    (negative control: the test detects an LSE stride that is ignored)."""
    case = _case(LSE_FORMS[form])
    assert (case.length_mode == "ragged") == form.startswith("thd")
    fields, want, scatter = _lse_layout(case, form)
    run = run_case(
        case, arch=ARCH, config=config, request=fields, mutate_inputs=scatter
    )
    prep = run.plan.launches[0]
    assert prep.stage == "prep"
    assert (prep.scalars["l_b"], prep.scalars["l_h"], prep.scalars["l_t"]) == want
    _passes(run)
    ignored = run_case(case, arch=ARCH, config=config, mutate_inputs=scatter)
    assert compare_bwd(
        case, ignored.dq, ignored.dk, ignored.dv, ignored.ref
    ), "the packed-default read of the gapped LSE passed the oracle"


# ---------------------------------------------------------------------------
# ragged (THD) sequences: offset forms, counts, zero lengths, workspace bound
# ---------------------------------------------------------------------------

# Offset table forms: int32 / int64 x element offsets (multiplier 1), token
# offsets (multiplier = the token stride, one token stride for every tensor)
# and element offsets in units of 8 elements. The storage puts tokens before
# the first and after the last sequence and pads every token; those storage
# elements hold NaN inputs and must keep the gradient sentinel.
OFFSET_FORMS = {
    f"{dt}_{units}": ThdStorage(offset_dtype=dt, **kw)
    for dt in ("int32", "int64")
    for units, kw in (
        ("elements", dict(lead=3, tail=5, token_pad=8)),
        ("tokens", dict(lead=2, tail=1, uniform=True, multiplier=0)),
        ("units8", dict(lead=1, tail=4, token_pad=16, multiplier=8)),
    )
}
# A GQA group (direct; Q and K token strides differ unless uniform) and an
# h_k != h_v cell (atomic dK / dV through the kv token rows of the workspace).
OFFSET_FORM_CASES = ("bwd_ragged_gqa4_br", "bwd_kern_thd_hkv_842_tl_fp16_d64")


def _thd_params(keys):
    """``(case id, variant, config)`` params of the THD tests, dealt round
    robin over the shards by ``case id / variant``."""
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, None, id="no-device")]
    return [
        pytest.param(cid, var, cfg, id=f"{cid}-{var}-{cfg}")
        for cid, var in (k.split("/") for k in _round_robin(keys))
        for cfg in selected_configs(ARCH)
    ]


def _token_stride(run, role):
    return run.plan.launches[1].scalars[f"{role}_t"]


@pytest.mark.parametrize(
    "case_id,form,config",
    _thd_params([f"{c}/{f}" for c in OFFSET_FORM_CASES for f in OFFSET_FORMS]),
)
def test_ragged_offset_forms(case_id, form, config):
    """int32 and int64 offset tables, token and element units (times a
    multiplier): the kernels decode ``token = off * mult / token_stride`` in
    64 bits and touch nothing outside the sequences."""
    storage = OFFSET_FORMS[form]
    case = _case(case_id)
    run = run_case(case, arch=ARCH, config=config, thd=storage)
    for step in run.plan.launches:
        assert step.scalars["off64"] == (storage.offset_dtype == "int64")
    main = run.plan.launches[1].scalars
    mult = storage.multiplier or _token_stride(run, "q")
    assert (main["q_mult"], main["kv_mult"]) == (mult, mult)
    assert (main["q_div"], main["kv_div"]) == (
        _token_stride(run, "q"),
        _token_stride(run, "k"),
    )
    _passes(run)


# Per-sequence counts with THD offsets: every sequence ends ``gap`` tokens
# before the next offset (the length is the count, min-ed with the offset
# span); the gap tokens hold NaN inputs and must keep the sentinel.
COUNT_FORMS = {
    "gap4": ThdStorage(gap=4, tail=2, seq_len_counts=True),
    "gap1_int64": ThdStorage(lead=2, gap=1, seq_len_counts=True, offset_dtype="int64"),
}
COUNT_CASES = (
    "bwd_ragged_tails_br",
    "bwd_kern_thd_hkv_842_tl_fp16_d64",
    "bwd_kern_thd_zero_len_last_mha_br_fp16_d64",
)


def _thd_case(case_id):
    for c in THD_ZERO_LENGTH_CASES + HEAD_PACK_CASES:
        if c.id == case_id:
            return c
    return _case(case_id)


@pytest.mark.parametrize(
    "case_id,form,config",
    _thd_params([f"{c}/{f}" for c in COUNT_CASES for f in COUNT_FORMS]),
)
def test_ragged_sequence_counts(case_id, form, config):
    """THD with SEQ_LEN counts shorter than the offset spans: the length is
    ``min(count, offset span)``, and the tokens between the end of a sequence
    and the next offset are neither read nor written."""
    case = _thd_case(case_id)
    run = run_case(case, arch=ARCH, config=config, thd=COUNT_FORMS[form])
    assert all(step.scalars["has_len"] == 1 for step in run.plan.launches)
    _passes(run)


def _check_storage_run(run, storage):
    """A THD run on a non-compact storage: the plan's row mode matches the
    bound mode, the reported workspace holds every row, and the gradients
    pass (integrity: storage tokens outside the sequences untouched, the
    workspace canary intact)."""
    main = run.plan.launches[1].scalars
    if storage.bound == "none":
        assert all(step.scalars["ws_seg"] == 1 for step in run.plan.launches)
        b, s_q = len(run.case.lens()[0]), main["S_q_max"]
        assert main["ws_rows_q"] == b * s_q
    else:
        assert all(step.scalars["ws_seg"] == 0 for step in run.plan.launches)
    _passes(run)


@pytest.mark.parametrize(
    "case_id,form,config",
    _thd_params([f"{c}/{f}" for c in THD_STORAGE_CASES for f in THD_STORAGE_FORMS]),
)
def test_ragged_storage_forms(case_id, form, config):
    """Non-compact THD storages with and without a caller bound: the
    workspace is indexed by sequence slot without one, so tokens before,
    between or after the sequences never push a row past the reported
    workspace."""
    case = _thd_case(case_id)
    storage = ThdStorage(**resolve_storage_form(case, THD_STORAGE_FORMS[form]))
    run = run_case(case, arch=ARCH, config=config, thd=storage)
    _check_storage_run(run, storage)


def _fuzz_keys():
    n = thd_fuzz_count()
    return [
        f"{c}/{name}"
        for k, c in enumerate(THD_STORAGE_CASES)
        for name, _kw in thd_storage_fuzz(k + 1, n)
    ]


@pytest.mark.parametrize("case_id,form,config", _thd_params(_fuzz_keys()))
def test_ragged_storage_fuzz(case_id, form, config):
    """Seeded non-compact THD storages (lead / gap / tail / padding / offset
    width and units / bound mode drawn per seed) pass."""
    k = THD_STORAGE_CASES.index(case_id)
    case = _thd_case(case_id)
    kw = dict(thd_storage_fuzz(k + 1, thd_fuzz_count()))[form]
    storage = ThdStorage(**resolve_storage_form(case, kw))
    run = run_case(case, arch=ARCH, config=config, thd=storage)
    _check_storage_run(run, storage)


# THD storages of the zero-length test: the compact case layout (the last
# sequence ends at the last token of every tensor) and a storage with NaN
# tokens before the first and after the last sequence.
ZERO_STORAGES = {
    "compact": ThdStorage(),
    "padded": ThdStorage(lead=2, tail=3, token_pad=8, offset_dtype="int64"),
}


def _segments(lens):
    o = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    return [slice(o[i], o[i + 1]) for i in range(len(lens))]


@pytest.mark.parametrize(
    "case_id,storage,config",
    _thd_params([f"{c.id}/{s}" for c in THD_ZERO_LENGTH_CASES for s in ZERO_STORAGES]),
)
def test_zero_length_segments(case_id, storage, config):
    """THD sequences of length 0 (the last q and kv sequences, middle ones, all
    but one): dK / dV of a kv sequence without queries and dQ of a q sequence
    without keys are exactly 0, the rest equals the oracle, and nothing outside
    the sequences (nor past the last offset) is touched."""
    case = _thd_case(case_id)
    lq, lk = case.lens()
    assert 0 in lq + lk
    run = run_case(case, arch=ARCH, config=config, thd=ZERO_STORAGES[storage])
    assert run.plan.launches[1].scalars["has_len"] == 0
    _passes(run)
    for qs, ks, nq, nk in zip(_segments(lq), _segments(lk), lq, lk):
        if nq == 0:
            assert np.all(run.dk[ks] == 0.0), "dK of a kv sequence without queries"
            assert np.all(run.dv[ks] == 0.0), "dV of a kv sequence without queries"
        if nk == 0:
            assert np.all(run.dq[qs] == 0.0), "dQ of a q sequence without keys"


# Understated workspace bounds: the last sequence crosses max_total_q (and,
# for the atomic case, max_total_kv).
MAX_TOTAL_CASES = (
    "bwd_ragged_tails_tl",
    "bwd_ragged_gqa4_br",
    "bwd_kern_thd_hkv_842_tl_fp16_d64",
)


@pytest.mark.parametrize(
    "case_id,bound,config", _thd_params([f"{c}/under" for c in MAX_TOTAL_CASES])
)
def test_max_total_guard(case_id, bound, config):
    """An understated ``max_total_q`` (and ``max_total_kv`` in atomic mode) is
    memory-safe: the workspace canary and every byte outside the sequences
    stay intact, outputs are finite, the dQ rows of tokens past the bound are
    exact zeros, and the rows that depend only on tokens inside the bound
    (dQ of those tokens; dK / dV of every sequence but the last) are correct.
    Both bounds are passed (a single bound is not used: rows stay per-sequence
    slots); in direct mode the kv bound is the exact kv token count."""
    case = _case(case_id)
    lq, lk = case.lens()
    bound_q = sum(lq[:-1]) + lq[-1] // 2
    atomic = case.h_k != case.h_v
    bound_kv = sum(lk[:-1]) + lk[-1] // 2 if atomic else sum(lk)
    storage = ThdStorage(max_total_q=bound_q, max_total_kv=bound_kv)
    run = run_case(case, arch=ARCH, config=config, thd=storage)
    main = run.plan.launches[1].scalars
    assert main["ws_rows_q"] == bound_q and main["ws_seg"] == 0
    assert main["ws_rows_kv"] == (bound_kv if atomic else 0)
    assert not run.integrity, run.integrity
    for name in ("dq", "dk", "dv"):
        assert np.isfinite(getattr(run, name)).all(), f"non-finite {name}"
    assert np.all(run.dq[bound_q:] == 0.0), "dQ past max_total_q is not zero"
    if atomic:
        assert np.all(run.dk[bound_kv:] == 0.0), "dK past max_total_kv"
        assert np.all(run.dv[bound_kv:] == 0.0), "dV past max_total_kv"
    ref = run.ref
    valid_q = np.array(ref.valid_q, copy=True)
    valid_q[bound_q:] = False
    valid_kv = np.array(ref.valid_kv, copy=True)
    valid_kv[sum(lk[:-1]) :] = False
    dead_q = np.array(ref.dead_q, copy=True)
    dead_q[bound_q:] = False
    inside = dataclasses.replace(ref, valid_q=valid_q, valid_kv=valid_kv, dead_q=dead_q)
    probs = compare_bwd(case, run.dq, run.dk, run.dv, inside)
    assert not probs, probs


# Token-major workspace ([rows][h][X]) against the head-major default: a THD
# GQA group (direct), a THD h_k != h_v cell (atomic dK / dV), a packed THD
# group and a batched band case.
TOKEN_MAJOR_CASES = (
    "bwd_ragged_gqa4_br",
    "bwd_kern_thd_hkv_842_br_fp16_d64",
    "bwd_kern_pack_thd_sq2_g4_br_fp16_d64",
    "bwd_seqlen_33x72_br",
)


@pytest.mark.parametrize(
    "case_id,layout,config",
    _thd_params([f"{c}/token_major" for c in TOKEN_MAJOR_CASES]),
)
def test_token_major_workspace(case_id, layout, config):
    """``ws_layout = "token_major"`` in prep, main and the converts: the run
    passes, and in direct mode dK / dV equal the head-major run bit for bit
    (the stats rows hold the same values at other addresses)."""
    case = _thd_case(case_id)
    base = run_case(case, arch=ARCH, config=config)
    _passes(base)
    run = run_case(case, arch=ARCH, config=config, overrides={"ws_layout": layout})
    assert run.plan.specs["main"].ws_layout == layout
    assert run.plan.specs["main"].aux_spec("prep").ws_layout == layout
    assert run.plan.workspace_bytes == base.plan.workspace_bytes
    _passes(run)
    if run.plan.specs["main"].dkv_mode == "direct":
        assert np.array_equal(run.dk, base.dk), "dK differs across ws_layout"
        assert np.array_equal(run.dv, base.dv), "dV differs across ws_layout"


# ---------------------------------------------------------------------------
# narrow access (stage_vec = 1): any B/H/S stride, any element-aligned base
# ---------------------------------------------------------------------------


def _narrow_params():
    """``(cell index, config)`` of the narrow cells of ``ROCKE_BWD_TIER``,
    dealt round robin over the shards."""
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    tier = os.environ.get("ROCKE_BWD_TIER", "smoke")
    keys = [c.id for c, _v, _b in NARROW_CASES if tier in c.tags]
    index = {c.id: i for i, (c, _v, _b) in enumerate(NARROW_CASES)}
    return [
        pytest.param(index[cid], cfg, id=f"{cid}-{cfg}")
        for cid in _round_robin(keys)
        for cfg in selected_configs(ARCH)
    ]


@pytest.mark.parametrize("cell,config", _narrow_params())
def test_unaligned_strides(cell, config):
    """Odd S strides, H strides that are not multiples of 8 and tensor bases 2,
    4 or 8 bytes off a 16-byte boundary: the plan picks the narrow-access
    instances for prep, main and the converts, and the run passes with every
    stride gap and every byte outside the views untouched."""
    case, view, base_bytes = NARROW_CASES[cell]
    base = base_bytes // 2
    if view == "thd":
        stride_pad = 3  # every token stride odd
        run = run_case(
            case,
            arch=ARCH,
            config=config,
            thd=ThdStorage(token_pad=stride_pad, base_offset=base, lead=1, tail=2),
        )
        main = run.plan.launches[1].scalars
        assert main["q_t"] % 2 == 1 and main["k_t"] % 2 == 1
    else:
        inputs, request = narrow_inputs(case, view, base)
        run = run_case(case, arch=ARCH, config=config, inputs=inputs, request=request)
        main = run.plan.launches[1].scalars
        assert main["q_t"] % 2 == 1 and main["q_h"] % 8 != 0
    for stage in ("main", "prep", "convert"):
        assert run.plan.specs[stage].stage_vec == 1, stage
    assert all("_sv1" in step.kernel for step in run.plan.launches)
    _passes(run)


def _twin_params():
    if ARCH not in DEVICE_ARCHS:
        return [pytest.param(None, None, id="no-device")]
    return [
        pytest.param(cid, cfg, id=f"{cid}-{cfg}")
        for cid in _round_robin(NARROW_TWIN_CASES)
        for cfg in selected_configs(ARCH)
    ]


# Spec fields that come from the request, not from the tuning knobs.
_REQUEST_FIELDS = ("head_size", "dtype", "seq_mode", "dkv_mode", "stage_vec",
                   "mask_class", "name")  # fmt: skip


@pytest.mark.parametrize("case_id,config", _twin_params())
def test_narrow_twin_agrees(case_id, config):
    """A ``stage_vec = 8`` run and its ``stage_vec = 1`` twin on the same tile
    and storage: direct-mode dK / dV agree bit for bit (only the global access
    width differs)."""
    case = _case(case_id)
    narrow = run_case(case, arch=ARCH, config=config, request={"tensor_alignment": 8})
    spec = narrow.plan.specs["main"]
    assert spec.stage_vec == 1 and spec.dkv_mode == "direct"
    _passes(narrow)
    knobs = {
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec)
        if f.name not in _REQUEST_FIELDS
    }
    wide = run_case(case, arch=ARCH, config=config, overrides=knobs)
    wspec = wide.plan.specs["main"]
    assert wspec.stage_vec == 8
    assert dataclasses.replace(wspec, stage_vec=1) == spec
    _passes(wide)
    assert np.array_equal(narrow.dk, wide.dk), "dK differs across stage_vec"
    assert np.array_equal(narrow.dv, wide.dv), "dV differs across stage_vec"
