# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CTA work decode of the backward main kernels (``_attention_bwd_worklist``).

The emitters run on the typed host evaluator (the exact integer formulas the
kernel executes):

* every load-balance order (natural, reverse, mirror pairing, odd and even
  tile counts, a single tile) visits each kv tile of a (unit, batch) exactly
  once over the grid, in the documented order; the decode without pairing
  (the main kernel's) is one kv tile per CTA;
* the head unit / split decode covers every ``y`` of ``units * g_split``;
* direct mode maps a unit to its kv head and whole group; atomic mode maps
  every (unit, split) to disjoint query-head ranges that cover all ``h_q``
  heads, each sharing one K head ``h / gk`` and one V head ``h / gv``;
* the emitters also build on the IR builder (CTA-uniform integer ops only).
"""

from __future__ import annotations

import math

import pytest
from rocke.core.ir import I32, IRBuilder

from kernels.common._attention_bwd_worklist import (
    LB_MIRROR,
    LB_NATURAL,
    LB_REVERSE,
    decode_cta,
    head_range,
    main_grid_x,
)

from ._attention_bwd_int_eval import IntEvalBuilder, i32


def _tiles(n_kv_tiles, lb_order, pair, *, y=0, g_split=None):
    """kv tiles each CTA x visits, through the emitted decode (``pair=None``:
    the decode without mirror pairing, as the main kernel emits it)."""
    b = IntEvalBuilder()
    out = []
    for x in range(main_grid_x(n_kv_tiles, pair or 0)):
        w = decode_cta(
            b,
            x=i32(b, x),
            y=i32(b, y),
            z=i32(b, 0),
            n_kv_tiles=i32(b, n_kv_tiles),
            lb_order=i32(b, lb_order),
            pair=None if pair is None else i32(b, pair),
            g_split=None if g_split is None else i32(b, g_split),
        )
        assert w.n_pass.v in (1, 2)
        out.append([w.kv_tile(b, i32(b, p)).v for p in range(w.n_pass.v)])
    return out


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 9, 17])
@pytest.mark.parametrize(
    "lb_order,pair",
    [(LB_NATURAL, 0), (LB_REVERSE, 0), (LB_MIRROR, 1)],
    ids=["natural", "reverse", "mirror"],
)
def test_every_order_visits_each_kv_tile_once(n, lb_order, pair):
    per_cta = _tiles(n, lb_order, pair)
    flat = [t for ts in per_cta for t in ts]
    assert sorted(flat) == list(range(n))
    if lb_order == LB_NATURAL:
        assert per_cta == [[t] for t in range(n)]
    elif lb_order == LB_REVERSE:
        assert per_cta == [[n - 1 - t] for t in range(n)]
    else:
        assert len(per_cta) == math.ceil(n / 2)
        for t, ts in enumerate(per_cta):
            assert ts == ([t] if t == n - 1 - t else [t, n - 1 - t])


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_decode_without_pairing_is_one_tile_per_cta(n):
    # the main kernel's decode: natural or reverse, one kv tile per CTA
    assert _tiles(n, LB_NATURAL, None) == _tiles(n, LB_NATURAL, 0)
    assert _tiles(n, LB_REVERSE, None) == _tiles(n, LB_REVERSE, 0)
    assert all(len(ts) == 1 for ts in _tiles(n, LB_REVERSE, None))


def test_pairing_takes_precedence_over_the_order():
    # pair = 1 always visits (t, n - 1 - t), whatever lb_order says
    for lb in (LB_NATURAL, LB_REVERSE, LB_MIRROR):
        assert _tiles(6, lb, 1) == [[0, 5], [1, 4], [2, 3]]


def test_main_grid_x():
    assert [main_grid_x(n, 0) for n in range(5)] == [0, 1, 2, 3, 4]
    assert [main_grid_x(n, 1) for n in range(5)] == [0, 1, 1, 2, 2]
    with pytest.raises(ValueError):
        main_grid_x(-1, 0)


@pytest.mark.parametrize("units,g_split", [(1, 1), (3, 1), (4, 2), (2, 4), (5, 3)])
def test_unit_and_split_decode(units, g_split):
    b = IntEvalBuilder()
    seen = set()
    for y in range(units * g_split):
        w = decode_cta(
            b, x=i32(b, 0), y=i32(b, y), z=i32(b, 7), n_kv_tiles=i32(b, 1),
            lb_order=i32(b, 0), pair=i32(b, 0), g_split=i32(b, g_split),
        )  # fmt: skip
        assert 0 <= w.split.v < g_split and w.batch.v == 7
        seen.add((w.unit.v, w.split.v))
    assert seen == {(u, s) for u in range(units) for s in range(g_split)}
    # without a split the unit is y
    w = decode_cta(
        b, x=i32(b, 0), y=i32(b, 5), z=i32(b, 0), n_kv_tiles=i32(b, 1),
        lb_order=i32(b, 0), pair=i32(b, 0),
    )  # fmt: skip
    assert (w.unit.v, w.split.v) == (5, 0)


def _heads(h_q, h_k, h_v, g_split):
    """Query heads of every (unit, split), through the emitted head map."""
    gk, gv = h_q // h_k, h_q // h_v
    atomic = h_k != h_v or g_split > 1
    group = math.gcd(gk, gv) if atomic else gk
    units = h_q // group
    b = IntEvalBuilder()
    out = []
    for unit in range(units):
        for split in range(g_split):
            hr = head_range(
                b, unit=i32(b, unit), split=i32(b, split), group=i32(b, group),
                gk=i32(b, gk), gv=i32(b, gv),
                g_split=i32(b, g_split) if atomic else None,
            )  # fmt: skip
            heads = list(range(hr.h0.v, hr.h0.v + hr.n_heads.v))
            out.append((heads, hr.hk.v, hr.hv.v))
            if not atomic:
                break
    return out, gk, gv


@pytest.mark.parametrize(
    "h_q,h_k,h_v,g_split",
    [
        (8, 8, 8, 1),  # MHA, direct
        (8, 2, 2, 1),  # GQA, direct
        (8, 1, 1, 1),  # MQA, direct
        (8, 4, 2, 1),  # h_k != h_v
        (8, 2, 8, 1),  # K fewer than V
        (12, 4, 6, 1),  # gk = 3, gv = 2: G = 1
        (8, 2, 2, 2),  # GQA, split
        (8, 1, 1, 4),  # MQA, split
        (16, 4, 2, 2),  # h_k != h_v, split
    ],
)
def test_head_ranges_cover_every_query_head_once(h_q, h_k, h_v, g_split):
    out, gk, gv = _heads(h_q, h_k, h_v, g_split)
    flat = [h for heads, _hk, _hv in out for h in heads]
    assert sorted(flat) == list(range(h_q))
    for heads, hk, hv in out:
        assert heads and all(h // gk == hk and h // gv == hv for h in heads)


def test_direct_mode_unit_is_the_kv_head():
    out, gk, _ = _heads(8, 2, 2, 1)
    assert [(hs[0], len(hs), hk) for hs, hk, _hv in out] == [(0, 4, 0), (4, 4, 1)]


def test_emitters_build_on_the_ir_builder():
    b = IRBuilder("worklist_probe")
    a = {n: b.param(n, I32) for n in ("n", "lb", "pair", "gs", "G", "gk", "gv")}
    w = decode_cta(
        b, x=b.block_id_x(), y=b.block_id_y(), z=b.block_id_z(),
        n_kv_tiles=a["n"], lb_order=a["lb"], pair=a["pair"], g_split=a["gs"],
    )  # fmt: skip
    kt = w.kv_tile(b, b.const_i32(1))
    hr = head_range(
        b, unit=w.unit, split=w.split, group=a["G"], gk=a["gk"], gv=a["gv"],
        g_split=a["gs"],
    )  # fmt: skip
    for v in (kt, w.n_pass, hr.h0, hr.n_heads, hr.hk, hr.hv):
        assert v.type == I32
    names = {op.name for op in b.kernel.body.ops}
    assert not any(n.startswith(("memref.", "tile.mma", "scf.")) for n in names)
