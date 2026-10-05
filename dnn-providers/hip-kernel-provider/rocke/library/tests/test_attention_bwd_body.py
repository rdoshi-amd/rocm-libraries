# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU tests of the shared backward body (geometry, LDS views, helpers).

The emitted kernel is covered by ``test_attention_bwd_ir.py`` (structure,
counts, resources) and the device tests; this file checks the pieces.
"""

from __future__ import annotations

import dataclasses

import pytest
from rocke.core.ir import F16, F32, I32, I64, IRBuilder, PtrType

from kernels.common._attention_bwd_addr import WorkspaceAddr
from kernels.common._attention_bwd_body import (
    BwdLds,
    BwdTileGeometry,
    DkDvAcc,
    StatsView,
    copy_rows_to_lds,
)
from kernels.common._attention_bwd_lds import PHASE_EPILOGUE
from kernels.common.attention_bwd import AttnBwdSpec, lds_plan, validate_attn_bwd_spec

from .sdpa.bwd_kernel_cases import config_knobs

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")


def _resolved(arch, cfg="default", d=64, **kw):
    knobs = dict(config_knobs(cfg, arch))
    knobs.update(kw)
    spec = AttnBwdSpec(head_size=d, mask_class="none", **knobs)
    return validate_attn_bwd_spec(spec, arch)


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("d", (32, 64, 128))
def test_geometry_gemm_shapes(arch, d):
    r = _resolved(arch, d=d)
    g = BwdTileGeometry.from_spec(r, arch)
    assert (g.g0.m, g.g0.n, g.g0.k) == (r.block_m, r.block_n, d)
    assert (g.g1.m, g.g1.n, g.g1.k) == (r.block_n, d, r.block_m)
    assert (g.g4.m, g.g4.n, g.g4.k) == (r.block_m, d, r.block_n)
    assert g.g0.grid == (1, r.waves) and g.g1.grid == (r.waves, 1)
    assert g.g4.grid == tuple(r.warp_grid_g4)
    assert g.threads == r.waves * g.wave_size
    st = g.stage_table()
    assert st.mfma.g0 == g.g0.mma_per_wave and st.mfma.g1 == g.g1.mma_per_wave
    assert st.mfma.g4 == g.g4.mma_per_wave
    # the frozen geometry is hashable and equal for equal specs
    assert hash(g) == hash(BwdTileGeometry.from_spec(r, arch))


def test_geometry_needs_a_resolved_spec():
    with pytest.raises(ValueError, match="resolved"):
        BwdTileGeometry.from_spec(AttnBwdSpec(head_size=64), "gfx942")


def test_relabel_needs_the_natural_grids():
    r = _resolved("gfx942", d=64)
    bad = dataclasses.replace(r, warp_grid_g02=(2, 2))
    with pytest.raises(ValueError):
        BwdTileGeometry.from_spec(bad, "gfx942")
    ok = dataclasses.replace(bad, pt_route="lds")
    assert BwdTileGeometry.from_spec(ok, "gfx942").relabel_plan is None


@pytest.mark.parametrize("arch", ARCHS)
def test_lds_views_follow_the_plan(arch):
    for cfg in ("debug", "default"):
        r = _resolved(arch, cfg, d=128)
        plan = lds_plan(r, arch)
        b = IRBuilder("view_probe")
        lds = BwdLds(b, plan, r.dtype)
        assert lds.smem.type.shape == (plan.total_bytes // 2,)
        for buf in plan.buffers:
            if buf.layout.startswith("vector"):
                continue
            cols = r.block_n if buf.name in ("ds", "pt") else r.head_size
            v = lds.view(buf.name, cols, buf.slot)
            assert v.base * 2 == buf.offset
            if buf.layout == "rowmajor:f32":
                assert v.scale == 2 and v.rows * 4 * cols == buf.nbytes
            else:
                assert v.scale == 1 and v.rows * 2 * v.pitch == buf.nbytes
        # the epilogue transpose buffer exists unless the accumulators are in LDS
        assert plan.has("dkv_out") is (not r.acc_in_lds)
        if plan.has("dkv_out"):
            assert plan.buffer("dkv_out").phases == (PHASE_EPILOGUE,)


def test_padded_rows_and_unknown_layouts():
    r = _resolved("gfx942", "debug", d=64, lds_swizzle="q=pad16")
    plan = lds_plan(r, "gfx942")
    lds = BwdLds(IRBuilder("pad_probe"), plan, r.dtype)
    assert lds.view("q", 64).pitch == 64 + 8
    with pytest.raises(KeyError):
        lds.view("kt_res", 16)


def test_dkdv_acc_round_trip():
    b = IRBuilder("acc_probe")
    vals = [b.zero_vec_f32(4) for _ in range(12)]
    acc = DkDvAcc.from_flat(vals, 2, 3)
    assert len(acc.dk) == 2 and len(acc.dk[0]) == 3
    assert acc.flat() == vals
    with pytest.raises(ValueError):
        DkDvAcc.from_flat(vals[:10], 2, 3)


def test_stats_view_takes_one_value_per_row():
    b = IRBuilder("stats_probe")
    p = b.param("WS", PtrType(F32, "global"))
    row0 = b.zext(b.const_i32(0), I64)
    one = WorkspaceAddr(b, p, ws_rows=b.const_i32(8), width=1, row0=row0)
    wide = WorkspaceAddr(b, p, ws_rows=b.const_i32(8), width=4, row0=row0)
    StatsView(one, one)
    with pytest.raises(ValueError):
        StatsView(one, wide)


@pytest.mark.parametrize(
    "rows,threads,guards", [(16, 64, 0), (16, 256, 1), (32, 96, 1)]
)
def test_copy_rows_guards_only_a_partial_last_pass(rows, threads, guards):
    from kernels.common._attention_bwd_addr import StridedTensorAddr, decode_batched
    from kernels.common._attention_bwd_frag import LdsView

    b = IRBuilder("copy_probe")
    p = b.param("X", PtrType(F16, "global"))
    n = b.param("n", I32)
    seq = decode_batched(b, b.const_i32(0), s_max=n)
    slab = StridedTensorAddr(b, p, dtype=F16, strides=(0, 0, 64), seq=seq).slab(0, 0)
    smem = b.smem_alloc(F16, (rows * 64,), "flat")
    copy_rows_to_lds(
        b, slab, LdsView(smem, 0, 64, rows), row0=b.const_i32(0), rows=rows, cols=64,
        tid=b.thread_id_x(), threads=threads,
    )  # fmt: skip
    ops = b.kernel.body.ops
    assert sum(op.name == "scf.if" for op in ops) == guards
    stores = sum(
        1
        for op in ops
        for r in ([op] + [x for reg in op.regions for x in reg.ops])
        if r.name == "tile.smem_store_vN"
    )
    assert stores == -(-rows * 8 // threads)
