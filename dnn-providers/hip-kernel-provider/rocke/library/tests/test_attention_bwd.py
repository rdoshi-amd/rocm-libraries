# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only tests of the layout-general backward spec (no device, no build).

Covers the defaulted spec fields, the per-arch default geometry, every static
legality rule of the knob catalog (value sets, couplings, LDS footprint, the
two-budget register model reproduced on the worked start-point tables), kernel
name salting and the ordered kernarg lists of the ``rocke.attn_bwd.v3`` ABI.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import itertools

import pytest

from kernels.common.attention_bwd import (
    AGPR_ALLOCS,
    ATTN_BWD_ABI,
    BLOCK_K4_VALUES,
    BWD_ARCHS,
    BWD_DTYPES,
    BWD_HEAD_SIZES,
    NOT_YET_EFFECTIVE_KNOBS,
    AttnBwdSpec,
    attn_bwd_arch_facts,
    attn_bwd_default_geometry,
    attn_bwd_params,
    build_attn_bwd_main,
    lds_footprint_bytes,
    lds_plan,
    lds_request,
    register_estimates,
    validate_attn_bwd_spec,
)

# ---------------------------------------------------------------------------
# fields and defaults
# ---------------------------------------------------------------------------


def test_only_head_size_is_required():
    required = [
        f.name
        for f in dataclasses.fields(AttnBwdSpec)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    assert required == ["head_size"]


def test_every_catalog_knob_is_a_field():
    names = {f.name for f in dataclasses.fields(AttnBwdSpec)}
    knobs = {
        "dtype",
        "seq_mode",
        "dkv_mode",
        "stage_vec",
        "mask_class",
        "dq_mode",
        "waves",
        "block_m",
        "block_n",
        "block_k4",
        "warp_grid_g4",
        "atom_g02",
        "atom_g13",
        "atom_g4",
        "pt_route",
        "kv_residency",
        "kt_source",
        "transpose_source",
        "ring_depth",
        "global_path",
        "lds_swizzle",
        "sched",
        "setprio",
        "waves_per_eu",
        "edge_tiles",
        "head_pack",
        "acc_in_lds",
        "ws_layout",
        "agpr_alloc",
        "s_in_agpr",
        "scheduler_strategy",
        "warp_grid_g02",
        "warp_grid_g13",
    }
    assert knobs <= names
    # runtime plan values are never compile-time fields
    assert not names & {
        "g_split",
        "lb_order",
        "pair",
        "xcd_n",
        "xcd_chunk",
        "pack_heads",
    }


def test_abi_tag():
    assert ATTN_BWD_ABI == "rocke.attn_bwd.v3"


@pytest.mark.parametrize("arch", BWD_ARCHS)
@pytest.mark.parametrize("d", BWD_HEAD_SIZES)
@pytest.mark.parametrize("stage_vec", (8, 1))
@pytest.mark.parametrize("seq_mode", ("batched", "thd"))
def test_default_geometry_is_legal(arch, d, stage_vec, seq_mode):
    for dt in BWD_DTYPES:
        for dkv in ("direct", "atomic"):
            for mask in ("band", "none"):
                spec = AttnBwdSpec(
                    head_size=d,
                    dtype=dt,
                    stage_vec=stage_vec,
                    seq_mode=seq_mode,
                    dkv_mode=dkv,
                    mask_class=mask,
                )
                r = validate_attn_bwd_spec(spec, arch)
                for f in dataclasses.fields(r):
                    if f.name in ("lds_swizzle", "scheduler_strategy", "waves_per_eu"):
                        continue
                    if f.name == "agpr_alloc":
                        # VGPR-form MFMA on CDNA (every start point fits it)
                        mfma = attn_bwd_arch_facts(arch).matrix_path == "mfma"
                        assert r.agpr_alloc == ((0, 0) if mfma else None)
                        continue
                    assert getattr(r, f.name) is not None, f.name


def test_default_geometry_start_points():
    g = attn_bwd_default_geometry("gfx942", 128)
    assert (g["block_m"], g["block_n"], g["block_k4"], g["waves"]) == (16, 128, 32, 8)
    assert g["warp_grid_g4"] == (1, 8)
    g = attn_bwd_default_geometry("gfx942", 64)
    assert (g["block_m"], g["block_n"], g["block_k4"], g["waves"]) == (32, 128, 32, 8)
    assert g["warp_grid_g4"] == (2, 4)
    assert g["transpose_source"] == "lds_plain" and g["global_path"] == "vgpr"
    assert g["sched"] == "none"
    g = attn_bwd_default_geometry("gfx950", 128)
    assert (g["block_m"], g["block_n"], g["waves"]) == (16, 128, 8)
    assert g["kt_source"] == "lds" and g["transpose_source"] == "tr_read"
    assert g["atom_g02"] == "16x16x32" and g["atom_g13"] == "16x16x16"
    g = attn_bwd_default_geometry("gfx950", 64)
    assert (g["block_m"], g["block_n"], g["waves"]) == (32, 256, 8)
    g = attn_bwd_default_geometry("gfx1151", 64)
    assert (g["block_m"], g["block_n"], g["waves"]) == (16, 16, 1)
    assert g["acc_in_lds"] is True and g["transpose_source"] == "wmma_lds"
    assert g["scheduler_strategy"] is None
    for arch in ("gfx1151", "gfx1201"):
        g = attn_bwd_default_geometry(arch, 128)
        assert g["scheduler_strategy"] == "iterative-minreg"
    for arch in ("gfx942", "gfx950"):
        g = attn_bwd_default_geometry(arch, 64, stage_vec=1)
        assert (g["block_m"], g["block_n"]) == (16, 64)
        assert g["transpose_source"] == "lds_plain"
        assert g["scheduler_strategy"] is None
        # the narrow d128 tile uses the register-minimising scheduler
        g = attn_bwd_default_geometry(arch, 128, stage_vec=1)
        assert g["scheduler_strategy"] == "iterative-minreg"
        assert attn_bwd_default_geometry(arch, 128)["scheduler_strategy"] is None


@pytest.mark.parametrize("arch", BWD_ARCHS)
def test_one_wave_d128_tiles_resolve_to_the_register_minimising_scheduler(arch):
    one_wave = dict(waves=1, block_m=16, block_n=16, block_k4=16, warp_grid_g4=(1, 1))
    if attn_bwd_arch_facts(arch).matrix_path == "mfma":
        one_wave.update(
            atom_g02="16x16x16", atom_g13="16x16x16", atom_g4="16x16x16",
            kv_residency="lds", kt_source="lds", transpose_source="lds_plain",
        )  # fmt: skip
    for d in BWD_HEAD_SIZES:
        r = AttnBwdSpec(head_size=d, **one_wave).resolved(arch)
        assert r.scheduler_strategy == ("iterative-minreg" if d == 128 else None)
        # an explicit strategy is kept
        r = AttnBwdSpec(head_size=d, scheduler_strategy="max-ilp", **one_wave)
        assert r.resolved(arch).scheduler_strategy == "max-ilp"
    # the multi-wave CDNA start points keep the default scheduler
    if arch in ("gfx942", "gfx950"):
        assert AttnBwdSpec(head_size=128).resolved(arch).scheduler_strategy is None


def test_resolved_agpr_alloc_prefers_the_vgpr_form():
    # An explicit reservation puts the S / dP / dQ MFMAs in AGPR form too (the
    # hot-loop AGPR copy rule fails): VGPR-form whenever the estimate fits it,
    # else the backend's choice.
    assert AttnBwdSpec(head_size=128).resolved("gfx942").agpr_alloc == (0, 0)
    assert AttnBwdSpec(head_size=128).resolved("gfx950").agpr_alloc == (0, 0)
    assert AttnBwdSpec(head_size=128).resolved("gfx1151").agpr_alloc is None
    big = AttnBwdSpec(
        head_size=128, block_m=16, block_n=192, waves=4, warp_grid_g4=(1, 4)
    ).resolved("gfx950")
    assert big.agpr_alloc is None
    explicit = AttnBwdSpec(head_size=128, agpr_alloc=(192, 192)).resolved("gfx950")
    assert explicit.agpr_alloc == (192, 192)


def test_arch_facts():
    f942, f950 = attn_bwd_arch_facts("gfx942"), attn_bwd_arch_facts("gfx950")
    assert f942.wide_k_atom is False and f950.wide_k_atom is True
    assert f942.lds_capacity_bytes == 65536 and f950.lds_capacity_bytes == 163840
    assert (f942.arch_vgprs, f942.agprs) == (256, 256)
    assert attn_bwd_arch_facts("gfx1151").matrix_path == "wmma"
    assert "dma" not in f942.legal_global_paths and "dma" in f950.legal_global_paths
    with pytest.raises(ValueError):
        attn_bwd_arch_facts("gfx90a")


# ---------------------------------------------------------------------------
# register model: the worked start-point tables
# ---------------------------------------------------------------------------


def _cdna(arch, d, km, kn, kk, *, kt, g4=(1, 4), s_in_agpr=False, agpr=None, **kw):
    if arch == "gfx942":
        base = dict(
            transpose_source="xt_lds",
            sched="stage_table",
            global_path="vgpr",
            ring_depth=1,
            atom_g02="16x16x16",
            atom_g13="16x16x16",
            atom_g4="16x16x16",
        )
    else:
        base = dict(
            transpose_source="tr_read",
            sched="stage_table",
            global_path="dma",
            ring_depth=2,
            atom_g02="16x16x32",
            atom_g13="16x16x16",
            atom_g4="16x16x32",
        )
    base.update(kw)
    return AttnBwdSpec(
        head_size=d,
        waves=4,
        block_m=km,
        block_n=kn,
        block_k4=kk,
        warp_grid_g4=g4,
        kv_residency="reg",
        kt_source=kt,
        s_in_agpr=s_in_agpr,
        agpr_alloc=agpr,
        **base,
    )


# (arch, d, kM0, kN0, kK4, kt_source, s_in_agpr, G4 grid) -> (arch-VGPR, AGPR)
_WORKED = [
    ("gfx942", 128, 16, 128, 32, "reg", False, (1, 4), 200, 128),
    ("gfx942", 128, 16, 128, 32, "lds", False, (1, 4), 168, 128),
    ("gfx942", 64, 32, 128, 32, "reg", False, (1, 4), 184, 64),
    ("gfx942", 32, 32, 128, 64, "reg", False, (2, 2), 152, 32),
    ("gfx950", 128, 16, 192, 32, "reg", False, (1, 4), 252, 192),
    ("gfx950", 128, 16, 192, 32, "lds", False, (1, 4), 204, 192),
    ("gfx950", 128, 32, 128, 32, "reg", False, (1, 4), 264, 128),
    ("gfx950", 128, 32, 128, 32, "lds", False, (1, 4), 232, 128),
    ("gfx950", 128, 32, 128, 32, "reg", True, (1, 4), 232, 160),
    ("gfx950", 64, 32, 256, 32, "reg", False, (1, 4), 272, 128),
    ("gfx950", 64, 32, 256, 32, "lds", False, (1, 4), 240, 128),
    ("gfx950", 64, 32, 256, 32, "reg", True, (1, 4), 208, 192),
]


@pytest.mark.parametrize("row", _WORKED, ids=lambda r: "_".join(map(str, r[:7])))
def test_register_model_reproduces_worked_tables(row):
    arch, d, km, kn, kk, kt, sag, g4, vgpr, agpr = row
    # the worked tables count the accumulators as AGPRs (an explicit reservation)
    spec = _cdna(
        arch, d, km, kn, kk, kt=kt, g4=g4, s_in_agpr=sag, agpr=(256, 256)
    ).resolved(arch)
    est = register_estimates(spec, arch)
    assert (est["arch_vgpr"], est["agpr"]) == (vgpr, agpr)
    # every worked row is inside the static prune allowance, so it is compiled
    validate_attn_bwd_spec(spec, arch)


def test_vgpr_form_fits_only_where_the_tables_say():
    # gfx942 d64 fits VGPR-form (248), d32 too (184); d128 does not (328)
    for d, km, kk, g4, total in ((64, 32, 32, (1, 4), 248), (32, 32, 64, (2, 2), 184)):
        s = _cdna("gfx942", d, km, 128, kk, kt="reg", g4=g4, agpr=(0, 0))
        r = validate_attn_bwd_spec(s, "gfx942")
        assert register_estimates(r, "gfx942") == {"arch_vgpr": total, "agpr": 0}
    with pytest.raises(ValueError, match="328"):
        validate_attn_bwd_spec(
            _cdna("gfx942", 128, 16, 128, 32, kt="reg", agpr=(0, 0)), "gfx942"
        )


def test_agpr_alloc_must_hold_the_estimate():
    s = _cdna("gfx950", 64, 32, 256, 32, kt="reg", s_in_agpr=True, agpr=(128, 128))
    with pytest.raises(ValueError, match="below the AGPR estimate"):
        validate_attn_bwd_spec(s, "gfx950")
    s = _cdna("gfx950", 64, 32, 256, 32, kt="reg", s_in_agpr=True, agpr=(192, 192))
    validate_attn_bwd_spec(s, "gfx950")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("d", [64, 128])
def test_agpr_reservation_must_leave_arch_vgprs(arch, d):
    """An AGPR reservation that fills the per-wave share of the unified file
    is refused: (256, 256) on the eight-wave default tile (two waves per SIMD,
    256 registers per wave) leaves no arch VGPR and fails codegen; the same
    reservation on four waves, and (192, 192) on eight, are legal."""
    eight = AttnBwdSpec(head_size=d, agpr_alloc=(256, 256))
    assert validate_attn_bwd_spec(AttnBwdSpec(head_size=d), arch).waves == 8
    with pytest.raises(ValueError, match="leaves 0 of the 256 registers per wave"):
        validate_attn_bwd_spec(eight, arch)
    validate_attn_bwd_spec(dataclasses.replace(eight, agpr_alloc=(192, 192)), arch)
    four = AttnBwdSpec(
        head_size=d, agpr_alloc=(256, 256), waves=4, block_m=16, block_n=64,
        block_k4=32, warp_grid_g4=(1, 4),
    )  # fmt: skip
    validate_attn_bwd_spec(four, arch)
    # waves_per_eu = 2 halves the share on four waves as well
    with pytest.raises(ValueError, match="registers per wave"):
        validate_attn_bwd_spec(dataclasses.replace(four, waves_per_eu=2), arch)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_accepted_agpr_reservations_compile(arch):
    """Every reservation the validator accepts on the default d64 tile and a
    four-wave tile compiles (the rule matches the backend)."""
    pytest.importorskip("rocke.runtime.comgr")
    from rocke.helpers.compile import compile_kernel

    tiles = (
        {},
        dict(waves=4, block_m=16, block_n=64, block_k4=32, warp_grid_g4=(1, 4)),
    )
    built = 0
    for tile in tiles:
        for alloc in AGPR_ALLOCS[2:]:
            spec = AttnBwdSpec(head_size=64, agpr_alloc=alloc, **tile)
            try:
                validate_attn_bwd_spec(spec, arch)
            except ValueError:
                continue
            try:
                compile_kernel(
                    build_attn_bwd_main(spec, arch=arch), arch=arch,
                    capture_ir_text=False, backend="python",
                )  # fmt: skip
            except Exception as exc:  # noqa: BLE001
                if built == 0:  # the first spec is a known-good reservation
                    pytest.skip(f"comgr cannot compile for {arch}: {exc}")
                raise
            built += 1
    assert built >= 5


def test_register_budget_beyond_allowance_is_pruned():
    # K, K^T, V resident at kN0 256 / d128 is far above the arch budget
    s = _cdna("gfx950", 128, 32, 256, 32, kt="reg")
    with pytest.raises(ValueError, match="arch-VGPR estimate"):
        validate_attn_bwd_spec(s, "gfx950")


def test_waves_per_eu_needs_the_halved_budget():
    with pytest.raises(ValueError, match="waves_per_eu=2"):
        validate_attn_bwd_spec(
            AttnBwdSpec(head_size=128, waves=4, warp_grid_g4=(1, 4), waves_per_eu=2),
            "gfx942",
        )
    with pytest.raises(ValueError, match="waves_per_eu=3"):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=32, waves_per_eu=3), "gfx950")


# ---------------------------------------------------------------------------
# LDS footprint
# ---------------------------------------------------------------------------


def test_lds_gfx942_d128_stage0_is_exactly_the_capacity():
    s = _cdna("gfx942", 128, 16, 128, 32, kt="reg").resolved("gfx942")
    assert lds_footprint_bytes(s, "gfx942") == 65536
    validate_attn_bwd_spec(s, "gfx942")
    # K^T kept in LDS reuses the staging slot, so it still fits
    validate_attn_bwd_spec(dataclasses.replace(s, kt_source="lds"), "gfx942")
    # K and V both kept in LDS next to the step ring do not
    with pytest.raises(ValueError, match="LDS plan"):
        validate_attn_bwd_spec(
            dataclasses.replace(s, kv_residency="lds", kt_source="lds"), "gfx942"
        )


def test_lds_gfx950_has_room_for_a_three_deep_ring():
    s = _cdna("gfx950", 128, 16, 192, 32, kt="lds", ring_depth=3).resolved("gfx950")
    assert lds_footprint_bytes(s, "gfx950") <= 163840
    validate_attn_bwd_spec(s, "gfx950")


def test_lds_counterexample_resident_kt_next_to_lds_kv_is_rejected():
    # K, V and the K^T image all resident next to a two-deep XT ring: the
    # planner places 3 x 12 KiB resident plus a 29 KiB loop phase (> 64 KiB).
    s = _cdna(
        "gfx942",
        32,
        32,
        192,
        64,
        kt="lds",
        g4=(2, 2),
        ring_depth=2,
        sched="none",
    )
    s = dataclasses.replace(s, kv_residency="lds").resolved("gfx942")
    assert (s.kv_residency, s.kt_source, s.transpose_source) == ("lds", "lds", "xt_lds")
    plan = lds_plan(s, "gfx942")
    assert plan.has("kt_res") and plan.has("k_res") and plan.has("v_res")
    assert not plan.fits
    assert lds_footprint_bytes(s, "gfx942") == plan.total_bytes > 65536
    with pytest.raises(ValueError, match="LDS plan"):
        validate_attn_bwd_spec(s, "gfx942")
    # one ring slot fewer fits
    validate_attn_bwd_spec(dataclasses.replace(s, ring_depth=1), "gfx942")


def test_lds_request_carries_every_planner_input():
    s = AttnBwdSpec(head_size=64, dtype="bf16", lds_swizzle="qt=pad8,dot=xor").resolved(
        "gfx950"
    )
    req = lds_request(s, "gfx950")
    assert (req.head_size, req.k_m0, req.k_n0, req.waves) == (
        64,
        s.block_m,
        s.block_n,
        s.waves,
    )
    assert req.wave_size == 64 and req.dtype == "bf16"
    assert req.swizzle_of("qt", "xor") == "pad8"
    assert req.acc_in_lds is False and req.dq_mode == "atomic"
    r = AttnBwdSpec(head_size=64).resolved("gfx1151")
    rq = lds_request(r, "gfx1151")
    assert rq.wave_size == 32 and rq.acc_in_lds is True
    with pytest.raises(ValueError, match="not resolved"):
        lds_request(AttnBwdSpec(head_size=64), "gfx942")


def _lds_sweep_specs(arch):
    """Every LDS-relevant knob combination of ``arch`` (other knobs fixed legal)."""
    facts = attn_bwd_arch_facts(arch)
    mfma = facts.matrix_path == "mfma"
    atom = facts.legal_atoms[0]
    for d, km, kn, w, (kv, kt), tsrc, ring, gpath, acc, pt in itertools.product(
        BWD_HEAD_SIZES,
        (16, 32, 64),
        (16, 32, 64, 128, 192, 256),
        facts.legal_waves,
        (("reg", "reg"), ("reg", "lds"), ("lds", "lds")),
        facts.legal_transpose_sources,
        facts.legal_ring_depths,
        facts.legal_global_paths,
        (False, True) if mfma else (True,),
        ("relabel", "lds") if mfma else ("lds",),
    ):
        if not mfma and kv == "reg":
            continue
        if gpath == "dma" and tsrc == "xt_lds":
            continue
        if w == 1 and (km, kn) != (16, 16):
            continue
        g4 = (1, w) if d % (16 * w) == 0 else (2, max(1, w // 2))
        yield AttnBwdSpec(
            head_size=d,
            waves=w,
            block_m=km,
            block_n=kn,
            block_k4=16,
            warp_grid_g4=g4,
            atom_g02=atom,
            atom_g13=atom,
            atom_g4=atom,
            pt_route=pt,
            kv_residency=kv,
            kt_source=kt,
            transpose_source=tsrc,
            ring_depth=ring,
            global_path=gpath,
            sched="none",
            setprio=False,
            acc_in_lds=acc,
            agpr_alloc=(0, 0) if (mfma and acc) else None,
        )


@pytest.mark.parametrize("arch", BWD_ARCHS)
def test_validator_and_phase_planner_agree(arch):
    # Over the whole LDS-relevant knob space of the arch: a spec the validator
    # accepts fits the phase plan and reports its footprint; a spec rejected on
    # LDS does not fit; a spec rejected on registers (checked after LDS) fits.
    counts = {"legal": 0, "lds": 0, "reg": 0}
    for spec in _lds_sweep_specs(arch):
        r = spec.resolved(arch)
        plan = lds_plan(r, arch)
        try:
            validate_attn_bwd_spec(spec, arch)
        except ValueError as exc:
            msg = str(exc)
            if msg.startswith("LDS plan"):
                assert not plan.fits, (spec, plan.total_bytes)
                counts["lds"] += 1
            elif "estimate" in msg:
                assert plan.fits, (spec, msg)
                counts["reg"] += 1
            continue
        assert plan.fits, (spec, plan.total_bytes)
        assert lds_footprint_bytes(r, arch) == plan.total_bytes
        counts["legal"] += 1
    assert counts["legal"] > 0 and counts["lds"] > 0, counts


# ---------------------------------------------------------------------------
# legality rules (value sets and couplings)
# ---------------------------------------------------------------------------

_BAD_FIELDS = [
    dict(head_size=256),
    dict(head_size=96),
    dict(head_size=64, dtype="fp32"),
    dict(head_size=64, stage_vec=4),
    dict(head_size=64, seq_mode="paged"),
    dict(head_size=64, dq_mode="split"),
    dict(head_size=64, block_n=48),
    dict(head_size=64, block_m=128),
    dict(head_size=64, atom_g02="32x32x8"),
    dict(head_size=64, sched="iglp2"),
    dict(head_size=64, ws_layout="row_major"),
    dict(head_size=64, agpr_alloc=(64, 64)),
    dict(head_size=64, kv_residency="lds", kt_source="reg"),
    dict(head_size=64, global_path="dma", stage_vec=1),
    dict(head_size=64, global_path="dma", transpose_source="xt_lds"),
    dict(head_size=64, setprio=True, sched="iglp0"),
    dict(head_size=64, s_in_agpr=True, agpr_alloc=(0, 0)),
    dict(head_size=64, acc_in_lds=True, agpr_alloc=(128, 128)),
    dict(head_size=64, atom_g13="16x16x32", block_m=16),
    dict(head_size=64, warp_grid_g4=(1, 2, 3)),
    dict(head_size=64, lds_swizzle="k=zigzag"),
    dict(head_size=64, lds_swizzle="k=xor,k=pad8"),
    dict(head_size=64, lds_swizzle="kxor"),
    dict(head_size=64, scheduler_strategy="not-a-strategy"),
    dict(head_size=64, name="bad name"),
]


@pytest.mark.parametrize(
    "kw",
    _BAD_FIELDS,
    ids=lambda kw: "-".join(f"{k}" for k in kw if k != "head_size") or "head",
)
def test_arch_independent_rules_raise(kw):
    with pytest.raises(ValueError):
        AttnBwdSpec(**kw)


_BAD_FOR_ARCH = [
    ("gfx942", dict(transpose_source="tr_read")),
    ("gfx942", dict(global_path="dma", transpose_source="lds_plain")),
    ("gfx942", dict(atom_g02="16x16x32")),
    ("gfx942", dict(ring_depth=3)),
    ("gfx942", dict(waves=4, block_n=32)),  # kN0 % (16 * W)
    ("gfx942", dict(waves=1, block_m=32, block_n=64, warp_grid_g4=(1, 1))),
    ("gfx942", dict(block_k4=64, block_n=32, waves=2, warp_grid_g4=(1, 2))),
    ("gfx942", dict(warp_grid_g4=(4, 1))),
    ("gfx942", dict(setprio=True)),
    ("gfx950", dict(atom_g4="16x16x32", block_k4=16)),
    ("gfx950", dict(waves_per_eu=3)),
    ("gfx1151", dict(kv_residency="reg", kt_source="lds")),
    ("gfx1151", dict(acc_in_lds=False)),
    ("gfx1151", dict(waves=8)),
    ("gfx1151", dict(pt_route="relabel")),
    ("gfx1151", dict(atom_g02="16x16x16")),
    ("gfx1151", dict(agpr_alloc=(128, 128))),
]


@pytest.mark.parametrize(
    "arch,kw", _BAD_FOR_ARCH, ids=lambda x: x if isinstance(x, str) else "-".join(x)
)
def test_arch_rules_raise(arch, kw):
    with pytest.raises(ValueError):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=64, **kw), arch)


# Small tiles whose register estimates sit far below every budget, so only the
# rule under test can reject them.
_SMALL_W2 = dict(
    head_size=32,
    waves=2,
    block_m=16,
    block_n=32,
    block_k4=16,
    warp_grid_g4=(1, 2),
    atom_g02="16x16x16",
    atom_g13="16x16x16",
    atom_g4="16x16x16",
)
_SMALL_W4 = dict(
    head_size=32,
    waves=4,
    block_m=32,
    block_n=64,
    block_k4=16,
    warp_grid_g4=(2, 2),
    atom_g02="16x16x16",
    atom_g13="16x16x16",
    atom_g4="16x16x16",
)


def test_waves_per_eu_3_only_on_gfx942_with_at_most_two_waves():
    validate_attn_bwd_spec(AttnBwdSpec(**_SMALL_W2, waves_per_eu=3), "gfx942")
    for arch, kw in (("gfx950", _SMALL_W2), ("gfx942", _SMALL_W4)):
        validate_attn_bwd_spec(AttnBwdSpec(**kw), arch)  # legal without it
        with pytest.raises(ValueError, match="waves_per_eu=3 is legal only on gfx942"):
            validate_attn_bwd_spec(AttnBwdSpec(**kw, waves_per_eu=3), arch)


def test_g4_warp_grid_must_divide_d_and_block_m():
    with pytest.raises(ValueError, match="does not divide"):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=32, warp_grid_g4=(1, 4)), "gfx942")
    with pytest.raises(ValueError, match="does not divide"):
        validate_attn_bwd_spec(
            AttnBwdSpec(head_size=128, block_m=16, waves=4, warp_grid_g4=(2, 2)),
            "gfx942",
        )


def test_g02_g13_warp_grids_resolve_and_need_the_lds_route_to_change():
    r = validate_attn_bwd_spec(AttnBwdSpec(**_SMALL_W4), "gfx942")
    assert (r.warp_grid_g02, r.warp_grid_g13) == ((1, 4), (4, 1))
    assert validate_attn_bwd_spec(
        AttnBwdSpec(head_size=64), "gfx1151"
    ).warp_grid_g02 == (1, 1)
    for gname in ("warp_grid_g02", "warp_grid_g13"):
        with pytest.raises(ValueError, match="needs pt_route='lds'"):
            validate_attn_bwd_spec(
                AttnBwdSpec(**_SMALL_W4, **{gname: (2, 2)}), "gfx942"
            )
        lds = validate_attn_bwd_spec(
            AttnBwdSpec(**_SMALL_W4, pt_route="lds", **{gname: (2, 2)}), "gfx942"
        )
        assert getattr(lds, gname) == (2, 2)
        with pytest.raises(ValueError, match="must be"):
            validate_attn_bwd_spec(
                AttnBwdSpec(**_SMALL_W4, pt_route="lds", **{gname: (4, 4)}), "gfx942"
            )
    # (2, W/2) must still divide the tile: G0/G2 rows are block_m
    with pytest.raises(ValueError, match="does not divide"):
        validate_attn_bwd_spec(
            AttnBwdSpec(head_size=64, block_m=16, pt_route="lds", warp_grid_g02=(2, 2)),
            "gfx942",
        )
    with pytest.raises(ValueError):
        AttnBwdSpec(head_size=64, warp_grid_g13=(0, 4))


def test_dma_buffers_read_by_tr_read_force_the_xor_swizzle():
    kw = dict(
        head_size=128, global_path="dma", transpose_source="tr_read", kt_source="lds"
    )
    for bad in ("k=pad8,kt=none,q=none", "q=pad16", "do=none"):
        with pytest.raises(ValueError, match="need 'xor'"):
            validate_attn_bwd_spec(AttnBwdSpec(**kw, lds_swizzle=bad), "gfx950")
    validate_attn_bwd_spec(
        AttnBwdSpec(**kw, lds_swizzle="do=xor,k=xor,q=xor,ds=pad8"), "gfx950"
    )
    validate_attn_bwd_spec(AttnBwdSpec(**kw), "gfx950")  # predictor default
    # without DMA the coupling does not apply
    validate_attn_bwd_spec(
        AttnBwdSpec(head_size=128, transpose_source="tr_read", lds_swizzle="q=pad8"),
        "gfx950",
    )


def test_debug_tile_is_legal_on_every_arch():
    for arch in BWD_ARCHS:
        kw = dict(waves=1, block_m=16, block_n=16, block_k4=16, warp_grid_g4=(1, 1))
        if attn_bwd_arch_facts(arch).matrix_path == "mfma":
            kw.update(
                atom_g02="16x16x16",
                atom_g13="16x16x16",
                atom_g4="16x16x16",
                kv_residency="lds",
                kt_source="lds",
                transpose_source="lds_plain",
                waves_per_eu=None,
            )
        for d in BWD_HEAD_SIZES:
            validate_attn_bwd_spec(AttnBwdSpec(head_size=d, **kw), arch)


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        validate_attn_bwd_spec(AttnBwdSpec(head_size=64), "gfx90a")


def test_swizzle_is_canonical_and_hashable():
    a = AttnBwdSpec(head_size=64, lds_swizzle=" q=pad8 , k=xor")
    b = AttnBwdSpec(head_size=64, lds_swizzle="k=xor,q=pad8")
    assert a.lds_swizzle == "k=xor,q=pad8" and a == b and hash(a) == hash(b)


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


def test_kernel_name_salted_by_every_non_default_knob():
    base = AttnBwdSpec(head_size=128, dtype="bf16")
    assert base.kernel_name() == "rocke_attn_bwd_main_bf16_d128_batched_direct_band_sv8"
    variants = [
        base,
        dataclasses.replace(base, waves=2),
        dataclasses.replace(base, block_n=64),
        dataclasses.replace(base, edge_tiles=True),
        dataclasses.replace(base, ws_layout="token_major"),
        dataclasses.replace(base, agpr_alloc=(256, 256)),
        dataclasses.replace(base, s_in_agpr=True),
        dataclasses.replace(base, lds_swizzle="k=xor"),
        dataclasses.replace(base, scheduler_strategy=None, sched="iglp0"),
        dataclasses.replace(base, seq_mode="thd"),
        dataclasses.replace(base, stage_vec=1),
        dataclasses.replace(base, mask_class="none"),
        dataclasses.replace(base, dkv_mode="atomic"),
        dataclasses.replace(base, warp_grid_g02=(2, 2)),
        dataclasses.replace(base, warp_grid_g13=(2, 2)),
    ]
    names = [v.kernel_name() for v in variants]
    assert len(set(names)) == len(names)
    assert base.kernel_name() == AttnBwdSpec(head_size=128, dtype="bf16").kernel_name()


# ---------------------------------------------------------------------------
# kernarg ABI (independent copy of the ordered lists)
# ---------------------------------------------------------------------------

_V3 = {
    "main": (
        "Q K V dO dK dV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV SEQ_Q SEQ_KV OFF_Q OFF_KV WORKLIST",
        "q_b q_h k_b k_h v_b v_h do_b do_h dk_b dk_h dv_b dv_h",
        "scale_log2 ds_mult dk_mult dq_mult",
        "q_t k_t v_t do_t dk_t dv_t "
        "h_q h_k h_v G gk gv S_q_max S_kv_max has_len len_stride off64 q_mult q_div kv_mult kv_div "
        "left right bottom_right ws_rows_q ws_rows_kv ws_seg "
        "g_split n_kv_tiles lb_order pair xcd_n xcd_chunk pack_heads use_worklist",
    ),
    "prep": (
        "O dO LSE SEQ_Q SEQ_KV OFF_Q OFF_KV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV WORKLIST",
        "o_b o_h do_b do_h l_b l_h",
        "",
        "o_t do_t l_t h_q h_k h_v S_q_max S_kv_max "
        "has_len len_stride off64 q_mult q_div kv_mult kv_div ws_rows_q ws_rows_kv ws_seg "
        "zero_kv zero_dq use_worklist n_batch wl_kn0",
    ),
    "convert": (
        "SRC DST SEQ OFF",
        "d_b d_h",
        "mult",
        "H d_t S_max has_len len_stride off64 tok_mult tok_div ws_rows ws_seg",
    ),
    "dq": (
        "Q K V dO dQ WS_LSE2 WS_DSUM SEQ_Q SEQ_KV OFF_Q OFF_KV",
        "q_b q_h k_b k_h v_b v_h do_b do_h dq_b dq_h",
        "scale_log2 ds_mult dq_mult",
        "q_t k_t v_t do_t dq_t "
        "h_q h_k h_v G S_q_max S_kv_max has_len len_stride off64 q_mult q_div kv_mult kv_div "
        "left right bottom_right ws_rows_q ws_seg xcd_n xcd_chunk",
    ),
}

# The superseded v2 lists (all-i32 scalars): every v2 name keeps its meaning in
# v3; v3 moves the batch / head strides to i64 and adds the THD row flag.
_V2_NAMES = {
    "main": "q_b q_h q_t k_b k_h k_t v_b v_h v_t do_b do_h do_t dk_b dk_h dk_t dv_b dv_h "
    "dv_t h_q h_k h_v G gk gv S_q_max S_kv_max has_len len_stride off64 q_mult q_div "
    "kv_mult kv_div left right bottom_right ws_rows_q ws_rows_kv g_split n_kv_tiles "
    "lb_order pair xcd_n xcd_chunk pack_heads use_worklist",
    "prep": "o_b o_h o_t do_b do_h do_t l_b l_h l_t h_q h_k h_v S_q_max S_kv_max has_len "
    "len_stride off64 q_mult q_div kv_mult kv_div ws_rows_q ws_rows_kv zero_kv zero_dq "
    "use_worklist n_batch wl_kn0",
    "convert": "H d_b d_h d_t S_max has_len len_stride off64 tok_mult tok_div ws_rows",
}


def _split(params):
    return tuple(tuple(n for n, f in params if f == k) for k in ("Q", "q", "f", "i"))


@pytest.mark.parametrize("stage", sorted(_V3))
def test_params_match_the_v3_lists(stage):
    got = _split(attn_bwd_params(stage, AttnBwdSpec(head_size=64)))
    want = tuple(tuple(s.split()) for s in _V3[stage])
    assert got == want
    params = attn_bwd_params(stage)
    # pointers first, then i64, f32, i32 (the i64 block is 8-byte aligned)
    kinds = [f for _, f in params]
    assert kinds == sorted(kinds, key="Qqfi".index)
    assert len({n for n, _ in params}) == len(params)


@pytest.mark.parametrize("stage", sorted(_V2_NAMES))
def test_v2_scalars_survive_in_v3(stage):
    """Every v2 scalar is still a v3 kernarg; only the batch / head strides
    changed type (i64) and ``ws_seg`` is new."""
    v3 = dict(attn_bwd_params(stage))
    v2 = _V2_NAMES[stage].split()
    assert set(v2) <= set(v3)
    widened = {n for n in v2 if v3[n] == "q"}
    assert widened and all(n.endswith(("_b", "_h")) for n in widened)
    ptrs_f32 = set(_V3[stage][0].split()) | set(_V3[stage][2].split())
    assert set(v3) - set(v2) - ptrs_f32 == {"ws_seg"}


def _ir_without_name(spec, arch):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

    name = validate_attn_bwd_spec(spec, arch).kernel_name("main")
    ir = lower(build_attn_bwd_main(spec, arch=arch), arch=arch, llvm_flavor="llvm22")
    return ir.replace(name, "KNAME")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151"])
def test_not_yet_effective_knobs_emit_the_default_kernel(arch):
    """Pins the status of the not-yet-effective knobs: every legal value
    of ``block_k4`` (and ``s_in_agpr = True`` on MFMA) emits the same IR as the
    default apart from the kernel name. When one of them becomes effective
    this test fails: take it out of ``NOT_YET_EFFECTIVE_KNOBS`` (and the
    sweep then counts its values)."""
    assert set(NOT_YET_EFFECTIVE_KNOBS) == {"block_k4", "s_in_agpr"}
    mfma = arch != "gfx1151"
    if mfma:
        base = AttnBwdSpec(head_size=64, agpr_alloc=(128, 128))
    else:  # the two-wave RDNA tile (the one-wave tile has one legal block_k4)
        base = AttnBwdSpec(
            head_size=64, waves=2, block_m=16, block_n=32, block_k4=32,
            warp_grid_g4=(1, 2),
        )  # fmt: skip
    ref = _ir_without_name(base, arch)
    k4 = validate_attn_bwd_spec(base, arch).block_k4
    tried = 0
    for alt in BLOCK_K4_VALUES:
        if alt == k4:
            continue
        spec = dataclasses.replace(base, block_k4=alt)
        try:
            validate_attn_bwd_spec(spec, arch)
        except ValueError:
            continue
        assert spec.kernel_name() != base.kernel_name()
        assert _ir_without_name(spec, arch) == ref, f"block_k4={alt} is effective"
        tried += 1
    assert tried
    if mfma:
        spec = dataclasses.replace(base, s_in_agpr=True)
        assert _ir_without_name(spec, arch) == ref, "s_in_agpr is effective"


def test_params_reject_unknown_stage():
    with pytest.raises(ValueError):
        attn_bwd_params("epilogue")


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def test_builder_signature_and_unbuilt_knobs():
    sig = inspect.signature(build_attn_bwd_main)
    assert list(sig.parameters) == ["spec", "arch"]
    assert sig.parameters["arch"].kind is inspect.Parameter.KEYWORD_ONLY
    # knob values whose emission is not built yet raise before any IR
    for kw in (
        dict(transpose_source="xt_lds"),
        dict(sched="stage_table"),
        dict(ring_depth=2),
    ):
        with pytest.raises(NotImplementedError, match="not built for"):
            build_attn_bwd_main(AttnBwdSpec(head_size=64, **kw), arch="gfx942")
    # masks, edge tiles, head packing, atomic dK / dV, THD sequences and the
    # token-major workspace are built
    for kw in (
        dict(mask_class="band"),
        dict(seq_mode="thd"),
        dict(seq_mode="thd", dkv_mode="atomic", head_pack=True),
        dict(seq_mode="thd", ws_layout="token_major"),
        dict(mask_class="band", edge_tiles=True),
        dict(mask_class="none", head_pack=True),
        dict(dkv_mode="atomic"),
        dict(dkv_mode="atomic", head_pack=True, edge_tiles=True),
    ):
        assert build_attn_bwd_main(AttnBwdSpec(head_size=64, **kw), arch="gfx942")
    # illegal specs are rejected first (ValueError)
    with pytest.raises(ValueError):
        build_attn_bwd_main(
            AttnBwdSpec(head_size=64, transpose_source="tr_read"), arch="gfx942"
        )


def test_aux_spec_matches_when_the_aux_module_exists():
    if importlib.util.find_spec("kernels.common.attention_bwd_aux") is None:
        pytest.skip("the shared prep / convert module is not in this tree")
    spec = AttnBwdSpec(
        head_size=64, dtype="bf16", seq_mode="thd", stage_vec=1, ws_layout="token_major"
    )
    aux = spec.aux_spec("prep")
    for f in ("head_size", "dtype", "seq_mode", "stage_vec", "ws_layout"):
        assert getattr(aux, f) == getattr(spec, f)
