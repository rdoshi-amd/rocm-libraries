# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests of the pruning set and of the pruning-set gate of the benchmark.

CPU: the composition of every class's pruning set, that every pruning case
plans to the very main kernel the timed cells of that class launch (so the
gate exercises the timed code object), and the gate's failure paths with a
stubbed device run. Device (gfx942 / gfx950 / gfx1151 / gfx1201): the gate on
the shipped configuration passes, and a corrupted run is rejected.
"""

from __future__ import annotations

import dataclasses

import pytest

from benchmarks.common import attention_bwd_bench as bb
from benchmarks.common import attention_bwd_rocke_arm as ra
from benchmarks.common import attention_bwd_sweep as sw
from kernels.common.attention_bwd_plan import attn_bwd_plan

from . import _attention_bwd_perf_gate as pg
from ._attention_bwd_harness import ARCH, DEVICE_ARCHS
from .sdpa.bwd_kernel_cases import (
    PRUNE_TAILS,
    config_policy,
    perf_prune_cases,
    request_from_case,
)
from .test_attention_bwd_rocke_arm import _tensors

CLASSES = [
    dict(mask_class=m, seq_mode=s, dkv_mode=k, g_split=g)
    for m in ("band", "none")
    for s in ("batched", "thd")
    for k, g in (("direct", 1), ("atomic", 1), ("atomic", 2))
]


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: "-".join(map(str, c.values())))
@pytest.mark.parametrize("d", [32, 128])
def test_pruning_set_composition(cls, d):
    cases = perf_prune_cases(head_size=d, dtype="fp16", **cls)
    assert 5 <= len(cases) <= 12
    lens = {n for c in cases for n in (c.lens()[0] + c.lens()[1])}
    assert set(PRUNE_TAILS) <= lens
    assert all(c.d == d and c.dtype == "fp16" for c in cases)
    assert any(c.h_q == 4 * c.h_k for c in cases)  # a GQA group of 4
    atomic = cls["dkv_mode"] == "atomic"
    assert any(c.h_k != c.h_v for c in cases) == atomic
    if atomic and cls["g_split"] == 1:
        assert all(c.h_k != c.h_v for c in cases)
    assert any(c.h_q == c.h_k for c in cases) == (cls["g_split"] == 1)
    if cls["mask_class"] == "band":
        assert any(c.fully_masked_rows for c in cases)
        assert any(c.left_bound >= 0 for c in cases)  # a window
        assert any(c.s_q != c.s_kv for c in cases)
    else:
        assert all(c.left_bound == c.right_bound == -1 for c in cases)
    thd = cls["seq_mode"] == "thd"
    assert all((c.length_mode == "ragged") == thd for c in cases)
    if thd:
        assert any(c.lens()[0][-1] == 0 and c.lens()[1][-1] == 0 for c in cases)
    with pytest.raises(ValueError):
        perf_prune_cases(head_size=d, g_split=2)  # a head split is atomic


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1151"])
@pytest.mark.parametrize("cfg", ["default", "debug"])
def test_every_pruning_case_runs_the_timed_main_kernel(arch, cfg):
    """Per census class: the pruning cases plan to the main kernel that the
    census cells of the class launch under the same configuration."""
    from .sdpa.bwd_kernel_cases import config_knobs

    knobs = config_knobs(cfg, arch)
    seen = 0
    for cell in bb.census_cells(bb.load_census()):
        req = ra.request_for_cell(cell, _tensors(cell), scale=ra.scale_of(cell))
        rc = ra.RockeConfig("c", knobs)
        plan = attn_bwd_plan(req, arch, policy=ra.config_policy(rc))
        cases = pg.cases_for(plan)
        for case in cases:
            p = attn_bwd_plan(
                request_from_case(case), arch, policy=config_policy(cfg, arch)
            )
            assert p.specs["main"] == plan.specs["main"], (cell.id, case.id)
            assert {s.kernel for s in plan.launches} <= {
                s.kernel for c in cases for s in _plan(c, arch, cfg).launches
            }
        seen += 1
    assert seen == 84


def _plan(case, arch, cfg):
    return attn_bwd_plan(request_from_case(case), arch, policy=config_policy(cfg, arch))


def test_sweep_runtime_configs_keep_the_kernel():
    sl = sw.primary_slice("gfx942", "general", 128)
    c = sw.geometry_stage(sl, sw.Release(frozenset({1}))).candidates[0]
    x = c.with_runtime(g_split=2)
    plan_spec = sw.legality(x).resolved
    fake = type("P", (), {"specs": {"main": plan_spec}})()
    cases = pg.cases_for(fake, x.r)
    assert cases and all((c.h_q // c.h_k) % 2 == 0 for c in cases)
    for case in cases:
        p = attn_bwd_plan(request_from_case(case), "gfx942",
                          policy=config_policy("default", "gfx942",
                                               {**x.k, **x.r}))  # fmt: skip
        assert p.specs["main"] == plan_spec


# --------------------------------------------------------------------------- failure paths
class _Run:
    def __init__(self, plan, case, bump=0.0, integrity=()):
        from ._attention_bwd_harness import reference

        _inp, ref = reference(case)
        self.plan, self.ref = plan, ref
        self.dq, self.dk, self.dv = ref.dq + bump, ref.dk, ref.dv
        self.integrity = list(integrity)


def _cfg_plan(arch="gfx942"):
    cells = sw.representative_cells(sw.primary_slice(arch, "general", 64))
    req = ra.request_for_cell(cells[0], _tensors(cells[0]), scale=0.125)
    return attn_bwd_plan(req, arch, policy=ra.config_policy(ra.SHIPPED))


def test_gate_failure_paths(monkeypatch, tmp_path):
    plan = _cfg_plan()
    cases = pg.cases_for(plan)
    gate = pg.PruneGate()

    def runner(bump=0.0, integrity=(), other_main=False):
        def run_case(case, *, arch, config, overrides, cache, **kw):
            p = _plan(case, arch, "default")
            if other_main:
                main = dataclasses.replace(p.specs["main"], edge_tiles=True)
                p = dataclasses.replace(p, specs={**p.specs, "main": main})
            return _Run(p, case, bump, integrity)

        return run_case

    monkeypatch.setattr(pg, "run_case", runner())
    assert gate.check("gfx942", ra.SHIPPED, plan, {}, tmp_path) == []
    monkeypatch.setattr(pg, "run_case", runner(bump=1.0))
    probs = gate.check("gfx942", ra.SHIPPED, plan, {}, tmp_path)
    assert len(probs) == len(cases) and all("dq" in p for p in probs)
    monkeypatch.setattr(pg, "run_case", runner(integrity=["dq: tail canary"]))
    assert gate.check("gfx942", ra.SHIPPED, plan, {}, tmp_path)
    monkeypatch.setattr(pg, "run_case", runner(other_main=True))
    probs = gate.check("gfx942", ra.SHIPPED, plan, {}, tmp_path)
    assert all(p.endswith("gate_kernel_mismatch") for p in probs)
    # a timed kernel no pruning case exercised fails the coverage check
    monkeypatch.setattr(pg, "run_case", runner())
    extra = dataclasses.replace(plan.launches[0], kernel=plan.launches[0].kernel + "_x")
    p2 = dataclasses.replace(plan, launches=plan.launches + (extra,))
    assert gate.check("gfx942", ra.SHIPPED, p2, {}, tmp_path)[0].startswith(
        "gate_coverage"
    )


def test_prepare_stores_the_oracles(tmp_path):
    plan = _cfg_plan()
    gate = pg.PruneGate()
    n = gate.prepare([(ra.SHIPPED, plan)], tmp_path)
    assert n == len(pg.cases_for(plan))
    assert gate.prepare([(ra.SHIPPED, plan)], tmp_path) == 0
    assert len(list(tmp_path.glob("prune-*.pkl"))) == n


# --------------------------------------------------------------------------- device
@pytest.mark.gpu
@pytest.mark.skipif(ARCH not in DEVICE_ARCHS, reason=f"no backward device ({ARCH})")
def test_gate_on_device_passes_the_shipped_config_and_rejects_a_bad_kernel(
    tmp_path, monkeypatch
):
    from . import _attention_bwd_harness as h

    arch = ARCH if ARCH != "gfx1201" else "gfx1151"
    sl = sw.primary_slice(arch, "general", 64)
    cell = sw.representative_cells(sl)[0]
    small = dataclasses.replace(cell, b=1, sq=64, skv=64, hq=8, hkv=2)
    req = ra.request_for_cell(small, _tensors(small), scale=0.125)
    plan = attn_bwd_plan(req, arch, policy=ra.config_policy(ra.SHIPPED))
    if ARCH == "gfx1201":
        plan = dataclasses.replace(plan, arch="gfx1201")
    gate = pg.PruneGate()
    gate.prepare([(ra.SHIPPED, plan)], tmp_path)
    cache: dict = {}
    assert gate.check(ARCH, ra.SHIPPED, plan, cache, tmp_path) == []
    # negative control: the main kernel launched with a wrong scale kernarg
    real = h.run_case

    def bad(case, **kw):
        def mutate(p):
            return h._replace_launch(p, "main", scalars={"ds_mult": 0.5})

        return real(case, mutate_plan=mutate, **kw)

    monkeypatch.setattr(pg, "run_case", bad)
    assert gate.check(ARCH, ra.SHIPPED, plan, cache, tmp_path)
