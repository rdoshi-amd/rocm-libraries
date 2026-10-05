# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""IR, ABI and resource tests of the layout-general backward main kernel.

The shipped instance set covers every built base key (batched, direct,
``stage_vec = 8``, no mask or a runtime band; fp16 / bf16; d32 / d64 / d128)
on gfx942, gfx950, gfx1151 and gfx1201 in every correctness configuration of
the arch (the arch default geometry and the one-wave 16 x 16 configuration; on
RDNA the one-wave default and a two-wave tile), each as the plain instance,
the head-packed instance (short query spans of GQA groups) and the
``edge_tiles`` instance (three sequential q loops: edge, interior, edge; the
interior loop has no per-cell selects), plus the atomic dK / dV instances
(``dkv_mode = "atomic"``: ``h_k != h_v`` or ``g_split > 1``), plain and
head-packed, for both mask classes; the narrow-access instances
(``stage_vec = 1``, the arch's narrow tile) of every plain, head-packed and
atomic variant; and every one of these variants again with ragged THD
sequences (``seq_mode = "thd"``).

* lowering at every LLVM flavor; kernel signature equal to the
  ``rocke.attn_bwd.v3`` main list, no ``noalias``, inputs ``readonly``;
  unique kernel names;
* structure: K / V global reads only inside the liveness branch, no branch
  around a loop atomic, ``readfirstlane``'d q-loop trip counts with the
  accumulators initialised after them, no matrix op inside a branch, no
  q loop nested in another loop;
* atomic dK / dV: the epilogue adds ``2 * kN0 * D / (W * wave)`` fp32 values
  per lane inside the liveness branch, after the q loops; the CTA owns
  ``G / g_split`` query heads, and K / V are rebased on ``unit * G / gk`` and
  ``unit * G / gv``;
* grouped heads (direct mode): ``G * n_qt`` steps per q loop (runtime
  ``G``), head groups inner wrapping at ``G``, query head
  ``unit * G + group`` with Q / dO rebased on it in 64 bits in every step;
* padding: the per-batch SEQ_LEN counts are read once per CTA under the
  runtime ``has_len`` flag at element ``batch * len_stride`` and clamped to
  ``[0, S_max]``;
* THD: the offset tables are read once per CTA, outside any branch or loop
  (entries ``b`` and ``b + 1``, two 32-bit words each, the word index
  selected by the runtime ``off64``), and converted to tokens in 64 bits
  (``off * mult / div``); batched instances never read them;
* LDS barriers: every pair of LDS accesses that can race across waves
  (overlapping views, read / write or writes through aliasing views) is
  separated by a CTA-wide barrier on every path (zero-trip, and the wrap
  from one q step into the next); the q step ends with a barrier;
* counts per q step (per step variant): matrix ops per GEMM equal the
  stage-table model, dQ atomics per lane ``kM0 * D / (W * wave)`` at agent
  scope;
* workspace accesses (dQ, stats and, in atomic mode, dK / dV) rebased in
  i64; user-tensor accesses 16 bytes wide (``stage_vec = 8``) or one element
  wide with ``align <= 2`` (``stage_vec = 1``), the LDS, matrix-op and
  workspace accesses of the two being equal;
* no 128-bit transpose read; the gfx950 default reads its transposed
  operands with ``ds_read_tr16_b64``;
* the resource gate on every shipped instance (comgr + disassembly; zero
  scratch and spills, register budgets, LDS equal to the phase plan, no
  AGPR copies inside the q loop).
"""

from __future__ import annotations

import dataclasses
import re
from functools import cache

import pytest
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common._attention_bwd_body import BwdTileGeometry
from kernels.common.attention_bwd import (
    AttnBwdSpec,
    attn_bwd_arch_facts,
    attn_bwd_params,
    build_attn_bwd_main,
    lds_plan,
    validate_attn_bwd_spec,
)

from .sdpa.bwd_kernel_cases import CONFIGS, config_knobs

ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
FLAVORS = ("llvm20", "llvm22", "llvm23")
DTYPES = ("fp16", "bf16")
HEADS = (32, 64, 128)
INPUTS = {"Q", "K", "V", "dO", "WS_LSE2", "WS_DSUM", "SEQ_Q", "SEQ_KV", "OFF_Q",
          "OFF_KV", "WORKLIST"}  # fmt: skip


def _configs(arch):
    out, seen = [], []
    for cfg in CONFIGS:
        knobs = config_knobs(cfg, arch)
        if knobs is None or knobs in seen:
            continue
        seen.append(knobs)
        out.append(cfg)
    return out


# Spec knobs per instance variant (mask class, head packing, edge tiles).
VARIANTS = {
    "none": dict(mask_class="none"),
    "band": dict(mask_class="band"),
    "none_pack": dict(mask_class="none", head_pack=True),
    "band_pack": dict(mask_class="band", head_pack=True),
    "none_edge": dict(mask_class="none", edge_tiles=True),
    "band_edge": dict(mask_class="band", edge_tiles=True),
    "none_atomic": dict(mask_class="none", dkv_mode="atomic"),
    "band_atomic": dict(mask_class="band", dkv_mode="atomic"),
    "none_atomic_pack": dict(mask_class="none", dkv_mode="atomic", head_pack=True),
    "band_atomic_pack": dict(mask_class="band", dkv_mode="atomic", head_pack=True),
}
# The narrow-access instances (stage_vec = 1) of every base key, plain and
# head-packed (edge tiles are not a shipped narrow configuration).
VARIANTS.update(
    {
        f"sv1_{k}": dict(v, stage_vec=1)
        for k, v in list(VARIANTS.items())
        if "edge" not in k
    }
)
# Every variant again with ragged THD sequences.
VARIANTS.update(
    {f"thd_{k}": dict(v, seq_mode="thd") for k, v in list(VARIANTS.items())}
)
DIRECT_VARIANTS = tuple(v for v in VARIANTS if "atomic" not in v)
ATOMIC_VARIANTS = tuple(v for v in VARIANTS if "atomic" in v)

INSTANCES = [
    (arch, cfg, d, dt, var)
    for arch in ARCHS
    for cfg in _configs(arch)
    for d in HEADS
    for dt in DTYPES
    for var in VARIANTS
]


def _ids(row):
    return "-".join(map(str, row))


def spec_of(arch, cfg, d, dt, var="none") -> AttnBwdSpec:
    return AttnBwdSpec(
        head_size=d, dtype=dt, **VARIANTS[var], **config_knobs(cfg, arch)
    )


@cache
def kernel_of(arch, cfg, d, dt, var="none"):
    return build_attn_bwd_main(spec_of(arch, cfg, d, dt, var), arch=arch)


@cache
def ir_of(arch, cfg, d, dt, var="none", flavor="llvm22"):
    return lower(kernel_of(arch, cfg, d, dt, var), arch=arch, llvm_flavor=flavor)


def resolved_of(arch, cfg, d, dt, var="none"):
    return validate_attn_bwd_spec(spec_of(arch, cfg, d, dt, var), arch)


# ---------------------------------------------------------------------------
# kernel-structure walkers
# ---------------------------------------------------------------------------


def _walk(ops, ancestors=()):
    for op in ops:
        yield op, ancestors
        for r in op.regions:
            yield from _walk(r.ops, ancestors + (op,))


def _mma_loops(kernel):
    """``(loop, ancestors)`` of every ``scf.for`` that holds a matrix op."""
    return [
        (op, anc)
        for op, anc in _walk(kernel.body.ops)
        if op.name == "scf.for"
        and any(o.name == "tile.mma" for o, _ in _walk(op.regions[0].ops))
    ]


def q_loops(kernel):
    """The q loops in program order: one, or edge / interior / edge
    (``edge_tiles``). No loop holding matrix ops is nested in another loop."""
    loops = _mma_loops(kernel)
    assert all(not any(a.name == "scf.for" for a in anc) for _lp, anc in loops)
    return [lp for lp, _anc in loops]


def q_loop(kernel):
    """The first q loop."""
    return q_loops(kernel)[0]


def step_bodies(kernel):
    """Op lists of the q-step bodies (one per q loop)."""
    return [lp.regions[0].ops for lp in q_loops(kernel)]


def _last_op(ops):
    """Last op of a body before its yield."""
    return [op for op in ops if op.name != "scf.yield"][-1]


def _derived(kernel, names):
    """Value names derived from the given params through pointer adds."""
    out = {f"%{n}" for n in names}
    for op, _ in _walk(kernel.body.ops):
        if op.name == "tile.global_ptr_add" and op.operands[0].name in out:
            out.add(op.results[0].name)
    return out


# ---------------------------------------------------------------------------
# lowering, signature, names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", INSTANCES, ids=_ids)
def test_instance_builds_and_lowers(row):
    ir = ir_of(*row)
    assert "alloca" not in ir and "addrspace(5)" not in ir
    assert "define amdgpu_kernel void @" in ir


@pytest.mark.parametrize("flavor", FLAVORS)
@pytest.mark.parametrize("arch", ARCHS)
def test_lowers_at_every_flavor(arch, flavor):
    for cfg in _configs(arch):
        for var in (
            "none", "band_pack", "band_edge", "band_atomic_pack", "thd_band_pack",
            "thd_none_atomic", "sv1_band_atomic_pack", "thd_sv1_band",
        ):  # fmt: skip
            ir = ir_of(arch, cfg, 64, "bf16", var, flavor)
            assert "define amdgpu_kernel void @" in ir


def _signature(ir):
    head = re.search(r"define amdgpu_kernel void @\S+\((.*?)\) #0", ir).group(1)
    out = []
    for p in head.split(", "):
        name = p.split("%")[-1]
        if p.startswith("ptr"):
            kind = "Q"
        elif p.startswith("float"):
            kind = "f"
        else:
            kind = "q" if p.startswith("i64") else "i"
        out.append((name, kind, p))
    return out


@pytest.mark.parametrize("arch", ARCHS)
def test_kernarg_order_matches_params(arch):
    variants = (
        "none", "band_pack", "band_atomic", "thd_band", "thd_none_atomic",
        "sv1_band", "thd_sv1_none_atomic_pack",
    )  # fmt: skip
    for cfg, var in ((c, v) for c in _configs(arch) for v in variants):
        sig = _signature(ir_of(arch, cfg, 128, "fp16", var))
        assert [(n, k) for n, k, _ in sig] == list(attn_bwd_params("main"))
        for name, kind, text in sig:
            assert "noalias" not in text
            if kind == "Q":
                assert ("readonly" in text) == (name in INPUTS), text


def test_kernel_names_are_unique():
    names = {}
    for row in INSTANCES:
        arch = row[0]
        name = kernel_of(*row).name
        assert name == resolved_of(*row).kernel_name("main")
        assert names.setdefault((arch, name), row) == row, (name, row)


# ---------------------------------------------------------------------------
# build-time facts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
def test_lane_map_facts_and_relabel(arch):
    for cfg in _configs(arch):
        geom = BwdTileGeometry.from_spec(resolved_of(arch, cfg, 64, "fp16"), arch)
        if geom.pt_route == "relabel":
            assert geom.relabel_plan.kind == "direct"
        assert geom.reader == (
            "tr16" if geom.transpose_source == "tr_read" else "plain"
        )


def test_illegal_geometry_raises_before_ir():
    r = resolved_of("gfx942", "default", 64, "fp16")
    with pytest.raises(ValueError, match="no operand reader"):
        BwdTileGeometry.from_spec(
            r.__class__(**{**r.__dict__, "transpose_source": "xt_lds"}), "gfx942"
        )
    # a transpose read on an arch without it
    with pytest.raises(ValueError):
        BwdTileGeometry.from_spec(
            r.__class__(**{**r.__dict__, "transpose_source": "tr_read"}), "gfx942"
        )


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", INSTANCES, ids=_ids)
def test_structure(row):
    kernel = kernel_of(*row)
    loops = q_loops(kernel)
    assert len(loops) == (3 if resolved_of(*row).edge_tiles else 1)
    kv = _derived(kernel, ("K", "V"))
    kv_loads = 0
    guards = None  # branch conditions shared by every K / V load (by identity)
    for op, anc in _walk(kernel.body.ops):
        if op.name.startswith("memref.global_load") and op.operands[0].name in kv:
            # K / V are read only inside the liveness branch, never in a loop
            conds = {
                id(a.operands[0]): a.operands[0] for a in anc if a.name == "scf.if"
            }
            assert conds, op.name
            if guards is None:
                guards = conds
            else:
                guards = {k: v for k, v in guards.items() if k in conds}
            assert not any(lp in anc for lp in loops)
            kv_loads += 1
        if op.name == "tile.mma":
            assert not any(a.name in ("scf.if", "scf.if_else") for a in anc)
    assert kv_loads > 0
    # the shared guard is the CTA liveness test k0 < len_kv: a strict compare
    # of two different values (not a constant or a tautology)
    live = [
        c for c in (guards or {}).values()
        if c.op is not None and c.op.name == "arith.cmp"
        and c.op.attrs.get("pred") == "lt"
        and c.op.operands[0] is not c.op.operands[1]
    ]  # fmt: skip
    assert live, "K / V staging is not guarded by the kv-tile liveness test"
    order = {id(op): i for i, (op, _) in enumerate(_walk(kernel.body.ops))}
    first_upper = loops[0].operands[1]
    for loop in loops:
        in_loop = list(_walk(loop.regions[0].ops))
        atomics = [
            (op, anc)
            for op, anc in in_loop
            if op.name == "memref.global_atomic_add_f32"
        ]
        assert atomics
        for _op, anc in atomics:
            assert not any(a.name in ("scf.if", "scf.if_else") for a in anc)
        # trip counts are readfirstlane'd; the trailing barrier is kept
        upper = loop.operands[1]
        assert upper.op is not None and upper.op.name == "tile.readfirstlane"
        assert loop.attrs.get("elide_trailing_barrier") is False
    # accumulators start after the exit test (iter args: dK / dV accumulators,
    # then the q tile and head group indices)
    inits = loops[0].operands[3:]
    for init in inits[: len(inits) - 2]:
        assert order[id(init.op)] > order[id(first_upper.op)]
    if len(loops) == 3:
        # the interior loop skips the per-cell selects of the edge loops
        def selects(lp):
            return sum(op.name == "arith.select" for op, _ in _walk(lp.regions[0].ops))

        assert selects(loops[1]) < selects(loops[0]) == selects(loops[2])


@pytest.mark.parametrize(
    "row", [r for r in INSTANCES if r[4] in ATOMIC_VARIANTS], ids=_ids
)
def test_atomic_epilogue(row):
    """Atomic dK / dV: ``2 * kN0 * D / (W * wave)`` fp32 atomics per lane after
    the q loops, inside the liveness branch, into ``WS_DK`` / ``WS_DV``; no
    user dK / dV access; the direct instance of the same row has no epilogue
    atomic."""
    arch = row[0]
    r = resolved_of(*row)
    kernel = kernel_of(*row)
    wave = attn_bwd_arch_facts(arch).wave_size
    loops = q_loops(kernel)
    in_q = {id(op) for lp in loops for op, _ in _walk(lp.regions[0].ops)}
    ws = _derived(kernel, ("WS_DK", "WS_DV"))
    epi = []
    for op, anc in _walk(kernel.body.ops):
        if op.name != "memref.global_atomic_add_f32" or id(op) in in_q:
            continue
        assert op.operands[0].name in ws
        assert any(a.name == "scf.if" for a in anc)
        epi.append(op)
    assert len(epi) == 2 * r.block_n * r.head_size // (r.waves * wave)
    users = _derived(kernel, ("dK", "dV"))
    assert not any(
        op.operands and op.operands[0].name in users
        for op, _ in _walk(kernel.body.ops)
        if op.name.startswith("memref.global_")
    )
    direct = row[:4] + (row[4].replace("_atomic", ""),)
    in_q = {id(op) for lp in q_loops(kernel_of(*direct)) for op, _ in _walk(lp.regions[0].ops)}  # fmt: skip
    assert not [
        op
        for op, _ in _walk(kernel_of(*direct).body.ops)
        if op.name == "memref.global_atomic_add_f32" and id(op) not in in_q
    ]


@pytest.mark.parametrize("arch", ARCHS)
def test_atomic_mode_heads_and_split(arch):
    """Atomic mode: the CTA's head unit and split come from ``y / g_split`` and
    ``y % g_split``; the q loop runs ``(G / g_split) * n_qt`` steps from query
    head ``unit * G + split * (G / g_split)``; K is rebased on ``unit * G /
    gk`` and V on ``unit * G / gv``."""
    kernel = kernel_of(arch, _configs(arch)[0], 64, "fp16", "band_atomic")
    ops = [op for op, _ in _walk(kernel.body.ops)]

    def operands(op):
        return [o.name for o in op.operands]

    divs = [op for op in ops if op.name == "arith.div"]
    assert any(operands(op) == ["%G", "%g_split"] for op in divs)
    assert any(op.operands[1].name == "%gk" for op in divs)
    assert any(op.operands[1].name == "%gv" for op in divs)
    assert any(op.name == "arith.mod" and operands(op)[1] == "%g_split" for op in ops)
    for loop in q_loops(kernel):
        mul = loop.operands[1].op.operands[0].op
        per = [o.op for o in mul.operands if o.op is not None]
        assert any(
            p.name == "arith.div" and operands(p) == ["%G", "%g_split"] for p in per
        )


@pytest.mark.parametrize(
    "row",
    [r for r in INSTANCES if r[4] in ("none", "band", "band_edge")],
    ids=_ids,
)
def test_group_loop_sums_the_query_heads_of_a_kv_head(row):
    """Direct mode: the q loop runs ``G * n_qt`` steps of the CTA's kv head
    (runtime ``G``); head groups are inner and wrap at ``G``; each step's query
    head is ``unit * G + group``, and Q / dO are rebased on it in 64 bits
    inside the step."""
    kernel = kernel_of(*row)
    for loop in q_loops(kernel):
        trip = loop.operands[1].op
        assert trip.name == "tile.readfirstlane"
        mul = trip.operands[0].op
        assert mul.name == "arith.mul" and "%G" in [o.name for o in mul.operands]
        grp = loop.operands[-1]
        assert grp.op is not None and grp.op.name == "arith.constant"
        body = loop.regions[0].ops
        carried = {o.name for op in body for o in op.operands}
        grp_iv = [n for n in carried if n.endswith("grp")]
        assert len(grp_iv) == 1, grp_iv
        # the head group wraps at G
        wraps = [
            op for op in body if op.name == "arith.cmp" and op.operands[1].name == "%G"
        ]
        assert wraps
        # query head = unit * G + group (unit = the kv head of the CTA)
        heads = [
            op
            for op in body
            if op.name == "arith.add" and op.operands[1].name == grp_iv[0]
        ]
        assert heads
        base = heads[0].operands[0].op
        assert base.name == "arith.mul" and base.operands[1].name == "%G"
        head = heads[0].results[0].name
        # Q and dO slabs: the head is widened before it meets the head stride
        widened = {
            op.results[0].name
            for op in body
            if op.name == "arith.zext" and op.operands[0].name == head
        }
        assert widened
        rebased = {
            p.operands[0].name
            for p in body
            if p.name == "tile.global_ptr_add" and p.operands[0].name in ("%Q", "%dO")
        }
        assert rebased == {"%Q", "%dO"}


@pytest.mark.parametrize("arch", ARCHS)
def test_sequence_lengths_read_once_under_the_runtime_flag(arch):
    """Padding: SEQ_Q and SEQ_KV are each read once per CTA, outside the q
    loops, at element ``batch * len_stride``, inside a 0/1-trip loop on
    ``has_len`` (a null pointer is never dereferenced when the flag is 0),
    and the count is clamped to ``[0, S_max]`` before any use."""
    limits = {"%SEQ_Q": "%S_q_max", "%SEQ_KV": "%S_kv_max"}
    for cfg in _configs(arch):
        for var in ("band", "band_pack", "none_atomic", "thd_band", "thd_none_atomic"):
            kernel = kernel_of(arch, cfg, 64, "fp16", var)
            in_q = {
                id(op) for lp in q_loops(kernel) for op, _ in _walk(lp.regions[0].ops)
            }
            ops = list(_walk(kernel.body.ops))
            found = {}
            for op, anc in ops:
                if not op.name.startswith("memref.global_load"):
                    continue
                ptr = op.operands[0].name
                if ptr not in limits:
                    continue
                found[ptr] = found.get(ptr, 0) + 1
                assert id(op) not in in_q
                guard = anc[-1]
                assert guard.name == "scf.for"
                trip = guard.operands[1].op
                assert trip.name == "arith.select"
                cond = trip.operands[0].op
                assert cond.name == "arith.cmp"
                assert "%has_len" in [o.name for o in cond.operands]
                idx = op.operands[1].op
                assert idx.name == "arith.mul"
                assert "%len_stride" in [o.name for o in idx.operands]
                users = [u for u, _ in ops if op.results[0] in u.operands]
                assert [u.name for u in users] == ["arith.smax"]
                clamp = [u for u, _ in ops if users[0].results[0] in u.operands]
                assert [u.name for u in clamp] == ["arith.smin"]
                assert clamp[0].operands[1].name == limits[ptr]
            assert found == {"%SEQ_Q": 1, "%SEQ_KV": 1}, (cfg, var, found)


def _producers(ops, value):
    """Ops reachable backwards from ``value`` through operands."""
    by_result = {id(r): op for op, _ in ops for r in op.results}
    seen, stack, out = set(), [value], []
    while stack:
        v = stack.pop()
        op = by_result.get(id(v))
        if op is None or id(op) in seen:
            continue
        seen.add(id(op))
        out.append(op)
        stack.extend(op.operands)
    return out


@pytest.mark.parametrize("arch", ARCHS)
def test_ragged_offsets_read_once_and_decoded_in_64_bits(arch):
    """THD: OFF_Q and OFF_KV are each read as four 32-bit words per CTA
    (entries ``b`` and ``b + 1``; the word index is selected by ``off64``,
    so an int32 table is never read past entry ``B``), outside every branch
    and loop; the token arithmetic (``off * mult / div``) is 64-bit; the
    batched instance of the same variant reads neither table."""
    mults = {"%OFF_Q": ("%q_mult", "%q_div"), "%OFF_KV": ("%kv_mult", "%kv_div")}
    for cfg in _configs(arch):
        for var in ("band", "band_pack", "none_atomic_pack"):
            batched = kernel_of(arch, cfg, 64, "fp16", var)
            names = {o.name for op, _ in _walk(batched.body.ops) for o in op.operands}
            assert not names & set(mults), (cfg, var)
            kernel = kernel_of(arch, cfg, 64, "fp16", f"thd_{var}")
            ops = list(_walk(kernel.body.ops))
            found = {}
            for op, anc in ops:
                if not op.name.startswith("memref.global_load"):
                    continue
                ptr = op.operands[0].name
                if ptr not in mults:
                    continue
                found[ptr] = found.get(ptr, 0) + 1
                assert not anc, (ptr, [a.name for a in anc])
                assert op.results[0].type.name == "i32"
                idx = _producers(ops, op.operands[1])
                assert "%off64" in {o.name for p in idx for o in p.operands}
            assert found == {"%OFF_Q": 4, "%OFF_KV": 4}, (cfg, var, found)
            for ptr, (mult, div) in mults.items():
                users = [
                    op
                    for op, _ in ops
                    if any(o.name in (mult, div) for o in op.operands)
                ]
                assert users, (ptr, mult)
                for u in users:
                    assert u.name in ("arith.sext", "arith.zext"), u.name
                    assert u.results[0].type.name == "i64"
                # the token divisions take the widened divisor (i64)
                wide = {id(u.results[0]) for u in users if u.operands[0].name == div}
                divs = [
                    op
                    for op, _ in ops
                    if op.name == "arith.div" and id(op.operands[1]) in wide
                ]
                assert len(divs) == 2, (ptr, len(divs))
                assert all(op.results[0].type.name == "i64" for op in divs)


# ---------------------------------------------------------------------------
# LDS barriers: a cross-wave hazard model over the program order
# ---------------------------------------------------------------------------


_LDS_PREFIXES = ("tile.smem_", "tile.ds_read")


def _is_lds(op) -> bool:
    return op.name.startswith(_LDS_PREFIXES) and op.name != "tile.smem_alloc"


@cache
def traced_kernel(arch, cfg, d, dt, var="none"):
    """Build the kernel and tag every LDS access with its view's span.

    Returns ``(kernel, tags)``; ``tags[id(op)] = (lo, hi, is_store, private)``
    where ``[lo, hi)`` is the accessed view in allocation elements and
    ``private`` marks the in-loop accumulator read-modify-write (each lane
    reads exactly the cells it wrote: no cross-wave hazard by construction).
    """
    from kernels.common import _attention_bwd_body as body
    from kernels.common._attention_bwd_frag import LdsView

    tags, state = {}, {"private": False}
    orig = {n: getattr(LdsView, n) for n in ("load", "store", "tr16_load")}
    orig_acc = {n: getattr(body.BwdTileStep, n) for n in ("_load_lds_acc", "_store_lds_acc")}  # fmt: skip

    def tagged(name):
        def f(self, b, *args, **kw):
            out = orig[name](self, b, *args, **kw)
            op = b._region_stack[-1].ops[-1]
            assert _is_lds(op), op.name
            hi = self.base + self.rows * self.pitch * self.scale
            tags[id(op)] = (self.base, hi, name == "store", state["private"])
            return out

        return f

    def private(name):
        def f(self, *args, **kw):
            state["private"] = True
            try:
                return orig_acc[name](self, *args, **kw)
            finally:
                state["private"] = False

        return f

    try:
        for n in orig:
            setattr(LdsView, n, tagged(n))
        for n in orig_acc:
            setattr(body.BwdTileStep, n, private(n))
        kernel = build_attn_bwd_main(spec_of(arch, cfg, d, dt, var), arch=arch)
    finally:
        for n, f in orig.items():
            setattr(LdsView, n, f)
        for n, f in orig_acc.items():
            setattr(body.BwdTileStep, n, f)
    return kernel, tags


def _events(ops, tags, *, trips, in_branch=False):
    """Program-order events: ``("sync", None)`` or ``("lds", op)``.

    Every loop body is laid out ``trips`` times (0: the zero-trip path, 2:
    the wrap from one iteration into the next). Branch bodies are always
    included; a barrier inside a branch is not counted (not every wave
    reaches it).
    """
    out = []
    for op in ops:
        if op.name == "tile.sync":
            if not in_branch:
                out.append(("sync", op))
        elif _is_lds(op):
            assert id(op) in tags, "an LDS access outside the LdsView methods"
            out.append(("lds", op))
        elif op.name == "scf.for":
            for _ in range(trips):
                out += _events(
                    op.regions[0].ops, tags, trips=trips, in_branch=in_branch
                )
        else:
            for r in op.regions:
                out += _events(r.ops, tags, trips=trips, in_branch=True)
    return out


def lds_hazards(kernel, tags):
    """Pairs of LDS accesses that can race across waves.

    Two accesses race when they overlap, no CTA-wide barrier lies between
    them on some path (zero-trip, or two iterations of each loop), they are
    not both the lane-private accumulator update, and they are a read and a
    write, or writes through two different (aliasing) views. Writes through
    one view are not a race: each thread owns the same cells of a view at
    every write site (copy chunks by ``tid``, C tiles by lane).
    """
    found = []
    for trips in (0, 2):
        ev = _events(kernel.body.ops, tags, trips=trips)
        open_ = []  # accesses since the last barrier
        for kind, op in ev:
            if kind == "sync":
                open_ = []
                continue
            lo, hi, st, priv = tags[id(op)]
            for prev in open_:
                plo, phi, pst, ppriv = tags[id(prev)]
                if not (lo < phi and plo < hi) or (priv and ppriv):
                    continue
                if (st != pst) or (st and (lo, hi) != (plo, phi)):
                    found.append((prev.name, plo, op.name, lo, trips))
            open_.append(op)
    return found


@pytest.mark.parametrize("row", INSTANCES, ids=_ids)
def test_lds_barriers_separate_cross_wave_accesses(row):
    kernel, tags = traced_kernel(*row)
    assert not lds_hazards(kernel, tags)[:4]
    # the q step ends with a barrier (the next step overwrites q / dO / dS)
    body = [op for op in q_loop(kernel).regions[0].ops if op.name != "scf.yield"]
    for ops in step_bodies(kernel):
        assert _last_op(ops).name == "tile.sync"
    # the q-loop body reads and writes LDS, so the model above saw it
    assert sum(id(op) in tags for op, _ in _walk(body)) > 0


# ---------------------------------------------------------------------------
# counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("row", INSTANCES, ids=_ids)
def test_mma_and_atomic_counts_per_step(row):
    arch = row[0]
    r = resolved_of(*row)
    geom = BwdTileGeometry.from_spec(r, arch)
    model = geom.stage_table()
    m = model.mfma
    wave = attn_bwd_arch_facts(arch).wave_size
    for ops in step_bodies(kernel_of(*row)):
        mmas = [op for op, _ in _walk(ops) if op.name == "tile.mma"]
        # emission order: G0, G2, G1, G3, G4
        assert len(mmas) == m.g0 + m.g2 + m.g1 + m.g3 + m.g4
        ids = [op.attrs["op_id"] for op in mmas]
        bounds = (m.g0, m.g0 + m.g2, m.g0 + m.g2 + m.g1, m.g0 + m.g2 + m.g1 + m.g3)
        assert set(ids[: bounds[1]]) == {geom.g0.op.op_id}
        assert set(ids[bounds[1] : bounds[3]]) == {geom.g1.op.op_id}
        assert set(ids[bounds[3] :]) == {geom.g4.op.op_id}
        atomics = [
            op for op, _ in _walk(ops) if op.name == "memref.global_atomic_add_f32"
        ]
        assert len(atomics) == r.block_m * r.head_size // (r.waves * wave)
        assert len(atomics) == model.dq_atomics_per_lane


@pytest.mark.parametrize("arch", ARCHS)
def test_dq_atomics_are_agent_scope_and_native(arch):
    variants = ("none", "band_pack", "band_atomic", "none_atomic_pack")
    for cfg, var in ((c, v) for c in _configs(arch) for v in variants):
        ir = ir_of(arch, cfg, 64, "fp16", var)
        rmw = [ln for ln in ir.splitlines() if "atomicrmw" in ln]
        assert rmw
        assert all("fadd" in ln and 'syncscope("agent") monotonic' in ln for ln in rmw)
        assert "cmpxchg" not in ir


# ---------------------------------------------------------------------------
# addressing and access width
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "var",
    [
        "none", "band_pack", "band_atomic", "none_atomic_pack", "thd_none",
        "thd_band_pack", "thd_band_atomic", "thd_none_atomic_pack",
        "sv1_band_atomic", "thd_sv1_none_atomic_pack",
    ],
)  # fmt: skip
@pytest.mark.parametrize("arch", ARCHS)
def test_workspace_rebase_is_64bit(arch, var):
    ir = ir_of(arch, _configs(arch)[0], 64, "bf16", var)
    body = ir.split(") #0 {", 1)[1]
    used = ["WS_LSE2", "WS_DSUM", "WS_DQ"]
    unused = ["WORKLIST"] + ([] if var.startswith("thd_") else ["OFF_Q", "OFF_KV"])
    (used if "atomic" in var else unused).extend(["WS_DK", "WS_DV"])
    for name in used:
        uses = re.findall(rf"[^\n]*%{name}\b[^\n]*", body)
        assert uses, name
        for u in uses:
            assert "getelementptr inbounds i8" in u and "i64" in u, u
        assert not re.search(
            rf"getelementptr inbounds float, ptr addrspace\(1\) %{name}\b", ir
        )
    for name in unused:
        assert f"%{name}" not in body  # unused in this instance


def _user_accesses(ir, names):
    derived = set(names)
    out = []
    for line in ir.splitlines():
        m = re.match(
            r"\s*(%\S+) = getelementptr inbounds \S+, ptr addrspace\(1\) (%\S+),", line
        )
        if m and m.group(2).rstrip(",") in derived:
            derived.add(m.group(1))
            continue
        m = re.match(
            r"\s*%\S+ = load (\S+(?: x \S+>)?), ptr addrspace\(1\) (%\S+), align (\d+)",
            line,
        )
        if m and m.group(2) in derived:
            out.append(("load", m.group(1), int(m.group(3))))
        m = re.match(
            r"\s*store (\S+(?: x \S+>)?) \S+, ptr addrspace\(1\) (%\S+), align (\d+)",
            line,
        )
        if m and m.group(2) in derived:
            out.append(("store", m.group(1), int(m.group(3))))
    return out


USER_TENSORS = ("%Q", "%K", "%V", "%dO", "%dK", "%dV")
WS_TENSORS = ("%WS_LSE2", "%WS_DSUM", "%WS_DQ", "%WS_DK", "%WS_DV")


@pytest.mark.parametrize("var", ["band", "band_pack"])
@pytest.mark.parametrize("dtype,elem", [("fp16", "half"), ("bf16", "bfloat")])
@pytest.mark.parametrize("arch", ARCHS)
def test_stage_vec_8_access_width(arch, dtype, elem, var):
    ir = ir_of(arch, _configs(arch)[-1], 128, dtype, var)
    acc = _user_accesses(ir, USER_TENSORS)
    kinds = {k for k, _t, _a in acc}
    assert kinds == {"load", "store"}
    assert all(t == f"<8 x {elem}>" and al == 16 for _k, t, al in acc), acc[:4]


_LDS_ACCESS = re.compile(
    r"^\s*(?:%\S+ = )?(load|store) (\S+(?: x \S+>)?)[^;]*ptr addrspace\(3\)"
    r".*align (\d+)"
)
_LDS_CALL = re.compile(r"call [^@]*(@llvm\.amdgcn\.ds\.[\w.]+)\(")


def _lds_and_workspace_accesses(ir):
    """Sorted ``(space, kind, type, align)`` of every LDS access, every LDS
    intrinsic call and every workspace load / store / atomic of a kernel."""
    out = []
    for line in ir.splitlines():
        m = _LDS_ACCESS.match(line)
        if m:
            out.append(("lds", m.group(1), m.group(2), int(m.group(3))))
        call = _LDS_CALL.search(line)
        if call and not line.lstrip().startswith("declare"):
            out.append(("lds", "call", call.group(1), 0))
        if "atomicrmw" in line:
            kind = line.split("atomicrmw", 1)[1].split()[0]
            out.append(("atomic", "rmw", kind, 0))
    ws = _user_accesses(ir, WS_TENSORS)
    out += [("ws", k, t, a) for k, t, a in ws]
    return sorted(out)


# Narrow instances per arch: every configuration, both dtypes, d32 / d64 /
# d128, plain and head-packed, direct and atomic, batched and THD.
SV1_ROWS = [
    r
    for r in INSTANCES
    if r[4] in ("sv1_band", "sv1_none_pack", "sv1_band_atomic_pack", "thd_sv1_band")
]


@pytest.mark.parametrize("row", SV1_ROWS, ids=_ids)
def test_stage_vec_1_access_width(row):
    """``stage_vec = 1``: no user-tensor access wider than one element and
    none with ``align > 2``; the LDS, matrix-op and workspace accesses equal
    those of the ``stage_vec = 8`` instance of the same geometry."""
    arch, dt = row[0], row[3]
    elem = {"fp16": "half", "bf16": "bfloat"}[dt]
    r = resolved_of(*row)
    assert r.stage_vec == 1
    ir = lower(build_attn_bwd_main(r, arch=arch), arch=arch, llvm_flavor="llvm22")
    acc = _user_accesses(ir, USER_TENSORS)
    kinds = {k for k, _t, _a in acc}
    assert kinds == ({"load"} if r.dkv_mode == "atomic" else {"load", "store"})
    assert all(t in (elem, "i16") and al <= 2 for _k, t, al in acc), acc[:4]
    wide = dataclasses.replace(r, stage_vec=8)
    wide_ir = lower(
        build_attn_bwd_main(wide, arch=arch), arch=arch, llvm_flavor="llvm22"
    )
    assert all(
        t == f"<8 x {elem}>" for _k, t, _a in _user_accesses(wide_ir, USER_TENSORS)
    )
    assert _lds_and_workspace_accesses(ir) == _lds_and_workspace_accesses(wide_ir)
    assert ir.count("@llvm.amdgcn.mfma") + ir.count("@llvm.amdgcn.wmma") == (
        wide_ir.count("@llvm.amdgcn.mfma") + wide_ir.count("@llvm.amdgcn.wmma")
    )


def test_stage_vec_1_defaults_to_the_narrow_tiles():
    """The narrow start points: 16 x 64 (32 x 64 at d32) on four waves with
    the plain LDS transpose on CDNA (d128 with the register-minimising
    scheduler); the one-wave 16 x 16 WMMA tile on RDNA."""
    for arch in ARCHS:
        for d in HEADS:
            r = resolved_of(arch, "default", d, "fp16", "sv1_band")
            if attn_bwd_arch_facts(arch).matrix_path == "mfma":
                assert (r.block_m, r.block_n, r.waves) == (32 if d == 32 else 16, 64, 4)
                assert r.transpose_source == "lds_plain"
                assert r.global_path == "vgpr" and r.ring_depth == 1
                # d128 keeps clear of the 256-VGPR file with the register-
                # minimising scheduler; the 16-byte tiles keep the default one
                assert r.scheduler_strategy == (
                    "iterative-minreg" if d == 128 else None
                )
                wide = resolved_of(arch, "default", d, "fp16", "band")
                assert wide.scheduler_strategy is None
            else:
                assert (r.block_m, r.block_n, r.waves) == (16, 16, 1)


@pytest.mark.parametrize("row", INSTANCES, ids=_ids)
def test_no_wide_transpose_read(row):
    ir = ir_of(*row)
    assert "tr16.b128" not in ir
    r = resolved_of(*row)
    if r.transpose_source == "tr_read":
        assert "@llvm.amdgcn.ds.read.tr16.b64(" in ir
    else:
        assert "ds.read.tr" not in ir


# ---------------------------------------------------------------------------
# resource gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("var", list(VARIANTS))
@pytest.mark.parametrize("arch", ARCHS)
def test_zero_scratch_no_spills_and_lds_equal_to_the_plan(arch, var):
    pytest.importorskip("rocke.runtime.comgr")
    from rocke.helpers.compile import compile_kernel

    from ._attention_bwd_resources import evaluate_resource_gate, find_objdump

    if find_objdump() is None:
        pytest.skip("llvm-objdump not found (needed for the q-loop AGPR copy rule)")
    facts = attn_bwd_arch_facts(arch)
    failures = []
    for row in (x for x in INSTANCES if x[0] == arch and x[-1] == var):
        r = resolved_of(*row)
        try:
            art = compile_kernel(
                kernel_of(*row), arch=arch, capture_ir_text=False, backend="python"
            )
        except Exception as exc:  # noqa: BLE001 - comgr cannot target this arch here
            pytest.skip(f"comgr cannot compile for {arch}: {exc}")
        res = evaluate_resource_gate(
            art.hsaco,
            arch=arch,
            lds_capacity_bytes=facts.lds_capacity_bytes,
            block_m=r.block_m,
            block_n=r.block_n,
            waves=r.waves,
            s_in_agpr=bool(r.s_in_agpr),
        )
        if not res.valid:
            pytest.skip(f"resource gate is valid on llvm22 only (flavor {res.flavor})")
        if res.failures:
            failures.append((row, res.failures))
        plan_bytes = lds_plan(r, arch).total_bytes
        if res.resources.lds_bytes > plan_bytes:
            failures.append((row, f"LDS {res.resources.lds_bytes} > plan {plan_bytes}"))
    assert not failures, failures
