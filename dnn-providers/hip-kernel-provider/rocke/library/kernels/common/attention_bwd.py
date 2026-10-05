# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Layout-general SDPA backward (dQ, dK, dV): spec, legality, kernarg ABI.

This module owns the compile-time description of the layout-general backward
family: :class:`AttnBwdSpec` (every tile, residency, path, schedule and codegen
knob as a defaulted field; only ``head_size`` is required), the per-arch default
geometry policy that resolves ``None`` knobs, the static legality rules
(value sets, couplings, the LDS footprint of the phase planner, the two-budget
register estimate), kernel names salted by every non-default compile-time
field, and the ordered kernarg lists of the ``rocke.attn_bwd.v3`` ABI.

Runtime plan values (``g_split``, scale placement multipliers, load-balance
order, XCD remap, head-pack count, work list) are never spec fields; the host
plan in :mod:`kernels.common.attention_bwd_plan` chooses them.

:func:`build_attn_bwd_main` emits the main kernel: the CTA decode, sequence
lengths (batched: ``S_max`` or the per-batch SEQ_LEN counts; THD: the int32 /
int64 ragged offsets, converted to tokens in 64 bits, with the workspace rows
of a sequence starting at its first token or, under the runtime ``ws_seg``,
at its slot ``b * S_max``), user-tensor and workspace addressing, the mask policy (no mask, or
the runtime band: top-left / bottom-right per sequence, two-sided windows),
the one-time K / V staging and the flattened q loop (q tiles of the inverse
band outer, query-head groups inner) over the shared tile step of
:mod:`kernels.common._attention_bwd_body`, then the dK / dV epilogue. With
``edge_tiles`` the loop runs as edge / interior / edge loops over contiguous
q-tile ranges in the same order; with ``head_pack`` a group of query heads
shares the M rows of a step (runtime ``pack_heads``). In direct mode a CTA
owns one kv head and sums the dK / dV contributions of its ``G = h_q / h_k``
query heads on chip (MHA, GQA, MQA; the ``G`` kernarg); every step rebases Q,
dO and the dQ workspace on its query head in 64 bits. In atomic mode
(``dkv_mode = "atomic"``: ``h_k != h_v``, or a runtime ``g_split > 1``) a
CTA owns ``G / g_split`` query heads of a unit of ``G = gcd(h_q / h_k, h_q /
h_v)`` heads, which share one K head and one V head, and adds its fp32 dK / dV
into the ``WS_DK`` / ``WS_DV`` workspace with atomics (two converts follow).
The CTA decode (:mod:`kernels.common._attention_bwd_worklist`) applies the
runtime load-balance order (natural or reverse; one kv tile per CTA, so the
``pair`` kernarg must be 0 and the host plan refuses mirror pairing).
``stage_vec`` sets the global access width of the user tensors only: 8 moves
16-byte vectors (every B/H/S stride a multiple of 8 elements, bases 16-byte
aligned), 1 moves single elements, so any element stride and any
element-aligned base is served; LDS, matrix ops and workspace accesses are
the same for both, and ``stage_vec = 1`` instances default to the narrow
tiles of :func:`attn_bwd_default_geometry`. Knob values whose emission is
not built yet raise :class:`NotImplementedError`.

Register model (CDNA wave64 and RDNA wave32, per lane, 4-byte registers, two
16-bit operands per register), used by :func:`register_estimates`:

* resident K, V (``kv_residency = "reg"``) and K^T (``kt_source = "reg"``):
  ``kN0*D / (2*wave*W)`` each;
* dK, dV fp32 accumulators: ``kN0*D / (wave*W)`` each (AGPRs on MFMA unless
  ``agpr_alloc = (0, 0)``; none in registers when ``acc_in_lds``);
* per-step operands (Q and dO as A and as B): ``2*kM0*D / (2*wave)``;
* register prefetch of the next Q / dO (``global_path = "vgpr"`` with
  ``ring_depth >= 2``, or the gfx942 one-deep prefetch under a stage table):
  ``kM0*D / (wave*W)``;
* S and dP fp32 accumulators ``2*kM0*kN0 / (wave*W)`` (AGPRs when
  ``s_in_agpr``) plus the P and dS operands ``kM0*kN0 / (wave*W)``;
* dQ partial ``kM0*D / (wave*W)``; dS slices for G4 ``2*kM0*kK4 / (2*wave)``;
* an allowance of 24 for addresses, loop state, lse2 / Dsum and masks.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, fields
from functools import cache
from types import MappingProxyType
from typing import Optional

from rocke.core.arch import ArchTarget
from rocke.core.codegen_policy import (
    SCHEDULER_STRATEGY_ATTR,
    normalize_scheduler_strategy,
)
from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType

from rocke.helpers.attention_band import AttnRuntimeBounds

from kernels.common._attention_bwd_addr import (
    PackedRows,
    StridedTensorAddr,
    WorkspaceAddr,
    decode_batched,
    decode_thd,
)
from kernels.common._attention_bwd_band import (
    NoMaskKTail,
    RuntimeBand,
    dkv_store_rule,
)
from kernels.common._attention_bwd_body import (
    BwdLds,
    BwdTileGeometry,
    BwdTileStep,
    DkDvAcc,
    PackedStep,
    StatsView,
)
from kernels.common._attention_bwd_frag import ir_type
from kernels.common._attention_bwd_gemm import lane_and_wave
from kernels.common._attention_bwd_lds import BwdLdsRequest, LdsPlan, plan_bwd_lds
from kernels.common._attention_bwd_sinks import (
    AtomicWorkspaceSink,
    PackedAtomicWorkspaceSink,
)
from kernels.common._attention_bwd_worklist import decode_cta, head_range

__all__ = [
    "ATTN_BWD_ABI",
    "BWD_ARCHS",
    "BWD_DTYPES",
    "BWD_HEAD_SIZES",
    "AttnBwdSpec",
    "BwdArchFacts",
    "attn_bwd_arch_facts",
    "attn_bwd_default_geometry",
    "attn_bwd_params",
    "NOT_YET_EFFECTIVE_KNOBS",
    "build_attn_bwd_main",
    "lds_footprint_bytes",
    "lds_plan",
    "lds_request",
    "register_estimates",
    "validate_attn_bwd_spec",
]

ATTN_BWD_ABI = "rocke.attn_bwd.v3"

# Targets the family knows (all four have a hardware validation run).
BWD_ARCHS = ("gfx942", "gfx950", "gfx1151", "gfx1201")
BWD_DTYPES = ("fp16", "bf16")
BWD_HEAD_SIZES = (32, 64, 128)

SEQ_MODES = ("batched", "thd")
DKV_MODES = ("direct", "atomic")
STAGE_VECS = (8, 1)
MASK_CLASSES = ("band", "none")
DQ_MODES = ("atomic", "split")
WS_LAYOUTS = ("head_major", "token_major")
BLOCK_M_VALUES = (16, 32, 64)
BLOCK_N_VALUES = (16, 32, 64, 128, 192, 256)
BLOCK_K4_VALUES = (16, 32, 64)
ATOMS = ("16x16x16", "16x16x32", "wmma16x16x16")
PT_ROUTES = ("relabel", "lds")
RESIDENCIES = ("reg", "lds")
TRANSPOSE_SOURCES = ("xt_lds", "tr_read", "lds_plain", "wmma_lds")
GLOBAL_PATHS = ("vgpr", "dma")
SWIZZLES = ("xor", "pad8", "pad16", "pad32", "none")
SCHEDS = ("none", "iglp0", "iglp1", "fences_only", "stage_table", "stage_table_atomics")
STAGE_TABLE_SCHEDS = ("stage_table", "stage_table_atomics")
AGPR_ALLOCS = (None, (0, 0), (128, 128), (192, 192), (256, 256))
# LDS buffers staged by DMA that the transpose read consumes (Q^T and dO^T for
# the dV / dK GEMMs, K^T for dQ); their swizzle is fixed to the slab XOR.
TR_READ_DMA_BUFFERS = ("q", "do", "k")

# The separate dQ kernel (Q-parallel recompute, no dQ workspace) is not built,
# so ``dq_mode = "split"`` is rejected until it is.
SPLIT_DQ_AVAILABLE = False

# Knobs that are accepted, validated and salted into the kernel name but whose
# emission is not built yet: the main kernel's IR equals the default's apart
# from the name. They stay in the knob catalog (levers are retained) and a
# sweep must not count their values as distinct configurations.
NOT_YET_EFFECTIVE_KNOBS = MappingProxyType(
    {
        "block_k4": "the G4 dS key slicing is not built: G4 reads the whole kN0 "
        "dS tile; the value is read only by the stage-table scheduler, which "
        "is not built",
        "s_in_agpr": "S / dP accumulator placement in AGPRs is not built: the "
        "value enters the register estimate only; the backend places the "
        "accumulators",
    }
)

# Static register prune: an estimate may exceed its budget by up to this many
# registers and still be compiled (the operand-liveness term is conservative);
# the compiled resource notes then decide.
REGISTER_PRUNE_ALLOWANCE = 32
_REG_ALLOWANCE_FIXED = 24
# SIMDs per CDNA compute unit: a CTA of W waves puts ceil(W / 4) waves on each
# SIMD, which share its unified (arch VGPR + AGPR) register file.
_CDNA_SIMDS_PER_CU = 4

_ARCH_TRANSPOSE = {
    "mfma_tr": ("tr_read", "xt_lds", "lds_plain"),
    "mfma": ("xt_lds", "lds_plain"),
    "wmma": ("wmma_lds",),
}


# ---------------------------------------------------------------------------
# per-arch facts (from the arch catalog)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BwdArchFacts:
    """Backward-relevant facts of one gfx target, read from ``ArchTarget``."""

    arch: str
    wave_size: int
    matrix_path: str  # "mfma" | "wmma"
    lds_capacity_bytes: int
    arch_vgprs: int
    agprs: int
    has_tr_read: bool
    has_lds_dma: bool
    wide_k_atom: bool  # native 16x16x32 f16/bf16 atom

    @property
    def legal_waves(self) -> tuple[int, ...]:
        return (1, 2, 4, 8) if self.matrix_path == "mfma" else (1, 2, 4)

    @property
    def legal_atoms(self) -> tuple[str, ...]:
        if self.matrix_path == "wmma":
            return ("wmma16x16x16",)
        return ("16x16x16", "16x16x32") if self.wide_k_atom else ("16x16x16",)

    @property
    def legal_transpose_sources(self) -> tuple[str, ...]:
        if self.matrix_path == "wmma":
            return _ARCH_TRANSPOSE["wmma"]
        return _ARCH_TRANSPOSE["mfma_tr" if self.has_tr_read else "mfma"]

    @property
    def legal_global_paths(self) -> tuple[str, ...]:
        # LDS DMA wide enough for a 16-byte row segment exists only where the
        # transpose read exists too (gfx950); gfx942's one-dword DMA is not used.
        if self.matrix_path == "mfma" and self.has_lds_dma and self.has_tr_read:
            return ("vgpr", "dma")
        return ("vgpr",)

    @property
    def legal_ring_depths(self) -> tuple[int, ...]:
        if self.matrix_path == "wmma":
            return (1,)
        return (1, 2, 3) if self.has_tr_read else (1, 2)


@cache
def attn_bwd_arch_facts(arch: str) -> BwdArchFacts:
    """Facts for ``arch`` (a gfx id); raises ``ValueError`` for other targets."""
    if arch not in BWD_ARCHS:
        raise ValueError(
            f"attention backward does not target {arch!r}; known: {BWD_ARCHS}"
        )
    t = ArchTarget.from_gfx(arch)
    path = "mfma" if t.has_mfma else "wmma"
    agprs = int(t.limits.agprs)
    wide = False
    if path == "mfma":
        for dt in BWD_DTYPES:
            ops = t.mma.enumerate(
                family="mma", a_dtype=dt, b_dtype=dt, c_dtype="fp32", m=16, n=16
            )
            if any(op.k == 32 and op.wave_size == t.wave_size for op in ops):
                wide = True
    return BwdArchFacts(
        arch=arch,
        wave_size=int(t.wave_size),
        matrix_path=path,
        lds_capacity_bytes=int(t.lds_capacity_bytes),
        arch_vgprs=int(t.limits.vgprs) - agprs,
        agprs=agprs,
        has_tr_read=bool(t.memory.has_ds_read_tr),
        has_lds_dma=bool(t.memory.has_async_lds),
        wide_k_atom=wide,
    )


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------


def _atom_k(atom: str) -> int:
    return int(atom.rsplit("x", 1)[1])


def _canonical_swizzle(value: str) -> str:
    """Normalise ``"k=xor,q=pad8"`` to sorted ``name=value`` pairs."""
    pairs = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, val = item.partition("=")
        name, val = name.strip(), val.strip()
        if not sep or not name.isidentifier():
            raise ValueError(f"lds_swizzle entry {item!r} is not 'buffer=value'")
        if val not in SWIZZLES:
            raise ValueError(
                f"lds_swizzle value {val!r} for {name!r} not in {SWIZZLES}"
            )
        if name in pairs:
            raise ValueError(f"lds_swizzle names buffer {name!r} twice")
        pairs[name] = val
    return ",".join(f"{k}={pairs[k]}" for k in sorted(pairs))


def _check_in(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name}={value!r} not in {tuple(allowed)}")


@dataclass(frozen=True)
class AttnBwdSpec:
    """Compile-time spec of the layout-general backward main kernel.

    ``head_size`` is the only required field. A ``None`` knob is resolved by
    :func:`attn_bwd_default_geometry` for the target arch (see :meth:`resolved`).
    On MFMA targets ``agpr_alloc = None`` resolves to ``(0, 0)`` (VGPR-form
    MFMA) when the register estimate fits the arch-VGPR file, else stays
    ``None`` (backend choice); see :func:`_policy_agpr_alloc`. A one-wave
    ``head_size = 128`` tile with ``scheduler_strategy = None`` resolves to
    the register-minimising machine scheduler on every arch (see
    :data:`_D128_LOW_WAVE_SCHED`), and so does the CDNA narrow d128 tile
    (``stage_vec = 1``; see :data:`_NARROW_D128_SCHED`).
    """

    head_size: int
    dtype: str = "fp16"
    seq_mode: str = "batched"  # "batched" | "thd"
    dkv_mode: str = "direct"  # "direct" | "atomic"
    stage_vec: int = 8  # global access width of user tensors: 8 | 1
    mask_class: str = "band"  # "band" | "none"
    dq_mode: str = "atomic"  # "atomic" | "split"
    waves: Optional[int] = None
    block_m: Optional[int] = None  # kM0
    block_n: Optional[int] = None  # kN0
    block_k4: Optional[int] = None  # kK4
    warp_grid_g4: Optional[tuple[int, int]] = None
    atom_g02: Optional[str] = None
    atom_g13: Optional[str] = None
    atom_g4: Optional[str] = None
    pt_route: Optional[str] = None
    kv_residency: Optional[str] = None
    kt_source: Optional[str] = None
    transpose_source: Optional[str] = None
    ring_depth: Optional[int] = None
    global_path: Optional[str] = None
    lds_swizzle: Optional[str] = None  # canonical "buf=value,..."; None = predictor
    sched: Optional[str] = None
    setprio: Optional[bool] = None
    waves_per_eu: Optional[int] = None
    edge_tiles: Optional[bool] = None
    head_pack: Optional[bool] = None
    acc_in_lds: Optional[bool] = None
    ws_layout: str = "head_major"
    agpr_alloc: Optional[tuple[int, int]] = None
    s_in_agpr: Optional[bool] = None
    scheduler_strategy: Optional[str] = None  # codegen scheduler strategy
    # Warp grids (rows, cols) of G0/G2 (S, dP) and G1/G3 (dV, dK). None resolves
    # to (1, W) / (W, 1); (2, W/2) is legal only with pt_route = "lds".
    warp_grid_g02: Optional[tuple[int, int]] = None
    warp_grid_g13: Optional[tuple[int, int]] = None
    name: str = "rocke_attn_bwd"

    def __post_init__(self) -> None:
        """Arch-independent value sets and couplings (rules of the knob catalog)."""
        _check_in("head_size", self.head_size, BWD_HEAD_SIZES)
        _check_in("dtype", self.dtype, BWD_DTYPES)
        _check_in("seq_mode", self.seq_mode, SEQ_MODES)
        _check_in("dkv_mode", self.dkv_mode, DKV_MODES)
        _check_in("stage_vec", self.stage_vec, STAGE_VECS)
        _check_in("mask_class", self.mask_class, MASK_CLASSES)
        _check_in("dq_mode", self.dq_mode, DQ_MODES)
        _check_in("ws_layout", self.ws_layout, WS_LAYOUTS)
        if self.dq_mode == "split" and not SPLIT_DQ_AVAILABLE:
            raise ValueError(
                "dq_mode='split' needs the split dQ kernel, which is not built"
            )
        opt = {
            "waves": (1, 2, 4, 8),
            "block_m": BLOCK_M_VALUES,
            "block_n": BLOCK_N_VALUES,
            "block_k4": BLOCK_K4_VALUES,
            "atom_g02": ATOMS,
            "atom_g13": ATOMS,
            "atom_g4": ATOMS,
            "pt_route": PT_ROUTES,
            "kv_residency": RESIDENCIES,
            "kt_source": RESIDENCIES,
            "transpose_source": TRANSPOSE_SOURCES,
            "ring_depth": (1, 2, 3),
            "global_path": GLOBAL_PATHS,
            "sched": SCHEDS,
            "waves_per_eu": (1, 2, 3),
        }
        for fname, allowed in opt.items():
            v = getattr(self, fname)
            if v is not None:
                _check_in(fname, v, allowed)
        for fname in ("setprio", "edge_tiles", "head_pack", "acc_in_lds", "s_in_agpr"):
            v = getattr(self, fname)
            if v is not None and not isinstance(v, bool):
                raise ValueError(f"{fname} must be a bool or None, got {v!r}")
        for gname in ("warp_grid_g4", "warp_grid_g02", "warp_grid_g13"):
            gv = getattr(self, gname)
            if gv is None:
                continue
            g = tuple(gv)
            if len(g) != 2 or any(not isinstance(x, int) or x < 1 for x in g):
                raise ValueError(f"{gname}={gv!r} must be (rows, cols)")
            object.__setattr__(self, gname, g)
        if self.agpr_alloc is not None:
            a = tuple(self.agpr_alloc)
            _check_in("agpr_alloc", a, AGPR_ALLOCS)
            object.__setattr__(self, "agpr_alloc", a)
        if self.lds_swizzle is not None:
            object.__setattr__(
                self, "lds_swizzle", _canonical_swizzle(self.lds_swizzle)
            )
        if self.scheduler_strategy is not None:
            normalize_scheduler_strategy(self.scheduler_strategy)
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"name={self.name!r} must be a non-empty identifier")
        # couplings that need no arch
        if self.kt_source == "reg" and self.kv_residency == "lds":
            raise ValueError("kt_source='reg' requires kv_residency='reg'")
        if self.global_path == "dma" and self.stage_vec != 8:
            raise ValueError("global_path='dma' requires stage_vec=8")
        if self.global_path == "dma" and self.transpose_source == "xt_lds":
            raise ValueError("global_path='dma' excludes transpose_source='xt_lds'")
        if (
            self.setprio
            and self.sched is not None
            and self.sched not in STAGE_TABLE_SCHEDS
        ):
            raise ValueError("setprio=True is legal only with a stage-table sched")
        if self.s_in_agpr and self.agpr_alloc == (0, 0):
            raise ValueError("s_in_agpr=True requires agpr_alloc != (0, 0)")
        if self.acc_in_lds and self.agpr_alloc not in (None, (0, 0)):
            raise ValueError("acc_in_lds=True implies agpr_alloc=(0, 0)")
        if (
            self.atom_g13 == "16x16x32"
            and self.block_m is not None
            and self.block_m < 32
        ):
            raise ValueError("atom_g13='16x16x32' requires block_m >= 32")

    # -- derived -----------------------------------------------------------

    def resolved(self, arch: str) -> "AttnBwdSpec":
        """A copy with every ``None`` knob replaced by the arch policy value."""
        defaults = attn_bwd_default_geometry(
            arch, self.head_size, self.dtype, self.stage_vec, self.seq_mode
        )
        updates = {k: v for k, v in defaults.items() if getattr(self, k) is None}
        out = dataclasses.replace(self, **updates)
        # the G0/G2 and G1/G3 grids follow the resolved wave count
        grids = {}
        if out.warp_grid_g02 is None:
            grids["warp_grid_g02"] = (1, out.waves)
        if out.warp_grid_g13 is None:
            grids["warp_grid_g13"] = (out.waves, 1)
        if grids:
            out = dataclasses.replace(out, **grids)
        if out.agpr_alloc is None and attn_bwd_arch_facts(arch).matrix_path == "mfma":
            out = dataclasses.replace(out, agpr_alloc=_policy_agpr_alloc(out, arch))
        if out.scheduler_strategy is None and out.waves == 1 and out.head_size == 128:
            out = dataclasses.replace(out, scheduler_strategy=_D128_LOW_WAVE_SCHED)
        return out

    def non_default_fields(self) -> tuple[tuple[str, object], ...]:
        """Compile-time fields that differ from their declared default."""
        out = []
        for f in fields(self):
            if f.name in ("head_size", "name") or f.default is dataclasses.MISSING:
                continue
            v = getattr(self, f.name)
            if v != f.default:
                out.append((f.name, v))
        return tuple(out)

    def kernel_name(self, stage: str = "main") -> str:
        """Kernel name: base key plus a salt over every non-default knob."""
        _check_in("stage", stage, ("main", "dq"))
        base = (
            f"{self.name}_{stage}_{self.dtype}_d{self.head_size}_"
            f"{self.seq_mode}_{self.dkv_mode}_{self.mask_class}_sv{self.stage_vec}"
        )
        salted = [
            (k, v)
            for k, v in self.non_default_fields()
            if k not in ("dtype", "seq_mode", "dkv_mode", "stage_vec", "mask_class")
        ]
        if not salted:
            return base
        digest = hashlib.sha1(
            repr(sorted(salted, key=lambda kv: kv[0])).encode()
        ).hexdigest()
        return f"{base}_{digest[:10]}"

    def aux_spec(self, stage: str = "prep"):
        """The shared prep / convert spec matching this main spec.

        Imported lazily: the aux module is a separate file of the family.
        """
        _check_in("stage", stage, ("prep", "convert"))
        try:
            from kernels.common.attention_bwd_aux import AttnBwdAuxSpec
        except ImportError as exc:  # pragma: no cover - depends on the tree
            raise NotImplementedError(
                "AttnBwdAuxSpec is not available: kernels/common/attention_bwd_aux.py "
                "is missing from this tree"
            ) from exc
        return AttnBwdAuxSpec(
            head_size=self.head_size,
            dtype=self.dtype,
            seq_mode=self.seq_mode,
            stage_vec=self.stage_vec,
            ws_layout=self.ws_layout,
            name=self.name,
        )


# ---------------------------------------------------------------------------
# default geometry policy (start points; the tuned table overrides it)
# ---------------------------------------------------------------------------

# (kM0, kN0, kK4, W, warp_grid_g4, kv_residency, kt_source) per (arch class, d).
# The d64 / d128 start points run the 3.2 / 3.3 tiles on eight waves (gfx950
# d128: 16 x 128, as 192 keys do not split over eight waves). On four waves
# the per-wave state (resident K / V / K^T and the dK / dV accumulators) fits
# only with the accumulators in AGPRs, and the production backend then puts
# the S / dP / dQ accumulators in AGPRs as well, which the VALU reads inside
# the q loop (the hot-loop AGPR copy rule of the resource gate fails). Eight
# waves halve the per-wave state, so the VGPR-form MFMA holds all of it.
_CDNA_TILES = {
    ("gfx942", 32): (32, 128, 64, 4, (2, 2), "reg", "reg"),
    ("gfx942", 64): (32, 128, 32, 8, (2, 4), "reg", "reg"),
    ("gfx942", 128): (16, 128, 32, 8, (1, 8), "reg", "reg"),
    ("gfx950", 32): (32, 128, 64, 4, (2, 2), "reg", "reg"),
    ("gfx950", 64): (32, 256, 32, 8, (2, 4), "reg", "lds"),
    ("gfx950", 128): (16, 128, 32, 8, (1, 8), "reg", "lds"),
}
# stage_vec = 1 tiles: 16 x 64, except d32 where a four-wave G4 grid needs two
# warp rows (D / (16 * 4) < 1), hence kM0 = 32.
_NARROW_TILE = (16, 64, 32, 4, (1, 4), "reg", "reg")
_NARROW_TILE_D32 = (32, 64, 32, 4, (2, 2), "reg", "reg")
_DEBUG_TILE = (16, 16, 16, 1, (1, 1), "lds", "lds")
# Machine-scheduler strategy of the RDNA d128 start points (LDS accumulators,
# one or two waves) and of every one-wave d128 tile (the CDNA 16 x 16
# correctness tile): one wave holds all of D = 128 of a 16-row tile, the default
# scheduler fills the 256-VGPR file and the production backend then spills a
# few registers on some instances (atomic dK / dV with head packing).
_RDNA_D128_SCHED = "iterative-minreg"
_D128_LOW_WAVE_SCHED = _RDNA_D128_SCHED
# The CDNA narrow d128 tile (stage_vec = 1, 16 x 64, K / V / K^T in registers)
# sits at the 256-VGPR file under the default scheduler, and the production
# backend spills one register on gfx950 (atomic dK / dV under a band); the
# register-minimising scheduler keeps every narrow d128 instance spill-free.
_NARROW_D128_SCHED = _RDNA_D128_SCHED


def attn_bwd_default_geometry(
    arch: str,
    head_size: int,
    dtype: str = "fp16",
    stage_vec: int = 8,
    seq_mode: str = "batched",
) -> dict:
    """Per-arch start point for every ``None`` knob (no AGPR choice).

    The shipped correctness configuration uses the portable paths:
    ``global_path = "vgpr"``, ring depth 1, no stage table; on gfx942 the plain
    LDS transpose, on gfx950 the transpose read. RDNA uses the one-wave 16 x 16
    WMMA schedule with LDS accumulators (d128 with the register-minimising
    machine scheduler). ``seq_mode`` does not change the start point today; it
    is part of the signature so a tuned table may key on it.
    """
    facts = attn_bwd_arch_facts(arch)
    _check_in("head_size", head_size, BWD_HEAD_SIZES)
    _check_in("dtype", dtype, BWD_DTYPES)
    _check_in("stage_vec", stage_vec, STAGE_VECS)
    _check_in("seq_mode", seq_mode, SEQ_MODES)
    if facts.matrix_path == "wmma":
        km, kn, kk, w, g4, kv, kt = _DEBUG_TILE
        return {
            "waves": w,
            "block_m": km,
            "block_n": kn,
            "block_k4": kk,
            "warp_grid_g4": g4,
            "atom_g02": "wmma16x16x16",
            "atom_g13": "wmma16x16x16",
            "atom_g4": "wmma16x16x16",
            "pt_route": "lds",
            "kv_residency": kv,
            "kt_source": kt,
            "transpose_source": "wmma_lds",
            "ring_depth": 1,
            "global_path": "vgpr",
            "sched": "none",
            "setprio": False,
            "waves_per_eu": None,
            "edge_tiles": False,
            "head_pack": False,
            "acc_in_lds": True,
            "s_in_agpr": False,
            # d128 on one / two waves sits at the 256-VGPR file: the
            # register-minimising machine scheduler keeps it free of spills.
            "scheduler_strategy": _RDNA_D128_SCHED if head_size == 128 else None,
        }
    key = "gfx950" if facts.has_tr_read else "gfx942"
    if stage_vec == 8:
        tile = _CDNA_TILES[(key, head_size)]
    else:
        tile = _NARROW_TILE_D32 if head_size == 32 else _NARROW_TILE
    km, kn, kk, w, g4, kv, kt = tile
    if facts.has_tr_read and stage_vec == 8:
        transpose = "tr_read"
    else:
        transpose = "lds_plain"
    wide = facts.wide_k_atom and kk % 32 == 0 and head_size % 32 == 0
    return {
        "waves": w,
        "block_m": km,
        "block_n": kn,
        "block_k4": kk,
        "warp_grid_g4": g4,
        "atom_g02": "16x16x32" if wide else "16x16x16",
        "atom_g13": "16x16x16",
        "atom_g4": "16x16x32" if wide else "16x16x16",
        "pt_route": "relabel",
        "kv_residency": kv,
        "kt_source": kt,
        "transpose_source": transpose,
        "ring_depth": 1,
        "global_path": "vgpr",
        "sched": "none",
        "setprio": False,
        "waves_per_eu": 1,
        "edge_tiles": False,
        "head_pack": False,
        "acc_in_lds": False,
        "s_in_agpr": False,
        "scheduler_strategy": (
            _NARROW_D128_SCHED if stage_vec == 1 and head_size == 128 else None
        ),
    }


# ---------------------------------------------------------------------------
# static legality: LDS footprint and register estimates
# ---------------------------------------------------------------------------


def _policy_agpr_alloc(spec: AttnBwdSpec, arch: str) -> Optional[tuple[int, int]]:
    """``(0, 0)`` (VGPR-form MFMA) when the whole register estimate fits the
    arch-VGPR file (always with LDS accumulators), else ``None`` (backend).

    An explicit AGPR reservation makes the backend select every MFMA in AGPR
    form, including S / dP / dQ, whose values the VALU then reads with AGPR
    copies inside the q loop; the VGPR form has no such copies.
    """
    if spec.acc_in_lds:
        return (0, 0)
    vgpr, _ = _raw_estimates(spec, arch, agpr_form=False)
    return (0, 0) if vgpr <= attn_bwd_arch_facts(arch).arch_vgprs else None


def _require_resolved(spec: AttnBwdSpec) -> None:
    missing = [
        f.name
        for f in fields(spec)
        if getattr(spec, f.name) is None
        and f.name
        not in ("lds_swizzle", "agpr_alloc", "waves_per_eu", "scheduler_strategy")
    ]
    if missing:
        raise ValueError(f"spec is not resolved (call .resolved(arch)): {missing}")


def lds_request(spec: AttnBwdSpec, arch: str) -> BwdLdsRequest:
    """The LDS phase-planner request of a resolved spec on ``arch``."""
    _require_resolved(spec)
    return BwdLdsRequest(
        head_size=spec.head_size,
        k_m0=spec.block_m,
        k_n0=spec.block_n,
        waves=spec.waves,
        wave_size=attn_bwd_arch_facts(arch).wave_size,
        dtype=spec.dtype,
        kv_residency=spec.kv_residency,
        kt_source=spec.kt_source,
        transpose_source=spec.transpose_source,
        ring_depth=spec.ring_depth,
        global_path=spec.global_path,
        pt_route=spec.pt_route,
        acc_in_lds=bool(spec.acc_in_lds),
        dq_mode=spec.dq_mode,
        lds_swizzle=BwdLdsRequest.canonical_swizzle(spec.lds_swizzle),
    )


def lds_plan(spec: AttnBwdSpec, arch: str) -> LdsPlan:
    """The LDS phase plan of a resolved spec (not checked against capacity)."""
    return plan_bwd_lds(lds_request(spec, arch), arch=arch, strict=False)


def lds_footprint_bytes(spec: AttnBwdSpec, arch: str) -> int:
    """LDS bytes of a resolved spec: the phase planner's ``total_bytes``.

    The phase planner (:func:`plan_bwd_lds`) is the only LDS model of the
    family: resident buffers (LDS K / V, the K^T image or copy, LDS
    accumulators) plus the largest of the phase-aliased stage-0 K, stage-0 V,
    loop and epilogue phases, every buffer aligned to a 128-byte line.
    """
    return lds_plan(spec, arch).total_bytes


def _raw_estimates(spec: AttnBwdSpec, arch: str, *, agpr_form: bool) -> tuple[int, int]:
    facts = attn_bwd_arch_facts(arch)
    wave, w = facts.wave_size, spec.waves
    d, km, kn, kk = spec.head_size, spec.block_m, spec.block_n, spec.block_k4
    half = 2 * wave * w  # 16-bit elements per register, spread over the CTA
    vgpr = _REG_ALLOWANCE_FIXED
    if spec.kv_residency == "reg":
        vgpr += 2 * (kn * d // half)
    if spec.kt_source == "reg":
        vgpr += kn * d // half
    vgpr += 2 * km * d // (2 * wave)
    prefetch = spec.global_path == "vgpr" and (
        spec.ring_depth >= 2 or (arch == "gfx942" and spec.sched in STAGE_TABLE_SCHEDS)
    )
    if prefetch:
        vgpr += km * d // (wave * w)
    s_dp = 2 * km * kn // (wave * w)
    vgpr += km * kn // (wave * w)
    vgpr += km * d // (wave * w)
    vgpr += 2 * km * kk // (2 * wave)
    acc = 0 if spec.acc_in_lds else 2 * (kn * d // (wave * w))
    agpr = 0
    if facts.matrix_path == "mfma" and agpr_form:
        agpr += acc
        if spec.s_in_agpr:
            agpr += s_dp
        else:
            vgpr += s_dp
    else:
        vgpr += acc + s_dp
    return vgpr, agpr


def register_estimates(spec: AttnBwdSpec, arch: str) -> dict:
    """Two-budget register estimate of a resolved spec.

    Returns ``{"arch_vgpr": n, "agpr": n}``: the accumulators count as AGPRs on
    MFMA unless ``agpr_alloc = (0, 0)`` (VGPR-form) or ``acc_in_lds``.
    """
    _require_resolved(spec)
    facts = attn_bwd_arch_facts(arch)
    agpr_form = facts.matrix_path == "mfma" and spec.agpr_alloc != (0, 0)
    vgpr, agpr = _raw_estimates(spec, arch, agpr_form=agpr_form)
    return {"arch_vgpr": vgpr, "agpr": agpr}


def validate_attn_bwd_spec(spec: AttnBwdSpec, arch: str) -> AttnBwdSpec:
    """Resolve ``spec`` for ``arch`` and apply every static legality rule.

    Returns the resolved spec; raises ``ValueError`` naming the first broken
    rule. Rules: catalog (waves, atoms, transpose read, DMA, ring depth),
    geometry divisibility and couplings, the LDS phase plan within the
    per-arch capacity, and the two register budgets (arch-VGPR and AGPR, their sum
    under ``waves_per_eu``) with the static prune allowance.
    """
    facts = attn_bwd_arch_facts(arch)
    s = spec.resolved(arch)
    w, km, kn, kk, d = s.waves, s.block_m, s.block_n, s.block_k4, s.head_size
    mfma = facts.matrix_path == "mfma"
    # catalog
    _check_in(f"waves on {arch}", w, facts.legal_waves)
    for fname in ("atom_g02", "atom_g13", "atom_g4"):
        _check_in(f"{fname} on {arch}", getattr(s, fname), facts.legal_atoms)
    _check_in(
        f"transpose_source on {arch}", s.transpose_source, facts.legal_transpose_sources
    )
    _check_in(f"global_path on {arch}", s.global_path, facts.legal_global_paths)
    _check_in(f"ring_depth on {arch}", s.ring_depth, facts.legal_ring_depths)
    # geometry
    if kn % (16 * w) != 0:
        raise ValueError(f"block_n={kn} must be a multiple of 16*waves={16 * w}")
    if w == 1 and (km, kn) != (16, 16):
        raise ValueError("waves=1 is legal only for the 16 x 16 debug / fallback tile")
    k4a = _atom_k(s.atom_g4)
    if kk % k4a != 0:
        raise ValueError(f"block_k4={kk} must be a multiple of the G4 atom K ({k4a})")
    if kn % kk != 0:
        raise ValueError(f"block_n={kn} must be a multiple of block_k4={kk}")
    if d % _atom_k(s.atom_g02) != 0:
        raise ValueError(f"head_size={d} must be a multiple of the G0/G2 atom K")
    if s.atom_g13 == "16x16x32" and km < 32:
        raise ValueError("atom_g13='16x16x32' requires block_m >= 32")
    g4r, g4c = s.warp_grid_g4
    if (g4r, g4c) not in ((1, w), (2, w // 2)) or g4r * g4c != w:
        raise ValueError(
            f"warp_grid_g4={s.warp_grid_g4} must be (1, {w}) or (2, {w // 2})"
        )
    if d % (16 * g4c) != 0 or km % (16 * g4r) != 0:
        raise ValueError(
            f"warp_grid_g4={s.warp_grid_g4} does not divide D={d} / block_m={km}"
        )
    if km % 16 != 0:
        raise ValueError(f"block_m={km} must be a multiple of 16")
    # G0/G2 tile kM0 x kN0, G1/G3 tile kN0 x D: fixed grids unless P^T goes via LDS
    for gname, natural, rows, cols in (
        ("warp_grid_g02", (1, w), km, kn),
        ("warp_grid_g13", (w, 1), kn, d),
    ):
        g = getattr(s, gname)
        if g == natural:
            continue
        alt = (2, w // 2)
        if w < 2 or g != alt:
            raise ValueError(
                f"{gname}={g} must be {natural} (or {alt} with pt_route='lds')"
            )
        if s.pt_route != "lds":
            raise ValueError(
                f"{gname}={g} needs pt_route='lds' (the relabel route fixes it)"
            )
        if rows % 32 != 0 or cols % (16 * (w // 2)) != 0:
            raise ValueError(f"{gname}={g} does not divide its {rows} x {cols} tile")
    # couplings
    if s.kt_source == "reg" and s.kv_residency != "reg":
        raise ValueError("kt_source='reg' requires kv_residency='reg'")
    if not mfma and s.kv_residency == "reg":
        raise ValueError("kv_residency='reg' is not legal on WMMA targets")
    if not mfma and s.pt_route == "relabel":
        raise ValueError(
            "pt_route='relabel' is not legal on WMMA targets (C is not A^T)"
        )
    if s.global_path == "dma" and (s.stage_vec != 8 or s.transpose_source == "xt_lds"):
        raise ValueError("global_path='dma' needs stage_vec=8 and excludes xt_lds")
    if s.global_path == "dma" and s.transpose_source == "tr_read" and s.lds_swizzle:
        for item in s.lds_swizzle.split(","):
            buf, _, val = item.partition("=")
            if buf in TR_READ_DMA_BUFFERS and val != "xor":
                raise ValueError(
                    f"lds_swizzle {buf}={val}: DMA buffers read by tr_read need 'xor'"
                )
    if s.sched.startswith("iglp") and s.setprio:
        raise ValueError("setprio is legal only inside a stage table")
    if s.setprio and s.sched not in STAGE_TABLE_SCHEDS:
        raise ValueError("setprio is legal only inside a stage table")
    if not mfma:
        if not s.acc_in_lds:
            raise ValueError("acc_in_lds=False is not legal on WMMA targets")
        if s.agpr_alloc is not None or s.s_in_agpr:
            raise ValueError("agpr_alloc / s_in_agpr apply to MFMA targets only")
        if s.waves_per_eu not in (None, 1, 2):
            raise ValueError("waves_per_eu on WMMA targets must be None, 1 or 2")
    else:
        if s.acc_in_lds and s.agpr_alloc != (0, 0):
            raise ValueError("acc_in_lds=True implies agpr_alloc=(0, 0)")
        if s.s_in_agpr and s.agpr_alloc == (0, 0):
            raise ValueError("s_in_agpr=True requires agpr_alloc != (0, 0)")
        if s.waves_per_eu == 3 and not (arch == "gfx942" and w <= 2):
            raise ValueError("waves_per_eu=3 is legal only on gfx942 with waves <= 2")
    # LDS: the phase planner decides
    plan = lds_plan(s, arch)
    if not plan.fits:
        raise ValueError(
            f"LDS plan {plan.total_bytes} B exceeds {plan.capacity_bytes} B on {arch}"
        )
    # registers
    est = register_estimates(s, arch)
    vg, ag = est["arch_vgpr"], est["agpr"]
    slack = REGISTER_PRUNE_ALLOWANCE
    if mfma:
        if vg > facts.arch_vgprs + slack:
            raise ValueError(
                f"arch-VGPR estimate {vg} exceeds {facts.arch_vgprs} (+{slack})"
            )
        if s.agpr_alloc == (0, 0):
            if vg > facts.arch_vgprs:
                raise ValueError(
                    f"agpr_alloc=(0, 0) needs arch-VGPR + AGPR estimate {vg} <= 256"
                )
        elif s.agpr_alloc is not None:
            n = s.agpr_alloc[0]
            # The reservation is a minimum the backend must honour inside the
            # per-wave share of the unified file; it must leave arch VGPRs (at
            # least the fixed allowance of the estimate), or codegen fails
            # (for example (256, 256) on an eight-wave tile: 256 per wave).
            per_simd = max(-(-w // _CDNA_SIMDS_PER_CU), s.waves_per_eu or 1)
            share = (facts.arch_vgprs + facts.agprs) // per_simd
            if share - n < _REG_ALLOWANCE_FIXED:
                raise ValueError(
                    f"agpr_alloc={s.agpr_alloc} leaves {share - n} of the {share} "
                    f"registers per wave ({per_simd} waves per SIMD at waves={w}) "
                    f"for arch VGPRs; at least {_REG_ALLOWANCE_FIXED} are needed"
                )
            if n < ag:
                raise ValueError(
                    f"agpr_alloc={s.agpr_alloc} below the AGPR estimate {ag}"
                )
            if ag > min(facts.agprs, n) + slack:
                raise ValueError(
                    f"AGPR estimate {ag} exceeds {min(facts.agprs, n)} (+{slack})"
                )
        elif ag > facts.agprs + slack:
            raise ValueError(f"AGPR estimate {ag} exceeds {facts.agprs} (+{slack})")
        wpe = s.waves_per_eu
        if wpe in (2, 3) and vg + ag > (facts.arch_vgprs + facts.agprs) // wpe:
            raise ValueError(
                f"waves_per_eu={wpe} needs arch-VGPR + AGPR estimate {vg + ag} "
                f"<= {(facts.arch_vgprs + facts.agprs) // wpe}"
            )
    else:
        if vg > facts.arch_vgprs + slack:
            raise ValueError(
                f"register estimate {vg} exceeds {facts.arch_vgprs} (+{slack})"
            )
        if s.waves_per_eu == 2 and vg > facts.arch_vgprs // 2:
            raise ValueError(
                "waves_per_eu=2 needs the register estimate <= half the file"
            )
    return s


# ---------------------------------------------------------------------------
# kernarg ABI rocke.attn_bwd.v3
# ---------------------------------------------------------------------------
#
# Order: pointers, i64 scalars, f32 scalars, i32 scalars. Every pointer is 8
# bytes, so the i64 block starts 8-byte aligned and the packed argument bytes
# need no padding. The batch and head strides are i64 (a tensor may span more
# than 2^31 elements between batches or heads; the kernels rebase in i64);
# the token strides stay i32 (the host predicate bounds S_max * stride_t).

_MAIN_PTRS = (
    "Q K V dO dK dV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV SEQ_Q SEQ_KV OFF_Q OFF_KV WORKLIST"
).split()
_MAIN_I64 = "q_b q_h k_b k_h v_b v_h do_b do_h dk_b dk_h dv_b dv_h".split()
_MAIN_F32 = ("scale_log2", "ds_mult", "dk_mult", "dq_mult")
_MAIN_I32 = (
    "q_t k_t v_t do_t dk_t dv_t "
    "h_q h_k h_v G gk gv S_q_max S_kv_max has_len len_stride off64 q_mult q_div kv_mult kv_div "
    "left right bottom_right ws_rows_q ws_rows_kv ws_seg "
    "g_split n_kv_tiles lb_order pair xcd_n xcd_chunk pack_heads use_worklist"
).split()
_PREP_PTRS = (
    "O dO LSE SEQ_Q SEQ_KV OFF_Q OFF_KV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV WORKLIST"
).split()
_PREP_I64 = "o_b o_h do_b do_h l_b l_h".split()
_PREP_I32 = (
    "o_t do_t l_t h_q h_k h_v S_q_max S_kv_max "
    "has_len len_stride off64 q_mult q_div kv_mult kv_div ws_rows_q ws_rows_kv ws_seg "
    "zero_kv zero_dq use_worklist n_batch wl_kn0"
).split()
_CONVERT_PTRS = ("SRC", "DST", "SEQ", "OFF")
_CONVERT_I64 = ("d_b", "d_h")
_CONVERT_F32 = ("mult",)
_CONVERT_I32 = (
    "H d_t S_max has_len len_stride off64 tok_mult tok_div ws_rows ws_seg"
).split()
_DQ_PTRS = "Q K V dO dQ WS_LSE2 WS_DSUM SEQ_Q SEQ_KV OFF_Q OFF_KV".split()
_DQ_I64 = "q_b q_h k_b k_h v_b v_h do_b do_h dq_b dq_h".split()
_DQ_F32 = ("scale_log2", "ds_mult", "dq_mult")
_DQ_I32 = (
    "q_t k_t v_t do_t dq_t "
    "h_q h_k h_v G S_q_max S_kv_max has_len len_stride off64 q_mult q_div kv_mult kv_div "
    "left right bottom_right ws_rows_q ws_seg xcd_n xcd_chunk"
).split()

_ABI_LISTS = {
    "main": (_MAIN_PTRS, _MAIN_I64, _MAIN_F32, _MAIN_I32),
    "prep": (_PREP_PTRS, _PREP_I64, (), _PREP_I32),
    "convert": (_CONVERT_PTRS, _CONVERT_I64, _CONVERT_F32, _CONVERT_I32),
    "dq": (_DQ_PTRS, _DQ_I64, _DQ_F32, _DQ_I32),
}

# struct format of each kernarg kind (little endian): pointer, i64, f32, i32.
KERNARG_FORMATS = ("Q", "q", "f", "i")


def attn_bwd_params(
    stage: str, spec: Optional[AttnBwdSpec] = None
) -> tuple[tuple[str, str], ...]:
    """Ordered ``(name, struct format)`` kernargs of ``stage`` under the v3 ABI.

    ``"Q"`` is a 64-bit global pointer, ``"q"`` an i64, ``"f"`` an f32, ``"i"``
    an i32. The lists do not depend on the spec (every compile-time knob keeps
    the ABI); ``spec`` is accepted so callers can pass the instance they launch.
    """
    _check_in("stage", stage, tuple(_ABI_LISTS))
    ptrs, i64s, f32s, i32s = _ABI_LISTS[stage]
    return (
        tuple((p, "Q") for p in ptrs)
        + tuple((n, "q") for n in i64s)
        + tuple((f, "f") for f in f32s)
        + tuple((i, "i") for i in i32s)
    )


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------

_MAIN_READONLY = frozenset(
    "Q K V dO WS_LSE2 WS_DSUM SEQ_Q SEQ_KV OFF_Q OFF_KV WORKLIST".split()
)
# The lowering's flat-work-group-size attribute has a fixed minimum of 64.
_MIN_WG_ATTR = 64


def _not_built(spec: AttnBwdSpec) -> list[str]:
    """Resolved knob values whose emission is not part of the body yet."""
    out = []
    if spec.transpose_source == "xt_lds":
        out.append("transpose_source='xt_lds'")
    if spec.global_path != "vgpr":
        out.append(f"global_path={spec.global_path!r}")
    if spec.ring_depth != 1:
        out.append(f"ring_depth={spec.ring_depth}")
    if spec.sched != "none" or spec.setprio:
        out.append(f"sched={spec.sched!r}, setprio={spec.setprio}")
    if spec.lds_swizzle and any(
        item.endswith("=xor") for item in spec.lds_swizzle.split(",")
    ):
        out.append(f"lds_swizzle={spec.lds_swizzle!r}")
    if (
        spec.transpose_source == "tr_read"
        and spec.pt_route == "relabel"
        and spec.atom_g13 == "16x16x32"
    ):
        out.append("tr_read with a K-permuted G1/G3 B operand")
    return out


def _declare_main(b, io) -> dict:
    args = {}
    for name, fmt in attn_bwd_params("main"):
        if fmt == "Q":
            if name in ("Q", "K", "V", "dO", "dK", "dV"):
                t = PtrType(io, "global")
            elif name.startswith("WS_"):
                t = PtrType(F32, "global")
            else:
                t = PtrType(I32, "global")
            attrs = {"readonly": True} if name in _MAIN_READONLY else {}
            args[name] = b.param(name, t, **attrs)
        elif fmt == "q":
            args[name] = b.param(name, I64)
        elif fmt == "f":
            args[name] = b.param(name, F32)
        else:
            args[name] = b.param(name, I32)
    return args


def _decode_sequences(b, s: AttnBwdSpec, a: dict, bz):
    """``(seq_q, seq_kv)`` of batch / THD sequence ``bz``.

    Batched: the lengths are ``S_max`` or the clamped SEQ_LEN counts (runtime
    ``has_len``). THD: ``OFF_Q`` / ``OFF_KV`` entries ``bz`` and ``bz + 1``
    (int32 or int64 by the runtime ``off64``) give the first token
    ``off * mult / div`` and the length (i64), clamped to ``[0, S_max]`` and
    ``min``-ed with the SEQ_LEN count under ``has_len``; the workspace rows of
    the sequence start at its first token (``ws_seg = 0``) or at its slot
    ``bz * S_max`` (``ws_seg != 0``).
    """
    return _decode_side(b, s, a, bz, "q"), _decode_side(b, s, a, bz, "kv")


def _decode_side(b, s: AttnBwdSpec, a: dict, bz, side: str):
    """The q-side or kv-side ``SeqInfo`` of batch / THD sequence ``bz``."""
    smax, off, seq = {
        "q": ("S_q_max", "OFF_Q", "SEQ_Q"),
        "kv": ("S_kv_max", "OFF_KV", "SEQ_KV"),
    }[side]
    if s.seq_mode == "thd":
        return decode_thd(
            b,
            bz,
            s_max=a[smax],
            off_ptr=a[off],
            off64=a["off64"],
            mult=a[f"{side}_mult"],
            div=a[f"{side}_div"],
            seq_ptr=a[seq],
            has_len=a["has_len"],
            len_stride=a["len_stride"],
            ws_seg=a["ws_seg"],
        )
    return decode_batched(
        b,
        bz,
        s_max=a[smax],
        seq_ptr=a[seq],
        has_len=a["has_len"],
        len_stride=a["len_stride"],
    )


def _emit_main(spec: AttnBwdSpec, arch: str, *, sentinel: bool = True):
    """Emit the main kernel of ``spec`` on ``arch``.

    ``sentinel=False`` drops the dead-row term ``lse2 < +inf`` from the dS
    select; it exists only for the negative-control tests and is not a spec
    field.
    """
    s = validate_attn_bwd_spec(spec, arch)
    missing = _not_built(s)
    if missing:
        raise NotImplementedError(
            "the layout-general attention backward main kernel is not built for "
            + ", ".join(missing)
        )
    facts = attn_bwd_arch_facts(arch)
    geom = BwdTileGeometry.from_spec(s, arch)
    plan = lds_plan(s, arch)
    io = ir_type(s.dtype)
    D, km, kn = s.head_size, s.block_m, s.block_n

    b = IRBuilder(s.kernel_name("main"))
    a = _declare_main(b, io)
    b.kernel.attrs["max_workgroup_size"] = max(geom.threads, _MIN_WG_ATTR)
    if facts.matrix_path == "mfma" and s.agpr_alloc is not None:
        b.kernel.attrs["agpr_alloc"] = tuple(s.agpr_alloc)
    if s.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = int(s.waves_per_eu)
    if s.scheduler_strategy is not None:
        b.kernel.attrs[SCHEDULER_STRATEGY_ATTR] = normalize_scheduler_strategy(
            s.scheduler_strategy
        )

    tid, lane, wave = lane_and_wave(b, facts.wave_size)
    atomic = s.dkv_mode == "atomic"
    # CTA decode: kv tile fastest, then the head unit (and split), then b; the
    # load-balance order permutes the kv tiles of one (unit, b).
    split_arg = a["g_split"] if atomic else None
    work = decode_cta(
        b, x=b.block_id_x(), y=b.block_id_y(), z=b.block_id_z(),
        n_kv_tiles=a["n_kv_tiles"], lb_order=a["lb_order"], g_split=split_arg,
    )  # fmt: skip
    bz = work.batch
    seq_q, seq_kv = _decode_sequences(b, s, a, bz)
    # Direct mode: the unit is the kv head and owns its query heads
    # [unit * G, unit * G + G). Atomic mode: the unit's G / g_split heads of
    # this split share one K head and one V head.
    heads = head_range(
        b, unit=work.unit, split=work.split, group=a["G"], gk=a["gk"], gv=a["gv"],
        g_split=split_arg,
    )  # fmt: skip
    hk, hv, h0, n_h = heads.hk, heads.hv, heads.h0, heads.n_heads

    def tensor(name, prefix, seq):
        return StridedTensorAddr(
            b, a[name], dtype=io, seq=seq, stage_vec=s.stage_vec,
            strides=(a[f"{prefix}_b"], a[f"{prefix}_h"], a[f"{prefix}_t"]),
        )  # fmt: skip

    q_addr = tensor("Q", "q", seq_q)
    do_addr = tensor("dO", "do", seq_q)
    k_addr = tensor("K", "k", seq_kv)
    v_addr = tensor("V", "v", seq_kv)
    nh = a["h_q"] if s.ws_layout == "token_major" else None

    def ws(name, width):
        return WorkspaceAddr(
            b, a[name], ws_rows=a["ws_rows_q"], width=width, row0=seq_q.ws_row0,
            layout=s.ws_layout, n_heads=nh,
        )  # fmt: skip

    stats = StatsView(ws("WS_LSE2", 1), ws("WS_DSUM", 1))
    sink_cls = PackedAtomicWorkspaceSink if s.head_pack else AtomicWorkspaceSink
    sink = sink_cls(b, ws("WS_DQ", D), seq=seq_q, scale=a["dq_mult"])
    zero = b.const_i32(0)
    if s.mask_class == "band":
        # Per-sequence diagonal: 0 (top-left) or len_kv - len_q (bottom-right).
        off = b.select(
            b.cmp_ne(a["bottom_right"], zero),
            b.sub(seq_kv.length, seq_q.length),
            zero,
        )
        bounds = AttnRuntimeBounds(
            seq_q.length, seq_kv.length, off, a["left"], a["right"]
        )
        mask = RuntimeBand(b, bounds, k_m0=km, k_n0=kn)
    else:
        mask = NoMaskKTail(b, seq_q.length, seq_kv.length, k_m0=km, k_n0=kn)
    lds = BwdLds(b, plan, s.dtype)
    step = BwdTileStep(
        b, geom, mask=mask, stats=stats, dq_sink=sink, lds=lds, tid=tid, lane=lane,
        wave=wave, scale_log2=a["scale_log2"], ds_mult=a["ds_mult"],
        sentinel=sentinel,
    )  # fmt: skip

    # One kv tile per CTA (natural or reverse order; the plan refuses mirror
    # pairing, whose second kv tile per CTA is not built).
    _kv_tile(
        b, s, a, step, mask, kt=work.first_kt, hk=hk, hv=hv, h0=h0, n_h=n_h,
        seq_q=seq_q, seq_kv=seq_kv, q_addr=q_addr, do_addr=do_addr, k_addr=k_addr,
        v_addr=v_addr, io=io, tid=tid,
    )  # fmt: skip
    b.ret()
    return b.kernel


def _kv_tile(
    b, s, a, step, mask, *, kt, hk, hv, h0, n_h, seq_q, seq_kv, q_addr, do_addr,
    k_addr, v_addr, io, tid,
) -> None:  # fmt: skip
    """Everything the CTA does for kv tile ``kt``: staging, q loops, epilogue."""
    geom = step.geom
    km, kn = s.block_m, s.block_n
    zero, one = b.const_i32(0), b.const_i32(1)
    k0 = b.mul(kt, b.const_i32(kn))
    live = b.cmp_lt(k0, seq_kv.length)  # CTA-uniform
    k_slab = k_addr.slab(hk, k0)
    v_slab = v_addr.slab(hv, k0)
    step.stage_kv(b, k_slab, v_slab, k0, live)

    if s.head_pack:
        # pack_heads = P > 0: P query heads share the M rows of the single q
        # tile (the plan guarantees P * len_q <= kM0, so n_qt <= 1); the loop
        # visits ceil(n_h / P) head groups. pack_heads = 0 is the unpacked loop.
        pk = a["pack_heads"]
        pack_on = b.cmp_ne(pk, zero)
        per = b.select(pack_on, pk, one)
        rows_per_head = b.select(pack_on, b.smax(seq_q.length, one), b.const_i32(km))
        n_grp = b.div(b.add(n_h, b.sub(per, one)), per)
    else:
        pack_on = per = rows_per_head = None
        n_grp = n_h

    # Inverse band (empty band, a tile past len_kv or len_q == 0: 0 tiles).
    qt_start, n_qt = mask.q_tile_range(b, k0)
    # Flattened order: q tiles outer, head groups inner, so every q-tile range
    # is a contiguous run of steps; the trip counts live in SGPRs.
    qt_end = b.add(qt_start, n_qt)
    if s.edge_tiles:
        # Interior q tiles [i0, i1) skip the per-cell selects: the steps run
        # as edge [qt_start, i0), interior [i0, i1) and edge [i1, qt_end)
        # loops in the single loop's order, so the accumulation order (and
        # dK / dV bit for bit) is the same. A packed tile is an edge tile.
        i0, i1 = mask.interior_q_range(b, k0, qt_start, n_qt)
        if pack_on is not None:
            i1 = b.select(pack_on, i0, i1)
        ranges = ((qt_start, i0, True), (i0, i1, False), (i1, qt_end, True))
    else:
        ranges = ((qt_start, qt_end, True),)
    trips = [b.readfirstlane(b.mul(n_grp, b.sub(e, s0))) for s0, e, _ in ranges]
    acc0 = step.zero_acc(b)  # after the exit test: accumulators start here
    fm, fn = geom.g1.frags_m, geom.g1.frags_n
    n_acc = len(acc0.flat())

    def carried(prefix, acc, qt0):
        vals = acc.flat()
        half = len(vals) // 2
        names = [f"{prefix}dk{i}" for i in range(half)]
        names += [f"{prefix}dv{i}" for i in range(half)]
        names += [f"{prefix}qt", f"{prefix}grp"]
        return list(zip(names, vals + [qt0, zero]))

    def q_steps(acc, qt0, trip, *, edge, name):
        """``trip`` steps from q tile ``qt0``, head group 0 (q tile outer)."""
        loop = b.scf_for_iter(
            zero, trip, one, carried(f"bwd_{name}", acc, qt0),
            iv_name=f"bwd_{name}it", elide_trailing_barrier=False,
        )  # fmt: skip
        with loop as (_it, cur):
            qt, gi = cur[n_acc], cur[n_acc + 1]
            # next (q tile, head group): head groups inner
            nxt = b.add(gi, one)
            wrap = b.cmp_ge(nxt, n_grp)
            qt_next = b.add(qt, b.select(wrap, one, zero))
            gi_next = b.select(wrap, zero, nxt)
            q0 = b.mul(qt, b.const_i32(km))
            if s.head_pack:
                first = b.mul(gi, per)
                h = b.add(h0, first)
                rows = PackedRows(
                    head0=h, n_heads=b.smin(per, b.sub(n_h, first)),
                    rows_per_head=rows_per_head, row0=q0,
                )  # fmt: skip
                packed = PackedStep(rows=rows, q_addr=q_addr, do_addr=do_addr)
                q_slab = do_slab = None
            else:
                h = b.add(h0, gi)
                packed = None
                q_slab, do_slab = q_addr.slab(h, q0), do_addr.slab(h, q0)
            step.sink.begin_head(b, h)
            out = step.run_q_step(
                b, DkDvAcc.from_flat(cur[:n_acc], fm, fn) if n_acc else acc0,
                q_slab, do_slab, qt, h, k0, edge=edge, packed=packed,
            )  # fmt: skip
            b.scf_yield(*out.flat(), qt_next, gi_next)
        res = list(loop.results)[:n_acc]
        return DkDvAcc.from_flat(res, fm, fn) if n_acc else acc0

    acc = acc0
    names = ("q", "qi_", "qe_")
    for (s0, _e, edge), trip, name in zip(ranges, trips, names):
        acc = q_steps(acc, s0, trip, edge=edge, name=name)
    acc = step.finish(b, acc)
    if s.dkv_mode == "atomic":
        _atomic_epilogue(
            b, s, a, step, acc, hk=hk, hv=hv, k0=k0, seq_kv=seq_kv, live=live
        )
    else:
        _direct_epilogue(
            b, s, a, step, acc, hk=hk, k0=k0, seq_kv=seq_kv, io=io, tid=tid
        )


def _atomic_epilogue(b, s, a, step, acc, *, hk, hv, k0, seq_kv, live) -> None:
    """fp32 dK / dV of the key tile added into the ``WS_DK`` / ``WS_DV`` workspace.

    Inside ``scf_if(live)`` (a tile past ``len_kv`` adds nothing). Every
    accumulator element of the lane is one ``global_atomic_add_f32`` (agent
    scope) into kv head ``hk`` (dK, times ``dk_mult``) or ``hv`` (dV), with the
    workspace rebased in i64 per head and tile row. One branch per key row of
    the lane: a row is added only when it is inside the sequence (``k <
    len_kv``) and the workspace (``< ws_rows_kv``), and the atomics of a row
    share its row base. Batched rows ``[len_kv, S_kv_max)`` keep the zeros
    written by prep, and the dK / dV converts store them as zeros.
    """
    D = s.head_size
    tok_major = s.ws_layout == "token_major"
    with b.scf_if(live):
        for which, ptr, head, n_heads, mult in (
            ("dk", "WS_DK", hk, a["h_k"], a["dk_mult"]),
            ("dv", "WS_DV", hv, a["h_v"], None),
        ):
            slab = WorkspaceAddr(
                b, a[ptr], ws_rows=a["ws_rows_kv"], width=D, row0=seq_kv.ws_row0,
                layout=s.ws_layout, n_heads=n_heads if tok_major else None,
            ).slab(head, k0)  # fmt: skip
            for row, items in step.dkv_row_slots(b, which, acc):
                if mult is not None:
                    items = [(c, b.fmul(v, mult)) for c, v in items]
                in_seq = b.cmp_lt(b.add(k0, row), seq_kv.length)
                slab.atomic_add_row(row, items, in_seq)


def _direct_epilogue(b, s, a, step, acc, *, hk, k0, seq_kv, io, tid) -> None:
    """dK / dV rows of the key tile to the output dtype at their strides.

    Rows ``k < len_kv`` get the values (``dk_mult`` on dK); batched rows
    ``len_kv <= k < S_kv_max`` get exact zeros; rows at or past ``S_kv_max``
    are not touched. The stores are predicated per row (after the loop).
    """
    D, kn = s.head_size, s.block_n
    threads = step.geom.threads
    per_row = D // 8
    total = kn * per_row
    zero8 = b.zero_vec(io, 8)
    for which, ptr, prefix, mult in (
        ("dk", "dK", "dk", a["dk_mult"]),
        ("dv", "dV", "dv", None),
    ):
        slab = StridedTensorAddr(
            b, a[ptr], dtype=io, seq=seq_kv, stage_vec=s.stage_vec,
            strides=(a[f"{prefix}_b"], a[f"{prefix}_h"], a[f"{prefix}_t"]),
        ).slab(hk, k0, clamp=False)  # fmt: skip
        read = step.dkv_rows(b, which, acc, mult)
        for it in range(-(-total // threads)):
            chunk = tid if it == 0 else b.add(tid, b.const_i32(it * threads))
            r = b.div(chunk, b.const_i32(per_row))
            c = b.mul(b.mod(chunk, b.const_i32(per_row)), b.const_i32(8))
            k = b.add(k0, r)
            store, row_live = dkv_store_rule(
                b, k, seq_kv.length, seq_mode=s.seq_mode, s_kv_max=a["S_kv_max"]
            )
            if (it + 1) * threads > total:
                store = b.land(store, b.cmp_lt(chunk, b.const_i32(total)))
            with b.scf_if(store):
                v = b.select(row_live, read(r, c), zero8)
                slab.store_vec(k, c, v)
        b.sync()  # reads of the epilogue buffer done before it is reused


def build_attn_bwd_main(spec: AttnBwdSpec, *, arch: str):
    """Build the layout-general backward main kernel (``rocke.attn_bwd.v3``).

    The spec is resolved and validated for ``arch`` first (an illegal
    geometry raises ``ValueError``); knob values whose emission is not built
    yet raise ``NotImplementedError`` before any IR is emitted.
    """
    return _emit_main(spec, arch)
