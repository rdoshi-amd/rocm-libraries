# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stage-scheduler table for the attention backward tile step.

One q step of the backward body runs five GEMMs (G0 ``S = Q K^T``, G1
``dV += P^T dO``, G2 ``dP = dO V^T``, G3 ``dK += dS^T Q``, G4
``dQ = dS K``). :class:`StageTable` groups them into four stages and gives,
per wave and per lane, how many MFMA (or WMMA) instructions and how many
VMEM-read, VMEM-write, DS-read and DS-write instructions each stage issues.
The counts come from the tile, the warp grids and each GEMM's own atom, never
from G0's atom alone, and K = 32 steps on an arch without a native 16x16x32
atom count as two 16x16x16 instructions.

Default stage order (``sched = "stage_table"``):

* A: G0 MFMAs with the next tile's global reads and the dO^T fragment reads
  (and the K fragment reads when ``kv_residency = "lds"``);
* B: G1 + G2 with the Q^T fragment reads and the next tile's LDS writes (and
  the V fragment reads when ``kv_residency = "lds"``, the dV accumulator
  read-modify-write when ``acc_in_lds``);
* C: G3 with the dS writes, the first dS slice and the next Q / lse2 reads
  (and the dK accumulator read-modify-write when ``acc_in_lds``);
* D: G4 with the remaining dS slices (and K^T reads when K^T comes from LDS)
  and the next dO / Dsum reads.

``stage_table_atomics`` also places the previous step's dQ atomics
(``VMEM_WRITE``) into stage A. Each stage's work is interleaved with
``sched_group_barrier(mask, count, 0)`` groups (one MFMA per group, memory
work spread evenly between them) and the stages are fenced with
``sched_barrier(0)`` except after D. ``fences_only`` emits the fences alone,
``iglp0`` / ``iglp1`` emit ``iglp_opt`` once at the loop head and nothing
else (never together with a stage table), ``none`` emits nothing.

The memory counts are the count *model* of the body's data movement (one
instruction per MMA fragment read, vector width per buffer kind, the XT
writer of ``_attention_bwd_lds``); :meth:`StageTable.ir_count_model` gives the
IR op counts the emission produces, so a test can compare model and IR.

Residency: with ``kv_residency = "lds"`` the K (G0) and V (G2) B fragments are
read from their resident LDS images (D contiguous, so one read per 16 bytes)
once per distinct (n tile, k step) of the wave in every q step. With
``acc_in_lds`` the fp32 dV (G1) and dK (G3) accumulators live in row-major
fp32 LDS images; each 16x16 output tile of the wave is read before its MMA
chain and written back after it, one ``b32`` access per accumulator element
(the lanes of an MMA C fragment hold distinct rows, so the elements are not
contiguous in a row-major image).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rocke.core.arch import ArchTarget
from rocke.helpers.schedule import DS_READ, DS_WRITE, MFMA, VMEM_READ, VMEM_WRITE

from ._attention_bwd_lds import default_xt_writer

__all__ = [
    "GEMMS",
    "SCHEDS",
    "STAGES",
    "STAGE_GEMMS",
    "STAGE_TABLE_SCHEDS",
    "GemmCounts",
    "StageCounts",
    "StageTable",
    "instruction_k",
    "interleave_groups",
    "native_atom_ks",
]

SCHEDS = ("none", "iglp0", "iglp1", "fences_only", "stage_table", "stage_table_atomics")
STAGE_TABLE_SCHEDS = ("stage_table", "stage_table_atomics")
STAGES = ("A", "B", "C", "D")
GEMMS = ("g0", "g1", "g2", "g3", "g4")
STAGE_GEMMS: Mapping[str, tuple[str, ...]] = {
    "A": ("g0",),
    "B": ("g1", "g2"),
    "C": ("g3",),
    "D": ("g4",),
}
_TRANSPOSE_SOURCES = ("xt_lds", "tr_read", "lds_plain", "wmma_lds")
_DTYPES = {"fp16": "fp16", "f16": "fp16", "bf16": "bf16"}
_ELEM_BYTES = 2
# Memory classes in emission order around each MFMA group: VMEM reads go
# before the MFMA (they have the longest latency), the rest after it.
_PRE = (VMEM_READ,)
_POST = (DS_READ, DS_WRITE, VMEM_WRITE)


def _cdiv(a: int, b: int) -> int:
    return -(-int(a) // int(b))


def _mma_ops(arch: str, dtype: str):
    tgt = ArchTarget.from_gfx(arch)
    dt = _DTYPES.get(dtype)
    if dt is None:
        raise ValueError(f"unsupported dtype {dtype!r}")
    family = "mma" if tgt.has_mfma else "wmma"
    ops = tgt.mma.enumerate(
        family=family, a_dtype=dt, b_dtype=dt, c_dtype="fp32", m=16, n=16
    )
    ops = [op for op in ops if op.wave_size == tgt.wave_size and op.a_frag_len > 0]
    if not ops:
        raise ValueError(f"no 16x16 {dt} MMA atom on {arch}")
    return sorted(ops, key=lambda op: op.k)


def native_atom_ks(arch: str, dtype: str = "fp16") -> tuple[int, ...]:
    """K extents of the native 16x16 MMA atoms of ``arch`` (core catalog)."""
    return tuple(op.k for op in _mma_ops(arch, dtype))


def instruction_k(arch: str, atom_k: int, dtype: str = "fp16") -> int:
    """K of the instruction that implements a 16x16x``atom_k`` step on ``arch``.

    A native atom is one instruction; a K = 32 step on an arch whose widest
    native atom is K = 16 is two 16x16x16 instructions.
    """
    ks = native_atom_ks(arch, dtype)
    if atom_k in ks:
        return int(atom_k)
    smaller = [k for k in ks if atom_k % k == 0]
    if not smaller:
        raise ValueError(f"no instruction can implement a K={atom_k} step on {arch}")
    return max(smaller)


def _atom_k(value: Any) -> int:
    if value is None:
        return 16
    if isinstance(value, int):
        return int(value)
    k = getattr(value, "k", None)
    if k is not None:
        return int(k)
    if isinstance(value, str):
        text = value.lower().replace("wmma", "")
        parts = text.split("x")
        if len(parts) == 3:
            return int(parts[2])
    raise ValueError(f"cannot read an atom K from {value!r}")


def _atom_from(atoms: Any, *names: str) -> Any:
    if atoms is None:
        return None
    for name in names:
        if isinstance(atoms, Mapping) and name in atoms:
            return atoms[name]
        if hasattr(atoms, name):
            return getattr(atoms, name)
    return None


def interleave_groups(
    mfma: int, mem: Sequence[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Spread memory work evenly between ``mfma`` single-MFMA groups.

    Group ``i`` receives ``floor((i+1) c / m) - floor(i c / m)`` of a class
    with total ``c``; VMEM reads go before the MFMA, the other classes after.
    Adjacent groups of one class are merged. With no MFMA the memory groups
    are emitted as they are. The per-mask sums always equal the inputs.
    """
    raw: list[tuple[int, int]] = []
    if mfma <= 0:
        raw = [(m, c) for m, c in mem if c > 0]
    else:
        pre = [(m, c) for m, c in mem if m in _PRE and c > 0]
        post = [(m, c) for m, c in mem if m not in _PRE and c > 0]
        for i in range(mfma):
            for m, c in pre:
                s = (i + 1) * c // mfma - i * c // mfma
                if s:
                    raw.append((m, s))
            raw.append((MFMA, 1))
            for m, c in post:
                s = (i + 1) * c // mfma - i * c // mfma
                if s:
                    raw.append((m, s))
    out: list[tuple[int, int]] = []
    for m, c in raw:
        if out and out[-1][0] == m:
            out[-1] = (m, out[-1][1] + c)
        else:
            out.append((m, c))
    return out


@dataclass(frozen=True)
class GemmCounts:
    """MMA instructions per wave per q step, per GEMM."""

    g0: int
    g1: int
    g2: int
    g3: int
    g4: int

    def as_dict(self) -> dict[str, int]:
        return {g: getattr(self, g) for g in GEMMS}

    @property
    def total(self) -> int:
        return self.g0 + self.g1 + self.g2 + self.g3 + self.g4


@dataclass(frozen=True)
class StageCounts:
    """Per-wave instruction counts of one stage (memory counts per lane)."""

    stage: str
    mfma: int
    vmem_read: int = 0
    vmem_write: int = 0
    ds_read: int = 0
    ds_write: int = 0

    def memory(self) -> tuple[tuple[int, int], ...]:
        return (
            (VMEM_READ, self.vmem_read),
            (DS_READ, self.ds_read),
            (DS_WRITE, self.ds_write),
            (VMEM_WRITE, self.vmem_write),
        )

    def by_mask(self) -> dict[int, int]:
        return {
            MFMA: self.mfma,
            VMEM_READ: self.vmem_read,
            VMEM_WRITE: self.vmem_write,
            DS_READ: self.ds_read,
            DS_WRITE: self.ds_write,
        }


@dataclass(frozen=True)
class StageTable:
    """Backward stage table for one resolved tile geometry on one arch.

    Use :meth:`from_geometry` (any object with the ``BwdTileGeometry`` field
    names) or :meth:`from_params`. ``atom_g02`` / ``atom_g13`` / ``atom_g4``
    are the GEMM step K extents (16 or 32); the instruction K is resolved
    from the catalog of ``arch``.
    """

    arch: str
    head_size: int
    k_m0: int
    k_n0: int
    k_k4: int
    waves: int
    grid_g4: tuple[int, int] = (1, 0)
    atom_g02: int = 16
    atom_g13: int = 16
    atom_g4: int = 16
    dtype: str = "fp16"
    transpose_source: str = "xt_lds"
    kt_source: str = "reg"
    global_path: str = "vgpr"
    pt_route: str = "relabel"
    stage_vec: int = 8
    dq_mode: str = "atomic"
    sched: str = "stage_table"
    setprio: bool = False
    kv_residency: str = "reg"
    acc_in_lds: bool = False

    def __post_init__(self) -> None:
        if self.grid_g4[1] == 0:
            object.__setattr__(self, "grid_g4", (1, int(self.waves)))
        g4r, g4c = self.grid_g4
        w, d, m0, n0, k4 = self.waves, self.head_size, self.k_m0, self.k_n0, self.k_k4
        if self.sched not in SCHEDS:
            raise ValueError(f"unknown sched {self.sched!r}; expected one of {SCHEDS}")
        if self.setprio and self.sched not in STAGE_TABLE_SCHEDS:
            raise ValueError("setprio is legal only inside a stage table")
        if self.transpose_source not in _TRANSPOSE_SOURCES:
            raise ValueError(f"unknown transpose_source {self.transpose_source!r}")
        if self.kt_source not in ("reg", "lds"):
            raise ValueError("kt_source must be 'reg' or 'lds'")
        if self.kv_residency not in ("reg", "lds"):
            raise ValueError("kv_residency must be 'reg' or 'lds'")
        if self.kt_source == "reg" and self.kv_residency != "reg":
            raise ValueError("kt_source='reg' requires kv_residency='reg'")
        if not isinstance(self.acc_in_lds, bool):
            raise ValueError("acc_in_lds must be a bool")
        if self.global_path not in ("vgpr", "dma"):
            raise ValueError("global_path must be 'vgpr' or 'dma'")
        if self.pt_route not in ("relabel", "lds"):
            raise ValueError("pt_route must be 'relabel' or 'lds'")
        if self.dq_mode not in ("atomic", "split"):
            raise ValueError("dq_mode must be 'atomic' or 'split'")
        if self.stage_vec not in (1, 8):
            raise ValueError("stage_vec must be 8 or 1")
        if g4r * g4c != w:
            raise ValueError(f"warp grid G4 {self.grid_g4} does not hold {w} waves")
        checks = (
            (n0 % (16 * w) == 0, f"kN0 {n0} % (16 * waves {w}) != 0"),
            (m0 % 16 == 0, f"kM0 {m0} % 16 != 0"),
            (d % (16 * g4c) == 0, f"D {d} % (16 * g4c {g4c}) != 0"),
            (m0 % (16 * g4r) == 0, f"kM0 {m0} % (16 * g4r {g4r}) != 0"),
            (n0 % k4 == 0, f"kN0 {n0} % kK4 {k4} != 0"),
            (k4 % self.atom_g4 == 0, f"kK4 {k4} % atom_g4 K {self.atom_g4} != 0"),
            (d % self.atom_g02 == 0, f"D {d} % atom_g02 K {self.atom_g02} != 0"),
            (m0 % self.atom_g13 == 0, f"kM0 {m0} % atom_g13 K {self.atom_g13} != 0"),
            (self.atom_g13 != 32 or m0 >= 32, "atom_g13 K = 32 needs kM0 >= 32"),
        )
        for ok, why in checks:
            if not ok:
                raise ValueError(f"illegal backward stage-table geometry: {why}")
        for k in (self.atom_g02, self.atom_g13, self.atom_g4):
            instruction_k(self.arch, k, self.dtype)  # raises when not implementable

    # ----- construction ---------------------------------------------------

    @classmethod
    def from_params(cls, **kwargs: Any) -> StageTable:
        return cls(**kwargs)

    @classmethod
    def from_geometry(
        cls,
        geom: Any,
        arch: str,
        *,
        dtype: str = "fp16",
        stage_vec: int = 8,
        dq_mode: str = "atomic",
        acc_in_lds: bool = False,
    ) -> StageTable:
        """Read a ``BwdTileGeometry``-like object (duck-typed attributes).

        ``acc_in_lds`` is a family-spec field rather than a geometry field;
        an ``acc_in_lds`` attribute on ``geom`` takes precedence over the
        keyword.
        """
        atoms = getattr(geom, "atoms", None)
        grid = getattr(geom, "grid_g4", None)
        return cls(
            arch=arch,
            head_size=int(geom.head_size),
            k_m0=int(geom.k_m0),
            k_n0=int(geom.k_n0),
            k_k4=int(geom.k_k4),
            waves=int(geom.waves),
            grid_g4=tuple(grid) if grid is not None else (1, 0),
            atom_g02=_atom_k(_atom_from(atoms, "g02", "g0", "atom_g02")),
            atom_g13=_atom_k(_atom_from(atoms, "g13", "g1", "atom_g13")),
            atom_g4=_atom_k(_atom_from(atoms, "g4", "atom_g4")),
            dtype=dtype,
            transpose_source=str(getattr(geom, "transpose_source", "xt_lds")),
            kt_source=str(getattr(geom, "kt_source", "reg")),
            global_path=str(getattr(geom, "global_path", "vgpr")),
            pt_route=str(getattr(geom, "pt_route", "relabel")),
            stage_vec=int(getattr(geom, "stage_vec", stage_vec)),
            dq_mode=str(getattr(geom, "dq_mode", dq_mode)),
            sched=str(getattr(geom, "sched", "stage_table")),
            setprio=bool(getattr(geom, "setprio", False)),
            kv_residency=str(getattr(geom, "kv_residency", "reg")),
            acc_in_lds=bool(getattr(geom, "acc_in_lds", acc_in_lds)),
        )

    # ----- arch facts -----------------------------------------------------

    @property
    def wave_size(self) -> int:
        return int(ArchTarget.from_gfx(self.arch).wave_size)

    def _op(self, k: int):
        for op in _mma_ops(self.arch, self.dtype):
            if op.k == k:
                return op
        raise ValueError(f"no native 16x16x{k} atom on {self.arch}")

    @property
    def instr_k(self) -> dict[str, int]:
        """Instruction K per GEMM after splitting non-native K = 32 steps."""
        k02 = instruction_k(self.arch, self.atom_g02, self.dtype)
        k13 = instruction_k(self.arch, self.atom_g13, self.dtype)
        k4 = instruction_k(self.arch, self.atom_g4, self.dtype)
        return {"g0": k02, "g1": k13, "g2": k02, "g3": k13, "g4": k4}

    def mma_op_id(self, gemm: str) -> str:
        return self._op(self.instr_k[gemm]).op_id

    def frag_len(self, gemm: str) -> int:
        return int(self._op(self.instr_k[gemm]).a_frag_len)

    @property
    def c_frag_len(self) -> int:
        return int(self._op(16).c_frag_len)

    # ----- MMA counts -----------------------------------------------------

    @property
    def mfma(self) -> GemmCounts:
        """MMA instructions per wave per q step, per GEMM, from the warp grids."""
        w, d, m0, n0 = self.waves, self.head_size, self.k_m0, self.k_n0
        g4r, g4c = self.grid_g4
        k = self.instr_k
        g0 = (m0 // 16) * (n0 // (16 * w)) * (d // k["g0"])
        g1 = (n0 // (16 * w)) * (d // 16) * (m0 // k["g1"])
        g4 = 0
        if self.dq_mode == "atomic":
            g4 = (m0 // (16 * g4r)) * (d // (16 * g4c)) * (n0 // k["g4"])
        return GemmCounts(g0=g0, g1=g1, g2=g0, g3=g1, g4=g4)

    @property
    def dq_atomics_per_lane(self) -> int:
        """fp32 dQ atomics per lane per q step: ``kM0 * D / (W * wave_size)``."""
        if self.dq_mode != "atomic":
            return 0
        return self.k_m0 * self.head_size // (self.waves * self.wave_size)

    # ----- memory model ---------------------------------------------------

    def _reads_per_frag(self, gemm: str, source: str) -> int:
        """LDS read instructions for one lane's operand fragment of ``gemm``.

        ``source``: "contig" (K contiguous in LDS: plain image for A operands,
        XT image, or the WMMA LDS transpose), "tr_read" (``ds_read_tr16_b64``,
        four elements per read) or "lds_plain" (one element per read).
        """
        n = self.frag_len(gemm)
        if source == "tr_read":
            return _cdiv(n, 4)
        if source == "lds_plain":
            return n
        return _cdiv(n * _ELEM_BYTES, 16)

    @property
    def _tsrc(self) -> str:
        if self.transpose_source in ("xt_lds", "wmma_lds"):
            return "contig"
        return self.transpose_source

    def _prefetch_loads(self) -> int:
        vec = 8 if self.global_path == "dma" else self.stage_vec
        threads = self.waves * self.wave_size
        tile = _cdiv(self.k_m0 * self.head_size, threads * vec)
        stats = _cdiv(self.k_m0, self.wave_size)
        return 2 * tile + 2 * stats

    def _next_tile_lds_writes(self) -> int:
        if self.global_path == "dma":
            return 0  # the DMA fills LDS; no ds_write
        threads = self.waves * self.wave_size
        tile = _cdiv(self.k_m0 * self.head_size, threads * self.stage_vec)
        n = 2 * tile
        if self.transpose_source == "xt_lds":
            w = default_xt_writer(self.k_m0, self.head_size, threads)
            n += 2 * w.stores_per_thread
        n += 2 * _cdiv(self.k_m0, threads)
        return n

    def _ds_tile_writes(self) -> int:
        tiles = (self.k_m0 // 16) * (self.k_n0 // (16 * self.waves))
        per_tile = 1 if self.wave_size == 64 else self.c_frag_len
        return tiles * per_tile

    def _kv_frag_reads(self, gemm: str) -> int:
        """LDS reads of the wave's resident K (G0) or V (G2) B fragments."""
        if self.kv_residency != "lds":
            return 0
        k = self.instr_k[gemm]
        frags = (self.k_n0 // (16 * self.waves)) * (self.head_size // k)
        return frags * self._reads_per_frag(gemm, "contig")

    def _acc_rmw(self) -> int:
        """DS reads (= DS writes) of one of dK / dV per wave per q step."""
        if not self.acc_in_lds:
            return 0
        tiles = (self.k_n0 // (16 * self.waves)) * (self.head_size // 16)
        return tiles * self.c_frag_len

    def stages(
        self, *, prefetch: bool = True, prev_atomics: bool = True
    ) -> tuple[StageCounts, ...]:
        """Per-stage counts of one q step.

        ``prefetch``: the step loads / stages the next q tile (false on the
        peeled last step). ``prev_atomics``: a previous step's dQ atomics are
        pending (false on the first step); used only by
        ``stage_table_atomics``.
        """
        w, d, m0, n0, kk4 = self.waves, self.head_size, self.k_m0, self.k_n0, self.k_k4
        g4r, g4c = self.grid_g4
        k = self.instr_k
        mf = self.mfma
        t = self._tsrc
        g4_on = self.dq_mode == "atomic"

        # A operands of G0/G2 (next Q / dO), B operands of G1/G3 (dO^T / Q^T).
        bt_frags = (d // 16) * (m0 // k["g1"])
        bt_reads = bt_frags * self._reads_per_frag("g1", t)
        next_a_reads = (
            (m0 // 16) * (d // k["g0"]) * self._reads_per_frag("g0", "contig")
        )
        stats_reads = m0 // 16
        slice_reads = (
            (m0 // (16 * g4r)) * (kk4 // k["g4"]) * self._reads_per_frag("g4", "contig")
        )
        n_slices = n0 // kk4 if g4_on else 0

        a_vmem = self._prefetch_loads() if prefetch else 0
        a_vmem_w = (
            self.dq_atomics_per_lane
            if self.sched == "stage_table_atomics" and prev_atomics
            else 0
        )
        a_ds_r = bt_reads
        a_ds_w = 0
        b_ds_r = bt_reads
        b_ds_w = self._next_tile_lds_writes() if prefetch else 0
        if self.pt_route == "lds":
            a_ds_w += self._ds_tile_writes()  # P to LDS after G0
            pt_frags = (n0 // (16 * w)) * (m0 // k["g1"])
            b_ds_r += pt_frags * self._reads_per_frag("g1", "contig")
        c_ds_w = self._ds_tile_writes() if (g4_on or self.pt_route == "lds") else 0
        c_ds_r = (slice_reads if n_slices else 0) + (
            next_a_reads + stats_reads if prefetch else 0
        )
        if self.pt_route == "lds":
            c_ds_r += (
                (n0 // (16 * w))
                * (m0 // k["g3"])
                * self._reads_per_frag("g3", "contig")
            )
        # Resident K / V fragments and LDS accumulators (zero when absent).
        a_ds_r += self._kv_frag_reads("g0")
        b_ds_r += self._kv_frag_reads("g2")
        b_ds_r += self._acc_rmw()
        b_ds_w += self._acc_rmw()
        c_ds_r += self._acc_rmw()
        c_ds_w += self._acc_rmw()
        d_ds_r = max(0, n_slices - 1) * slice_reads
        if g4_on and self.kt_source == "lds":
            kt_frags = (d // (16 * g4c)) * (n0 // k["g4"])
            d_ds_r += kt_frags * self._reads_per_frag("g4", t)
        if prefetch:
            d_ds_r += next_a_reads + stats_reads
        return (
            StageCounts(
                "A",
                mf.g0,
                vmem_read=a_vmem,
                vmem_write=a_vmem_w,
                ds_read=a_ds_r,
                ds_write=a_ds_w,
            ),
            StageCounts("B", mf.g1 + mf.g2, ds_read=b_ds_r, ds_write=b_ds_w),
            StageCounts("C", mf.g3, ds_read=c_ds_r, ds_write=c_ds_w),
            StageCounts("D", mf.g4, ds_read=d_ds_r),
        )

    def stage(self, name: str, **kw: bool) -> StageCounts:
        return self.stages(**kw)[STAGES.index(name)]

    def groups(
        self, name: str, *, prefetch: bool = True, prev_atomics: bool = True
    ) -> list[tuple[int, int]]:
        """``(mask, count)`` sched-group sequence of one stage (empty if none)."""
        if self.sched not in STAGE_TABLE_SCHEDS:
            return []
        sc = self.stage(name, prefetch=prefetch, prev_atomics=prev_atomics)
        return interleave_groups(sc.mfma, sc.memory())

    # ----- emission -------------------------------------------------------

    def emit_loop_head(self, b: Any) -> None:
        """Once at the top of the q-loop body: ``iglp_opt`` for the iglp modes."""
        if self.sched == "iglp0":
            b.iglp_opt(0)
        elif self.sched == "iglp1":
            b.iglp_opt(1)

    def emit_stage_begin(self, b: Any, name: str) -> None:
        """Before a stage's ops: raise the wave priority when ``setprio``."""
        if self.setprio:
            b.s_setprio(1)

    def emit_stage_end(
        self, b: Any, name: str, *, prefetch: bool = True, prev_atomics: bool = True
    ) -> None:
        """After a stage's ops: the interleave groups, priority reset, fence."""
        if name not in STAGES:
            raise ValueError(f"unknown stage {name!r}")
        for mask, count in self.groups(
            name, prefetch=prefetch, prev_atomics=prev_atomics
        ):
            b.sched_group_barrier(mask, count, 0)
        if self.setprio:
            b.s_setprio(0)
        if name != "D" and self.sched in STAGE_TABLE_SCHEDS + ("fences_only",):
            b.sched_barrier(0)

    # ----- IR count model -------------------------------------------------

    def ir_count_model(
        self, *, prefetch: bool = True, prev_atomics: bool = True
    ) -> dict[str, Any]:
        """IR op counts one q step's emission produces, plus the MMA model.

        Keys: ``tile.sched_group_barrier``, ``tile.sched_barrier``,
        ``tile.s_setprio``, ``tile.iglp_opt`` (IR op counts of
        :meth:`emit_loop_head` + the four stage begin/end calls),
        ``group_counts`` (``{mask: summed count}`` over the groups),
        ``mfma`` (per GEMM), ``mfma_total``, ``stage_mfma`` (per stage).
        """
        groups = 0
        sums: dict[int, int] = {}
        for name in STAGES:
            for mask, count in self.groups(
                name, prefetch=prefetch, prev_atomics=prev_atomics
            ):
                groups += 1
                sums[mask] = sums.get(mask, 0) + count
        fences = 3 if self.sched in STAGE_TABLE_SCHEDS + ("fences_only",) else 0
        st = self.stages(prefetch=prefetch, prev_atomics=prev_atomics)
        return {
            "tile.sched_group_barrier": groups,
            "tile.sched_barrier": fences,
            "tile.s_setprio": 2 * len(STAGES) if self.setprio else 0,
            "tile.iglp_opt": 1 if self.sched in ("iglp0", "iglp1") else 0,
            "group_counts": sums,
            "mfma": self.mfma.as_dict(),
            "mfma_total": self.mfma.total,
            "stage_mfma": {s.stage: s.mfma for s in st},
        }

    def describe(self) -> dict[str, Any]:
        """Plain-data summary (for sweep records and the ``--count`` output)."""
        return {
            "arch": self.arch,
            "sched": self.sched,
            "setprio": self.setprio,
            "kv_residency": self.kv_residency,
            "acc_in_lds": self.acc_in_lds,
            "instr_k": self.instr_k,
            "mfma": self.mfma.as_dict(),
            "dq_atomics_per_lane": self.dq_atomics_per_lane,
            "stages": {
                s.stage: {
                    "mfma": s.mfma,
                    "vmem_read": s.vmem_read,
                    "vmem_write": s.vmem_write,
                    "ds_read": s.ds_read,
                    "ds_write": s.ds_write,
                }
                for s in self.stages()
            },
        }
