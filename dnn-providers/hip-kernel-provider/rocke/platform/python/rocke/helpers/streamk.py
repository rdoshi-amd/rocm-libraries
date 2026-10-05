# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""StreamK tile partitioner helpers (CK Tile parity).

CK Tile's :class:`StreamKTilePartitioner` decomposes a GEMM workload
into ``(m_tile, n_tile, k_iter)`` triples and decides how each
partial-K accumulation feeds the output. Two reduction strategies are
exposed:

* ``ReductionStrategy.Atomic`` -- each CTA atomically adds its
 partial K-sum into a shared f32 workspace at the output's
 ``(m_tile, n_tile)`` position. Simpler; relies on
 ``global_atomic_add_f32`` (gfx940+). Requires a finalisation pass
 to convert f32 -> output dtype.

* ``ReductionStrategy.Reduction`` -- the CTAs that contribute to the
 same ``(m_tile, n_tile)`` cooperate through a tile-major workspace
 + a flag table; the *last* contributor performs the reduction +
 finalisation in-kernel. More complex but avoids the second launch
 and is the path CK Tile defaults to.

This module ships the partitioner math (a pure-Python view) and the
IR-side glue that
``instances/streamk_gemm.py`` uses. The kernel-shape primitives stay
in :mod:`rocke.helpers.atoms` / :mod:`rocke.helpers.loads` /
:mod:`rocke.helpers.epilogues`; this file is purely about
*partitioning* the work, not about running the per-tile GEMM.

What we ship today:

* :class:`StreamKReductionStrategy` -- enum of supported strategies.
* :class:`StreamKPartition` -- the decoded ``(m_tile, n_tile, k_iter,
 is_first, is_last)`` for a given linear partition id.
* :func:`compute_streamk_grid_size` -- worst-case CTA count for a
 spec.
* :func:`emit_streamk_decode` -- IR-side decode of a linear partition
 id into the SSA ``(m_tile, n_tile, k_iter, is_first, is_last)``
 bundle the kernel needs.

v1 implements the ``Atomic`` strategy end-to-end and the
``Reduction`` strategy's *decode* surface (so the spec layer can
emit the right kernel shape); the per-tile reduction pass that the
``Reduction`` strategy needs lands with the StreamK GEMM kernel
itself in .
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from ..core.ir import IRBuilder, Value


__all__ = [
    "StreamKIterPartition",
    "StreamKPartition",
    "StreamKReductionStrategy",
    "compute_streamk_grid_size",
    "emit_streamk_decode",
    "emit_streamk_iter_range",
    "emit_streamk_partial_load_accumulate",
    "emit_streamk_partial_store",
    "emit_streamk_sk_start_iter",
    "streamk_end_iter",
    "streamk_iter_partition",
    "streamk_num_macro_tiles",
    "streamk_start_iter",
]


class StreamKReductionStrategy(Enum):
    """Per CK Tile naming."""

    Atomic = "atomic"  # global_atomic_add_f32 into workspace
    Reduction = "reduction"  # cooperative + flag-table reduction


@dataclass(frozen=True)
class StreamKPartition:
    """The compile-time-fixed shape inputs to the StreamK partitioner.

    The actual ``(m_tile, n_tile, k_iter, is_first, is_last)``
    decoding happens at runtime via :func:`emit_streamk_decode`;
    this dataclass just collects the compile-time constants the
    decode helper needs.
    """

    m_tiles: int  # ceil(M / tile_m); the number of M tiles
    n_tiles: int  # ceil(N / tile_n)
    k_iters: int  # K / tile_k

    @property
    def num_macro_tiles(self) -> int:
        """``m_tiles * n_tiles * k_iters`` -- the total chunk count."""
        return self.m_tiles * self.n_tiles * self.k_iters

    @property
    def k_iters_per_output_tile(self) -> int:
        """``k_iters`` -- the number of CTAs that touch each
        ``(m, n)`` output tile in the Atomic strategy.
        """
        return self.k_iters


class _DecodedTile(NamedTuple):
    """The SSA bundle returned by :func:`emit_streamk_decode`."""

    m_tile: Value  # i32
    n_tile: Value  # i32
    k_iter: Value  # i32 in [0, k_iters)
    is_first: Value  # i1: k_iter == 0
    is_last: Value  # i1: k_iter == k_iters - 1


def streamk_num_macro_tiles(spec: StreamKPartition) -> int:
    """Plain Python view of :attr:`StreamKPartition.num_macro_tiles`."""
    return spec.num_macro_tiles


def compute_streamk_grid_size(
    spec: StreamKPartition,
    *,
    num_cus: int = 304,
    blocks_per_cu: int = 1,
) -> int:
    """Recommended persistent-launch grid size for ``spec``.

    Cap = ``min(num_macro_tiles, num_cus * blocks_per_cu)``. The
    persistent kernel re-fetches macro tiles via atomic_add until
    the global counter exhausts; the launch grid stays small and
    constant so launch overhead is one-shot.

    Default ``num_cus = 304`` matches MI300X (the canonical CK Tile
    deployment target). Override for MI355X (304) or other parts.
    """
    if spec.num_macro_tiles <= 0:
        raise ValueError("spec has zero macro tiles")
    return min(spec.num_macro_tiles, num_cus * blocks_per_cu)


def emit_streamk_decode(
    b: IRBuilder, linear_id: Value, spec: StreamKPartition
) -> _DecodedTile:
    """Decode a linear macro-tile id into ``(m_tile, n_tile, k_iter,
    is_first, is_last)``.

    Layout (matches CK Tile's "K-major within a (m, n) tile" walk):

    .. code-block:: text

    k_iter = linear_id % k_iters
    nn = linear_id // k_iters
    n_tile = nn % n_tiles
    m_tile = nn // n_tiles

    The ``is_first`` / ``is_last`` predicates let the kernel decide
    whether it owns the *seed* of the output (must clear) or the
    *finalize* (must write the converted output dtype to ``C``); the
    middle iterations atomic-add their partial into the workspace.
    """
    c_k_iters = b.const_i32(spec.k_iters)
    c_n_tiles = b.const_i32(spec.n_tiles)

    k_iter = b.mod(linear_id, c_k_iters)
    nn = b.div(linear_id, c_k_iters)
    n_tile = b.mod(nn, c_n_tiles)
    m_tile = b.div(nn, c_n_tiles)

    is_first = b.cmp_eq(k_iter, b.const_i32(0))
    is_last = b.cmp_eq(k_iter, b.const_i32(spec.k_iters - 1))
    return _DecodedTile(
        m_tile=m_tile,
        n_tile=n_tile,
        k_iter=k_iter,
        is_first=is_first,
        is_last=is_last,
    )


def emit_streamk_partial_store(
    b: IRBuilder,
    *,
    workspace: Value,
    flag_table: Value,
    workspace_off: Value,
    flag_off: Value,
    value: Value,
    is_first: Value,
) -> None:
    """Cooperative ``Reduction`` strategy partial-store + signal.

    P37: each contributing CTA stores its partial K-sum at the
    ``(m_tile, n_tile, k_iter)`` slot in the workspace and bumps the
    flag-table counter for ``(m_tile, n_tile)`` via ``atomic_add(+1)``.
    The first contributor's store is unconditionally a fresh write
    (``is_first=True``); later contributors atomic-add into the
    workspace too so the order doesn't matter.

    Reference: CK Tile ``streamk_common.hpp::SignalStorePartialDone``
    + ``streamk_gemm_kernel.hpp:448-504`` (Tree-reduction).
    """
    # Initial store: lane-0-only write (the workspace is per-lane sized
    # already in the caller's view) — the consumer in
    # :func:`emit_streamk_partial_load_accumulate` reads each slot once.
    with b.scf_if(is_first):
        b.global_store(workspace, workspace_off, value, align=4)
    # Non-first contributors: atomic add into the same f32 slot (the
    # ``is_last`` reducer reads the converged value).
    with b.scf_if(b.lnot(is_first)):
        b.global_atomic_add(workspace, workspace_off, value)
    # Bump the flag counter so the last-finishing CTA can detect "I'm
    # the reducer" via flag == k_iters_per_tile.
    b.global_atomic_add(flag_table, flag_off, b.const_i32(1))


def emit_streamk_partial_load_accumulate(
    b: IRBuilder,
    *,
    workspace: Value,
    workspace_off: Value,
    is_last: Value,
) -> Value:
    """Read the converged f32 partial sum at ``workspace_off`` once
    every contributor has signalled (``is_last == True`` is the
    caller's responsibility).

    The flag-table-based wait protocol is a busy-loop on
    ``atomic_load(flag_table[tile_id]) == k_iters_per_tile``; we expose
    it as a separate helper because the busy-loop shape differs
    between Linear / Tree reduction. For now the Atomic strategy is
    the canonical path; this helper plus
    :func:`emit_streamk_partial_store` give callers the surface
    needed to opt into the Reduction strategy.

    Reference: CK Tile ``streamk_common.hpp:34-73``
    (``WaitStorePartialDone``).
    """
    return b.global_load_f32(workspace, workspace_off)


# ---------------------------------------------------------------------------
# Iteration-balanced stream-K (CK Tile StreamKTilePartitioner)
# ---------------------------------------------------------------------------
#
# The partition above hands every CTA exactly one ``tile_k`` slice. CK Tile's
# production partitioner instead balances *iterations*: output tiles that
# divide evenly across the CTA pool run data-parallel (one CTA per tile, the
# full K loop), and only the remainder is spread over ``max_active_wgs``
# stream-K CTAs, each owning a contiguous range of MAC-loop iterations that
# may straddle output-tile boundaries. rocKE compiles one kernel per shape, so
# the whole partition is resolved host-side and folded into the IR as
# constants; the device only maps its linear CTA id onto an iteration range.


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class StreamKIterPartition:
    """CK Tile ``StreamKTilePartitionerBase``, resolved at build time.

    ``m_tiles`` / ``n_tiles`` count output tiles (a caller folding a batch
    dimension into M passes the folded count). ``iters_per_tile`` is the
    number of ``tile_k`` MAC iterations per output tile. ``max_active_wgs``
    is the CTA pool the stream-K remainder is spread over; every stream-K
    CTA must be co-resident with the CTAs it waits on, so it must not exceed
    the number of workgroups the device can hold at once.
    """

    m_tiles: int
    n_tiles: int
    iters_per_tile: int
    max_active_wgs: int
    persistent: bool = False

    @property
    def num_tiles(self) -> int:
        return self.m_tiles * self.n_tiles

    @property
    def sk_tiles(self) -> int:
        """Output tiles handled by stream-K CTAs (0 means all-DP).

        CK's "DP + 2-tile SK": when the tiles do not divide the CTA pool,
        one full wave of tiles plus the remainder go stream-K so the last
        wave is balanced; a remainder too small to give every stream-K CTA
        at least one iteration falls back to all-DP.
        """
        rem = self.num_tiles % self.max_active_wgs
        if rem == 0:
            return 0
        if self.num_tiles > self.max_active_wgs:
            tiles = self.max_active_wgs + rem
        else:
            tiles = self.num_tiles
        if tiles * self.iters_per_tile < self.max_active_wgs:
            return 0
        return tiles

    @property
    def sk_ctas(self) -> int:
        return self.max_active_wgs if self.sk_tiles else 0

    @property
    def total_sk_iters(self) -> int:
        return self.sk_tiles * self.iters_per_tile

    @property
    def iters_per_sk_cta(self) -> int:
        return self.total_sk_iters // self.sk_ctas if self.sk_ctas else 0

    @property
    def extra_iters(self) -> int:
        """Stream-K CTAs ``[0, extra_iters)`` own one extra iteration."""
        return self.total_sk_iters % self.sk_ctas if self.sk_ctas else 0

    @property
    def dp_tiles(self) -> int:
        return self.num_tiles - self.sk_tiles

    @property
    def total_dp_iters(self) -> int:
        return self.dp_tiles * self.iters_per_tile

    @property
    def grid_size(self) -> int:
        """Launch grid (x): the CTA pool when persistent, else DP + SK CTAs."""
        if self.persistent:
            return self.max_active_wgs
        return self.dp_tiles + self.sk_ctas

    @property
    def max_linear_partners(self) -> int:
        """Upper bound on the CTAs a tile owner has to fold in (Linear).

        The owner is whichever CTA covers the tile's first iteration; it may
        have started in the previous tile, so it is only guaranteed one
        iteration here. Every later contributor covers at least
        ``iters_per_sk_cta`` iterations until the tile ends.
        """
        if not self.sk_ctas:
            return 0
        return _ceil_div(self.iters_per_tile - 1, self.iters_per_sk_cta)

    @property
    def tree_rounds(self) -> int:
        """Pairwise fan-in rounds bounding the Tree fixup: ``ceil(log2)`` of
        the most CTAs that can contribute to one tile."""
        contributors = self.max_linear_partners + 1
        return (contributors - 1).bit_length()

    @property
    def flags_bytes(self) -> int:
        """One i32 flag per stream-K CTA, padded to 256 bytes (CK layout)."""
        return _ceil_div(4 * self.sk_ctas, 256) * 256


def streamk_iter_partition(
    *,
    m_tiles: int,
    n_tiles: int,
    iters_per_tile: int,
    max_active_wgs: int,
    persistent: bool = False,
) -> StreamKIterPartition:
    """Validated constructor for :class:`StreamKIterPartition`."""
    for name, v in (
        ("m_tiles", m_tiles),
        ("n_tiles", n_tiles),
        ("iters_per_tile", iters_per_tile),
        ("max_active_wgs", max_active_wgs),
    ):
        if v <= 0:
            raise ValueError(f"stream-K partition: {name} must be > 0 (got {v})")
    return StreamKIterPartition(
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        iters_per_tile=iters_per_tile,
        max_active_wgs=max_active_wgs,
        persistent=persistent,
    )


def streamk_start_iter(part: StreamKIterPartition, sk_cta: int) -> int:
    """Global first MAC iteration of stream-K CTA ``sk_cta``."""
    return (
        part.total_dp_iters
        + sk_cta * part.iters_per_sk_cta
        + min(sk_cta, part.extra_iters)
    )


def streamk_end_iter(part: StreamKIterPartition, sk_cta: int) -> int:
    """One past the last MAC iteration of stream-K CTA ``sk_cta``."""
    return (
        streamk_start_iter(part, sk_cta)
        + part.iters_per_sk_cta
        + (1 if sk_cta < part.extra_iters else 0)
    )


def emit_streamk_sk_start_iter(
    b: IRBuilder, sk_cta: Value, part: StreamKIterPartition
) -> Value:
    """SSA :func:`streamk_start_iter` for a runtime stream-K CTA index."""
    c_dp_iters = b.const_i32(part.total_dp_iters)
    c_per_cta = b.const_i32(part.iters_per_sk_cta)
    c_extra = b.const_i32(part.extra_iters)
    body = b.mul(sk_cta, c_per_cta)
    lead = b.smin(sk_cta, c_extra)
    return b.add(c_dp_iters, b.add(body, lead))


class _IterRange(NamedTuple):
    """The SSA bundle returned by :func:`emit_streamk_iter_range`."""

    start: Value  # i32 global first iteration (SGPR)
    end: Value  # i32 one past the last iteration (SGPR)
    sk_cta: Value  # i32 stream-K CTA index; meaningless for a DP CTA


def emit_streamk_iter_range(
    b: IRBuilder, cta: Value, part: StreamKIterPartition, *, dp: bool
) -> _IterRange:
    """Map a CTA onto its contiguous range of global MAC iterations.

    ``dp=True`` (non-persistent launch): CTAs ``[0, dp_tiles)`` each own one
    whole tile and the rest are stream-K CTAs, selected branch-free so the
    GEMM body is emitted once. ``dp=False``: ``cta`` already is a stream-K
    CTA index (the persistent launch runs its DP sweep separately).
    """
    if dp:
        c_dp_tiles = b.const_i32(part.dp_tiles)
        sk_cta = b.sub(cta, c_dp_tiles)
    else:
        sk_cta = cta
    sk_start = emit_streamk_sk_start_iter(b, sk_cta, part)
    c_per_cta = b.const_i32(part.iters_per_sk_cta)
    c_extra = b.const_i32(part.extra_iters)
    c_one = b.const_i32(1)
    c_zero = b.const_i32(0)
    sk_len = b.add(c_per_cta, b.select(b.cmp_lt(sk_cta, c_extra), c_one, c_zero))
    if dp:
        c_ipt = b.const_i32(part.iters_per_tile)
        is_dp = b.cmp_lt(cta, c_dp_tiles)
        start = b.select(is_dp, b.mul(cta, c_ipt), sk_start)
        length = b.select(is_dp, c_ipt, sk_len)
    else:
        start = sk_start
        length = sk_len
    start = b.to_sgpr_u32(start)
    end = b.to_sgpr_u32(b.add(start, length))
    return _IterRange(start=start, end=end, sk_cta=sk_cta)
