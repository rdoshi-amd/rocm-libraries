# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CTA work decode of the KV-parallel attention backward main kernels.

Family glue for the backward kernels (Python-only, no C++ twin). Everything
here emits CTA-uniform integer IR through the builder it is handed and only
calls integer methods of ``IRBuilder``, so the duck-typed host evaluator of the
tests can execute it.

Grid ``(gx, units * g_split, B)`` of the main kernel; one CTA decodes, in this
fixed order:

1. the linear id ``L = x + gx * (y + gy * z)``; the XCD remap is the identity
   here (``xcd_n = 0``), so the decode below reads ``(x, y, z)`` directly;
2. the kv-tile index ``t = x``, the head unit and split of ``y`` (``unit = y /
   g_split``, ``split = y % g_split``; ``unit = y`` when the kernel has no
   split) and the batch ``z``;
3. the load-balance order on ``t`` (runtime ``lb_order`` / ``pair``):

   * natural (0): one kv tile ``kt = t``;
   * reverse (1): one kv tile ``kt = n_kv_tiles - 1 - t``;
   * mirror pairing (``pair = 1``, ``gx = ceil(n_kv_tiles / 2)``): the tiles
     ``t`` and ``n_kv_tiles - 1 - t``, once when they are equal. ``pair = 1``
     takes precedence over ``lb_order``.

   Every order visits each kv tile of a (unit, batch) exactly once. The main
   kernel decodes with ``pairing=False`` (one kv tile per CTA: natural or
   reverse); the two-tile CTA of mirror pairing is not built (a CTA-uniform
   loop over two kv tiles keeps the staging and epilogue values live across
   the q loops and spills registers), so the host plan refuses it.

:func:`head_range` maps a (unit, split) to its query heads and kv heads:
direct mode (``h_k == h_v``, no split) ``h0 = unit * G``, ``G`` heads,
``hk = hv = unit``; atomic mode (``h_k != h_v`` or ``g_split > 1``, ``G =
gcd(h_q / h_k, h_q / h_v)``) ``G / g_split`` heads from ``unit * G + split *
G / g_split``, ``hk = unit * G / gk`` and ``hv = unit * G / gv`` (every query
head of a unit shares one K head and one V head, because ``G`` divides ``gk``
and ``gv``).
"""

from __future__ import annotations

from dataclasses import dataclass

from rocke.core.ir import IRBuilder, Value

__all__ = [
    "LB_MIRROR",
    "LB_NATURAL",
    "LB_REVERSE",
    "CtaWork",
    "HeadRange",
    "decode_cta",
    "head_range",
    "main_grid_x",
]

LB_NATURAL = 0
LB_REVERSE = 1
LB_MIRROR = 2


def main_grid_x(n_kv_tiles: int, pair: int) -> int:
    """Grid x of the main kernel: ``ceil(n / 2)`` under mirror pairing, else ``n``."""
    if n_kv_tiles < 0:
        raise ValueError("n_kv_tiles must be >= 0")
    return -(-n_kv_tiles // 2) if pair else n_kv_tiles


@dataclass(frozen=True, eq=False)
class CtaWork:
    """Decoded work of one CTA (every value CTA-uniform, i32).

    ``first_kt`` / ``second_kt`` are the kv tiles of pass 0 / pass 1;
    ``n_pass`` is 1, or 2 under mirror pairing when the two tiles differ.
    """

    t: Value
    unit: Value
    split: Value
    batch: Value
    first_kt: Value
    second_kt: Value
    n_pass: Value

    def kv_tile(self, b: IRBuilder, p: Value) -> Value:
        """kv tile of pass ``p`` (0 or 1)."""
        return b.select(b.cmp_eq(p, b.const_i32(0)), self.first_kt, self.second_kt)


def decode_cta(
    b: IRBuilder,
    *,
    x: Value,
    y: Value,
    z: Value,
    n_kv_tiles: Value,
    lb_order: Value,
    pair: Value | None = None,
    g_split: Value | None = None,
) -> CtaWork:
    """CTA decode of the main grid (see the module docstring).

    ``pair = None``: no mirror pairing (one kv tile per CTA; ``second_kt`` is
    the first and ``n_pass`` is 1). ``g_split = None``: the kernel has no head
    split (``unit = y``, split 0).
    """
    zero, one = b.const_i32(0), b.const_i32(1)
    if g_split is None:
        unit, split = y, zero
    else:
        unit, split = b.div(y, g_split), b.mod(y, g_split)
    mirror_t = b.sub(b.sub(n_kv_tiles, one), x)
    reverse = b.cmp_eq(lb_order, b.const_i32(LB_REVERSE))
    if pair is None:
        first = b.select(reverse, mirror_t, x)
        return CtaWork(
            t=x, unit=unit, split=split, batch=z, first_kt=first, second_kt=first,
            n_pass=one,
        )  # fmt: skip
    paired = b.cmp_ne(pair, zero)
    first = b.select(b.land(reverse, b.lnot(paired)), mirror_t, x)
    two = b.land(paired, b.cmp_ne(mirror_t, first))
    n_pass = b.select(two, b.const_i32(2), one)
    return CtaWork(
        t=x,
        unit=unit,
        split=split,
        batch=z,
        first_kt=first,
        second_kt=mirror_t,
        n_pass=n_pass,
    )


@dataclass(frozen=True, eq=False)
class HeadRange:
    """Query heads ``[h0, h0 + n_heads)`` of a CTA and its K / V heads."""

    h0: Value
    n_heads: Value
    hk: Value
    hv: Value


def head_range(
    b: IRBuilder,
    *,
    unit: Value,
    split: Value,
    group: Value,
    gk: Value,
    gv: Value,
    g_split: Value | None = None,
) -> HeadRange:
    """Query and kv heads of a (unit, split) (see the module docstring).

    ``g_split = None`` is direct mode: the unit is the kv head and owns its
    whole group (``group = h_q / h_k``). Otherwise atomic mode: ``group`` is
    ``gcd(h_q / h_k, h_q / h_v)`` and ``g_split`` divides it.
    """
    base = b.mul(unit, group)
    if g_split is None:
        return HeadRange(h0=base, n_heads=group, hk=unit, hv=unit)
    per = b.div(group, g_split)
    return HeadRange(
        h0=b.add(base, b.mul(split, per)),
        n_heads=per,
        hk=b.div(base, gk),
        hv=b.div(base, gv),
    )
