# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Probe kernels for the backward block GEMM and the C-to-A relabel.

Shared by the CPU IR tests and the device numeric tests. Two one-CTA kernels:

* ``gemm``: ``C[M x N] = A[M x K] * B[K x N]`` with operands staged to LDS in a
  chosen storage form and fed to the block GEMM either from LDS (per K step)
  or from registers (all fragments loaded first);
* ``chain``: ``S = Q K^T`` on warp grid ``(1, W)``, then ``dV = P^T dO`` on
  ``(W, 1)`` with ``P^T`` produced by the register relabel of ``S``
  (``P = round(S)`` to the operand dtype) and ``dO`` loaded with the plan's K
  permutation when it needs one.

Host buffers are given in the LDS storage form, so staging is a plain copy.
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np
from rocke.core.ir import F32, IRBuilder, PtrType

from kernels.common._attention_bwd_frag import ir_type
from kernels.common._attention_bwd_gemm import (
    BlockGemm,
    LdsOperand,
    kperm_b_operand_coords,
    lane_and_wave,
    relabel_acc_to_a,
)

# ---------------------------------------------------------------------------
# kernel construction
# ---------------------------------------------------------------------------


def _stage(b, src, smem, shape, tid, threads, elem):
    """Copy a row-major ``shape`` tile from global ``src`` into ``smem``."""
    rows, cols = shape
    total, vec = rows * cols, 8
    if cols % vec:
        raise ValueError("staged tiles need a multiple-of-8 inner dimension")
    step = threads * vec
    for it in range((total + step - 1) // step):
        idx = b.add(b.mul(tid, b.const_i32(vec)), b.const_i32(it * step))

        def body(idx=idx):
            v = b.global_load_vN(src, idx, elem, vec)
            r = b.div(idx, b.const_i32(cols))
            c = b.mod(idx, b.const_i32(cols))
            b.smem_store_vN(smem, [r, c], v, vec)

        if (it + 1) * step <= total:
            body()
        else:
            with b.scf_if(b.cmp_lt(idx, b.const_i32(total))):
                body()


def _store_acc(b, g, acc, lane, origin, out, ld):
    for fm in range(g.frags_m):
        for fn in range(g.frags_n):
            for s, (r, c) in enumerate(g.acc_coords(b, lane, origin, fm, fn)):
                idx = b.add(b.mul(r, b.const_i32(ld)), c)
                b.global_store(out, idx, b.vec_extract(acc[fm][fn], s), align=4)


def _storage_shape(storage, mn, k):
    return (mn, k) if storage == "k_inner" else (k, mn)


def gemm_probe_name(arch, dtype, m, n, k, grid, atom_k, a_st, b_st, a_src, b_src):
    return (
        f"bwd_gemm_probe_{arch}_{dtype}_{m}x{n}x{k}_g{grid[0]}x{grid[1]}_k{atom_k}"
        f"_{a_st[2]}{b_st[2]}_{a_src[0]}{b_src[0]}"
    )


def make_gemm_probe(
    arch,
    dtype,
    *,
    m,
    n,
    k,
    grid,
    atom_k=16,
    a_storage="k_inner",
    b_storage="k_inner",
    a_src="lds",
    b_src="lds",
):
    """Return ``(kernel, geometry)`` of the plain block-GEMM probe."""
    g = BlockGemm(arch, dtype, m, n, k, tuple(grid), atom_k)
    elem = ir_type(dtype)
    name = gemm_probe_name(
        arch, dtype, m, n, k, grid, atom_k, a_storage, b_storage, a_src, b_src
    )
    b = IRBuilder(name)
    pa = b.param("A", PtrType(elem, "global"), readonly=True)
    pb = b.param("B", PtrType(elem, "global"), readonly=True)
    pc = b.param("C", PtrType(F32, "global"))
    b.kernel.attrs["max_workgroup_size"] = max(64, g.threads)
    tid, lane, wave = lane_and_wave(b, g.wave_size)
    sha, shb = _storage_shape(a_storage, m, k), _storage_shape(b_storage, n, k)
    sa = b.smem_alloc(elem, sha, "sa")
    sb = b.smem_alloc(elem, shb, "sb")
    _stage(b, pa, sa, sha, tid, g.threads, elem)
    _stage(b, pb, sb, shb, tid, g.threads, elem)
    b.sync()
    origin = g.wave_origin(b, wave)
    a_op = LdsOperand(sa, a_storage)
    b_op = LdsOperand(sb, b_storage)
    if a_src == "reg":
        a_op = g.load_all_a(b, a_op, lane, origin[0])
    if b_src == "reg":
        b_op = g.load_all_b(b, b_op, lane, origin[1])
    acc = g.mma(b, g.zero_acc(b), a_op, b_op, lane=lane, origin=origin)
    _store_acc(b, g, acc, lane, origin, pc, n)
    b.ret()
    return b.kernel, g


def make_chain_probe(
    arch, dtype, *, k_m0, k_n0, d, waves, atom_k0=16, atom_k1=16, do_storage="k_outer"
):
    """Return ``(kernel, g0, g1, plan)`` of the relabel chain probe."""
    g0 = BlockGemm(arch, dtype, k_m0, k_n0, d, (1, waves), atom_k0)
    g1 = BlockGemm(arch, dtype, k_n0, d, k_m0, (waves, 1), atom_k1)
    elem = ir_type(dtype)
    name = (
        f"bwd_chain_probe_{arch}_{dtype}_m{k_m0}_n{k_n0}_d{d}_w{waves}"
        f"_k{atom_k0}k{atom_k1}_{do_storage[2]}"
    )
    b = IRBuilder(name)
    pq = b.param("Q", PtrType(elem, "global"), readonly=True)
    pk = b.param("K", PtrType(elem, "global"), readonly=True)
    pdo = b.param("dO", PtrType(elem, "global"), readonly=True)
    ps = b.param("S", PtrType(F32, "global"))
    pdv = b.param("dV", PtrType(F32, "global"))
    b.kernel.attrs["max_workgroup_size"] = max(64, g0.threads)
    tid, lane, wave = lane_and_wave(b, g0.wave_size)
    sq = b.smem_alloc(elem, (k_m0, d), "sq")
    sk = b.smem_alloc(elem, (k_n0, d), "sk")
    sdo_shape = _storage_shape(do_storage, d, k_m0)
    sdo = b.smem_alloc(elem, sdo_shape, "sdo")
    _stage(b, pq, sq, (k_m0, d), tid, g0.threads, elem)
    _stage(b, pk, sk, (k_n0, d), tid, g0.threads, elem)
    _stage(b, pdo, sdo, sdo_shape, tid, g0.threads, elem)
    b.sync()
    o0 = g0.wave_origin(b, wave)
    acc0 = g0.mma(
        b, g0.zero_acc(b), LdsOperand(sq), LdsOperand(sk), lane=lane, origin=o0
    )
    _store_acc(b, g0, acc0, lane, o0, ps, k_n0)
    a1, plan = relabel_acc_to_a(b, g0, g1, acc0, lane)
    coords = kperm_b_operand_coords(plan, g0, g1)
    o1 = g1.wave_origin(b, wave)
    acc1 = g1.mma(
        b,
        g1.zero_acc(b),
        a1,
        LdsOperand(sdo, do_storage, coords=coords),
        lane=lane,
        origin=o1,
    )
    _store_acc(b, g1, acc1, lane, o1, pdv, d)
    b.ret()
    return b.kernel, g0, g1, plan


# ---------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------


def enc(x, dtype):
    """float32 -> raw fp16 / bf16 storage (round to nearest even)."""
    x = np.asarray(x, np.float32)
    if dtype == "fp16":
        return x.astype(np.float16)
    u = x.view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def dec(u, dtype):
    if dtype == "fp16":
        return np.asarray(u).astype(np.float32)
    return (np.asarray(u).astype(np.uint32) << 16).view(np.float32)


def _rand(rng, shape, dtype):
    raw = enc(rng.standard_normal(shape).astype(np.float32) * 0.5, dtype)
    return raw, dec(raw, dtype).astype(np.float64)


def _launch(arch, kernel, threads, inputs, outputs):
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    art = compile_kernel(kernel, arch=arch, capture_ir_text=False, backend="python")
    rt = Runtime()
    mod = rt.load_module(art.hsaco)
    fn = mod.get_function(art.kernel_name)

    def u8(a):
        return (ctypes.c_uint8 * int(a.nbytes)).from_buffer(a)

    host = [np.ascontiguousarray(a) for a in inputs] + [
        np.zeros(n, np.float32) for n in outputs
    ]
    dev = []
    for a in host:
        p = rt.alloc(max(a.nbytes, 16))
        rt.memcpy_h2d(p, u8(a), a.nbytes)
        dev.append(p)
    packed = b"".join(struct.pack("<Q", p) for p in dev)
    rt.launch(fn, (1, 1, 1), (threads, 1, 1), packed)
    rt.sync()
    outs = host[len(inputs) :]
    for a, p in zip(outs, dev[len(inputs) :]):
        rt.memcpy_d2h(u8(a), p, a.nbytes)
    for p in dev:
        rt.free(p)
    mod.unload()
    return outs


def run_gemm_probe(arch, dtype, *, seed=0, **kw):
    """Launch the GEMM probe; returns ``(got, ref)`` as float64 ``[M, N]``."""
    kernel, g = make_gemm_probe(arch, dtype, **kw)
    rng = np.random.default_rng(seed)
    a_raw, a = _rand(rng, (g.m, g.k), dtype)
    b_raw, bm = _rand(rng, (g.k, g.n), dtype)
    a_store = a_raw if kw.get("a_storage", "k_inner") == "k_inner" else a_raw.T
    b_store = b_raw.T if kw.get("b_storage", "k_inner") == "k_inner" else b_raw
    (c,) = _launch(arch, kernel, g.threads, [a_store, b_store], [g.m * g.n])
    return c.reshape(g.m, g.n).astype(np.float64), a @ bm


def run_chain_probe(arch, dtype, *, seed=0, **kw):
    """Launch the chain probe.

    Returns ``(s_got, s_ref, dv_got, dv_ref)``; ``dv_ref`` uses the device ``S``
    rounded to the operand dtype, so it isolates the relabel and the K
    permutation from the rounding of ``S``.
    """
    kernel, g0, g1, _ = make_chain_probe(arch, dtype, **kw)
    rng = np.random.default_rng(seed)
    q_raw, q = _rand(rng, (g0.m, g0.k), dtype)
    k_raw, kk = _rand(rng, (g0.n, g0.k), dtype)
    do_raw, do = _rand(rng, (g1.k, g1.n), dtype)
    do_store = do_raw if kw.get("do_storage", "k_outer") == "k_outer" else do_raw.T
    s, dv = _launch(
        arch,
        kernel,
        g0.threads,
        [q_raw, k_raw, do_store],
        [g0.m * g0.n, g1.m * g1.n],
    )
    s = s.reshape(g0.m, g0.n).astype(np.float64)
    p = dec(enc(s.astype(np.float32), dtype), dtype).astype(np.float64)
    return s, q @ kk.T, dv.reshape(g1.m, g1.n).astype(np.float64), p.T @ do
