# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real-GPU page/tile loader numerics with CPU references after quantization."""

import math

import pytest

from tests import test_strided_kv_decode_numeric as decode_numeric

gpu = decode_numeric.gpu
_reference = decode_numeric._reference


def _shuffled_cache(torch, logical, page, *, poison_tail=True):
    batch, heads, capacity, dim = logical.shape
    count = (capacity + page - 1) // page
    # Scatter logical pages into nonadjacent physical pages. Unreferenced pages
    # and final-page padding are NaN so an incorrect mapping cannot pass quietly.
    ids = torch.randperm(
        batch * count * 2, generator=torch.Generator().manual_seed(51)
    )[: batch * count].reshape(batch, count)
    pool = torch.full(
        (batch * count * 2, page, heads, dim), float("nan"), dtype=logical.dtype
    )
    for b in range(batch):
        for i in range(count):
            if not poison_tail:
                pool[ids[b, i]].zero_()
            n = min(page, capacity - i * page)
            pool[ids[b, i], :n] = logical[b, :, i * page : i * page + n].permute(
                1, 0, 2
            )
    return pool.cuda(), ids.to(torch.int32).cuda()


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_paged_tile_decode(
    gpu, monkeypatch, page, dim, dtype, graph=False, default_stream=False
):
    torch, arch = gpu
    if arch not in ("gfx942", "gfx950"):
        pytest.skip("token-mapped loader targets gfx942/gfx950")
    from dispatch.attention import AttentionRequest, dispatch_attention
    from kernels.common import attention_unified as au

    monkeypatch.setenv(f"HIPDNN_{arch.upper()}_3D_GRAPH", str(int(graph)))
    batch, heads, capacity = 3, 2, 97
    ratio = {64: 1, 128: 4, 256: 16}[dim]
    qheads = heads * ratio
    window = 0 if graph else (33 if dim == 128 else 0)
    tdtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    rng = torch.Generator().manual_seed(913 + dim)
    qh = torch.randn(batch, qheads, dim, generator=rng).to(tdtype)
    kh = torch.randn(batch, heads, capacity, dim, generator=rng).to(tdtype)
    vh = torch.randn(batch, heads, capacity, dim, generator=rng).to(tdtype)
    for b in range(batch):
        vh[b] += b
    q = qh.cuda()
    # Legacy page-contained loaders require finite page padding: PV can read
    # masked V rows. New gather paths must safely handle NaN tails as well.
    poison_tail = page < 32 or (arch == "gfx950" and page == 64)
    k, table = _shuffled_cache(torch, kh, page, poison_tail=poison_tail)
    v, _ = _shuffled_cache(torch, vh, page, poison_tail=poison_tail)
    cu = torch.arange(batch + 1, dtype=torch.int32).cuda()
    used = torch.tensor([capacity, capacity - 1, 0], dtype=torch.int32).cuda()
    storage = torch.full((q.numel() + 64,), 117, dtype=tdtype).cuda()
    out = storage[32:-32].view(q.shape)
    request = AttentionRequest(
        batch=batch,
        nhead_q=qheads,
        nhead_k=heads,
        seqlen_q=1,
        seqlen_k=capacity,
        hdim_q=dim,
        hdim_v=dim,
        arch=arch,
        dtype=dtype,
        kv_block_size=page,
        sliding_window=window,
        target_ctas=8,
        algorithm="paged_decode_t32",
    )
    result = dispatch_attention(request)
    from dispatch.attention.common import _problem

    problem = _problem(request)
    assert result.spec.kernel_spec.tile_size == 32
    stream = torch.cuda.default_stream() if default_stream else torch.cuda.Stream()
    torch.cuda.synchronize()
    before = set(au._3D_GRAPHS)
    captured = {}
    # Cross both page and tile boundaries, then empty and one-token rows. No
    # padding in the page table beyond ceil(capacity/page).
    lengths_cases = (
        [capacity, capacity - 1, 0],
        [min(page + 1, capacity), 31, 32],
        [1, 33, 0],
    )
    for lengths in lengths_cases:
        with torch.cuda.stream(stream):
            used.copy_(torch.tensor(lengths, dtype=torch.int32).cuda())
            au.run_unified_attention_torch(
                problem=problem,
                q=q,
                k=k,
                v=v,
                out=out,
                cu_seqlens_q=cu,
                seqused_k=used,
                softmax_scale=1 / math.sqrt(dim),
                block_table=table,
                softcap=0,
                backend="3d",
                stream=stream.cuda_stream,
                tuning_spec=result.spec,
            )
        stream.synchronize()
        expected = _reference(torch, qh, kh, vh, lengths, window)
        atol, rtol = (0.003, 0.005) if dtype == "fp16" else (0.01, 0.02)
        torch.testing.assert_close(out.cpu().float(), expected, atol=atol, rtol=rtol)
        assert bool((storage[:32].cpu() == 117).all()) and bool(
            (storage[-32:].cpu() == 117).all()
        )
        if graph:
            keys = set(au._3D_GRAPHS) - before
            assert len(keys) == 1
            now = {key: au._3D_GRAPHS[key] for key in keys}
            assert all(now[key] is val for key, val in captured.items())
            captured = now
    if page == 1 and not graph:
        au.run_unified_attention_torch(
            problem=problem,
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu,
            seqused_k=used,
            softmax_scale=1 / math.sqrt(dim),
            block_table=table,
            softcap=0,
            backend="3d",
            stream=stream.cuda_stream,
        )
        stream.synchronize()
        torch.testing.assert_close(out.cpu().float(), expected, atol=atol, rtol=rtol)
    # Exercise the dispatch binding as well as the direct explicit runtime.
    binding = result.candidate.bind_torch(
        request,
        result.spec,
        {
            "problem": problem,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "cu_seqlens_q": cu,
            "seqused_k": used,
            "block_table": table,
        },
        stream=stream.cuda_stream,
    )
    binding.launch()
    stream.synchronize()
    torch.testing.assert_close(out.cpu().float(), expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("default_stream", [False, True])
def test_paged_tile_graph(gpu, monkeypatch, page, default_stream):
    test_paged_tile_decode(
        gpu, monkeypatch, page, 64, "bf16", graph=True, default_stream=default_stream
    )
