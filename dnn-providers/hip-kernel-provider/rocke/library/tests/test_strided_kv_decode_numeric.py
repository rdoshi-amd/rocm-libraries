# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Actual tiled split-KV launches on contiguous and padded non-paged caches.

Set ROCKE_REQUIRE_DECODE_GPU=gfx942/gfx950 to require hardware (no skips).
For shared paged graph coverage on gfx1250, set that target and select
``-k 'graph_replay and paged and bf16'`` (D64, GQA-8).
Reference attention is computed independently in FP32 on the CPU from logical
KV tensors, after rounding inputs to the tested fp16/bf16 dtype.
"""

import math
import os

import pytest


@pytest.fixture(scope="module")
def gpu():
    required = os.environ.get("ROCKE_REQUIRE_DECODE_GPU")
    if required:
        import torch
    else:
        torch = pytest.importorskip("torch")
    from rocke.runtime.hip_module import get_device_arch

    if not torch.cuda.is_available():
        if required:
            pytest.fail("required decode GPU is unavailable")
        pytest.skip("requires a ROCm GPU")
    arch = get_device_arch()
    if required and arch != required:
        pytest.fail(f"requested {required}, found {arch}")
    if arch not in ("gfx942", "gfx950", "gfx1250"):
        if required:
            pytest.fail(f"unsupported required decode target {arch}")
        pytest.skip("decode validation targets gfx942/gfx950/gfx1250")
    return torch, arch


def _reference(torch, q, k, v, lengths, window):
    output = torch.zeros_like(q, dtype=torch.float32)
    ratio = q.shape[1] // k.shape[1]
    for batch, length in enumerate(lengths):
        if length == 0:
            continue
        begin = max(0, length - window) if window else 0
        for head in range(q.shape[1]):
            keys = k[batch, head // ratio, begin:length].float()
            values = v[batch, head // ratio, begin:length].float()
            scores = (keys @ q[batch, head].float()) / math.sqrt(q.shape[2])
            output[batch, head] = scores.softmax(0) @ values
    return output


def _cache(torch, logical, layout):
    batch, heads, capacity, dim = logical.shape
    if layout == "bhsd":
        return logical.cuda()
    if layout == "bshd":
        return logical.permute(0, 2, 1, 3).contiguous().cuda().permute(0, 2, 1, 3)
    # Unit-D rows separated by 8 unused elements, plus batch/head padding.
    row_stride = dim + 8
    head_stride = (capacity + 1) * row_stride
    batch_stride = (heads + 1) * head_stride
    span = (
        (batch - 1) * batch_stride
        + (heads - 1) * head_stride
        + (capacity - 1) * row_stride
        + dim
    )
    backing = torch.full((span,), float("nan"), dtype=logical.dtype)
    view = backing.as_strided(logical.shape, (batch_stride, head_stride, row_stride, 1))
    view.copy_(logical)
    return backing.cuda().as_strided(logical.shape, view.stride())


def _paged_cache(torch, logical, block_size):
    batch, heads, capacity, dim = logical.shape
    blocks = (capacity + block_size - 1) // block_size
    backing = torch.zeros(batch, blocks * block_size, heads, dim, dtype=logical.dtype)
    backing[:, :capacity].copy_(logical.permute(0, 2, 1, 3))
    pages = backing.reshape(batch * blocks, block_size, heads, dim).flip(0).contiguous()
    table = torch.arange(batch * blocks - 1, -1, -1, dtype=torch.int32).view(
        batch, blocks
    )
    return pages.cuda(), table.cuda()


# Distinct batches, empty/partial segments, both physical orders, independent
# K/V strides, and final tiles without allocation padding. Repeated passes change
# the lengths in place to exercise binding/graph cache reuse.
_CASES = [
    (1, 64, 2, 4, "bhsd", "bhsd", 0),
    (17, 64, 2, 4, "bshd", "bshd", 0),
    (65, 128, 2, 8, "bhsd", "bshd", 0),
    (97, 128, 2, 8, "bshd", "bhsd", 17),
    (65, 256, 1, 8, "bhsd", "bhsd", 0),
    (129, 256, 2, 8, "padded", "bshd", 33),
    (1031, 64, 2, 4, "bshd", "padded", 0),
]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("kv_layout", ["strided", "paged"])
def test_strided_kv_decode(
    gpu,
    monkeypatch,
    dtype,
    case,
    kv_layout,
    graph=False,
    default_stream=False,
    block_size=16,
):
    torch, arch = gpu
    from kernels.common import attention_unified as au

    if arch == "gfx1250" and (
        kv_layout == "strided"
        or dtype != "bf16"
        or case[1] != 64
        or case[3] // case[2] != 8
    ):
        pytest.skip("gfx1250 coverage is paged BF16 D64 GQA-8 decode")
    monkeypatch.setenv("HIPDNN_GFX942_3D_GRAPH", str(int(graph)))
    monkeypatch.setenv("HIPDNN_GFX950_3D_GRAPH", str(int(graph)))
    monkeypatch.setenv("HIPDNN_GFX1250_3D_GRAPH", str(int(graph)))
    capacity, dim, kv_heads, q_heads, k_layout, v_layout, window = case
    batch = 3
    tdtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    rng = torch.Generator().manual_seed(701 + capacity + dim)
    q_host = torch.randn(batch, q_heads, dim, generator=rng).to(tdtype)
    k_host = torch.randn(batch, kv_heads, capacity, dim, generator=rng).to(tdtype)
    v_host = torch.randn(batch, kv_heads, capacity, dim, generator=rng).to(tdtype)
    # Large per-batch shifts make shared-cache bugs observable.
    for b in range(batch):
        v_host[b] += b
    q = q_host.cuda()
    if kv_layout == "strided":
        k, v = _cache(torch, k_host, k_layout), _cache(torch, v_host, v_layout)
        block_table = None
    else:
        k, block_table = _paged_cache(torch, k_host, block_size)
        v, _ = _paged_cache(torch, v_host, block_size)
    cu_q = torch.arange(batch + 1, dtype=torch.int32).cuda()
    lengths_host = [capacity, max(0, capacity - 3), 0]
    lengths = torch.tensor(lengths_host, dtype=torch.int32).cuda()
    count = q.numel()
    storage = torch.full((count + 64,), 117, dtype=tdtype).cuda()
    output = storage[32:-32].view(q.shape)
    problem = au.UnifiedAttentionProblem(
        total_q=batch,
        num_seqs=batch,
        num_query_heads=q_heads,
        num_kv_heads=kv_heads,
        head_size=dim,
        block_size=block_size,
        max_seqlen_q=1,
        max_seqlen_k=capacity,
        dtype=dtype,
        sliding_window=window,
        clamp_arch=arch,
        target_ctas=8,
    )
    stream = torch.cuda.default_stream() if default_stream else torch.cuda.Stream()
    torch.cuda.synchronize()
    # Paged cases exercise the direct runtime; strided cases also test dispatch.
    if kv_layout == "strided":
        from dispatch.attention import AttentionRequest, dispatch_attention

        request = AttentionRequest(
            batch=batch,
            nhead_q=q_heads,
            nhead_k=kv_heads,
            seqlen_q=1,
            seqlen_k=capacity,
            hdim_q=dim,
            hdim_v=dim,
            arch=arch,
            dtype=dtype,
            kv_layout="strided",
            kv_block_size=block_size,
            sliding_window=window,
            target_ctas=8,
        )
        result = dispatch_attention(request)
        assert result.candidate.name == "attention_strided_decode"
        binding = result.candidate.bind_torch(
            request,
            result.spec,
            dict(
                problem=problem,
                q=q,
                k=k,
                v=v,
                out=output,
                cu_seqlens_q=cu_q,
                seqused_k=lengths,
            ),
            stream=stream.cuda_stream,
        )
    graphs_before = set(au._3D_GRAPHS)
    captured_graphs = {}
    for step in range(3):
        if step:
            lengths_host = (
                [max(0, capacity - 1), capacity, min(1, capacity)]
                if step == 1
                else [capacity, min(1, capacity), capacity]
            )
            lengths.copy_(torch.tensor(lengths_host, dtype=torch.int32))
            torch.cuda.synchronize()
        if step and kv_layout == "strided":
            binding.launch()
        else:
            au.run_unified_attention_torch(
                problem=problem,
                q=q,
                k=k,
                v=v,
                out=output,
                cu_seqlens_q=cu_q,
                seqused_k=lengths,
                softmax_scale=1 / math.sqrt(dim),
                block_table=block_table,
                softcap=0.0,
                backend="3d",
                kv_layout=kv_layout,
                stream=stream.cuda_stream,
            )
        stream.synchronize()
        if graph:
            graph_keys = set(au._3D_GRAPHS) - graphs_before
            # Direct and explicit-spec launches have separate cache identities.
            assert len(graph_keys) == (2 if kv_layout == "strided" and step else 1)
            current_graphs = {key: au._3D_GRAPHS[key] for key in graph_keys}
            assert all(
                current_graphs[key] is value for key, value in captured_graphs.items()
            )
            captured_graphs = current_graphs
        expected = _reference(torch, q_host, k_host, v_host, lengths_host, window)
        atol, rtol = (0.01, 0.02) if dtype == "bf16" else (0.003, 0.005)
        torch.testing.assert_close(output.cpu().float(), expected, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            storage[:32].cpu(), torch.full((32,), 117, dtype=tdtype), atol=0, rtol=0
        )
        torch.testing.assert_close(
            storage[-32:].cpu(), torch.full((32,), 117, dtype=tdtype), atol=0, rtol=0
        )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("kv_layout", ["strided", "paged"])
@pytest.mark.parametrize("default_stream", [False, True])
def test_strided_decode_graph_replay(
    gpu, monkeypatch, dtype, kv_layout, default_stream
):
    case = (17, 64, 2, 16, "bshd", "bshd", 0) if gpu[1] == "gfx1250" else _CASES[1]
    test_strided_kv_decode(
        gpu,
        monkeypatch,
        dtype,
        case,
        kv_layout,
        graph=True,
        default_stream=default_stream,
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("window", [0, 17])
def test_strided_decode_tile_policy(gpu, monkeypatch, dtype, block_size, window):
    case = (*_CASES[2][:-1], window)
    test_strided_kv_decode(
        gpu,
        monkeypatch,
        dtype,
        case,
        "strided",
        block_size=block_size,
    )
