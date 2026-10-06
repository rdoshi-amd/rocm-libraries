# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host admission for token-mapped tiled decode. No device metadata reads."""

BUFFER_LIMIT = 0x7FFF0000


def validate_paged_decode(problem, tensors, tuning_spec=None):
    """Validate the new loader ABI without synchronization or graph allocation.

    Caller-owned values: lengths lie in [0,max_seqlen_k], query offsets are
    [0,1,...,B], and referenced page IDs are in [0,physical_pages). Inactive
    table entries need not be initialized. K and V use the same page mapping.
    """
    p = problem
    if p.max_seqlen_q != 1 or p.total_q != p.num_seqs:
        raise ValueError("paged gather requires one query per sequence")
    if p.dtype not in ("fp16", "bf16") or p.use_fp8:
        raise ValueError("paged gather requires fp16/bf16")
    if p.use_sinks or p.use_alibi or p.use_qq_bias or p.softcap:
        raise ValueError("paged gather does not support sinks, bias or softcap")
    if not 0 <= p.max_seqlen_k <= BUFFER_LIMIT:
        raise ValueError("paged gather sequence length exceeds token-offset bound")
    if tuning_spec is not None:
        segment = tuning_spec.kernel_spec
        if tuning_spec.path != "3d" or tuning_spec.reduce_spec is None:
            raise ValueError("paged gather requires segment and reduce specs")
        for name, expected in (
            ("kv_layout", "paged"),
            ("head_size", p.head_size),
            ("block_size", p.block_size),
            ("num_query_heads", p.num_query_heads),
            ("num_kv_heads", p.num_kv_heads),
            ("num_seqs", p.num_seqs),
            ("dtype", p.dtype),
            ("sliding_window", p.sliding_window),
            ("kv_storage_dtype", None),
            ("use_i64_kv_addr", False),
            ("use_sinks", False),
            ("use_alibi", False),
            ("use_qq_bias", False),
            ("has_softcap", False),
        ):
            if not hasattr(segment, name) or getattr(segment, name) != expected:
                raise ValueError(
                    f"paged kernel_spec.{name} missing or disagrees with problem"
                )
        for name in (
            "head_size",
            "num_query_heads",
            "num_kv_heads",
            "dtype",
            "num_segments",
        ):
            if getattr(tuning_spec.reduce_spec, name, None) != getattr(segment, name):
                raise ValueError(f"paged reduce_spec.{name} disagrees with segment")
    # The segment workspace uses bounded scalar offsets too. Default routing
    # selects at most 128 segments; use that ceiling before building its spec.
    segments = (
        128
        if tuning_spec is None
        else getattr(tuning_spec.kernel_spec, "num_segments", 0)
    )
    if not 1 <= segments <= 128:
        raise ValueError("paged gather requires 1..128 segments")
    if p.num_seqs * p.num_query_heads * p.head_size * segments * 4 > BUFFER_LIMIT:
        raise ValueError("paged gather workspace exceeds buffer-offset bound")
    q, k, v, out, cu, used, table = tensors
    dtype = "torch.float16" if p.dtype == "fp16" else "torch.bfloat16"
    q_shape = (p.num_seqs, p.num_query_heads, p.head_size)
    tail = (p.block_size, p.num_kv_heads, p.head_size)
    if tuple(q.shape) != q_shape or tuple(out.shape) != q_shape:
        raise ValueError("paged decode Q/O shape disagrees with problem")
    if (
        len(k.shape) != 4
        or tuple(k.shape[1:]) != tail
        or tuple(v.shape) != tuple(k.shape)
        or k.shape[0] <= 0
    ):
        raise ValueError("paged decode K/V must be matching [pages,P,Hkv,D] pools")
    # Check both actual allocations; num_kv_blocks supplied by callers is not
    # proof of representability. Element counts use Python's unbounded integer.
    for tensor in (k, v):
        if int(tensor.numel()) * 2 > BUFFER_LIMIT:
            raise ValueError("paged gather KV pool exceeds buffer-offset bound")
    if any(str(t.dtype) != dtype for t in (q, k, v, out)):
        raise ValueError("paged decode Q/K/V/O dtype disagrees with problem")
    for tensor, shape in ((cu, (p.num_seqs + 1,)), (used, (p.num_seqs,))):
        if str(tensor.dtype) != "torch.int32" or tuple(tensor.shape) != shape:
            raise ValueError("paged decode lengths/offsets must be int32 vectors")
    pages = max(1, (p.max_seqlen_k + p.block_size - 1) // p.block_size)
    if (
        table is None
        or len(table.shape) != 2
        or table.shape[0] != p.num_seqs
        or table.shape[1] < pages
        or str(table.dtype) != "torch.int32"
    ):
        raise ValueError(
            "paged decode block table must be int32 [B,ceil(capacity/P)] or wider"
        )
    if int(table.numel()) * 4 > BUFFER_LIMIT:
        raise ValueError("paged gather page table exceeds index bound")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("paged decode tensors must be contiguous")
    if not q.is_cuda or any(t.device != q.device for t in tensors):
        raise ValueError("paged decode tensors must be on the same device")
    if any(int(t.data_ptr()) % 16 for t in (q, k, v, out)):
        raise ValueError("paged decode Q/K/V/O must be 16-byte aligned")
