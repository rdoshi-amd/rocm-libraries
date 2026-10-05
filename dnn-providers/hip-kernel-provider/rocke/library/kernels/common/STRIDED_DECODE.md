# Non-paged KV input to tiled decode

The gfx942 and gfx950 unified 3D split-KV kernels accept non-paged KV through
`kv_layout="strided"`. This specialization changes global-memory addressing;
it reuses the tiled matrix-core computation, LDS layout, segment workspace,
and reducer. It does not copy KV or construct an identity page table.
The default `kv_layout="paged"` retains its existing ABI and addressing.

## Input contract

K and V have logical shape `[B, Hkv, capacity, D]`. Contiguous BHSD storage
works directly. For contiguous BSHD storage, pass a metadata-only
`cache.permute(0, 2, 1, 3)` view. K and V may have different outer strides,
including padding, with these bounds:

- fp16 or bf16, D in 64/128/256; one query per sequence;
- positive, nonoverlapping strides, unit D stride, 16-byte-aligned rows/base;
- per-head byte span and token stride at most `0x7fff0000`;
- query-to-KV head ratio in 1/2/4/8/16;
- contiguous Q and output `[B, Hq, D]`, on the same device as KV and metadata.

Capacity need not be a page or tile multiple. Device `seqused_k` is contiguous
int32 `[B]`, with each value in `[0, problem.max_seqlen_k]`, and
`problem.max_seqlen_k <= capacity`. Device `cu_seqlens_q` is contiguous int32
`[B+1]` with contents `[0,1,...,B]`. These **contents are caller contracts**:
validation checks tensor metadata without copying device values to the host.
The query position is `seqused_k[b] - 1`, supporting bottom-right causal
alignment and an optional sliding window. Empty sequences produce zero output.

The registered `attention_strided_decode` candidate declines bias, sinks, FP8,
top-left causal alignment, and multi-query prefill. Arbitrary D strides and
overlapping or unaligned views and K/V/output scaling are not supported. This change adds no LSE output.

## Usage and validation

Use the existing [runtime entry point](attention_unified.py) with the following
arguments, after creating the problem and device tensors described above:

```python
run_unified_attention_torch(
    problem=problem,
    q=q,
    k=k_bhsd,  # or k_bshd.permute(0, 2, 1, 3)
    v=v_bhsd,
    out=out,
    cu_seqlens_q=query_offsets,
    seqused_k=kv_lengths,
    block_table=None,
    softmax_scale=problem.head_size**-0.5,
    softcap=0.0,
    backend="3d",
    kv_layout="strided",
    stream=stream,
)
```

For registry consumers, set `AttentionRequest.kv_layout="strided"`; the
[candidate](../../dispatch/attention/strided_decode.py) provides spec building
and `bind_torch`. Direct and dispatched launches share the same spec policy.
The binding accepts only `softmax_scale` and `stream` keyword overrides; it
rejects softcap, bias/sinks tensors, and unknown arguments. Explicit segment
and reducer specs must agree with the runtime problem's shapes, dtype, window,
and enabled features. The low-level [layout adapter](attention_kv_cache.py) packs
independent K/V batch/head strides as i64 byte offsets and token strides/span
as i32 bytes. Separate C builder entry points preserve installed paged spec
structs; strided symbols carry `_stridedkv`.

The [numeric tests](../../tests/test_strided_kv_decode_numeric.py) compare
the direct runtime and registered binding against an independent CPU FP32
reference after input quantization. They include batched BHSD/BSHD, different
K/V strides, padding, unpadded capacity tails, changing valid lengths, empty
sequences, windows, paged regressions, and graph replay. Set
`ROCKE_REQUIRE_DECODE_GPU=gfx942` or `gfx950` to require the target GPU rather
than skip. No performance claim follows from this correctness coverage.
