<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# What the FlyDSL packs cover

Two packs, built once for `gfx11-generic` and shipped to all eight RDNA3 /
RDNA3.5 arches it covers (gfx1100–gfx1103, gfx1150–gfx1153): **RMSNorm forward**
(12 kernel objects, §1–§4) and **SDPA forward** (12 kernel objects, §5). Every graph a pack accepts is
computed by one of its objects; everything else is **declined**, so another
engine gets the plan.

That last point is the design rule this whole file exists to document. The
kernels are pre-built HSACO — [`FlydslRmsNormNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslRmsNormNative.cpp)
cannot change what they compute, only decide whether they are the right thing to
run. So every condition below is enforced by *refusing*, never by adapting. A
matcher that stretched to accept a near-miss would compute the wrong answer
silently; one that declines it costs a fallback.

---

## 1. The 12 instances

Generated from [`generators/_instances.py`](generators/_instances.py), which is
the single source of truth — the compiler, the packer and the descriptor emitter
all read it, so adding an instance is a row there rather than an edit in three
places that can disagree.

| Instance | Tier | dtype | `N` | `block_threads` | priority |
|---|---|---|---|---|---|
| `rmsnorm_bf16_ngeneric_bt256` | generic | bf16 | runtime | 256 | 10 |
| `rmsnorm_f16_ngeneric_bt256` | generic | f16 | runtime | 256 | 10 |
| `rmsnorm_bf16_n3072_bt256` | specialized | bf16 | 3072 | 256 | 100 |
| `rmsnorm_bf16_n3584_bt256` | specialized | bf16 | 3584 | 256 | 100 |
| `rmsnorm_bf16_n4096_bt256` | specialized | bf16 | 4096 | 256 | 100 |
| `rmsnorm_bf16_n5120_bt256` | specialized | bf16 | 5120 | 256 | 100 |
| `rmsnorm_bf16_n8192_bt256` | specialized | bf16 | 8192 | 256 | 100 |
| `rmsnorm_f16_n3072_bt256` | specialized | f16 | 3072 | 256 | 100 |
| `rmsnorm_f16_n3584_bt256` | specialized | f16 | 3584 | 256 | 100 |
| `rmsnorm_f16_n4096_bt256` | specialized | f16 | 4096 | 256 | 100 |
| `rmsnorm_f16_n5120_bt256` | specialized | f16 | 5120 | 256 | 100 |
| `rmsnorm_f16_n8192_bt256` | specialized | f16 | 8192 | 256 | 100 |

2 dtypes × (1 generic + 5 specialized widths). `block_threads` is single-valued
today: it is launch geometry, not correctness, so a second value only earns its
6 extra objects once benchmarking says it wins somewhere. It stays in the knob
dict so the descriptors and the matcher already carry the axis when it grows.

### Why these five widths

3072 / 4096 / 5120 / 8192 are Llama-family hidden sizes; 3584 is Qwen2-7B. This
is a list of what the targeted models actually run, not a sweep.

**Widening it is cheap, and is not a coverage fix** — every other `N` is already
served, by the generic tier. Adding a width buys speed on that width, nothing
else.

### Why the two tiers exist

FlyDSL's vectorized 128-bit path caches tiles in a Python list of registers,
which only exists while the tile loop is unrolled, and unrolling needs a
compile-time count. So the fast path **genuinely cannot take a runtime `N`** —
the tier split is forced by the kernel, not chosen for convenience.

The generic tier is the vendored runtime-`N` variant (modification 1 in
[`kernels_src/kernels/norm/rmsnorm_kernel.py`](kernels_src/kernels/norm/rmsnorm_kernel.py)):
it reads `N` from the input tensor descriptor and predicates the tail, so one
binary covers every width at the cost of the fast path.

Both tiers stay in the candidate set when both apply — the matcher admits both
for a baked width, and the tie is broken by ranking, not by refusal.

Which one wins is expressed purely as **data**. `IKernelHeuristic::rank` orders
`score desc → priority desc → kernelId asc`, and `flydslRmsNormScore` returns
`kernel.priority` unchanged rather than inventing a second opinion — so the
`priority` column above *is* the ranking, decided at the first comparison
(100 > 10, baked before generic). The later tiebreaks never come into play here,
which is the point: selection is authored in the descriptors and refined by the
autotuner, not computed from the shape. A score derived from the shape would
also make the winner cache's key insufficient.

### The gap that is deliberate: N ≤ 2048

`RMSNORM_SMALL_N_THRESHOLD = 2048`, and 2048 is **absent** from the specialized
list on purpose. At or below that width `build_rmsnorm_module` does not
specialize the kernel above — it returns `_build_rmsnorm_large_m_small_n_module`,
a *different* kernel with a different ABI (an extra `m_in` kernarg) and a
different launch geometry (grid divided by `BLOCK_M` rather than one block per
row). Asking for `N=2048` here would not produce a specialization of this kernel
at all.

That path is worth having — it is upstream's large-M/small-N specialization —
but it is a second launch record, so it lands as its own instance family rather
than hiding inside this one. **Until then small `N` is served by the generic
tier**, correctly but without the specialization. The generator asserts the
Python threshold and the table's constant still agree, because drift there is
the kind that produces a correct-looking object dispatched with the wrong grid.

---

## 2. What a graph must be

Two matchers run, in this order.

### Graph-scoped — `flydslRmsNormGraphMatches`

Evaluated once per graph. Every one of these is a hard refusal.

| Condition | Why |
|---|---|
| Exactly **one node**, and its attributes are `RMSNormAttributes` | Each kernel serves one complete graph. `RMSNormAttributes` is its own union arm, so matching it is the whole node-type test — there is no mode enum to disambiguate it from layer norm. |
| No `bias_tensor_uid` | Every instance was compiled with no bias term and adds nothing after scaling. |
| No `inv_rms_tensor_uid`, and `forward_phase != TRAINING` | All 12 were built `store_rstd=False`. A TRAINING graph wants the statistic even when it forgot to name the tensor, so the phase is refused too. |
| x, scale, y all present and all **device operands** | Not virtual, not pass-by-value: the kernel reads each through a pointer. |
| One dtype for all three, in {`BFLOAT16`, `HALF`} | The kernel loads gamma with the same element type it loads x with, and writes y in that type. |
| x, y rank ≥ 2; scale rank ≥ 1; `dims` and `strides` both present and equal length | Rank ≥ 2 so there is a row to reduce over. The matcher runs on an *unvalidated* graph, so it must be total against null or mismatched dims. |
| `sameShape(x, y)` | |
| x, y, scale all **packed row-major** | The innermost stride is a compile-time literal 1 in the kernel and has no descriptor slot. The check is stronger than the descriptor strictly needs — it could carry a padded row stride — but it is what makes collapsing rank > 2 to (M, N) sound, since one row stride has to describe all the leading axes at once. |
| rows > 0 and columns > 0 | |
| gamma is **exactly** the normalised axis: `rowCount(scale) == 1 && innermostExtent(scale) == columns` | Any other extent would be a broadcast the kernel does not perform. |
| epsilon is runtime-user-supplied, **or** within 1% of `1e-5` | See below. |

Rank > 2 is accepted, not rejected: leading axes collapse to a row count
(`rowCount` = product of every extent but the innermost). A rank-4 NCHW graph
with a `(1,1,1,N)` broadcast scale is a match.

On success it binds `flydsl_rmsnorm.{x,scale,epsilon,y}.uid` plus `.rows` and
`.columns`, so nothing downstream re-derives them from the graph.

### Kernel-scoped — `flydslRmsNormKernelMatches`

Evaluated once per candidate kernel, against the KMD metadata:

- `dtype` metadata must equal the graph dtype's kernel spelling. This function
  is the one place the two vocabularies meet — descriptors say `bf16`/`f16`, the
  flatbuffer enum says `BFLOAT16`/`HALF`.
- The baked `N`: `GENERIC_N_SENTINEL = 0` admits any width; any other value is
  admitted only when it equals the bound `columns`. A real graph never has a
  zero-width normalised axis, so the value is free to carry the sentinel.

### Epsilon, which is checked twice

Epsilon is a **literal in the kernel body** (`EPS = 1e-5`), not a kernarg. So:

- A graph that **bakes** its epsilon into the op-graph is compared at match time
  against `BAKED_EPSILON = 1e-5` with `EPSILON_RELATIVE_TOLERANCE = 1e-2`, and
  declined when it disagrees — falling through to an engine that can honour it,
  rather than failing the whole plan build.
- A graph that **defers** epsilon to execute cannot be inspected then, so
  `launch()` re-checks the resolved value and throws
  `HIPDNN_PLUGIN_STATUS_INVALID_VALUE` on a mismatch.

The 1% tolerance is wide enough that `1e-5` surviving a half-precision
round-trip (~1.4e-3 relative error) still matches, and far too narrow to admit a
caller who genuinely wanted a different epsilon.

---

## 3. Out of scope

Not gaps to be filled before this ships — each is something the shipped objects
do not compute, declined rather than approximated.

| | Why, and what handles it |
|---|---|
| **RMSNorm backward** | Forward-only. The vendored copy deliberately drops the `rmsnorm_bwd_kernel` re-export (modification 2). |
| **Training phase / saved `inv_rms`** | `store_rstd=False` in all 12. |
| **Bias** | No instance has the term. |
| **dtypes other than bf16 / f16** | fp32, fp8 and the rest are not built. Declined on dtype in both matchers. |
| **Mixed dtypes across x / scale / y** | The kernel uses one element type throughout. |
| **Non-packed / padded / transposed operands** | Innermost stride is a literal 1 with no descriptor slot. |
| **Rank < 2 on x or y** | No row to reduce over. |
| **Broadcast gamma that is not the normalised axis** | Not a broadcast the kernel performs. |
| **Multi-node graphs, fused add+RMSNorm** | One node per kernel. Fusion would be its own pack. |
| **Architectures outside gfx11-generic** | Objects are checked in for the gfx11 generic family (gfx1100–gfx1103, gfx1150–gfx1153). gfx1170/gfx1171 belong to `gfx11-7-generic`, and CDNA / gfx12 are other families. With `GPU_TARGETS` naming no arch we ship kernels for, the integration reports dormant at configure time and stages nothing — it does not fail, and it does not silently produce an empty shard. Adding a family is a regeneration ([REGEN.md](REGEN.md) §3), not a code change. |
| **Ops other than RMSNorm** | `OPS` in `_instances.py` has one entry. |

---

## 4. Where each claim is checked

| Claim | Checked by |
|---|---|
| The 12 objects are what the pinned toolchain produces | [REGEN.md](REGEN.md) §2 — SHA256 against `manifest.json`, currently 12/12 |
| Descriptors agree with the objects they describe | `gen_descriptors.py --check`, run by the build before staging |
| Each object is built for the target its directory names | `gen_descriptors.py --check` reads `amdhsa.target` and, for a generic target, the ELF machine |
| Every member arch's shard holds exactly the checked-in objects | `tools/check_shards.py`, run by the build after the product pack |
| The vendored sources match upstream but for recorded modifications | `tools/diff_upstream.py` ([REGEN.md](REGEN.md) §5) |
| The pack ships both tiers for both dtypes, and every kernel declares the 8-slot signature | `TestFlydslRmsNormPacks` — and the shard census that runs it, which fails when the suite is green against an empty shard |
| The accept/refuse rules above hold | The matcher-acceptance suites in `TestFlydslRmsNormEngine.cpp` |

---

## 5. SDPA forward — `hipkernel:flydsl_sdpa`

The kernel is FlyDSL's RDNA4 flash-attention forward from AITER, ported to the
gfx11 WMMA ABI and widened so a handful of objects cover many shapes
([`kernels_src/kernels/attention/flash_attn_func_gfx1151.py`](kernels_src/kernels/attention/flash_attn_func_gfx1151.py),
whose header lists every modification). The native half is
[`FlydslSdpaNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaNative.cpp).

### 5.1 What is baked, and what is not

| Baked per object | Runtime kernel argument |
|---|---|
| dtype (`bf16`, `f16`) | batch |
| `head_dim` | `seq_len_q`, `seq_len_kv` — independently, any value |
| causal variant: a right bound is applied, or not | query heads, and the GQA/MQA group size |
| tile (`block_m` 128, `block_n` 32) | every operand's (batch, sequence, head) stride |
| | softmax scale |
| | `right_bound`, `left_bound`, diagonal alignment |
| | whether to write the LSE output, and its strides |

So the instance key is `dtype × head_dim × causal`, and every axis it
enumerates is **model-determined**, never request-determined. That is what
keeps the table at 12 rows:

| | `head_dim` 64 | `head_dim` 96 | `head_dim` 128 |
|---|---|---|---|
| bf16 | causal, non-causal | causal, non-causal | causal, non-causal |
| f16 | causal, non-causal | causal, non-causal | causal, non-causal |

The causal variant is the one that applies a right bound — top-left or
bottom-right causal, and any band to the right of the diagonal. It is two
objects rather than a runtime flag because it changes the loop: it bounds the
KV loop at the diagonal and skips fully masked tiles, and the non-causal one
carries a V prefetch across iterations that the causal one drops for register
pressure. Everything else about the mask is a runtime argument, served by both.

### 5.2 What a graph must be

One SDPA-forward node, and:

| Condition | Why |
|---|---|
| Q, K, V, O present, rank 4, device operands, **not ragged** | The kernel reads each through a pointer as a dense batch; a ragged (THD) operand would be read as one. |
| One dtype for all four, `BFLOAT16` or `HALF` | One element type throughout; no fp32 WMMA operand form exists. |
| Head-dim stride 1 on all four; other strides positive | Each lane reads a row's head-dim values as one contiguous vector. Every other stride is a kernel argument, so BSHD, BHSD and packed-QKV views are all served. |
| K and V share every extent; O has Q's; V's head dim equals Q/K's | The kernel has one `head_dim` and does not broadcast. |
| `Hq % Hkv == 0` | The kv head is the query head divided by the group size. |
| Any `left_bound`/`right_bound` ≥ −1, either alignment, or the deprecated causal booleans (not both) | The kernel applies the reference's two-sided rule (`CpuFpReferenceSdpa` `isMasked`) directly. |
| Scale: `attn_scale_value` (> 0), a pass-by-value scalar tensor, or absent | Absent is `1/√head_dim`, the reference's default. The running max is over unscaled scores, which orders them correctly only for a positive scale. |
| Stats, if requested: a named f32 `[B, H, Sq, 1]` device tensor | The kernel writes the natural-log LSE of the scaled scores there. |
| `Sq`, `Skv` < 2²⁹; one (batch, kv head) slice of K or V under 2 GiB; grid under 2³¹ | int32 mask arithmetic, and 32-bit bounds-checked buffer descriptors. |

Any sequence length is correct: K and V reads past `seq_len_kv` return zero and
every such score is masked, so nothing is padded and nothing pads the softmax.
A row no key reaches — a narrow window, or bottom-right causal with more
queries than keys — gets O = 0 and LSE = −inf, as the reference defines it.

### 5.3 Against hipDNN's attention Tier 0 / Tier 1

| # | Requirement | Status |
|---|---|---|
| 1 | fp16, bf16; fp32 accumulate | **served** |
| 2 | BSHD, BHSD, general B/S/H strides, packed QKV | **served** |
| 3 | MHA, MQA, GQA | **served** (`Hk == Hv`) |
| 4 | attention scale | **served** (positive) |
| 5 | causal, top-left | **served** |
| 6 | causal, bottom-right, `Sq ≠ Skv` | **served** |
| 7 | softmax stats / LSE | **served** |
| 8 | arbitrary sequence lengths | **served** |
| 9 | head dims 64, 128, 256 | **64, 128 served (and 96)**; 256 missing |
| 10 | padding mask via `SEQ_LEN_Q`/`SEQ_LEN_KV` | missing |
| 11 | sliding window (`left_bound`) | **served** (and right-side bands) |
| 12 | varlen / THD | missing (declined) |
| 13 | decode (`Sq == 1`) | **served**; idle waves skip the matrix work, but one query row still occupies a 128-row tile |
| 14 | additive bias | missing |

### 5.4 What is still missing, and what closing it takes

Each is declined today, so the graph plans on another engine or fails cleanly
at `check_support()`, never at `execute()`.

| | Effort | What it takes |
|---|---|---|
| **`head_dim` 256** (Tier 0) | large | Its O accumulators and register-resident Q operands alone fill the 256-VGPR ceiling, and the object spills. Needs a schedule that stages Q through LDS, or splits O into two passes. |
| **Additive bias** (Tier 1; `F.sdpa(attn_mask=)`) | medium | A bias pointer and four broadcast strides, two bounds-checked loads per tile, and an add before the mask. The d128 objects sit at 255–256 VGPRs, so it may need a baked `has_bias` variant rather than a runtime flag. |
| **Padding mask** (Tier 1) | medium | Two per-batch length pointers read once per workgroup. The diagonal offset is already computed in-kernel from the lengths for exactly this. hipDNN's CPU reference does not implement padding yet, so its semantics must be defined there first. |
| **Varlen / THD** (Tier 1) | medium, after padding | Ragged offsets replace the batch stride; the grid stays over the longest sequence and tiles past a sequence's end exit early. |
| **Decode throughput** | medium / large | Pack the GQA group into the 128 query rows (medium), then split-KV with an LSE-combine pass (large). |
| `head_dim` 80, 112 | small–medium | The kernel asserts `head_dim % 32`; 16 should suffice but the cooperative loads must be re-verified. 40/72/104 (`% 8`) need a WMMA K-tail. |
| `Dv ≠ Dqk` (MLA) | medium–large | A second head dim for V's LDS tile, the O accumulators and the store. |
| `Hk ≠ Hv` | small, low value | A second group size. |
| fp32 I/O, FP8, dropout, paged KV, block masks, sinks, ALiBi, softcap | out of scope | As hipDNN's requirements state; ALiBi and softcap have no reference semantics. |
| **Backward** | out of scope here | Inference-only, as for hipDNN's Tier 0/1. LSE is emitted so a backward can be added without an ABI break. |
| **Other architectures** | see below | The objects are `gfx11-generic` builds, so every RDNA3 / RDNA3.5 part is served. |

**Every gfx11 generic member** (gfx1100–gfx1103, gfx1150–gfx1153) is served by
the one object set; execution is verified on gfx1151, and the other seven rest on
the loader's generic-target rule, the ELF machine check and the shard check.
FlyDSL 0.3.4 needs the `_generic_targets.py` shim to lower for a generic target
(REGEN.md §3), and the SDPA objects are built with `amdgpu-use-amdgpu-trackers`,
without which d128 causal spills a few VGPRs: the generic ISA lacks gfx115x's
scalar-float instructions. **gfx1170/gfx1171** need `gfx11-7-generic` objects.
**gfx12** needs per-family WMMA codegen: its operand ABI differs, so one object
cannot span both; the kernel source can, behind four small helpers.

### 5.5 Where each claim is checked

| Claim | Checked by |
|---|---|
| The 12 objects are what the pinned toolchain produces | [REGEN.md](REGEN.md) §2 — 12/12 byte-identical |
| Each object has the declared 29-argument layout and spills no registers | `gen_sdpa.py`, before the object is written |
| The vendored kernel matches AITER but for its recorded modifications | `tools/diff_upstream.py --aiter` ([REGEN.md](REGEN.md) §5) |
| The shard ships exactly one object per `dtype × head_dim × causal` class, each with the 29-slot signature | `TestFlydslSdpaPacks`, and the shard census that runs it |
| The accept/decline rules above, the bindings, and candidate selection | `TestFlydslSdpaGraphAccepts`, `TestFlydslSdpaGraphDeclines`, `TestFlydslSdpaBinding`, `TestFlydslSdpaKernelMatch` |
| The staged objects compute attention and LSE — GQA, ragged lengths, cross-attention, both causal corners, windows, bands, keyless rows, decode, `head_dim` 96, runtime and default scale — against a double-precision reference | `TestGpuFlydslSdpaDispatch` |

---

## 6. The instance budget: what is baked, what is runtime, and why

Every feature is either a **runtime argument** (one object serves it, at whatever
cost it adds when unused) or a **compile-time axis** (more objects, each paying
only for what it computes). The count of objects is a cost too — every one is
built, reviewed, packed into every member shard and verified — so neither side
wins by default. The rule this provider follows:

1. **Default classification.** A per-launch scalar that changes only loop
   bounds, addresses or the epilogue is a runtime argument: lengths, head
   counts, strides, scale, mask offsets and bounds, `lse_on`. Anything that adds
   work per score in the inner loop or keeps extra state live in registers (an
   additive bias tile, softcap, ALiBi), or that changes the schedule (a
   `head_dim` 256 layout, a decode tile), is a compile-time axis.
2. **The gate.** A runtime feature stays runtime only if, with the feature
   unused, the shipped object matches a build with the feature compiled out: no
   spill, no lost occupancy, and time within the measurement noise on a fixed
   shape set (repeated, interleaved runs; a difference counts only when its sign
   repeats). One that fails becomes a compile-time axis or is restructured.
3. **Axes are model-determined, never request-determined.** `dtype`,
   `head_dim`, `causal`, a future `has_bias` — properties of the model a graph
   comes from. Never a sequence length, head count or window width: those stay
   runtime, so the count does not grow with the shapes a model is run at.
4. **An axis splits only what needs it.** A bias variant adds bias objects; it
   does not double the table. Specialized fast objects are added beside a
   generic one only where they measurably win, ranked by `priority` (the RMSNorm
   pattern, §1).

The measurements behind each decision are kept outside the repository; what is
recorded here is the decision and the gate it passed.

### 6.1 Runtime features audited against compiled-out builds

| Feature (SDPA) | Where its cost would be | Result | Decision |
|---|---|---|---|
| Bottom-right alignment, right-side bands, sliding windows | scalar setup, the KV loop's start; the per-score compare exists without them | ≤ 9 VGPRs, ~10 SGPRs; time within noise on every shape | runtime |
| Keyless-row guard (O = 0, LSE = −inf) | one select per KV step, one in the epilogue | 0 VGPRs; time within noise | runtime |
| LSE output (`lse_on`) | epilogue only | 0 VGPRs; time within noise | runtime |
| Idle-wave skip (rows past `seq_len_q`) | one branch per KV tile | removing it doubles decode time and slows ragged tails; it costs a few percent on small f16 non-causal prefill (`head_dim` 64, S ≈ 1K) — over the gate on that one class | kept: the cost moves to the decode instance family when one exists, which takes the skip with it |

### 6.2 Instance ledger

Objects per generic family; each family's set is packed into every member arch's
shard, so the shipped count per arch is the same as the authored count.

| Op | Axes (compile-time) | Objects | Status |
|---|---|---|---|
| RMSNorm forward | dtype (2) × {generic N, N ∈ 3072/3584/4096/5120/8192} | 12 | shipped, `gfx11-generic` |
| SDPA forward | dtype (2) × head_dim {64, 96, 128} × causal (2) | 12 | shipped, `gfx11-generic` |
| **Total, gfx11-generic** | | **24** | |
| SDPA `head_dim` 256 | dtype (2) × causal (2), own schedule | +4 | planned |

Adding a row here, with its gate result in §6.1 where it introduces a runtime
feature, is part of adding the feature.
