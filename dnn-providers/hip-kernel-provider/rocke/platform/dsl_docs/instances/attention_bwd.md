# Attention Backward (dQ, dK, dV): layout-general family contract

Status: draft. The spec, legality rules, kernarg ABI, host plan, exact
predicate, workspace formulas and declaration exist. `build_attn_bwd_main`
emits the main kernel for batched lengths (fixed, or padded with per-batch
`SEQ_LEN_Q` / `SEQ_LEN_KV` counts: exact-zero gradient rows past each
length, per-batch bottom-right diagonal, zero-length batches) and for ragged
THD sequences (`[B + 1]` int32 or int64 offset tables, `token = off * mult /
div` in 64 bits, optional `min` with the SEQ_LEN counts, zero-length
sequences including the last one, rows outside the sequences untouched, the
workspace rows indexed by storage token under the caller's `max_total_q` /
`max_total_kv`, or by sequence slot in `B * S_max` rows without them), with a
head-major or token-major workspace, direct or
atomic dK / dV and `stage_vec = 8` or `1` (16-byte or single-element user
tensor accesses: any B/H/S stride and any element-aligned base), with no mask or the runtime band (top-left / bottom-right
per sequence, two-sided windows, inverse-band q-tile range), `edge_tiles`
(interior q tiles without per-cell selects, run in the same accumulation
order) and `head_pack` (the query heads of a group share the M rows of one q
tile when `G * len_q <= kM0`). In direct mode a CTA owns one kv head and sums the
dK / dV contributions of its `G = h_q / h_k` query heads on chip (MHA, GQA,
MQA, packed QKV views). In atomic mode (`dkv_mode = "atomic"`: `h_k != h_v`,
or a runtime `g_split > 1`) a CTA owns `G / g_split` query heads of a unit of
`G = gcd(h_q / h_k, h_q / h_v)` heads (one K head and one V head) and adds its
fp32 dK / dV into the `WS_DK` / `WS_DV` workspace with atomics; two converts
follow. The CTA decode applies the runtime load-balance order (natural or
reverse, one kv tile per CTA); mirror pairing (two kv tiles per CTA) is not
built and the host plan refuses it, and the XCD remap kernargs are not read
yet (identity order). Every other knob value raises `NotImplementedError`
until its functional group is built. The shipped policy declares the `base`,
`masks_and_stats`, `grouped_heads`, `split_kv_heads`, `padding`, `ragged`
and `narrow_access` groups (every functional group), and no performance class
is measured yet, so gfx942 / gfx950 requests are declined `PERF_UNVERIFIED`
and gfx1151 and gfx1201 serve the cells of those seven groups (see
section 7).

Files (relative to the rocke root):

| File | Content |
|---|---|
| `library/kernels/common/attention_bwd.py` | `AttnBwdSpec`, per-arch default geometry, legality rules, register and LDS estimates, kernel names, `attn_bwd_params` (ABI lists), `build_attn_bwd_main` (CTA decode, addressing, q loop, dK / dV epilogue) |
| `library/kernels/common/_attention_bwd_body.py` | shared inner body: `BwdTileGeometry`, `BwdTileStep` (G0..G4, selects, dQ sink; edge and interior steps; head-packed steps), `StatsView`, the flat LDS realisation of the phase plan |
| `library/kernels/common/_attention_bwd_worklist.py` | CTA work decode: kv tile, head unit and split, batch; load-balance order; query / K / V heads of a unit (direct and atomic mode) |
| `library/kernels/common/attention_bwd_run.py` | launch glue: `run_attn_bwd` (prep, main, convert on one stream; caller-owned workspace, no allocation, no memset) |
| `library/kernels/common/attention_bwd_plan.py` | `AttnBwdRequest`, `attn_bwd_support` (predicate), `attn_bwd_plan`, `attn_bwd_workspace_bytes` / `_layout`, `attn_bwd_declaration`, routing and performance classes, `AttnBwdPolicy` |
| `library/tests/test_attention_bwd.py` | spec, legality, register model, ABI lists, builder signature |
| `library/tests/test_attention_bwd_plan.py` | predicate (every code reachable, accept rows), case-table coverage, workspace, declaration agreement, no device query, feature groups, routing |

Everything in the plan module is host-only: no device query, no allocation,
no kernel build. Targets are named by gfx id only.

## 1. Statistics contract (LSE)

* Input statistic: **natural-log** log-sum-exp, **fp32**, one value per query
  row and query head. Accepted forms (`lse_dims` / `lse_strides` of the
  request; `()` dims are the canonical dims of the layout, `()` strides their
  packed strides):
  * dense (batched, padded): `[B, H_q, S_q, 1]` with rank-4 `(b, h, t, 1)`
    strides; packed: `(H_q * S_q, S_q, 1, 1)`;
  * THD: `[T_q, H_q, 1]` (any `T_q`) with rank-3 `(t, h, 1)` strides; packed:
    `t = H_q`, `h = 1`;
  * THD: the batched shape `[B, H_q, S_q, 1]` with explicit rank-4
    `(b, h, t, 1)` strides whose B stride is unused (tokens are located
    through the ragged offsets, so the token stride is the S stride). This
    shape has no packed default.

  Every stride is a non-negative integer; the stride of the trailing extent-1
  dim never moves an address and may be any such value. The stride tuple has
  the rank of the dims. Any other form (another rank, other dims, a negative
  or non-integer entry, the THD batched shape without strides) is declined
  `LSE_FORMAT` with the reason in the detail; no form is replaced by a
  default.
  A log2 LSE is wrong input (it gives wrong gradients; the tests include it as
  a negative control).
* Dead rows: a row is dead when `!(lse > -inf && lse < +inf)` (so NaN is
  dead too) or when the row is outside the sequence. The forward marks fully
  masked rows with `-inf`. On a dead row the backward produces `dQ = 0` and the
  row contributes nothing to dK / dV, whatever `O` holds on it.
* The prep kernel converts once: `lse2 = lse * log2(e)` with a `+inf`
  sentinel on dead rows, and `Dsum = select(row_live, rowsum(dO * O), 0)`.
* Only this contract couples the backward to a forward: any forward that
  stores `O` and a natural-log fp32 LSE as above (including one from an
  earlier release) is consumable. `library/tests/test_attention_bwd_fwd_chain.py`
  checks this on device statistics: O / LSE from the forward extension-seam
  probe (inline LSE epilogue and the `_lse_store` epilogue, the latter with
  the rows past `len_q` left unwritten) and from a frozen fixture
  (`library/tests/sdpa/bwd_fixture_v1.py`, never regenerated in place).

## 2. Problem coverage (functional matrix of the family)

| Axis | Values |
|---|---|
| dtype | fp16, bf16 for every tensor (Q, K, V, O, dO, dQ, dK, dV); fp32 accumulation |
| head size | 32, 64, 128 (`d_qk == d_v`); 256 declined `HEAD_DIM_256_BWD` |
| layout | any non-negative B/H/S element strides with D stride 1 (BSHD, BHSD, packed QKV views), per tensor |
| lengths | fixed; padded (`SEQ_LEN_Q/KV` with `padding_mask`); THD (`[B+1]` int32 or int64 offsets, multiplier) |
| heads | MHA, GQA, MQA; `h_k != h_v` (atomic dK/dV) |
| masks | none, causal top-left / bottom-right, sliding window (left/right, either alignment), deprecated causal booleans |
| `s_q = 1` | served by the general kernel with head packing |
| scale | host f32 (default `1/sqrt(d)`); pass-by-value scale tensors resolved on the host |

Strides that are not multiples of 8 elements, or base alignment in `[2, 16)`
bytes, select the narrow global-access instance (`stage_vec = 1`); LDS, MMA,
workspace and ABI are identical to the 16-byte instance. The narrow instances
default to the narrow tiles (16 x 64, 32 x 64 at d32, four waves, plain LDS
transpose on gfx942 / gfx950; the one-wave 16 x 16 tile on RDNA).

## 3. Spec and knobs (`AttnBwdSpec`)

`head_size` is the only required field; every other field is defaulted, and a
`None` knob is resolved by `attn_bwd_default_geometry(arch, head_size, dtype,
stage_vec, seq_mode)` (plus a per-arch AGPR reservation on MFMA targets).

* Problem class fields: `dtype`, `seq_mode` (`batched`/`thd`), `dkv_mode`
  (`direct`/`atomic`), `stage_vec` (8/1), `mask_class` (`band`/`none`),
  `dq_mode` (`atomic`; `split` is rejected until the split dQ kernel exists).
* Tile and residency: `waves`, `block_m`, `block_n`, `block_k4`,
  `warp_grid_g4`, `warp_grid_g02` (default `(1, W)`), `warp_grid_g13`
  (default `(W, 1)`; both may be `(2, W/2)` only with `pt_route = "lds"`),
  `atom_g02`, `atom_g13`, `atom_g4`, `pt_route`,
  `kv_residency`, `kt_source`, `transpose_source`, `ring_depth`, `global_path`.
* Schedule and codegen: `lds_swizzle` (canonical `buffer=value` string),
  `sched`, `setprio`, `waves_per_eu`, `scheduler_strategy`, `edge_tiles`,
  `head_pack`, `acc_in_lds`, `ws_layout`, `agpr_alloc`, `s_in_agpr`.

Not yet effective (`NOT_YET_EFFECTIVE_KNOBS`): `block_k4` (the G4 dS key
slicing is not built; G4 reads the whole `kN0` dS tile, and the value is read
only by the stage-table scheduler, which is not built) and `s_in_agpr` (the
value enters the register estimate only; the backend places the S / dP
accumulators). Both are accepted, validated and salted into the kernel name,
and the emitted IR equals the default's apart from the name (a test pins
this). They stay in the catalog; the sweep funnel compiles one representative
per configuration that differs only in them.

Static legality (`validate_attn_bwd_spec(spec, arch)`, raises `ValueError`
before any IR is emitted): catalog facts per arch (waves, MMA atoms, transpose
read, LDS DMA, ring depth), geometry divisibility (`block_n % (16*waves)`,
`block_k4` against the G4 atom and `block_n`, G4 warp grid against `D` and
`block_m`, the one-wave tile only at 16 x 16), couplings (`kt_source = "reg"`
needs register K/V; DMA needs `stage_vec = 8` and excludes the XT transpose;
DMA buffers read by the transpose read (`q`, `do`, `k`) keep the `xor`
swizzle; `waves_per_eu = 3` only on gfx942 with at most two waves;
`setprio` only with a stage table; `s_in_agpr` vs `agpr_alloc` vs
`acc_in_lds`), the LDS footprint estimate within the per-arch capacity, and
the two-budget register estimate (arch VGPRs and AGPRs, their sum under
`waves_per_eu`; a static prune allowance applies before the compiled resource
notes decide). An explicit `agpr_alloc = (n, n)` must leave at least 24 arch
VGPRs in the per-wave share of the unified file, `512 / max(ceil(W / 4),
waves_per_eu)`: `(256, 256)` on an eight-wave tile (256 registers per wave)
is refused, as it fails codegen.

Runtime plan values are never spec fields: `g_split`, scale placement
(`ds_mult`, `dk_mult`, `dq_mult`, convert `mult`), load-balance order
(`lb_order`, `pair`), XCD remap (`xcd_n`, `xcd_chunk`), `pack_heads`,
`use_worklist`. Kernel names are salted by every non-default compile-time
field.

## 4. Kernarg ABI `rocke.attn_bwd.v3`

All pointers are global, not `noalias`; inputs are `readonly`; unused
pointers are 0 and never dereferenced. Order is exact (`attn_bwd_params`):
pointers, then i64, f32 and i32 scalars (every pointer is 8 bytes, so the i64
block is 8-byte aligned and the packed bytes need no padding):

```
main   ptr : Q K V dO dK dV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV SEQ_Q SEQ_KV OFF_Q OFF_KV WORKLIST
       i64 : q_b q_h  k_b k_h  v_b v_h  do_b do_h  dk_b dk_h  dv_b dv_h
       f32 : scale_log2 ds_mult dk_mult dq_mult
       i32 : q_t k_t v_t do_t dk_t dv_t
             h_q h_k h_v G gk gv  S_q_max S_kv_max  has_len len_stride  off64 q_mult q_div kv_mult kv_div
             left right bottom_right  ws_rows_q ws_rows_kv ws_seg
             g_split n_kv_tiles lb_order pair xcd_n xcd_chunk pack_heads use_worklist
prep   ptr : O dO LSE SEQ_Q SEQ_KV OFF_Q OFF_KV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV WORKLIST
       i64 : o_b o_h  do_b do_h  l_b l_h
       i32 : o_t do_t l_t  h_q h_k h_v  S_q_max S_kv_max
             has_len len_stride  off64 q_mult q_div kv_mult kv_div  ws_rows_q ws_rows_kv ws_seg
             zero_kv zero_dq use_worklist n_batch wl_kn0
convert ptr : SRC DST SEQ OFF
       i64 : d_b d_h
       f32 : mult
       i32 : H  d_t  S_max  has_len len_stride  off64 tok_mult tok_div  ws_rows ws_seg
dq     ptr : Q K V dO dQ WS_LSE2 WS_DSUM SEQ_Q SEQ_KV OFF_Q OFF_KV            (split dQ kernel, not built)
       i64 : q_b q_h  k_b k_h  v_b v_h  do_b do_h  dq_b dq_h
       f32 : scale_log2 ds_mult dq_mult
       i32 : q_t k_t v_t do_t dq_t
             h_q h_k h_v G  S_q_max S_kv_max  has_len len_stride  off64 q_mult q_div kv_mult kv_div
             left right bottom_right  ws_rows_q ws_seg  xcd_n xcd_chunk
```

Every v2 scalar keeps its name and meaning in v3; v3 moves the batch and head
strides to i64 (a tensor may span `2^31` elements or more between batches or
heads; the kernels already rebase in 64 bits) and adds `ws_seg` (THD
workspace rows by sequence slot, section 5). Token strides stay i32: the
predicate bounds `S_max * stride_t`. Every request the predicate admits packs
its kernargs without narrowing (`pack_attn_bwd_args` raises, naming the
kernarg, if a value does not fit its type). Strides are in elements; THD
tensors ignore the `*_b` strides. `G = h_q / h_k` in direct mode, `gcd(h_q/h_k, h_q/h_v)` in atomic
mode. Ragged element offsets become tokens as `off * mult / div` in 64-bit
integer arithmetic (`div` = the tensor's token stride).

Launch sequence (one stream): `prep -> main -> convert(dQ)`, plus
`convert(dK)` and `convert(dV)` in atomic mode.

## 5. Workspace

`A(x) = round_up(x, 256)`; every array is fp32; sub-buffers are packed in this
order at 256-byte aligned offsets (`attn_bwd_workspace_layout`):

```
rows_q, rows_kv  : fixed / padded: B * S (the tensor dims)
                   THD with both max_total_q and max_total_kv: those bounds
                     (rows indexed by storage token, ws_seg = 0)
                   THD otherwise: B * S_q_max, B * S_kv_max
                     (rows indexed by sequence slot b * S_max + i, ws_seg = 1)
WS_DQ    = A(4 * rows_q * h_q * D)                 dq_mode == "atomic"
WS_LSE2  = A(4 * rows_q * h_q)
WS_DSUM  = A(4 * rows_q * h_q)
WS_DK    = A(4 * rows_kv * h_k * D)                h_k != h_v or g_split > 1
WS_DV    = A(4 * rows_kv * h_v * D)                same condition
WORKLIST = A(8 * (B + ceil(rows_kv / kN0)))        work list only (not built)
workspace_bytes = sum of the above
```

Worked example: B = 2, S_q = S_kv = 128, h_q = h_k = h_v = 8, D = 128, fixed
lengths, `g_split = 1`: `1,048,576 + 8,192 + 8,192 = 1,064,960` bytes; with
`g_split = 2` add `2,097,152` bytes.

The reported size is an upper bound for every admitted request without a
device query. Without both caller bounds (hipDNN has no max-total API), the
rows of THD sequence `b` are its slot `[b * S_max, b * S_max + len_b)`, and
`len_b <= S_max`, so tokens before the first sequence, between sequences
(SEQ_LEN counts shorter than the allocated segments), after the last one, and
empty segments never push a row past `B * S_max`. With both bounds, the rows
are the storage tokens `[tok0_b, tok0_b + len_b)`, and the caller asserts
that each bound covers the end token of every segment (the last storage token
used plus one, not the sum of the lengths); a single bound is not used.

No hidden allocation: the plan reports the size before execution; the kernels
allocate nothing and the runtime issues no memset (prep initialises every
workspace byte that is later read); shipped instances have no scratch and no
spills. Caller contract: the workspace base is 256-byte aligned; an
understated THD `max_total` gives wrong gradients for the excess tokens but no
out-of-bounds access (every workspace row is clamped to `ws_rows - 1`). A
supplied `max_total_q` or `max_total_kv` must be at least 1 for a non-empty
problem (declined `RAGGED_FORMAT` otherwise), so `ws_rows >= 1` always holds.

## 6. Predicate and reason codes

`attn_bwd_support(req, arch, *, policy=None) -> Verdict(ok, code, detail)`
is a pure function of the request (and the in-repo policy tables). The first
failing check wins, in this order:

| # | Code | Decline when |
|---|---|---|
| 1 | `ARCH_UNSUPPORTED` | arch not in {gfx942, gfx950, gfx1151, gfx1201} |
| 2 | `ARCH_NOT_VALIDATED` | the arch is in the policy's `not_validated_archs` (empty in the shipped policy: every target has a hardware run) |
| 3 | `DROPOUT`, `BIAS_IN_BACKWARD`, `DBIAS`, `ALIBI`, `PAGED`, `FP8`, `DETERMINISTIC`, `UNSUPPORTED_FEATURE` | the feature is requested (in this order) |
| 4 | `SCALE_DEVICE_TENSOR` | scale is a device-buffer tensor |
| 5 | `DTYPE` | Q, K, V, O, dO not all fp16 or all bf16 |
| 6 | `OUT_DTYPE` | dQ, dK, dV differ from the input dtype |
| 7 | `LSE_FORMAT` | stats not fp32 or not in a form of section 1 (dims, stride rank, a negative or non-integer entry; the detail names which) |
| 8 | `DQK_NE_DV` | `d_qk != d_v` |
| 9 | `HEAD_DIM_256_BWD` | d == 256 (out of scope within Tier S per the requirements' "Explicitly out of scope" list; item 9 lists d = 256 for the forward only; the detail cites this) |
| 10 | `HEAD_DIM` | d not in {32, 64, 128} |
| 11 | `GQA_RATIO` | `h_q % h_k != 0` or `h_q % h_v != 0` |
| 12 | `BAND_BOUND` | a bound `< -1` or `>= 2^30`; both causal booleans; a causal boolean with explicit bounds |
| 13 | `STRIDE_D` | a D stride != 1 |
| 14 | `STRIDE_NEGATIVE` | a negative stride |
| 15 | `OUTPUT_OVERLAP` | two distinct indices of dQ, dK or dV share an element (exact: a sorted-extent test proves most layouts disjoint, the rest are decided by enumerating index differences, so interleaved disjoint rows are admitted; a layout beyond the enumeration budget is declined conservatively; zero strides are allowed for inputs only; an output with a zero extent never overlaps) |
| 16 | `STRIDE_ALIGN` | base alignment below the element size (2 bytes) |
| 17 | `INDEX_RANGE` | `S_max * stride_s + D >= 2^31` for a tensor; `S_q * stride_t + 1 >= 2^31` for the LSE; a batch or head stride `>= 2^63` (i64 kernarg) or a largest byte offset `>= 2^63`; `rows_q >= 2^31`; `rows_kv >= 2^31` (atomic mode); `S_max * D >= 2^31`; `S_max >= 2^30` |
| 18 | `GRID_LIMIT` | `B > 65535`; main units (`h_k`, or `h_q / G` in atomic mode) times `g_split` `> 65535`; prep y `> 65535`; prep/convert x `>= 2^31` |
| 19 | `SEQ_LEN_FORMAT` | SEQ_LEN tensors without `padding_mask`; padding without both tensors; not int32 `(B,1,1,1)` |
| 20 | `RAGGED_FORMAT` | THD offsets not int32/int64 `[B+1]`; q-side or kv-side tensors not on one table; multiplier or token stride outside `[1, 2^31)`; a supplied `max_total_q` / `max_total_kv` below 1 for a non-empty problem |
| 21 | `EMPTY` | B, a head count, a sequence dim or D is 0 |
| 22 | `INSTANCE_UNAVAILABLE` | the serving instance failed the resource gate |
| 23 | `PERF_BELOW_BASELINE` | the request's performance class on its serving route is `below_baseline` |
| 24 | `PERF_UNVERIFIED` | the class is `unverified` and the policy does not waive unverified classes |
| 25 | `NOT_YET_DECLARED` | the request needs a functional feature group whose correctness gate has not passed |
| - | `OK` | otherwise |

Accepted without a decline: THD without `max_total` (rows by sequence slot,
`B * S_max`), any storage layout of the ragged tensors (offsets need not
start at 0 or be contiguous), int64 offsets and multipliers, `s_q = 1`,
`h_k != h_v`, any non-negative B/H/S stride (batch and head strides of any
size below `2^63`) and any element-aligned base.

Functional feature groups (in gate order): `base`, `masks_and_stats` (any
mask, explicit scale, d32, `s_q = 1`), `grouped_heads`, `split_kv_heads`,
`padding`, `ragged`, `narrow_access`. `attn_bwd_feature_groups(req)` lists
the groups a request needs.

## 7. Performance classes, routing and the shipping policy

Performance class (request fields only): `(arch, route, d, dtype, mask kind
{none, causal_tl, causal_br, window}, head mode {mha, gqa, hk_ne_hv}, length
relation {square, unequal, s_q_1}, length mode {fixed, padded, thd},
stage_vec)`; for the dense route the class is the exact dense key. The device
hints `num_cus` and `num_xcds` (both default `None`) never enter the class;
they only choose runtime values inside it (grid-fill bucket, mirror pairing,
XCD remap).

Routing (`attn_bwd_route`): a request takes the dense route only when it is
dense-eligible (gfx942/gfx950, contiguous BSHD/BHSD, fixed lengths,
`h_k == h_v`, d 64/128, base alignment >= 16 B) and its exact key is in the
dense pack; otherwise the general route. Routing never adds or removes a
declared cell.

`AttnBwdPolicy` (default: `shipped_policy()`, reading the in-repo tuned-knob
table module when it exists):

* `declared_groups`: functional groups whose correctness gate has passed
  (today: `base`, `masks_and_stats`, `grouped_heads`, `split_kv_heads`,
  `padding`, `ragged`, `narrow_access`);
* `waive_unverified`: `False` declines classes without a measured cell
  (`PERF_UNVERIFIED`); `True` is an explicit, recorded waiver;
* `perf_status`: `{perf_class: "verified" | "unverified" | "below_baseline"}`;
  a missing class is `unverified`; RDNA targets (gfx1151, gfx1201) are
  declared degraded on performance and never declined on performance;
* `not_validated_archs`: targets declined `ARCH_NOT_VALIDATED` (empty: gfx1201
  is declared on its hardware run; its forward-chain tests skip because the
  forward probe shell has no gfx12 body, recorded in the declaration notes);
* `instance_unavailable`, `dense_pack`, and the `tuning` hook
  `(req, arch, route, perf_class) -> {knob: value}` returning spec knobs and
  runtime values (`g_split`, `lb_order`, `xcd_chunk`, `scale_placement`,
  `use_worklist`).

## 8. Declaration

`attn_bwd_declaration(arch, *, policy=None)` returns a JSON-serialisable dict:
`version` (3), `abi` (`rocke.attn_bwd.v3`), `arch`, `status` (`declared`,
`not_validated` or `unsupported`), `rdna_status`, accepted value sets per
axis (`head_dim_declined` carries the code and the requirements reason), the
ordered `reason_codes`, `feature_groups` and `declared_feature_groups`, the
`lse` contract, the `workspace` formula, the `determinism` statement, the
`perf_policy`, `perf` (one entry per (route, class) with its status),
`deferral_register` (the requirements deferral-register rows of this family:
requirement, reason code, decision, rationale and the consequence for hipDNN)
and `notes`.

Deferral-register rows (`DEFERRAL_REGISTER`):

| Requirement | Code | Consequence for hipDNN |
|---|---|---|
| item 9 (d = 256) / item 15 | `HEAD_DIM_256_BWD` | d = 256 models are servable for inference but not trainable through this provider; their training attention stays on CK / AOTriton |
| item 14 (bias, forward only) / item 15 | `BIAS_IN_BACKWARD` | `F.sdpa(attn_mask=...)` under autograd falls back to CK / AOTriton on every target |
| item 15 (no dBias) | `DBIAS` | graphs that need dBias train on CK / AOTriton |
| Tier S (deterministic backward) | `DETERMINISTIC` | a deterministic backward request falls back to CK / AOTriton |

## 9. Determinism

* dQ is accumulated with fp32 atomics: **non-deterministic in the last bits**
  (run-to-run differences stay within the numeric tolerance; not bitwise).
* dK and dV are **deterministic in direct mode** (`h_k == h_v`, `g_split = 1`):
  accumulated on chip and written once.
* In atomic mode (`h_k != h_v`, or `g_split > 1`) dK and dV are added through
  fp32 atomics and are non-deterministic in the last bits.
* A deterministic backward is declined (`DETERMINISTIC`).

## 10. Out of scope (declined)

dBias (`DBIAS`), bias or attention mask in the backward (`BIAS_IN_BACKWARD`),
dropout (`DROPOUT`), deterministic backward (`DETERMINISTIC`), d = 256
backward (`HEAD_DIM_256_BWD`: out of scope within Tier S in the requirements'
"Explicitly out of scope" list, deliberately asymmetric with item 9, which
serves d = 256 in the forward), fp8 (`FP8`), ALiBi (`ALIBI`), paged KV through
this entry (`PAGED`; paged problems belong to the rocKE-native unified entry),
block masks, sink tokens and extra statistics outputs (`UNSUPPORTED_FEATURE`).
