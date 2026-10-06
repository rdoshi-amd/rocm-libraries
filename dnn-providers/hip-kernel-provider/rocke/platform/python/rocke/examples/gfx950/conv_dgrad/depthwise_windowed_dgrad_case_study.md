<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Windowed depthwise dgrad — gfx950 case study

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.
What is recorded here is mechanism, instruction-count evidence, the levers that
were tried (kept and reverted), and the replay path.

Kernel: `build_direct_depthwise_dgrad_windowed` /
`DirectDepthwiseDgradWindowedSpec` in `library/kernels/common/conv_direct_grouped.py`,
C++ mirror `platform/cpp/instances/common/conv_direct_grouped_build_depthwise_dgrad_win.cpp`,
dispatch candidate `direct_depthwise_dgrad_win` in
`library/dispatch/grouped_convolution.py` (gfx950, cpg = kpg = 1, stride 1).

## Starting point

Depthwise dgrad had no dispatch path on gfx950: the implicit-GEMM dgrad candidate
declines `cpg = 1`, and the only fast kernel was the benchmark-only
`build_direct_depthwise_dgrad_streaming`. Its ISA on a 7×7, W = 14 shape showed
three costs that have nothing to do with the arithmetic:

| Symptom (per lane) | Cause |
| --- | --- |
| hundreds of `buffer_load_ushort`, several times the unique dY values | every `(r, s, j)` tap issues its own guarded load |
| over a thousand `v_cndmask_b32` | each tap selects both the address *and* the loaded value, although the OOB sentinel already returns zero |
| tile tail waste | the swept `block_w` values (4/8/16/32) do not divide W = 14 or W = 12 |

It also took minutes to compile a single 7×7 config at `block_w = 8`, and never
finished at `block_w = 32`, which stalled `--direction dgrad` sweeps.

## Lever 1 — one window load per dY row (KEEP)

For dY row `ho`, block columns `[wi0, wi0 + block_w)` need dY columns
`wi0 + PAD - (KW - 1) + t` for `t < block_w + KW - 1`. Load that window once and
read tap `(s, j)` from `window[j + KW - 1 - s]`. Out-of-range columns load through
the sentinel offset and read as zero, so the FMA chain has no select. Loads per
row drop from `KW * block_w` to `block_w + KW - 1`, and the selects collapse
into one per window column. Compile time drops by orders of magnitude because
the unrolled body loses its per-tap address arithmetic.

## Lever 2 — lane/row split addressing with two sentinels (KEEP, second form)

Every dY/dX offset is `lane + row`:

- the **lane** part is the row-invariant `(channel, column)` byte offset,
  computed once per window column, or `DW_DGRAD_WIN_OOB_LANE = 2**30` when the
  channel or column is out of range;
- the **row** part is the block-uniform row offset, or
  `DW_DGRAD_WIN_OOB_UNIFORM = 2**30 - 1` for a padding row of a split-H block.

The IR `add` lowers to `add nsw`, so a single `2**31 - 1` sentinel plus an offset
would be signed overflow (poison). The split sentinels sum to `2**31 - 1`, which
keeps every add in range and every invalid access past the buffer; the validator
therefore caps dY and dX below `2**30` bytes (dispatch falls back above that).

**First form REVERTED.** Folding the *column* validity into the uniform part
made all `(row, column)` offsets scalar: LLVM hoisted every `s_cselect`, ran out
of SGPRs and spilled them to VGPR lanes (`v_writelane` / `v_readlane` storm,
more `s_waitcnt`). Moving column validity into the per-column lane VGPR (one
`v_cndmask` per window column, ever) removed the spills and nearly all selects.

## Lever 3 — software prefetch of the next row + `sched_barrier` (KEEP, hardwired)

With cheap addressing the machine scheduler issued each window load just before
its first use (`load, short FMA run, s_waitcnt`), exposing memory latency; the
result was bimodal across otherwise identical binaries. Issuing the next live
row's window before the current row's FMAs, followed by `sched_barrier(0)`,
spreads the loads through the previous row's FMA block and moves the waits to
row boundaries. It also lowered VGPR use. A two-row lookahead and the no-barrier
variant were measured and dropped. One split-H, `ch_per_lane = 2` configuration
regressed; it is never a selected configuration. Hardwired (no knob) because no
configuration that dispatch selects preferred the old schedule.

## Lever 4 — `dot2` tap pairing (KEEP as knob; dispatch: KW ≥ 5)

7×7 depthwise is bound on the VALU issue rate, not memory. `v_dot2c_f32_bf16` /
`v_dot2c_f32_f16` compute `a.x*b.x + a.y*b.y + c` in one issue, so pairing taps
`(s, s+1)` halves the multiply-add issue count and removes the 16-to-32-bit
widening of every window value (the window stays packed). Window columns are
packed once per row as `(x[t], x[t-1])` (`v_perm_b32`); weights are packed once
in the prologue, with a zero partner for an odd KW.

**Odd-KW tail pair (correctness fix, review round 1).** The first form fed the
last pair `(w[KW-1], 0)` the operand `(x[j], x[j-1])`, so the zero weight met
window column `j-1`, one column outside the receptive field (or a duplicate of
`x[0]`). Numerically harmless for finite data, but an Inf there gave
`0 * Inf = NaN` in a dX column whose true value is finite (or Inf). The tail now
reads `(x[j], 0)`, so the zero weight only ever meets a zero. Three forms were
tried, each verified and measured same-session against the first form:

| Form | ISA (7×7, W = 14 / 16) | Outcome |
| --- | --- | --- |
| `vec_pack(x[j], 0.0)` | one extra `v_perm_b32` per column per row; more VGPRs | REVERT — measurably slower on the W = 16 shape |
| zero-extend the 16-bit bits (`bitcast` → `zext` i32 → `vector.bitcast`) | no pack at all: `buffer_load_ushort` already zero-fills the high half; fewer `v_perm_b32` than the first form (full pairs only need `t >= 2`). But the raw columns stay live across the full-pair `fdot2`s, VGPR rises past 256 on W = 16 and extra `v_mov_b32` appear | REVERT |
| zero-extend + tail pair issued first in the tap loop | raw columns die before the full pairs; VGPRs back under 256, `v_mov_b32` back to the first form's count, fewer instructions than the first form | KEEP — at parity with the first form |

`tests/test_conv_dgrad_depthwise_windowed.py::TestNonFiniteGradients` injects
Inf into dY on the first, an inner and the last column and requires dX to be
finite exactly where the reference is; it fails on the first form (every odd-KW
dot2 case) and passes on the kept one. Parity configs 40 (KW = 1 with dot2: tail
only) and 41 (non-square 3×5 fp16 dot2 with an H split) cover the new paths.

This needed a new IR op, `arith.fdot2`, in both engines (Python `IRBuilder.fdot2`,
`lower_llvm._op_arith_fdot2`; C++ `ROCKE_OP_ARITH_FDOT2` appended to the opcode
enum so no existing opcode value moves, name/purity tables, `rocke_b_fdot2`, LLVM
handler, declaration-table entries at the same relative position as the Python
table). gfx942's backend cannot select `llvm.amdgcn.fdot2.f32.bf16`, so the knob
is gfx950-only in both validators. Neutral on memory-bound 3×3, where dispatch
leaves it off.

## Other levers

| Lever | Outcome |
| --- | --- |
| `block_w` dividing W (whole row up to 16 columns, else about 8) | KEEP — the non-dividing widths are always worse; full-width rows become viable once `dot2` lowers VGPR use |
| `ch_per_lane = 2` (dword channel pairs, `v_pk_fma_f32`) | KEEP for 3×3 with enough channels (memory-bound, wider accesses); loses on 7×7 (VGPR pressure, occupancy 1) and when C is small (idle lanes) |
| `ch_per_lane = 4` | never wins (VGPR) — kept legal, not selected |
| `block_h` split | KEEP for small grids (large H × small N·C); loses on the target shapes because a split block cannot prune padding rows at build time and re-reads the halo |
| `block_waves` | 1–4, chosen so `block_ch` does not exceed C by a whole wave |
| Unroll budget on large filters (31×31, 33×33) | KEEP balanced shrink — rows are first halved down to about one filter height; past that the larger of rows and `block_w` is halved (the first form halved rows only, down to 1, so each block re-read a whole filter-height halo for one output row). Faster on every large-filter shape tried, same unroll size, so no compile-time change |
| Matrix-core (Toeplitz) formulation | KEEP — see Lever 5; the dispatch default inside a measured box of 7×7 layers |

Open lever: with the odd-KW tail fixed, `dot2` on 3×3 (fp16, 1536 channels)
measured slightly ahead of the selected `ch_per_lane = 2` form. Dispatch still
gates `dot2` on KW ≥ 5; widening it needs the 3×3 cohort re-swept first.

Honest losses of the dispatch heuristic against the per-shape sweep optimum are
recorded with the measurements outside the repo; the largest is a 3×3 shape with
192 channels where `block_waves = 1` beats the selected `block_waves = 2`.

## Lever 5 — Toeplitz MFMA form (KEEP; dispatch default inside a measured 7×7 box)

Knobs: `mfma`, `w_fold`, `prefetch_rows` on `DirectDepthwiseDgradWindowedSpec`
(builder `_build_dw_dgrad_win_mfma`, C++ `dw_win_build_mfma` in the same
translation unit as the VALU form). gfx950 only, `groups % 8 == 0`.

**Why.** After `dot2` the 7×7 kernel was still bound on the VALU issue rate
with one wave per SIMD and no latency hiding. A VALU microbenchmark showed
that `v_dot2c_f32_*` and `v_pk_fma_f32` issue at a lower rate than `v_fmac_f32`,
so they save instructions but not multiply-add throughput: the arithmetic
itself had to move to the matrix core.

**Formulation.** Each wave owns 8 channels and one 16×16 fp32 tile per dX row
in flight. Tile row `m = 8q + k` is channel `k` at column parity `q`; tile
column `n` is a column pair of one of `w_fold` images (`32 / w_fold` columns
each). A is a one-hot diagonal weight fragment (`W[k, r, KW-1-s]` at slot `k`
of its 8-wide K group, zero elsewhere), B holds 8 channels × 4 window
columns, so one `v_mfma_f32_16x16x32_{f16,bf16}` covers 4 taps of both column
parities, and a filter row takes `ceil((KW + 1) / 4)` passes. Seven of every
eight products are discarded work; the MFMA pipe still has ample headroom, so
the kernel becomes a memory-access problem. The KH-slot accumulator ring, the
row streaming and the two-sentinel addressing are those of the VALU form.

**Memory path — what was tried** (each step verified, then measured
same-session against the previous one):

| Step | Outcome |
| --- | --- |
| B loaded per lane straight from global (16-byte loads), D stored as 8-byte writes per lane | REVERT — correct but slower than `dot2`: neighbouring lanes and passes re-load the same window columns (TA busy and TCP accesses well above the VALU kernel), and the 8-byte stores double the write requests |
| `b_share`: one load per window column per wave, B fragments built with `ds_bpermute` | REVERT (superseded) — first form ahead of the VALU kernel; dropped once the LDS-staged loads below beat it |
| `d_pack`: lane pairs exchange halves so half the lanes write 16 bytes | REVERT — neutral |
| `xcd_chunk`: remap workgroups so neighbours sharing dY lines share an XCD | REVERT — helps only small workgroups; the selected shapes use large ones |
| deeper prefetch / H split / 4-image fold on the per-lane-load form | REVERT — more dY re-reads; the load path was throughput-limited |
| dX rows staged in an LDS double buffer (one barrier per row), drained as whole 16-byte channel runs per pixel | KEEP (hardwired) — a store-free diagnostic build showed the dX writes were the largest cost; staging turns them into full-line writes |
| dY window rows loaded block-coalesced (16-byte chunks) into a second LDS double buffer written in the same barrier phase; B read with `ds_read_b128` | KEEP (hardwired) |
| `prefetch_rows = 2` (two dY rows in flight) on the staged form | KEEP — ahead of 1 and 3 on the 7×7 shapes |

Only the winning memory path ships: the trimmed knob set is `mfma`, `w_fold`
and `prefetch_rows`; the LDS staging of both directions is implied by `mfma`.
`block_w` and `ch_per_lane` do not apply (`ch_per_lane` must stay 1, `block_w`
is left out of the kernel name, so specs that differ only there build one
kernel), and `w_fold` / `prefetch_rows` are rejected on the VALU form.

**Validator guards** (identical reason text in both engines):

- unrolled MFMAs per wave (`rows_per_block × KH × passes`) at most
  `DW_DGRAD_MFMA_MAX_UNROLL`: a tall image without `block_h` would otherwise
  unroll thousands of MFMAs into one tiny grid;
- `block_waves ≤ 8` (two waves per SIMD, so a wave keeps 256 VGPRs) and a
  register estimate of the resident fragments (one-hot weights, accumulators,
  one row of B, the prefetched rows) at most `DW_DGRAD_MFMA_MAX_FRAG_VGPRS`;
  calibrated against the compiled VGPR count up to 12×11 filters with no
  spills (a 16-wave block spilled at 11×11 before this guard);
- LDS bytes (`mfma_lds_bytes`, equal to the compiled group segment size) at
  most `DW_DGRAD_MFMA_LDS_BUDGET`;
- `prefetch_rows ≤ 4`, `w_fold ∈ {1, 2, 4}`, `groups % 8 == 0`, gfx950.

**Non-finite semantics (accepted).** Finite inputs give the VALU results up to
fp32 summation order. A non-finite dY value does not stay in its channel: the
zero off-diagonal weights give `0 * Inf = NaN` in the other 7 channels of its
group, and the zero-weight taps of the passes widen the receptive field to
`4 * ceil((KW + 1) / 4)` columns: one extra column for KW = 3, 7, 11, two
for KW = 6, 10, three for KW = 5, 9 and four for KW = 4, 8, 12 (which side
depends on the output column parity). Rows stay exact, and nothing spreads
across images. This is inherent to the one-hot formulation, and gradient overflow
checks treat NaN and Inf alike. `TestNonFiniteGradients` keeps the strict
per-channel assertion for the VALU / `dot2` forms and asserts for the MFMA
form that every out-of-tolerance element lies in the 8-channel group of a
non-finite input, inside that widened field, and that no non-finite
reference value comes out finite; a second test places one Inf per image and
group (5×5 to 9×9, odd and even widths) and bounds each one's spread to its
field rows, its group and `4 * ceil((KW + 1) / 4)` contiguous columns. A
single-Inf probe over 3×3 to 11×11 (square and non-square, odd and even) and
every `w_fold` confirmed the extent exactly.

**ISA (7×7, W = 14, 8 waves, `w_fold = 2`).** One MFMA per tap row and pass of
each live row; the multiply-add VALU work disappears; per dY row one barrier,
a handful of `ds_write` / `ds_read_b128` and 16-byte global loads and stores;
no spills, no scratch, about half the VGPRs of the `dot2` kernel; the
instruction count per wave drops several-fold.

**Admission box** (`_dw_dgrad_mfma_admits`). Dispatch gives a request the
MFMA form only inside a box where it was measured ahead of the `dot2` kernel
with warm and with cold caches; every other request keeps the `dot2` pick
unchanged:

- square 7×7 filters with 'same' padding (`pad = 3`), stride 1, dilation 1;
- C a multiple of 64 (128-byte dY pixels), at most 2048;
- N at most 256, and a dY tensor of at most 512 MiB;
- H from 7 to 16 rows;
- W of 7–8 columns (one 8-column fold-4 tile), 13–16 (one 16-column fold-2
  tile) or 19–112.

How the box was found. Earlier forms of the rule admitted every filter of at
least 5 rows and 5 columns and added exceptions (narrow-filter size gates,
channel-alignment gates, holds for one-block-per-CU filters on large grids
and on padded tiles) after each review found cold losses in a corner the
previous probes had not covered: 1×K and 3×K filters, 7×9 / 9×7 / 8×8 / 9×9
layers on large grids, channel counts off a multiple of 64, images just past
32 columns, grids past the launch limit. The rule was then inverted: a dense
probe of 5×5 and 7×7 layers (W from 6 to 112, H from 3 to 112, N from 1 to
256, C from 64 to 2048, fp16 and bf16, every width band with its corners)
timed the MFMA form against the `dot2` pick warm and cold, and the box keeps
only the region where it won throughout:

- 5×5 is close to memory-bound in the `dot2` kernel; the MFMA form won on
  small tensors but lost cold on large ones at most widths, so 5×5 stays on
  the `dot2` kernel (a missed gain on small 5×5 problems);
- images taller than 16 rows are split into row chunks that each re-read a
  6-row dY halo; on large tensors that lost cold at 13–16 columns from 17
  to 24 rows, and broke even on some other widths just past 16 rows, so the
  box stops at 16 rows (a missed gain on tall, wide images such as 28- and
  56-row layers, which won);
- images shorter than the filter (3–6 rows) lost cold on large batches;
- 9–12 columns fill a 16-column tile only partly and lost on large tensors;
  17–18 columns fill a 32-column tile just over half and only broke even;
- channel counts off a multiple of 64 and the other filters stay where the
  earlier reviews put them, on the `dot2` kernel.

A fresh hold-out sample inside the final box (box corners and edges, heavy
tensors, every width from 13 to 29, random interior) then showed the MFMA form
ahead on every shape, warm and cold; samples just outside each edge keep the
previous pick name for name. The NaN / Inf semantics above are unchanged.

**Knob rule** (`_dw_dgrad_mfma_spec`, for requests inside the box; it also
builds the form for explicit requests outside it):

- `w_fold` 4 up to 8 columns, 2 up to 16, 1 up to 32 (the tile columns are
  then mostly real columns); wider images take the fold with the least work
  (`_dw_dgrad_mfma_fold`, ties to the smaller fold): every block multiplies
  a full 32-column tile, empty image slots and padded columns included
  (`ceil(N / w_fold) * ceil(W / tile_w) * 32`), and every real image reads a
  `tile_w + KW - 1` column window per column tile
  (`N * ceil(W / tile_w) * (tile_w + KW - 1)`). A single 32-column tile
  would leave up to half of the second tile empty on images just past 32
  columns, so W = 33 to 48 takes 8- or 16-column tiles whenever there are
  at least two images to fill them; a single image (whose empty image slots
  would cost more), and W = 49 to 64 (two nearly full 32-column tiles), keep
  one image per tile;
- `block_waves = min(8, ceil(C / 8))`, but 4 where an 8-wave block fits only
  once per CU and the 8-wave grid is not a whole number of 256-block rounds
  (`_dw_dgrad_mfma_blocks_per_cu`, the VGPR estimate of the resident
  fragments against the 512-VGPR SIMD file). 7×7 fits two 8-wave blocks per
  CU, so inside the box this only matters for explicit specs of larger
  filters;
- images up to 16 rows stay whole, taller ones start from balanced chunks of
  about 14 rows;
- while the grid has fewer than 256 workgroups (about one per CU): first 4
  waves, then balanced chunks of about 7 rows, and below 160 workgroups about
  4 rows;
- `prefetch_rows = 2`;
- a grid whose y or z extent would pass 65535 keeps the VALU form (outside
  the box anyway; the guard stays for explicit requests).

Every comparison was run twice: warm (back-to-back launches) and cold (a
large buffer cleared before every launch). The gain over the `dot2` kernel is
smaller cold than warm, because the MFMA kernel is memory-bound and the `dot2`
one at 7×7 is not.

Honest losses against the per-shape sweep optimum are recorded with the
measurements outside the repo; they are on mid-sized grids where 8 waves with
a smaller H chunk and 4 waves with a larger one are close (rows-first would
win some of them, waves-first others). A first form of the rule also stepped
down to chunks of about 4 rows on grids that were already mid-sized; that
lost to chunks of about 7 on every such shape re-measured, so the 4-row step
is now limited to grids below 160 workgroups.

**Parity gate:** configs 49-56 in
`library/tests/parity/conv_direct_grouped_emit.{c,py}` (odd N under a 2-image
fold, ragged H split with two W tiles and a partial channel block, 9×9 with a
4-image fold, 11×11 split, 3×3 single pass, non-square 7×5, and a gfx942 and a
`groups % 8 != 0` config both engines reject). No new IR op was needed.

## ISA evidence (per lane, 7×7, W = 14, before → after)

- `v_cndmask_b32`: over a thousand → a few dozen.
- `buffer_load_ushort`: several hundred at `block_w = 8` → fewer at
  `block_w = 14` (window loads only).
- `s_waitcnt`: several hundred → about a hundred.
- multiply-add issue: `v_fmac_f32` per tap → `v_dot2c_f32_bf16` per tap pair.
- instructions per output pixel: less than half.
- no scratch, no VGPR/SGPR spills.

## Replay

```bash
cd rocke/library
export PYTHONPATH=<engine-build>/cpp/bindings:../platform/python:. ROCKE_CPP_STRICT=1

# correctness (adversarial shapes for the VALU and MFMA forms, Inf / NaN
# propagation bounds, dispatch end-to-end), manifest rule bad == 0
python -m pytest -q tests/test_conv_dgrad_depthwise_windowed.py
python -m pytest -q tests/dispatch/test_grouped_conv_dgrad_depthwise_dispatch.py

# sweep (stream + windowed variants, each verified) for one depthwise layer
python -m benchmarks.common.benchmark_direct_conv --direction dgrad --verify \
    --N 128 --Hi 14 --Wi 14 --C 512 --K 512 --groups 512 --Y 7 --X 7 --pH 3 --pW 3 --dtype bf16

# byte identity (Python vs C++ engine), scoped to the family
cd ../platform && ROCKE=$(pwd) PYTHONPATH=$(pwd)/python \
    python tools/check_byte_identity.py --only conv_direct_grouped
```

ISA: compile the dispatched spec with `rocke.compile_kernel` and disassemble the
HSACO with `llvm-objdump -d --mcpu=gfx950`; count `v_cndmask_b32`,
`buffer_load_*`, `s_waitcnt`, `v_fmac_f32` / `v_dot2c_*` and check the
`.vgpr_spill_count` note (`llvm-readelf --notes`).

ABI note: the kernel takes only the six-argument prefix
`(A, B, D, A_bytes, B_bytes, D_bytes)` shared by every conv-grouped candidate;
the implicit-GEMM dgrad kernel appends two more. A launcher sizes the argument
list from the kernel, not from `CONV_GROUPED_ABI_VERSION`.

Test isolation note: the GPU test module imports torch (when installed) before
rocke's launcher so only torch's bundled HIP runtime is loaded; otherwise later
torch-based tests in the same pytest process report no GPUs.

Toolchain note: measured with the LLVM 20 comgr shipped with ROCm 7.1 (the
`llvm22` flavor cannot run on that toolchain); the byte-identity gate was run
for llvm20, llvm22 and llvm23.
