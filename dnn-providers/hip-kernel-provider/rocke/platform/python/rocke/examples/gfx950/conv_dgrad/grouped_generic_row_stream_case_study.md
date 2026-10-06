<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Grouped dgrad, generic kernel: row-stream knobs (gfx950)

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.

**Kernel.** `build_direct_conv` (`DirectConvSpec`), the generic row-streaming
kernel the grouped direct-MFMA dgrad runs on the transposed problem
(`make_dgrad_fprop_spec`) for every admitted shape that is not the 4c row:
`cpg` / `kpg` multiples of 4 up to 32, 1x1 to 7x7, fused weight transform.

**Dispatch.** `_direct_dgrad_stream_knobs` in
`library/dispatch/grouped_convolution.py` adds the knobs below to fused
generic picks inside a measured region: square channel groups (`cpg == kpg`),
16-column strips (`block_q` 16), output channels that do not split into two 16-wide tile halves, the
image box of `_direct_dgrad_stream_box_admits` (untiled images up to 64
output columns for every channel group; 1x1 filters on H-tiled images from 12
channels per group), and at least 8 image row bands (the XCD
order applies). Every other pick keeps the
previous spec (`_DIRECT_DGRAD_STREAM_KNOBS = False` restores the previous
pick everywhere; `_DIRECT_DGRAD_STREAM_SPLIT_M = True` opts the 32-channel
picks into the `waves_m = 2` stack).

## What limited the kernel

The kernel streams one input row at a time through a double-buffered LDS row,
computes all `KH` output rows that use it, and flushes the finished output
row. Hardware counters against a hand-written reference on the 16- and
32-channel group shapes showed that it was not MFMA-bound and not simply
latency-bound (a deeper register prefetch alone did nothing). Four memory-path
effects remained:

1. **Output stores in the MFMA C layout.** Each lane stores 8 bytes and one
   store instruction covers 16 pixels a whole channel row apart, so the
   kernel issued twice the cache write requests of a 16-byte-per-lane store
   over the workgroup's contiguous channel span.
2. **LDS bank conflicts on the fragment reads.** A staged column of
   `block_groups * cpg` 16-bit elements that is an even number of 16-byte
   units puts the 16 q-lanes of a fragment read on few banks. The same effect
   made `block_groups = 4` (a 64-channel span) slower instead of faster.
3. **Row barrier waits for memory.** The per-row `s_barrier` came with a
   `vmcnt(0)` wait, so every row waited for its own prefetched loads and the
   previous row's output stores.
4. **Each pixel line read by two XCDs.** With 64 bytes of a 128-byte pixel
   line per workgroup and neighbouring group tiles launched round-robin across
   XCDs, every line was fetched into two L2s.

On the 32-channel groups a fifth one: one wave held both 16-wide output tiles'
preloaded weights and accumulators, so the kernel ran at a low wave count.

## Levers (each verified before it was timed)

| Knob | Mechanism | Decision |
| --- | --- | --- |
| `stage_out` | Finished row through a double-buffered LDS tile (16-byte column pad), stored with 16-byte lanes; one barrier per row, shared with the row barrier. Works with H tiles (the global row offset is a runtime product, masked past the image) | kept where `kpg % 16 == 0` |
| `lds_pad` | 8 extra elements per staged column: an odd number of 16-byte units per column | kept, derived per spec: only where the column is an even number of 16-byte units (on an odd one it only adds LDS and conflicts) |
| `lds_only_sync` + `prefetch_rows` | The row barrier drains LDS only; rows are loaded two ahead so loads and stores stay in flight across it | kept as a pair, `prefetch_rows = 2` (3 gave no further robust gain and costs registers) |
| `xcd_tiles` | Launch-order remap: all `q_tiles * group_tiles` workgroups of an image row band on one XCD, so both halves of every pixel line share an L2 | kept; the chunk is derived from the grid (`direct_conv_xcd_chunk`), not hand-set; needs at least 8 row bands |
| `waves_m` | Two waves per group, one per 16-wide output tile half, sharing the staged row | knob kept, dispatch opt-in only (see below) |
| `wide_rows` (16-byte row loader) | Row copies with 8-channel vectors | dropped: net loss once the pad and staged stores were in |
| `block_groups = 4` | 64-channel span like the reference | not adopted: with the pad it ties the XCD remap, without it hits the bank cliff |

Only the knob combination pays off on the 16-channel class: the XCD remap
alone did nothing while the stores and banks were the limit, and the row
barrier change alone did little.

## Two accounting fixes the stack needed

* **LDS footprint.** `direct_conv_lds_bytes` assumed the staged weight slice
  of `dgrad_weights_lds` overlays both row buffers. The LDS pool packer
  reuses a dead slot once: the first row buffer takes the weight slice's slot,
  the second opens a new one. The function now follows the packer (and counts
  the `stage_out` tiles); `test_lds_bytes_match_the_lowered_pool` compares it
  with the `@smem_pool` size of the lowered kernel.
* **Register budget.** `_preload_weight_vgprs` ignored `waves_m`, so every
  output wider than 32 channels was rejected even with the tiles split over
  waves. The budget now counts per wave.

## Dispatch rule

Fused generic picks with `cpg == kpg`, `block_q` 16, `waves_m = 2` not valid,
inside the image box of `_direct_dgrad_stream_box_admits` (see "Image box"
below) and `xcd_tiles` valid take `prefetch_rows = 2` with `lds_only_sync` and
`xcd_tiles`; every other pick keeps the previous spec. `lds_pad` and
`stage_out` add LDS, and on grids sized to exactly one launch round at the
old occupancy (for example 2048 single-wave workgroups at 8 per CU) the extra
LDS pushes a handful of workgroups into a nearly empty second round. The rule
therefore drops them (`stage_out` first) when `_direct_dgrad_lds_rounds`
(workgroups over CUs times the LDS- or wave-limited workgroups per CU)
increases. The guard is conservative: on a few H-tiled shapes the full stack
would still have been faster despite the extra round; it never picks a spec
slower than the previous one.

Fitting method:

1. Per-arm kernel screens on the fused generic picks of the earlier review
   cohorts (layer by layer: barrier pair, pad, staged stores, row loader, XCD
   order, `waves_m`), with LDS and register counts recorded for every build.
2. The rule implemented in the dispatcher, timed against the knob-off pick on
   the same cohort (dispatch level, interleaved arms, order alternated per
   repetition).
3. A disjoint random hold-out draw of fused generic shapes, same procedure;
   the lowest ratios re-measured with more repetitions.

**`waves_m = 2` is opt-in.** The first form of the rule also gave the stack,
on top of `waves_m = 2`, to every pick whose output channels split into two
16-wide halves (32-channel output groups). A review found that form behind
the previous pick on small batches (one to a few images) with many groups,
warm above all: few row bands leave the XCD remap off, and both waves' stores
and shared row reads stay on the uncoordinated memory path. The same stack
without `waves_m` lost on some of those shapes too, so neither form was
admitted there: those picks keep the previous spec, and
`_DIRECT_DGRAD_STREAM_SPLIT_M` opts them into the `waves_m = 2` stack (it won
on the large-batch shapes measured, a gain left for a future, separately
validated region). The `waves_m` spec knob stays available to explicit specs,
sweeps and the benchmark.

**The XCD order and square groups are required.** Fresh cohorts during the
same review round (small batches of many-group images, groups with more input
than output channels or the reverse) found the stack without `waves_m` behind
the previous pick too: warm on 2 to 4 images of 56 rows with 192 to 512
groups, where fewer than 8 row bands leave the XCD remap off, and warm or cold
on several 20- and 24-output-channel groups with 8 or 16 input channels.
Every pick with the XCD order on square groups won. Rather than another
threshold, the stack is now admitted only on square groups with the XCD
order; a dense hold-out over 4- to 24-channel square groups (8 to 512
groups, 1 to 256 images, 1x1 to 7x7) then showed it ahead on every problem
but one 20-channel 3x3 layer on 4 images, slightly behind cold on
re-measurement (ahead warm). The excluded picks keep the previous spec (a missed gain
on non-square groups that did win).

**16-column strips only.** The next review timed the admitted picks on
32-column strips (`block_q` 32, wide H-tiled images) separately and found the
bare form of the stack (prefetch, LDS-only barrier and XCD order, with the
rounds guard having dropped `lds_pad` and `stage_out`) behind the previous
pick on some 8-, 12- and 20-channel groups, warm and cold, most of all on
20-channel groups. Every 16-column-strip pick, of every channel count, was
ahead. The stack is now admitted only on 16-column strips; 32-column picks
keep the previous spec (a missed gain on the 16- and 24-channel 32-column
picks that did win).

**Image box.** A later review sampled the admitted 16-column picks along
image width and found a reproducible pocket behind the previous pick, warm and
cold: untiled images (`block_h` 0) of 20-, 24- and 28-channel groups (two groups
per workgroup) at 128 and 256 output columns, 1x1 and 3x3, plus a weaker
8-channel 1x1 H-tiled image with cold caches. Neither the prefetch pair nor the
XCD order alone recovered it. A dense cohort then crossed channel groups 8 to
32 with output widths 64 to 256, 1x1 and 3x3, 8 to 32 rows, many and few
images, and a second one extended it to 512 columns and 64 rows. The losses
sat on untiled images and grew with width and height: for the
non-power-of-two groups (12 to 28 channels, 40 to 56 channels per workgroup
row) from 80 columns on tall images and above all at 128 and 256 columns; for
the power-of-two groups (4, 8, 16) only at 256 columns and beyond. On H-tiled
images every 3x3 pick and every 1x1 pick of 12 or more channels per group won,
while 8-channel 1x1 picks were marginal. `_direct_dgrad_stream_box_admits`
therefore keeps:

* untiled images: every channel group up to
  `_DIRECT_DGRAD_STREAM_UNTILED_MAX_WO` (64) output columns, the width at which
  every measured group and height (8 to 64 rows) won. A wider cap for
  16-channel groups was tried and dropped: a later review found losses at 128
  columns where the launch-rounds rule removes the LDS pad, and on single-row
  1x1 images from about 176 columns;
* H-tiled images: every group for 3x3 and larger filters; 1x1 filters from
  `_DIRECT_DGRAD_STREAM_TILED_1X1_MIN_CPG` (12) channels per group.

Every other pick keeps the previous spec (a missed gain on the untiled widths
between the cap and the losing widths, where many of those picks did win).
`test_generic_row_stream_box_edges` pins both edges.

Where the stack is admitted, apart from that layer no shape that changed pick was reproducibly
slower than the previous pick. Only fused generic picks change; the 4c row,
the pre-pass pipeline, igemm and depthwise are untouched.

## Engines

The generic kernel now has a C++ mirror of every path and knob
(`rocke_build_direct_conv`; spec, validator, LDS / register / grid helpers in
`conv_direct_grouped_spec_and_validate.cpp`). The validator rejects every
unsupported combination with the same text in both engines; parity configs
61-72 of `conv_direct_grouped_emit.{py,c}` cover the default-knob paths and
the knob stacks, and `lower_conv_direct_grouped(spec, kind="generic")`
forwards every field.

## Correctness

dX pre-filled with NaN, NaN counted as a failure, fp64 / fp32 references:

* `test_direct_mfma_dgrad_correctness.py`: dispatched stream shapes (odd
  H / W, partial q strips and H tiles, 5x5 fused gathers; the opt-in
  `waves_m` stack with and without the guard), a knob grid on small adversarial problems (each knob
  alone and stacked, fused gathers, LDS-staged weights, pre-pass), and the
  CPU checks (LDS footprint vs the lowered pool, validator reasons, names,
  per-wave register budget).
* `test_conv_direct_grouped_backend_parity.py`: both engines, IR and `.ll`,
  every field forwarded, identical rejects.
* `tests/dispatch/test_grouped_conv_wgrad_dispatch.py`
  (`test_generic_row_stream_knobs_policy`): the picks per class, the
  `waves_m`-valid picks unchanged by default and the opt-in stack, the rounds
  guard, the pre-pass and knob-off picks unchanged.

## Replay

```bash
cd rocke/library
# Knob grid on one shape: every fused point with and without the stack (+st).
python benchmarks/common/benchmark_direct_conv.py --direction dgrad \
    --dgrad-family fused --verify --N 128 --Hi 14 --Wi 14 \
    --C 512 --K 512 --groups 32 --dtype bf16
# Dispatch picks with and without the stack (set the flag in a REPL):
#   import dispatch.grouped_convolution as gc
#   gc._DIRECT_DGRAD_STREAM_KNOBS = False   # previous picks everywhere
#   gc._DIRECT_DGRAD_STREAM_SPLIT_M = True  # opt in to the waves_m = 2 stack
python -m pytest tests/test_direct_mfma_dgrad_correctness.py \
    tests/test_conv_direct_grouped_backend_parity.py \
    tests/dispatch/test_grouped_conv_wgrad_dispatch.py -k "stream or generic or knob"
cd ../platform && python tools/check_byte_identity.py --only conv_direct_grouped
```

## Not done here

* Async (`buffer_load ... lds`) row loads with a swizzled LDS row instead of
  register staging and a pad.
* Fitting `stage_out` on the LDS-bound 32-channel kernels without an extra
  launch round (stage the weights in two halves, or reuse the dead weight slot
  for the output tiles).
* The pre-pass pipeline's main kernel (non-fused) does not take the stack.
