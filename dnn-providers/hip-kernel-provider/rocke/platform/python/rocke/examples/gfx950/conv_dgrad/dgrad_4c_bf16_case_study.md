<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# 4c dgrad on the batched 4x4x4 MFMA — gfx950 case study

Measured numbers are deliberately absent — see `platform/AGENTS.md` §Compliance.
This study records the mechanism, the levers that were swept, the keep/revert
decisions and the commands to replay them.

## What was investigated

Grouped backward-data with four channels per group (`cpg = kpg = 4`), bf16 and
fp16, stride 1, 3x3 "same" padding. Before this change the only direct dgrad
path for that shape was the generic `DirectConvSpec` pipeline
(`make_dgrad_fprop_spec`): a weight transpose pre-pass, then the generic
direct-MFMA kernel on `(dY, W_T)`, which uses the `16x16x16` atom with one wave
per group.

## Finding 1: at cpg = 4 the 16x16x16 atom does one sixteenth useful work

The generic kernel maps one group onto a `16x16x16` MFMA. With four input and
four output channels per group only a 4x4 corner of the M x K operand is real:
the other rows and K lanes are masked to zero (`v_cndmask` on every fragment)
and only 16 of 64 lanes store. Hardware counters on the generic kernel show the
MFMA pipe as the dominant busy unit even though almost all of that work is
padding.

The `DirectConv4cSpec` kernel already solved this for fprop: the wave64
`4x4x4` MFMA computes sixteen independent 4x4x4 products, so one wave covers
sixteen groups with no wasted lanes. It was fp16-only because the IR had no
bf16 twin of the atom.

## Change 1: `mfma_f32_4x4x4_bf16`

`llvm.amdgcn.mfma.f32.4x4x4bf16.1k` exists on CDNA2+ and selects
`v_mfma_f32_4x4x4_16b_bf16` on gfx950. It follows the same `_1k` convention as
`mfma_f32_16x16x16_bf16`: the IR operands are `<4 x bfloat>`, bitcast to
`<4 x i16>` at the intrinsic boundary.

Wired in both engines:

| Piece | Python | C++ |
| --- | --- | --- |
| builder method | `IRBuilder.mfma_f32_4x4x4_bf16` (`core/ir.py`) | `rocke_b_mfma_f32_4x4x4_bf16` (`include/rocke/ir.h`, `core/ir/ir_tile.cpp`) |
| accumulator fragment length | `_MMA_FRAGMENT_INFO` (`core/arch/target.py`) | `rocke_ati_mma_frag` (`core/arch/data.cpp`) |
| intrinsic declaration | `_INTRINSIC_DECLS` (`core/lower_llvm.py`) | `core/lower_llvm/data.cpp` (same table position) |
| LLVM lowering | `_op_tile_mfma_f32_4x4x4_bf16` | `MFMA_SPECS` row with `bitcast_to = "<4 x i16>"` (`core/lower_llvm/mma.cpp`) |
| HIP lowering | `_op_tile_mfma_f32_4x4x4_bf16` (`core/lower_hip.py`) | — (the C++ HIP path has no per-atom MFMA handlers) |

The op goes through the neutral `tile.mma` op with an `op_id` attribute, so no
new opcode enumerator was needed.

The lane map was pinned on hardware before relying on it:
`platform/tests/core/test_mfma_4x4x4_numerics.py` issues one MFMA per dtype and
checks lane `l` (batch `l // 4`, `i = l % 4`): operand `a` is row `i` of
`A_batch`, operand `b` is column `i` of `B_batch`, result slot `m` is
`D_batch[m][i]`. The f16 and bf16 atoms agree.

## Change 2: bf16 in the 4c kernel

`build_direct_conv_4c` now takes its I/O type from `problem.dtype`
(`_io_type`, `_buf_load_vN`, `_buf_store_vN`, `_mfma(b, dtype, "4x4x4", ...)`)
and the kernel name gains a `_bf16` suffix for bf16. The fp16 emission is
unchanged byte for byte (the representative-IR golden for the existing fp16 4c
cases did not move). The C++ mirror carries `io_type` / `is_bf16` in
`rocke_dconv_4c_ctx_t` and branches at the same three emission points.

## Change 3: the 4c dgrad entry

For stride 1, same padding and a 1x1 or 3x3 filter,

```text
dX = conv(dY, W_T),   W_T[c, r', s', k] = W[k, KH-1-r', KW-1-s', c]  (per group),  PAD' = KH-1-PAD
```

so the 4c kernel runs unchanged on `(dY, W_T)`. `make_dgrad_4c_spec` builds the
transposed spec, `build_direct_4c_dgrad` returns `(transpose_kernel,
main_kernel)` (the transpose is the existing
`build_direct_transpose_weights_dgrad`), `direct_4c_dgrad_launch` gives both
launch geometries and the workspace size, and `dgrad_4c_spec_for_problem` is
the dispatch hook (returns `None` when the 4c path does not apply). The 4c
kernel keeps `KH` accumulator slots and one weight fragment per tap in
registers, which is why only `KH in (1, 3)` qualifies.

## Step 0: knob sweep

Knobs: `block_q in {4, 8, 16, 32}` (multiple of 4; the C++ engine bounds
`block_q / 4` by its tile array) and `block_groups in {16, 32, 64}` (one wave
per 16 groups, `groups % block_groups == 0`).

- `block_q = 4` was best or tied on every cpg = 4 shape swept (large and small
  spatial extents, batch 1 and large batch). Larger `block_q` raises live
  accumulators and the unrolled-row code size; the ISA shows VGPR + AGPR use
  growing with `block_q` and the MFMA count per wave scaling with it, with no
  gain in reuse because each q-tile reloads its own input windows.
- `block_groups = 16` and `32` tie; `64` (four waves) was never better and lost
  on one shape. **Kept:** `block_q = 4`, `block_groups = 16`
  (`DGRAD_4C_DEFAULT_BLOCK_Q` / `DGRAD_4C_DEFAULT_BLOCK_GROUPS`); the hook falls
  back to `block_groups = 16` when `groups` is not a multiple of the default.

## Why it is faster (ISA)

Same problem, generic kernel (`block_q 16, block_groups 4`) versus 4c
(`block_q 4, block_groups 16`), per wave:

| | generic | 4c |
| --- | --- | --- |
| MFMA | `v_mfma_f32_16x16x16_bf16` | `v_mfma_f32_4x4x4_16b_bf16` |
| MFMA instructions per wave | equal counts | equal counts |
| waves in the grid | more than four times as many | baseline |
| LDS / barriers | `ds_write_b64`, `ds_read_b64`, `ds_read2_b64`, `s_barrier` per row | none |
| registers | higher VGPR + AGPR | lower VGPR + AGPR, no spills on either |

Both kernels issue the same number of MFMAs per wave, but the 4c grid needs
fewer than a quarter of the waves for the same output, and each 4x4x4 MFMA is a
cheaper instruction than a 16x16x16 one. The counters agree: the MFMA busy
fraction drops from the dominant unit to a minor one, and memory fetch drops
because a 4c workgroup reads 16 groups x 4 channels = 128 contiguous bytes per
pixel instead of 32.

## What still separates it from a single-kernel implementation

Read off the 4c ISA; these are the next levers, not done here:

1. **Weight pre-pass.** The transpose kernel plus its launch gap is still on
   the critical path. Next stage: read `W` with flipped, transposed addressing
   in the 4c prologue (the weights already live in registers for the whole
   kernel).
2. **Input re-loads.** Each output row issues `KW` separate
   `buffer_load_dwordx2` per q-tile — one per filter column — although the
   windows overlap by `KW - 1` columns. A lane-shift (DPP / `ds_bpermute`) or an
   LDS row would load each input once.
3. **Per-load OOB select.** Every input load is followed by a
   `v_cndmask_b32` pair to zero padding; a descriptor-clamped buffer load
   (out-of-range offset returns zero) would remove them.
4. **Fully unrolled H loop.** Code size grows linearly with `H`; a runtime row
   loop with a compile-time accumulator ring would shrink it.
5. **8-byte stores from the MFMA layout.** An LDS-staged 16-byte store would
   cut store instructions.

## Correctness gates

- `platform/tests/core/test_mfma_4x4x4_numerics.py` — on-device lane map +
  numerics for both atoms, LLVM and HIP lowering text.
- `library/tests/test_conv_dgrad_4c.py` — spec plumbing, emitted atom per dtype
  and arch, bf16 4c fprop, 4c dgrad fp16/bf16 on odd H/W, N = 1, W not a
  multiple of `block_q`, the full `block_q x block_groups` grid, and 1x1. Rule:
  an element is bad unless `|out - ref| <= tol + tol * |ref|`, `tol = 1e-2`
  (written that way so a NaN, i.e. an unwritten element of the NaN-filled
  output, counts as bad); zero bad elements required.
- Filter taps are bounded at `KH*KW <= DCONV4C_MAX_TAPS` (Python) /
  `ROCKE_DCONV4C_MAX_TAPS` (C++, which sizes the builder's per-tap `weights[]`
  and `s_consts[]` arrays), checked in `is_valid_spec_4c` on both engines with
  the same reason text, so a directly built 5x5 spec is rejected instead of
  overflowing the C++ arrays. The C++ `is_valid_spec_4c` also gained the
  `stride > 1` reject the Python one already had. LDS weight staging stays
  bounded at `DCONV4C_MAX_WL_PASSES` / `ROCKE_DCONV4C_MAX_WL_PASSES`; with
  the tap bound in place no legal spec reaches it.
- Public dual-engine entry: `rocke.core.backend.lower_conv_direct_grouped`
  flattens the spec for the C++ binding. The flattened dict must carry every
  kernel-shaping field — the problem `dtype` and the 4c
  `dgrad_fused_weights` / `dgrad_weights_lds` knobs — and the binding
  (`fill_direct_conv_problem`, `dg4_build_spec`) must read them. The first
  version of this change wired the knobs only into the binding, so the
  default cpp backend silently emitted the fp16, non-fused kernel for bf16
  and fused specs; the byte-identity gate did not see it because it drives
  the `_emit.c` parity pair, not this path.
  `library/tests/test_conv_direct_grouped_backend_parity.py` runs
  `backend="both"` over 4c fp16/bf16 fprop, 4c dgrad with and without the
  fused transform and LDS staging (gfx950 and gfx942), and 16c fp16/bf16, and
  checks the 5x5 reject on both engines. Removing the forwarded fields makes
  it fail with `BackendMismatch`.
- Byte-identity: `library/tests/parity/conv_direct_grouped_emit.{py,c}` configs
  42–44 (bf16 4c on gfx950 / gfx942, two-wave dgrad shape) and two bf16 4c
  cases in the representative-IR golden.

## Replay

```bash
cd rocke/library
# sweep (4c only), verify every config
python benchmarks/common/benchmark_direct_conv.py --direction dgrad \
    --N 8 --Hi 56 --Wi 56 --C 128 --K 128 --groups 32 --dtype bf16 \
    --dgrad-family 4c --verify --jobs 8
# generic pipeline on the same shape, for the same-session A/B
python benchmarks/common/benchmark_direct_conv.py --direction dgrad \
    --N 8 --Hi 56 --Wi 56 --C 128 --K 128 --groups 32 --dtype bf16 \
    --dgrad-family generic --verify --jobs 8
# tests
python -m pytest tests/test_conv_dgrad_4c.py tests/test_conv_direct_grouped_backend_parity.py
cd ../platform && python -m pytest tests/core/test_mfma_4x4x4_numerics.py
# private scratch + build root: run_diff.py compiles the C emitters under
# $TMPDIR, which concurrent worktrees otherwise share
TMPDIR=<private dir> python tools/check_byte_identity.py \
    --only conv_direct_grouped,target_intrinsics --build-root <private dir>
```

Short kernels should be timed from `rocprofv3 --kernel-trace` rather than
the host-side event timer; compare only within one session.

## Follow-up

The weight-transpose pre-pass listed above as remaining work is removed by the
fused weight transform (`dgrad_fused_weights` / `dgrad_weights_lds`), now the
default of `dgrad_4c_spec_for_problem`; see
`dgrad_fused_weights_case_study.md` in this folder.

## In production dispatch

The grouped-convolution dispatcher selects this kernel (fused weights,
LDS-staged on gfx950) for `cpg == kpg == 4` with a 1x1 or 3x3 filter and
`groups % 16 == 0`, as the `"4c"` variant of `direct_mfma_conv_dgrad`, but only
when the grid at the default `block_q` / `block_groups` has enough workgroups
to fill the device: one wave is one workgroup and the kernel streams whole
columns without H tiling, so on one or two images the generic kernel (which
tiles H) is faster. The floor and how it was measured are in
`grouped_direct_dgrad_dispatch_case_study.md`.

## Row-staged form (`stage_rows`)

**Diagnosis.** With the fused LDS weights in place, hardware counters on the
4c kernel showed the MFMA count equal to the reference grouped kernel's, but
the texture-address unit busy for most of the kernel and the memory unit
stalled: in the 4x4x4 B layout lane `group*4 + column` reads one 8-byte channel
run, so consecutive lanes of every `buffer_load_dwordx2` sit on different
pixels a channel row apart, and each wave issues three loads and one store per
input row.

**Levers tried first (kept off).**

- Register prefetch of the next row (`load_row(y + d)` before row `y`'s
  MFMAs): the waits moved off the critical path in the ISA but the kernel did
  not get faster; enough waves per SIMD already hid the latency.
- An XCD-contiguous tile order (neighbouring q tiles share halo columns):
  within noise.

**Change.** `DirectConv4cSpec.stage_rows` (with `waves_q`): every thread copies
16-byte vectors of the workgroup's input row (`block_q + KW - 1` columns x
`block_groups * 4` channels) into one of two LDS row buffers padded by 64 bytes
per column, one row ahead through registers; each wave then reads its
`KW` fragments per row with `ds_read_b64` and runs the same `KH*KW` MFMAs.
Rows outside the image are skipped, which also drops the MFMAs that only fed
the out-of-image output rows. Per row the global side is one 16-byte load per
thread instead of three strided 8-byte loads per lane.

**Variants swept on the target shape and kept off.** LDS-staged 16-byte output
stores, an LDS-only row barrier, the XCD tile order on top, two or four q
waves per workgroup and 32 groups per workgroup: none beat one wave per
workgroup with `block_q = 4`, `block_groups = 16`, so only `stage_rows` and
`waves_q` were productionised.

**Validation.** `validate()` / `is_valid_spec_4c` (and the C++ mirrors, same
reason text) reject `stage_rows` without the fused LDS weight form,
`block_q != 4 * waves_q`, `waves_q > 1` without `stage_rows`, stride > 1 or
non-"same" padding (the kernel streams output rows 1:1 with input rows and
stores `W` columns), more than 1024 threads, and an LDS footprint over the
target's capacity.

**Dispatch.** The rule in `_select_direct_dgrad_4c_spec` /
`_direct_dgrad_4c_takes_stage_rows` was fitted on a same-session cohort of
`cpg == kpg == 4` shapes (1x1 and 3x3, 16 to 128 groups, images 5 to 112,
grids from a few to tens of thousands of workgroups; arms interleaved with the
order alternated per repetition) against the previous pick, and checked on a
disjoint hold-out draw around the floor. Every shape at or above the 4c grid
floor takes the staged kernel except 1x1 filters on images shorter than 8
rows (and, after the final review below, 3x3 filters on images shorter than 4
rows), where the staged prologue's extra barriers do not pay for so few rows and
the direct-load kernel stays. Below the floor the generic kernel still wins on
images it splits into H tiles (more than 8 rows), and on 3x3 images of at
most 8 rows, which it streams whole, the staged kernel wins down to
`_DIRECT_DGRAD_4C_STAGED_SHORT_MIN_GRID` workgroups.

The first version of the rule also took the staged kernel below the floor on
9- to 16-row images from a slightly larger grid, and on 5-row images from a
smaller one. Both won with warm caches, but a review re-timed them with the
caches flushed before every launch, and there most of them lost to the
previous pick. On 9- to 16-row images the generic kernel's 4-row H tiles give
several times the workgroups of the one-wave 4c grid, and those extra waves
hide DRAM latency once nothing is cached. On the smallest grids the previous
pick is igemm, which also holds up better cold. Both regions were removed. A
probe that forced the staged kernel on every 3x3 shape below the floor (3- to
16-row images, grids from a few dozen to just under the floor, timed cold and
warm) set the bounds: every image taller than 8 rows lost cold below the
floor, except where the generic kernel's own grid crosses a wave-count cliff
(that is a generic-kernel tuning gap, not a reason to route around it), and
5-row images lost on 64-workgroup grids. A separate hold-out draw inside the
new bounds, plus 4c shapes above the floor, did not regress cold or warm. The
dispatch test pins the losing shapes to the previous pick. The measurements
live outside the tree.

A later review timed the short branch by the previous pick's family and image
width: shapes 1 to 3 columns wide that came from igemm lost to it cold (warm
they were level or slightly behind), while every wider shape and every shape
that came from the generic kernel held up. The short branch now also needs
images at least `_DIRECT_DGRAD_4C_STAGED_SHORT_MIN_W` (4) columns wide; a
re-measured cohort of 4- and 5-column shapes from both previous families held
up cold and warm. The 4c floor branch is unchanged.

The next review, and a denser grid over the short branch (1 to 8 rows, 4 to 8
columns, grids from the short floor to just under the 4c floor, both dtypes),
found the staged kernel ahead warm everywhere but only level with the previous
pick cold on images of 4 rows or fewer, and slightly behind cold on some 4-row,
5-column images at the short floor. Those images gain nothing cold, so the
short branch now also needs `_DIRECT_DGRAD_4C_STAGED_SHORT_MIN_H` (5) rows;
1- to 4-row images keep the previous pick, and the dispatch test pins both
sides of the edge.

The final review found the same short-image effect above the floor: 3x3
images of 3 rows and 1 or 2 columns, at grids of several hundred workgroups,
were consistently slightly behind the direct-load 4c kernel warm (level cold),
while 1- and 2-row images were level and images of 4 or more rows won. The
floor branch now also needs `_DIRECT_DGRAD_4C_STAGED_MIN_H_3X3` (4) rows for
3x3 filters; shorter images keep the direct-load 4c kernel, which is exactly
the previous pick, and `test_4c_row_stage_rows_policy` pins both sides.

**Gates.** `library/tests/test_conv_dgrad_4c.py` (`Test4cStageRowsSpec`: name,
thread count, every reject, helper default, fprop unaffected, emitted ops;
`Test4cStageRowsDgrad` and the `staged` mode of the existing grids: on-device
fp16/bf16 with H = 1, W = 1, partial q tiles, 1x1, partial staging passes and
up to four q waves x four channel waves, NaN-filled output);
`test_direct_mfma_dgrad_correctness.py` (dispatched staged shapes below and
above the floor, a taller image below the floor on the generic kernel, and the
1x1 short-image fallback);
`test_conv_direct_grouped_backend_parity.py` (`backend="both"` over staged
specs, every spec field forwarded to the binding, identical reject reasons);
`tests/dispatch/test_grouped_conv_wgrad_dispatch.py`
(`test_4c_row_stage_rows_policy`,
`test_4c_stage_rows_keeps_previous_pick_below_floor`); parity configs 57–60 and the
`4c_dgrad_fwl_sr2_bf16_n2h9` representative-IR case.

**Replay.**

```bash
cd rocke/library
# staged vs direct-load 4c on one shape (the benchmark sweeps both forms)
python benchmarks/common/benchmark_direct_conv.py --direction dgrad \
    --N 128 --Hi 56 --Wi 56 --C 128 --K 128 --groups 32 --dtype bf16 \
    --dgrad-family fused --verify --jobs 8
python -m pytest tests/test_conv_dgrad_4c.py tests/test_conv_direct_grouped_backend_parity.py
cd ../platform
TMPDIR=<private dir> python tools/check_byte_identity.py \
    --only conv_direct_grouped --build-root <private dir>
```

**Remaining gap.** Output stores are still 8-byte MFMA-layout stores, halo
columns are fetched once per XCD, and the padded LDS row still has some bank
conflicts; LDS-staged stores did not pay on the target shape.
