# Conv Backward Data (dgrad) — Implicit-GEMM Instance

Computes the input gradient of a 2-D convolution:

```
dX[n, hi, wi, c] = sum_{y, x, k} dY[n, ho, wo, k] * W[k, y, x, c]
```

## GEMM Orientation

| Dim | Expression | Operand |
|-----|-----------|---------|
| M | `N * Hi * Wi` | rows of dX |
| N_dg | `C` | cols of dX |
| K_dg | `Y * X * K` | reduction |

- **A** = `dY` (NHWK) — output gradient
- **B** = `W` (KYXC) — weight tensor
- **D** = `dX` (NHWC) — input gradient (output of this kernel)

## Tilde Decomposition

For stride > 1 the backward convolution decomposes into
`y_tilde × x_tilde` independent sub-GEMMs, where:

```
y_tilde = sH / gcd(sH, dH)
x_tilde = sW / gcd(sW, dW)
```

For stride=1, dilation=1: `y_tilde = x_tilde = 1` → single sub-GEMM.

### Why tilde decomposition?

When stride > 1 not all `(hi, y)` pairs produce a valid output row `ho`:

```
ho = (hi + pH - y * dH) / sH   (must be an integer and in [0, Ho))
```

The tilde decomposition partitions the filter positions `y` into `y_tilde`
groups such that within each group the integrality constraint is always
satisfied. Each group becomes one independent sub-GEMM.

## Pipeline Variants

| Pipeline | Description |
|----------|-------------|
| `mem` | Single-buffer LDS, synchronous loads, no scheduler hints. Default. |
| `wavelet` | Load/math wave specialization for **gfx1250/WMMA only**. Extra `num_load_waves` waves handle all DRAM→LDS transfers while the `warp_m × warp_n` math waves run WMMA exclusively. Requires gfx1250's separate VMEM and WMMA issue slots to achieve true hardware concurrency. Incompatible with `async_dma=True`, `split_k > 1` and `lds_k_outer=True`. Single-buffer LDS shared by both roles; synchronization via a `barrier_0 / barrier_A / barrier_B` protocol. |

On MFMA targets `compv3` / `compv4` are schedule policies only (hint placement, `s_setprio`); the K loop stays single-buffered and is charged one LDS buffer. WMMA dgrad accepts only `mem` and `wavelet`. `async_dma`, `unroll_k` and `chiplet_swizzle` are rejected by `is_valid_dgrad_spec`: the builder implements none of them.

## Kernel Architecture

All convolutions — stride=1 and strided — use a **single unified tiled kernel**:

- The host packs per-sub-GEMM constants into `sub_gemm_buf` (a flat `i32` array).
- Each CTA binary-searches the buffer to find its sub-GEMM and loads record fields.
  With exactly one sub-GEMM (stride 1) the record is folded into immediates
  instead (`static_sub_gemm`, default on): no search, no record loads; the
  buffer stays in the ABI, unread. The ungrouped pointwise problem is the
  exception: its descriptors are already divide-free, so it keeps the runtime
  record (a constant trip count only invites a K-loop unroll that costs it).
- The K-loop uses runtime descriptor closures that compute `dY` and `W` offsets
  from the record's coefficients. On the stride-1 path with `kpg % tile_k == 0`
  it runs as (filter tap outer) x (output-channel chunk inner) (`tap_outer_k`,
  default on), with the pixel decode loop-invariant, the dY predicate
  tap-invariant, and the tile's global reads batched ahead of its LDS writes.
  See `platform/python/rocke/examples/gfx950/conv_dgrad/stride1_igemm_dgrad_case_study.md`.
- An accumulator tile above 256 fp32 registers per lane is rejected (it fills
  the whole register file). This is a correctness guard, not a no-spill
  guarantee: some 256-accumulator configs still spill.
- A flat K loop (no tap-outer loop) with more than 128 fp32 accumulators per
  lane keeps the runtime record: the folded build spills there.
- On gfx950, the flat K loop of a folded record with 16-byte dY/W loads gets
  a derived `waves_per_eu = (2, 6)` when the spec sets none
  (`flat_fold_acc_waves_per_eu`), but only on the two dispatch warp tiles
  (64x64x64 with 2x2 warps of 32x32x16, and 128x128x64 with 2x2 warps of
  16x16x32; `_ACC_HINT_TILES`): it selects the VGPR-form MFMA and removes a
  per-iteration AGPR <-> VGPR accumulator copy. On other tiles the same hint
  speeds some configs up and slows others down or makes them spill, with no
  accumulator-count, atom, warp-count or epilogue rule that separates them,
  so explicit configs keep the backend default. The ceiling of 6 keeps the
  scheduler from serializing staged loads to reach full occupancy.
- On gfx950, the flat K loop of a folded record without that hint batches the
  tile's global reads ahead of its LDS writes, as the tap-outer loop does
  (`_FLAT_FOLD_BATCH_ARCHES`): with per-vector load -> store pairs the
  scheduler serialized each load behind its store, and the fold then ran
  slower than the runtime record on narrow-load problems (grouped
  `cpg % 8 != 0`, dense `cpg % 4 != 0`). Other targets keep the pairs.
- On gfx950, a grouped problem with a folded record and no split-K launches
  its `(group, tile)` space in XCD-contiguous order
  (`xcd_contiguous_tile_order`): the launch-order linear id is remapped with
  `chiplet_transform_chunked`, one chunk per XCD, so each XCD runs whole groups
  and fetches each group's dY and weights into its own L2. Launch order spreads
  every group over all XCDs, and on many-group problems whose operands
  outgrow the L2 the refetch, not the K loop, set the kernel time. Ungrouped
  problems and the runtime record keep launch order, and so do 1x1 problems
  with a single N tile (`cpg <= tile_n`): there no dY row is shared between
  tiles, the order removes little traffic, and launch order was faster with
  cold caches. Python only (the C++ builder refuses grouped dgrad).
- On the tap-outer path of a stride-1 problem whose output has the input's
  size (`Ho == Hi`, `Wo == Wi`) and a multi-tap filter, `dy_halo` swaps the
  loop to output-channel chunk outer, filter tap inner and reuses one staged
  dY halo tile per chunk for every tap. See [dY halo reuse](#dy-halo-reuse-dy_halo).
- **Epilogue dispatch** based on `needs_atomic`:
  - `False` (1 sub-GEMM, split_k=1): direct `buffer_store` into `dX`.
  - `True` (stride > 1 or split_k > 1): `global_atomic_fadd` into `dX`
    (caller must zero-initialise `dX` before launch).

### Sub-GEMM record layout (22 × i32)

| Field | Index | Description |
|-------|-------|-------------|
| `block_start` | 0 | first flat tile index for this sub-GEMM |
| `num_m_tiles` | 1 | M-tile count |
| `num_n_tiles` | 2 | N-tile count |
| `gemm_m` | 3 | `N * HTildeSlice * WTildeSlice` |
| `gemm_k` | 4 | `YDotSlice * XDotSlice * K` |
| `h_tilde_slice` | 5 | HTildeSlice |
| `w_tilde_slice` | 6 | WTildeSlice |
| `h_tilde_slice_begin` | 7 | HTildeSliceBegin |
| `w_tilde_slice_begin` | 8 | WTildeSliceBegin |
| `y_dot_slice` | 9 | YDotSlice |
| `x_dot_slice` | 10 | XDotSlice |
| `a_embed_h_coeff` | 11 | `ho = htl + h_begin + ydot * coeff_h` |
| `a_embed_w_coeff` | 12 | `wo = wtl + w_begin + xdot * coeff_w` |
| `b_y_stride` | 13 | `y = ydot * b_y_stride + b_y_offset` |
| `b_y_offset` | 14 | |
| `b_x_stride` | 15 | `x = xdot * b_x_stride + b_x_offset` |
| `b_x_offset` | 16 | |
| `d_h_stride` | 17 | `hi = htl * d_h_stride + d_h_offset` |
| `d_h_offset` | 18 | |
| `d_w_stride` | 19 | `wi = wtl * d_w_stride + d_w_offset` |
| `d_w_offset` | 20 | |
| `gemm_k_padded` | 21 | padded K for split-K |

## Kernel ABI

```
(dY, W, dX, dY_bytes, W_bytes, dX_bytes, sub_gemm_buf, num_sub_gemms)
```

All kernels — stride=1 and strided — share this 8-param ABI. For stride=1
`sub_gemm_buf` holds exactly one record and the binary search trivially
returns index 0.

## Grid Layout

```
grid = (flat_tiles, 1, split_k)
```

where `flat_tiles = sub_gemms[-1].block_end` (sum of all sub-GEMMs' tile counts).

## Split-K

When `split_k > 1` the K reduction is partitioned across `split_k` Z-grid CTAs.
The caller must zero-initialise `dX`. Supported dtypes:

- `fp32` — scalar `global_atomic_add` (f32 fadd)
- `bf16` — packed `global_atomic_fadd_v2bf16` (`<2 x bfloat>`)
- `fp16` — packed `global_atomic_fadd_v2f16` (`<2 x half>`)

## Vector Loads

Both A and B tiles are loaded via `CoalescedTileLoader` with dtype-aware vector
widths. The widths are derived from `DgradConvSpec.default_vector_sizes(C, K, dtype)`
which returns `(vec_a, vec_b, vec_c)`.

### A (dY, NHWK)

`k_out` is the innermost index of the GEMM-K decomposition, which maps
contiguously onto the last dim of `dY` (dim K). Vector width is therefore
constrained by `K % load_vec_a == 0`. Split-K forces `load_vec_a = 1` because
the per-CTA K-slice boundary may not be K-aligned.

The loader uses `vector_axis="col"` (the standard column-axis path).

### B (W, KYXC)

The GEMM row axis is `N_dg = C` (input channels), which is the **stride-1** axis
of `W` in `KYXC` layout. Vector loads therefore go along the free (row) axis and
the loader transposes the tile into row-major LDS layout on store — exactly the
same mechanism used by wgrad for its B operand (`X`, `NHWC`).

That transpose-on-store is what `lds_k_outer` removes; see
[LDS Tile Layout](#lds-tile-layout-lds_k_outer) below. Under `lds_k_outer=True`
this tile is stored K-outer and the scatter disappears, but the width selection
below is unchanged.

Constraint: `C % load_vec_b == 0`. The loader uses `vector_axis="row"`.

Width selection (Python / C++):

1. Compute `max_from_C` — largest power-of-two dividing `C` up to 8 (fp16/bf16)
   or 4 (fp32) from `default_vector_sizes`.
2. Call `CoalescedTileLoader.choose_vec(tile_rows=block_n, tile_cols=block_k, ...,
   max_vec=max_from_C, vector_axis="row")`.
3. If `spec.vector_size_b` is set explicitly, use that; else use the chosen value
   if `> 1`, otherwise fall back to 1 with `vector_axis="col"`.

### D (dX, NHWC)

The epilogue writes `dX` whose last dim is also `C`. Store vector width follows
`C % store_vec == 0`, derived from `default_vector_sizes` (third element).

## LDS Tile Layout (`lds_k_outer`)

`lds_k_outer=True` stores the **B tile only** K-outer (`LDS[k][n]`, row stride
`block_n + _KOUTER_PAD` with `_KOUTER_PAD = 8`) and recovers the MFMA/WMMA
operand layout with transpose reads, instead of transposing on store.

**Why B only.** This is a deliberate asymmetry with wgrad, which flips both
operands:

- **B (`W`, KYXC)** has the GEMM free axis `c` stride-1, forcing
  `vector_axis="row"` and a per-element `ds_write_b16` scatter whose inter-lane
  dword delta is a multiple of the 32-dword bank period. This is the cost worth
  removing.
- **A (`dY`, NHWK)** already has a stride-1 reduction axis (`k_out` innermost),
  so its loader is already `vector_axis="col"` with one wide `smem_store_vN`, and
  its M-outer fragment read is already conflict-free via `lds_k_pad`. Flipping A
  would put the global vector along `m = (n, hi, wi)` — stride `K` in NHWK — and
  destroy coalescing for zero write-side gain.

**Instruction effect on the B side.** The store collapses to one wide
`smem_store_vN` (`b128` for a 16-bit 8-wide vector), dropping `load_vec − 1`
address adds and `load_vec` `vec_extract`s per chunk. The read pays `n / 4`
`ds_read_b64_tr_b16` (wave64) or `n / 8` `ds_load_tr16_b128` (wave32) for a
per-lane fragment length `n`, where the M-outer path issued a single
`smem_load_vN`: **+1** read per fragment on the `n = 8` atoms (`32x32x16`,
`16x16x32`), exactly **zero** on the `n = 4` atoms (`16x16x16`, `32x32x8`).
Global `buffer_load`s are unchanged.

**Gating.** Two regimes, `_LDS_K_OUTER_ARCH_WAVE = {"gfx950": 64,
"gfx1250": 32}`, each arch pinned to its wave size so a mismatched spec is
rejected rather than emitting a lane formula the hardware does not implement.
gfx950 wave64 admits `warp_tile_n ∈ (16, 32)`; gfx1250 wave32 admits only the
`16x16x32` atom (whose 16-element fragment is two `ds_load_tr16_b128` reads).
Plus 16-bit B. Rejected with `async_dma=True` (the tilde builder has no direct
global→LDS path) and with `pipeline="wavelet"` (`build_wavelet_loaders` pins the
B tile to `(block_n, block_k)` and takes the unswapped descriptor, so it would
write M-outer into a K-outer allocation).

**Not a knob.** The spec field defaults `False`; the value is deduced by
the keyword-only `DgradConvSpec.default_lds_k_outer(*, arch, dtype_b,
warp_tile_n, cpg, wave_size=64, pipeline="mem")`, which both library dispatch
(`library/dispatch/grouped_convolution.py`,
`_dgrad_lds_k_outer`) and the sweep driver call. The predicate is
asymmetric with wgrad's — B-side dtype and warp tile only, never the A-side
counterparts — and additionally keys on `cpg`: the saving is proportional to the
B load width, which collapses to 1 on an odd channel run, where `axis_b` is
already `"col"` and there is no scatter to remove.

## dY halo reuse (`dy_halo`)

On a stride-1 problem whose output has the input's size, filter tap `(y, x)`
reads dY pixel `m + (pH - y)*Wo + (pW - x)` for dX pixel `m` in the linear
`(n, h, w)` order. Every tap of a tile of `tile_m` consecutive dX pixels
therefore reads one contiguous dY range: the tile plus `(Y-1)*Wo + (X-1)`
halo pixels. The tap-outer loop re-gathers dY for every `(tap, chunk)`; with
`dy_halo` the K loop runs output-channel chunk outer and filter tap inner
(unrolled), loads that range into LDS once per chunk, and serves each tap from
it at a constant row shift.

| Field | Default | Effect | Name tag |
|-------|---------|--------|----------|
| `dy_halo` | `0` | `1`: staged halo, B (W) loaded per `(chunk, tap)` with two barriers per tap. `2`: plus a double-buffered B tile whose next-tap global reads are issued before the current tap's MFMAs and stored after them, one barrier per tap. The prefetch is fenced with `sched_barrier` on both sides; without the fence the scheduler sinks the loads next to their LDS store and nothing overlaps. | `halo1` / `halo2` |
| `dy_halo_2d` | `False` | Stage the halo 2-D with a zero border, `(tile_m/Wo + Y-1)` image rows of `(Wo + X-1)` pixels: the loader zero-fills out-of-image pixels and no tap needs a mask. Needs a tile of whole image rows of one image (`tile_m % Wo == 0`, `Hi % (tile_m/Wo) == 0`). | `h2d` |
| `dy_halo_setprio` | `0` | `s_setprio` level (1..3) around each tap's MFMA block. | `hprio<n>` |
| `dy_halo_kouter_pad` | `0` | Row pad of the K-outer B tile instead of `_KOUTER_PAD`, a multiple of 8 (16-byte rows for the wide LDS store and the transpose read; an unaligned pad gave wrong dX). A wider pad removes transpose-read bank conflicts but costs LDS. | `hkp<pad>` |

In the 1-D layout a lane whose shifted pixel leaves the image (row wrap, image
edge, batch edge) reads a trailing all-zero LDS row instead: one select on the
LDS row index per fragment row. The halo tile is padded to whole loader passes
(`dy_halo_lds_rows`); the LDS charge (`dgrad_lds_bytes`) and the allocation
read the same shape. A non-default `lds_k_pad` is tagged `kp<pad>` in the
kernel name, because it changes the A tile layout.

The validator rejects every combination that would be ignored or is unsafe,
with the same reason text in both engines: `dy_halo` without the tap-outer
loop (stride 1, dilation 1, folded record, `kpg % tile_k == 0`, no split-K,
wave64), without a same-size output or on a 1x1 filter; `dy_halo_2d` on an
ineligible tile; the companion knobs without `dy_halo`; a pad that is not a
multiple of 8, or without `lds_k_outer`; and a halo that outgrows the LDS (it
grows by `(Y-1)*Wo` rows, so wide images and large filters can).

**Dispatch.** gfx950 dispatch (`_gfx950_dgrad_halo_pick`) takes `dy_halo=2`
on problems the validator admits it on, except where the rules below keep the
tile table. One-dimensional filters (3x1, 5x1, 7x1, 1x3, 1x5, 1x7: `X == 1`
or `Y == 1`) always keep it: on vertical-only ones the halo pick won on some
problems and lost on others on every tile and grid size, with no pattern in
dY reuse, grid size or channel counts, and 1x3 lost on wide images on the
256x64 tile. It
picks one of three tiles by grid size and LDS occupancy: 64x64
`w2x2` for small grids, 128x64 `w4x1` for mid-size grids, 256x64 `w4x1` for
large grids. On mid-size grids with at most 9 taps, 256x64 is also used when
its LDS occupancy matches the 128x64 tile's and it either has more than one
workgroup per CU or the per-tap halo `(Y-1)*Wo + (X-1)` covers at least two
128x64 tiles. The 4x1-wave tiles add
`dy_halo_setprio=1`, and `dy_halo_kouter_pad=32` only where the pad keeps the
LDS-limited workgroups per CU. A pick that fits one workgroup per CU on a grid
that needs more falls back (256x64 to 128x64, otherwise to the tile table).
The tile table is also kept when the picked tile's dY reuse is too low. Reuse
is `taps * tile_m / (tile_m + (Y-1)*Wo + (X-1))`, how often each staged dY
row feeds an MFMA. A filter with a tall halo on a very wide image has a reuse
near 1: the halo then stages about as many rows as the tap-outer loop gathers
and only adds per-chunk staging. The floor is 1.3 on the 4x1-wave tiles; on
the 64x64 tile it is 1.0 on grids of more than one workgroup per CU and 0.7
on smaller grids.
The tile table is also kept where it has the 128x128 tile, its N tiles cover
the group's input channels exactly in one or two tiles (`cpg` 128 or 256), and
the halo pick would be the 128x64 tile at two or fewer workgroups per CU: the
128x64 tile reads each dY row twice as often, and on wide images it measured
slower (at `cpg` 256 with cold caches). Filters with more than 81 taps keep the tile table too: the
halo loop unrolls every tap, so code size and compile time grow with it. The
4x1-wave tiles (128x64, 256x64) take at most 49 taps
(`_GFX950_DGRAD_HALO_4X1_MAX_TAPS`): 9x9, 7x9 and 9x7 filters that would pick
them keep the tile table (they lost with cold caches on large weights); the
64x64 tile keeps them.
`dy_halo_2d` is not selected by dispatch. The C++ builder mirrors the loop for
the ungrouped layout; grouped halo builds are Python only, like every grouped
dgrad build. See `platform/python/rocke/examples/gfx950/conv_dgrad/stride1_igemm_dgrad_case_study.md`.

## Key Files

| File | Purpose |
|------|---------|
| `conv_implicit_gemm_dgrad.py` | Python builder (this instance) |
| `../../benchmarks/common/benchmark_implicit_gemm_conv.py` | `--direction dgrad` sweep |
| `../../builders/common/conv_reference.py` | `dgrad_reference()` via `torch.nn.grad.conv2d_input` |
| `../../../platform/cpp/instances/common/conv_implicit_gemm_dgrad.cpp` | C++ port (byte-identical) |
| `../../../platform/cpp/include/rocke/instance_conv_implicit_gemm_dgrad.h` | C99 header |
| `../../tests/parity/conv_implicit_gemm_dgrad_emit.{c,py}` | C-vs-Python parity emitters |

Library dispatch reaches this kernel through the gfx950 candidate in
`../../dispatch/grouped_convolution.py`: the dY halo pick
`_gfx950_dgrad_halo_pick` where it applies, otherwise the shape-keyed table
`_gfx950_dgrad_tile`.

## Differences from Wgrad

| Aspect | Wgrad | Dgrad |
|--------|-------|-------|
| Number of GEMMs | 1 | `y_tilde × x_tilde` |
| GEMM-M | `K` | `N * HTildeSlice * WTildeSlice` |
| GEMM-N | `Y*X*C` | `C` |
| GEMM-K | `N*Ho*Wo` | `YDotSlice * XDotSlice * K` |
| A operand | `dY` (NHWK) | `dY` (NHWK) |
| B operand | `X` (NHWC) — reuses fwd A desc | `W` (KYXC) |
| Output | `dW` (KYXC) | `dX` (NHWC) |
| Tilde decomposition | Not needed | Required for stride > 1 |
| Output accumulation | Atomic only when split_k > 1 | Atomic when num_sub_gemms > 1 OR split_k > 1 |
| `lds_k_outer` scope | Flips **both** A and B | Flips **B only** (A already has a stride-1 reduction axis) |
| `lds_k_outer` + `async_dma` | Required together | Mutually exclusive |
