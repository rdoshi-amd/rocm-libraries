/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_implicit_gemm_dgrad_emit.c -- C-side emitter for the
 * implicit-GEMM backward-data convolution parity harness.  Selects one of N
 * sampled spec configs by argv[1] (the config index), builds
 * rocke_dgrad_conv_spec_t identically to the Python emitter
 * conv_implicit_gemm_dgrad_emit.py, builds the kernel via
 * rocke_build_implicit_gemm_conv_dgrad_new and lowers via
 * rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO), printing the .ll
 * to stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_implicit_gemm_dgrad.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

/* Fill the config for index `idx`.  Returns 0 on success, -1 if unknown.
 * On success sets *spec and *arch. */
static int make_cfg(int idx, rocke_dgrad_conv_spec_t* spec, const char** arch)
{
    *spec = rocke_dgrad_conv_spec_default();
    spec->tile_m = 64;
    spec->tile_n = 64;
    spec->tile_k = 64;
    spec->warp_m = 2;
    spec->warp_n = 2;
    spec->warp_tile_m = 32;
    spec->warp_tile_n = 32;
    spec->warp_tile_k = 16;
    spec->pipeline = "mem";
    spec->epilogue = "default";

    switch(idx)
    {
    case 0:
        /* Baseline stride=1, default tile geometry, gfx950 */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        *arch = "gfx950";
        return 0;
    case 1:
        /* Larger tile, compv4 pipeline, gfx950 */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->tile_k = 64;
        spec->pipeline = "compv4";
        *arch = "gfx950";
        return 0;
    case 2:
        /* cshuffle epilogue, gfx950 */
        spec->problem = rocke_conv_problem_make(16, 112, 112, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->epilogue = "cshuffle";
        *arch = "gfx950";
        return 0;
    case 3:
        /* Runtime sub-GEMM record on a stride-1 problem (static_sub_gemm off):
         * binary search + record loads, flat K loop. (Previously async_dma,
         * which dgrad now rejects instead of silently ignoring.) */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->static_sub_gemm = false;
        *arch = "gfx950";
        return 0;
    case 4:
        /* 1x1 filter (no spatial reduction), gfx950 */
        spec->problem = rocke_conv_problem_default(8, 56, 56, 64, 64, 1, 1);
        *arch = "gfx950";
        return 0;
    case 5:
        /* stride=2, pad=1, 3x3 -- tilde path (2x2 = 4 sub-GEMMs), gfx950 */
        spec->problem = rocke_conv_problem_make(4, 28, 28, 64, 64, 3, 3, 2, 2, 1, 1, 1, 1);
        *arch = "gfx950";
        return 0;
    case 6:
        /* stride=2, pad=1, 4x4 -- JIRA ticket scope, gfx950 */
        spec->problem = rocke_conv_problem_make(4, 28, 28, 64, 64, 4, 4, 2, 2, 1, 1, 1, 1);
        *arch = "gfx950";
        return 0;
    case 7:
        /* split_k=2, stride=1, gfx950 */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->split_k = 2;
        *arch = "gfx950";
        return 0;
    case 8:
        /* WMMA wave32, gfx1151 */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->wave_size = 32;
        *arch = "gfx1151";
        return 0;
    case 9:
        /* WMMA wave32, gfx1201 */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->wave_size = 32;
        *arch = "gfx1201";
        return 0;
    case 10:
        /* Folded record with the flat K loop (tap_outer_k off); not a
         * dispatch warp tile, so no waves_per_eu accumulator hint.
         * (Previously chiplet_swizzle, which dgrad now rejects instead of
         * ignoring.) */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->tile_k = 64;
        spec->tap_outer_k = false;
        *arch = "gfx950";
        return 0;
    case 11:
        /* K-outer B tile + ds_read_tr16_b64 transpose read, 32x32x16 atom.
         * cshuffle deliberately: the validator rejects 16-bit dtype_d with the
         * default epilogue, and a rejected config lands as BOTH_REJECTED, which
         * the gate counts as a pass -- it would compare nothing. */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->tile_k = 64;
        spec->warp_tile_m = 32;
        spec->warp_tile_n = 32;
        spec->warp_tile_k = 16;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 12:
        /* Same path on the 16x16x16 atom, where b_frag_len is 4 rather than 8. */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 64;
        spec->tile_n = 64;
        spec->tile_k = 32;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 13:
        /* gfx1250 wave32 WMMA 16x16x32 K-outer. The shared transpose-read
         * helper takes its wave32 branch here and lowers to ds_load_tr16_b128
         * (8 per lane), so a 16-element fragment is two reads. Configs 11 and
         * 12 are both wave64. */
        spec->problem = rocke_conv_problem_make(8, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 32;
        spec->tile_n = 32;
        spec->tile_k = 32;
        spec->warp_m = 1;
        spec->warp_n = 1;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 32;
        spec->wave_size = 32;
        spec->pipeline = "mem";
        spec->epilogue = "default";
        spec->lds_k_outer = true;
        *arch = "gfx1250";
        return 0;
    case 14:
        /* K not a multiple of tile_k: a K tile straddles two taps, so the
         * folded record keeps the flat K loop. */
        spec->problem = rocke_conv_problem_make(2, 14, 14, 64, 48, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 15:
        /* Tap-outer K loop on an odd, non-square, partial-tile problem: N=1,
         * 13x17, pad 2 with a 5x5 filter, C not a multiple of tile_n. */
        spec->problem = rocke_conv_problem_make(1, 13, 17, 48, 64, 5, 5, 1, 1, 2, 2, 1, 1);
        spec->tile_k = 32;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 16;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 16:
        /* Tap-outer K loop with the M-outer B tile, compv3 schedule hints. */
        spec->problem = rocke_conv_problem_make(4, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 128;
        spec->tile_n = 64;
        spec->warp_m = 4;
        spec->warp_n = 2;
        spec->pipeline = "compv3";
        spec->epilogue = "cshuffle";
        *arch = "gfx950";
        return 0;
    case 17:
        /* Ungrouped pointwise (1x1, stride 1, pad 0): static_sub_gemm stays
         * on but the record is not folded (runtime record, opaque trip
         * count); C not a multiple of tile_n. */
        spec->problem = rocke_conv_problem_make(2, 14, 14, 96, 128, 1, 1, 1, 1, 0, 0, 1, 1);
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 18:
        /* Folded record, flat K loop, 4-element dY and W loads (K and C not
         * multiples of 8): the waves_per_eu accumulator hint is withheld. */
        spec->problem = rocke_conv_problem_make(2, 13, 11, 100, 68, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 19:
        /* Folded record, flat K loop, 8-element loads, explicit waves_per_eu:
         * the explicit value wins over the accumulator hint. */
        spec->problem = rocke_conv_problem_make(2, 14, 14, 64, 48, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        spec->has_waves_per_eu = true;
        spec->waves_per_eu = 1;
        *arch = "gfx950";
        return 0;
    case 20:
        /* The large-problem dispatch tile: 128x128x64, 2x2 waves, 16x16x32
         * atom, K-outer B, tap-outer K loop. fp16: the C++ CoalescedTileLoader
         * has no elem_dtype yet, so a bf16 config cannot be compared. */
        spec->problem = rocke_conv_problem_make(2, 16, 16, 256, 256, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 32;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 21:
        /* Flat K loop (kpg not a multiple of tile_k), 8-element loads, 256
         * fp32 accumulators per lane (256x128 tile, 1x2 waves, 32x32 atom):
         * above 128 the record is not folded and the waves_per_eu accumulator
         * hint is withheld (both would add spills). */
        spec->problem = rocke_conv_problem_make(2, 14, 14, 128, 48, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 256;
        spec->tile_n = 128;
        spec->warp_m = 1;
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 22:
        /* Flat K loop, 8-element loads, 128 fp32 accumulators per lane with
         * the 16x16x32 atom on a single warp (64x128 tile, 1x1 waves): the
         * record is folded, but the waves_per_eu accumulator hint is withheld
         * (not a dispatch warp tile; under its 256-register cap this tile
         * spills). */
        spec->problem = rocke_conv_problem_make(2, 14, 14, 128, 48, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_n = 128;
        spec->warp_m = 1;
        spec->warp_n = 1;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 32;
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 23:
        /* Flat K loop, 8-element loads, 64 fp32 accumulators per lane on a
         * warp tile that is not a dispatch tile (256x64x64, 2x2 waves, 32x32
         * atom): the record is folded but the waves_per_eu accumulator hint
         * is withheld (it is applied only to the two dispatch tiles). */
        spec->problem = rocke_conv_problem_make(2, 14, 14, 128, 48, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 256;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 24:
        /* Tap-aligned K, 256 fp32 accumulators per lane (256x128 tile, 1x2
         * waves, 32x32 atom) and an N tile wider than the input channels
         * (C=64 < tile_n): the folded record with the tap-outer K loop (the
         * 128-accumulator fold limit applies to the flat loop only). */
        spec->problem = rocke_conv_problem_make(2, 14, 14, 64, 128, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 256;
        spec->tile_n = 128;
        spec->warp_m = 1;
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 25:
        /* The large-problem dispatch tile (128x128x64, 2x2 waves, 16x16x32
         * atom) on a flat K loop (kpg 96 not a multiple of tile_k): folded
         * record with the waves_per_eu accumulator hint. fp16, see 20. */
        spec->problem = rocke_conv_problem_make(2, 16, 16, 256, 96, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 128;
        spec->tile_n = 128;
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 32;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        *arch = "gfx950";
        return 0;
    case 26:
        /* dY halo reuse (dy_halo=1): one staged 1-D halo tile per
         * output-channel chunk serving all nine taps, two barriers per tap.
         * Odd image, N=2 so a 64-row tile straddles an image boundary, and a
         * partial last M tile. */
        spec->problem = rocke_conv_problem_make(2, 13, 17, 64, 128, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        spec->dy_halo = 1;
        *arch = "gfx950";
        return 0;
    case 27:
        /* dy_halo=2 on the 128x64 4x1-wave dispatch tile: double-buffered B
         * with the pinned prefetch, s_setprio around each tap's MFMAs and the
         * wider K-outer B row pad. */
        spec->problem = rocke_conv_problem_make(2, 16, 16, 128, 128, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 128;
        spec->warp_m = 4;
        spec->warp_n = 1;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        spec->dy_halo = 2;
        spec->dy_halo_setprio = 1;
        spec->dy_halo_kouter_pad = 32;
        *arch = "gfx950";
        return 0;
    case 28:
        /* dy_halo=2 with the 2-D zero-bordered halo (tile_m = 8 whole rows of
         * a 16-wide image): no row masks, the loader zero-fills the border. */
        spec->problem = rocke_conv_problem_make(2, 16, 16, 64, 128, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 128;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        spec->dy_halo = 2;
        spec->dy_halo_2d = true;
        spec->dy_halo_setprio = 2;
        *arch = "gfx950";
        return 0;
    case 29:
        /* dy_halo=2 with the M-outer B tile (no K-outer pad) on a 5x5 filter,
         * pad 2, image narrower than a tile row (9x11, N=1). */
        spec->problem = rocke_conv_problem_make(1, 9, 11, 64, 64, 5, 5, 1, 1, 2, 2, 1, 1);
        spec->epilogue = "cshuffle";
        spec->dy_halo = 2;
        *arch = "gfx950";
        return 0;
    case 30:
        /* dy_halo=2 on the 256x64 4x1-wave tile with a non-default A row pad
         * (lds_k_pad=16, tagged kp16 in the kernel name) and the 1-D halo of
         * a 20-wide image. */
        spec->problem = rocke_conv_problem_make(1, 20, 20, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->tile_m = 256;
        spec->warp_m = 4;
        spec->warp_n = 1;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        spec->has_lds_k_pad = true;
        spec->lds_k_pad = 16;
        spec->dy_halo = 2;
        *arch = "gfx950";
        return 0;
    case 31:
        /* dy_halo=2 on the 16x16x32 atom (4 fragment rows per lane group). */
        spec->problem = rocke_conv_problem_make(1, 8, 8, 64, 64, 3, 3, 1, 1, 1, 1, 1, 1);
        spec->warp_tile_m = 16;
        spec->warp_tile_n = 16;
        spec->warp_tile_k = 32;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        spec->dy_halo = 2;
        *arch = "gfx950";
        return 0;
    case 32:
        /* dy_halo=2 on a 1x7 filter (pad 0x3): a one-row halo per tile, the
         * horizontal taps only, on the 128x64 4x1-wave tile with s_setprio
         * and the K-outer B row pad. */
        spec->problem = rocke_conv_problem_make(2, 12, 20, 64, 128, 1, 7, 1, 1, 0, 3, 1, 1);
        spec->tile_m = 128;
        spec->warp_m = 4;
        spec->warp_n = 1;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        spec->dy_halo = 2;
        spec->dy_halo_setprio = 1;
        spec->dy_halo_kouter_pad = 32;
        *arch = "gfx950";
        return 0;
    case 33:
        /* dy_halo=2 on a 7x1 filter (pad 3x0): vertical taps only, image
         * narrower than the filter is tall (21x9, N=1), default epilogue. */
        spec->problem = rocke_conv_problem_make(1, 21, 9, 64, 64, 7, 1, 1, 1, 3, 0, 1, 1);
        spec->lds_k_outer = true;
        spec->dy_halo = 2;
        *arch = "gfx950";
        return 0;
    case 34:
        /* dy_halo=2 on a 3x5 filter (pad 1x2) on the 256x64 4x1-wave tile:
         * row and column offsets of the taps differ in extent. */
        spec->problem = rocke_conv_problem_make(1, 10, 14, 128, 64, 3, 5, 1, 1, 1, 2, 1, 1);
        spec->tile_m = 256;
        spec->warp_m = 4;
        spec->warp_n = 1;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        spec->dy_halo = 2;
        spec->dy_halo_setprio = 1;
        *arch = "gfx950";
        return 0;
    case 35:
        /* dy_halo=2 with the 2-D zero-bordered halo on a 1x7 filter (tile_m =
         * 8 whole rows of a 16-wide image). */
        spec->problem = rocke_conv_problem_make(2, 16, 16, 64, 64, 1, 7, 1, 1, 0, 3, 1, 1);
        spec->tile_m = 128;
        spec->epilogue = "cshuffle";
        spec->lds_k_outer = true;
        spec->dy_halo = 2;
        spec->dy_halo_2d = true;
        *arch = "gfx950";
        return 0;
    default:
        return -1;
    }
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: %s <config_index>\n", argv[0]);
        return 2;
    }
    int idx = atoi(argv[1]);
    const char* mode = (argc > 2) ? argv[2] : "ll";

    rocke_dgrad_conv_spec_t spec;
    const char* arch = "gfx950";
    if(make_cfg(idx, &spec, &arch) != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = rocke_build_implicit_gemm_conv_dgrad_new(&b, &spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        fprintf(stderr, "build failed: %s\n", m ? m : "(no message)");
        rocke_ir_builder_free(&b);
        return 1;
    }

    int ret = 0;
    if(strcmp(mode, "ll") == 0)
    {
        char* llvm_text = NULL;
        rocke_status_t st
            = rocke_lower_kernel_to_llvm(kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &llvm_text);
        if(st != ROCKE_OK || !llvm_text)
        {
            fprintf(stderr, "lower failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(llvm_text, stdout);
        free(llvm_text);
    }
    else if(strcmp(mode, "ir") == 0)
    {
        char* t = NULL;
        rocke_status_t st = rocke_ir_serialize(kernel, &t);
        if(st != ROCKE_OK || !t)
        {
            fprintf(stderr, "ir_serialize failed: status=%d\n", (int)st);
            rocke_ir_builder_free(&b);
            return 1;
        }
        fputs(t, stdout);
        free(t);
    }
    else if(strcmp(mode, "verify") == 0)
    {
        rocke_diag_t* d = NULL;
        size_t n = 0;
        rocke_verify(kernel, &d, &n);
        for(size_t i = 0; i < n; i++)
        {
            char* s = rocke_diag_to_string(&d[i]);
            if(s)
            {
                puts(s);
                free(s);
            }
        }
        rocke_diags_free(d, n);
    }
    else
    {
        fprintf(stderr, "unknown mode %s\n", mode);
        rocke_ir_builder_free(&b);
        return 2;
    }
    rocke_ir_builder_free(&b);
    return ret;
}
