/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_direct_grouped_emit.c -- C-side emitter for the direct
 * grouped convolution parity harness. Selects one of N sampled spec configs by
 * argv[1] (the config index), builds the rocke_direct_conv_16c_spec_t /
 * rocke_direct_conv_4c_spec_t / rocke_direct_conv_8c_spec_t /
 * rocke_direct_conv_32c_spec_t / rocke_direct_conv_spec_t / rocke_direct_depthwise_spec_t /
 * rocke_direct_depthwise_dgrad_win_spec_t / rocke_direct_conv_wgrad_spec_t identically to
 * the Python emitter conv_direct_grouped_emit.py, builds the kernel via the
 * matching rocke_build_direct_conv_*_new function and lowers via
 * rocke_lower_kernel_to_llvm (per-config arch, flavor AUTO) and prints the .ll
 * to stdout so the two outputs can be byte-compared.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

enum
{
    KIND_16C = 0,
    KIND_4C = 1,
    KIND_8C = 2,
    KIND_32C = 3,
    KIND_DW = 4,
    KIND_SPATIAL = 5,
    KIND_DGRAD = 6,
    KIND_DW_DGRAD = 7,
    KIND_WGRAD = 8,
    KIND_DW_DGRAD_WIN = 9,
    KIND_GENERIC = 10
};

/* Fill the config for index `idx`. Returns 0 on success, -1 if unknown.
 * On success sets *kind, the matching spec struct, and *arch. */
static int make_cfg(int idx,
                    int* kind,
                    rocke_direct_conv_16c_spec_t* s16,
                    rocke_direct_conv_4c_spec_t* s4,
                    rocke_direct_conv_8c_spec_t* s8,
                    rocke_direct_conv_32c_spec_t* s32,
                    rocke_direct_depthwise_spec_t* sdw,
                    rocke_direct_depthwise_spatial_spec_t* ssp,
                    rocke_direct_conv_dgrad_spec_t* sdgrad,
                    rocke_direct_depthwise_dgrad_spec_t* sdw_dgrad,
                    rocke_direct_conv_wgrad_spec_t* swg,
                    rocke_direct_depthwise_dgrad_win_spec_t* swin,
                    rocke_direct_conv_spec_t* sgen,
                    const char** arch)
{
    rocke_direct_conv_problem_t p = rocke_direct_conv_problem_default();
    p.KH = 3;
    p.KW = 3;
    p.PAD = 1;
    p.stride = 1;

    switch(idx)
    {
    case 0:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 16;
        p.kpg = 16;
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 4;
        s16->fold_k32 = true;
        *kind = KIND_16C;
        *arch = "gfx950";
        return 0;
    case 1:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 16;
        p.kpg = 16;
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 8;
        s16->fold_k32 = true;
        *kind = KIND_16C;
        *arch = "gfx950";
        return 0;
    case 2:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 64;
        p.cpg = 4;
        p.kpg = 4;
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->block_q = 4;
        s4->block_groups = 16;
        *kind = KIND_4C;
        *arch = "gfx950";
        return 0;
    case 3:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 64;
        p.cpg = 4;
        p.kpg = 4;
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->block_q = 8;
        s4->block_groups = 16;
        *kind = KIND_4C;
        *arch = "gfx950";
        return 0;
    case 4:
        p.N = 1;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 1;
        s16->fold_k32 = false;
        *kind = KIND_16C;
        *arch = "gfx942";
        return 0;
    case 5:
        p.N = 1;
        p.H = 8;
        p.W = 8;
        p.groups = 16;
        p.cpg = 4;
        p.kpg = 4;
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->block_q = 4;
        s4->block_groups = 16;
        *kind = KIND_4C;
        *arch = "gfx950";
        return 0;
    case 6:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 8;
        p.kpg = 8;
        *s8 = rocke_direct_conv_8c_spec_default();
        s8->problem = p;
        s8->block_q = 16;
        s8->block_groups = 8;
        s8->double_buffer = true;
        *kind = KIND_8C;
        *arch = "gfx950";
        return 0;
    case 7:
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 8;
        p.cpg = 32;
        p.kpg = 32;
        *s32 = rocke_direct_conv_32c_spec_default();
        s32->problem = p;
        s32->block_q = 32;
        s32->block_groups = 4;
        s32->double_buffer = true;
        *kind = KIND_32C;
        *arch = "gfx950";
        return 0;
    case 8:
        /* groups must be divisible by block_ch = block_waves * wave_size (2 * 64 = 128) */
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 128;
        p.cpg = 1;
        p.kpg = 1;
        *sdw = rocke_direct_depthwise_spec_default();
        sdw->problem = p;
        sdw->block_w = 16;
        sdw->block_waves = 2;
        *kind = KIND_DW;
        *arch = "gfx950";
        return 0;
    case 9:
        /* depthwise stride=2: exercises Ho/Wo descriptors and stride-aware flush */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.stride = 2;
        *sdw = rocke_direct_depthwise_spec_default();
        sdw->problem = p;
        sdw->block_w = 8;
        sdw->block_waves = 1;
        *kind = KIND_DW;
        *arch = "gfx950";
        return 0;
    case 10:
        /* spatial layout: groups=3 (non-power-of-two, exercises partial wave) */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 3;
        p.cpg = 1;
        p.kpg = 1;
        *ssp = rocke_direct_depthwise_spatial_spec_default();
        ssp->problem = p;
        ssp->block_waves = 2;
        *kind = KIND_SPATIAL;
        *arch = "gfx950";
        return 0;
    case 11:
        /* spatial layout with stride=2: exercises Ho/Wo + spatial thread mapping */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 3;
        p.cpg = 1;
        p.kpg = 1;
        p.stride = 2;
        *ssp = rocke_direct_depthwise_spatial_spec_default();
        ssp->problem = p;
        ssp->block_waves = 1;
        *kind = KIND_SPATIAL;
        *arch = "gfx950";
        return 0;
    case 12:
        /* dgrad: baseline grouped dgrad stride=1 */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *sdgrad = rocke_direct_conv_dgrad_spec_default();
        sdgrad->problem = p;
        sdgrad->block_q = 16;
        sdgrad->block_groups = 8;
        *kind = KIND_DGRAD;
        *arch = "gfx950";
        return 0;
    case 13:
        /* dgrad: larger groups / different block_groups */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 32;
        p.kpg = 32;
        *sdgrad = rocke_direct_conv_dgrad_spec_default();
        sdgrad->problem = p;
        sdgrad->block_q = 16;
        sdgrad->block_groups = 4;
        *kind = KIND_DGRAD;
        *arch = "gfx950";
        return 0;
    case 14:
        /* dgrad: gfx942 target */
        p.N = 1;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *sdgrad = rocke_direct_conv_dgrad_spec_default();
        sdgrad->problem = p;
        sdgrad->block_q = 16;
        sdgrad->block_groups = 8;
        *kind = KIND_DGRAD;
        *arch = "gfx942";
        return 0;
    case 15:
        /* depthwise_dgrad: stride=1 */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        *sdw_dgrad = rocke_direct_depthwise_dgrad_spec_default();
        sdw_dgrad->problem = p;
        sdw_dgrad->block_w = 8;
        sdw_dgrad->block_waves = 1;
        *kind = KIND_DW_DGRAD;
        *arch = "gfx950";
        return 0;
    case 16:
        /* depthwise_dgrad: stride=2 exercises divisibility checks */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.stride = 2;
        *sdw_dgrad = rocke_direct_depthwise_dgrad_spec_default();
        sdw_dgrad->problem = p;
        sdw_dgrad->block_w = 8;
        sdw_dgrad->block_waves = 1;
        *kind = KIND_DW_DGRAD;
        *arch = "gfx950";
        return 0;
    case 17:
        /* 16c bf16: exercises bf16 I/O, bf16 load/store taps, mfma_f32_16x16x16_bf16 */
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 16;
        p.kpg = 16;
        p.dtype = "bf16";
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 8;
        s16->fold_k32 = false;
        *kind = KIND_16C;
        *arch = "gfx950";
        return 0;
    case 18:
        /* 8c bf16: exercises bf16 I/O, bf16 load/store taps, mfma_f32_16x16x16_bf16 */
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 8;
        p.kpg = 8;
        p.dtype = "bf16";
        *s8 = rocke_direct_conv_8c_spec_default();
        s8->problem = p;
        s8->block_q = 16;
        s8->block_groups = 8;
        s8->double_buffer = true;
        *kind = KIND_8C;
        *arch = "gfx950";
        return 0;
    case 19:
        /* dgrad bf16: exercises bf16 I/O on the scalar-FMA grouped dgrad path */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        p.dtype = "bf16";
        *sdgrad = rocke_direct_conv_dgrad_spec_default();
        sdgrad->problem = p;
        sdgrad->block_q = 16;
        sdgrad->block_groups = 8;
        *kind = KIND_DGRAD;
        *arch = "gfx950";
        return 0;
    case 20:
        /* 16c bf16 with fold_k32=True: pins the non-default fold_k32 path under bf16 */
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 16;
        p.cpg = 16;
        p.kpg = 16;
        p.dtype = "bf16";
        *s16 = rocke_direct_conv_16c_spec_default();
        s16->problem = p;
        s16->block_groups = 8;
        s16->fold_k32 = true;
        *kind = KIND_16C;
        *arch = "gfx950";
        return 0;
    case 21:
        /* 32c bf16: exercises bf16 I/O on the 32c MFMA path */
        p.N = 32;
        p.H = 200;
        p.W = 200;
        p.groups = 32;
        p.cpg = 32;
        p.kpg = 32;
        p.dtype = "bf16";
        *s32 = rocke_direct_conv_32c_spec_default();
        s32->problem = p;
        s32->block_groups = 8;
        *kind = KIND_32C;
        *arch = "gfx950";
        return 0;
    case 22:
        /* depthwise forward bf16: exercises bf16 I/O on the scalar-FMA depthwise path */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.dtype = "bf16";
        *sdw = rocke_direct_depthwise_spec_default();
        sdw->problem = p;
        sdw->block_w = 8;
        sdw->block_waves = 1;
        *kind = KIND_DW;
        *arch = "gfx950";
        return 0;
    case 23:
        /* depthwise spatial bf16: exercises bf16 I/O on the small-group spatial path */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 16;
        p.cpg = 1;
        p.kpg = 1;
        p.dtype = "bf16";
        *ssp = rocke_direct_depthwise_spatial_spec_default();
        ssp->problem = p;
        ssp->block_waves = 1;
        *kind = KIND_SPATIAL;
        *arch = "gfx950";
        return 0;
    case 24:
        /* depthwise dgrad bf16: exercises bf16 I/O on the scalar-FMA depthwise dgrad path */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.dtype = "bf16";
        *sdw_dgrad = rocke_direct_depthwise_dgrad_spec_default();
        sdw_dgrad->problem = p;
        sdw_dgrad->block_w = 8;
        sdw_dgrad->block_waves = 1;
        *kind = KIND_DW_DGRAD;
        *arch = "gfx950";
        return 0;
    /* ---- wgrad (backward weights) ---- */
    case 25:
        /* Defaults: mfma_k=32 (VEC_CH=8, two ds_read_tr per fragment),
         * waves_k=waves_c=waves_q=1, ho_per_block=4. */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 26:
        /* Narrow MFMA: mfma_k=16 (VEC_CH=4, one ds_read_tr per fragment). */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        swg->mfma_k = 16;
        swg->ho_per_block = 2;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 27:
        /* Multi-wave: K/C/Q all split, so n_k_tiles = n_c_tiles = 2,
         * STRIP_GROUPS = 2 and the kernel name carries the _wq flag. */
        p.N = 4;
        p.H = 16;
        p.W = 16;
        p.groups = 4;
        p.cpg = 64;
        p.kpg = 64;
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        swg->waves_k = 2;
        swg->waves_c = 2;
        swg->waves_q = 2;
        swg->ho_per_block = 3;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 28:
        /* gfx942 has no 16x16x32 f16 atom -> both engines reject (mfma_k=32). */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        *kind = KIND_WGRAD;
        *arch = "gfx942";
        return 0;
    case 29:
        /* gfx942 with mfma_k=16 clears the atom gate and is rejected one check
         * later, on the missing ds_read_tr16_b64 the LDS staging needs. */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        swg->mfma_k = 16;
        *kind = KIND_WGRAD;
        *arch = "gfx942";
        return 0;
    case 30:
        /* wgrad bf16: bf16 I/O and the bf16 MFMA atom, same LDS transpose
         * staging. Pins that only the atom and the element type move. */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        p.dtype = "bf16";
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 31:
        /* wgrad bf16 at mfma_k=16: the narrow atom under bf16, one ds_read_tr
         * per fragment. */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 8;
        p.cpg = 16;
        p.kpg = 16;
        p.dtype = "bf16";
        *swg = rocke_direct_conv_wgrad_spec_default();
        swg->problem = p;
        swg->mfma_k = 16;
        swg->ho_per_block = 2;
        *kind = KIND_WGRAD;
        *arch = "gfx950";
        return 0;
    case 32:
        /* windowed depthwise dgrad: 7x7, scalar f32 FMA path */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 7;
        p.KW = 7;
        p.PAD = 3;
        p.dtype = "bf16";
        *swin = rocke_direct_depthwise_dgrad_win_spec_default();
        swin->problem = p;
        swin->block_w = 7;
        swin->block_waves = 1;
        swin->ch_per_lane = 1;
        swin->block_h = 0;
        swin->dot2 = false;
        *kind = KIND_DW_DGRAD_WIN;
        *arch = "gfx950";
        return 0;
    case 33:
        /* windowed depthwise dgrad: 7x7 with fdot2 tap pairing (odd KW pads a zero weight) */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 7;
        p.KW = 7;
        p.PAD = 3;
        p.dtype = "bf16";
        *swin = rocke_direct_depthwise_dgrad_win_spec_default();
        swin->problem = p;
        swin->block_w = 7;
        swin->block_waves = 1;
        swin->ch_per_lane = 1;
        swin->block_h = 0;
        swin->dot2 = true;
        *kind = KIND_DW_DGRAD_WIN;
        *arch = "gfx950";
        return 0;
    case 34:
        /* windowed depthwise dgrad: ch_per_lane=2 (dword loads/stores, packed f32 FMA) */
        p.N = 2;
        p.H = 12;
        p.W = 12;
        p.groups = 128;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 3;
        p.KW = 3;
        p.PAD = 1;
        p.dtype = "fp16";
        *swin = rocke_direct_depthwise_dgrad_win_spec_default();
        swin->problem = p;
        swin->block_w = 6;
        swin->block_waves = 2;
        swin->ch_per_lane = 2;
        swin->block_h = 0;
        swin->dot2 = false;
        *kind = KIND_DW_DGRAD_WIN;
        *arch = "gfx950";
        return 0;
    case 35:
        /* windowed depthwise dgrad: ragged H split (block_h=4, H=13) re-reads the halo */
        p.N = 2;
        p.H = 13;
        p.W = 13;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 5;
        p.KW = 5;
        p.PAD = 2;
        p.dtype = "fp16";
        *swin = rocke_direct_depthwise_dgrad_win_spec_default();
        swin->problem = p;
        swin->block_w = 5;
        swin->block_waves = 1;
        swin->ch_per_lane = 1;
        swin->block_h = 4;
        swin->dot2 = false;
        *kind = KIND_DW_DGRAD_WIN;
        *arch = "gfx950";
        return 0;
    case 36:
        /* windowed depthwise dgrad: even H split + dot2, W=13 not a multiple of block_w */
        p.N = 1;
        p.H = 8;
        p.W = 13;
        p.groups = 40;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 3;
        p.KW = 3;
        p.PAD = 1;
        p.dtype = "bf16";
        *swin = rocke_direct_depthwise_dgrad_win_spec_default();
        swin->problem = p;
        swin->block_w = 4;
        swin->block_waves = 1;
        swin->ch_per_lane = 1;
        swin->block_h = 4;
        swin->dot2 = true;
        *kind = KIND_DW_DGRAD_WIN;
        *arch = "gfx950";
        return 0;
    case 37:
        /* windowed depthwise dgrad: even KW=4 with dot2 (no zero pad) */
        p.N = 2;
        p.H = 9;
        p.W = 9;
        p.groups = 72;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 4;
        p.KW = 4;
        p.PAD = 1;
        p.dtype = "fp16";
        *swin = rocke_direct_depthwise_dgrad_win_spec_default();
        swin->problem = p;
        swin->block_w = 8;
        swin->block_waves = 1;
        swin->ch_per_lane = 1;
        swin->block_h = 0;
        swin->dot2 = true;
        *kind = KIND_DW_DGRAD_WIN;
        *arch = "gfx950";
        return 0;
    case 38:
        /* windowed depthwise dgrad: dot2 on gfx942 -> both engines reject */
        p.N = 2;
        p.H = 8;
        p.W = 8;
        p.groups = 64;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 3;
        p.KW = 3;
        p.PAD = 1;
        p.dtype = "bf16";
        *swin = rocke_direct_depthwise_dgrad_win_spec_default();
        swin->problem = p;
        swin->block_w = 4;
        swin->block_waves = 1;
        swin->ch_per_lane = 1;
        swin->block_h = 0;
        swin->dot2 = true;
        *kind = KIND_DW_DGRAD_WIN;
        *arch = "gfx942";
        return 0;
    case 39:
        /* windowed depthwise dgrad: ch_per_lane=4 (dwordx2) bf16 */
        p.N = 2;
        p.H = 7;
        p.W = 7;
        p.groups = 128;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 3;
        p.KW = 3;
        p.PAD = 1;
        p.dtype = "bf16";
        *swin = rocke_direct_depthwise_dgrad_win_spec_default();
        swin->problem = p;
        swin->block_w = 4;
        swin->block_waves = 1;
        swin->ch_per_lane = 4;
        swin->block_h = 0;
        swin->dot2 = false;
        *kind = KIND_DW_DGRAD_WIN;
        *arch = "gfx950";
        return 0;
    case 40:
        /* windowed depthwise dgrad: KW=1 with dot2 (tail pair only, no full pairs) */
        p.N = 2;
        p.H = 9;
        p.W = 11;
        p.groups = 96;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 3;
        p.KW = 1;
        p.PAD = 1;
        p.dtype = "fp16";
        *swin = rocke_direct_depthwise_dgrad_win_spec_default();
        swin->problem = p;
        swin->block_w = 5;
        swin->block_waves = 1;
        swin->ch_per_lane = 1;
        swin->block_h = 0;
        swin->dot2 = true;
        *kind = KIND_DW_DGRAD_WIN;
        *arch = "gfx950";
        return 0;
    case 41:
        /* windowed depthwise dgrad: non-square 3x5 fp16 dot2 with a ragged H split */
        p.N = 1;
        p.H = 13;
        p.W = 10;
        p.groups = 72;
        p.cpg = 1;
        p.kpg = 1;
        p.KH = 3;
        p.KW = 5;
        p.PAD = 2;
        p.dtype = "fp16";
        *swin = rocke_direct_depthwise_dgrad_win_spec_default();
        swin->problem = p;
        swin->block_w = 10;
        swin->block_waves = 2;
        swin->ch_per_lane = 1;
        swin->block_h = 5;
        swin->dot2 = true;
        *kind = KIND_DW_DGRAD_WIN;
        *arch = "gfx950";
        return 0;
    case 42:
        /* 4c bf16: bf16 I/O and the mfma_f32_4x4x4_bf16 (`_1k`) atom; odd H/W
         * so the right-edge column mask and the H-edge rows are exercised. */
        p.N = 2;
        p.H = 13;
        p.W = 13;
        p.groups = 32;
        p.cpg = 4;
        p.kpg = 4;
        p.dtype = "bf16";
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->block_q = 4;
        s4->block_groups = 16;
        *kind = KIND_4C;
        *arch = "gfx950";
        return 0;
    case 43:
        /* 4c bf16 on gfx942: the 4x4x4 bf16 `_1k` atom is CDNA2+, so the
         * kernel stays arch-neutral. */
        p.N = 1;
        p.H = 8;
        p.W = 8;
        p.groups = 16;
        p.cpg = 4;
        p.kpg = 4;
        p.dtype = "bf16";
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->block_q = 8;
        s4->block_groups = 16;
        *kind = KIND_4C;
        *arch = "gfx942";
        return 0;
    case 44:
        /* 4c bf16 with two waves per block (block_groups=32) and four q-tiles
         * per wave: the shape the 4c dgrad entry builds for cpg=kpg=4. */
        p.N = 2;
        p.H = 14;
        p.W = 14;
        p.groups = 32;
        p.cpg = 4;
        p.kpg = 4;
        p.dtype = "bf16";
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->name = "direct_conv_4c_dgrad";
        s4->block_q = 16;
        s4->block_groups = 32;
        *kind = KIND_4C;
        *arch = "gfx950";
        return 0;
    case 45:
    case 46:
    case 47:
    case 48:
    {
        /* 4c fused dgrad weights (the 4c dgrad entry with dgrad_fused_weights):
         * B is the original weight, read flipped / k<->c transposed in the
         * prologue -- 45/46 per-element gathers (46 = the gfx942 path),
         * 47/48 LDS staging + ds_read_b64_tr_b16 (48 = 1x1, partial pass). */
        static const int cfg[4][8] = {
            /* N, H, W, groups, KH, PAD, block_q, block_groups */
            {2, 13, 13, 32, 3, 1, 4, 16},
            {1, 8, 8, 16, 3, 1, 8, 16},
            {2, 14, 14, 32, 3, 1, 16, 32},
            {1, 7, 9, 64, 1, 0, 4, 64},
        };
        const int* c = cfg[idx - 45];
        p.N = c[0];
        p.H = c[1];
        p.W = c[2];
        p.groups = c[3];
        p.cpg = 4;
        p.kpg = 4;
        p.KH = c[4];
        p.KW = c[4];
        p.PAD = c[5];
        p.dtype = (idx == 46 || idx == 48) ? "fp16" : "bf16";
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->name = "direct_conv_4c_dgrad";
        s4->block_q = c[6];
        s4->block_groups = c[7];
        s4->dgrad_fused_weights = true;
        s4->dgrad_weights_lds = (idx >= 47);
        *kind = KIND_4C;
        *arch = (idx == 46) ? "gfx942" : "gfx950";
        return 0;
    }
    case 49:
    case 50:
    case 51:
    case 52:
    case 53:
    case 54:
    case 55:
    case 56:
    {
        /* windowed depthwise dgrad, Toeplitz MFMA form (mfma): one-hot weights
         * x 8 channels per 16x16x32 MFMA, LDS-staged dY windows and dX rows.
         * 49 odd N with w_fold 2 (empty second image of the last tile);
         * 50 ragged H split + 2 W tiles + a partial channel block, w_fold 1;
         * 51 9x9 w_fold 4 even H split; 52 11x11 ragged split (3 passes);
         * 53 gfx942 -> both reject; 54 3x3 pad 0 single pass, one wave;
         * 55 groups % 8 != 0 -> both reject; 56 non-square 7x5. */
        static const int cfg[8][12] = {
            /* N, H, W, groups, KH, KW, PAD, bf16, block_waves, w_fold, prefetch_rows, block_h */
            {3, 14, 14, 64, 7, 7, 3, 1, 8, 2, 2, 0},
            {2, 13, 37, 40, 5, 5, 2, 0, 4, 1, 1, 4},
            {2, 10, 11, 16, 9, 9, 4, 1, 2, 4, 3, 5},
            {1, 13, 13, 64, 11, 11, 5, 0, 8, 2, 2, 4},
            {2, 14, 14, 64, 7, 7, 3, 1, 8, 2, 2, 0},
            {1, 9, 12, 8, 3, 3, 0, 0, 1, 2, 1, 0},
            {2, 14, 14, 12, 7, 7, 3, 1, 1, 2, 2, 0},
            {2, 11, 15, 32, 7, 5, 2, 1, 4, 2, 2, 0},
        };
        const int* c = cfg[idx - 49];
        p.N = c[0];
        p.H = c[1];
        p.W = c[2];
        p.groups = c[3];
        p.cpg = 1;
        p.kpg = 1;
        p.KH = c[4];
        p.KW = c[5];
        p.PAD = c[6];
        p.dtype = c[7] ? "bf16" : "fp16";
        *swin = rocke_direct_depthwise_dgrad_win_spec_default();
        swin->problem = p;
        swin->block_waves = c[8];
        swin->block_h = c[11];
        swin->mfma = true;
        swin->w_fold = c[9];
        swin->prefetch_rows = c[10];
        *kind = KIND_DW_DGRAD_WIN;
        *arch = (idx == 53) ? "gfx942" : "gfx950";
        return 0;
    }
    case 57:
    case 58:
    case 59:
    case 60:
    {
        /* 4c row-staged dgrad (stage_rows, fused LDS weights): 57 bf16 3x3 odd
         * W (partial q tile), 58 fp16 1x1 with a partial row-staging pass,
         * 59 bf16 two q waves x two channel waves, 60 fp16 H=1 (the halo rows
         * outside the image are skipped). */
        static const int cfg[4][8] = {
            /* N, H, W, groups, KH, block_q, block_groups, waves_q */
            {2, 13, 13, 32, 3, 4, 16, 1},
            {1, 7, 9, 64, 1, 4, 64, 1},
            {2, 9, 21, 64, 3, 8, 32, 2},
            {1, 1, 6, 16, 3, 4, 16, 1},
        };
        const int* c = cfg[idx - 57];
        p.N = c[0];
        p.H = c[1];
        p.W = c[2];
        p.groups = c[3];
        p.cpg = 4;
        p.kpg = 4;
        p.KH = c[4];
        p.KW = c[4];
        p.PAD = (c[4] - 1) / 2;
        p.dtype = (idx == 58 || idx == 60) ? "fp16" : "bf16";
        *s4 = rocke_direct_conv_4c_spec_default();
        s4->problem = p;
        s4->name = "direct_conv_4c_dgrad";
        s4->block_q = c[5];
        s4->block_groups = c[6];
        s4->dgrad_fused_weights = true;
        s4->dgrad_weights_lds = true;
        s4->stage_rows = true;
        s4->waves_q = c[7];
        *kind = KIND_4C;
        *arch = "gfx950";
        return 0;
    }
    case 61:
    case 62:
    case 63:
    case 64:
    case 65:
    case 66:
    case 67:
    case 68:
    case 69:
    case 70:
    case 71:
    case 72:
    {
        /* Generic DirectConvSpec (build_direct_conv, the grouped direct-MFMA
         * dgrad main kernel): 61-66 the default-knob paths, 67-70 the
         * row-stream knobs, 71-72 preloaded weights / single buffer and the
         * knobs on gfx942 (see the Python emitter for the per-config notes). */
        static const int cfg[12][8] = {
            /* N, H, W, groups, cpg, kpg, K, bf16 */
            {2, 9, 17, 8, 16, 16, 3, 0},
            {2, 10, 13, 32, 16, 16, 3, 1},
            {2, 9, 14, 16, 32, 32, 3, 1},
            {2, 9, 17, 4, 24, 12, 5, 0},
            {2, 9, 17, 4, 16, 16, 3, 1},
            {2, 7, 17, 4, 32, 16, 3, 0},
            {9, 10, 13, 32, 16, 16, 3, 1},
            {8, 11, 19, 32, 16, 16, 3, 0},
            {8, 7, 14, 16, 32, 32, 3, 1},
            {2, 9, 14, 8, 8, 32, 5, 0},
            {2, 9, 17, 4, 16, 16, 3, 0},
            {2, 9, 17, 8, 16, 16, 3, 1},
        };
        const int* c = cfg[idx - 61];
        p.N = c[0];
        p.H = c[1];
        p.W = c[2];
        p.groups = c[3];
        p.cpg = c[4];
        p.kpg = c[5];
        p.KH = c[6];
        p.KW = c[6];
        p.PAD = (c[6] - 1) / 2;
        p.dtype = c[7] ? "bf16" : "fp16";
        *sgen = rocke_direct_conv_spec_default();
        sgen->problem = p;
        sgen->name = "direct_mfma_dgrad";
        *arch = "gfx950";
        switch(idx)
        {
        case 61:
            sgen->block_groups = 2;
            sgen->dgrad_fused_weights = true;
            break;
        case 62:
            sgen->block_groups = 2;
            sgen->dgrad_fused_weights = true;
            sgen->dgrad_weights_lds = true;
            sgen->waves_per_eu = 4;
            break;
        case 63:
            sgen->block_groups = 1;
            sgen->fold_k32 = true;
            sgen->dgrad_fused_weights = true;
            sgen->dgrad_weights_lds = true;
            break;
        case 64:
            sgen->block_groups = 1;
            sgen->block_h = 4;
            break;
        case 65:
            sgen->block_groups = 1;
            sgen->block_h = 4;
            sgen->persistent_grid = true;
            sgen->runtime_k_loop = true;
            break;
        case 66:
            sgen->block_q = 32;
            sgen->block_groups = 1;
            sgen->waves_q = 2;
            sgen->waves_k = 2;
            break;
        case 67:
            sgen->block_groups = 2;
            sgen->dgrad_fused_weights = true;
            sgen->dgrad_weights_lds = true;
            sgen->waves_per_eu = 4;
            sgen->prefetch_rows = 2;
            sgen->lds_only_sync = true;
            sgen->lds_pad = 8;
            sgen->stage_out = true;
            sgen->xcd_tiles = true;
            break;
        case 68:
            sgen->block_q = 32;
            sgen->block_groups = 2;
            sgen->block_h = 8;
            sgen->dgrad_fused_weights = true;
            sgen->dgrad_weights_lds = true;
            sgen->prefetch_rows = 2;
            sgen->lds_only_sync = true;
            sgen->lds_pad = 8;
            sgen->stage_out = true;
            sgen->xcd_tiles = true;
            break;
        case 69:
            sgen->block_groups = 1;
            sgen->fold_k32 = true;
            sgen->dgrad_fused_weights = true;
            sgen->dgrad_weights_lds = true;
            sgen->prefetch_rows = 2;
            sgen->lds_only_sync = true;
            sgen->waves_m = 2;
            sgen->lds_pad = 8;
            sgen->stage_out = true;
            sgen->xcd_tiles = true;
            break;
        case 70:
            sgen->block_groups = 1;
            sgen->block_h = 4;
            sgen->dgrad_fused_weights = true;
            sgen->prefetch_rows = 3;
            sgen->lds_only_sync = true;
            sgen->waves_m = 2;
            sgen->stage_out = true;
            break;
        case 71:
            sgen->block_groups = 2;
            sgen->preload_weights = true;
            sgen->double_buffer = false;
            *arch = "gfx942";
            break;
        default: /* 72 */
            sgen->block_groups = 2;
            sgen->block_h = 4;
            sgen->dgrad_fused_weights = true;
            sgen->prefetch_rows = 2;
            sgen->lds_only_sync = true;
            sgen->lds_pad = 8;
            *arch = "gfx942";
            break;
        }
        *kind = KIND_GENERIC;
        return 0;
    }
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

    int kind = KIND_16C;
    rocke_direct_conv_16c_spec_t s16;
    rocke_direct_conv_4c_spec_t s4;
    rocke_direct_conv_8c_spec_t s8;
    rocke_direct_conv_32c_spec_t s32;
    rocke_direct_depthwise_spec_t sdw;
    rocke_direct_depthwise_spatial_spec_t ssp;
    rocke_direct_conv_dgrad_spec_t sdgrad;
    rocke_direct_depthwise_dgrad_spec_t sdw_dgrad;
    rocke_direct_conv_wgrad_spec_t swg;
    rocke_direct_depthwise_dgrad_win_spec_t swin;
    rocke_direct_conv_spec_t sgen;
    const char* arch = "gfx950";
    if(make_cfg(idx,
                &kind,
                &s16,
                &s4,
                &s8,
                &s32,
                &sdw,
                &ssp,
                &sdgrad,
                &sdw_dgrad,
                &swg,
                &swin,
                &sgen,
                &arch)
       != 0)
    {
        fprintf(stderr, "unknown config index %d\n", idx);
        return 2;
    }

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = NULL;
    if(kind == KIND_16C)
        kernel = rocke_build_direct_conv_16c_new(&b, &s16, arch);
    else if(kind == KIND_4C)
        kernel = rocke_build_direct_conv_4c_new(&b, &s4, arch);
    else if(kind == KIND_8C)
        kernel = rocke_build_direct_conv_8c_new(&b, &s8, arch);
    else if(kind == KIND_32C)
        kernel = rocke_build_direct_conv_32c_new(&b, &s32, arch);
    else if(kind == KIND_SPATIAL)
        kernel = rocke_build_direct_depthwise_spatial_new(&b, &ssp, arch);
    else if(kind == KIND_DGRAD)
        kernel = rocke_build_direct_conv_dgrad_new(&b, &sdgrad, arch);
    else if(kind == KIND_DW_DGRAD)
        kernel = rocke_build_direct_depthwise_dgrad_new(&b, &sdw_dgrad, arch);
    else if(kind == KIND_WGRAD)
        kernel = rocke_build_direct_conv_wgrad_new(&b, &swg, arch);
    else if(kind == KIND_DW_DGRAD_WIN)
        kernel = rocke_build_direct_depthwise_dgrad_win_new(&b, &swin, arch);
    else if(kind == KIND_GENERIC)
        kernel = rocke_build_direct_conv_new(&b, &sgen, arch);
    else
        kernel = rocke_build_direct_depthwise_new(&b, &sdw, arch);
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
