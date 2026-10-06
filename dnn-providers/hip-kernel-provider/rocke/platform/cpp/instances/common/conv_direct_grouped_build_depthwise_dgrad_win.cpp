// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_depthwise_dgrad_win.cpp -- C99 port of
 * build_direct_depthwise_dgrad_windowed (library/kernels/common/conv_direct_grouped.py).
 *
 * Windowed ho-streaming depthwise dgrad (stride 1): each dY row is loaded once
 * per block as a window of block_w + KW - 1 columns, the next live row is
 * prefetched before the current row's FMAs (sched_barrier pinned), and every
 * buffer offset is a row-invariant lane part plus a block-uniform row part with
 * out-of-range sentinels instead of per-tap selects. Optional knobs: channel
 * packing (ch_per_lane), H split (block_h), fdot2 tap pairing (dot2) and the
 * Toeplitz MFMA form (mfma, with w_fold and prefetch_rows).
 *
 * IMPORTANT: every call that has side effects on the IR builder (rocke_b_*) is
 * issued as a separate statement in Python's left-to-right evaluation order, so
 * the emitted IR numbering matches the Python engine byte for byte.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/instance_conv_direct_grouped_internal.h"
#include "rocke/ir.h"

typedef struct dw_win_ctx
{
    rocke_ir_builder_t* b;
    int is_bf16;
    int KH, KW, PAD, Ho, ROWS, WIN, CPL;
    bool split_h;
    rocke_value_t* c0;
    rocke_value_t* oob_uniform;
    rocke_value_t* h0;
    rocke_value_t* c_Ho;
    rocke_value_t* a_rsrc;
    rocke_value_t* dy_n_row;
    rocke_value_t* c_dy_row_bytes;
    rocke_value_t** win_lane;
} dw_win_ctx;

/* Python load_half(rsrc, off). */
static rocke_value_t* dw_win_load_half(const dw_win_ctx* cx, rocke_value_t* rsrc, rocke_value_t* off)
{
    if(cx->is_bf16)
        return rocke_b_buffer_load_bf16(cx->b, rsrc, off, cx->c0);
    return rocke_b_buffer_load_f16(cx->b, rsrc, off, cx->c0);
}

/* Python row_part(n_row, row, row_bytes, row_ok). */
static rocke_value_t* dw_win_row_part(const dw_win_ctx* cx,
                               rocke_value_t* n_row,
                               rocke_value_t* row,
                               rocke_value_t* row_bytes,
                               rocke_value_t* row_ok)
{
    rocke_value_t* sum = rocke_b_add(cx->b, n_row, row);
    rocke_value_t* part = rocke_b_mul(cx->b, sum, row_bytes);
    if(row_ok == NULL)
        return part;
    return rocke_b_select(cx->b, row_ok, part, cx->oob_uniform);
}

/* Python row_taps(y): writes the owned filter rows into rs, returns the count. */
static int dw_win_row_taps(const dw_win_ctx* cx, int y, int* rs)
{
    int n = 0;
    for(int r = 0; r < cx->KH; r++)
    {
        const int hl = y + r - (cx->KH - 1);
        if(hl >= 0 && hl < cx->ROWS)
            rs[n++] = r;
    }
    const int rel_ho = y + cx->PAD - (cx->KH - 1);
    if(n > 0 && (cx->split_h || (rel_ho >= 0 && rel_ho < cx->Ho)))
        return n;
    return 0;
}

/* Python load_window(y): fills raw[0..WIN). */
static void dw_win_load_window(const dw_win_ctx* cx, int y, rocke_value_t** raw)
{
    rocke_ir_builder_t* b = cx->b;
    const int rel_ho = y + cx->PAD - (cx->KH - 1);
    rocke_value_t* ho;
    rocke_value_t* row_ok = NULL;
    if(cx->split_h)
    {
        rocke_value_t* c_rel = rocke_b_const_i32(b, rel_ho);
        ho = rocke_b_add(b, cx->h0, c_rel);
        rocke_value_t* ge = rocke_b_cmp_ge(b, ho, cx->c0);
        rocke_value_t* lt = rocke_b_cmp_lt(b, ho, cx->c_Ho);
        row_ok = rocke_b_land(b, ge, lt);
    }
    else
    {
        ho = rocke_b_const_i32(b, rel_ho);
    }
    rocke_value_t* dy_row = dw_win_row_part(cx, cx->dy_n_row, ho, cx->c_dy_row_bytes, row_ok);
    for(int t = 0; t < cx->WIN; t++)
    {
        rocke_value_t* off = rocke_b_add(b, cx->win_lane[t], dy_row);
        if(cx->CPL == 1)
            raw[t] = dw_win_load_half(cx, cx->a_rsrc, off);
        else if(cx->is_bf16)
            raw[t] = rocke_b_buffer_load_vN_bf16(b, cx->a_rsrc, off, cx->c0, cx->CPL / 2);
        else
            raw[t] = rocke_b_buffer_load_vN_f16(b, cx->a_rsrc, off, cx->c0, cx->CPL / 2);
    }
    rocke_b_sched_barrier(b, 0);
}

/* Python widen(v). */
static rocke_value_t* dw_win_widen(const dw_win_ctx* cx, rocke_value_t* v)
{
    if(cx->CPL == 1)
        return rocke_b_cast_to_f32(cx->b, v);
    rocke_value_t* comps[8];
    for(int i = 0; i < cx->CPL; i++)
    {
        rocke_value_t* e = rocke_b_vec_extract(cx->b, v, i);
        comps[i] = rocke_b_cast_to_f32(cx->b, e);
    }
    return rocke_b_vec_pack(cx->b, comps, cx->CPL, rocke_f32());
}

/* ===================================================================== *
 *  Toeplitz MFMA form (spec->mfma): C port of _build_dw_dgrad_win_mfma.
 *
 *  Each wave owns 8 channels and one 16x16 fp32 tile per in-flight dX row;
 *  one-hot weights (A) meet 8 channels x 4 window columns (B) per 16x16x32
 *  MFMA. dY window rows and finished dX rows are staged through two LDS
 *  double buffers that share one barrier per row. Every builder call is a
 *  separate statement in Python evaluation order.
 * ===================================================================== */

typedef struct dw_mfma_plan
{
    rocke_value_t* px;
    rocke_value_t* ch8;
    rocke_value_t* g_off;
} dw_mfma_plan;

typedef struct dw_mfma_ctx
{
    rocke_ir_builder_t* b;
    int is_bf16;
    int KH, PAD, Ho, Wo, G, ROWS, W;
    bool split_h;
    rocke_value_t* c0;
    rocke_value_t* oob_uniform;
    rocke_value_t* h0;
    rocke_value_t* c_Ho;
    rocke_value_t* c_H;
    rocke_value_t* a_rsrc;
    rocke_value_t* d_rsrc;
    rocke_value_t* o_smem;
    rocke_value_t* c_dy_row_bytes;
    rocke_value_t* c_dx_row_bytes;
    const rocke_type_t* io_type;
    const dw_mfma_plan* in_plan;
    int in_passes;
    const dw_mfma_plan* drain;
    int n_drain;
} dw_mfma_ctx;

/* Python load_row(y) of the MFMA form: fills vals[0..in_passes). */
static void dw_mfma_load_row(const dw_mfma_ctx* cx, int y, rocke_value_t** vals)
{
    rocke_ir_builder_t* b = cx->b;
    const int rel_ho = y + cx->PAD - (cx->KH - 1);
    rocke_value_t* dy_row;
    if(cx->split_h)
    {
        rocke_value_t* c_rel = rocke_b_const_i32(b, rel_ho);
        rocke_value_t* ho = rocke_b_add(b, cx->h0, c_rel);
        rocke_value_t* ge = rocke_b_cmp_ge(b, ho, cx->c0);
        rocke_value_t* lt = rocke_b_cmp_lt(b, ho, cx->c_Ho);
        rocke_value_t* row_ok = rocke_b_land(b, ge, lt);
        rocke_value_t* part = rocke_b_mul(b, ho, cx->c_dy_row_bytes);
        dy_row = rocke_b_select(b, row_ok, part, cx->oob_uniform);
    }
    else
    {
        dy_row = rocke_b_const_i32(b, rel_ho * cx->Wo * cx->G * 2);
    }
    for(int i = 0; i < cx->in_passes; i++)
    {
        rocke_value_t* off = rocke_b_add(b, cx->in_plan[i].g_off, dy_row);
        vals[i] = cx->is_bf16 ? rocke_b_buffer_load_vN_bf16(b, cx->a_rsrc, off, cx->c0, 4)
                              : rocke_b_buffer_load_vN_f16(b, cx->a_rsrc, off, cx->c0, 4);
    }
    rocke_b_sched_barrier(b, 0);
}

/* Python dx_row_part(hi_local). */
static rocke_value_t* dw_mfma_dx_row_part(const dw_mfma_ctx* cx, int hi_local)
{
    rocke_ir_builder_t* b = cx->b;
    if(!cx->split_h)
        return rocke_b_const_i32(b, hi_local * cx->W * cx->G * 2);
    rocke_value_t* c_hl = rocke_b_const_i32(b, hi_local);
    rocke_value_t* hi = rocke_b_add(b, cx->h0, c_hl);
    rocke_value_t* part = rocke_b_mul(b, hi, cx->c_dx_row_bytes);
    if(cx->c_H == NULL)
        return part;
    rocke_value_t* lt = rocke_b_cmp_lt(b, hi, cx->c_H);
    return rocke_b_select(b, lt, part, cx->oob_uniform);
}

/* Python drain_out(buf, dx_row). */
static void dw_mfma_drain_out(const dw_mfma_ctx* cx, rocke_value_t* buf, rocke_value_t* dx_row)
{
    rocke_ir_builder_t* b = cx->b;
    for(int i = 0; i < cx->n_drain; i++)
    {
        rocke_value_t* idx[3] = {buf, cx->drain[i].px, cx->drain[i].ch8};
        rocke_value_t* v = rocke_b_smem_load_vN(b, cx->o_smem, idx, 3, cx->io_type, 8);
        rocke_value_t* off = rocke_b_add(b, cx->drain[i].g_off, dx_row);
        if(cx->is_bf16)
            rocke_b_buffer_store_vN_bf16(b, cx->d_rsrc, off, cx->c0, v, 4);
        else
            rocke_b_buffer_store_vN_f16(b, cx->d_rsrc, off, cx->c0, v, 4);
    }
}

static rocke_kernel_def_t* dw_win_build_mfma(rocke_ir_builder_t* b,
                                             const rocke_direct_depthwise_dgrad_win_spec_t* spec)
{
    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int KH = p->KH;
    const int KW = p->KW;
    const int PAD = p->PAD;
    const int Ho = p->H + 2 * PAD - KH + 1; /* stride 1 (validated) */
    const int Wo = p->W + 2 * PAD - KW + 1;
    const int G = p->groups;
    const int F = spec->w_fold;
    const int SUB = 16 / F;
    const int TW = rocke_direct_depthwise_dgrad_win_tile_w(spec);
    const int KWP = (KW + 4) / 4;
    const int U = TW + 4 * KWP - 2;
    const int ROWS = rocke_direct_depthwise_dgrad_win_rows_per_block(spec);
    const int H_TILES = rocke_direct_depthwise_dgrad_win_h_tiles(spec);
    const int THREADS = rocke_direct_depthwise_dgrad_win_threads_per_block(spec);
    const int BLOCK_CH = rocke_direct_depthwise_dgrad_win_block_ch(spec);
    const int PER_PX = BLOCK_CH / ROCKE_DW_DGRAD_MFMA_CH;
    const int CH_PAD = BLOCK_CH + ROCKE_DW_DGRAD_MFMA_CH;
    const int PF = spec->prefetch_rows;
    const bool split_h = H_TILES > 1;
    const int is_bf16 = (p->dtype && strcmp(p->dtype, "bf16") == 0);
    const int n_out = F * TW * PER_PX;
    const int n_drain = (n_out + THREADS - 1) / THREADS;
    const int n_in = F * U * PER_PX;
    const int in_passes = (n_in + THREADS - 1) / THREADS;
    const int in_px = (in_passes * THREADS + PER_PX - 1) / PER_PX;
    const int n_rows = ROWS + KH - 1;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);

    const rocke_type_t* io_type = rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16");
    const rocke_type_t* ioptr = rocke_ptr_type(b, io_type, "global");
    rocke_param_opts_t ro = {0};
    ro.noalias = true;
    ro.noalias_set = true;
    ro.readonly = true;
    ro.readonly_set = true;
    ro.align = 16;
    ro.align_set = true;
    rocke_param_opts_t wo_opts = {0};
    wo_opts.noalias = true;
    wo_opts.noalias_set = true;
    wo_opts.writeonly = true;
    wo_opts.writeonly_set = true;
    wo_opts.align = 16;
    wo_opts.align_set = true;
    rocke_param_opts_t none = {0};

    rocke_value_t* A = rocke_b_param(b, "A", ioptr, &ro);
    rocke_value_t* Bp = rocke_b_param(b, "B", ioptr, &ro);
    rocke_value_t* D = rocke_b_param(b, "D", ioptr, &wo_opts);
    rocke_value_t* A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), &none);
    rocke_value_t* B_bytes = rocke_b_param(b, "B_bytes", rocke_i32(), &none);
    rocke_value_t* D_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), &none);

    /* Scratch arrays (freed before every return below). */
    rocke_value_t** weights = (rocke_value_t**)calloc((size_t)(KH * KWP), sizeof(rocke_value_t*));
    dw_mfma_plan* drain = (dw_mfma_plan*)calloc((size_t)n_drain, sizeof(dw_mfma_plan));
    dw_mfma_plan* in_plan = (dw_mfma_plan*)calloc((size_t)in_passes, sizeof(dw_mfma_plan));
    rocke_value_t** b_px = (rocke_value_t**)calloc((size_t)KWP, sizeof(rocke_value_t*));
    rocke_value_t** xs = (rocke_value_t**)calloc((size_t)KWP, sizeof(rocke_value_t*));
    rocke_value_t** pending
        = (rocke_value_t**)calloc((size_t)(PF * in_passes), sizeof(rocke_value_t*));
    rocke_value_t** acc = (rocke_value_t**)calloc((size_t)KH, sizeof(rocke_value_t*));
    int* rs = (int*)calloc((size_t)KH, sizeof(int));
    int* live_rows = (int*)calloc((size_t)n_rows, sizeof(int));
    rocke_kernel_def_t* result = NULL;
    if(!weights || !drain || !in_plan || !b_px || !xs || !pending || !acc || !rs || !live_rows)
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_OOM;
        goto done;
    }

    {
        rocke_value_t* c0 = rocke_b_const_i32(b, 0);
        rocke_value_t* oob_lane = rocke_b_const_i32(b, ROCKE_DW_DGRAD_WIN_OOB_LANE);
        rocke_value_t* oob_uniform = rocke_b_const_i32(b, ROCKE_DW_DGRAD_WIN_OOB_UNIFORM);
        rocke_value_t* zero_acc = rocke_b_zero_vec_f32(b, 4);
        rocke_value_t* zero_f32 = rocke_b_const_f32(b, 0.0f);
        rocke_value_t* zero_half = is_bf16 ? rocke_b_trunc_f32_to_bf16(b, zero_f32)
                                           : rocke_b_trunc_f32_to_f16(b, zero_f32);

        rocke_value_t* tid = rocke_b_thread_id_x(b);
        rocke_value_t* bx = rocke_b_block_id_x(b);
        rocke_value_t* by = rocke_b_block_id_y(b);
        rocke_value_t* bz = rocke_b_block_id_z(b);
        rocke_value_t* n_tile;
        rocke_value_t* h0 = NULL;
        if(split_h)
        {
            rocke_value_t* c_h_tiles = rocke_b_const_i32(b, H_TILES);
            n_tile = rocke_b_div(b, bz, c_h_tiles);
            rocke_value_t* tile = rocke_b_mod(b, bz, c_h_tiles);
            rocke_value_t* c_rows = rocke_b_const_i32(b, ROWS);
            h0 = rocke_b_mul(b, tile, c_rows);
        }
        else
        {
            n_tile = bz;
        }
        rocke_value_t* c64 = rocke_b_const_i32(b, 64);
        rocke_value_t* lane = rocke_b_mod(b, tid, c64);
        rocke_value_t* wave = rocke_b_div(b, tid, c64);
        rocke_value_t* c_tw0 = rocke_b_const_i32(b, TW);
        rocke_value_t* x0 = rocke_b_mul(b, bx, c_tw0);
        rocke_value_t* c_f = rocke_b_const_i32(b, F);
        rocke_value_t* img0 = rocke_b_mul(b, n_tile, c_f);
        rocke_value_t* c_block_ch = rocke_b_const_i32(b, BLOCK_CH);
        rocke_value_t* ch_blk = rocke_b_mul(b, by, c_block_ch);
        rocke_value_t* c_mch = rocke_b_const_i32(b, ROCKE_DW_DGRAD_MFMA_CH);
        rocke_value_t* wave_ch = rocke_b_mul(b, wave, c_mch);
        rocke_value_t* c_wave = rocke_b_add(b, ch_blk, wave_ch);
        rocke_value_t* c_G = rocke_b_const_i32(b, G);
        rocke_value_t* ch_ok = rocke_b_cmp_lt(b, c_wave, c_G);
        rocke_value_t* c16 = rocke_b_const_i32(b, 16);
        rocke_value_t* c8 = rocke_b_const_i32(b, 8);
        rocke_value_t* col16 = rocke_b_mod(b, lane, c16);
        rocke_value_t* t = rocke_b_div(b, lane, c16);
        rocke_value_t* k = rocke_b_mod(b, col16, c8);
        rocke_value_t* q = rocke_b_div(b, col16, c8);

        rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, A, A_bytes);
        rocke_value_t* b_rsrc = rocke_b_buffer_rsrc(b, Bp, B_bytes);
        rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, D, D_bytes);

        /* One-hot A fragments weights[r][ph]: W[k, r, KW - 1 - s] at slot k. */
        rocke_value_t* wk = rocke_b_add(b, c_wave, k);
        rocke_value_t* c_tap_bytes = rocke_b_const_i32(b, KH * KW * 2);
        rocke_value_t* w_base = rocke_b_mul(b, wk, c_tap_bytes);
        rocke_value_t* c_KW = rocke_b_const_i32(b, KW);
        rocke_value_t* is_k[ROCKE_DW_DGRAD_MFMA_CH];
        for(int j = 0; j < ROCKE_DW_DGRAD_MFMA_CH; j++)
        {
            rocke_value_t* c_j = rocke_b_const_i32(b, j);
            is_k[j] = rocke_b_cmp_eq(b, k, c_j);
        }
        for(int r = 0; r < KH; r++)
        {
            for(int ph = 0; ph < KWP; ph++)
            {
                rocke_value_t* c_4ph = rocke_b_const_i32(b, 4 * ph);
                rocke_value_t* t4 = rocke_b_add(b, t, c_4ph);
                rocke_value_t* s = rocke_b_sub(b, t4, q);
                rocke_value_t* ge = rocke_b_cmp_ge(b, s, c0);
                rocke_value_t* lt = rocke_b_cmp_lt(b, s, c_KW);
                rocke_value_t* in_row = rocke_b_land(b, ge, lt);
                rocke_value_t* ok = rocke_b_land(b, ch_ok, in_row);
                rocke_value_t* c_tap = rocke_b_const_i32(b, r * KW + KW - 1);
                rocke_value_t* tap = rocke_b_sub(b, c_tap, s);
                rocke_value_t* c2 = rocke_b_const_i32(b, 2);
                rocke_value_t* tap_bytes = rocke_b_mul(b, tap, c2);
                rocke_value_t* w_off = rocke_b_add(b, w_base, tap_bytes);
                rocke_value_t* off = rocke_b_select(b, ok, w_off, oob_lane);
                rocke_value_t* w = is_bf16 ? rocke_b_buffer_load_bf16(b, b_rsrc, off, c0)
                                           : rocke_b_buffer_load_f16(b, b_rsrc, off, c0);
                rocke_value_t* comps[ROCKE_DW_DGRAD_MFMA_CH];
                for(int j = 0; j < ROCKE_DW_DGRAD_MFMA_CH; j++)
                    comps[j] = rocke_b_select(b, is_k[j], w, zero_half);
                weights[r * KWP + ph] = rocke_b_vec_pack(b, comps, ROCKE_DW_DGRAD_MFMA_CH, io_type);
            }
        }

        /* Lane parts shared by the B reads and the D staging. */
        rocke_value_t* c_sub = rocke_b_const_i32(b, SUB);
        rocke_value_t* f = rocke_b_div(b, col16, c_sub);
        rocke_value_t* pair = rocke_b_mod(b, col16, c_sub);
        rocke_value_t* c2p = rocke_b_const_i32(b, 2);
        rocke_value_t* pair2 = rocke_b_mul(b, pair, c2p);
        rocke_value_t* c_px_bytes = rocke_b_const_i32(b, G * 2);
        rocke_value_t* c_dy_img_bytes = rocke_b_const_i32(b, Ho * Wo * G * 2);
        rocke_value_t* c_dx_img_bytes = rocke_b_const_i32(b, p->H * p->W * G * 2);
        rocke_value_t* c_N = rocke_b_const_i32(b, p->N);
        rocke_value_t* c_W = rocke_b_const_i32(b, p->W);
        rocke_value_t* c_Wo = rocke_b_const_i32(b, Wo);
        rocke_value_t* c_per_px = rocke_b_const_i32(b, PER_PX);

        /* dX staging [buffer, image column, channel] and its drain plan. */
        const int o_shape[3] = {2, F * TW, CH_PAD};
        rocke_value_t* o_smem = rocke_b_smem_alloc(b, io_type, o_shape, 3, "dx_stage");
        rocke_value_t* c_tw1 = rocke_b_const_i32(b, TW);
        rocke_value_t* f_tw = rocke_b_mul(b, f, c_tw1);
        rocke_value_t* c2q = rocke_b_const_i32(b, 2);
        rocke_value_t* t_half = rocke_b_div(b, t, c2q);
        rocke_value_t* col_in = rocke_b_add(b, pair2, t_half);
        rocke_value_t* o_col = rocke_b_add(b, f_tw, col_in);
        rocke_value_t* c2r = rocke_b_const_i32(b, 2);
        rocke_value_t* t_par = rocke_b_mod(b, t, c2r);
        rocke_value_t* c4 = rocke_b_const_i32(b, 4);
        rocke_value_t* k0 = rocke_b_mul(b, t_par, c4);
        rocke_value_t* o_ch = rocke_b_add(b, wave_ch, k0);
        rocke_value_t* c_TW = rocke_b_const_i32(b, TW);
        for(int i = 0; i < n_drain; i++)
        {
            rocke_value_t* c_base = rocke_b_const_i32(b, i * THREADS);
            rocke_value_t* idx = rocke_b_add(b, tid, c_base);
            rocke_value_t* px = rocke_b_div(b, idx, c_per_px);
            rocke_value_t* chunk = rocke_b_mod(b, idx, c_per_px);
            rocke_value_t* ch8 = rocke_b_mul(b, chunk, c8);
            rocke_value_t* f_o = rocke_b_div(b, px, c_TW);
            rocke_value_t* img_o = rocke_b_add(b, img0, f_o);
            rocke_value_t* col_o = rocke_b_mod(b, px, c_TW);
            rocke_value_t* wi_o = rocke_b_add(b, x0, col_o);
            rocke_value_t* ch_o = rocke_b_add(b, ch_blk, ch8);
            rocke_value_t* c_n_out = rocke_b_const_i32(b, n_out);
            rocke_value_t* ok0 = rocke_b_cmp_lt(b, idx, c_n_out);
            rocke_value_t* ok1 = rocke_b_cmp_lt(b, img_o, c_N);
            rocke_value_t* ok01 = rocke_b_land(b, ok0, ok1);
            rocke_value_t* ok2 = rocke_b_cmp_lt(b, wi_o, c_W);
            rocke_value_t* ok3 = rocke_b_cmp_lt(b, ch_o, c_G);
            rocke_value_t* ok23 = rocke_b_land(b, ok2, ok3);
            rocke_value_t* ok = rocke_b_land(b, ok01, ok23);
            rocke_value_t* img_b = rocke_b_mul(b, img_o, c_dx_img_bytes);
            rocke_value_t* col_b = rocke_b_mul(b, wi_o, c_px_bytes);
            rocke_value_t* c2s = rocke_b_const_i32(b, 2);
            rocke_value_t* ch_b = rocke_b_mul(b, ch_o, c2s);
            rocke_value_t* lane_b = rocke_b_add(b, col_b, ch_b);
            rocke_value_t* g_off = rocke_b_add(b, img_b, lane_b);
            drain[i].px = px;
            drain[i].ch8 = ch8;
            drain[i].g_off = rocke_b_select(b, ok, g_off, oob_lane);
        }

        /* dY staging [buffer, window column, channel] and its load plan. */
        const int i_shape[3] = {2, in_px, CH_PAD};
        rocke_value_t* i_smem = rocke_b_smem_alloc(b, io_type, i_shape, 3, "dy_stage");
        rocke_value_t* c_U = rocke_b_const_i32(b, U);
        for(int i = 0; i < in_passes; i++)
        {
            rocke_value_t* c_base = rocke_b_const_i32(b, i * THREADS);
            rocke_value_t* idx = rocke_b_add(b, tid, c_base);
            rocke_value_t* px = rocke_b_div(b, idx, c_per_px);
            rocke_value_t* chunk = rocke_b_mod(b, idx, c_per_px);
            rocke_value_t* ch8 = rocke_b_mul(b, chunk, c8);
            rocke_value_t* f_i = rocke_b_div(b, px, c_U);
            rocke_value_t* img_i = rocke_b_add(b, img0, f_i);
            rocke_value_t* col_i = rocke_b_mod(b, px, c_U);
            rocke_value_t* c_shift = rocke_b_const_i32(b, PAD - (KW - 1));
            rocke_value_t* col_s = rocke_b_add(b, col_i, c_shift);
            rocke_value_t* wo = rocke_b_add(b, x0, col_s);
            rocke_value_t* ch_i = rocke_b_add(b, ch_blk, ch8);
            rocke_value_t* c_n_in = rocke_b_const_i32(b, n_in);
            rocke_value_t* ok0 = rocke_b_cmp_lt(b, idx, c_n_in);
            rocke_value_t* ok1 = rocke_b_cmp_lt(b, img_i, c_N);
            rocke_value_t* ok01 = rocke_b_land(b, ok0, ok1);
            rocke_value_t* ge = rocke_b_cmp_ge(b, wo, c0);
            rocke_value_t* lt = rocke_b_cmp_lt(b, wo, c_Wo);
            rocke_value_t* in_row = rocke_b_land(b, ge, lt);
            rocke_value_t* ok3 = rocke_b_cmp_lt(b, ch_i, c_G);
            rocke_value_t* ok23 = rocke_b_land(b, in_row, ok3);
            rocke_value_t* ok = rocke_b_land(b, ok01, ok23);
            rocke_value_t* img_b = rocke_b_mul(b, img_i, c_dy_img_bytes);
            rocke_value_t* col_b = rocke_b_mul(b, wo, c_px_bytes);
            rocke_value_t* c2s = rocke_b_const_i32(b, 2);
            rocke_value_t* ch_b = rocke_b_mul(b, ch_i, c2s);
            rocke_value_t* lane_b = rocke_b_add(b, col_b, ch_b);
            rocke_value_t* g_off = rocke_b_add(b, img_b, lane_b);
            in_plan[i].px = px;
            in_plan[i].ch8 = ch8;
            in_plan[i].g_off = rocke_b_select(b, ok, g_off, oob_lane);
        }
        rocke_value_t* f_u = rocke_b_mul(b, f, c_U);
        for(int ph = 0; ph < KWP; ph++)
        {
            rocke_value_t* c_4ph = rocke_b_const_i32(b, 4 * ph);
            rocke_value_t* t4 = rocke_b_add(b, t, c_4ph);
            rocke_value_t* col = rocke_b_add(b, pair2, t4);
            b_px[ph] = rocke_b_add(b, f_u, col);
        }

        dw_mfma_ctx cx;
        memset(&cx, 0, sizeof cx);
        cx.b = b;
        cx.is_bf16 = is_bf16;
        cx.KH = KH;
        cx.PAD = PAD;
        cx.Ho = Ho;
        cx.Wo = Wo;
        cx.G = G;
        cx.ROWS = ROWS;
        cx.W = p->W;
        cx.split_h = split_h;
        cx.c0 = c0;
        cx.oob_uniform = oob_uniform;
        cx.h0 = h0;
        cx.a_rsrc = a_rsrc;
        cx.d_rsrc = d_rsrc;
        cx.o_smem = o_smem;
        cx.io_type = io_type;
        cx.in_plan = in_plan;
        cx.in_passes = in_passes;
        cx.drain = drain;
        cx.n_drain = n_drain;
        cx.c_dy_row_bytes = rocke_b_const_i32(b, Wo * G * 2);
        cx.c_dx_row_bytes = rocke_b_const_i32(b, p->W * G * 2);
        cx.c_Ho = split_h ? rocke_b_const_i32(b, Ho) : NULL;
        cx.c_H = (split_h && p->H % ROWS != 0) ? rocke_b_const_i32(b, p->H) : NULL;

        /* Live dY rows (Python row_taps: same rule as the VALU form). */
        dw_win_ctx tc;
        memset(&tc, 0, sizeof tc);
        tc.KH = KH;
        tc.PAD = PAD;
        tc.Ho = Ho;
        tc.ROWS = ROWS;
        tc.split_h = split_h;
        int n_live = 0;
        for(int y = 0; y < n_rows; y++)
        {
            if(dw_win_row_taps(&tc, y, rs) > 0)
                live_rows[n_live++] = y;
        }
        /* pending rows live in a ring of PF slots: live row li in slot li % PF.
         * Row li is consumed (stored to LDS) before row li + PF is loaded. */
        for(int li = 0; li < n_live && li < PF; li++)
            dw_mfma_load_row(&cx, live_rows[li], &pending[(li % PF) * in_passes]);

        int phase = 0;
        int live_idx = 0;
        rocke_value_t* pend_val = NULL;
        rocke_value_t* pend_row = NULL;
        for(int i = 0; i < KH; i++)
            acc[i] = zero_acc;
        for(int y = 0; y < n_rows; y++)
        {
            const int n_rs = dw_win_row_taps(&tc, y, rs);
            rocke_value_t** raw = (n_rs > 0) ? &pending[(live_idx % PF) * in_passes] : NULL;
            if(raw != NULL || pend_val != NULL)
            {
                rocke_value_t* buf = rocke_b_const_i32(b, phase % 2);
                phase++;
                if(raw != NULL)
                {
                    for(int i = 0; i < in_passes; i++)
                    {
                        rocke_value_t* idx[3] = {buf, in_plan[i].px, in_plan[i].ch8};
                        rocke_b_smem_store_vN(b, i_smem, idx, 3, raw[i], 8);
                    }
                }
                if(pend_val != NULL)
                {
                    rocke_value_t* idx[3] = {buf, o_col, o_ch};
                    rocke_b_smem_store_vN(b, o_smem, idx, 3, pend_val, 4);
                }
                rocke_b_sync_lds_only(b);
                if(raw != NULL)
                {
                    for(int ph = 0; ph < KWP; ph++)
                    {
                        rocke_value_t* idx[3] = {buf, b_px[ph], wave_ch};
                        xs[ph] = rocke_b_smem_load_vN(b, i_smem, idx, 3, io_type, 8);
                    }
                }
                if(pend_val != NULL)
                {
                    dw_mfma_drain_out(&cx, buf, pend_row);
                    pend_val = NULL;
                    pend_row = NULL;
                }
            }
            if(n_rs > 0)
            {
                const int k_next = live_idx + PF;
                if(k_next < n_live)
                    dw_mfma_load_row(&cx, live_rows[k_next], &pending[(k_next % PF) * in_passes]);
                for(int ph = 0; ph < KWP; ph++)
                {
                    for(int ri = 0; ri < n_rs; ri++)
                    {
                        const int r = rs[ri];
                        const int slot = (y + r - (KH - 1)) % KH;
                        acc[slot] = is_bf16 ? rocke_b_mfma_f32_16x16x32_bf16(
                                                  b, weights[r * KWP + ph], xs[ph], acc[slot])
                                            : rocke_b_mfma_f32_16x16x32_f16(
                                                  b, weights[r * KWP + ph], xs[ph], acc[slot]);
                    }
                }
                live_idx++;
            }
            const int hi_local = y - (KH - 1);
            if(hi_local < 0 || hi_local >= ROWS)
                continue;
            const int slot = hi_local % KH;
            pend_val = is_bf16 ? rocke_b_vec_trunc_f32_to_bf16(b, acc[slot])
                               : rocke_b_vec_trunc_f32_to_f16(b, acc[slot]);
            pend_row = dw_mfma_dx_row_part(&cx, hi_local);
            acc[slot] = zero_acc;
        }
        if(pend_val != NULL)
        {
            rocke_value_t* buf = rocke_b_const_i32(b, phase % 2);
            rocke_value_t* idx[3] = {buf, o_col, o_ch};
            rocke_b_smem_store_vN(b, o_smem, idx, 3, pend_val, 4);
            rocke_b_sync_lds_only(b);
            dw_mfma_drain_out(&cx, buf, pend_row);
        }
    }
    result = rocke_ir_builder_kernel(b);

done:
    free(weights);
    free(drain);
    free(in_plan);
    free(b_px);
    free(xs);
    free(pending);
    free(acc);
    free(rs);
    free(live_rows);
    return result;
}

rocke_kernel_def_t* rocke_build_direct_depthwise_dgrad_win(
    rocke_ir_builder_t* b, const rocke_direct_depthwise_dgrad_win_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];

    if(rocke_direct_depthwise_dgrad_win_validate(spec, reason, sizeof reason) != ROCKE_OK)
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE;
        return NULL;
    }
    if(!rocke_direct_depthwise_dgrad_win_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_VALUE;
        return NULL;
    }
    if(spec->mfma)
        return dw_win_build_mfma(b, spec);

    const rocke_direct_conv_problem_t* p = &spec->problem;
    const int BW = spec->block_w;
    const int CPL = spec->ch_per_lane;
    const int G = p->groups;
    const int KH = p->KH;
    const int KW = p->KW;
    const int PAD = p->PAD;
    const int Ho = p->H + 2 * PAD - KH + 1; /* stride 1 (validated) */
    const int Wo = p->W + 2 * PAD - KW + 1;
    const int ROWS = rocke_direct_depthwise_dgrad_win_rows_per_block(spec);
    const int H_TILES = rocke_direct_depthwise_dgrad_win_h_tiles(spec);
    const bool split_h = H_TILES > 1;
    const int WIN = BW + KW - 1;
    const int N_PAIRS = (KW + 1) / 2;
    const int THREADS = rocke_direct_depthwise_dgrad_win_threads_per_block(spec);
    const int BLOCK_CH = rocke_direct_depthwise_dgrad_win_block_ch(spec);
    const int is_bf16 = (p->dtype && strcmp(p->dtype, "bf16") == 0);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);

    const rocke_type_t* io_type = rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16");
    const rocke_type_t* ioptr = rocke_ptr_type(b, io_type, "global");
    rocke_param_opts_t ro = {0};
    ro.noalias = true;
    ro.noalias_set = true;
    ro.readonly = true;
    ro.readonly_set = true;
    ro.align = 16;
    ro.align_set = true;
    rocke_param_opts_t wo_opts = {0};
    wo_opts.noalias = true;
    wo_opts.noalias_set = true;
    wo_opts.writeonly = true;
    wo_opts.writeonly_set = true;
    wo_opts.align = 16;
    wo_opts.align_set = true;
    rocke_param_opts_t none = {0};

    rocke_value_t* A = rocke_b_param(b, "A", ioptr, &ro);
    rocke_value_t* Bp = rocke_b_param(b, "B", ioptr, &ro);
    rocke_value_t* D = rocke_b_param(b, "D", ioptr, &wo_opts);
    rocke_value_t* A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), &none);
    rocke_value_t* B_bytes = rocke_b_param(b, "B_bytes", rocke_i32(), &none);
    rocke_value_t* D_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), &none);

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* oob_lane = rocke_b_const_i32(b, ROCKE_DW_DGRAD_WIN_OOB_LANE);
    rocke_value_t* oob_uniform = rocke_b_const_i32(b, ROCKE_DW_DGRAD_WIN_OOB_UNIFORM);
    rocke_value_t* zero_f32 = rocke_b_const_f32(b, 0.0f);
    rocke_value_t* zero_acc = (CPL > 1) ? rocke_b_vector_splat(b, zero_f32, CPL) : zero_f32;

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* by = rocke_b_block_id_y(b);
    rocke_value_t* bz = rocke_b_block_id_z(b);
    rocke_value_t* n;
    rocke_value_t* h0 = NULL;
    if(split_h)
    {
        rocke_value_t* c_h_tiles = rocke_b_const_i32(b, H_TILES);
        n = rocke_b_div(b, bz, c_h_tiles);
        rocke_value_t* tile = rocke_b_mod(b, bz, c_h_tiles);
        rocke_value_t* c_rows = rocke_b_const_i32(b, ROWS);
        h0 = rocke_b_mul(b, tile, c_rows);
    }
    else
    {
        n = bz;
    }
    rocke_value_t* c_bw = rocke_b_const_i32(b, BW);
    rocke_value_t* wi0 = rocke_b_mul(b, bx, c_bw);
    rocke_value_t* lane_ch = tid;
    if(CPL > 1)
    {
        rocke_value_t* c_cpl = rocke_b_const_i32(b, CPL);
        lane_ch = rocke_b_mul(b, tid, c_cpl);
    }
    rocke_value_t* c_block_ch = rocke_b_const_i32(b, BLOCK_CH);
    rocke_value_t* ch_base = rocke_b_mul(b, by, c_block_ch);
    rocke_value_t* ch = rocke_b_add(b, ch_base, lane_ch);
    rocke_value_t* c_groups = rocke_b_const_i32(b, G);
    rocke_value_t* ch_ok = rocke_b_cmp_lt(b, ch, c_groups);
    rocke_value_t* c_two = rocke_b_const_i32(b, 2);
    rocke_value_t* ch_bytes = rocke_b_mul(b, ch, c_two);
    rocke_value_t* c_tap_bytes = rocke_b_const_i32(b, KH * KW * 2);
    rocke_value_t* w_base = rocke_b_mul(b, ch, c_tap_bytes);
    rocke_value_t* w_lane = rocke_b_select(b, ch_ok, w_base, oob_lane);

    rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, A, A_bytes);
    rocke_value_t* b_rsrc = rocke_b_buffer_rsrc(b, Bp, B_bytes);
    rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, D, D_bytes);

    dw_win_ctx cx;
    memset(&cx, 0, sizeof cx);
    cx.b = b;
    cx.is_bf16 = is_bf16;
    cx.KH = KH;
    cx.KW = KW;
    cx.PAD = PAD;
    cx.Ho = Ho;
    cx.ROWS = ROWS;
    cx.WIN = WIN;
    cx.CPL = CPL;
    cx.split_h = split_h;
    cx.c0 = c0;
    cx.oob_uniform = oob_uniform;
    cx.h0 = h0;
    cx.a_rsrc = a_rsrc;

    /* Scratch arrays (freed before every return below). */
    const int n_wt = (KH * KW > 0) ? KH * KW : 1;
    rocke_value_t** raw_w = (rocke_value_t**)calloc((size_t)n_wt, sizeof(rocke_value_t*));
    rocke_value_t** weights = (rocke_value_t**)calloc((size_t)n_wt, sizeof(rocke_value_t*));
    rocke_value_t** win_lane = (rocke_value_t**)calloc((size_t)WIN, sizeof(rocke_value_t*));
    rocke_value_t** out_lane = (rocke_value_t**)calloc((size_t)BW, sizeof(rocke_value_t*));
    rocke_value_t** acc = (rocke_value_t**)calloc((size_t)KH * BW, sizeof(rocke_value_t*));
    rocke_value_t** buf_a = (rocke_value_t**)calloc((size_t)WIN, sizeof(rocke_value_t*));
    rocke_value_t** buf_b = (rocke_value_t**)calloc((size_t)WIN, sizeof(rocke_value_t*));
    rocke_value_t** opnd = (rocke_value_t**)calloc((size_t)WIN, sizeof(rocke_value_t*));
    rocke_value_t** tail = (rocke_value_t**)calloc((size_t)BW, sizeof(rocke_value_t*));
    int* rs = (int*)calloc((size_t)KH, sizeof(int));
    int* live_rows = (int*)calloc((size_t)(ROWS + KH - 1), sizeof(int));
    rocke_kernel_def_t* result = NULL;
    if(!raw_w || !weights || !win_lane || !out_lane || !acc || !buf_a || !buf_b || !opnd || !tail
       || !rs || !live_rows)
    {
        if(b->status == ROCKE_OK)
            b->status = ROCKE_ERR_OOM;
        goto done;
    }
    cx.win_lane = win_lane;

    /* Preload W: f32 per tap (CPL-wide) or 16-bit tap pairs for dot2. */
    if(spec->dot2)
    {
        for(int r = 0; r < KH; r++)
        {
            for(int s = 0; s < KW; s++)
            {
                rocke_value_t* c_off = rocke_b_const_i32(b, (r * KW + s) * 2);
                rocke_value_t* off = rocke_b_add(b, w_lane, c_off);
                raw_w[r * KW + s] = dw_win_load_half(&cx, b_rsrc, off);
            }
        }
        rocke_value_t* zero_half = NULL;
        if(KW % 2)
            zero_half = is_bf16 ? rocke_b_trunc_f32_to_bf16(b, zero_f32)
                                : rocke_b_trunc_f32_to_f16(b, zero_f32);
        for(int r = 0; r < KH; r++)
        {
            for(int k = 0; k < N_PAIRS; k++)
            {
                rocke_value_t* comps[2];
                comps[0] = raw_w[r * KW + 2 * k];
                comps[1] = (2 * k + 1 < KW) ? raw_w[r * KW + 2 * k + 1] : zero_half;
                weights[r * N_PAIRS + k] = rocke_b_vec_pack(b, comps, 2, io_type);
            }
        }
    }
    else
    {
        for(int r = 0; r < KH; r++)
        {
            for(int s = 0; s < KW; s++)
            {
                rocke_value_t* comps[8];
                for(int i = 0; i < CPL; i++)
                {
                    rocke_value_t* c_off = rocke_b_const_i32(b, ((i * KH + r) * KW + s) * 2);
                    rocke_value_t* off = rocke_b_add(b, w_lane, c_off);
                    rocke_value_t* h = dw_win_load_half(&cx, b_rsrc, off);
                    comps[i] = rocke_b_cast_to_f32(b, h);
                }
                weights[r * KW + s]
                    = (CPL > 1) ? rocke_b_vec_pack(b, comps, CPL, rocke_f32()) : comps[0];
            }
        }
    }

    {
        /* Lane parts of the window and store offsets (row-invariant). */
        rocke_value_t* c_px_bytes = rocke_b_const_i32(b, G * 2);
        rocke_value_t* c_Wo = rocke_b_const_i32(b, Wo);
        for(int t = 0; t < WIN; t++)
        {
            rocke_value_t* c_t = rocke_b_const_i32(b, PAD - (KW - 1) + t);
            rocke_value_t* wo = rocke_b_add(b, wi0, c_t);
            rocke_value_t* ge = rocke_b_cmp_ge(b, wo, c0);
            rocke_value_t* lt = rocke_b_cmp_lt(b, wo, c_Wo);
            rocke_value_t* in_row = rocke_b_land(b, ge, lt);
            rocke_value_t* ok = rocke_b_land(b, ch_ok, in_row);
            rocke_value_t* col = rocke_b_mul(b, wo, c_px_bytes);
            rocke_value_t* lane = rocke_b_add(b, ch_bytes, col);
            win_lane[t] = rocke_b_select(b, ok, lane, oob_lane);
        }
        rocke_value_t* c_W = rocke_b_const_i32(b, p->W);
        for(int j = 0; j < BW; j++)
        {
            rocke_value_t* c_j = rocke_b_const_i32(b, j);
            rocke_value_t* wi = rocke_b_add(b, wi0, c_j);
            rocke_value_t* lt = rocke_b_cmp_lt(b, wi, c_W);
            rocke_value_t* ok = rocke_b_land(b, ch_ok, lt);
            rocke_value_t* col = rocke_b_mul(b, wi, c_px_bytes);
            rocke_value_t* lane = rocke_b_add(b, ch_bytes, col);
            out_lane[j] = rocke_b_select(b, ok, lane, oob_lane);
        }
    }

    {
        rocke_value_t* c_dy_row_bytes = rocke_b_const_i32(b, Wo * G * 2);
        rocke_value_t* c_dx_row_bytes = rocke_b_const_i32(b, p->W * G * 2);
        rocke_value_t* c_Ho_n = rocke_b_const_i32(b, Ho);
        rocke_value_t* dy_n_row = rocke_b_mul(b, n, c_Ho_n);
        rocke_value_t* c_H_n = rocke_b_const_i32(b, p->H);
        rocke_value_t* dx_n_row = rocke_b_mul(b, n, c_H_n);
        rocke_value_t* c_Ho = split_h ? rocke_b_const_i32(b, Ho) : NULL;
        rocke_value_t* c_H = (split_h && p->H % ROWS != 0) ? rocke_b_const_i32(b, p->H) : NULL;
        cx.c_Ho = c_Ho;
        cx.dy_n_row = dy_n_row;
        cx.c_dy_row_bytes = c_dy_row_bytes;

        const int n_rows = ROWS + KH - 1;
        int n_live = 0;
        for(int y = 0; y < n_rows; y++)
        {
            if(dw_win_row_taps(&cx, y, rs) > 0)
                live_rows[n_live++] = y;
        }
        rocke_value_t** pending = buf_a;
        rocke_value_t** spare = buf_b;
        if(n_live > 0)
            dw_win_load_window(&cx, live_rows[0], pending);

        for(int i = 0; i < KH * BW; i++)
            acc[i] = zero_acc;

        int live_idx = 0;
        for(int y = 0; y < n_rows; y++)
        {
            const int n_rs = dw_win_row_taps(&cx, y, rs);
            if(n_rs > 0)
            {
                rocke_value_t** raw = pending;
                if(live_idx + 1 < n_live)
                    dw_win_load_window(&cx, live_rows[live_idx + 1], spare);
                if(spec->dot2)
                {
                    /* An odd KW's last pair (w[KW - 1], 0) reads tail[j] = (x[j], 0)
                     * so the zero weight never meets a dY value outside the
                     * receptive field; full pairs then only read t >= 2. The
                     * tail is the zero-extended 16-bit bits (the u16 load
                     * already zero-fills the high half: no pack), and runs
                     * first so the raw columns die early. */
                    const int t_lo = (KW % 2 == 0) ? 0 : (KW > 1 ? 2 : WIN);
                    for(int t = t_lo; t < WIN; t++)
                    {
                        rocke_value_t* comps[2];
                        comps[0] = raw[t];
                        comps[1] = (t > 0) ? raw[t - 1] : raw[t];
                        opnd[t] = rocke_b_vec_pack(b, comps, 2, io_type);
                    }
                    if(KW % 2)
                    {
                        const rocke_type_t* pair_t = rocke_vector_type(b, io_type, 2);
                        for(int j = 0; j < BW; j++)
                        {
                            rocke_value_t* bits = rocke_b_bitcast(b, raw[j], rocke_i16());
                            tail[j] = rocke_b_vec_bitcast(
                                b, rocke_b_zext(b, bits, rocke_i32()), pair_t);
                        }
                    }
                    for(int ri = 0; ri < n_rs; ri++)
                    {
                        const int r = rs[ri];
                        const int slot = (y + r - (KH - 1)) % KH;
                        for(int ki = 0; ki < N_PAIRS; ki++)
                        {
                            const int k = (KW % 2) ? (ki + N_PAIRS - 1) % N_PAIRS : ki;
                            for(int j = 0; j < BW; j++)
                            {
                                rocke_value_t* x = (2 * k + 1 == KW)
                                                       ? tail[j]
                                                       : opnd[j + KW - 1 - 2 * k];
                                acc[slot * BW + j] = rocke_b_fdot2(
                                    b, weights[r * N_PAIRS + k], x, acc[slot * BW + j]);
                            }
                        }
                    }
                }
                else
                {
                    for(int t = 0; t < WIN; t++)
                        opnd[t] = dw_win_widen(&cx, raw[t]);
                    for(int ri = 0; ri < n_rs; ri++)
                    {
                        const int r = rs[ri];
                        const int slot = (y + r - (KH - 1)) % KH;
                        for(int s = 0; s < KW; s++)
                        {
                            for(int j = 0; j < BW; j++)
                            {
                                rocke_value_t* x = opnd[j + KW - 1 - s];
                                rocke_value_t* w = weights[r * KW + s];
                                acc[slot * BW + j]
                                    = (CPL > 1) ? rocke_b_vector_fma(b, w, x, acc[slot * BW + j])
                                                : rocke_b_fma(b, w, x, acc[slot * BW + j]);
                            }
                        }
                    }
                }
                rocke_value_t** tmp = pending;
                pending = spare;
                spare = tmp;
                live_idx++;
            }

            const int hi_local = y - (KH - 1);
            if(hi_local < 0 || hi_local >= ROWS)
                continue;
            const int slot = hi_local % KH;
            rocke_value_t* hi;
            rocke_value_t* hi_ok = NULL;
            if(split_h)
            {
                rocke_value_t* c_hl = rocke_b_const_i32(b, hi_local);
                hi = rocke_b_add(b, h0, c_hl);
                if(c_H != NULL)
                    hi_ok = rocke_b_cmp_lt(b, hi, c_H);
            }
            else
            {
                hi = rocke_b_const_i32(b, hi_local);
            }
            rocke_value_t* dx_row = dw_win_row_part(&cx, dx_n_row, hi, c_dx_row_bytes, hi_ok);
            for(int j = 0; j < BW; j++)
            {
                rocke_value_t* off = rocke_b_add(b, out_lane[j], dx_row);
                rocke_value_t* a = acc[slot * BW + j];
                if(CPL == 1)
                {
                    if(is_bf16)
                    {
                        rocke_value_t* v = rocke_b_trunc_f32_to_bf16(b, a);
                        rocke_b_buffer_store_bf16(b, d_rsrc, off, c0, v);
                    }
                    else
                    {
                        rocke_value_t* v = rocke_b_trunc_f32_to_f16(b, a);
                        rocke_b_buffer_store_f16(b, d_rsrc, off, c0, v);
                    }
                }
                else if(is_bf16)
                {
                    rocke_value_t* v = rocke_b_vec_trunc_f32_to_bf16(b, a);
                    rocke_b_buffer_store_vN_bf16(b, d_rsrc, off, c0, v, CPL / 2);
                }
                else
                {
                    rocke_value_t* v = rocke_b_vec_trunc_f32_to_f16(b, a);
                    rocke_b_buffer_store_vN_f16(b, d_rsrc, off, c0, v, CPL / 2);
                }
                acc[slot * BW + j] = zero_acc;
            }
        }
    }
    result = rocke_ir_builder_kernel(b);

done:
    free(raw_w);
    free(weights);
    free(win_lane);
    free(out_lane);
    free(acc);
    free(buf_a);
    free(buf_b);
    free(opnd);
    free(tail);
    free(rs);
    free(live_rows);
    return result;
}
