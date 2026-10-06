// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * conv_direct_grouped_build_generic.cpp -- C99 port of build_direct_conv, the
 * generic row-streaming direct grouped convolution kernel (any cpg / kpg that is
 * a multiple of 4; the transposed-fprop main kernel of the grouped direct-MFMA
 * dgrad) in rocke/instances/common/conv_direct_grouped.py.
 *
 *   Python (conv_direct_grouped.py)        C99 (this file)
 *   ------------------------------------   -------------------------------------
 *   build_direct_conv(spec, arch)          rocke_build_direct_conv / _new
 *     issue_dram_load(y, g_tile_val)       rocke_dcg__issue_dram_load
 *     store_to_lds(loads, lds)             rocke_dcg__store_to_lds
 *     _alloc_row_buffers()                 rocke_dcg__alloc_row_buffers
 *     _row_y(y_l)                          rocke_dcg__row_y
 *   (+ convenience: build -> lower .ll)    rocke_direct_conv_lower_to_llvm
 *
 * Every builder call is issued in the Python evaluation order: Python evaluates
 * call arguments left to right, C leaves sibling-argument order unspecified, so
 * every expression with two or more IR-emitting operands is split into
 * sequenced locals. Python closures read their enclosing locals at call time
 * (late binding: the persistent cell loop rebinds n / q_tile_start_lds), so the
 * closure state lives in a context struct that the body updates in place.
 *
 * Unrolled-structure tables (chunk meta, the loads of a staged row, the
 * accumulator ring, the preloaded weight fragments, the stage_out store meta)
 * are sized from the spec and allocated from the builder arena, so the port has
 * no fixed caps the Python builder does not have.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.grid.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/instance_conv_direct_grouped.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

/* XCD_TILES_NUM_XCDS (the xcd_tiles remap) and the persistent grid's layout. */
#define ROCKE_DCG_XCD_TILES_NUM_XCDS 8
#define ROCKE_DCG_NUM_XCD 8
#define ROCKE_DCG_BLOCKS_PER_XCD 32

/* One staged-row load of issue_dram_load: (a_vec, lds_idx). */
typedef struct rocke_dcg_load
{
    rocke_value_t* vec;
    rocke_value_t* lds_idx;
} rocke_dcg_load_t;

/* chunk_meta entry (Python dict). */
typedef struct rocke_dcg_chunk
{
    rocke_value_t* chunk_idx;
    rocke_value_t* ch_block;
    rocke_value_t* W_lds;
    rocke_value_t* in_bounds;
    rocke_value_t* group_in_wg;
} rocke_dcg_chunk_t;

/* stage_out store meta: (_ok, global byte offset, LDS index). */
typedef struct rocke_dcg_so
{
    rocke_value_t* ok;
    rocke_value_t* gbytes;
    rocke_value_t* lidx;
} rocke_dcg_so_t;

/* Closure state of build_direct_conv (the locals issue_dram_load reads). */
typedef struct rocke_dcg_ctx
{
    rocke_ir_builder_t* b;
    const rocke_direct_conv_problem_t* p;
    const rocke_type_t* io_type;
    bool is_bf16;
    int LOAD_VEC;
    int LDS_PAD;
    rocke_value_t* c0;
    rocke_value_t* c_BG;
    rocke_value_t* c_cpg;
    rocke_value_t* c_half_bytes;
    rocke_value_t* oob_sentinel;
    rocke_value_t* by;
    rocke_value_t* n;
    rocke_value_t* q_tile_start_lds;
    rocke_value_t* a_rsrc;
    const rocke_tensor_descriptor_t* a_desc;
    rocke_dcg_chunk_t* chunk_meta;
    int PASSES;
} rocke_dcg_ctx_t;

static void* rocke_dcg__alloc(rocke_ir_builder_t* b, size_t bytes)
{
    void* p = rocke_arena_alloc(&b->arena, bytes ? bytes : 1);
    if(p == NULL)
    {
        if(b->status == ROCKE_OK)
        {
            b->status = ROCKE_ERR_OOM;
            snprintf(b->err, sizeof b->err, "build_direct_conv: OOM");
        }
        return NULL;
    }
    memset(p, 0, bytes ? bytes : 1);
    return p;
}

/* Arena copy of a name (scf iv / iter-arg names). */
static const char* rocke_dcg__strdup(rocke_ir_builder_t* b, const char* src)
{
    size_t n = strlen(src) + 1;
    char* out = (char*)rocke_dcg__alloc(b, n);
    if(out != NULL)
    {
        memcpy(out, src, n);
    }
    return out;
}

/* Arena copy of "<fmt % (y, qt, r)>" (the y{y}_qt{qt}_r{r} loop tags). */
static const char* rocke_dcg__name(rocke_ir_builder_t* b, const char* fmt, int y, int qt, int r)
{
    char buf[96];
    snprintf(buf, sizeof buf, fmt, y, qt, r);
    return rocke_dcg__strdup(b, buf);
}

/* Python modulo (result has the divisor's sign) for the accumulator ring. */
static int rocke_dcg__pymod(int a, int m)
{
    int r = a % m;
    return r < 0 ? r + m : r;
}

static rocke_value_t* rocke_dcg__buf_load(const rocke_dcg_ctx_t* ctx,
                                          rocke_value_t* rsrc,
                                          rocke_value_t* voff,
                                          int dwords)
{
    if(ctx->is_bf16)
    {
        return rocke_b_buffer_load_vN_bf16(ctx->b, rsrc, voff, ctx->c0, dwords);
    }
    return rocke_b_buffer_load_vN_f16(ctx->b, rsrc, voff, ctx->c0, dwords);
}

static void rocke_dcg__buf_store(const rocke_dcg_ctx_t* ctx,
                                 rocke_value_t* rsrc,
                                 rocke_value_t* voff,
                                 rocke_value_t* val,
                                 int dwords)
{
    if(ctx->is_bf16)
    {
        rocke_b_buffer_store_vN_bf16(ctx->b, rsrc, voff, ctx->c0, val, dwords);
    }
    else
    {
        rocke_b_buffer_store_vN_f16(ctx->b, rsrc, voff, ctx->c0, val, dwords);
    }
}

static rocke_value_t* rocke_dcg__trunc(const rocke_dcg_ctx_t* ctx, rocke_value_t* v)
{
    if(ctx->is_bf16)
    {
        return rocke_b_vec_trunc_f32_to_bf16(ctx->b, v);
    }
    return rocke_b_vec_trunc_f32_to_f16(ctx->b, v);
}

static rocke_value_t* rocke_dcg__mfma(const rocke_dcg_ctx_t* ctx,
                                      bool fold_k32,
                                      rocke_value_t* a,
                                      rocke_value_t* bv,
                                      rocke_value_t* acc)
{
    if(fold_k32)
    {
        return ctx->is_bf16 ? rocke_b_mfma_f32_16x16x32_bf16(ctx->b, a, bv, acc)
                            : rocke_b_mfma_f32_16x16x32_f16(ctx->b, a, bv, acc);
    }
    return ctx->is_bf16 ? rocke_b_mfma_f32_16x16x16_bf16(ctx->b, a, bv, acc)
                        : rocke_b_mfma_f32_16x16x16_f16(ctx->b, a, bv, acc);
}

/* TensorDescriptor.offset(b, **coords) -> (offset, valid). */
static rocke_value_t* rocke_dcg__offset(rocke_ir_builder_t* b,
                                        const rocke_tensor_descriptor_t* desc,
                                        const char** names,
                                        rocke_value_t** values,
                                        int n)
{
    rocke_value_t* off = NULL;
    rocke_value_t* valid = NULL;
    if(!rocke_transforms_descriptor_offset(b, desc, names, values, n, &off, &valid))
    {
        return NULL;
    }
    return off;
}

/* issue_dram_load(y_iter_val, g_tile_val=None) -> PASSES (a_vec, lds_idx). */
static rocke_dcg_load_t* rocke_dcg__issue_dram_load(rocke_dcg_ctx_t* ctx,
                                                    rocke_value_t* y_iter_val,
                                                    rocke_value_t* g_tile_val)
{
    rocke_ir_builder_t* b = ctx->b;
    rocke_value_t* g_tile = g_tile_val != NULL ? g_tile_val : ctx->by;
    rocke_dcg_load_t* out
        = (rocke_dcg_load_t*)rocke_dcg__alloc(b, sizeof(rocke_dcg_load_t) * (size_t)ctx->PASSES);
    int i;
    if(out == NULL)
    {
        return NULL;
    }
    for(i = 0; i < ctx->PASSES; ++i)
    {
        const rocke_dcg_chunk_t* cm = &ctx->chunk_meta[i];
        rocke_value_t* abs_group;
        rocke_value_t* c_val;
        rocke_value_t* a_off;
        rocke_value_t* addr_valid = NULL;
        rocke_value_t* valid;
        rocke_value_t* safe_off;
        rocke_value_t* a_vec;
        rocke_value_t* zero_vec;
        rocke_value_t* lds_idx;
        const char* names[5] = {"n", "y_iter", "q_pos", "W_lds_pos", "c"};
        rocke_value_t* vals[5];

        abs_group = rocke_b_add(b, rocke_b_mul(b, g_tile, ctx->c_BG), cm->group_in_wg);
        {
            rocke_value_t* m1 = rocke_b_mul(b, abs_group, ctx->c_cpg);
            rocke_value_t* cv = rocke_b_const_i32(b, ctx->LOAD_VEC);
            rocke_value_t* m2 = rocke_b_mul(b, cm->ch_block, cv);
            c_val = rocke_b_add(b, m1, m2);
        }
        vals[0] = ctx->n;
        vals[1] = y_iter_val;
        vals[2] = ctx->q_tile_start_lds;
        vals[3] = cm->W_lds;
        vals[4] = c_val;
        if(!rocke_transforms_descriptor_offset(b, ctx->a_desc, names, vals, 5, &a_off, &addr_valid))
        {
            return NULL;
        }
        valid = rocke_b_land(b, addr_valid, cm->in_bounds);
        {
            rocke_value_t* bytes = rocke_b_mul(b, a_off, ctx->c_half_bytes);
            safe_off = rocke_b_select(b, valid, bytes, ctx->oob_sentinel);
        }
        a_vec = rocke_dcg__buf_load(ctx, ctx->a_rsrc, safe_off, ctx->LOAD_VEC / 2);
        zero_vec = rocke_b_zero_vec(b, ctx->io_type, ctx->LOAD_VEC);
        a_vec = rocke_b_select(b, valid, a_vec, zero_vec);
        lds_idx = rocke_b_mul(b, cm->chunk_idx, rocke_b_const_i32(b, ctx->LOAD_VEC));
        if(ctx->LDS_PAD)
        {
            rocke_value_t* cp = rocke_b_const_i32(b, ctx->LDS_PAD);
            rocke_value_t* padv = rocke_b_mul(b, cm->W_lds, cp);
            lds_idx = rocke_b_add(b, lds_idx, padv);
        }
        out[i].vec = a_vec;
        out[i].lds_idx = lds_idx;
    }
    return out;
}

/* store_to_lds(loads, lds). */
static void
    rocke_dcg__store_to_lds(rocke_dcg_ctx_t* ctx, const rocke_dcg_load_t* loads, rocke_value_t* lds)
{
    int i;
    if(loads == NULL)
    {
        return;
    }
    for(i = 0; i < ctx->PASSES; ++i)
    {
        rocke_value_t* idx[2];
        idx[0] = ctx->c0;
        idx[1] = loads[i].lds_idx;
        rocke_b_smem_store_vN(ctx->b, lds, idx, 2, loads[i].vec, ctx->LOAD_VEC);
    }
}

/* _alloc_row_buffers() -> (A_smem, B_smem). */
static void rocke_dcg__alloc_row_buffers(rocke_dcg_ctx_t* ctx,
                                         bool double_buffer,
                                         int lds_total_fp16,
                                         rocke_value_t** A_smem,
                                         rocke_value_t** B_smem)
{
    int shape[2];
    shape[0] = 1;
    shape[1] = lds_total_fp16;
    *A_smem = rocke_b_smem_alloc(ctx->b, ctx->io_type, shape, 2, "lds_a");
    *B_smem = double_buffer ? rocke_b_smem_alloc(ctx->b, ctx->io_type, shape, 2, "lds_b") : *A_smem;
}

rocke_kernel_def_t* rocke_build_direct_conv(rocke_ir_builder_t* b,
                                            const rocke_direct_conv_spec_t* spec,
                                            const char* arch)
{
    rocke_dcg_ctx_t ctx;
    const rocke_direct_conv_problem_t* p;
    char reason[ROCKE_ERR_MSG_CAP];
    rocke_status_t vst;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* spec.validate(); ok, why = is_valid_spec(spec, arch=arch) */
    vst = rocke_direct_conv_validate(spec, reason, sizeof reason);
    if(vst != ROCKE_OK)
    {
        snprintf(b->err, sizeof b->err, "%s", reason);
        b->status = vst;
        return NULL;
    }
    if(!rocke_direct_conv_is_valid_spec(spec, arch, reason, sizeof reason))
    {
        ROCKE_ERR_SNPRINTF(
            b->err, sizeof b->err, "invalid DirectConvSpec for %s: %s", arch, reason);
        b->status = ROCKE_ERR_VALUE;
        return NULL;
    }

    p = &spec->problem;
    memset(&ctx, 0, sizeof ctx);
    ctx.b = b;
    ctx.p = p;
    ctx.io_type = rocke_b_io_ir_type(b, p->dtype ? p->dtype : "fp16");
    if(ctx.io_type == NULL)
    {
        return NULL;
    }
    ctx.is_bf16 = p->dtype != NULL && strcmp(p->dtype, "bf16") == 0;

    const int BLOCK_Q = spec->block_q;
    const int BLOCK_GROUPS = spec->block_groups;
    const int WAVES_Q = spec->waves_q;
    const int WAVES_K = spec->waves_k;
    const int WAVE = spec->wave_size;
    const int THREADS = rocke_direct_conv_threads_per_block(spec);
    const int Ho = (p->H + 2 * p->PAD - p->KH) / p->stride + 1; /* stride == 1 (validated) */
    const int Wo = (p->W + 2 * p->PAD - p->KW) / p->stride + 1;
    const int total_c = rocke_direct_conv_problem_total_c(p);
    const int total_k = rocke_direct_conv_problem_total_k(p);

    const bool FOLD_K32 = spec->fold_k32;
    const int K_ATOM_SIZE = FOLD_K32 ? 32 : 16;
    const int LOAD_VEC = FOLD_K32 ? 8 : 4;
    const int N_K_ATOMS
        = FOLD_K32 ? p->cpg / K_ATOM_SIZE : (p->cpg + K_ATOM_SIZE - 1) / K_ATOM_SIZE;
    const int N_K_LOCAL = N_K_ATOMS / WAVES_K;
    const int N_M_TILES = (p->kpg + 15) / 16;
    const int WAVES_M = spec->waves_m;
    const int M_LOCAL = N_M_TILES / WAVES_M;
    const int N_VECS = p->cpg / LOAD_VEC;
    const int BLOCK_Q_WAVE = BLOCK_Q / WAVES_Q;
    const int q_subtiles = BLOCK_Q_WAVE / 16;
    const int LDS_W = (BLOCK_Q - 1) * p->stride + p->KW;
    const int NUM_CHUNKS = LDS_W * BLOCK_GROUPS * N_VECS;
    const int PASSES = (NUM_CHUNKS + THREADS - 1) / THREADS;
    int lds_total_fp16 = PASSES * THREADS * LOAD_VEC;
    const int LDS_PAD = spec->lds_pad;
    if(LDS_PAD)
    {
        lds_total_fp16
            += ((PASSES * THREADS + BLOCK_GROUPS * N_VECS - 1) / (BLOCK_GROUPS * N_VECS) + 1)
               * LDS_PAD;
    }
    const bool PRELOADS = WAVES_K > 1 || spec->preload_weights || spec->dgrad_fused_weights;
    const bool cpg_partial = (p->cpg % K_ATOM_SIZE) != 0;

    ctx.LOAD_VEC = LOAD_VEC;
    ctx.LDS_PAD = LDS_PAD;
    ctx.PASSES = PASSES;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", THREADS);
    if(spec->waves_per_eu > 0)
    {
        rocke_attr_set_int(b, &b->kernel->attrs, "waves_per_eu", spec->waves_per_eu);
    }

    rocke_value_t* A;
    rocke_value_t* Bp;
    rocke_value_t* D;
    rocke_value_t* A_bytes;
    rocke_value_t* B_bytes;
    rocke_value_t* D_bytes;
    {
        const rocke_type_t* io_ptr = rocke_ptr_type(b, ctx.io_type, "global");
        rocke_param_opts_t ro = {0};
        rocke_param_opts_t wo = {0};
        rocke_param_opts_t none = {0};
        ro.noalias = true;
        ro.noalias_set = true;
        ro.readonly = true;
        ro.readonly_set = true;
        ro.align = 16;
        ro.align_set = true;
        wo.noalias = true;
        wo.noalias_set = true;
        wo.writeonly = true;
        wo.writeonly_set = true;
        wo.align = 16;
        wo.align_set = true;
        A = rocke_b_param(b, "A", io_ptr, &ro);
        Bp = rocke_b_param(b, "B", io_ptr, &ro);
        D = rocke_b_param(b, "D", io_ptr, &wo);
        A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), &none);
        B_bytes = rocke_b_param(b, "B_bytes", rocke_i32(), &none);
        D_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), &none);
    }

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c1 = rocke_b_const_i32(b, 1);
    rocke_value_t* c_wave = rocke_b_const_i32(b, WAVE);
    rocke_value_t* c_BG = rocke_b_const_i32(b, BLOCK_GROUPS);
    rocke_value_t* c_BQ = rocke_b_const_i32(b, BLOCK_Q);
    rocke_value_t* c_cpg = rocke_b_const_i32(b, p->cpg);
    rocke_value_t* c_kpg = rocke_b_const_i32(b, p->kpg);
    rocke_value_t* c_W = rocke_b_const_i32(b, Wo);
    rocke_value_t* c_KW = rocke_b_const_i32(b, p->KW);
    rocke_value_t* c_N_K_LOCAL = rocke_b_const_i32(b, N_K_LOCAL);
    rocke_value_t* c_BG_cpg = rocke_b_const_i32(b, BLOCK_GROUPS * p->cpg + spec->lds_pad);
    rocke_value_t* c_half_bytes = rocke_b_const_i32(b, 2);
    rocke_value_t* oob_sentinel = rocke_b_const_i32(b, ((int64_t)1 << 31) - 1);
    ctx.c0 = c0;
    ctx.c_BG = c_BG;
    ctx.c_cpg = c_cpg;
    ctx.c_half_bytes = c_half_bytes;
    ctx.oob_sentinel = oob_sentinel;

    (void)rocke_b_zero_vec(b, ctx.io_type, 4); /* fp16x4_zero */
    rocke_value_t* zero_acc = rocke_b_zero_vec_f32(b, 4);

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    const int waves_per_group = WAVES_Q * WAVES_K;
    rocke_value_t* wave_id_full = rocke_b_div(b, tid, c_wave);
    rocke_value_t* wave_id_in_group
        = rocke_b_mod(b, wave_id_full, rocke_b_const_i32(b, waves_per_group));
    rocke_value_t* wave_group_idx
        = rocke_b_div(b, wave_id_full, rocke_b_const_i32(b, waves_per_group));
    rocke_value_t* wave_id_q = rocke_b_div(b, wave_id_in_group, rocke_b_const_i32(b, WAVES_K));
    rocke_value_t* wave_id_k = rocke_b_mod(b, wave_id_in_group, rocke_b_const_i32(b, WAVES_K));
    rocke_value_t* wm_base16 = NULL;
    if(WAVES_M > 1)
    {
        rocke_value_t* wave_id_m;
        wave_group_idx = rocke_b_div(b, wave_id_full, rocke_b_const_i32(b, WAVES_M));
        wave_id_m = rocke_b_mod(b, wave_id_full, rocke_b_const_i32(b, WAVES_M));
        wave_id_q = c0;
        wave_id_k = c0;
        wm_base16 = rocke_b_mul(b, wave_id_m, rocke_b_const_i32(b, M_LOCAL * 16));
    }
    rocke_value_t* k_atom_base = rocke_b_mul(b, wave_id_k, rocke_b_const_i32(b, N_K_LOCAL));

    rocke_value_t* lane = rocke_b_mod(b, tid, c_wave);
    rocke_value_t* c4 = rocke_b_div(b, lane, rocke_b_const_i32(b, 16));
    rocke_value_t* q_in_lane = rocke_b_mod(b, lane, rocke_b_const_i32(b, 16));

    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* by = rocke_b_block_id_y(b);
    rocke_value_t* bz = rocke_b_block_id_z(b);
    if(spec->xcd_tiles)
    {
        int grid[3];
        rocke_value_t* c_gx;
        rocke_value_t* c_gxy;
        rocke_value_t* lin;
        rocke_direct_conv_grid(spec, grid);
        c_gx = rocke_b_const_i32(b, grid[0]);
        c_gxy = rocke_b_const_i32(b, grid[0] * grid[1]);
        {
            rocke_value_t* m1 = rocke_b_mul(b, by, c_gx);
            rocke_value_t* m2 = rocke_b_mul(b, bz, c_gxy);
            rocke_value_t* inner = rocke_b_add(b, m1, m2);
            lin = rocke_b_add(b, bx, inner);
        }
        lin = rocke_chiplet_transform_chunked(b,
                                              lin,
                                              grid[0] * grid[1] * grid[2],
                                              ROCKE_DCG_XCD_TILES_NUM_XCDS,
                                              rocke_direct_conv_xcd_chunk(spec));
        if(lin == NULL)
        {
            return NULL;
        }
        bx = rocke_b_mod(b, lin, c_gx);
        {
            rocke_value_t* dv = rocke_b_div(b, lin, c_gx);
            rocke_value_t* cy = rocke_b_const_i32(b, grid[1]);
            by = rocke_b_mod(b, dv, cy);
        }
        bz = rocke_b_div(b, lin, c_gxy);
    }
    ctx.by = by;

    const int BLOCK_H = spec->block_h;
    const bool PERSISTENT = spec->persistent_grid;

    rocke_value_t* n = NULL;
    rocke_value_t* h_tile_start = NULL;
    rocke_value_t* xcd_start = NULL;
    rocke_value_t* wg_in_xcd = NULL;
    int n_h_tiles = 1;
    int q_tiles_p = 0, g_tiles_p = 0, n_cells_p = 0, rounds_per_block = 0;
    if(PERSISTENT)
    {
        int cells_per_xcd;
        rocke_value_t* xcd_id;
        n_h_tiles = (p->H + BLOCK_H - 1) / BLOCK_H;
        q_tiles_p = (Wo + BLOCK_Q - 1) / BLOCK_Q;
        g_tiles_p = p->groups / BLOCK_GROUPS;
        n_cells_p = p->N * n_h_tiles * q_tiles_p * g_tiles_p;
        cells_per_xcd = (n_cells_p + ROCKE_DCG_NUM_XCD - 1) / ROCKE_DCG_NUM_XCD;
        rounds_per_block
            = (cells_per_xcd + ROCKE_DCG_BLOCKS_PER_XCD - 1) / ROCKE_DCG_BLOCKS_PER_XCD;
        xcd_id = rocke_b_mod(b, bx, rocke_b_const_i32(b, ROCKE_DCG_NUM_XCD));
        wg_in_xcd = rocke_b_div(b, bx, rocke_b_const_i32(b, ROCKE_DCG_NUM_XCD));
        xcd_start = rocke_b_mul(b, xcd_id, rocke_b_const_i32(b, cells_per_xcd));
        n = rocke_b_const_i32(b, 0);
        h_tile_start = rocke_b_const_i32(b, 0);
        (void)rocke_b_const_i32(b, 0); /* q_tile_start_cell */
        (void)rocke_b_const_i32(b, 0); /* g_tile_cell */
    }
    else if(BLOCK_H > 0)
    {
        rocke_value_t* c_n_h_tiles;
        rocke_value_t* h_tile_idx;
        n_h_tiles = (p->H + BLOCK_H - 1) / BLOCK_H;
        c_n_h_tiles = rocke_b_const_i32(b, n_h_tiles);
        n = rocke_b_div(b, bz, c_n_h_tiles);
        h_tile_idx = rocke_b_mod(b, bz, c_n_h_tiles);
        h_tile_start = rocke_b_mul(b, h_tile_idx, rocke_b_const_i32(b, BLOCK_H));
    }
    else
    {
        n = bz;
    }
    rocke_value_t* g_tile = PERSISTENT ? NULL : by;
    rocke_value_t* g;
    if(PERSISTENT)
    {
        g = rocke_b_const_i32(b, 0);
    }
    else
    {
        g = rocke_b_add(b, rocke_b_mul(b, g_tile, c_BG), wave_group_idx);
    }
    rocke_value_t* block_q_start = rocke_b_mul(b, bx, c_BQ);
    rocke_value_t* q_tile_start;
    {
        rocke_value_t* cq = rocke_b_const_i32(b, BLOCK_Q_WAVE);
        rocke_value_t* mq = rocke_b_mul(b, wave_id_q, cq);
        q_tile_start = rocke_b_add(b, block_q_start, mq);
    }
    ctx.n = n;
    ctx.q_tile_start_lds = block_q_start;

    rocke_value_t* A_smem = NULL;
    rocke_value_t* B_smem = NULL;
    if(!spec->dgrad_weights_lds)
    {
        rocke_dcg__alloc_row_buffers(&ctx, spec->double_buffer, lds_total_fp16, &A_smem, &B_smem);
    }

    rocke_value_t* red_lds = NULL;
    rocke_value_t* lds_row_idx = NULL;
    if(WAVES_K > 1)
    {
        int shape[2];
        shape[0] = WAVES_Q * WAVES_K;
        shape[1] = WAVE * 4;
        red_lds = rocke_b_smem_alloc_f32(b, shape, 2, "red_lds");
        {
            rocke_value_t* ck = rocke_b_const_i32(b, WAVES_K);
            rocke_value_t* mq = rocke_b_mul(b, wave_id_q, ck);
            lds_row_idx = rocke_b_add(b, mq, wave_id_k);
        }
    }

    rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, A, A_bytes);
    rocke_value_t* b_rsrc = rocke_b_buffer_rsrc(b, Bp, B_bytes);
    rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, D, D_bytes);
    ctx.a_rsrc = a_rsrc;

    /* a_desc: A[N, H, W, total_c] with the h / w embeds. */
    {
        static const char* const a_coords[4] = {"n", "h", "w", "c"};
        int a_lengths[4];
        rocke_tensor_descriptor_t* a_naive;
        const rocke_transform_t* xforms[2];
        a_lengths[0] = p->N;
        a_lengths[1] = p->H;
        a_lengths[2] = p->W;
        a_lengths[3] = total_c;
        a_naive = rocke_tensor_descriptor_naive(b, "A", a_lengths, 4, NULL, a_coords, 4);
        {
            static const char* const h_upper[1] = {"y_iter"};
            int h_strides[1] = {1};
            xforms[0] = rocke_embed_bounded(b, h_upper, 1, "h", h_strides, -p->PAD, 0, p->H);
        }
        {
            static const char* const w_upper[2] = {"q_pos", "W_lds_pos"};
            int w_strides[2];
            w_strides[0] = p->stride;
            w_strides[1] = 1;
            xforms[1] = rocke_embed_bounded(b, w_upper, 2, "w", w_strides, -p->PAD, 0, p->W);
        }
        ctx.a_desc = rocke_tensor_descriptor_transform(b, a_naive, xforms, 2);
    }
    const int c_stride_gen = p->stride;
    const rocke_tensor_descriptor_t* b_desc;
    {
        static const char* const b_coords[4] = {"k_out", "r", "s", "c"};
        int lengths[4];
        lengths[0] = total_k;
        lengths[1] = p->KH;
        lengths[2] = p->KW;
        lengths[3] = p->cpg;
        b_desc = rocke_tensor_descriptor_naive(b, "B", lengths, 4, NULL, b_coords, 4);
    }
    const rocke_tensor_descriptor_t* d_desc;
    {
        static const char* const d_coords[4] = {"n", "h", "w", "k"};
        int lengths[4];
        lengths[0] = p->N;
        lengths[1] = Ho;
        lengths[2] = Wo;
        lengths[3] = total_k;
        d_desc = rocke_tensor_descriptor_naive(b, "D", lengths, 4, NULL, d_coords, 4);
    }
    const rocke_tensor_descriptor_t* chunk_desc;
    {
        static const char* const coords[3] = {"W_lds", "group_in_wg", "ch_block"};
        int lengths[3];
        rocke_tensor_descriptor_t* naive;
        const rocke_transform_t* xf[1];
        lengths[0] = LDS_W;
        lengths[1] = BLOCK_GROUPS;
        lengths[2] = N_VECS;
        naive = rocke_tensor_descriptor_naive(b, "chunk_unmerge", lengths, 3, NULL, coords, 3);
        xf[0] = rocke_unmerge_magic(b, "chunk_idx", coords, 3, lengths);
        chunk_desc = rocke_tensor_descriptor_transform(b, naive, xf, 1);
    }

    ctx.chunk_meta
        = (rocke_dcg_chunk_t*)rocke_dcg__alloc(b, sizeof(rocke_dcg_chunk_t) * (size_t)PASSES);
    if(ctx.chunk_meta == NULL)
    {
        return NULL;
    }
    for(int pass_idx = 0; pass_idx < PASSES; ++pass_idx)
    {
        rocke_value_t* chunk_idx = rocke_b_add(b, tid, rocke_b_const_i32(b, pass_idx * THREADS));
        const char* in_names[1] = {"chunk_idx"};
        rocke_value_t* in_values[1];
        const char* out_names[8];
        rocke_value_t* out_values[8];
        int n_out;
        rocke_dcg_chunk_t* cm = &ctx.chunk_meta[pass_idx];
        in_values[0] = chunk_idx;
        n_out = rocke_tensor_descriptor_unmerge_lower(
            b, chunk_desc, in_names, in_values, 1, out_names, out_values, 8);
        for(int i = 0; i < n_out; ++i)
        {
            if(out_names[i] == NULL)
                continue;
            if(strcmp(out_names[i], "ch_block") == 0)
                cm->ch_block = out_values[i];
            else if(strcmp(out_names[i], "W_lds") == 0)
                cm->W_lds = out_values[i];
            else if(strcmp(out_names[i], "group_in_wg") == 0)
                cm->group_in_wg = out_values[i];
        }
        cm->chunk_idx = chunk_idx;
        cm->in_bounds = rocke_b_cmp_lt(b, chunk_idx, rocke_b_const_i32(b, NUM_CHUNKS));
    }

    /* ---- Persistent cell loop ---- */
    rocke_for_t cell_loop;
    rocke_value_t* cell_idx = NULL;
    rocke_value_t* pg_in_bounds = NULL;
    rocke_value_t* pg_gt_v = NULL;
    memset(&cell_loop, 0, sizeof cell_loop);
    if(PERSISTENT)
    {
        rocke_iter_arg_t ia[1];
        rocke_value_t* rounds_c;
        ia[0].name = "pg_cell_idx_carry";
        ia[0].init = xcd_start;
        rounds_c = rocke_b_const_i32(b, rounds_per_block);
        cell_loop = rocke_b_scf_for_iter(b, c0, rounds_c, c1, ia, 1, "pg_round", false, false);
        if(cell_loop.body == NULL)
        {
            return NULL;
        }
        rocke_b_region_enter(b, cell_loop.body);
        rocke_value_t* pg_round_iv = cell_loop.iv;
        {
            rocke_value_t* cb = rocke_b_const_i32(b, ROCKE_DCG_BLOCKS_PER_XCD);
            rocke_value_t* mr = rocke_b_mul(b, pg_round_iv, cb);
            rocke_value_t* inner = rocke_b_add(b, wg_in_xcd, mr);
            cell_idx = rocke_b_add(b, xcd_start, inner);
        }
        pg_in_bounds = rocke_b_cmp_lt(b, cell_idx, rocke_b_const_i32(b, n_cells_p));
        rocke_value_t* c_qt_p = rocke_b_const_i32(b, q_tiles_p);
        (void)rocke_b_const_i32(b, n_h_tiles); /* c_nht */
        rocke_value_t* c_nhq = rocke_b_const_i32(b, n_h_tiles * q_tiles_p);
        (void)rocke_b_const_i32(b, n_h_tiles * q_tiles_p * g_tiles_p); /* c_nhqg */
        (void)rocke_b_div(b, cell_idx, c_nhq); /* pg_g_tile */
        rocke_value_t* pg_rem1 = rocke_b_mod(b, cell_idx, c_nhq);
        (void)rocke_b_div(b, pg_rem1, c_nhq); /* pg_n */
        rocke_value_t* pg_n_v
            = rocke_b_div(b, cell_idx, rocke_b_const_i32(b, n_h_tiles * q_tiles_p * g_tiles_p));
        rocke_value_t* pg_rem_n
            = rocke_b_mod(b, cell_idx, rocke_b_const_i32(b, n_h_tiles * q_tiles_p * g_tiles_p));
        pg_gt_v = rocke_b_div(b, pg_rem_n, rocke_b_const_i32(b, n_h_tiles * q_tiles_p));
        rocke_value_t* pg_rem_gt
            = rocke_b_mod(b, pg_rem_n, rocke_b_const_i32(b, n_h_tiles * q_tiles_p));
        rocke_value_t* pg_ht_v = rocke_b_div(b, pg_rem_gt, c_qt_p);
        rocke_value_t* pg_qt_v = rocke_b_mod(b, pg_rem_gt, c_qt_p);

        n = rocke_b_select(b, pg_in_bounds, pg_n_v, c0);
        {
            rocke_value_t* cbh = rocke_b_const_i32(b, BLOCK_H);
            rocke_value_t* mh = rocke_b_mul(b, pg_ht_v, cbh);
            h_tile_start = rocke_b_select(b, pg_in_bounds, mh, c0);
        }
        {
            rocke_value_t* mg = rocke_b_mul(b, pg_gt_v, c_BG);
            rocke_value_t* ag = rocke_b_add(b, mg, wave_group_idx);
            g = rocke_b_select(b, pg_in_bounds, ag, c0);
        }
        {
            rocke_value_t* m1 = rocke_b_mul(b, pg_qt_v, c_BQ);
            rocke_value_t* cq = rocke_b_const_i32(b, BLOCK_Q_WAVE);
            rocke_value_t* m2 = rocke_b_mul(b, wave_id_q, cq);
            rocke_value_t* aq = rocke_b_add(b, m1, m2);
            q_tile_start = rocke_b_select(b, pg_in_bounds, aq, c0);
        }
        {
            rocke_value_t* ml = rocke_b_mul(b, pg_qt_v, c_BQ);
            ctx.q_tile_start_lds = rocke_b_select(b, pg_in_bounds, ml, c0);
        }
        ctx.n = n;
    }

    rocke_value_t* load_g_tile = PERSISTENT ? pg_gt_v : NULL;
    rocke_value_t* prologue_y = BLOCK_H == 0 ? c0 : h_tile_start;
    rocke_value_t* wl_lds = NULL;
    rocke_dcg_load_t* wl_prologue_loads = NULL;
    const int WL_GROUP = p->cpg * p->KH * p->KW * p->kpg;
    if(spec->dgrad_weights_lds)
    {
        const int WL_VECS = BLOCK_GROUPS * WL_GROUP / 8;
        const int WL_PASSES = (WL_VECS + THREADS - 1) / THREADS;
        rocke_value_t** wl_vecs;
        rocke_value_t** wl_idx;
        rocke_value_t* wl_base;
        {
            int shape[2];
            shape[0] = 1;
            shape[1] = WL_PASSES * THREADS * 8;
            wl_lds = rocke_b_smem_alloc(b, ctx.io_type, shape, 2, "lds_w");
        }
        wl_base = rocke_b_mul(b, by, rocke_b_const_i32(b, BLOCK_GROUPS * WL_GROUP));
        wl_vecs = (rocke_value_t**)rocke_dcg__alloc(b, sizeof(rocke_value_t*) * (size_t)WL_PASSES);
        wl_idx = (rocke_value_t**)rocke_dcg__alloc(b, sizeof(rocke_value_t*) * (size_t)WL_PASSES);
        if(wl_vecs == NULL || wl_idx == NULL)
        {
            return NULL;
        }
        for(int pi = 0; pi < WL_PASSES; ++pi)
        {
            rocke_value_t* wl_v = rocke_b_add(b, tid, rocke_b_const_i32(b, pi * THREADS));
            rocke_value_t* wl_off;
            {
                rocke_value_t* c8 = rocke_b_const_i32(b, 8);
                rocke_value_t* m8 = rocke_b_mul(b, wl_v, c8);
                rocke_value_t* ad = rocke_b_add(b, wl_base, m8);
                wl_off = rocke_b_mul(b, ad, c_half_bytes);
            }
            if((pi + 1) * THREADS > WL_VECS)
            {
                rocke_value_t* cv = rocke_b_const_i32(b, WL_VECS);
                rocke_value_t* lt = rocke_b_cmp_lt(b, wl_v, cv);
                wl_off = rocke_b_select(b, lt, wl_off, oob_sentinel);
            }
            wl_vecs[pi] = rocke_dcg__buf_load(&ctx, b_rsrc, wl_off, 4);
            wl_idx[pi] = rocke_b_mul(b, wl_v, rocke_b_const_i32(b, 8));
        }
        wl_prologue_loads = rocke_dcg__issue_dram_load(&ctx, prologue_y, load_g_tile);
        if(wl_prologue_loads == NULL)
        {
            return NULL;
        }
        for(int pi = 0; pi < WL_PASSES; ++pi)
        {
            rocke_value_t* idx[2];
            idx[0] = c0;
            idx[1] = wl_idx[pi];
            rocke_b_smem_store_vN(b, wl_lds, idx, 2, wl_vecs[pi], 8);
        }
        rocke_b_sync(b);
    }
    else
    {
        rocke_dcg_load_t* loads = rocke_dcg__issue_dram_load(&ctx, prologue_y, load_g_tile);
        if(loads == NULL)
        {
            return NULL;
        }
        rocke_dcg__store_to_lds(&ctx, loads, A_smem);
        rocke_b_sync(b);
    }

    /* acc_tiles[qt][m][slot] (q_subtiles x M_LOCAL x KH); the runtime-K paths
     * index it up to N_M_TILES, equal to M_LOCAL there (waves_m == 1). */
    const int KH = p->KH;
    const int KW = p->KW;
    rocke_value_t** acc_tiles = (rocke_value_t**)rocke_dcg__alloc(
        b, sizeof(rocke_value_t*) * (size_t)(q_subtiles * N_M_TILES * KH));
    if(acc_tiles == NULL)
    {
        return NULL;
    }
#define ACC(qt, m, slot) acc_tiles[((qt) * N_M_TILES + (m)) * KH + (slot)]
    for(int qt = 0; qt < q_subtiles; ++qt)
        for(int m = 0; m < N_M_TILES; ++m)
            for(int s = 0; s < KH; ++s)
                ACC(qt, m, s) = zero_acc;

    /* preloaded_w[(r, s, local_atom, m)] (m < N_M_TILES). */
    rocke_value_t** preloaded_w = (rocke_value_t**)rocke_dcg__alloc(
        b,
        sizeof(rocke_value_t*) * (size_t)(KH * KW * (N_K_LOCAL > 0 ? N_K_LOCAL : 1) * N_M_TILES));
    if(preloaded_w == NULL)
    {
        return NULL;
    }
#define PW(r, s, la, m) preloaded_w[(((r) * KW + (s)) * N_K_LOCAL + (la)) * N_M_TILES + (m)]

    if(WAVES_K > 1)
    {
        rocke_value_t* lane_id_pw = rocke_b_mod(b, tid, rocke_b_const_i32(b, WAVE));
        rocke_value_t* c_block_sz = rocke_b_const_i32(b, WAVE * LOAD_VEC);
        const int pw_n_dwords = LOAD_VEC / 2;
        const int pw_blocks_per_group = KH * KW * N_K_ATOMS * N_M_TILES;
        rocke_value_t* pw_g_abs = PERSISTENT ? rocke_b_const_i32(b, 0) : g;
        rocke_value_t* pw_group_base
            = rocke_b_mul(b, pw_g_abs, rocke_b_const_i32(b, pw_blocks_per_group));
        for(int r = 0; r < KH; ++r)
        {
            for(int s = 0; s < KW; ++s)
            {
                for(int la = 0; la < N_K_LOCAL; ++la)
                {
                    rocke_value_t* atom_global_val
                        = rocke_b_add(b, k_atom_base, rocke_b_const_i32(b, la));
                    rocke_value_t* rs_base_pw;
                    {
                        rocke_value_t* cm = rocke_b_const_i32(b, N_M_TILES);
                        rocke_value_t* ma = rocke_b_mul(b, atom_global_val, cm);
                        rocke_value_t* crs = rocke_b_const_i32(
                            b, (r * KW * N_K_ATOMS + s * N_K_ATOMS) * N_M_TILES);
                        rocke_value_t* inner = rocke_b_add(b, ma, crs);
                        rs_base_pw = rocke_b_add(b, pw_group_base, inner);
                    }
                    for(int m = 0; m < N_M_TILES; ++m)
                    {
                        if(m * 16 >= p->kpg)
                        {
                            PW(r, s, la, m) = NULL;
                            continue;
                        }
                        rocke_value_t* block_idx
                            = rocke_b_add(b, rs_base_pw, rocke_b_const_i32(b, m));
                        rocke_value_t* elem_off;
                        {
                            rocke_value_t* m1 = rocke_b_mul(b, block_idx, c_block_sz);
                            rocke_value_t* cl = rocke_b_const_i32(b, LOAD_VEC);
                            rocke_value_t* m2 = rocke_b_mul(b, lane_id_pw, cl);
                            elem_off = rocke_b_add(b, m1, m2);
                        }
                        rocke_value_t* bytes = rocke_b_mul(b, elem_off, c_half_bytes);
                        PW(r, s, la, m) = rocke_dcg__buf_load(&ctx, b_rsrc, bytes, pw_n_dwords);
                    }
                }
            }
        }
    }
    else if(PRELOADS)
    {
        const int c4_step_pl = K_ATOM_SIZE / 4;
        rocke_value_t* wl_lane_row = NULL;
        rocke_value_t* wl_lane_col = NULL;
        rocke_value_t* wl_grp_off = NULL;
        rocke_value_t* c_wl_row = NULL;
        const rocke_tensor_descriptor_t* fw_desc = NULL;
        const int fw_k_stride = KH * KW * p->kpg;
        if(spec->dgrad_weights_lds)
        {
            {
                rocke_value_t* c4a = rocke_b_const_i32(b, 4);
                rocke_value_t* dv = rocke_b_div(b, lane, c4a);
                rocke_value_t* c4b = rocke_b_const_i32(b, 4);
                wl_lane_row = rocke_b_mod(b, dv, c4b);
            }
            {
                rocke_value_t* c4a = rocke_b_const_i32(b, 4);
                rocke_value_t* md = rocke_b_mod(b, lane, c4a);
                rocke_value_t* c4b = rocke_b_const_i32(b, 4);
                wl_lane_col = rocke_b_mul(b, md, c4b);
            }
            wl_grp_off = rocke_b_mul(b, wave_group_idx, rocke_b_const_i32(b, WL_GROUP));
            c_wl_row = rocke_b_const_i32(b, KH * KW * p->kpg);
        }
        else if(spec->dgrad_fused_weights)
        {
            static const char* const fw_coords[4] = {"k", "r", "s", "c"};
            int lengths[4];
            lengths[0] = p->groups * p->cpg;
            lengths[1] = KH;
            lengths[2] = KW;
            lengths[3] = p->kpg;
            fw_desc = rocke_tensor_descriptor_naive(b, "B", lengths, 4, NULL, fw_coords, 4);
        }
        for(int r = 0; r < KH; ++r)
        {
            for(int s = 0; s < KW; ++s)
            {
                for(int la = 0; la < N_K_LOCAL; ++la)
                {
                    rocke_value_t* ch_off_pl;
                    {
                        rocke_value_t* cka = rocke_b_const_i32(b, K_ATOM_SIZE);
                        rocke_value_t* m1 = rocke_b_mul(b, k_atom_base, cka);
                        rocke_value_t* cla = rocke_b_const_i32(b, la * K_ATOM_SIZE);
                        rocke_value_t* cst = rocke_b_const_i32(b, c4_step_pl);
                        rocke_value_t* m2 = rocke_b_mul(b, c4, cst);
                        rocke_value_t* inner = rocke_b_add(b, cla, m2);
                        ch_off_pl = rocke_b_add(b, m1, inner);
                    }
                    for(int m = 0; m < M_LOCAL; ++m)
                    {
                        rocke_value_t* w_frag_pl;
                        if(m * 16 >= p->kpg)
                        {
                            PW(r, s, la, m) = NULL;
                            continue;
                        }
                        if(spec->dgrad_weights_lds)
                        {
                            rocke_value_t* wl_row = rocke_b_add(b, ch_off_pl, wl_lane_row);
                            rocke_value_t* wl_idx2;
                            {
                                rocke_value_t* mr = rocke_b_mul(b, wl_row, c_wl_row);
                                rocke_value_t* a1 = rocke_b_add(b, wl_grp_off, mr);
                                rocke_value_t* cx = rocke_b_const_i32(
                                    b, ((KH - 1 - r) * KW + (KW - 1 - s)) * p->kpg + m * 16);
                                rocke_value_t* a2 = rocke_b_add(b, cx, wl_lane_col);
                                wl_idx2 = rocke_b_add(b, a1, a2);
                            }
                            if(wm_base16 != NULL)
                            {
                                wl_idx2 = rocke_b_add(b, wl_idx2, wm_base16);
                            }
                            {
                                rocke_value_t* idx[2];
                                idx[0] = c0;
                                idx[1] = wl_idx2;
                                w_frag_pl
                                    = rocke_b_ds_read_tr16_b64(b, wl_lds, idx, 2, ctx.io_type);
                            }
                            for(int rd = 1; rd < LOAD_VEC / 4; ++rd)
                            {
                                rocke_value_t* cr = rocke_b_const_i32(b, 4 * rd * KH * KW * p->kpg);
                                rocke_value_t* ar = rocke_b_add(b, wl_idx2, cr);
                                rocke_value_t* idx[2];
                                rocke_value_t* rdv;
                                idx[0] = c0;
                                idx[1] = ar;
                                rdv = rocke_b_ds_read_tr16_b64(b, wl_lds, idx, 2, ctx.io_type);
                                w_frag_pl = rocke_b_vec_concat(b, w_frag_pl, rdv);
                            }
                            if(cpg_partial)
                            {
                                rocke_value_t* ge = rocke_b_cmp_ge(b, ch_off_pl, c_cpg);
                                rocke_value_t* zv = rocke_b_zero_vec(b, ctx.io_type, LOAD_VEC);
                                w_frag_pl = rocke_b_select(b, ge, zv, w_frag_pl);
                            }
                        }
                        else if(spec->dgrad_fused_weights)
                        {
                            const char* names[4] = {"k", "r", "s", "c"};
                            rocke_value_t* vals[4];
                            rocke_value_t* fw_off;
                            rocke_value_t** elems;
                            {
                                rocke_value_t* mg = rocke_b_mul(b, g, c_cpg);
                                vals[0] = rocke_b_add(b, mg, ch_off_pl);
                            }
                            vals[1] = rocke_b_const_i32(b, KH - 1 - r);
                            vals[2] = rocke_b_const_i32(b, KW - 1 - s);
                            if(wm_base16 == NULL)
                            {
                                rocke_value_t* cm = rocke_b_const_i32(b, m * 16);
                                vals[3] = rocke_b_add(b, cm, q_in_lane);
                            }
                            else
                            {
                                rocke_value_t* cm = rocke_b_const_i32(b, m * 16);
                                rocke_value_t* a1 = rocke_b_add(b, wm_base16, cm);
                                vals[3] = rocke_b_add(b, a1, q_in_lane);
                            }
                            fw_off = rocke_dcg__offset(b, fw_desc, names, vals, 4);
                            if(fw_off == NULL)
                            {
                                return NULL;
                            }
                            elems = (rocke_value_t**)rocke_dcg__alloc(
                                b, sizeof(rocke_value_t*) * (size_t)LOAD_VEC);
                            if(elems == NULL)
                            {
                                return NULL;
                            }
                            for(int e = 0; e < LOAD_VEC; ++e)
                            {
                                rocke_value_t* e_off;
                                {
                                    rocke_value_t* ce = rocke_b_const_i32(b, e * fw_k_stride);
                                    rocke_value_t* ae = rocke_b_add(b, fw_off, ce);
                                    e_off = rocke_b_mul(b, ae, c_half_bytes);
                                }
                                if(cpg_partial)
                                {
                                    rocke_value_t* ce = rocke_b_const_i32(b, e);
                                    rocke_value_t* ae = rocke_b_add(b, ch_off_pl, ce);
                                    rocke_value_t* e_ok = rocke_b_cmp_lt(b, ae, c_cpg);
                                    e_off = rocke_b_select(b, e_ok, e_off, oob_sentinel);
                                }
                                elems[e] = ctx.is_bf16
                                               ? rocke_b_buffer_load_bf16(b, b_rsrc, e_off, c0)
                                               : rocke_b_buffer_load_f16(b, b_rsrc, e_off, c0);
                            }
                            w_frag_pl = rocke_b_vec_pack(b, elems, LOAD_VEC, ctx.io_type);
                        }
                        else
                        {
                            rocke_value_t* k_out_pl;
                            const char* names[4] = {"k_out", "r", "s", "c"};
                            rocke_value_t* vals[4];
                            rocke_value_t* w_off_pl;
                            {
                                rocke_value_t* mg = rocke_b_mul(b, g, c_kpg);
                                rocke_value_t* cm = rocke_b_const_i32(b, m * 16);
                                rocke_value_t* aq = rocke_b_add(b, cm, q_in_lane);
                                k_out_pl = rocke_b_add(b, mg, aq);
                            }
                            if(wm_base16 != NULL)
                            {
                                k_out_pl = rocke_b_add(b, k_out_pl, wm_base16);
                            }
                            vals[0] = k_out_pl;
                            vals[1] = rocke_b_const_i32(b, r);
                            vals[2] = rocke_b_const_i32(b, s);
                            vals[3] = ch_off_pl;
                            w_off_pl = rocke_dcg__offset(b, b_desc, names, vals, 4);
                            if(w_off_pl == NULL)
                            {
                                return NULL;
                            }
                            {
                                rocke_value_t* bytes = rocke_b_mul(b, w_off_pl, c_half_bytes);
                                w_frag_pl = rocke_dcg__buf_load(&ctx, b_rsrc, bytes, LOAD_VEC / 2);
                            }
                            if(cpg_partial)
                            {
                                rocke_value_t* ge = rocke_b_cmp_ge(b, ch_off_pl, c_cpg);
                                rocke_value_t* zv = rocke_b_zero_vec(b, ctx.io_type, LOAD_VEC);
                                w_frag_pl = rocke_b_select(b, ge, zv, w_frag_pl);
                            }
                        }
                        PW(r, s, la, m) = w_frag_pl;
                    }
                }
            }
        }
        if(spec->dgrad_weights_lds)
        {
            rocke_b_sync(b);
            rocke_dcg__alloc_row_buffers(
                &ctx, spec->double_buffer, lds_total_fp16, &A_smem, &B_smem);
            rocke_dcg__store_to_lds(&ctx, wl_prologue_loads, A_smem);
            rocke_b_sync(b);
        }
    }

    const int n_iters = BLOCK_H > 0 ? (BLOCK_H + KH - 1) : (p->H + KH - 1);
    const int PF = spec->prefetch_rows > 1 ? spec->prefetch_rows : 1;

    /* ---- stage_out tiles + store meta ---- */
    rocke_value_t* so_bufs[2] = {NULL, NULL};
    rocke_value_t* c_so_stride = NULL;
    rocke_dcg_so_t* so_meta = NULL;
    int n_so_meta = 0;
    if(spec->stage_out)
    {
        const int SO_CH = BLOCK_GROUPS * p->kpg;
        const int SO_STRIDE = SO_CH + 8;
        const int SO_VPC = SO_CH / 8;
        const int SO_VECS = BLOCK_Q * SO_VPC;
        int shape[2];
        shape[0] = 1;
        shape[1] = BLOCK_Q * SO_STRIDE;
        so_bufs[0] = rocke_b_smem_alloc(b, ctx.io_type, shape, 2, "lds_out0");
        so_bufs[1] = rocke_b_smem_alloc(b, ctx.io_type, shape, 2, "lds_out1");
        c_so_stride = rocke_b_const_i32(b, SO_STRIDE);
        n_so_meta = (SO_VECS + THREADS - 1) / THREADS;
        so_meta = (rocke_dcg_so_t*)rocke_dcg__alloc(b, sizeof(rocke_dcg_so_t) * (size_t)n_so_meta);
        if(so_meta == NULL)
        {
            return NULL;
        }
        for(int pi = 0; pi < n_so_meta; ++pi)
        {
            rocke_value_t* t = rocke_b_add(b, tid, rocke_b_const_i32(b, pi * THREADS));
            rocke_value_t* q = rocke_b_div(b, t, rocke_b_const_i32(b, SO_VPC));
            rocke_value_t* v = rocke_b_mod(b, t, rocke_b_const_i32(b, SO_VPC));
            rocke_value_t* qg = rocke_b_add(b, block_q_start, q);
            rocke_value_t* ok = rocke_b_cmp_lt(b, qg, c_W);
            rocke_value_t* gel;
            if((pi + 1) * THREADS > SO_VECS)
            {
                rocke_value_t* cv = rocke_b_const_i32(b, SO_VECS);
                rocke_value_t* lt = rocke_b_cmp_lt(b, t, cv);
                ok = rocke_b_land(b, ok, lt);
            }
            {
                rocke_value_t* chw = rocke_b_const_i32(b, Ho * Wo);
                rocke_value_t* m1 = rocke_b_mul(b, n, chw);
                rocke_value_t* a1 = rocke_b_add(b, m1, qg);
                rocke_value_t* ctk = rocke_b_const_i32(b, total_k);
                rocke_value_t* m2 = rocke_b_mul(b, a1, ctk);
                rocke_value_t* cso = rocke_b_const_i32(b, SO_CH);
                rocke_value_t* m3 = rocke_b_mul(b, g_tile, cso);
                rocke_value_t* c8 = rocke_b_const_i32(b, 8);
                rocke_value_t* m4 = rocke_b_mul(b, v, c8);
                rocke_value_t* a2 = rocke_b_add(b, m3, m4);
                gel = rocke_b_add(b, m2, a2);
            }
            so_meta[pi].ok = ok;
            so_meta[pi].gbytes = rocke_b_mul(b, gel, c_half_bytes);
            {
                rocke_value_t* mq = rocke_b_mul(b, q, c_so_stride);
                rocke_value_t* c8 = rocke_b_const_i32(b, 8);
                rocke_value_t* mv = rocke_b_mul(b, v, c8);
                so_meta[pi].lidx = rocke_b_add(b, mq, mv);
            }
        }
    }

    /* _row_y(y_l) and the prefetched rows 1 .. PF-1 (pending_rows). */
    rocke_dcg_load_t** pending = (rocke_dcg_load_t**)rocke_dcg__alloc(
        b, sizeof(rocke_dcg_load_t*) * (size_t)(n_iters + 4));
    if(pending == NULL)
    {
        return NULL;
    }
#define ROW_Y(y_l)                                                           \
    (BLOCK_H > 0 ? rocke_b_add(b, h_tile_start, rocke_b_const_i32(b, (y_l))) \
                 : rocke_b_const_i32(b, (y_l)))
    for(int y_l = 1; y_l < (PF < n_iters ? PF : n_iters); ++y_l)
    {
        rocke_value_t* yv = ROW_Y(y_l);
        pending[y_l] = rocke_dcg__issue_dram_load(&ctx, yv, load_g_tile);
        if(pending[y_l] == NULL)
        {
            return NULL;
        }
    }

    for(int y_local = 0; y_local < n_iters; ++y_local)
    {
        if(BLOCK_H > 0)
        {
            rocke_value_t* cy = rocke_b_const_i32(b, y_local);
            (void)rocke_b_add(b, h_tile_start, cy); /* y */
        }
        else
        {
            (void)rocke_b_const_i32(b, y_local); /* y */
        }
        const bool even = (y_local % 2 == 0) || !spec->double_buffer;
        rocke_value_t* cur = even ? A_smem : B_smem;
        rocke_value_t* nxt = even ? B_smem : A_smem;

        rocke_dcg_load_t* loads_next = NULL;
        if(PF > 1)
        {
            if(y_local + PF < n_iters)
            {
                rocke_value_t* yv = ROW_Y(y_local + PF);
                pending[y_local + PF] = rocke_dcg__issue_dram_load(&ctx, yv, load_g_tile);
                if(pending[y_local + PF] == NULL)
                {
                    return NULL;
                }
            }
            if(y_local + 1 < n_iters + 4)
            {
                loads_next = pending[y_local + 1];
                pending[y_local + 1] = NULL;
            }
        }
        else if(y_local + 1 < n_iters)
        {
            rocke_value_t* next_y;
            if(BLOCK_H > 0)
            {
                rocke_value_t* cy = rocke_b_const_i32(b, y_local + 1);
                next_y = rocke_b_add(b, h_tile_start, cy);
            }
            else
            {
                next_y = rocke_b_const_i32(b, y_local + 1);
            }
            loads_next = rocke_dcg__issue_dram_load(&ctx, next_y, load_g_tile);
            if(loads_next == NULL)
            {
                return NULL;
            }
        }

        for(int qt = 0; qt < q_subtiles; ++qt)
        {
            const int qt_w_base = qt * 16;
            for(int r = 0; r < KH; ++r)
            {
                const int p_idx = rocke_dcg__pymod(y_local - r, KH);
                rocke_value_t* r_i = rocke_b_const_i32(b, r);

                if(PRELOADS)
                {
                    for(int s = 0; s < KW; ++s)
                    {
                        rocke_value_t* s_val = rocke_b_const_i32(b, s);
                        for(int la = 0; la < N_K_LOCAL; ++la)
                        {
                            const int c4_step = K_ATOM_SIZE / 4;
                            rocke_value_t* ch_off;
                            {
                                rocke_value_t* cka = rocke_b_const_i32(b, K_ATOM_SIZE);
                                rocke_value_t* m1 = rocke_b_mul(b, k_atom_base, cka);
                                rocke_value_t* cla = rocke_b_const_i32(b, la * K_ATOM_SIZE);
                                rocke_value_t* a1 = rocke_b_add(b, m1, cla);
                                rocke_value_t* cs = rocke_b_const_i32(b, c4_step);
                                rocke_value_t* m2 = rocke_b_mul(b, c4, cs);
                                ch_off = rocke_b_add(b, a1, m2);
                            }
                            rocke_value_t* W_lds_idx_pw;
                            {
                                rocke_value_t* cq = rocke_b_const_i32(b, qt_w_base);
                                rocke_value_t* aq = rocke_b_add(b, q_in_lane, cq);
                                rocke_value_t* cs = rocke_b_const_i32(b, c_stride_gen);
                                rocke_value_t* mq = rocke_b_mul(b, aq, cs);
                                W_lds_idx_pw = rocke_b_add(b, mq, s_val);
                            }
                            if(WAVES_Q > 1)
                            {
                                rocke_value_t* cq = rocke_b_const_i32(b, BLOCK_Q_WAVE);
                                rocke_value_t* mq = rocke_b_mul(b, wave_id_q, cq);
                                W_lds_idx_pw = rocke_b_add(b, mq, W_lds_idx_pw);
                            }
                            rocke_value_t* lds_idx_pw;
                            {
                                rocke_value_t* m1 = rocke_b_mul(b, W_lds_idx_pw, c_BG_cpg);
                                rocke_value_t* m2 = rocke_b_mul(b, wave_group_idx, c_cpg);
                                rocke_value_t* a1 = rocke_b_add(b, m1, m2);
                                lds_idx_pw = rocke_b_add(b, a1, ch_off);
                            }
                            rocke_value_t* x_frag;
                            {
                                rocke_value_t* idx[2];
                                idx[0] = c0;
                                idx[1] = lds_idx_pw;
                                x_frag
                                    = rocke_b_smem_load_vN(b, cur, idx, 2, ctx.io_type, LOAD_VEC);
                            }
                            if(cpg_partial)
                            {
                                rocke_value_t* cc = rocke_b_const_i32(b, p->cpg);
                                rocke_value_t* oob = rocke_b_cmp_ge(b, ch_off, cc);
                                rocke_value_t* zv = rocke_b_zero_vec(b, ctx.io_type, LOAD_VEC);
                                x_frag = rocke_b_select(b, oob, zv, x_frag);
                            }
                            for(int m = 0; m < M_LOCAL; ++m)
                            {
                                if(m * 16 >= p->kpg)
                                    continue;
                                ACC(qt, m, p_idx) = rocke_dcg__mfma(
                                    &ctx, FOLD_K32, PW(r, s, la, m), x_frag, ACC(qt, m, p_idx));
                            }
                        }
                    }
                    continue;
                }

                if(spec->runtime_k_loop)
                {
                    rocke_iter_arg_t* ia = (rocke_iter_arg_t*)rocke_dcg__alloc(
                        b, sizeof(rocke_iter_arg_t) * (size_t)N_M_TILES);
                    rocke_value_t** new_accs = (rocke_value_t**)rocke_dcg__alloc(
                        b, sizeof(rocke_value_t*) * (size_t)N_M_TILES);
                    if(ia == NULL || new_accs == NULL)
                    {
                        return NULL;
                    }
                    for(int m = 0; m < N_M_TILES; ++m)
                    {
                        {
                            char buf[96];
                            snprintf(buf, sizeof buf, "rk_acc_m%d_y%d_qt%d_r%d", m, y_local, qt, r);
                            ia[m].name = rocke_dcg__strdup(b, buf);
                            if(ia[m].name == NULL)
                                return NULL;
                        }
                        ia[m].init = ACC(qt, m, p_idx);
                    }
                    const char* iv_name = rocke_dcg__name(b, "rk_iv_y%d_qt%d_r%d", y_local, qt, r);
                    rocke_for_t k_loop_rk = rocke_b_scf_for_iter(
                        b, c0, c_N_K_LOCAL, c1, ia, N_M_TILES, iv_name, false, false);
                    if(k_loop_rk.body == NULL)
                    {
                        return NULL;
                    }
                    rocke_b_region_enter(b, k_loop_rk.body);
                    {
                        rocke_value_t* k_iv_rk = k_loop_rk.iv;
                        rocke_value_t* k_atom_global = rocke_b_add(b, k_atom_base, k_iv_rk);
                        const int rk_c4_step = K_ATOM_SIZE / 4;
                        rocke_value_t* ch_off_rk;
                        {
                            rocke_value_t* cka = rocke_b_const_i32(b, K_ATOM_SIZE);
                            rocke_value_t* m1 = rocke_b_mul(b, k_atom_global, cka);
                            rocke_value_t* cs = rocke_b_const_i32(b, rk_c4_step);
                            rocke_value_t* m2 = rocke_b_mul(b, c4, cs);
                            ch_off_rk = rocke_b_add(b, m1, m2);
                        }
                        for(int m = 0; m < N_M_TILES; ++m)
                        {
                            new_accs[m] = k_loop_rk.iter_vars[m];
                        }
                        for(int s = 0; s < KW; ++s)
                        {
                            rocke_value_t* s_val_rk = rocke_b_const_i32(b, s);
                            rocke_value_t* W_lds_idx_rk;
                            {
                                rocke_value_t* cq = rocke_b_const_i32(b, qt_w_base);
                                rocke_value_t* aq = rocke_b_add(b, q_in_lane, cq);
                                rocke_value_t* cs = rocke_b_const_i32(b, c_stride_gen);
                                rocke_value_t* mq = rocke_b_mul(b, aq, cs);
                                W_lds_idx_rk = rocke_b_add(b, mq, s_val_rk);
                            }
                            if(WAVES_Q > 1)
                            {
                                rocke_value_t* cq = rocke_b_const_i32(b, BLOCK_Q_WAVE);
                                rocke_value_t* mq = rocke_b_mul(b, wave_id_q, cq);
                                W_lds_idx_rk = rocke_b_add(b, mq, W_lds_idx_rk);
                            }
                            rocke_value_t* lds_idx_rk;
                            {
                                rocke_value_t* m1 = rocke_b_mul(b, W_lds_idx_rk, c_BG_cpg);
                                rocke_value_t* m2 = rocke_b_mul(b, wave_group_idx, c_cpg);
                                rocke_value_t* a1 = rocke_b_add(b, m1, m2);
                                lds_idx_rk = rocke_b_add(b, a1, ch_off_rk);
                            }
                            rocke_value_t* x_frag_rk;
                            {
                                rocke_value_t* idx[2];
                                idx[0] = c0;
                                idx[1] = lds_idx_rk;
                                x_frag_rk
                                    = rocke_b_smem_load_vN(b, cur, idx, 2, ctx.io_type, LOAD_VEC);
                            }
                            if(cpg_partial)
                            {
                                rocke_value_t* cc = rocke_b_const_i32(b, p->cpg);
                                rocke_value_t* oob = rocke_b_cmp_ge(b, ch_off_rk, cc);
                                rocke_value_t* zv = rocke_b_zero_vec(b, ctx.io_type, LOAD_VEC);
                                x_frag_rk = rocke_b_select(b, oob, zv, x_frag_rk);
                            }
                            for(int m = 0; m < N_M_TILES; ++m)
                            {
                                if(m * 16 >= p->kpg)
                                    continue;
                                const int rs_const_base
                                    = (r * KW * N_K_ATOMS + s * N_K_ATOMS) * N_M_TILES;
                                const int rk_blocks_per_group = KH * KW * N_K_ATOMS * N_M_TILES;
                                rocke_value_t* rk_g_abs = PERSISTENT ? rocke_b_const_i32(b, 0) : g;
                                rocke_value_t* rk_group_base = rocke_b_mul(
                                    b, rk_g_abs, rocke_b_const_i32(b, rk_blocks_per_group));
                                rocke_value_t* block_idx_rk;
                                {
                                    rocke_value_t* crs = rocke_b_const_i32(b, rs_const_base + m);
                                    rocke_value_t* cm = rocke_b_const_i32(b, N_M_TILES);
                                    rocke_value_t* mk = rocke_b_mul(b, k_atom_global, cm);
                                    rocke_value_t* inner = rocke_b_add(b, crs, mk);
                                    block_idx_rk = rocke_b_add(b, rk_group_base, inner);
                                }
                                rocke_value_t* elem_off_rk;
                                {
                                    rocke_value_t* cb = rocke_b_const_i32(b, WAVE * LOAD_VEC);
                                    rocke_value_t* m1 = rocke_b_mul(b, block_idx_rk, cb);
                                    rocke_value_t* cw = rocke_b_const_i32(b, WAVE);
                                    rocke_value_t* md = rocke_b_mod(b, tid, cw);
                                    rocke_value_t* cl = rocke_b_const_i32(b, LOAD_VEC);
                                    rocke_value_t* m2 = rocke_b_mul(b, md, cl);
                                    elem_off_rk = rocke_b_add(b, m1, m2);
                                }
                                rocke_value_t* bytes = rocke_b_mul(b, elem_off_rk, c_half_bytes);
                                rocke_value_t* w_frag_rk
                                    = rocke_dcg__buf_load(&ctx, b_rsrc, bytes, LOAD_VEC / 2);
                                new_accs[m] = rocke_dcg__mfma(
                                    &ctx, FOLD_K32, w_frag_rk, x_frag_rk, new_accs[m]);
                            }
                        }
                        rocke_b_scf_yield(b, new_accs, N_M_TILES);
                    }
                    rocke_b_region_leave(b);
                    for(int m = 0; m < N_M_TILES; ++m)
                    {
                        ACC(qt, m, p_idx) = k_loop_rk.op->results[m];
                    }
                    continue;
                }

                /* ---- runtime s / K-atom loops (waves_k == 1) ---- */
                {
                    rocke_iter_arg_t* s_ia = (rocke_iter_arg_t*)rocke_dcg__alloc(
                        b, sizeof(rocke_iter_arg_t) * (size_t)N_M_TILES);
                    rocke_iter_arg_t* a_ia = (rocke_iter_arg_t*)rocke_dcg__alloc(
                        b, sizeof(rocke_iter_arg_t) * (size_t)N_M_TILES);
                    rocke_value_t** new_atom_accs = (rocke_value_t**)rocke_dcg__alloc(
                        b, sizeof(rocke_value_t*) * (size_t)N_M_TILES);
                    rocke_value_t** atom_results = (rocke_value_t**)rocke_dcg__alloc(
                        b, sizeof(rocke_value_t*) * (size_t)N_M_TILES);
                    if(s_ia == NULL || a_ia == NULL || new_atom_accs == NULL
                       || atom_results == NULL)
                    {
                        return NULL;
                    }
                    for(int m = 0; m < N_M_TILES; ++m)
                    {
                        char buf[96];
                        snprintf(buf, sizeof buf, "siv_acc_m%d_y%d_qt%d_r%d", m, y_local, qt, r);
                        s_ia[m].name = rocke_dcg__strdup(b, buf);
                        if(s_ia[m].name == NULL)
                            return NULL;
                        s_ia[m].init = ACC(qt, m, p_idx);
                    }
                    const char* s_iv_name = rocke_dcg__name(b, "s_iv_y%d_qt%d_r%d", y_local, qt, r);
                    rocke_for_t s_loop = rocke_b_scf_for_iter(
                        b, c0, c_KW, c1, s_ia, N_M_TILES, s_iv_name, false, false);
                    if(s_loop.body == NULL)
                    {
                        return NULL;
                    }
                    rocke_b_region_enter(b, s_loop.body);
                    {
                        rocke_value_t* s_iv = s_loop.iv;
                        for(int m = 0; m < N_M_TILES; ++m)
                        {
                            char buf[96];
                            snprintf(
                                buf, sizeof buf, "katom_acc_m%d_y%d_qt%d_r%d", m, y_local, qt, r);
                            a_ia[m].name = rocke_dcg__strdup(b, buf);
                            if(a_ia[m].name == NULL)
                                return NULL;
                            a_ia[m].init = s_loop.iter_vars[m];
                        }
                        const char* a_iv_name
                            = rocke_dcg__name(b, "atom_iv_y%d_qt%d_r%d", y_local, qt, r);
                        rocke_for_t atom_loop = rocke_b_scf_for_iter(
                            b, c0, c_N_K_LOCAL, c1, a_ia, N_M_TILES, a_iv_name, false, false);
                        if(atom_loop.body == NULL)
                        {
                            return NULL;
                        }
                        rocke_b_region_enter(b, atom_loop.body);
                        {
                            rocke_value_t* atom_iv_local = atom_loop.iv;
                            rocke_value_t* atom_iv_global
                                = rocke_b_add(b, k_atom_base, atom_iv_local);
                            const int std_c4_step = K_ATOM_SIZE / 4;
                            rocke_value_t* ch_off;
                            {
                                rocke_value_t* cka = rocke_b_const_i32(b, K_ATOM_SIZE);
                                rocke_value_t* m1 = rocke_b_mul(b, atom_iv_global, cka);
                                rocke_value_t* cs = rocke_b_const_i32(b, std_c4_step);
                                rocke_value_t* m2 = rocke_b_mul(b, c4, cs);
                                ch_off = rocke_b_add(b, m1, m2);
                            }
                            rocke_value_t* W_lds_idx;
                            {
                                rocke_value_t* cq = rocke_b_const_i32(b, qt_w_base);
                                rocke_value_t* aq = rocke_b_add(b, q_in_lane, cq);
                                rocke_value_t* cs = rocke_b_const_i32(b, c_stride_gen);
                                rocke_value_t* mq = rocke_b_mul(b, aq, cs);
                                W_lds_idx = rocke_b_add(b, mq, s_iv);
                            }
                            rocke_value_t* W_lds_idx_full = W_lds_idx;
                            if(WAVES_Q > 1)
                            {
                                rocke_value_t* cq = rocke_b_const_i32(b, BLOCK_Q_WAVE);
                                rocke_value_t* wave_q_lds_base = rocke_b_mul(b, wave_id_q, cq);
                                W_lds_idx_full = rocke_b_add(b, wave_q_lds_base, W_lds_idx);
                            }
                            rocke_value_t* lds_idx;
                            {
                                rocke_value_t* m1 = rocke_b_mul(b, W_lds_idx_full, c_BG_cpg);
                                rocke_value_t* m2 = rocke_b_mul(b, wave_group_idx, c_cpg);
                                rocke_value_t* a1 = rocke_b_add(b, m1, m2);
                                lds_idx = rocke_b_add(b, a1, ch_off);
                            }
                            rocke_value_t* x_frag;
                            {
                                rocke_value_t* idx[2];
                                idx[0] = c0;
                                idx[1] = lds_idx;
                                x_frag
                                    = rocke_b_smem_load_vN(b, cur, idx, 2, ctx.io_type, LOAD_VEC);
                            }
                            rocke_value_t* c4_oob = NULL;
                            rocke_value_t* zero_std = NULL;
                            if(cpg_partial)
                            {
                                rocke_value_t* cc = rocke_b_const_i32(b, p->cpg);
                                c4_oob = rocke_b_cmp_ge(b, ch_off, cc);
                                zero_std = rocke_b_zero_vec(b, ctx.io_type, LOAD_VEC);
                                x_frag = rocke_b_select(b, c4_oob, zero_std, x_frag);
                            }
                            for(int m = 0; m < N_M_TILES; ++m)
                            {
                                rocke_value_t* k_out_m;
                                {
                                    rocke_value_t* mg = rocke_b_mul(b, g, c_kpg);
                                    rocke_value_t* cm = rocke_b_const_i32(b, m * 16);
                                    rocke_value_t* aq = rocke_b_add(b, cm, q_in_lane);
                                    k_out_m = rocke_b_add(b, mg, aq);
                                }
                                const char* names[4] = {"k_out", "r", "s", "c"};
                                rocke_value_t* vals[4];
                                vals[0] = k_out_m;
                                vals[1] = r_i;
                                vals[2] = s_iv;
                                vals[3] = ch_off;
                                rocke_value_t* w_off = rocke_dcg__offset(b, b_desc, names, vals, 4);
                                if(w_off == NULL)
                                {
                                    return NULL;
                                }
                                rocke_value_t* bytes = rocke_b_mul(b, w_off, c_half_bytes);
                                rocke_value_t* w_frag
                                    = rocke_dcg__buf_load(&ctx, b_rsrc, bytes, LOAD_VEC / 2);
                                if(cpg_partial)
                                {
                                    w_frag = rocke_b_select(b, c4_oob, zero_std, w_frag);
                                }
                                new_atom_accs[m] = rocke_dcg__mfma(
                                    &ctx, FOLD_K32, w_frag, x_frag, atom_loop.iter_vars[m]);
                            }
                            rocke_b_scf_yield(b, new_atom_accs, N_M_TILES);
                        }
                        rocke_b_region_leave(b);
                        for(int m = 0; m < N_M_TILES; ++m)
                        {
                            atom_results[m] = atom_loop.op->results[m];
                        }
                        rocke_b_scf_yield(b, atom_results, N_M_TILES);
                    }
                    rocke_b_region_leave(b);
                    for(int m = 0; m < N_M_TILES; ++m)
                    {
                        ACC(qt, m, p_idx) = s_loop.op->results[m];
                    }
                }
            }
        }

        if(loads_next != NULL)
        {
            if(!spec->double_buffer)
            {
                rocke_b_sync(b);
            }
            rocke_dcg__store_to_lds(&ctx, loads_next, nxt);
        }
        {
            const int so_row = y_local - (KH - 1);
            if(spec->stage_out && 0 <= so_row && so_row < (BLOCK_H > 0 ? BLOCK_H : p->H))
            {
                rocke_value_t* so_buf = so_bufs[y_local % 2];
                const int so_slot = rocke_dcg__pymod(so_row, KH);
                for(int qt = 0; qt < q_subtiles; ++qt)
                {
                    rocke_value_t* col = rocke_b_add(b, q_in_lane, rocke_b_const_i32(b, qt * 16));
                    for(int m = 0; m < M_LOCAL; ++m)
                    {
                        rocke_value_t* ch;
                        {
                            rocke_value_t* m1 = rocke_b_mul(b, wave_group_idx, c_kpg);
                            rocke_value_t* cm = rocke_b_const_i32(b, m * 16);
                            rocke_value_t* c4c = rocke_b_const_i32(b, 4);
                            rocke_value_t* m2 = rocke_b_mul(b, c4, c4c);
                            rocke_value_t* a1 = rocke_b_add(b, cm, m2);
                            ch = rocke_b_add(b, m1, a1);
                        }
                        if(wm_base16 != NULL)
                        {
                            ch = rocke_b_add(b, ch, wm_base16);
                        }
                        {
                            rocke_value_t* mc = rocke_b_mul(b, col, c_so_stride);
                            rocke_value_t* ai = rocke_b_add(b, mc, ch);
                            rocke_value_t* idx[2];
                            rocke_value_t* tv;
                            idx[0] = c0;
                            idx[1] = ai;
                            tv = rocke_dcg__trunc(&ctx, ACC(qt, m, so_slot));
                            rocke_b_smem_store_vN(b, so_buf, idx, 2, tv, 4);
                        }
                    }
                }
            }
        }
        if(spec->lds_only_sync)
        {
            rocke_b_sync_lds_only(b);
        }
        else
        {
            rocke_b_sync(b);
        }

        const int p_flush_local = y_local - (KH - 1);
        const int P_FLUSH = rocke_dcg__pymod(p_flush_local, KH);
        bool should_flush = false;
        rocke_value_t* ho_row_val = NULL;
        rocke_value_t* in_h_range = NULL;
        if(BLOCK_H > 0)
        {
            if(p_flush_local >= 0)
            {
                rocke_value_t* p_flush_global
                    = rocke_b_add(b, h_tile_start, rocke_b_const_i32(b, p_flush_local));
                {
                    rocke_value_t* ge = rocke_b_cmp_ge(b, p_flush_global, c0);
                    rocke_value_t* ch = rocke_b_const_i32(b, p->H);
                    rocke_value_t* lt = rocke_b_cmp_lt(b, p_flush_global, ch);
                    in_h_range = rocke_b_land(b, ge, lt);
                }
                if(PERSISTENT)
                {
                    in_h_range = rocke_b_land(b, in_h_range, pg_in_bounds);
                }
                if(c_stride_gen > 1)
                {
                    rocke_value_t* cs = rocke_b_const_i32(b, c_stride_gen);
                    rocke_value_t* md = rocke_b_mod(b, p_flush_global, cs);
                    rocke_value_t* stride_ok = rocke_b_cmp_eq(b, md, c0);
                    in_h_range = rocke_b_land(b, in_h_range, stride_ok);
                }
                ho_row_val = rocke_b_div(b, p_flush_global, rocke_b_const_i32(b, c_stride_gen));
                should_flush = true;
            }
        }
        else
        {
            should_flush
                = 0 <= p_flush_local && p_flush_local < p->H && p_flush_local % c_stride_gen == 0;
            if(should_flush)
            {
                ho_row_val = rocke_b_const_i32(b, p_flush_local / c_stride_gen);
                in_h_range = NULL;
            }
        }

        if(should_flush && spec->stage_out)
        {
            rocke_value_t* so_buf = so_bufs[y_local % 2];
            rocke_value_t* row_bytes
                = rocke_b_mul(b, ho_row_val, rocke_b_const_i32(b, Wo * total_k * 2));
            for(int i = 0; i < n_so_meta; ++i)
            {
                rocke_value_t* ok = so_meta[i].ok;
                rocke_value_t* vec;
                rocke_value_t* off;
                if(in_h_range != NULL)
                {
                    ok = rocke_b_land(b, ok, in_h_range);
                }
                {
                    rocke_value_t* idx[2];
                    idx[0] = c0;
                    idx[1] = so_meta[i].lidx;
                    vec = rocke_b_smem_load_vN(b, so_buf, idx, 2, ctx.io_type, 8);
                }
                {
                    rocke_value_t* ad = rocke_b_add(b, so_meta[i].gbytes, row_bytes);
                    off = rocke_b_select(b, ok, ad, oob_sentinel);
                }
                rocke_dcg__buf_store(&ctx, d_rsrc, off, vec, 4);
            }
        }
        else if(should_flush)
        {
            for(int qt = 0; qt < q_subtiles; ++qt)
            {
                const int qt_w_base = qt * 16;
                rocke_value_t* out_q;
                {
                    rocke_value_t* cq = rocke_b_const_i32(b, qt_w_base);
                    rocke_value_t* a1 = rocke_b_add(b, q_tile_start, cq);
                    out_q = rocke_b_add(b, a1, q_in_lane);
                }
                rocke_value_t* out_q_ok = rocke_b_cmp_lt(b, out_q, c_W);
                for(int m = 0; m < M_LOCAL; ++m)
                {
                    if(m * 16 >= p->kpg)
                        continue;
                    rocke_value_t* acc_to_flush = ACC(qt, m, P_FLUSH);
                    rocke_value_t* k_val;
                    {
                        rocke_value_t* m1 = rocke_b_mul(b, g, c_kpg);
                        rocke_value_t* cm = rocke_b_const_i32(b, m * 16);
                        rocke_value_t* c4c = rocke_b_const_i32(b, 4);
                        rocke_value_t* m2 = rocke_b_mul(b, c4, c4c);
                        rocke_value_t* a1 = rocke_b_add(b, cm, m2);
                        k_val = rocke_b_add(b, m1, a1);
                    }
                    if(wm_base16 != NULL)
                    {
                        k_val = rocke_b_add(b, k_val, wm_base16);
                    }
                    const int rows_in_tile = p->kpg - m * 16;
                    rocke_value_t* store_ok;
                    if(rows_in_tile < 16)
                    {
                        rocke_value_t* c4c = rocke_b_const_i32(b, 4);
                        rocke_value_t* m4 = rocke_b_mul(b, c4, c4c);
                        rocke_value_t* cr = rocke_b_const_i32(b, rows_in_tile);
                        rocke_value_t* c4_ok = rocke_b_cmp_lt(b, m4, cr);
                        store_ok = rocke_b_land(b, out_q_ok, c4_ok);
                    }
                    else
                    {
                        store_ok = out_q_ok;
                    }
                    if(BLOCK_H > 0 && in_h_range != NULL)
                    {
                        store_ok = rocke_b_land(b, store_ok, in_h_range);
                    }
                    rocke_value_t* d_base;
                    {
                        const char* names[4] = {"n", "h", "w", "k"};
                        rocke_value_t* vals[4];
                        vals[0] = n;
                        vals[1] = ho_row_val;
                        vals[2] = out_q;
                        vals[3] = k_val;
                        d_base = rocke_dcg__offset(b, d_desc, names, vals, 4);
                        if(d_base == NULL)
                        {
                            return NULL;
                        }
                    }
                    if(WAVES_K > 1)
                    {
                        rocke_value_t* lane_col = rocke_b_mul(b, lane, rocke_b_const_i32(b, 4));
                        {
                            rocke_value_t* idx[2];
                            idx[0] = lds_row_idx;
                            idx[1] = lane_col;
                            rocke_b_smem_store_vN_f32(b, red_lds, idx, 2, acc_to_flush, 4);
                        }
                        rocke_b_sync(b);
                        rocke_value_t* is_k0 = rocke_b_cmp_eq(b, wave_id_k, c0);
                        rocke_if_t ifk = rocke_b_scf_if(b, is_k0);
                        if(ifk.then_region == NULL)
                        {
                            return NULL;
                        }
                        rocke_b_region_enter(b, ifk.then_region);
                        {
                            rocke_value_t* row_base
                                = rocke_b_mul(b, wave_id_q, rocke_b_const_i32(b, WAVES_K));
                            rocke_value_t** rows_f32 = (rocke_value_t**)rocke_dcg__alloc(
                                b, sizeof(rocke_value_t*) * (size_t)WAVES_K);
                            rocke_value_t* sum_slots[4];
                            if(rows_f32 == NULL)
                            {
                                return NULL;
                            }
                            for(int wk = 0; wk < WAVES_K; ++wk)
                            {
                                rocke_value_t* cw = rocke_b_const_i32(b, wk);
                                rocke_value_t* rw = rocke_b_add(b, row_base, cw);
                                rocke_value_t* idx[2];
                                idx[0] = rw;
                                idx[1] = lane_col;
                                rows_f32[wk] = rocke_b_smem_load_vN_f32(b, red_lds, idx, 2, 4);
                            }
                            for(int slot = 0; slot < 4; ++slot)
                            {
                                rocke_value_t* sv = rocke_b_vec_extract(b, rows_f32[0], slot);
                                for(int wk = 1; wk < WAVES_K; ++wk)
                                {
                                    rocke_value_t* ev = rocke_b_vec_extract(b, rows_f32[wk], slot);
                                    sv = rocke_b_fadd(b, sv, ev);
                                }
                                sum_slots[slot] = sv;
                            }
                            rocke_value_t* partial = rocke_b_vec_pack(b, sum_slots, 4, rocke_f32());
                            rocke_value_t* safe_d;
                            {
                                rocke_value_t* by2 = rocke_b_mul(b, d_base, c_half_bytes);
                                safe_d = rocke_b_select(b, store_ok, by2, oob_sentinel);
                            }
                            rocke_value_t* acc_h = rocke_dcg__trunc(&ctx, partial);
                            rocke_dcg__buf_store(&ctx, d_rsrc, safe_d, acc_h, 2);
                        }
                        rocke_b_region_leave(b);
                        rocke_b_sync(b);
                    }
                    else
                    {
                        rocke_value_t* safe_d;
                        {
                            rocke_value_t* by2 = rocke_b_mul(b, d_base, c_half_bytes);
                            safe_d = rocke_b_select(b, store_ok, by2, oob_sentinel);
                        }
                        rocke_value_t* acc_h = rocke_dcg__trunc(&ctx, acc_to_flush);
                        rocke_dcg__buf_store(&ctx, d_rsrc, safe_d, acc_h, 2);
                    }
                }
            }
        }

        for(int qt = 0; qt < q_subtiles; ++qt)
        {
            for(int m = 0; m < M_LOCAL; ++m)
            {
                ACC(qt, m, P_FLUSH) = zero_acc;
            }
        }
    }
#undef ROW_Y
#undef PW
#undef ACC

    if(PERSISTENT)
    {
        rocke_value_t* yv[1];
        yv[0] = cell_idx;
        rocke_b_scf_yield(b, yv, 1);
        rocke_b_region_leave(b);
    }
    if(!rocke_ir_builder_ok(b))
    {
        return NULL;
    }
    return rocke_ir_builder_kernel(b);
}

rocke_kernel_def_t* rocke_build_direct_conv_new(rocke_ir_builder_t* b,
                                                const rocke_direct_conv_spec_t* spec,
                                                const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        /* b = IRBuilder(spec.kernel_name()) */
        if(rocke_direct_conv_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_direct_conv(b, spec, arch);
    });
}

rocke_status_t rocke_direct_conv_lower_to_llvm(const rocke_direct_conv_spec_t* spec,
                                               const char* arch,
                                               rocke_llvm_flavor_t flavor,
                                               char** out_ll,
                                               char* err,
                                               size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        if(err != NULL && err_cap > 0)
        {
            snprintf(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    kernel = rocke_build_direct_conv_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            snprintf(
                err, err_cap, "%s", (m != NULL && m[0] != '\0') ? m : "build_direct_conv failed");
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
