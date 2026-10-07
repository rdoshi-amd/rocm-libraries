/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * gfx1250 TDM (Tensor Data Mover) descriptor construction.
 *
 * Mirrors python/rocke/core/tdm.py. The TDM engine copies a rectangular tile
 * of a global tensor into LDS from a single wave-uniform descriptor, replacing
 * the per-lane address math the direct-to-LDS path emits. The descriptor is
 * five SGPR groups (4, 8, 4, 4, 8) x i32 handed to tensor_load_to_lds; for the
 * rank<=2 non-gather shape the GEMM needs, groups 2/3/4 are zero and only
 * groups 0 and 1 carry fields.
 *
 * The bit layout is transcribed from the Python tables, which in turn mirror
 * TDM_GROUP0 / TDM_GROUP1 in CK's core/arch/amd_tdm_descriptor.hpp. Field
 * placement is the one part that cannot be derived: a wrong bit gives silent
 * garbage or a hang, never a compile error.
 */
#ifndef ROCKE_TDM_H
#define ROCKE_TDM_H

#include <stdbool.h>
#include <stdint.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* pad_interval is 3 bits and pad_amount 7 bits in TDM_GROUP1. */
#define ROCKE_TDM_PAD_INTERVAL_MAX 7
#define ROCKE_TDM_PAD_AMOUNT_MAX 127

/* tensor_dim1_stride_lo is a 16-bit field, so CK's 64-bit assignment drops
 * bits [16, 32) of the dim-1 stride. Reproduced verbatim, which caps the
 * usable leading dimension. */
#define ROCKE_TDM_MAX_DIM1_STRIDE 0xFFFF

/* TDM_GROUP1.data_size for an element width in bytes: {8:3, 4:2, 2:1, 1:0}.
 * Returns -1 for an unencodable width (Python raises ValueError). */
int rocke_tdm_data_size_code(int elem_bytes);

/* Encode a row-stride pad as (pad_interval, pad_amount), both dword-counted:
 *   pad_interval = log2(interval_bytes / 4) - 1
 *   pad_amount   = pad_bytes / 4 - 1
 * Returns false and leaves the outputs untouched where Python raises. */
bool rocke_encode_tdm_padding(int interval_bytes, int pad_bytes, int* out_interval,
                              int* out_amount);

/* (pad_enable, pad_interval, pad_amount) for a row_elems-wide tile.
 * pad_elems == 0 disables padding and zeroes the two encoded fields. */
bool rocke_tdm_padding_for_tile(int elem_bytes, int row_elems, int pad_elems, int* out_enable,
                                int* out_interval, int* out_amount);

/* Pack TDM_GROUP0 (4 x i32) from concrete integers into out_words[4]. */
void rocke_pack_tdm_group0(uint64_t lds_addr, uint64_t global_addr, int gather_index_size,
                           int gather_mode, uint32_t out_words[4]);

/* Pack TDM_GROUP1 (8 x i32) for a rank-2 non-gather transfer into
 * out_words[8]. Extents and tile dims are reversed relative to the natural
 * tile shape, so dim0 is the contiguous dimension; the strides are NOT
 * reversed. Strides and extents are in elements, not bytes. Returns false if
 * dim1_stride exceeds ROCKE_TDM_MAX_DIM1_STRIDE (Python raises). */
bool rocke_pack_tdm_group1_2d(int elem_bytes, uint64_t tensor_dim0, uint64_t tensor_dim1,
                              int tile_dim0, int tile_dim1, uint64_t dim0_stride,
                              uint64_t dim1_stride, int pad_enable, int pad_interval,
                              int pad_amount, int workgroup_mask, int atomic_barrier_enable,
                              int atomic_barrier_address, uint32_t out_words[8]);

/* Parameters for rocke_b_tdm_descriptor_2d. global_addr and lds_addr are i64
 * IR values; tensor_dim0/tensor_dim1 are i32 IR values (they shrink as the
 * window walks off the end of the tensor and the mover clips the copy).
 * dim0_stride_value, when non-NULL, supplies the row pitch at runtime and
 * dim0_stride is ignored. */
typedef struct rocke_tdm_desc_2d_params
{
    rocke_value_t* global_addr; /* i64 */
    rocke_value_t* lds_addr; /* i64 */
    rocke_value_t* tensor_dim0; /* i32 */
    rocke_value_t* tensor_dim1; /* i32 */
    rocke_value_t* dim0_stride_value; /* i32, or NULL for the constant below */
    int elem_bytes;
    int tile_dim0;
    int tile_dim1;
    uint64_t dim0_stride;
    uint64_t dim1_stride;
    int pad_enable;
    int pad_interval;
    int pad_amount;
} rocke_tdm_desc_2d_params_t;

/* Emit the five descriptor groups for a rank-2 non-gather TDM load, ready for
 * rocke_b_tensor_load_to_lds. out_groups receives (d0, d1, d2, d3, d4).
 * Mirrors tdm.build_tdm_descriptor_2d, including emission order: byte-identity
 * depends on the ops being issued in the same sequence as Python. */
bool rocke_b_tdm_descriptor_2d(rocke_ir_builder_t* b, const rocke_tdm_desc_2d_params_t* p,
                               rocke_value_t* out_groups[5]);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_TDM_H */
