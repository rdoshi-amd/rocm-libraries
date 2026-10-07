/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * gfx1250 TDM descriptor construction. Mirrors python/rocke/core/tdm.py.
 *
 * Two halves: pure bit packing (no builder), and the emitter that folds the
 * static words together with runtime values. The emitter's op ORDER is load
 * bearing -- byte-identity compares the emitted text, so every builder call
 * below is issued in the same sequence as the Python, including the operands
 * evaluated inside Python's dict literals before _insert_words runs.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/ir.h"
#include "rocke/ir_internal.h"
#include "rocke/tdm.h"

#define ROCKE_TDM_BYTES_PER_DWORD 4

/* CK's TDM_GROUP0 constructor: count = 1 and type = 2 ("set to 2 for spg")
 * are unexplained in the SPG and copied verbatim. */
#define ROCKE_TDM_GROUP0_COUNT 1
#define ROCKE_TDM_GROUP0_TYPE 2

/* (word, lsb, width) for every field these packers set. Groups 2/3/4 are zero
 * for rank<=2 non-gather transfers and have no table. Mirrors _GROUP0_FIELDS
 * and _GROUP1_FIELDS. */
typedef struct
{
    int word;
    int lsb;
    int width;
} rocke_tdm_field_t;

/* GROUP0 */
static const rocke_tdm_field_t G0_COUNT = {0, 0, 2};
static const rocke_tdm_field_t G0_LDS_ADDR = {1, 0, 32};
static const rocke_tdm_field_t G0_GLOBAL_ADDR_LO = {2, 0, 32};
static const rocke_tdm_field_t G0_GLOBAL_ADDR_HI = {3, 0, 25};
static const rocke_tdm_field_t G0_TYPE = {3, 30, 2};
static const rocke_tdm_field_t G0_GATHER_INDEX_SIZE = {0, 30, 1};
static const rocke_tdm_field_t G0_GATHER_MODE = {0, 31, 1};

/* GROUP1 */
static const rocke_tdm_field_t G1_WORKGROUP_MASK = {0, 0, 16};
static const rocke_tdm_field_t G1_DATA_SIZE = {0, 16, 2};
static const rocke_tdm_field_t G1_ATOMIC_BARRIER_ENABLE = {0, 18, 1};
static const rocke_tdm_field_t G1_ITERATE_ENABLE = {0, 19, 1};
static const rocke_tdm_field_t G1_PAD_ENABLE = {0, 20, 1};
static const rocke_tdm_field_t G1_EARLY_TIMEOUT = {0, 21, 1};
static const rocke_tdm_field_t G1_PAD_INTERVAL = {0, 22, 3};
static const rocke_tdm_field_t G1_PAD_AMOUNT = {0, 25, 7};
static const rocke_tdm_field_t G1_ATOMIC_BARRIER_ADDRESS = {1, 0, 16};
static const rocke_tdm_field_t G1_TENSOR_DIM0_LO = {1, 16, 16};
static const rocke_tdm_field_t G1_TENSOR_DIM0_HI = {2, 0, 16};
static const rocke_tdm_field_t G1_TENSOR_DIM1_LO = {2, 16, 16};
static const rocke_tdm_field_t G1_TENSOR_DIM1_HI = {3, 0, 16};
static const rocke_tdm_field_t G1_TILE_DIM0 = {3, 16, 16};
static const rocke_tdm_field_t G1_TILE_DIM1 = {4, 0, 16};
static const rocke_tdm_field_t G1_TILE_DIM2 = {4, 16, 16};
static const rocke_tdm_field_t G1_DIM0_STRIDE_LO = {5, 0, 32};
static const rocke_tdm_field_t G1_DIM0_STRIDE_HI = {6, 0, 16};
static const rocke_tdm_field_t G1_DIM1_STRIDE_LO = {6, 16, 16};
static const rocke_tdm_field_t G1_DIM1_STRIDE_HI = {7, 0, 32};

/* OR one field into a zeroed word array. Mirrors _pack; the Python raises on
 * an out-of-range value, which cannot happen for the call sites here because
 * every caller masks first, so the C clamps by masking rather than erroring. */
static void rocke_tdm_set(uint32_t* words, rocke_tdm_field_t f, uint64_t value)
{
    const uint64_t mask = (f.width >= 64) ? ~(uint64_t)0 : (((uint64_t)1 << f.width) - 1);
    words[f.word] |= (uint32_t)((value & mask) << f.lsb);
}

int rocke_tdm_data_size_code(int elem_bytes)
{
    switch(elem_bytes)
    {
    case 8:
        return 3;
    case 4:
        return 2;
    case 2:
        return 1;
    case 1:
        return 0;
    default:
        return -1; /* Python raises ValueError */
    }
}

bool rocke_encode_tdm_padding(int interval_bytes, int pad_bytes, int* out_interval,
                              int* out_amount)
{
    int interval_dwords;
    int pad_interval;
    int pad_amount;
    int bits;

    if(pad_bytes <= 0)
    {
        return false; /* pad_enable=0 is the way to express "no pad" */
    }
    if((interval_bytes % ROCKE_TDM_BYTES_PER_DWORD) != 0
       || (pad_bytes % ROCKE_TDM_BYTES_PER_DWORD) != 0)
    {
        return false; /* TDM padding is dword-granular */
    }
    interval_dwords = interval_bytes / ROCKE_TDM_BYTES_PER_DWORD;
    if(interval_dwords < 2 || (interval_dwords & (interval_dwords - 1)) != 0)
    {
        return false; /* must be a power-of-two dword count >= 2 */
    }
    /* Python: interval_dwords.bit_length() - 2 */
    bits = 0;
    while((interval_dwords >> bits) != 0)
    {
        ++bits;
    }
    pad_interval = bits - 2;
    pad_amount = pad_bytes / ROCKE_TDM_BYTES_PER_DWORD - 1;
    if(pad_interval > ROCKE_TDM_PAD_INTERVAL_MAX || pad_amount > ROCKE_TDM_PAD_AMOUNT_MAX)
    {
        return false;
    }
    if(out_interval != NULL)
    {
        *out_interval = pad_interval;
    }
    if(out_amount != NULL)
    {
        *out_amount = pad_amount;
    }
    return true;
}

bool rocke_tdm_padding_for_tile(int elem_bytes, int row_elems, int pad_elems, int* out_enable,
                                int* out_interval, int* out_amount)
{
    if(pad_elems == 0)
    {
        if(out_enable != NULL)
        {
            *out_enable = 0;
        }
        if(out_interval != NULL)
        {
            *out_interval = 0;
        }
        if(out_amount != NULL)
        {
            *out_amount = 0;
        }
        return true;
    }
    if(!rocke_encode_tdm_padding(elem_bytes * row_elems, elem_bytes * pad_elems, out_interval,
                                 out_amount))
    {
        return false;
    }
    if(out_enable != NULL)
    {
        *out_enable = 1;
    }
    return true;
}

void rocke_pack_tdm_group0(uint64_t lds_addr, uint64_t global_addr, int gather_index_size,
                           int gather_mode, uint32_t out_words[4])
{
    int i;

    for(i = 0; i < 4; ++i)
    {
        out_words[i] = 0;
    }
    rocke_tdm_set(out_words, G0_COUNT, ROCKE_TDM_GROUP0_COUNT);
    rocke_tdm_set(out_words, G0_LDS_ADDR, lds_addr & 0xFFFFFFFFu);
    rocke_tdm_set(out_words, G0_GLOBAL_ADDR_LO, global_addr & 0xFFFFFFFFu);
    rocke_tdm_set(out_words, G0_GLOBAL_ADDR_HI, (global_addr >> 32) & 0x1FFFFFFu);
    rocke_tdm_set(out_words, G0_TYPE, ROCKE_TDM_GROUP0_TYPE);
    rocke_tdm_set(out_words, G0_GATHER_INDEX_SIZE, (uint64_t)gather_index_size);
    rocke_tdm_set(out_words, G0_GATHER_MODE, (uint64_t)gather_mode);
}

bool rocke_pack_tdm_group1_2d(int elem_bytes, uint64_t tensor_dim0, uint64_t tensor_dim1,
                              int tile_dim0, int tile_dim1, uint64_t dim0_stride,
                              uint64_t dim1_stride, int pad_enable, int pad_interval,
                              int pad_amount, int workgroup_mask, int atomic_barrier_enable,
                              int atomic_barrier_address, uint32_t out_words[8])
{
    const int data_size = rocke_tdm_data_size_code(elem_bytes);
    int i;

    if(data_size < 0)
    {
        return false;
    }
    if(dim1_stride > ROCKE_TDM_MAX_DIM1_STRIDE)
    {
        return false; /* the descriptor would truncate it to 16 bits */
    }
    for(i = 0; i < 8; ++i)
    {
        out_words[i] = 0;
    }
    rocke_tdm_set(out_words, G1_WORKGROUP_MASK, (uint64_t)workgroup_mask);
    rocke_tdm_set(out_words, G1_DATA_SIZE, (uint64_t)data_size);
    rocke_tdm_set(out_words, G1_ATOMIC_BARRIER_ENABLE, (uint64_t)atomic_barrier_enable);
    rocke_tdm_set(out_words, G1_ATOMIC_BARRIER_ADDRESS, (uint64_t)atomic_barrier_address);
    rocke_tdm_set(out_words, G1_ITERATE_ENABLE, 0);
    rocke_tdm_set(out_words, G1_PAD_ENABLE, (uint64_t)pad_enable);
    rocke_tdm_set(out_words, G1_EARLY_TIMEOUT, 0);
    rocke_tdm_set(out_words, G1_PAD_INTERVAL, (uint64_t)pad_interval);
    rocke_tdm_set(out_words, G1_PAD_AMOUNT, (uint64_t)pad_amount);
    rocke_tdm_set(out_words, G1_TENSOR_DIM0_LO, tensor_dim0 & 0xFFFFu);
    rocke_tdm_set(out_words, G1_TENSOR_DIM0_HI, (tensor_dim0 >> 16) & 0xFFFFu);
    rocke_tdm_set(out_words, G1_TENSOR_DIM1_LO, tensor_dim1 & 0xFFFFu);
    rocke_tdm_set(out_words, G1_TENSOR_DIM1_HI, (tensor_dim1 >> 16) & 0xFFFFu);
    rocke_tdm_set(out_words, G1_TILE_DIM0, (uint64_t)tile_dim0);
    rocke_tdm_set(out_words, G1_TILE_DIM1, (uint64_t)tile_dim1);
    rocke_tdm_set(out_words, G1_TILE_DIM2, 0);
    rocke_tdm_set(out_words, G1_DIM0_STRIDE_LO, dim0_stride & 0xFFFFFFFFu);
    rocke_tdm_set(out_words, G1_DIM0_STRIDE_HI, (dim0_stride >> 32) & 0xFFFFu);
    /* CK truncates the dim-1 stride to 16 bits here; see ROCKE_TDM_MAX_DIM1_STRIDE. */
    rocke_tdm_set(out_words, G1_DIM1_STRIDE_LO, dim1_stride & 0xFFFFu);
    rocke_tdm_set(out_words, G1_DIM1_STRIDE_HI, (dim1_stride >> 32) & 0xFFFFFFFFu);
    return true;
}

/* Build an <n x i32> vector from constant and runtime words. Mirrors
 * _insert_words: every runtime word goes through readfirstlane first, because
 * the descriptor is read as SGPRs and a VGPR-resident word is silently wrong.
 * `dynamic[i] == NULL` means "no runtime word at i"; a static word is emitted
 * only when non-zero, matching Python's `elif static.get(index):`. */
static rocke_value_t* rocke_tdm_insert_words(rocke_ir_builder_t* b, int words,
                                             const uint32_t* static_words,
                                             rocke_value_t* const* dynamic)
{
    rocke_value_t* vec = rocke_b_zero_vec(b, rocke_i32(), words);
    int index;

    for(index = 0; index < words; ++index)
    {
        if(dynamic[index] != NULL)
        {
            rocke_value_t* lane = rocke_b_readfirstlane(b, dynamic[index]);
            vec = rocke_b_vec_insert(b, vec, lane, index);
        }
        else if(static_words[index] != 0)
        {
            rocke_value_t* word = rocke_b_const_i32(b, (int64_t)static_words[index]);
            vec = rocke_b_vec_insert(b, vec, word, index);
        }
    }
    return vec;
}

bool rocke_b_tdm_descriptor_2d(rocke_ir_builder_t* b, const rocke_tdm_desc_2d_params_t* p,
                               rocke_value_t* out_groups[5])
{
    uint32_t word0[4];
    uint32_t static1[8];
    uint32_t static1_masked[8];
    rocke_value_t* dyn0[4] = {NULL, NULL, NULL, NULL};
    rocke_value_t* dyn1[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    rocke_value_t* lds32;
    rocke_value_t* g_lo;
    rocke_value_t* g_hi;
    rocke_value_t* mask16;
    rocke_value_t* sixteen;
    rocke_value_t* dim0_lo;
    rocke_value_t* dim0_hi;
    rocke_value_t* dim1_lo;
    rocke_value_t* dim1_hi;
    const bool stride0_is_value = (p->dim0_stride_value != NULL);
    int i;

    if(b == NULL || p == NULL || out_groups == NULL)
    {
        return false;
    }

    /* Group 0: the two addresses are runtime, the rest is a constant word.
     * Emission order mirrors the Python statement order exactly. */
    /* Every builder call gets its own statement. C leaves the evaluation order
     * of function arguments unspecified, so a nested rocke_b_const_* inside an
     * argument list can be emitted before the sub-expression Python evaluates
     * first -- which shifts every later SSA number and fails byte-identity
     * while emitting semantically identical instructions. */
    lds32 = rocke_b_trunc(b, p->lds_addr, rocke_i32());
    g_lo = rocke_b_trunc(b, p->global_addr, rocke_i32());
    {
        rocke_value_t* c32 = rocke_b_const_i64(b, 32);
        rocke_value_t* shifted = rocke_b_lshr(b, p->global_addr, c32);
        rocke_value_t* narrowed = rocke_b_trunc(b, shifted, rocke_i32());
        rocke_value_t* hi_mask = rocke_b_const_i32(b, 0x1FFFFFF);
        g_hi = rocke_b_land(b, narrowed, hi_mask);
    }

    rocke_pack_tdm_group0(0, 0, 0, 0, word0);

    /* `type` shares word 3 with global_addr_hi, so that word is merged at
     * runtime rather than taken from the static pattern. Python evaluates this
     * inside the dict literal, i.e. BEFORE _insert_words emits its zero_vec. */
    dyn0[1] = lds32;
    dyn0[2] = g_lo;
    {
        rocke_value_t* type_word = rocke_b_const_i32(b, (int64_t)word0[3]);
        dyn0[3] = rocke_b_lor(b, g_hi, type_word);
    }
    {
        /* Python passes only {0: word0[0]} as the static map, so words 1..3
         * have no static fallback. */
        uint32_t static0[4] = {word0[0], 0, 0, 0};
        out_groups[0] = rocke_tdm_insert_words(b, 4, static0, dyn0);
    }

    /* Group 1: the tensor extents are runtime (they shrink as the window walks
     * off the end of the tensor and the mover clips the copy), and the row
     * pitch in slot 0 may be too. */
    if(!rocke_pack_tdm_group1_2d(p->elem_bytes, 0, 0, p->tile_dim0, p->tile_dim1,
                                 stride0_is_value ? 0 : p->dim0_stride, p->dim1_stride,
                                 p->pad_enable, p->pad_interval, p->pad_amount, 0, 0, 0,
                                 static1))
    {
        return false;
    }

    mask16 = rocke_b_const_i32(b, 0xFFFF);
    sixteen = rocke_b_const_i32(b, 16);
    dim0_lo = rocke_b_shl(b, rocke_b_land(b, p->tensor_dim0, mask16), sixteen);
    dim0_hi = rocke_b_land(b, rocke_b_lshr(b, p->tensor_dim0, sixteen), mask16);
    dim1_lo = rocke_b_shl(b, rocke_b_land(b, p->tensor_dim1, mask16), sixteen);
    dim1_hi = rocke_b_land(b, rocke_b_lshr(b, p->tensor_dim1, sixteen), mask16);

    {
        rocke_value_t* w1 = rocke_b_const_i32(b, (int64_t)static1[1]);
        dyn1[1] = rocke_b_lor(b, dim0_lo, w1);
    }
    {
        rocke_value_t* merged = rocke_b_lor(b, dim0_hi, dim1_lo);
        rocke_value_t* w2 = rocke_b_const_i32(b, (int64_t)static1[2]);
        dyn1[2] = rocke_b_lor(b, merged, w2);
    }
    {
        rocke_value_t* w3 = rocke_b_const_i32(b, (int64_t)static1[3]);
        dyn1[3] = rocke_b_lor(b, dim1_hi, w3);
    }
    if(stride0_is_value)
    {
        /* Slot 0 is 48 bits (word 5 plus word 6's low half); a row pitch that
         * fits an i32 leaves the high half at its static value. */
        dyn1[5] = p->dim0_stride_value;
    }

    /* Python's static map is {index: static1[index] for index in (0,4,5,6,7)
     * if index not in runtime1} -- words 1/2/3 are runtime-only, and 5 drops
     * out when the stride is a value. */
    for(i = 0; i < 8; ++i)
    {
        static1_masked[i] = 0;
    }
    static1_masked[0] = static1[0];
    static1_masked[4] = static1[4];
    static1_masked[6] = static1[6];
    static1_masked[7] = static1[7];
    if(!stride0_is_value)
    {
        static1_masked[5] = static1[5];
    }
    out_groups[1] = rocke_tdm_insert_words(b, 8, static1_masked, dyn1);

    out_groups[2] = rocke_b_zero_vec(b, rocke_i32(), 4);
    out_groups[3] = out_groups[2];
    out_groups[4] = rocke_b_zero_vec(b, rocke_i32(), 8);
    return true;
}
