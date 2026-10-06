// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a pack that
// names a kpack archive nothing staged.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/**
 * @file FlydslSdpaNative.cpp
 * @brief The flyDSL SDPA-forward pack's native half: matching, scoring, dispatch, and the
 *        one function that registers them.
 *
 * Every kernel this pack selects is a pre-built HSACO, checked in as an `hsaco` UKD of the
 * production descriptor root and packed into the per-arch `hip_kernel_provider_<arch>.kpack`.
 * It is compiled ahead of time from
 * `flydsl/kernels_src/kernels/attention/flash_attn_func_gfx1151.py`. That kernel bakes
 * only dtype, head_dim, the causal flag and its tile; batch, both sequence lengths, both
 * head counts, every stride, the softmax scale and the causal offset are kernel
 * arguments. So one object serves every shape and layout of its (dtype, head_dim,
 * causal) class, and the matcher's job is to decline exactly what the kernel cannot
 * compute -- see `flydsl/COVERAGE.md` §5 for the list, each with its reason.
 *
 * Symbol names are restated here rather than shared via a header, since a descriptor can't
 * reference a C++ constant; the loader pre-flights every symbol a descriptor names, so a
 * mismatched string is caught without becoming a compile error.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.flydsl_sdpa.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.flydsl_sdpa.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.flydsl_sdpa.dispatch";

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_DIM_FIELD = "head_dim";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view BLOCK_M_FIELD = "block_m";
constexpr std::string_view BLOCK_N_FIELD = "block_n";

constexpr std::string_view Q_TOKEN = "flydsl_sdpa.q.uid";
constexpr std::string_view K_TOKEN = "flydsl_sdpa.k.uid";
constexpr std::string_view V_TOKEN = "flydsl_sdpa.v.uid";
constexpr std::string_view O_TOKEN = "flydsl_sdpa.o.uid";
/// 0 or 1: which of the two kernel variants the graph needs.
constexpr std::string_view CAUSAL_TOKEN = "flydsl_sdpa.causal";
/// The mask, in the reference's own terms (CpuFpReferenceSdpa isMasked): right and left
/// bounds (-1 for none) and whether the diagonal is bottom-right aligned. The kernel derives
/// the diagonal's offset from the sequence lengths itself.
constexpr std::string_view RIGHT_BOUND_TOKEN = "flydsl_sdpa.right_bound";
constexpr std::string_view LEFT_BOUND_TOKEN = "flydsl_sdpa.left_bound";
constexpr std::string_view ALIGN_BOTTOM_RIGHT_TOKEN = "flydsl_sdpa.align_bottom_right";
/// The uid of the softmax-statistics (LSE) output, or NO_STATS when the graph wants none.
constexpr std::string_view STATS_TOKEN = "flydsl_sdpa.stats.uid";
/// Where the softmax scale comes from; see ScaleSource.
constexpr std::string_view SCALE_SOURCE_TOKEN = "flydsl_sdpa.scale_source";
/// The scale's f32 bit pattern when baked (BoundTokens carry int64_t), else the uid of
/// the pass-by-value scale tensor.
constexpr std::string_view SCALE_TOKEN = "flydsl_sdpa.scale";

/// hipDNN SDPA tensors are rank 4 in LOGICAL order (B, H, S, D) whatever the layout.
constexpr uint32_t SDPA_RANK = 4;
constexpr uint32_t BATCH_AXIS = 0;
constexpr uint32_t HEAD_AXIS = 1;
constexpr uint32_t SEQ_AXIS = 2;
constexpr uint32_t HEAD_DIM_AXIS = 3;

/// Unbounded, in the left_bound/right_bound convention.
constexpr int64_t UNBOUNDED = -1;

/// The stats token's value when the graph asks for no LSE output.
constexpr int64_t NO_STATS = -1;

/// Sequence lengths the kernel's int32 mask arithmetic is sound for: a row's attendable
/// range is computed as q + offset + bound with every term below 2^29, so the sum stays
/// inside int32, as does the "no window" sentinel's distance from any kv index.
constexpr int64_t SEQ_LEN_LIMIT = 536870912LL; // 2^29

/// Kernel geometry the generator records: each wave owns 16 query rows, 32 lanes wide.
constexpr int64_t WAVE_ROWS = 16;
constexpr int64_t WAVE_SIZE = 32;
/// The kernel's KV sub-tile; block_n must be a multiple of it.
constexpr int64_t KV_SUB_TILE = 32;

constexpr int64_t INT32_LIMIT = 2147483647LL;
/// K and V are read through a buffer descriptor scoped to one (batch, kv head) slice,
/// whose num_records is 32 bits. Held to 2^31 rather than 2^32 so no part of the path
/// has to be trusted to treat it as unsigned.
constexpr int64_t KV_SLICE_BYTE_LIMIT = 2147483648LL; // 2^31
constexpr int64_t BYTES_PER_ELEMENT = 2; // bf16 and fp16 only

/// Where the kernel's `scale` argument comes from.
enum class ScaleSource : int64_t
{
    /// attn_scale_value, or 1/sqrt(head_dim) when the graph gives none -- the default
    /// hipDNN's CPU reference applies (CpuFpReferenceSdpa.hpp).
    BAKED = 0,
    /// A pass-by-value scalar tensor, resolved at execute.
    TENSOR = 1,
};

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

/// The node this pack's matchers read, or nullptr if the graph isn't a single SDPA-forward
/// node.
const data_objects::SdpaAttributes* sdpaNode(const MatchContext& context)
{
    if(context.graph.nodeCount() != 1)
    {
        return nullptr;
    }

    const auto& node = context.graph.getNodeWrapper(0);
    if(node.attributesType() != data_objects::NodeAttributes::SdpaAttributes)
    {
        return nullptr;
    }

    return &node.attributesAs<data_objects::SdpaAttributes>();
}

/// The product of @p factors, or nullopt when it does not fit in int64_t. Every product of
/// graph-controlled extents goes through this: an unchecked one is signed overflow, and a
/// wrapped value can pass a bound it should fail.
std::optional<int64_t> checkedProduct(std::initializer_list<int64_t> factors)
{
    int64_t product = 1;
    for(const int64_t factor : factors)
    {
        if(__builtin_mul_overflow(product, factor, &product))
        {
            return std::nullopt;
        }
    }
    return product;
}

/**
 * @brief Total over an UNVALIDATED graph: rank, stride/dim agreement and positive extents,
 *        checked before anything indexes an axis; and the one layout rule the kernel has.
 *
 * The kernel takes a (batch, sequence, head) stride for every operand, so BSHD, BHSD and
 * any other arrangement are equally served -- with one exception. Each lane reads a row's
 * head-dim values as one contiguous vector, so the head-dim stride must be 1. Every other
 * stride must be positive on an axis with more than one element: the K/V slice extent is
 * computed from the sequence stride, and a non-positive one would place rows outside the
 * bounds-checked range the kernel builds from it. A unit-extent axis is exempt; its index
 * is always 0.
 */
bool isServableOperand(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return false;
    }

    for(flatbuffers::uoffset_t axis = 0; axis < SDPA_RANK; ++axis)
    {
        if(dims->Get(axis) <= 0 || dims->Get(axis) > INT32_LIMIT)
        {
            return false;
        }
        if(dims->Get(axis) > 1 && strides->Get(axis) <= 0)
        {
            return false;
        }
    }

    if(strides->Get(HEAD_DIM_AXIS) != 1)
    {
        return false;
    }

    // A ragged (THD) operand packs variable-length sequences end to end, which the kernel
    // would read as one dense batch -- wrong values, not an error. Declined until varlen
    // is implemented.
    if(tensor.ragged_offset_tensor_uid().has_value())
    {
        return false;
    }

    return !tensor.virtual_() && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor);
}

/// The mask a graph asks for, in the reference's terms (CpuFpReferenceSdpa isMasked): kv is
/// masked when kv >= q + 1 + align + right (right >= 0), or kv < q + align - left
/// (left >= 0), where align is 0 top-left and Skv - Sq bottom-right.
struct SdpaMask
{
    int64_t right = UNBOUNDED;
    int64_t left = UNBOUNDED;
    bool alignBottomRight = false;
};

/**
 * @brief The graph's mask, or nullopt for one the kernel cannot apply.
 *
 * The kernel applies the reference's two-sided rule directly, so every bound pair is
 * served: causal (right 0), a band (right > 0), a sliding window (left >= 0), and either
 * diagonal alignment. The deprecated causal booleans mean right 0 at their corner; a
 * right_bound the graph also sets wins over the boolean's implied 0. Setting both booleans
 * is contradictory and declined.
 */
std::optional<SdpaMask> maskFor(const data_objects::SdpaAttributes& attributes)
{
    const bool topLeftDeprecated = attributes.causal_mask();
    const bool bottomRightDeprecated = attributes.causal_mask_bottom_right();
    if(topLeftDeprecated && bottomRightDeprecated)
    {
        return std::nullopt;
    }

    SdpaMask mask;
    mask.left = attributes.left_bound().has_value() ? attributes.left_bound().value() : UNBOUNDED;
    mask.right
        = attributes.right_bound().has_value() ? attributes.right_bound().value() : UNBOUNDED;
    if(mask.left < UNBOUNDED || mask.right < UNBOUNDED)
    {
        return std::nullopt;
    }

    if(topLeftDeprecated || bottomRightDeprecated)
    {
        if(mask.right == UNBOUNDED)
        {
            mask.right = 0;
        }
        mask.alignBottomRight = bottomRightDeprecated;
    }
    else
    {
        mask.alignBottomRight
            = attributes.diagonal_alignment() == data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    }
    return mask;
}

/// The softmax-statistics output the kernel can write: f32 `[B, H, Sq, 1]`, a real device
/// tensor, with positive strides on its non-unit axes (the kernel takes its batch, sequence
/// and head strides).
bool isServableStats(const data_objects::TensorAttributes& stats,
                     int64_t batch,
                     int64_t heads,
                     int64_t seqLenQ)
{
    const auto* dims = stats.dims();
    const auto* strides = stats.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return false;
    }
    if(dims->Get(BATCH_AXIS) != batch || dims->Get(HEAD_AXIS) != heads
       || dims->Get(SEQ_AXIS) != seqLenQ || dims->Get(HEAD_DIM_AXIS) != 1)
    {
        return false;
    }
    for(flatbuffers::uoffset_t axis = 0; axis < SDPA_RANK; ++axis)
    {
        if(dims->Get(axis) > 1 && strides->Get(axis) <= 0)
        {
            return false;
        }
    }
    return stats.data_type() == data_objects::DataType::FLOAT && !stats.virtual_()
           && !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&stats)
           && !stats.ragged_offset_tensor_uid().has_value();
}

/// The kernel metadata spelling of a graph dtype. Descriptors say `bf16`/`f16`, the
/// flatbuffer enum says `BFLOAT16`/`HALF`, and this is the one place the two meet.
std::optional<std::string> kernelDataTypeName(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::BFLOAT16:
        return std::string("bf16");
    case data_objects::DataType::HALF:
        return std::string("f16");
    default:
        return std::nullopt;
    }
}

/// The graph's shape, read from Q and K. Callers must have validated both operands.
struct SdpaProblem
{
    int64_t batch = 0;
    int64_t numHeads = 0;
    int64_t numKvHeads = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t headDim = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

SdpaProblem problemFor(const data_objects::TensorAttributes& q,
                       const data_objects::TensorAttributes& k)
{
    SdpaProblem problem;
    problem.batch = q.dims()->Get(BATCH_AXIS);
    problem.numHeads = q.dims()->Get(HEAD_AXIS);
    problem.seqLenQ = q.dims()->Get(SEQ_AXIS);
    problem.headDim = q.dims()->Get(HEAD_DIM_AXIS);
    problem.numKvHeads = k.dims()->Get(HEAD_AXIS);
    problem.seqLenKv = k.dims()->Get(SEQ_AXIS);
    problem.dataType = q.data_type();
    return problem;
}

bool hasDims(const data_objects::TensorAttributes& tensor,
             int64_t batch,
             int64_t heads,
             int64_t sequence,
             int64_t headDim)
{
    const auto* dims = tensor.dims();
    return dims->Get(BATCH_AXIS) == batch && dims->Get(HEAD_AXIS) == heads
           && dims->Get(SEQ_AXIS) == sequence && dims->Get(HEAD_DIM_AXIS) == headDim;
}

/// Does one (batch, kv head) slice of @p tensor fit the kernel's bounds-checked range?
bool kvSliceFits(const data_objects::TensorAttributes& tensor, int64_t seqLenKv, int64_t headDim)
{
    const auto rows = checkedProduct({seqLenKv - 1, tensor.strides()->Get(SEQ_AXIS)});
    if(!rows.has_value())
    {
        return false;
    }
    const auto bytes = checkedProduct({*rows + headDim, BYTES_PER_ELEMENT});
    return bytes.has_value() && *bytes < KV_SLICE_BYTE_LIMIT;
}

/**
 * @brief Graph-scoped applicability: is this a single SDPA-forward node the pre-built
 *        flyDSL kernels compute exactly?
 *
 * Refuses rather than approximates: every optional feature the kernel does not implement
 * is declined explicitly, so a graph asking for one falls through to an engine that can
 * honour it rather than being served without it.
 *
 * @warning Returning std::nullopt empties this engine's whole catalog for the graph.
 */
std::optional<BoundTokens> flydslSdpaGraphMatches(const MatchContext& context)
{
    // --- 1. One SDPA-forward node; each kernel serves one complete graph.
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;

    // --- 2. Operands, each total-checked before any axis is read.
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr)
    {
        return std::nullopt;
    }
    if(!isServableOperand(*q) || !isServableOperand(*k) || !isServableOperand(*v)
       || !isServableOperand(*o))
    {
        return std::nullopt;
    }

    // --- 3. One dtype throughout, and one the kernel was built for.
    const auto problem = problemFor(*q, *k);
    if(k->data_type() != problem.dataType || v->data_type() != problem.dataType
       || o->data_type() != problem.dataType)
    {
        return std::nullopt;
    }
    if(!kernelDataTypeName(problem.dataType).has_value())
    {
        return std::nullopt;
    }

    // --- 4. Cross-tensor shape. No broadcasting: K and V share every extent, O is Q's
    // shape, and V's head dim must equal Q/K's -- the kernel has one head_dim.
    if(!hasDims(*k, problem.batch, problem.numKvHeads, problem.seqLenKv, problem.headDim)
       || !hasDims(*v, problem.batch, problem.numKvHeads, problem.seqLenKv, problem.headDim)
       || !hasDims(*o, problem.batch, problem.numHeads, problem.seqLenQ, problem.headDim))
    {
        return std::nullopt;
    }

    // GQA/MQA: the kernel maps a query head to its kv head by integer division, so a
    // non-divisible pair would silently attend to the wrong heads.
    if(problem.numHeads % problem.numKvHeads != 0)
    {
        return std::nullopt;
    }

    // --- 5. Range limits the kernel's arithmetic assumes. The grid is one workgroup per
    // (batch, query tile, head); its smallest tile is one wave's rows, so this bound holds
    // for every candidate whatever its block_m.
    const auto queryTiles = (problem.seqLenQ + WAVE_ROWS - 1) / WAVE_ROWS;
    const auto grid = checkedProduct({problem.batch, queryTiles, problem.numHeads});
    if(!grid.has_value() || *grid > INT32_LIMIT)
    {
        return std::nullopt;
    }
    if(!kvSliceFits(*k, problem.seqLenKv, problem.headDim)
       || !kvSliceFits(*v, problem.seqLenKv, problem.headDim))
    {
        return std::nullopt;
    }

    // --- 6. The mask. A row no key reaches -- a narrow window, or bottom-right causal with
    // more queries than keys -- gets O = 0 and LSE = -inf, as the reference defines it.
    if(problem.seqLenQ > SEQ_LEN_LIMIT || problem.seqLenKv > SEQ_LEN_LIMIT)
    {
        return std::nullopt;
    }
    const auto mask = maskFor(attributes);
    if(!mask.has_value())
    {
        return std::nullopt;
    }
    // A bound past the sequence masks nothing more than one at it; clamped so the kernel's
    // int32 sums stay in range.
    const int64_t causal = mask->right >= 0 ? 1 : 0;
    const int64_t rightBound = std::min(mask->right, problem.seqLenKv);
    const int64_t leftBound = std::min(mask->left, problem.seqLenQ + problem.seqLenKv);

    // --- 7. Every optional feature the kernel does not implement, declined explicitly.

    // Additive attention bias.
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // varlen, both spellings.
    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Dropout.
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        return std::nullopt;
    }
    // Paged KV.
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value())
    {
        return std::nullopt;
    }
    // Block-sparse masks and attention sinks.
    if(attributes.block_mask_tensor_uid().has_value()
       || attributes.sink_token_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // FP8 quantization.
    if(attributes.descale_q_tensor_uid().has_value()
       || attributes.descale_k_tensor_uid().has_value()
       || attributes.descale_v_tensor_uid().has_value()
       || attributes.descale_s_tensor_uid().has_value()
       || attributes.scale_s_tensor_uid().has_value() || attributes.scale_o_tensor_uid().has_value()
       || attributes.amax_s_tensor_uid().has_value() || attributes.amax_o_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Softmax statistics: the LSE output is served; the split max / sum-exp outputs and the
    // RNG dump are not. Asking for stats without naming a tensor to write them to is a graph
    // the kernel cannot satisfy.
    if(attributes.max_tensor_uid().has_value() || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    int64_t statsUid = NO_STATS;
    if(attributes.stats_tensor_uid().has_value())
    {
        const auto* stats = findTensor(context, attributes.stats_tensor_uid().value());
        if(stats == nullptr
           || !isServableStats(*stats, problem.batch, problem.numHeads, problem.seqLenQ))
        {
            return std::nullopt;
        }
        statsUid = attributes.stats_tensor_uid().value();
    }
    else if(attributes.generate_stats().has_value() && attributes.generate_stats().value())
    {
        return std::nullopt;
    }
    // ALiBi slopes and padding masks.
    if(attributes.alibi_mask() || attributes.padding_mask())
    {
        return std::nullopt;
    }
    // mma_core_mode is the MMA operand precision. The kernel's WMMA operands are the
    // graph's own fp16/bf16 inputs, so UNSET, HALF and BFLOAT16 describe what it runs.
    // HALF is accepted on bf16 graphs too, because the cuDNN-compat shim writes HALF when
    // the caller leaves the field unset. An allow-list, so a value added later declines.
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }
    // `implementation` is an execution-strategy hint; AUTO leaves the choice to us.
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        return std::nullopt;
    }

    // --- 8. The softmax scale: a baked value, a pass-by-value tensor, or the default.
    ScaleSource scaleSource = ScaleSource::BAKED;
    int64_t scaleToken = 0;
    if(attributes.scale_tensor_uid().has_value())
    {
        // Only a host scalar can become a kernarg; a device-resident scale would need a
        // device read the kernel does not do.
        const auto* scaleTensor = findTensor(context, attributes.scale_tensor_uid().value());
        if(scaleTensor == nullptr
           || !hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(scaleTensor))
        {
            return std::nullopt;
        }
        try
        {
            static_cast<void>(hipdnn_plugin_sdk::makeScalarOperand(
                context.graph.getTensorMap(), attributes.scale_tensor_uid().value(), "scale"));
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException&)
        {
            return std::nullopt;
        }
        scaleSource = ScaleSource::TENSOR;
        scaleToken = attributes.scale_tensor_uid().value();
    }
    else
    {
        const float scale = attributes.attn_scale_value().has_value()
                                ? attributes.attn_scale_value().value()
                                : 1.0F / std::sqrt(static_cast<float>(problem.headDim));
        // The kernel takes the running max over unscaled scores, which orders them the
        // same way as the scaled ones only for a positive scale.
        if(!(scale > 0.0F) || !std::isfinite(scale))
        {
            return std::nullopt;
        }
        int32_t scaleBits = 0;
        static_assert(sizeof(scaleBits) == sizeof(scale), "float must be 32-bit to round-trip");
        std::memcpy(&scaleBits, &scale, sizeof(scale));
        scaleToken = static_cast<int64_t>(scaleBits);
    }

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    bound[std::string(CAUSAL_TOKEN)] = causal;
    bound[std::string(RIGHT_BOUND_TOKEN)] = rightBound;
    bound[std::string(LEFT_BOUND_TOKEN)] = leftBound;
    bound[std::string(ALIGN_BOTTOM_RIGHT_TOKEN)] = mask->alignBottomRight ? 1 : 0;
    bound[std::string(STATS_TOKEN)] = statsUid;
    bound[std::string(SCALE_SOURCE_TOKEN)] = static_cast<int64_t>(scaleSource);
    bound[std::string(SCALE_TOKEN)] = scaleToken;
    return bound;
}

/// The bindings a match established, re-read for dispatch.
struct FlydslSdpaBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t rightBound = UNBOUNDED;
    int64_t leftBound = UNBOUNDED;
    int64_t alignBottomRight = 0;
    int64_t stats = NO_STATS;
    ScaleSource scaleSource = ScaleSource::BAKED;
    int64_t scale = 0;
};

FlydslSdpaBinding flydslSdpaBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_sdpa dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };

    FlydslSdpaBinding binding;
    binding.q = read(Q_TOKEN);
    binding.k = read(K_TOKEN);
    binding.v = read(V_TOKEN);
    binding.o = read(O_TOKEN);
    binding.rightBound = read(RIGHT_BOUND_TOKEN);
    binding.leftBound = read(LEFT_BOUND_TOKEN);
    binding.alignBottomRight = read(ALIGN_BOTTOM_RIGHT_TOKEN);
    binding.stats = read(STATS_TOKEN);
    binding.scaleSource = static_cast<ScaleSource>(read(SCALE_SOURCE_TOKEN));
    binding.scale = read(SCALE_TOKEN);
    return binding;
}

/// A candidate's integer metadata field, or nullopt when it is absent or holds another
/// type. Total, unlike KernelDefinition::getIntMetadata: a malformed record declines
/// rather than throwing out of a match.
std::optional<int64_t> integerMetadata(const KernelDefinition& kernel, std::string_view field)
{
    const auto value = kernel.tryGetMetadata(std::string(field));
    if(!value.has_value())
    {
        return std::nullopt;
    }
    const auto* integer = std::get_if<int64_t>(&*value);
    return integer == nullptr ? std::nullopt : std::optional<int64_t>(*integer);
}

/// The string counterpart of integerMetadata, equally total.
std::optional<std::string> stringMetadata(const KernelDefinition& kernel, std::string_view field)
{
    const auto value = kernel.tryGetMetadata(std::string(field));
    if(!value.has_value())
    {
        return std::nullopt;
    }
    const auto* text = std::get_if<std::string>(&*value);
    return text == nullptr ? std::nullopt : std::optional<std::string>(*text);
}

/// A candidate's launch tile, or nullopt when its metadata describes none the kernel can
/// have been built with.
struct SdpaTile
{
    int64_t blockM = 0;
    int64_t blockN = 0;
};

std::optional<SdpaTile> candidateTile(const KernelDefinition& kernel)
{
    const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
    const auto blockN = integerMetadata(kernel, BLOCK_N_FIELD);
    if(!blockM.has_value() || !blockN.has_value() || *blockM <= 0 || *blockM % WAVE_ROWS != 0
       || *blockN <= 0 || *blockN % KV_SUB_TILE != 0)
    {
        return std::nullopt;
    }
    return SdpaTile{*blockM, *blockN};
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked metadata fit the graph?
 *
 * dtype, head_dim and the causal variant are baked; everything else is a runtime
 * argument, so they are the whole comparison.
 */
bool flydslSdpaKernelMatches(const MatchContext& context,
                             const BoundTokens& bound,
                             const KernelDefinition& kernel)
{
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return false;
    }
    const auto* q = findTensor(context, attributesPtr->q_tensor_uid());
    const auto* k = findTensor(context, attributesPtr->k_tensor_uid());
    if(q == nullptr || k == nullptr)
    {
        return false;
    }
    const auto problem = problemFor(*q, *k);

    if(!candidateTile(kernel).has_value())
    {
        return false;
    }

    const auto dataTypeName = kernelDataTypeName(problem.dataType);
    if(!dataTypeName.has_value() || stringMetadata(kernel, DTYPE_FIELD) != *dataTypeName)
    {
        return false;
    }

    if(integerMetadata(kernel, HEAD_DIM_FIELD) != problem.headDim)
    {
        return false;
    }

    const auto causal = tryGetBoundInt(bound, CAUSAL_TOKEN);
    return causal.has_value() && integerMetadata(kernel, CAUSAL_FIELD) == *causal;
}

/**
 * @brief Ranks candidates that survived kernelMatches. Higher wins.
 *
 * The descriptor `priority`, unchanged. Today exactly one candidate survives a match
 * (dtype x head_dim x causal is the instance key), so this decides nothing yet; it is
 * here so a second tile for one class is ranked by data, as RMS norm's tiers are.
 */
double flydslSdpaScore(const MatchContext& /*context*/,
                       const BoundTokens& /*bound*/,
                       const KernelDefinition& kernel)
{
    return static_cast<double>(kernel.priority);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/**
 * @brief The argument list this pack launches with, mirroring launch() one for one, and
 *        the `signature` block every one of this pack's UKDs carries.
 *
 * Q, K, V, O, LSE pointers; seq_len_q, seq_len_kv, num_heads, kv_group, right_bound,
 * left_bound, align_bottom_right, lse_on (i32); scale (f32); then the (batch, sequence,
 * head) element strides of Q, K, V, O and LSE (i64).
 *
 * Names are empty and offsets zero because neither is compared for this producer -- FlyDSL
 * emits no argument names; see requireSignatureMatch. The semantic names are recorded per
 * object in the `manifest.json` beside it, under `descriptors/FlyDSL/sdpa/<arch>/`.
 */
const std::vector<KernelArgument>& flydslSdpaKernelSignature()
{
    static const KernelArgument s_pointer{
        "global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    static const KernelArgument s_i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    static const KernelArgument s_f32{"by_value", static_cast<uint32_t>(sizeof(float)), 0, ""};
    static const KernelArgument s_i64{"by_value", static_cast<uint32_t>(sizeof(int64_t)), 0, ""};
    static const std::vector<KernelArgument> s_signature{
        s_pointer, s_pointer, s_pointer, s_pointer, s_pointer, s_i32, s_i32, s_i32, s_i32, s_i32,
        s_i32,     s_i32,     s_i32,     s_f32,     s_i64,     s_i64, s_i64, s_i64, s_i64, s_i64,
        s_i64,     s_i64,     s_i64,     s_i64,     s_i64,     s_i64, s_i64, s_i64, s_i64};
    return s_signature;
}

/// One operand's (batch, sequence, head) element strides, in the kernel's argument order.
struct OperandStrides
{
    int64_t batch = 0;
    int64_t sequence = 0;
    int64_t head = 0;
};

OperandStrides stridesOf(const data_objects::TensorAttributes& tensor)
{
    const auto* strides = tensor.strides();
    return OperandStrides{
        strides->Get(BATCH_AXIS), strides->Get(SEQ_AXIS), strides->Get(HEAD_AXIS)};
}

const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    const auto* tensor = findTensor(context, uid);
    if(tensor == nullptr)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "matched flydsl_sdpa graph has no tensor for uid " + std::to_string(uid));
    }
    return *tensor;
}

/// The compiled kernel plus everything launch() needs, owning nothing that points back
/// into the MatchContext or BoundTokens it came from.
class PreparedFlydslSdpa : public PreparedDispatch
{
public:
    PreparedFlydslSdpa(IngestorKernelCode code,
                       FlydslSdpaBinding binding,
                       SdpaProblem problem,
                       std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand,
                       std::array<OperandStrides, 5> strides)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _scaleOperand(scaleOperand)
        , _strides(strides)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslSdpaBinding& binding() const
    {
        return _binding;
    }

    const SdpaProblem& problem() const
    {
        return _problem;
    }

    const std::optional<hipdnn_plugin_sdk::ScalarOperand>& scaleOperand() const
    {
        return _scaleOperand;
    }

    /// Q, K, V, O and the LSE output (zeros when there is none), in that order.
    const std::array<OperandStrides, 5>& strides() const
    {
        return _strides;
    }

private:
    IngestorKernelCode _code;
    FlydslSdpaBinding _binding;
    SdpaProblem _problem;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _scaleOperand;
    std::array<OperandStrides, 5> _strides;
};

/**
 * @brief The native dispatch behind this pack's UDD. Everything graph/kernel-derived
 *        resolves once at prepare(); launch() only resolves buffers and the scale, so
 *        nothing mutates once prepared and concurrent execution is safe.
 */
class FlydslSdpaDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    /// @param kernelCompiler Must outlive this handler; both are process-lifetime. Unused in
    /// practice -- every kernel this pack selects comes from a kpack archive -- but
    /// buildIngestorKernelCode decides that from the kernel's source kind, not from here.
    /// @param kpackLoader Same must-outlive contract.
    FlydslSdpaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                              const compilation::KpackKernelLoader& kpackLoader)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
    {
    }

    /// The kernel's only scratch is LDS and registers.
    size_t workspaceBytes(const MatchContext& /*context*/,
                          const BoundTokens& /*bound*/,
                          const KernelDefinition& /*kernel*/) const override
    {
        return 0;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = flydslSdpaBinding(bound);

        // kernel_match declines a candidate without a usable tile, so this is defence in
        // depth: the grid and block below come from block_m, and there is no tile to
        // substitute that would not launch this object with another's geometry.
        const auto tile = candidateTile(kernel);
        if(!tile.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa: kernel '" + toString(kernel.kernelId)
                    + "' declares no usable block_m/block_n tile");
        }

        const auto& q = requireTensor(context, binding.q);
        const auto& k = requireTensor(context, binding.k);
        const auto& v = requireTensor(context, binding.v);
        const auto& o = requireTensor(context, binding.o);
        const auto problem = problemFor(q, k);

        // Build defines are ignored on the kpack path, but the signature still wants an
        // options object. Built from the arch alone: the tensor overload classifies a 4-D
        // stride order as NCHW or NHWC and throws for anything else, which attention
        // layouts are.
        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);

        auto code = buildIngestorKernelCode(
            _kernelCompiler, _kpackLoader, context, kernel, options, flydslSdpaKernelSignature());

        // `batch_x_qtiles_x_heads`: one workgroup per (batch, query tile, head), each
        // block_m / 16 waves of 32. The match bounded this product for the smallest tile.
        const auto queryTiles = (problem.seqLenQ + tile->blockM - 1) / tile->blockM;
        const auto gridX = problem.batch * queryTiles * problem.numHeads;
        code.setBlockSize(static_cast<unsigned int>(tile->blockM / WAVE_ROWS * WAVE_SIZE), 1, 1);
        code.setGridSize(static_cast<unsigned int>(gridX), 1, 1);

        std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand;
        if(binding.scaleSource == ScaleSource::TENSOR)
        {
            scaleOperand = hipdnn_plugin_sdk::makeScalarOperand(
                context.graph.getTensorMap(), binding.scale, "scale");
        }

        return std::make_unique<PreparedFlydslSdpa>(
            std::move(code),
            binding,
            problem,
            scaleOperand,
            std::array<OperandStrides, 5>{stridesOf(q),
                                          stridesOf(k),
                                          stridesOf(v),
                                          stridesOf(o),
                                          binding.stats == NO_STATS
                                              ? OperandStrides{}
                                              : stridesOf(requireTensor(context, binding.stats))});
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* /*workspace*/) const override
    {
        const auto& preparedSdpa = dynamic_cast<const PreparedFlydslSdpa&>(prepared);
        const auto& binding = preparedSdpa.binding();
        const auto& problem = preparedSdpa.problem();

        float scale = 0.0F;
        if(preparedSdpa.scaleOperand().has_value())
        {
            scale = static_cast<float>(
                hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
                    *preparedSdpa.scaleOperand(), deviceBuffers, numDeviceBuffers)));
        }
        else
        {
            const auto scaleBits = static_cast<int32_t>(binding.scale);
            std::memcpy(&scale, &scaleBits, sizeof(scale));
        }

        const auto q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers);
        const auto k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers);
        const auto v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers);
        const auto o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers);
        // The kernel writes the LSE output only when lse_on is set, so with no stats tensor
        // the pointer is never dereferenced.
        void* stats = binding.stats == NO_STATS
                          ? nullptr
                          : hipdnn_plugin_sdk::findDeviceBuffer(
                                binding.stats, deviceBuffers, numDeviceBuffers)
                                .ptr;

        const auto& s = preparedSdpa.strides();

        // Changing this argument list means changing flydslSdpaKernelSignature() with it.
        preparedSdpa.kernelForStream(handle.getStream())
            .launch(handle.getStream(),
                    q.ptr,
                    k.ptr,
                    v.ptr,
                    o.ptr,
                    stats,
                    static_cast<int32_t>(problem.seqLenQ),
                    static_cast<int32_t>(problem.seqLenKv),
                    static_cast<int32_t>(problem.numHeads),
                    static_cast<int32_t>(problem.numHeads / problem.numKvHeads),
                    static_cast<int32_t>(binding.rightBound),
                    static_cast<int32_t>(binding.leftBound),
                    static_cast<int32_t>(binding.alignBottomRight),
                    static_cast<int32_t>(binding.stats == NO_STATS ? 0 : 1),
                    scale,
                    s[0].batch,
                    s[0].sequence,
                    s[0].head,
                    s[1].batch,
                    s[1].sequence,
                    s[1].head,
                    s[2].batch,
                    s[2].sequence,
                    s[2].head,
                    s[3].batch,
                    s[3].sequence,
                    s[3].head,
                    s[4].batch,
                    s[4].sequence,
                    s[4].head);
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& flydslSdpaKpackModuleCache()
{
    // Process-lifetime, and this pack's own: a cache key is (archive, toc_key, arch), so a
    // cache shared with another pack would answer that pack's lookups too -- including the
    // RMS-norm pack, which reads the same archive.
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetFlydslSdpaModuleCache()
{
    flydslSdpaKpackModuleCache().clear();
}

namespace
{

/// This pack's dispatch handler, process-lifetime: the registry holds a non-owning pointer
/// to it, but a provider's Container is created and destroyed per handle.
const FlydslSdpaDispatchHandler& flydslSdpaDispatchHandler()
{
    static const HipMlopsKernelCompiler s_kernelCompiler;
    static const compilation::KpackKernelLoader s_kpackLoader(flydslSdpaKpackModuleCache());
    static const FlydslSdpaDispatchHandler s_dispatchHandler(s_kernelCompiler, s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerFlydslSdpaSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &flydslSdpaGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &flydslSdpaKernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &flydslSdpaScore);
    scope.add(std::string(DISPATCH_SYMBOL), &flydslSdpaDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
