// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>

namespace hipdnn_integration_tests::test_utils
{

// Use the canonical implementation from data_sdk.
using hipdnn_data_sdk::utilities::generateStrides;

// Creates a minimal single-node SDPA-forward flatbuffer graph with packed Q/K/V/O tensors.
// The q/k/v/o tensor uids are written into `attrs`, so callers only need to set behavior
// fields (attn_scale_value, bounds, causal flags, attn_mask_tensor_uid, or any unsupported-mode
// uid/flag) on `attrs` to exercise the signature-key, applicability, and validation paths.
//
// When `statsUid` is provided, a FLOAT log-sum-exp (LSE) output tensor is added with dims
// [B, H, Sq, 1] (derived from `qDims`) and its uid is written into attrs.stats_tensor_uid,
// exercising the LSE/stats output path. Defaults to no stats output for existing callers.
inline flatbuffers::FlatBufferBuilder
    createSdpaFwdGraph(int64_t qUid,
                       int64_t kUid,
                       int64_t vUid,
                       int64_t oUid,
                       const std::vector<int64_t>& qDims,
                       const std::vector<int64_t>& kDims,
                       const std::vector<int64_t>& vDims,
                       const std::vector<int64_t>& oDims,
                       hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                       hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT attrs = {},
                       std::optional<int64_t> statsUid = std::nullopt)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    attrs.q_tensor_uid = qUid;
    attrs.k_tensor_uid = kUid;
    attrs.v_tensor_uid = vUid;
    attrs.o_tensor_uid = oUid;
    if(statsUid.has_value())
    {
        attrs.stats_tensor_uid = statsUid;
    }

    flatbuffers::FlatBufferBuilder builder;

    const auto qStrides = generateStrides(qDims);
    const auto kStrides = generateStrides(kDims);
    const auto vStrides = generateStrides(vDims);
    const auto oStrides = generateStrides(oDims);

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, qUid, "Q", dataType, &qStrides, &qDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, kUid, "K", dataType, &kStrides, &kDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, vUid, "V", dataType, &vStrides, &vDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, oUid, "O", dataType, &oStrides, &oDims));

    // Stats/LSE output tensor: rank-4 [B, H, Sq, 1] derived from Q dims, FLOAT typed.
    std::vector<int64_t> statsDims;
    std::vector<int64_t> statsStrides;
    if(statsUid.has_value())
    {
        statsDims = {qDims[0], qDims[1], qDims[2], 1};
        statsStrides = generateStrides(statsDims);
        tensors.push_back(CreateTensorAttributesDirect(
            builder, statsUid.value(), "Stats", DataType::FLOAT, &statsStrides, &statsDims));
    }

    auto sdpaAttrs = CreateSdpaAttributes(builder, &attrs);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttrs.Union()));

    auto graph = CreateGraphDirect(
        builder, "SdpaFwdTestGraph", dataType, dataType, DataType::FLOAT, &tensors, &nodes);

    builder.Finish(graph);
    return builder;
}

// BSHD-layout element strides for a packed rank-4 [B, H, S, D] SDPA tensor: the sequence axis is
// the outermost non-batch run, so stride = [S*H*D, D, H*D, 1] (seq stride = strides[2] = H*D).
inline std::vector<int64_t> bshdStrides(const std::vector<int64_t>& dims)
{
    const auto h = dims[1];
    const auto s = dims[2];
    const auto d = dims[3];
    return {s * h * d, d, h * d, 1};
}

// Layout of the optional stats (LSE) tensor of a ragged SDPA graph.
enum class RaggedStatsLayout
{
    // Dense [B, H, Sq_max, 1] with contiguous strides and no ragged_offset: the frontend's
    // default stats layout (SdpaFwdNode fills generateStrides when strides are unset).
    DENSE,
    // Packed [B, H, Sq_max, 1] with BSHD strides (seq stride = H) carrying its own ragged_offset.
    PACKED,
};

// Storage mode of a FLOAT operand tensor (attention scale, fp8 descale) in a graph.
enum class OperandStorage
{
    DEVICE, // ordinary tensor; the variant pack carries a device pointer
    BAKED, // value stored in the graph; the variant pack may omit the uid
    RUNTIME_PASS_BY_VALUE, // is_runtime_pass_by_value; the variant pack carries a host pointer
};

// A FLOAT operand tensor of a graph. BAKED/RUNTIME_PASS_BY_VALUE operands must be scalars ([1]);
// DEVICE operands may be e.g. per-KV-head descales [B, H_kv, 1, 1].
struct FloatOperandSpec
{
    int64_t uid = 0;
    OperandStorage storage = OperandStorage::DEVICE;
    float bakedValue = 0.0f; // BAKED only
    std::vector<int64_t> dims = {1};
};

// Optional features of createRaggedSdpaFwdGraph. Every field defaults to "absent".
struct RaggedSdpaFwdGraphOptions
{
    // Behavior fields (bounds, scale, unsupported-mode uids/flags); tensor uids are filled in.
    hipdnn_flatbuffers_sdk::data_objects::SdpaAttributesT attrs{};
    // Output dtype; UNSET means the shared input dtype (fp8 graphs use a distinct bf16 output).
    hipdnn_flatbuffers_sdk::data_objects::DataType oDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    // FLOAT LSE output [B, H, Sq, 1] wired into attrs.stats_tensor_uid.
    std::optional<int64_t> statsUid;
    RaggedStatsLayout statsLayout = RaggedStatsLayout::DENSE;
    // uid of the stats tensor's own INT32 ragged_offset aux; required for PACKED stats.
    std::optional<int64_t> raggedOffsetStatsUid;
    // FLOAT fp8 descale tensors wired into attrs.descale_q/k/v_tensor_uid.
    std::optional<FloatOperandSpec> descaleQ;
    std::optional<FloatOperandSpec> descaleK;
    std::optional<FloatOperandSpec> descaleV;
    // FLOAT attention scale tensor wired into attrs.scale_tensor_uid.
    std::optional<FloatOperandSpec> scale;
};

// Creates a single-node RAGGED (RFC-0014: packed [B,H,S,D] + ragged_offset) SDPA-forward graph.
//
// q/k/v/o are rank-4 [B,H,S,D] with BSHD-layout strides. Two INT32 ragged_offset aux tensors of
// shape [batch+1,1,1,1] are added: raggedOffsetQ is shared by q and o, raggedOffsetKv by k and v
// (valid only when the shared primaries have equal per-token element counts, i.e. D_q==D_v and
// Hk==Hv — which the plan/routing tests use). Each primary's ragged_offset_tensor_uid is set, which
// routes the node to the ragged reference. Optional stats/descale tensors come from `options`.
inline flatbuffers::FlatBufferBuilder
    createRaggedSdpaFwdGraph(int64_t qUid,
                             int64_t kUid,
                             int64_t vUid,
                             int64_t oUid,
                             int64_t raggedOffsetQUid,
                             int64_t raggedOffsetKvUid,
                             int64_t batch,
                             const std::vector<int64_t>& qDims,
                             const std::vector<int64_t>& kDims,
                             const std::vector<int64_t>& vDims,
                             const std::vector<int64_t>& oDims,
                             hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                             const RaggedSdpaFwdGraphOptions& options = {})
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    // Output dtype defaults to the shared dtype; fp8 uses a distinct bf16 output.
    const DataType outputDataType
        = (options.oDataType == DataType::UNSET) ? dataType : options.oDataType;

    auto attrs = options.attrs;
    attrs.q_tensor_uid = qUid;
    attrs.k_tensor_uid = kUid;
    attrs.v_tensor_uid = vUid;
    attrs.o_tensor_uid = oUid;
    if(options.statsUid.has_value())
    {
        attrs.stats_tensor_uid = options.statsUid;
    }
    if(options.descaleQ.has_value())
    {
        attrs.descale_q_tensor_uid = options.descaleQ->uid;
    }
    if(options.descaleK.has_value())
    {
        attrs.descale_k_tensor_uid = options.descaleK->uid;
    }
    if(options.descaleV.has_value())
    {
        attrs.descale_v_tensor_uid = options.descaleV->uid;
    }
    if(options.scale.has_value())
    {
        attrs.scale_tensor_uid = options.scale->uid;
    }

    flatbuffers::FlatBufferBuilder builder;

    const auto qStrides = bshdStrides(qDims);
    const auto kStrides = bshdStrides(kDims);
    const auto vStrides = bshdStrides(vDims);
    const auto oStrides = bshdStrides(oDims);

    // Primaries carry ragged_offset_tensor_uid (q,o -> Q offset; k,v -> KV offset).
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   qUid,
                                                   "Q",
                                                   dataType,
                                                   &qStrides,
                                                   &qDims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   /*is_runtime_pass_by_value=*/false,
                                                   raggedOffsetQUid));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   kUid,
                                                   "K",
                                                   dataType,
                                                   &kStrides,
                                                   &kDims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   /*is_runtime_pass_by_value=*/false,
                                                   raggedOffsetKvUid));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   vUid,
                                                   "V",
                                                   dataType,
                                                   &vStrides,
                                                   &vDims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   /*is_runtime_pass_by_value=*/false,
                                                   raggedOffsetKvUid));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   oUid,
                                                   "O",
                                                   outputDataType,
                                                   &oStrides,
                                                   &oDims,
                                                   /*virtual_=*/false,
                                                   TensorValue::NONE,
                                                   /*value=*/0,
                                                   /*is_runtime_pass_by_value=*/false,
                                                   raggedOffsetQUid));

    // ragged_offset aux tensors: INT32, rank-4 [batch+1, 1, 1, 1].
    const std::vector<int64_t> offsetDims = {batch + 1, 1, 1, 1};
    const auto offsetStrides = generateStrides(offsetDims);
    tensors.push_back(CreateTensorAttributesDirect(
        builder, raggedOffsetQUid, "RaggedOffsetQ", DataType::INT32, &offsetStrides, &offsetDims));
    tensors.push_back(CreateTensorAttributesDirect(builder,
                                                   raggedOffsetKvUid,
                                                   "RaggedOffsetKv",
                                                   DataType::INT32,
                                                   &offsetStrides,
                                                   &offsetDims));

    // LSE output tensor: rank-4 [B, H, Sq, 1], FLOAT typed; dense or packed (own ragged_offset).
    if(options.statsUid.has_value())
    {
        const std::vector<int64_t> statsDims = {qDims[0], qDims[1], qDims[2], 1};
        if(options.statsLayout == RaggedStatsLayout::DENSE)
        {
            const auto statsStrides = generateStrides(statsDims);
            tensors.push_back(CreateTensorAttributesDirect(builder,
                                                           options.statsUid.value(),
                                                           "Stats",
                                                           DataType::FLOAT,
                                                           &statsStrides,
                                                           &statsDims));
        }
        else
        {
            const auto statsStrides = bshdStrides(statsDims);
            tensors.push_back(CreateTensorAttributesDirect(builder,
                                                           options.statsUid.value(),
                                                           "Stats",
                                                           DataType::FLOAT,
                                                           &statsStrides,
                                                           &statsDims,
                                                           /*virtual_=*/false,
                                                           TensorValue::NONE,
                                                           /*value=*/0,
                                                           /*is_runtime_pass_by_value=*/false,
                                                           options.raggedOffsetStatsUid.value()));
            tensors.push_back(CreateTensorAttributesDirect(builder,
                                                           options.raggedOffsetStatsUid.value(),
                                                           "RaggedOffsetStats",
                                                           DataType::INT32,
                                                           &offsetStrides,
                                                           &offsetDims));
        }
    }

    // FLOAT operand tensor in the requested storage mode (see OperandStorage).
    const auto addFloatOperand = [&](const FloatOperandSpec& spec, const char* name) {
        const auto strides = generateStrides(spec.dims);
        const bool baked = spec.storage == OperandStorage::BAKED;
        const Float32Value bakedValue(spec.bakedValue);
        const flatbuffers::Offset<void> value
            = baked ? builder.CreateStruct(bakedValue).Union() : flatbuffers::Offset<void>();
        tensors.push_back(CreateTensorAttributesDirect(
            builder,
            spec.uid,
            name,
            DataType::FLOAT,
            &strides,
            &spec.dims,
            /*virtual_=*/false,
            baked ? TensorValue::Float32Value : TensorValue::NONE,
            value,
            /*is_runtime_pass_by_value=*/spec.storage == OperandStorage::RUNTIME_PASS_BY_VALUE));
    };

    for(const auto& [spec, name] : {std::pair{&options.scale, "Scale"},
                                    std::pair{&options.descaleQ, "DescaleQ"},
                                    std::pair{&options.descaleK, "DescaleK"},
                                    std::pair{&options.descaleV, "DescaleV"}})
    {
        if(spec->has_value())
        {
            addFloatOperand(spec->value(), name);
        }
    }

    auto sdpaAttrs = CreateSdpaAttributes(builder, &attrs);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_ragged_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttrs.Union()));

    auto graph = CreateGraphDirect(
        builder, "SdpaRaggedFwdTestGraph", dataType, dataType, DataType::FLOAT, &tensors, &nodes);

    builder.Finish(graph);
    return builder;
}

} // namespace hipdnn_integration_tests::test_utils
