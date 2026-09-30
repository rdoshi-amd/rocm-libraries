// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpaRagged.hpp>

#include "SdpaFwdGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaRaggedFwdPlan.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;

namespace
{

constexpr int64_t Q_UID = 10;
constexpr int64_t K_UID = 11;
constexpr int64_t V_UID = 12;
constexpr int64_t O_UID = 13;
constexpr int64_t RAGGED_OFFSET_Q_UID = 20;
constexpr int64_t RAGGED_OFFSET_KV_UID = 21;
constexpr int64_t STATS_UID = 14;
constexpr int64_t RAGGED_OFFSET_STATS_UID = 22;

// A uid intentionally absent from the graph's tensor map; used to set unsupported-mode optional
// uids whose mere presence must make the plan inapplicable.
constexpr int64_t UNUSED_UID = 99;

// Packed rank-4 [B=1, H=2, S=8, D=16] for a single batch (shape irrelevant to applicability).
const std::vector<int64_t> DIMS = {1, 2, 8, 16};

using Bf16Builder = GpuSdpaRaggedFwdPlanBuilder<DataType::BFLOAT16,
                                                DataType::BFLOAT16,
                                                DataType::BFLOAT16,
                                                DataType::BFLOAT16>;

// Build a bf16 ragged graph (ragged_offset on all primaries) with optional extra attrs.
flatbuffers::FlatBufferBuilder makeRaggedGraph(SdpaAttributesT attrs = {})
{
    RaggedSdpaFwdGraphOptions options;
    options.attrs = std::move(attrs);
    return createRaggedSdpaFwdGraph(Q_UID,
                                    K_UID,
                                    V_UID,
                                    O_UID,
                                    RAGGED_OFFSET_Q_UID,
                                    RAGGED_OFFSET_KV_UID,
                                    /*batch=*/1,
                                    DIMS,
                                    DIMS,
                                    DIMS,
                                    DIMS,
                                    DataType::BFLOAT16,
                                    options);
}

// ragged_offset aux [B+1,1,1,1] INT32 = cumTokens * seqStride (element offsets).
Tensor<int32_t> makeRaggedOffset(const std::vector<int64_t>& lengths, int64_t seqStride)
{
    Tensor<int32_t> off({static_cast<int64_t>(lengths.size()) + 1, 1, 1, 1});
    auto* p = off.memory().hostData();
    p[0] = 0;
    for(size_t i = 0; i < lengths.size(); ++i)
    {
        p[i + 1] = p[i] + static_cast<int32_t>(lengths[i] * seqStride);
    }
    off.memory().markHostModified();
    return off;
}

using Fp32Builder = GpuSdpaRaggedFwdPlanBuilder<DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT,
                                                DataType::FLOAT>;

// Wrap a borrowed packed host buffer as an RFC-0014 ragged tensor ([B,H,S,D], seqAxis=2, BSHD).
ShallowRaggedTensor<float> wrapRagged(float* buf,
                                      const std::vector<int64_t>& dims,
                                      int64_t seqStride,
                                      const std::vector<int64_t>& lengths)
{
    return ShallowRaggedTensor<float>(
        buf, dims, bshd(dims), SEQ_AXIS, makeRaggedOffsetAux(cumTokens(lengths), seqStride));
}

// fp32 ragged plan with unequal per-batch Q/KV lengths and an LSE output in `statsLayout`, checked
// against the CPU ragged reference (an oracle independent of the GPU kernel) writing through a
// tensor of the same layout. Both LSE buffers start at a sentinel, so padding rows must stay
// untouched and any mis-addressed row shows up as a mismatch.
void checkPlanLseAgainstCpu(RaggedStatsLayout statsLayout)
{
    const std::vector<int64_t> seqQ = {3, 5, 1};
    const std::vector<int64_t> seqKv = {4, 2, 6};
    const int64_t batch = 3;
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
    const int64_t sMaxQ = 5;
    const int64_t sMaxKv = 6;
    const int64_t totalQ = 9;
    const int64_t seqStride = numHeads * headDim;
    const bool packedStats = statsLayout == RaggedStatsLayout::PACKED;

    const std::vector<int64_t> qDims = {batch, numHeads, sMaxQ, headDim};
    const std::vector<int64_t> kvDims = {batch, numHeads, sMaxKv, headDim};
    const std::vector<int64_t> lseDims = {batch, numHeads, sMaxQ, 1};

    RaggedSdpaFwdGraphOptions options;
    options.statsUid = STATS_UID;
    options.statsLayout = statsLayout;
    if(packedStats)
    {
        options.raggedOffsetStatsUid = RAGGED_OFFSET_STATS_UID;
    }
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 batch,
                                                 qDims,
                                                 kvDims,
                                                 kvDims,
                                                 qDims,
                                                 DataType::FLOAT,
                                                 options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const Fp32Builder fp32Builder;
    ASSERT_TRUE(fp32Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto plan = fp32Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<float> q(qDims, bshd(qDims));
    Tensor<float> k(kvDims, bshd(kvDims));
    Tensor<float> v(kvDims, bshd(kvDims));
    q.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/11);
    k.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/22);
    v.fillWithRandomValues(-1.0f, 1.0f, /*seed=*/33);
    auto offQ = makeRaggedOffset(seqQ, seqStride);
    auto offKv = makeRaggedOffset(seqKv, seqStride);
    auto offLse = makeRaggedOffset(seqQ, numHeads); // packed LSE seq stride is H

    constexpr float SENTINEL = -99.0f;
    const auto makeLse = [&]() {
        Tensor<float> lse
            = packedStats ? Tensor<float>(lseDims, bshd(lseDims)) : Tensor<float>(lseDims);
        lse.fillWithValue(SENTINEL);
        return lse;
    };
    auto lsePlan = makeLse();
    auto lseCpu = makeLse();
    Tensor<float> oPlan(qDims, bshd(qDims));
    Tensor<float> oCpu(qDims, bshd(qDims));

    std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {RAGGED_OFFSET_Q_UID, offQ.memory().deviceData()},
        {RAGGED_OFFSET_KV_UID, offKv.memory().deviceData()},
        {STATS_UID, lsePlan.memory().deviceData()},
    };
    if(packedStats)
    {
        variantPack.emplace(RAGGED_OFFSET_STATS_UID, offLse.memory().deviceData());
    }
    plan->execute(variantPack);
    oPlan.markDeviceModified();
    lsePlan.markDeviceModified();

    {
        auto qR = wrapRagged(q.memory().hostData(), qDims, seqStride, seqQ);
        auto kR = wrapRagged(k.memory().hostData(), kvDims, seqStride, seqKv);
        auto vR = wrapRagged(v.memory().hostData(), kvDims, seqStride, seqKv);
        auto oR = wrapRagged(oCpu.memory().hostData(), qDims, seqStride, seqQ);
        if(packedStats)
        {
            auto lseR = wrapRagged(lseCpu.memory().hostData(), lseDims, numHeads, seqQ);
            CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                qR, kR, vR, oR, std::nullopt, -1, -1, true, &lseR);
        }
        else
        {
            CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                qR, kR, vR, oR, std::nullopt, -1, -1, true, &lseCpu);
        }
    }

    const float tolerance = 1e-4f;
    const auto* oPlanHost = oPlan.memory().hostData();
    const auto* oCpuHost = oCpu.memory().hostData();
    for(int64_t i = 0; i < totalQ * seqStride; ++i) // packed region only
    {
        EXPECT_NEAR(oPlanHost[i], oCpuHost[i], tolerance) << "output mismatch at element " << i;
    }
    const auto* lsePlanHost = lsePlan.memory().hostData();
    const auto* lseCpuHost = lseCpu.memory().hostData();
    for(int64_t i = 0; i < batch * numHeads * sMaxQ; ++i) // whole buffer, incl. padding sentinels
    {
        EXPECT_NEAR(lsePlanHost[i], lseCpuHost[i], tolerance) << "LSE mismatch at element " << i;
    }
}

} // namespace

TEST(TestGpuSdpaRaggedFwdPlanBuilder, IsApplicableForBf16RaggedNode)
{
    auto graphBuilder = makeRaggedGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const Bf16Builder bf16Builder;
    EXPECT_TRUE(bf16Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuSdpaRaggedFwdPlanBuilder, IsNotApplicableForDenseNode)
{
    // No ragged_offset on the primaries: a dense node belongs to the dense plan, not the ragged one.
    auto graphBuilder = createSdpaFwdGraph(
        Q_UID, K_UID, V_UID, O_UID, DIMS, DIMS, DIMS, DIMS, DataType::BFLOAT16);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const Bf16Builder bf16Builder;
    EXPECT_FALSE(bf16Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

TEST(TestGpuSdpaRaggedFwdPlanBuilder, IsNotApplicableForDtypeMismatch)
{
    auto graphBuilder = makeRaggedGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    // A half builder must not be applicable to a bf16 ragged graph.
    const GpuSdpaRaggedFwdPlanBuilder<DataType::HALF,
                                      DataType::HALF,
                                      DataType::HALF,
                                      DataType::HALF>
        halfBuilder;
    EXPECT_FALSE(halfBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));

    // A missing input tensor must make the plan inapplicable.
    const Bf16Builder bf16Builder;
    auto tensorMapCopy = graphWrap.getTensorMap();
    tensorMapCopy.erase(K_UID);
    EXPECT_FALSE(bf16Builder.isApplicable(graphWrap.getNode(0), tensorMapCopy));
}

TEST(TestGpuSdpaRaggedFwdPlanBuilder, IsNotApplicableForUnsupportedModes)
{
    const Bf16Builder bf16Builder;

    const auto isApplicableWith = [&](const SdpaAttributesT& attrs) {
        auto graphBuilder = makeRaggedGraph(attrs);
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
        return bf16Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap());
    };

    {
        // The padded seq-lens variant (ragged_offset + seq_len) is out of scope here.
        SdpaAttributesT attrs;
        attrs.seq_len_q_tensor_uid = UNUSED_UID;
        attrs.seq_len_kv_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        SdpaAttributesT attrs;
        attrs.alibi_mask = true;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        SdpaAttributesT attrs;
        attrs.padding_mask = true;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        // Additive bias is gated off on the ASM v3 path.
        SdpaAttributesT attrs;
        attrs.attn_mask_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        SdpaAttributesT attrs;
        attrs.dropout_probability = 0.1F;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        SdpaAttributesT attrs;
        attrs.page_table_k_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        SdpaAttributesT attrs;
        attrs.block_mask_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        // Softmax/output (re)quantization is unsupported (AITER fp8 fwd descales Q/K/V only).
        SdpaAttributesT attrs;
        attrs.descale_s_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
    {
        // max_tensor_uid (running max softmax stat) is not produced by the reference.
        SdpaAttributesT attrs;
        attrs.max_tensor_uid = UNUSED_UID;
        EXPECT_FALSE(isApplicableWith(attrs));
    }
}

TEST(TestGpuSdpaRaggedFwdPlanBuilder, PlanConstruction)
{
    auto graphBuilder = makeRaggedGraph();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const Bf16Builder bf16Builder;
    auto builtPlan = bf16Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    // Parenthesize the cast: the template's commas would otherwise be parsed as macro args.
    auto* casted
        = dynamic_cast<GpuSdpaRaggedFwdPlan<bfloat16, bfloat16, bfloat16, bfloat16, float>*>(
            builtPlan.get());
    EXPECT_NE(casted, nullptr);
}

// Wiring proof: plan execute() drives the same kernel as a direct fpropRagged call (including the
// bf16 provider probability mode and the LSE output), so identical inputs must produce identical
// output through the graph path. Equal per-batch lengths keep prod(dims) == packed, so the padded
// tensors are fully initialized and can be compared in full.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteMatchesDirectFpropRaggedBf16)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_gpu_ref::GpuFpReferenceSdpaRagged;

    const int64_t batch = 2;
    const int64_t numHeads = 2;
    const int64_t seqLen = 4; // equal per batch -> no padding
    const int64_t headDim = 16;
    const std::vector<int64_t> qkvDims = {batch, numHeads, seqLen, headDim};
    const std::vector<int64_t> lseDims = {batch, numHeads, seqLen, 1};
    const int64_t seqStride = numHeads * headDim;

    RaggedSdpaFwdGraphOptions options;
    options.statsUid = STATS_UID; // default dense stats layout
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 batch,
                                                 qkvDims,
                                                 qkvDims,
                                                 qkvDims,
                                                 qkvDims,
                                                 DataType::BFLOAT16,
                                                 options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const Bf16Builder bf16Builder;
    auto plan = bf16Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<bfloat16> q(qkvDims, bshd(qkvDims));
    Tensor<bfloat16> k(qkvDims, bshd(qkvDims));
    Tensor<bfloat16> v(qkvDims, bshd(qkvDims));
    q.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), /*seed=*/11);
    k.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), /*seed=*/22);
    v.fillWithRandomValues(bfloat16(-1.0f), bfloat16(1.0f), /*seed=*/33);
    auto offQ = makeRaggedOffset({seqLen, seqLen}, seqStride);
    auto offKv = makeRaggedOffset({seqLen, seqLen}, seqStride);

    Tensor<bfloat16> oPlan(qkvDims, bshd(qkvDims));
    Tensor<float> lsePlan(lseDims); // dense, matching the graph's stats strides
    lsePlan.fillWithValue(-987.0f); // sentinel: an unwritten LSE would retain this

    const std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {RAGGED_OFFSET_Q_UID, offQ.memory().deviceData()},
        {RAGGED_OFFSET_KV_UID, offKv.memory().deviceData()},
        {STATS_UID, lsePlan.memory().deviceData()},
    };
    plan->execute(variantPack);
    oPlan.markDeviceModified();
    lsePlan.markDeviceModified();

    // Direct reference with the same probability mode the plan selects for all-bf16.
    Tensor<bfloat16> oDirect(qkvDims, bshd(qkvDims));
    Tensor<float> lseDirect(lseDims);
    GpuFpReferenceSdpaRagged::fpropRagged<bfloat16, bfloat16, bfloat16, bfloat16, float>(
        q,
        k,
        v,
        oDirect,
        offQ,
        offKv,
        std::nullopt,
        /*leftBound=*/-1,
        /*rightBound=*/-1,
        /*topLeftAlignment=*/true,
        &lseDirect,
        /*raggedOffsetLse=*/nullptr,
        sdpaProbabilityMode<bfloat16, bfloat16, bfloat16, bfloat16>());

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<bfloat16> oValidation(tolerance, tolerance);
    EXPECT_TRUE(oValidation.allClose(oDirect, oPlan))
        << "Plan output differs from direct fpropRagged output";
    const CpuFpReferenceValidation<float> lseValidation(tolerance, tolerance);
    EXPECT_TRUE(lseValidation.allClose(lseDirect, lsePlan))
        << "Plan LSE differs from direct fpropRagged LSE";
}

// fp8 graph path: the plan must resolve the fp8 Q/K/V descale tensors from the variant pack and
// hand them to fpropRagged. Validated plan-vs-direct with identical inputs, descale, and mode.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteFp8MatchesDirectFpropRagged)
{
    SKIP_IF_NO_DEVICES();

    using hipdnn_gpu_ref::GpuFpReferenceSdpaRagged;

    constexpr int64_t DESCALE_Q_UID = 30;
    constexpr int64_t DESCALE_K_UID = 31;
    constexpr int64_t DESCALE_V_UID = 32;

    const int64_t batch = 2;
    const int64_t numHeads = 2;
    const int64_t seqLen = 4; // equal per batch -> no padding
    const int64_t headDim = 128;
    const std::vector<int64_t> qkvDims = {batch, numHeads, seqLen, headDim};
    const int64_t seqStride = numHeads * headDim;

    RaggedSdpaFwdGraphOptions options;
    options.descaleQUid = DESCALE_Q_UID;
    options.descaleKUid = DESCALE_K_UID;
    options.descaleVUid = DESCALE_V_UID;
    options.oDataType = DataType::BFLOAT16;
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 batch,
                                                 qkvDims,
                                                 qkvDims,
                                                 qkvDims,
                                                 qkvDims,
                                                 DataType::FP8_E4M3,
                                                 options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuSdpaRaggedFwdPlanBuilder<DataType::FP8_E4M3,
                                      DataType::FP8_E4M3,
                                      DataType::FP8_E4M3,
                                      DataType::BFLOAT16>
        fp8Builder;
    ASSERT_TRUE(fp8Builder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
    auto plan = fp8Builder.buildNodePlan(graphWrap, graphWrap.getNode(0));

    Tensor<fp8_e4m3> q(qkvDims, bshd(qkvDims));
    Tensor<fp8_e4m3> k(qkvDims, bshd(qkvDims));
    Tensor<fp8_e4m3> v(qkvDims, bshd(qkvDims));
    q.fillWithRandomValues(fp8_e4m3(-1.0f), fp8_e4m3(1.0f), /*seed=*/11);
    k.fillWithRandomValues(fp8_e4m3(-1.0f), fp8_e4m3(1.0f), /*seed=*/22);
    v.fillWithRandomValues(fp8_e4m3(-1.0f), fp8_e4m3(1.0f), /*seed=*/33);
    auto offQ = makeRaggedOffset({seqLen, seqLen}, seqStride);
    auto offKv = makeRaggedOffset({seqLen, seqLen}, seqStride);

    Tensor<float> descaleQ({1});
    Tensor<float> descaleK({1});
    Tensor<float> descaleV({1});
    descaleQ.memory().hostData()[0] = 0.5f;
    descaleK.memory().hostData()[0] = 0.25f;
    descaleV.memory().hostData()[0] = 2.0f;
    descaleQ.memory().markHostModified();
    descaleK.memory().markHostModified();
    descaleV.memory().markHostModified();

    Tensor<bfloat16> oPlan(qkvDims, bshd(qkvDims));
    const std::unordered_map<int64_t, void*> variantPack{
        {Q_UID, q.memory().deviceData()},
        {K_UID, k.memory().deviceData()},
        {V_UID, v.memory().deviceData()},
        {O_UID, oPlan.memory().deviceData()},
        {RAGGED_OFFSET_Q_UID, offQ.memory().deviceData()},
        {RAGGED_OFFSET_KV_UID, offKv.memory().deviceData()},
        {DESCALE_Q_UID, descaleQ.memory().deviceData()},
        {DESCALE_K_UID, descaleK.memory().deviceData()},
        {DESCALE_V_UID, descaleV.memory().deviceData()},
    };
    plan->execute(variantPack);
    oPlan.markDeviceModified();

    Tensor<bfloat16> oDirect(qkvDims, bshd(qkvDims));
    GpuFpReferenceSdpaRagged::fpropRagged<fp8_e4m3, fp8_e4m3, fp8_e4m3, bfloat16, float>(
        q,
        k,
        v,
        oDirect,
        offQ,
        offKv,
        std::nullopt,
        -1,
        -1,
        true,
        nullptr,
        nullptr,
        hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode::FLOAT,
        &descaleQ,
        &descaleK,
        &descaleV);

    const float tolerance = 1e-2f;
    const CpuFpReferenceValidation<bfloat16> validation(tolerance, tolerance);
    EXPECT_TRUE(validation.allClose(oDirect, oPlan))
        << "fp8 plan output differs from direct fpropRagged output";
}

// Frontend-default dense stats [B,H,Sq,1] with unequal per-batch lengths: each batch's LSE rows
// start at b * H * Sq_max, not at the batch's packed Q token.
TEST(TestGpuSdpaRaggedFwdPlan, ExecuteDenseStatsUnequalLengthsMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    checkPlanLseAgainstCpu(RaggedStatsLayout::DENSE);
}

// Packed stats carrying their own ragged_offset aux.
TEST(TestGpuSdpaRaggedFwdPlan, ExecutePackedStatsUnequalLengthsMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    checkPlanLseAgainstCpu(RaggedStatsLayout::PACKED);
}
