// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaTestGraphs.hpp"

/**
 * @file TestFlydslSdpaEngine.cpp
 * @brief The hipkernel:flydsl_sdpa pack's seams: what its graph matcher accepts and
 *        declines, what it binds, which candidate a graph selects, and real launches of
 *        the staged objects checked against a host reference.
 *
 * The census over the staged shard lives in TestFlydslSdpaPacks.cpp, in the census
 * binary. Everything here builds its graph in-process.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;
using hipdnn_flatbuffers_sdk::data_objects::DataType;

/// Runs the graph match over @p spec, which every binding case below needs to succeed.
BoundTokens bindingsFor(const SdpaGraphSpec& spec = {})
{
    const GraphFixture fixture(buildFlydslSdpaGraph(spec));
    const auto bound = matchesGraph(FLYDSL_SDPA, fixture.context());
    if(!bound.has_value())
    {
        throw std::runtime_error("flydsl_sdpa graph match declined a graph it must serve");
    }
    return *bound;
}

int64_t boundInt(const BoundTokens& bound, const char* token)
{
    const auto value = tryGetBoundInt(bound, token);
    if(!value.has_value())
    {
        throw std::runtime_error(std::string("no integer bound for ") + token);
    }
    return *value;
}

float boundScale(const BoundTokens& bound)
{
    const auto bits = static_cast<int32_t>(boundInt(bound, FLYDSL_SDPA_SCALE_TOKEN));
    float scale = 0.0F;
    std::memcpy(&scale, &bits, sizeof(scale));
    return scale;
}

/// A kernel carrying exactly the metadata this pack's KMD declares.
KernelDefinition makeFlydslSdpaKernel(const std::string& dtype,
                                      int64_t headDim,
                                      int64_t causal,
                                      int64_t blockM = 128,
                                      int64_t blockN = 32)
{
    KernelDefinition kernel;
    kernel.name = "sdpa_" + dtype + "_d" + std::to_string(headDim);
    kernel.metadata = {{FLYDSL_SDPA_DTYPE_FIELD, dtype},
                       {FLYDSL_SDPA_HEAD_DIM_FIELD, headDim},
                       {FLYDSL_SDPA_CAUSAL_FIELD, causal},
                       {FLYDSL_SDPA_BLOCK_M_FIELD, blockM},
                       {FLYDSL_SDPA_BLOCK_N_FIELD, blockN}};
    kernel.priority = 100;
    return kernel;
}

// ---------------------------------------------------------------------------
// graph_match: what this pack accepts
// ---------------------------------------------------------------------------

/// A spec the graph match must accept, named for what it exercises.
struct AcceptedCase
{
    std::string name;
    SdpaGraphSpec spec;
};

class TestFlydslSdpaGraphAccepts : public ::testing::TestWithParam<AcceptedCase>
{
};

TEST_P(TestFlydslSdpaGraphAccepts, BindsTheGraph)
{
    const GraphFixture fixture(buildFlydslSdpaGraph(GetParam().spec));
    EXPECT_TRUE(matchesGraph(FLYDSL_SDPA, fixture.context()).has_value());
}

SdpaGraphSpec accepted(void (*edit)(SdpaGraphSpec&))
{
    SdpaGraphSpec spec;
    edit(spec);
    return spec;
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslSdpaGraphAccepts,
    ::testing::Values(
        AcceptedCase{"Bf16Bshd", accepted([](SdpaGraphSpec&) {})},
        AcceptedCase{"F16Bhsd", accepted([](SdpaGraphSpec& s) {
                         s.dataType = DataType::HALF;
                         s.layout = SdpaLayout::BHSD;
                     })},
        AcceptedCase{"HeadDim128", accepted([](SdpaGraphSpec& s) { s.headDim = 128; })},
        // The kernel takes any head_dim that is a multiple of 32 from 64 up. Whether
        // an object was SHIPPED for one is kernel_match's question, not this one's.
        AcceptedCase{"HeadDim96", accepted([](SdpaGraphSpec& s) { s.headDim = 96; })},
        AcceptedCase{"GroupedQueryHeads", accepted([](SdpaGraphSpec& s) {
                         s.heads = 8;
                         s.kvHeads = 2;
                     })},
        AcceptedCase{"MultiQuery", accepted([](SdpaGraphSpec& s) { s.kvHeads = 1; })},
        AcceptedCase{"SequenceNotAMultipleOfAnyTile", accepted([](SdpaGraphSpec& s) {
                         s.seqLenQ = 77;
                         s.seqLenKv = 77;
                     })},
        AcceptedCase{"CrossAttention", accepted([](SdpaGraphSpec& s) {
                         s.seqLenQ = 64;
                         s.seqLenKv = 300;
                     })},
        AcceptedCase{"SingleTokenDecode", accepted([](SdpaGraphSpec& s) {
                         s.seqLenQ = 1;
                         s.seqLenKv = 513;
                     })},
        AcceptedCase{"TopLeftCausal",
                     accepted([](SdpaGraphSpec& s) { s.mask = SdpaMask::TOP_LEFT_CAUSAL; })},
        AcceptedCase{"BottomRightCausalWithMoreKeys", accepted([](SdpaGraphSpec& s) {
                         s.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
                         s.seqLenQ = 64;
                         s.seqLenKv = 300;
                     })},
        // A row no key reaches gets O = 0 and LSE = -inf, as the reference defines it.
        AcceptedCase{"BottomRightCausalWithMoreQueries", accepted([](SdpaGraphSpec& s) {
                         s.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
                         s.seqLenQ = 300;
                         s.seqLenKv = 64;
                     })},
        AcceptedCase{"SlidingWindow",
                     accepted([](SdpaGraphSpec& s) { s.mask = SdpaMask::SLIDING_WINDOW; })},
        AcceptedCase{"LeftOnlyWindow", accepted([](SdpaGraphSpec& s) { s.leftBound = 32; })},
        AcceptedCase{"RightBoundBand", accepted([](SdpaGraphSpec& s) {
                         s.mask = SdpaMask::TOP_LEFT_CAUSAL;
                         s.rightBound = 16;
                     })},
        AcceptedCase{"SoftmaxStats", accepted([](SdpaGraphSpec& s) { s.stats = SdpaStats::LSE; })},
        AcceptedCase{"ScaleAbsent",
                     accepted([](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::ABSENT; })},
        AcceptedCase{"ScaleConstantTensor",
                     accepted([](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::CONSTANT_TENSOR; })},
        AcceptedCase{"ScaleRuntimeTensor",
                     accepted([](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::RUNTIME_TENSOR; })}),
    [](const ::testing::TestParamInfo<AcceptedCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// graph_match: what this pack declines, each for the reason the kernel cannot serve it
// ---------------------------------------------------------------------------

class TestFlydslSdpaGraphDeclines : public ::testing::TestWithParam<AcceptedCase>
{
};

TEST_P(TestFlydslSdpaGraphDeclines, DeclinesTheGraph)
{
    const GraphFixture fixture(buildFlydslSdpaGraph(GetParam().spec));
    EXPECT_FALSE(matchesGraph(FLYDSL_SDPA, fixture.context()).has_value());
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TestFlydslSdpaGraphDeclines,
    ::testing::Values(
        // No f32 object exists, and the kernel's WMMA has no f32 operand form.
        AcceptedCase{"Fp32", accepted([](SdpaGraphSpec& s) { s.dataType = DataType::FLOAT; })},
        // One element type throughout the kernel.
        AcceptedCase{"MixedDtypes",
                     accepted([](SdpaGraphSpec& s) { s.keyDataType = DataType::HALF; })},
        // Each lane reads a row's head-dim values as one contiguous vector.
        AcceptedCase{"StridedHeadDim",
                     accepted([](SdpaGraphSpec& s) { s.queryHeadDimStride = 2; })},
        // One head_dim for Q, K and V.
        AcceptedCase{"ValueHeadDimDiffers",
                     accepted([](SdpaGraphSpec& s) { s.valueHeadDim = 128; })},
        // The kv head is the query head divided by the group size.
        AcceptedCase{"HeadsNotAMultipleOfKvHeads", accepted([](SdpaGraphSpec& s) {
                         s.heads = 6;
                         s.kvHeads = 4;
                     })},
        // A bound below -1 has no meaning in the reference's convention.
        AcceptedCase{"LeftBoundBelowMinusOne",
                     accepted([](SdpaGraphSpec& s) { s.leftBound = -2; })},
        AcceptedCase{"AdditiveBias", accepted([](SdpaGraphSpec& s) { s.withBias = true; })},
        // The LSE output the kernel writes is f32 [B, H, Sq, 1], and only when named.
        AcceptedCase{"StatsBf16",
                     accepted([](SdpaGraphSpec& s) { s.stats = SdpaStats::LSE_BF16; })},
        AcceptedCase{"StatsWrongShape",
                     accepted([](SdpaGraphSpec& s) { s.stats = SdpaStats::LSE_WRONG_SHAPE; })},
        AcceptedCase{"StatsRequestedWithoutTensor", accepted([](SdpaGraphSpec& s) {
                         s.stats = SdpaStats::REQUESTED_WITHOUT_TENSOR;
                     })},
        // A ragged (THD) operand would be read as one dense batch.
        AcceptedCase{"RaggedQuery", accepted([](SdpaGraphSpec& s) { s.raggedQuery = true; })},
        // The running max is taken over unscaled scores, which orders them like the
        // scaled ones only for a positive scale.
        AcceptedCase{"NegativeScale", accepted([](SdpaGraphSpec& s) { s.scale = -0.125F; })},
        AcceptedCase{"Dropout", accepted([](SdpaGraphSpec& s) { s.withDropout = true; })},
        // A scale behind a device pointer is not a kernel argument.
        AcceptedCase{"ScaleDeviceTensor",
                     accepted([](SdpaGraphSpec& s) { s.scaleKind = SdpaScale::DEVICE_TENSOR; })}),
    [](const ::testing::TestParamInfo<AcceptedCase>& info) { return info.param.name; });

// ---------------------------------------------------------------------------
// graph_match: what it binds
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaBinding, BindsTheFourOperandsByUid)
{
    const auto bound = bindingsFor();
    EXPECT_EQ(boundInt(bound, std::string(FLYDSL_SDPA.inputAToken).c_str()), FLYDSL_SDPA_Q_UID);
    EXPECT_EQ(boundInt(bound, std::string(FLYDSL_SDPA.inputBToken).c_str()), FLYDSL_SDPA_K_UID);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_V_TOKEN), FLYDSL_SDPA_V_UID);
    EXPECT_EQ(boundInt(bound, std::string(FLYDSL_SDPA.outputToken).c_str()), FLYDSL_SDPA_O_UID);
}

TEST(TestFlydslSdpaBinding, AnUnmaskedGraphSelectsTheNonCausalVariant)
{
    const auto bound = bindingsFor();
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_CAUSAL_TOKEN), 0);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RIGHT_BOUND_TOKEN), -1);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_LEFT_BOUND_TOKEN), -1);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_STATS_TOKEN), FLYDSL_SDPA_NO_STATS);
}

TEST(TestFlydslSdpaBinding, TopLeftCausalIsARightBoundOfZeroAtTheTopLeft)
{
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.seqLenQ = 64;
    spec.seqLenKv = 300;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RIGHT_BOUND_TOKEN), 0);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_ALIGN_BOTTOM_RIGHT_TOKEN), 0);
}

TEST(TestFlydslSdpaBinding, BottomRightCausalLeavesTheOffsetToTheKernel)
{
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 64;
    spec.seqLenKv = 300;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_CAUSAL_TOKEN), 1);
    // The kernel derives Skv - Sq itself, so per-batch lengths can move it later.
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RIGHT_BOUND_TOKEN), 0);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_ALIGN_BOTTOM_RIGHT_TOKEN), 1);
}

TEST(TestFlydslSdpaBinding, AWindowBindsItsLeftBoundAndClampsAnOversizedOne)
{
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_LEFT_BOUND_TOKEN), 64);

    // Past Sq + Skv a window masks nothing more, and the kernel's int32 sums need the
    // bound kept in range.
    spec.leftBound = int64_t{1} << 40;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_LEFT_BOUND_TOKEN),
              spec.seqLenQ + spec.seqLenKv);
}

TEST(TestFlydslSdpaBinding, ABandSelectsTheCausalVariantWithItsRightBound)
{
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.rightBound = 16;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_CAUSAL_TOKEN), 1);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_RIGHT_BOUND_TOKEN), 16);
}

TEST(TestFlydslSdpaBinding, BindsTheStatsOutputByUid)
{
    SdpaGraphSpec spec;
    spec.stats = SdpaStats::LSE;
    EXPECT_EQ(boundInt(bindingsFor(spec), FLYDSL_SDPA_STATS_TOKEN), FLYDSL_SDPA_STATS_UID);
}

TEST(TestFlydslSdpaBinding, BindsAnExplicitScaleAsItsBitPattern)
{
    // As an integer, 0.0625 would read on the device as a denormal near zero.
    SdpaGraphSpec spec;
    spec.scale = 0.0625F;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_SCALE_SOURCE_TOKEN), FLYDSL_SDPA_SCALE_BAKED);
    EXPECT_EQ(boundScale(bound), 0.0625F);
}

TEST(TestFlydslSdpaBinding, AnAbsentScaleDefaultsToTheInverseRootOfTheHeadDim)
{
    // hipDNN's CPU reference applies this same default (CpuFpReferenceSdpa.hpp).
    SdpaGraphSpec spec;
    spec.scaleKind = SdpaScale::ABSENT;
    spec.headDim = 128;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_SCALE_SOURCE_TOKEN), FLYDSL_SDPA_SCALE_BAKED);
    EXPECT_FLOAT_EQ(boundScale(bound), 1.0F / std::sqrt(128.0F));
}

TEST(TestFlydslSdpaBinding, AScaleTensorIsBoundByUidForExecuteToResolve)
{
    SdpaGraphSpec spec;
    spec.scaleKind = SdpaScale::RUNTIME_TENSOR;
    const auto bound = bindingsFor(spec);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_SCALE_SOURCE_TOKEN), FLYDSL_SDPA_SCALE_TENSOR);
    EXPECT_EQ(boundInt(bound, FLYDSL_SDPA_SCALE_TOKEN), FLYDSL_SDPA_SCALE_UID);
}

// ---------------------------------------------------------------------------
// kernel_match: which candidate a graph selects
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaKernelMatch, AdmitsTheCandidateBuiltForTheGraphsClass)
{
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    const GraphFixture fixture(buildFlydslSdpaGraph(spec));
    const auto bound = bindingsFor(spec);
    EXPECT_TRUE(
        matchesKernel(FLYDSL_SDPA, fixture.context(), makeFlydslSdpaKernel("bf16", 128, 1), bound));
}

TEST(TestFlydslSdpaKernelMatch, DeclinesEachBakedAxisThatDisagrees)
{
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    const GraphFixture fixture(buildFlydslSdpaGraph(spec));
    const auto bound = bindingsFor(spec);

    EXPECT_FALSE(
        matchesKernel(FLYDSL_SDPA, fixture.context(), makeFlydslSdpaKernel("f16", 128, 1), bound));
    EXPECT_FALSE(
        matchesKernel(FLYDSL_SDPA, fixture.context(), makeFlydslSdpaKernel("bf16", 64, 1), bound));
    EXPECT_FALSE(
        matchesKernel(FLYDSL_SDPA, fixture.context(), makeFlydslSdpaKernel("bf16", 128, 0), bound));
}

TEST(TestFlydslSdpaKernelMatch, DeclinesACandidateWhoseTileIsNotOneTheKernelBuilds)
{
    const GraphFixture fixture(buildFlydslSdpaGraph());
    const auto bound = bindingsFor();

    // block_m is whole waves of 16 rows; block_n whole 32-key sub-tiles.
    EXPECT_FALSE(matchesKernel(
        FLYDSL_SDPA, fixture.context(), makeFlydslSdpaKernel("bf16", 64, 0, 120, 32), bound));
    EXPECT_FALSE(matchesKernel(
        FLYDSL_SDPA, fixture.context(), makeFlydslSdpaKernel("bf16", 64, 0, 128, 48), bound));
}

TEST(TestFlydslSdpaKernelMatch, DeclinesARecordMissingAFieldRatherThanThrowing)
{
    const GraphFixture fixture(buildFlydslSdpaGraph());
    const auto bound = bindingsFor();

    auto kernel = makeFlydslSdpaKernel("bf16", 64, 0);
    kernel.metadata.erase(FLYDSL_SDPA_BLOCK_M_FIELD);
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA, fixture.context(), kernel, bound));

    kernel = makeFlydslSdpaKernel("bf16", 64, 0);
    kernel.metadata[FLYDSL_SDPA_HEAD_DIM_FIELD] = std::string("64");
    EXPECT_FALSE(matchesKernel(FLYDSL_SDPA, fixture.context(), kernel, bound));
}

TEST(TestFlydslSdpaKernelMatch, ScoresByDescriptorPriority)
{
    const GraphFixture fixture(buildFlydslSdpaGraph());
    auto kernel = makeFlydslSdpaKernel("bf16", 64, 0);
    kernel.priority = 37;
    EXPECT_EQ(scoreKernel(FLYDSL_SDPA, fixture.context(), kernel), 37.0);
}

// ---------------------------------------------------------------------------
// dispatch: the seams that need no device
// ---------------------------------------------------------------------------

TEST(TestFlydslSdpaDispatch, ResolvesFromTheDispatchRegistry)
{
    EXPECT_NO_THROW(static_cast<void>(dispatchHandler(FLYDSL_SDPA)));
}

TEST(TestFlydslSdpaDispatch, NeedsNoWorkspace)
{
    const GraphFixture fixture(buildFlydslSdpaGraph());
    EXPECT_EQ(
        dispatchHandler(FLYDSL_SDPA)
            .workspaceBytes(fixture.context(), bindingsFor(), makeFlydslSdpaKernel("bf16", 64, 0)),
        0U);
}

TEST(TestFlydslSdpaDispatch, RefusesToPrepareWithoutTheMatcherSBindings)
{
    const GraphFixture fixture(buildFlydslSdpaGraph());
    EXPECT_THROW(
        dispatchHandler(FLYDSL_SDPA)
            .prepare(fixture.context(), BoundTokens{}, makeFlydslSdpaKernel("bf16", 64, 0)),
        hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---------------------------------------------------------------------------
// dispatch on device: a real launch out of the staged archive
// ---------------------------------------------------------------------------

/// The device's architecture with feature suffixes removed, which is what the shard and
/// its kpack entries are keyed on.
DeviceProperties packedArchDeviceProperties()
{
    auto properties = currentDeviceProperties();
    const auto colon = properties.gcnArchName.find(':');
    if(colon != std::string::npos)
    {
        properties.gcnArchName.resize(colon);
    }
    return properties;
}

/// The staged candidate for (@p dtype, @p headDim, @p causal), if this build packed one.
std::optional<KernelDefinition>
    findStagedKernel(const std::string& dtype, int64_t headDim, int64_t causal)
{
    const auto& set = loadedSet(FLYDSL_SDPA.engineName);
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* name
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_SDPA_DTYPE_FIELD);
            const auto* dim
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_HEAD_DIM_FIELD);
            const auto* isCausal
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_CAUSAL_FIELD);
            if(name == nullptr || dim == nullptr || isCausal == nullptr || *name != dtype
               || *dim != headDim || *isCausal != causal)
            {
                continue;
            }
            return KernelDefinition{kernel.id,
                                    pack.id,
                                    pack.dispatchId,
                                    kernel.source,
                                    kernel.metadata,
                                    kernel.priority,
                                    kernel.arch.empty() ? pack.arch : kernel.arch,
                                    kernel.originDirectory,
                                    kernel.name,
                                    kernel.treeRoot};
        }
    }
    return std::nullopt;
}

/// One operand's host values and device copy, laid out per the graph spec's strides.
template <typename T>
class DeviceTensor
{
public:
    DeviceTensor(const std::vector<int64_t>& dims,
                 const std::vector<int64_t>& strides,
                 uint32_t seed,
                 bool fill)
        : _dims(dims)
        , _strides(strides)
    {
        int64_t span = 1;
        for(size_t axis = 0; axis < dims.size(); ++axis)
        {
            span += (dims[axis] - 1) * strides[axis];
        }
        _host.assign(static_cast<size_t>(span), T(0.0F));

        if(fill)
        {
            // k/32 for |k| <= 32: exact in both bf16 and f16, so the reference reads the
            // same values the device does and input rounding stays out of the budget.
            uint32_t state = seed;
            for(auto& value : _host)
            {
                state = state * 1664525U + 1013904223U;
                const auto k = static_cast<int32_t>((state >> 16) % 65U) - 32;
                value = T(static_cast<float>(k) / 32.0F);
            }
        }

        EXPECT_EQ(hipSuccess, hipMalloc(&_device, bytes()));
        EXPECT_EQ(hipSuccess, hipMemcpy(_device, _host.data(), bytes(), hipMemcpyHostToDevice));
    }

    ~DeviceTensor()
    {
        static_cast<void>(hipFree(_device));
    }

    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    void* device() const
    {
        return _device;
    }

    /// Element (b, h, s, d) as the reference reads it.
    double at(int64_t b, int64_t h, int64_t s, int64_t d) const
    {
        const auto index = b * _strides[0] + h * _strides[1] + s * _strides[2] + d * _strides[3];
        return static_cast<double>(static_cast<float>(_host[static_cast<size_t>(index)]));
    }

    /// Copies the device contents back over the host values.
    void readBack()
    {
        EXPECT_EQ(hipSuccess, hipMemcpy(_host.data(), _device, bytes(), hipMemcpyDeviceToHost));
    }

private:
    size_t bytes() const
    {
        return _host.size() * sizeof(T);
    }

    std::vector<int64_t> _dims;
    std::vector<int64_t> _strides;
    std::vector<T> _host;
    void* _device = nullptr;
};

/// Runs the staged kernel for @p spec through prepare() and launch(), and checks every
/// output element against a double-precision reference computed from the same inputs.
///
/// Skips when the staged set holds no kernel for the spec's class: on an arch this build
/// packed nothing for, there is no kernel to be wrong about.
template <typename T>
void expectSdpaMatchesReference(const SdpaGraphSpec& spec, double absoluteTolerance)
{
    // The bounds the graph builder writes for this spec, in CpuFpReferenceSdpa's terms.
    int64_t left = -1;
    int64_t right = -1;
    if(spec.mask == SdpaMask::TOP_LEFT_CAUSAL || spec.mask == SdpaMask::BOTTOM_RIGHT_CAUSAL)
    {
        right = 0;
    }
    if(spec.mask == SdpaMask::SLIDING_WINDOW)
    {
        left = 64;
        right = 0;
    }
    left = spec.leftBound.value_or(left);
    right = spec.rightBound.value_or(right);
    const int64_t align
        = spec.mask == SdpaMask::BOTTOM_RIGHT_CAUSAL ? spec.seqLenKv - spec.seqLenQ : 0;
    const auto isMasked = [&](int64_t row, int64_t key) {
        return (right >= 0 && key >= row + 1 + align + right)
               || (left >= 0 && key < row + align - left);
    };

    const std::string dtype = spec.dataType == DataType::BFLOAT16 ? "bf16" : "f16";
    const bool causal = right >= 0;
    const auto kernel = findStagedKernel(dtype, spec.headDim, causal ? 1 : 0);
    if(!kernel.has_value())
    {
        GTEST_SKIP() << "nothing packed for dtype=" << dtype << " head_dim=" << spec.headDim
                     << " causal=" << causal << " in this build's descriptor root";
    }

    const GraphFixture fixture(buildFlydslSdpaGraph(spec), packedArchDeviceProperties());
    const auto bound = matchesGraph(FLYDSL_SDPA, fixture.context());
    ASSERT_TRUE(bound.has_value());
    ASSERT_TRUE(matchesKernel(FLYDSL_SDPA, fixture.context(), *kernel, *bound));

    const auto& handler = dispatchHandler(FLYDSL_SDPA);
    const auto prepared = handler.prepare(fixture.context(), *bound, *kernel);
    ASSERT_NE(prepared, nullptr);

    const int64_t d = spec.headDim;
    const DeviceTensor<T> q({spec.batch, spec.heads, spec.seqLenQ, d},
                            sdpaStrides(spec.layout, spec.heads, spec.seqLenQ, d),
                            1U,
                            true);
    const DeviceTensor<T> k({spec.batch, spec.kvHeads, spec.seqLenKv, d},
                            sdpaStrides(spec.layout, spec.kvHeads, spec.seqLenKv, d),
                            2U,
                            true);
    const DeviceTensor<T> v({spec.batch, spec.kvHeads, spec.seqLenKv, d},
                            sdpaStrides(spec.layout, spec.kvHeads, spec.seqLenKv, d),
                            3U,
                            true);
    DeviceTensor<T> o({spec.batch, spec.heads, spec.seqLenQ, d},
                      sdpaStrides(spec.layout, spec.heads, spec.seqLenQ, d),
                      4U,
                      false);

    // f32 [B, H, Sq, 1], packed BHSD: the layout the graph builder declares.
    DeviceTensor<float> stats({spec.batch, spec.heads, spec.seqLenQ, 1},
                              {spec.heads * spec.seqLenQ, spec.seqLenQ, 1, 1},
                              5U,
                              false);
    const bool wantStats = spec.stats == SdpaStats::LSE;

    float runtimeScale = spec.scale;
    std::vector<hipdnnPluginDeviceBuffer_t> buffers{{FLYDSL_SDPA_Q_UID, q.device()},
                                                    {FLYDSL_SDPA_K_UID, k.device()},
                                                    {FLYDSL_SDPA_V_UID, v.device()},
                                                    {FLYDSL_SDPA_O_UID, o.device()}};
    if(wantStats)
    {
        buffers.push_back({FLYDSL_SDPA_STATS_UID, stats.device()});
    }
    if(spec.scaleKind == SdpaScale::RUNTIME_TENSOR)
    {
        buffers.push_back({FLYDSL_SDPA_SCALE_UID, &runtimeScale});
    }

    const Handle handle;
    handler.launch(
        handle, *prepared, buffers.data(), static_cast<uint32_t>(buffers.size()), nullptr);
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    o.readBack();
    stats.readBack();

    const double scale = spec.scaleKind == SdpaScale::ABSENT
                             ? 1.0 / std::sqrt(static_cast<double>(d))
                             : static_cast<double>(spec.scale);
    const int64_t group = spec.heads / spec.kvHeads;

    double worst = 0.0;
    std::string where;
    double worstLse = 0.0;
    std::string whereLse;
    std::vector<double> scores(static_cast<size_t>(spec.seqLenKv));
    for(int64_t b = 0; b < spec.batch; ++b)
    {
        for(int64_t h = 0; h < spec.heads; ++h)
        {
            const int64_t kvHead = h / group;
            for(int64_t s = 0; s < spec.seqLenQ; ++s)
            {
                double maxScore = -std::numeric_limits<double>::infinity();
                for(int64_t j = 0; j < spec.seqLenKv; ++j)
                {
                    if(isMasked(s, j))
                    {
                        continue;
                    }
                    double dot = 0.0;
                    for(int64_t e = 0; e < d; ++e)
                    {
                        dot += q.at(b, h, s, e) * k.at(b, kvHead, j, e);
                    }
                    scores[static_cast<size_t>(j)] = dot * scale;
                    maxScore = std::max(maxScore, scores[static_cast<size_t>(j)]);
                }
                double sum = 0.0;
                for(int64_t j = 0; j < spec.seqLenKv; ++j)
                {
                    scores[static_cast<size_t>(j)]
                        = isMasked(s, j) ? 0.0
                                         : std::exp(scores[static_cast<size_t>(j)] - maxScore);
                    sum += scores[static_cast<size_t>(j)];
                }
                // A row no key reaches: O = 0 and LSE = -inf, as the reference defines it.
                const bool hasKeys = sum > 0.0;
                if(wantStats)
                {
                    const double expectedLse = hasKeys ? maxScore + std::log(sum)
                                                       : -std::numeric_limits<double>::infinity();
                    const double actualLse = stats.at(b, h, s, 0);
                    // A keyless row must be exactly -inf; any other value is a full miss.
                    const bool negativeInfinity = std::isinf(actualLse) && actualLse < 0.0;
                    double lseError = std::numeric_limits<double>::max();
                    if(hasKeys)
                    {
                        lseError = std::abs(actualLse - expectedLse);
                    }
                    else if(negativeInfinity)
                    {
                        lseError = 0.0;
                    }
                    if(lseError > worstLse)
                    {
                        worstLse = lseError;
                        whereLse = "b=" + std::to_string(b) + " h=" + std::to_string(h) + " s="
                                   + std::to_string(s) + ": got " + std::to_string(actualLse)
                                   + ", expected " + std::to_string(expectedLse);
                    }
                }
                for(int64_t e = 0; e < d; ++e)
                {
                    double expected = 0.0;
                    for(int64_t j = 0; j < spec.seqLenKv; ++j)
                    {
                        expected += scores[static_cast<size_t>(j)] * v.at(b, kvHead, j, e);
                    }
                    expected = hasKeys ? expected / sum : 0.0;
                    const double error = std::abs(o.at(b, h, s, e) - expected);
                    if(error > worst)
                    {
                        worst = error;
                        where = "b=" + std::to_string(b) + " h=" + std::to_string(h)
                                + " s=" + std::to_string(s) + " d=" + std::to_string(e) + ": got "
                                + std::to_string(o.at(b, h, s, e)) + ", expected "
                                + std::to_string(expected);
                    }
                }
            }
        }
    }
    EXPECT_LE(worst, absoluteTolerance) << where;
    // LSE is computed in f32 from f32 accumulators, so it is far tighter than O.
    EXPECT_LE(worstLse, 1e-3) << whereLse;
}

// P is rounded to the input dtype before the second product, so bf16's 8-bit mantissa
// sets its budget; f16's 11 bits allow an order of magnitude less.
constexpr double BF16_TOLERANCE = 3e-2;
constexpr double F16_TOLERANCE = 4e-3;

TEST(TestGpuFlydslSdpaDispatch, ComputesBf16Bshd)
{
    SKIP_IF_NO_DEVICES();
    const SdpaGraphSpec spec;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaDispatch, ComputesF16BhsdAtHeadDim128)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.layout = SdpaLayout::BHSD;
    spec.headDim = 128;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaDispatch, ComputesCausalGroupedQueryAttentionAtARaggedLength)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 77;
    spec.seqLenKv = 77;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaDispatch, ComputesBottomRightCausalCrossAttention)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 128;
    spec.seqLenQ = 64;
    spec.seqLenKv = 300;
    spec.kvHeads = 2;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaDispatch, ComputesSingleTokenDecodeOverALongCache)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.seqLenQ = 1;
    spec.seqLenKv = 513;
    spec.kvHeads = 1;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaDispatch, ReadsARuntimeScaleAtExecute)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.scaleKind = SdpaScale::RUNTIME_TENSOR;
    spec.scale = 0.07F;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaDispatch, WritesTheLogSumExpOfEachRow)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 150;
    spec.seqLenKv = 150;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaDispatch, AppliesASlidingWindowNarrowerThanATile)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::SLIDING_WINDOW;
    spec.leftBound = 37;
    spec.seqLenQ = 300;
    spec.seqLenKv = 300;
    spec.kvHeads = 2;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaDispatch, AppliesALeftOnlyWindowWithTheNonCausalObject)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.leftBound = 64;
    spec.seqLenQ = 129;
    spec.seqLenKv = 129;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaDispatch, AppliesABandToTheRightOfTheDiagonal)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.rightBound = 5;
    spec.seqLenQ = 200;
    spec.seqLenKv = 200;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaDispatch, ZeroesRowsNoKeyReaches)
{
    SKIP_IF_NO_DEVICES();
    // Bottom-right causal with more queries than keys: the first Sq - Skv rows see none.
    SdpaGraphSpec spec;
    spec.mask = SdpaMask::BOTTOM_RIGHT_CAUSAL;
    spec.seqLenQ = 300;
    spec.seqLenKv = 64;
    spec.stats = SdpaStats::LSE;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaDispatch, ComputesHeadDim96)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.headDim = 96;
    spec.mask = SdpaMask::TOP_LEFT_CAUSAL;
    spec.heads = 8;
    spec.kvHeads = 2;
    spec.seqLenQ = 113;
    spec.seqLenKv = 113;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::bfloat16>(spec, BF16_TOLERANCE);
}

TEST(TestGpuFlydslSdpaDispatch, AppliesTheDefaultScaleWhenTheGraphGivesNone)
{
    SKIP_IF_NO_DEVICES();
    SdpaGraphSpec spec;
    spec.dataType = DataType::HALF;
    spec.scaleKind = SdpaScale::ABSENT;
    expectSdpaMatchesReference<hipdnn_data_sdk::types::half>(spec, F16_TOLERANCE);
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
