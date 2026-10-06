// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Redundant with the CMake gate on purpose: a future edit that puts this file back on
// an ungated source list then yields an empty translation unit rather than a suite that
// reads a staged shard nothing produced.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaTestGraphs.hpp"

/**
 * @file TestFlydslSdpaPacks.cpp
 * @brief A census of what hipkernel:flydsl_sdpa actually loads, read from the shard the
 *        census entry names rather than from anything built in-process.
 *
 * Belongs to the census binary for the reason TestFlydslRmsNormPacks.cpp does: every case
 * opens by reading loadedSet(), none skips, and none touches a device, so an empty shard
 * fails here instead of passing silently. The seam suites -- matchers, bindings, dispatch
 * and the real launches -- are in TestFlydslSdpaEngine.cpp.
 *
 * Registered with hkp_register_census_tests() in src/tests/CMakeLists.txt.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace hip_kernel_provider::kernel_ingestor_engine::testing;

TEST(TestFlydslSdpaPacks, IsOnePackWithAGraphMatchAndOneKernelScopedMatcher)
{
    const auto& set = loadedSet(FLYDSL_SDPA.engineName);

    EXPECT_EQ(distinctPackIdCount(set), 1U);
    EXPECT_EQ(set.engine.graphMatchNativeSymbol, FLYDSL_SDPA.graphMatcher);

    // SdpaAttributes is its own union arm, so admitting the node type IS the operation
    // test: there is no graph-scoped matcher to do it a second time.
    size_t graphScoped = 0;
    size_t kernelScoped = 0;
    for(const auto& matcher : set.matchers)
    {
        (matcher.scope == MatchScope::GRAPH ? graphScoped : kernelScoped) += 1;
    }
    EXPECT_EQ(graphScoped, 0U);
    EXPECT_EQ(kernelScoped, 1U);
}

TEST(TestFlydslSdpaPacks, ShipsEveryDtypeHeadDimAndCausalVariantOnce)
{
    const auto& set = loadedSet(FLYDSL_SDPA.engineName);

    // dtype x head_dim x causal is the whole instance key: everything else the kernel
    // needs is a runtime argument. A missing class would decline silently -- the graph
    // would simply plan on another engine -- and a duplicate would make the choice
    // between two objects for one class an accident of descriptor ids.
    std::set<std::tuple<std::string, int64_t, int64_t>> classes;
    size_t kernels = 0;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            const auto* dtype
                = tryGetMetadataField<std::string>(kernel.metadata, FLYDSL_SDPA_DTYPE_FIELD);
            const auto* headDim
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_HEAD_DIM_FIELD);
            const auto* causal
                = tryGetMetadataField<int64_t>(kernel.metadata, FLYDSL_SDPA_CAUSAL_FIELD);
            ASSERT_NE(dtype, nullptr) << kernel.name;
            ASSERT_NE(headDim, nullptr) << kernel.name;
            ASSERT_NE(causal, nullptr) << kernel.name;
            classes.emplace(*dtype, *headDim, *causal);
            ++kernels;

            // Only ever kpack: pre-built flyDSL code objects with no source to fall back to.
            EXPECT_EQ(kernel.source.kind, KernelSourceKind::KPACK) << kernel.name;
            EXPECT_FALSE(kernel.source.tocKey.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.symbol.empty()) << kernel.name;
            EXPECT_FALSE(kernel.source.sha256.empty()) << kernel.name;
        }
    }

    EXPECT_EQ(kernels, classes.size()) << "two objects claim one class";
    for(const char* dtype : {"bf16", "f16"})
    {
        for(const int64_t headDim : {64, 96, 128})
        {
            for(const int64_t causal : {0, 1})
            {
                EXPECT_EQ(classes.count({dtype, headDim, causal}), 1U)
                    << dtype << " head_dim=" << headDim << " causal=" << causal;
            }
        }
    }
}

TEST(TestFlydslSdpaPacks, EveryKernelDeclaresTheTwentyNineSlotSignature)
{
    const auto& set = loadedSet(FLYDSL_SDPA.engineName);

    // Q, K, V, O, LSE pointers; eight i32 (seq_len_q, seq_len_kv, num_heads, kv_group,
    // right_bound, left_bound, align_bottom_right, lse_on); the f32 scale; fifteen i64
    // strides from offset 80. Most slots share a size with a neighbour, so a permutation
    // is invisible to the loader's kind-and-size check -- the semantic names in the
    // manifest are the only record of which is which, and this table is what a
    // regenerated layout has to agree with.
    constexpr size_t SLOTS = 29;
    for(const auto& pack : set.packs)
    {
        for(const auto& kernel : pack.kernels)
        {
            ASSERT_EQ(kernel.source.signature.size(), SLOTS) << kernel.name;
            for(size_t slot = 0; slot < SLOTS; ++slot)
            {
                const auto& argument = kernel.source.signature[slot];
                std::string kind = "by_value";
                uint32_t size = 8;
                uint32_t offset = 0;
                if(slot < 5)
                {
                    kind = "global_buffer";
                    offset = static_cast<uint32_t>(8 * slot);
                }
                else if(slot < 14)
                {
                    size = 4;
                    offset = static_cast<uint32_t>(40 + 4 * (slot - 5));
                }
                else
                {
                    offset = static_cast<uint32_t>(80 + 8 * (slot - 14));
                }
                EXPECT_EQ(argument.kind, kind) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.size, size) << kernel.name << " slot " << slot;
                EXPECT_EQ(argument.offset, offset) << kernel.name << " slot " << slot;
            }
        }
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
