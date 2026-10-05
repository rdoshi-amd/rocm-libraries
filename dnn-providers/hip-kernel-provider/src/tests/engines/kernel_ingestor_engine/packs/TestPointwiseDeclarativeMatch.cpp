// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/CompiledGraphPattern.hpp>
#include <hipdnn_plugin_sdk/ingestor/DescriptorLoader.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphPatternMatcher.hpp>
#include <hipdnn_plugin_sdk/ingestor/IDeviceResolver.hpp>
#include <hipdnn_plugin_sdk/ingestor/MakeEngine.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/ConvFwdTestGraphs.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/PointwiseTestGraphs.hpp"

/**
 * @file TestPointwiseDeclarativeMatch.cpp
 * @brief The pointwise engine described with a declarative `graph_match.nodes` pattern
 *        instead of its native graph match, loaded from real descriptor files.
 *
 * The descriptor set mirrors the unit pointwise set's ADD pack -- its KMD, UHD, UMDs, UDD
 * and native symbols -- under its own engine name and ids, so the only difference from
 * the shipped engine is which arm of `graph_match` binds the graph. Matching needs no
 * device: the engine runs against a fixed device identity.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using namespace hipdnn_plugin_sdk::ingestor;
using hip_kernel_provider::kernel_ingestor_engine::testing::buildConvFwdGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::buildPointwiseGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::buildTwoNodePointwiseGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::GraphFixture;
using hip_kernel_provider::kernel_ingestor_engine::testing::matchesGraph;
using hip_kernel_provider::kernel_ingestor_engine::testing::POINTWISE_ADD;
using hip_kernel_provider::kernel_ingestor_engine::testing::testDeviceProperties;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

constexpr const char* SCRATCH_LABEL = "pointwisedeclarative";
constexpr const char* ENGINE_NAME = "hipkernel:PointwiseDeclarative";

constexpr const char* ENGINE_ID = "3e5b0c71-9a24-4d8f-b613-0f7e2c9a4d15";
constexpr const char* SCHEMA_ID = "6a0d4f92-1c37-4b85-9e20-d4b7f13a8c06";
constexpr const char* HEURISTIC_ID = "c27f8e40-5d19-4a63-8b0e-91a4d6c2f357";
constexpr const char* ADD_MATCHER_ID = "0b94e3d7-62a8-4f1c-a5d3-7e8c20b4f619";
constexpr const char* KERNEL_MATCHER_ID = "f4c1a8e2-3b70-4d96-8e15-a2d97c6b0e38";
constexpr const char* DISPATCH_ID = "8d27b6f5-0e43-4c18-b9a7-35f1e8d2c604";
constexpr const char* PACK_ID = "1e6f3a90-c82d-4b57-a4e1-6d0b9f2c7385";

/// The contract's pointwise pattern: one binary pointwise node, its inputs and output
/// bound to the roots the native match publishes.
nlohmann::json pointwisePattern()
{
    return nlohmann::json::array({{{"kind", "op"},
                                   {"id", "pointwise"},
                                   {"op", "pointwise"},
                                   {"operands", {{"in_0", "$input_a"}, {"in_1", "$input_b"}}},
                                   {"results", {{"out_0", "$output"}}}}});
}

nlohmann::json embeddedAddKernel(const char* id, int64_t blockSize)
{
    return {{"version", "1.0"},
            {"id", id},
            {"name", "pointwise_add.f32_block" + std::to_string(blockSize)},
            {"kernel_source",
             {{"kind", "embedded_source"},
              {"source_file", "kernels/PointwiseAdd.cpp"},
              {"entry_point", "PointwiseAdd"}}},
            {"metadata", {{"block_size", blockSize}, {"dtype", "FLOAT"}, {"operation", "ADD"}}},
            {"priority", 0}};
}

void writeJson(const std::filesystem::path& path, const nlohmann::json& body)
{
    std::ofstream(path, std::ios::binary) << body.dump(2) << '\n';
}

/// One descriptor file per type, as the unit pointwise set ships them, with the UED's
/// `graph_match` on the declarative arm.
void writeDeclarativeSet(const std::filesystem::path& root)
{
    writeJson(
        root / "pointwise.kmd.json",
        {{"version", "1.0"},
         {"id", SCHEMA_ID},
         {"name", "pointwise variant fields"},
         {"fields",
          nlohmann::json::array({{{"name", "block_size"}, {"type", "int"}, {"default_value", 64}},
                                 {{"name", "dtype"}, {"type", "string"}},
                                 {{"name", "operation"}, {"type", "string"}}})}});
    writeJson(root / "pointwise.uhd.json",
              {{"version", "1.0"},
               {"id", HEURISTIC_ID},
               {"name", "pointwise selector"},
               {"kind", "native"},
               {"payload", std::string(POINTWISE_ADD.score)}});
    writeJson(root / "pointwise.ued.json",
              {{"version", "1.0"},
               {"id", ENGINE_ID},
               {"name", ENGINE_NAME},
               {"graph_match", {{"nodes", pointwisePattern()}}},
               {"heuristic", HEURISTIC_ID},
               {"metadata", SCHEMA_ID},
               {"knobs", nlohmann::json::array({"block_size"})},
               {"behavior_notes", nlohmann::json::array({"runtime_compilation"})}});
    writeJson(root / "operation_is_add.umd.json",
              {{"version", "1.0"},
               {"id", ADD_MATCHER_ID},
               {"name", "graph operation is add"},
               {"scope", "graph"},
               {"match_symbol", std::string(POINTWISE_ADD.operationMatcher)}});
    writeJson(root / "kernel_dtype_matches_graph.umd.json",
              {{"version", "1.0"},
               {"id", KERNEL_MATCHER_ID},
               {"name", "kernel dtype matches the graph's dtype"},
               {"scope", "kernel"},
               {"match_symbol", std::string(POINTWISE_ADD.kernelMatcher)}});
    writeJson(root / "pointwise.udd.json",
              {{"version", "1.0"},
               {"id", DISPATCH_ID},
               {"name", "pointwise dispatch"},
               {"dispatch_symbol", std::string(POINTWISE_ADD.dispatch)}});
    writeJson(root / "pointwise_add.kdp.json",
              {{"version", "1.0"},
               {"id", PACK_ID},
               {"name", "hipkernel:pointwise_add_declarative"},
               {"arch", nlohmann::json::array()},
               {"matchers", nlohmann::json::array({ADD_MATCHER_ID, KERNEL_MATCHER_ID})},
               {"engine", ENGINE_ID},
               {"dispatch", DISPATCH_ID},
               {"kernelDescriptors",
                nlohmann::json::array(
                    {embeddedAddKernel("5f2a9c14-7e63-4b08-9d1f-c4a6e0b83d72", 64),
                     embeddedAddKernel("a93d1e75-2c40-4f8b-b6e9-07d5f8c1a246", 256)})}});
}

/// The declarative set, loaded and validated the way the provider loads its own:
/// native symbols registered first, then every file read from disk.
std::vector<DescriptorSet> loadDeclarativeSets(const ScopedDirectory& scratch)
{
    registerNativeIngestorSymbols();
    writeDeclarativeSet(scratch.path());
    return loadValidatedDescriptorSets<Handle>(scratch.path());
}

/// A fixed, device-less machine identity: matching reads the arch and nothing else.
class FixedDeviceResolver : public IDeviceResolver<Handle>
{
public:
    DeviceId deviceId(const Handle& /*handle*/) const override
    {
        return 0;
    }

    const DeviceProperties& deviceProperties(DeviceId /*deviceId*/) const override
    {
        return _properties;
    }

private:
    DeviceProperties _properties = testDeviceProperties();
};

struct ApplicabilityCase
{
    std::string name;
    std::function<flatbuffers::FlatBufferBuilder()> build;
    bool applicable;
};

class TestPointwiseDeclarativeMatch : public ::testing::TestWithParam<ApplicabilityCase>
{
};

std::set<std::string> tokenNames(const BoundTokens& bound)
{
    std::set<std::string> names;
    for(const auto& token : bound)
    {
        names.insert(token.first);
    }
    return names;
}

} // namespace

/// The pattern compiles at load, so the set validates like the native one and the engine
/// it builds answers applicability through the compiled pattern.
TEST_P(TestPointwiseDeclarativeMatch, EngineApplicabilityFollowsThePattern)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const auto sets = loadDeclarativeSets(scratch);
    ASSERT_EQ(sets.size(), 1U);
    ASSERT_TRUE(std::holds_alternative<std::shared_ptr<const CompiledGraphPattern>>(
        sets.front().engine.graphMatch));

    const FixedDeviceResolver resolver;
    const auto engine = makeEngine<Handle, Settings, Context>(sets.front(), resolver);
    const GraphFixture fixture(GetParam().build());
    Handle handle;

    EXPECT_EQ(engine->isApplicable(handle, fixture.context().graph), GetParam().applicable);
}

INSTANTIATE_TEST_SUITE_P(
    Graphs,
    TestPointwiseDeclarativeMatch,
    ::testing::Values(
        ApplicabilityCase{"BinaryAdd", [] { return buildPointwiseGraph(); }, true},
        // Full coverage: a pattern of one node matches only a graph of one node.
        ApplicabilityCase{"TwoPointwiseNodes", [] { return buildTwoNodePointwiseGraph(); }, false},
        // `in_2` is optional and the pattern leaves it unnamed, so a graph supplying it
        // is a ternary op the pattern does not describe.
        ApplicabilityCase{"TernaryPointwise",
                          [] {
                              return buildPointwiseGraph(data_objects::PointwiseMode::ADD,
                                                         data_objects::DataType::FLOAT,
                                                         {1, 1, 1, 1},
                                                         std::nullopt,
                                                         true,
                                                         std::nullopt,
                                                         std::nullopt,
                                                         /*includeThirdOperand=*/true);
                          },
                          false},
        ApplicabilityCase{"ConvolutionForward", [] { return buildConvFwdGraph(); }, false}),
    [](const ::testing::TestParamInfo<ApplicabilityCase>& info) { return info.param.name; });

/// The declarative pattern binds exactly what the native pointwise match binds on a graph
/// both accept, so every UMD and dispatch formula written against the native binding reads
/// the same tokens under either arm.
TEST(TestPointwiseDeclarativeMatchBinding, PublishesTheNativeBindingTokenForToken)
{
    const ScopedDirectory scratch = claimScratchDirectory(SCRATCH_LABEL);
    const auto sets = loadDeclarativeSets(scratch);
    ASSERT_EQ(sets.size(), 1U);
    const auto* pattern
        = std::get_if<std::shared_ptr<const CompiledGraphPattern>>(&sets.front().engine.graphMatch);
    ASSERT_NE(pattern, nullptr);
    ASSERT_NE(*pattern, nullptr);

    const GraphFixture fixture(buildPointwiseGraph());
    const auto native = matchesGraph(POINTWISE_ADD, fixture.context());
    const auto declarative = matchGraphPattern(**pattern, fixture.context());
    ASSERT_TRUE(native.has_value());
    ASSERT_TRUE(declarative.has_value());

    ASSERT_EQ(tokenNames(*declarative), tokenNames(*native));
    for(const auto& [name, value] : *native)
    {
        EXPECT_TRUE(declarative->at(name) == value) << name;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
