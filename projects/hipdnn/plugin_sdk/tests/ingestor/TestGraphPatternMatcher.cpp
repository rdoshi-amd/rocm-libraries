// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/BindingPublication.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphPatternMatcher.hpp>

#include "GraphPatternTestSupport.hpp"

namespace
{
using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::testing::compilePatternText;
using hipdnn_plugin_sdk::ingestor::testing::convolutionBwdDataNode;
using hipdnn_plugin_sdk::ingestor::testing::convolutionFwdNode;
using hipdnn_plugin_sdk::ingestor::testing::PatternGraphBuilder;
using hipdnn_plugin_sdk::ingestor::testing::pointwiseNode;

/// The pattern equivalent of the native single-node binary pointwise matcher.
const std::string BINARY_POINTWISE = R"([{"kind": "op", "id": "pointwise", "op": "pointwise",
    "operands": {"in_0": "$input_a", "in_1": "$input_b"}, "results": {"out_0": "$output"}}])";

const std::string OPTIONAL_SECOND_OPERAND = R"([{"kind": "op", "id": "pointwise",
    "op": "pointwise", "operands": {"in_0": "$input_a", "in_1": "$input_b?"},
    "results": {"out_0": "$output"}}])";

const std::string CONV_ADD_CHAIN = R"([
    {"kind": "op", "id": "conv", "op": "convolution_fwd",
     "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}},
    {"kind": "op", "id": "add", "op": "pointwise",
     "operands": {"in_0": "$conv_out", "in_1": "$bias"}, "results": {"out_0": "$y"}}
])";

std::optional<BoundTokens> match(const std::string& pattern, const PatternGraphBuilder& graph)
{
    const auto compiled = compilePatternText(pattern);
    const auto built = graph.build();
    return matchGraphPattern(*compiled, built.context());
}

bool hasTokenWithRoot(const BoundTokens& bound, const std::string& root)
{
    for(const auto& entry : bound)
    {
        if(entry.first == root || entry.first.rfind(root + ".", 0) == 0)
        {
            return true;
        }
    }
    return false;
}

PatternGraphBuilder binaryPointwiseGraph(PointwiseAttributesT attributes)
{
    attributes.in_0_tensor_uid = 1;
    attributes.in_1_tensor_uid = 2;
    attributes.out_0_tensor_uid = 3;
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3).node(pointwiseNode(attributes));
    return graph;
}

PointwiseAttributesT addAttributes()
{
    PointwiseAttributesT attributes;
    attributes.operation = PointwiseMode::ADD;
    return attributes;
}

TEST(TestGraphPatternMatcher, SinglePointwisePublishesTheCanonicalTokenMap)
{
    const auto compiled = compilePatternText(BINARY_POINTWISE);
    const auto graph = binaryPointwiseGraph(addAttributes()).build();
    const auto context = graph.context();

    const auto bound = matchGraphPattern(*compiled, context);
    ASSERT_TRUE(bound.has_value());

    BoundTokens expected;
    const auto& tensors = *context.graph.getGraph().tensors();
    ASSERT_TRUE(publishGraph(expected, context.graph));
    ASSERT_TRUE(publishTensor(expected, "input_a", tensors.Get(0)));
    ASSERT_TRUE(publishTensor(expected, "input_b", tensors.Get(1)));
    ASSERT_TRUE(publishTensor(expected, "output", tensors.Get(2)));
    expected.emplace("pointwise.operation", std::string("ADD"));
    expected.emplace("pointwise.compute_data_type", std::string("FLOAT"));
    EXPECT_EQ(*bound, expected);
    EXPECT_EQ(tryGetBoundInt(*bound, "input_b"), 2);
}

TEST(TestGraphPatternMatcher, PublishesOptionalAttributesOnlyWhenSet)
{
    auto attributes = addAttributes();
    attributes.relu_lower_clip = 0.5F;
    attributes.axis_tensor_uid = 7;
    const auto bound = match(BINARY_POINTWISE, binaryPointwiseGraph(attributes));
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(std::get<double>(bound->at("pointwise.relu_lower_clip")), 0.5);
    EXPECT_EQ(tryGetBoundInt(*bound, "pointwise.axis_tensor_uid"), 7);
    EXPECT_EQ(bound->count("pointwise.relu_upper_clip"), 0U);
    EXPECT_EQ(bound->count("pointwise.swish_beta"), 0U);
}

TEST(TestGraphPatternMatcher, DeclinesNonfiniteFloatAttribute)
{
    for(const float value :
        {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
    {
        auto attributes = addAttributes();
        attributes.swish_beta = value;
        EXPECT_FALSE(match(BINARY_POINTWISE, binaryPointwiseGraph(attributes)).has_value());
    }
}

TEST(TestGraphPatternMatcher, DeclinesNodeCountMismatch)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3).tensor(4);
    graph.node(pointwiseNode(PointwiseMode::ADD, 1, 2, 3));
    graph.node(pointwiseNode(PointwiseMode::RELU_FWD, 3, std::nullopt, 4));
    EXPECT_FALSE(match(BINARY_POINTWISE, graph).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesEmptyGraph)
{
    EXPECT_FALSE(match(BINARY_POINTWISE, PatternGraphBuilder{}).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesWrongOpType)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3).node(convolutionFwdNode(1, 2, 3));
    EXPECT_FALSE(match(BINARY_POINTWISE, graph).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesMissingRequiredOperand)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(3).node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 3));
    EXPECT_FALSE(match(BINARY_POINTWISE, graph).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesEdgeNamingAnUnknownTensor)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(3).node(pointwiseNode(PointwiseMode::ADD, 1, 9, 3));
    EXPECT_FALSE(match(BINARY_POINTWISE, graph).has_value());
}

TEST(TestGraphPatternMatcher, OptionalOperandMayBeAbsentAndPublishesOnlyWhenPresent)
{
    PatternGraphBuilder unary;
    unary.tensor(1).tensor(3).node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 3));
    const auto absent = match(OPTIONAL_SECOND_OPERAND, unary);
    ASSERT_TRUE(absent.has_value());
    EXPECT_FALSE(hasTokenWithRoot(*absent, "input_b"));
    EXPECT_EQ(tryGetBoundInt(*absent, "input_a"), 1);

    const auto present = match(OPTIONAL_SECOND_OPERAND, binaryPointwiseGraph(addAttributes()));
    ASSERT_TRUE(present.has_value());
    EXPECT_EQ(tryGetBoundInt(*present, "input_b"), 2);
    EXPECT_EQ(tryGetBoundInt(*present, "input_b.rank"), 4);
}

TEST(TestGraphPatternMatcher, DeclinesUnnamedOptionalOperandThatIsPresent)
{
    const std::string unaryPattern = R"([{"kind": "op", "id": "pw", "op": "pointwise",
        "operands": {"in_0": "$a"}, "results": {"out_0": "$out"}}])";
    EXPECT_FALSE(match(unaryPattern, binaryPointwiseGraph(addAttributes())).has_value());

    auto ternary = addAttributes();
    ternary.in_2_tensor_uid = 4;
    auto graph = binaryPointwiseGraph(ternary);
    graph.tensor(4);
    EXPECT_FALSE(match(BINARY_POINTWISE, graph).has_value());
}

TEST(TestGraphPatternMatcher, AliasedOperandsMatchDistinctVariables)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(3).node(pointwiseNode(PointwiseMode::MUL, 1, 1, 3));
    const auto bound = match(BINARY_POINTWISE, graph);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "input_a"), 1);
    EXPECT_EQ(tryGetBoundInt(*bound, "input_b"), 1);
}

PatternGraphBuilder convAddGraph(int64_t addInput)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3, true).tensor(4).tensor(5).tensor(6);
    graph.node(convolutionFwdNode(1, 2, 3));
    graph.node(pointwiseNode(PointwiseMode::ADD, addInput, 4, 5));
    return graph;
}

TEST(TestGraphPatternMatcher, ChainBindsIntermediateConsistently)
{
    const auto bound = match(CONV_ADD_CHAIN, convAddGraph(3));
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "x"), 1);
    EXPECT_EQ(tryGetBoundInt(*bound, "w"), 2);
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_out"), 3);
    EXPECT_TRUE(std::get<bool>(bound->at("conv_out.virtual")));
    EXPECT_EQ(tryGetBoundInt(*bound, "bias"), 4);
    EXPECT_EQ(tryGetBoundInt(*bound, "y"), 5);
    EXPECT_EQ(std::get<std::string>(bound->at("conv.conv_mode")), "CROSS_CORRELATION");
    EXPECT_EQ(std::get<std::string>(bound->at("add.operation")), "ADD");
    EXPECT_EQ(tryGetBoundInt(*bound, "graph.node_count"), 2);
}

TEST(TestGraphPatternMatcher, DeclinesChainWhoseConsumerReadsAnotherTensor)
{
    EXPECT_FALSE(match(CONV_ADD_CHAIN, convAddGraph(6)).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesGraphInputVariableProducedInsideTheGraph)
{
    // `$a` is not produced by any pattern node, so it must be a graph input; the graph
    // computes it with the convolution instead.
    const std::string disconnected = R"([
        {"kind": "op", "id": "conv", "op": "convolution_fwd",
         "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}},
        {"kind": "op", "id": "add", "op": "pointwise",
         "operands": {"in_0": "$a", "in_1": "$bias"}, "results": {"out_0": "$y"}}
    ])";
    EXPECT_FALSE(match(disconnected, convAddGraph(3)).has_value());
    EXPECT_TRUE(match(disconnected, convAddGraph(6)).has_value());
}

TEST(TestGraphPatternMatcher, OpcodeSetMatchesEitherMember)
{
    const std::string anyConvolution = R"([{"kind": "op", "id": "conv",
        "op": {"one_of": ["convolution_fwd", "convolution_bwd_data"]},
        "operands": {"w": "$w"}}])";

    PatternGraphBuilder forward;
    forward.tensor(1).tensor(2).tensor(3).node(convolutionFwdNode(1, 2, 3));
    const auto forwardBound = match(anyConvolution, forward);
    ASSERT_TRUE(forwardBound.has_value());
    EXPECT_EQ(tryGetBoundInt(*forwardBound, "w"), 2);
    EXPECT_EQ(std::get<std::string>(forwardBound->at("conv.conv_mode")), "CROSS_CORRELATION");

    PatternGraphBuilder backward;
    backward.tensor(1).tensor(2).tensor(3).node(convolutionBwdDataNode(1, 2, 3));
    const auto backwardBound = match(anyConvolution, backward);
    ASSERT_TRUE(backwardBound.has_value());
    EXPECT_EQ(tryGetBoundInt(*backwardBound, "w"), 2);
    EXPECT_EQ(std::get<std::string>(backwardBound->at("conv.conv_mode")), "CONVOLUTION");
}

TEST(TestGraphPatternMatcher, BacktracksWhenGraphOrderDiffersFromPatternOrder)
{
    const std::string unaryChain = R"([
        {"kind": "op", "id": "first", "op": "pointwise",
         "operands": {"in_0": "$a"}, "results": {"out_0": "$mid"}},
        {"kind": "op", "id": "second", "op": "pointwise",
         "operands": {"in_0": "$mid"}, "results": {"out_0": "$out"}}
    ])";
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3);
    graph.node(pointwiseNode(PointwiseMode::EXP, 2, std::nullopt, 3));
    graph.node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 2));

    const auto bound = match(unaryChain, graph);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "a"), 1);
    EXPECT_EQ(tryGetBoundInt(*bound, "mid"), 2);
    EXPECT_EQ(tryGetBoundInt(*bound, "out"), 3);
    EXPECT_EQ(std::get<std::string>(bound->at("first.operation")), "ABS");
    EXPECT_EQ(std::get<std::string>(bound->at("second.operation")), "EXP");
}

TEST(TestGraphPatternMatcher, SearchesEveryConsumerOfABoundTensor)
{
    // `$mid` has two consumers. The binary node must take the second one, and the unary
    // node must take the first, which the binary node already rejected.
    const std::string fanOut = R"([
        {"kind": "op", "id": "source", "op": "pointwise",
         "operands": {"in_0": "$a"}, "results": {"out_0": "$mid"}},
        {"kind": "op", "id": "binary", "op": "pointwise",
         "operands": {"in_0": "$mid", "in_1": "$b"}, "results": {"out_0": "$sum"}},
        {"kind": "op", "id": "unary", "op": "pointwise",
         "operands": {"in_0": "$mid"}, "results": {"out_0": "$exp"}}
    ])";
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3).tensor(4).tensor(5);
    graph.node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 2));
    graph.node(pointwiseNode(PointwiseMode::EXP, 2, std::nullopt, 4));
    graph.node(pointwiseNode(PointwiseMode::ADD, 2, 5, 3));

    const auto bound = match(fanOut, graph);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(std::get<std::string>(bound->at("binary.operation")), "ADD");
    EXPECT_EQ(tryGetBoundInt(*bound, "b"), 5);
    EXPECT_EQ(tryGetBoundInt(*bound, "sum"), 3);
    EXPECT_EQ(std::get<std::string>(bound->at("unary.operation")), "EXP");
    EXPECT_EQ(tryGetBoundInt(*bound, "exp"), 4);
}

TEST(TestGraphPatternMatcher, AmbiguousGraphResolvesToFirstAssignmentEveryTime)
{
    const std::string twoIndependent = R"([
        {"kind": "op", "id": "left", "op": "pointwise",
         "operands": {"in_0": "$a"}, "results": {"out_0": "$b"}},
        {"kind": "op", "id": "right", "op": "pointwise",
         "operands": {"in_0": "$c"}, "results": {"out_0": "$d"}}
    ])";
    const auto compiled = compilePatternText(twoIndependent);
    PatternGraphBuilder builder;
    builder.tensor(1).tensor(2).tensor(3).tensor(4);
    builder.node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 2));
    builder.node(pointwiseNode(PointwiseMode::EXP, 3, std::nullopt, 4));
    const auto graph = builder.build();

    const auto first = matchGraphPattern(*compiled, graph.context());
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(tryGetBoundInt(*first, "a"), 1);
    EXPECT_EQ(tryGetBoundInt(*first, "c"), 3);
    EXPECT_EQ(std::get<std::string>(first->at("left.operation")), "ABS");
    for(int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(matchGraphPattern(*compiled, graph.context()), first);
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
