// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "unit_conv_solver.hpp"

#include "get_handle.hpp"

#if defined(MIOPEN_USE_HIPCONV) && MIOPEN_USE_HIPCONV

using Config   = miopen::solver::conv::PerformanceConfigConvHipConv;
using TestCase = miopen::unit_tests::ConvTestCase;

// A kernel label contains commas, which field-wise perf-config serialization would split on.
TEST(CPU_PerfConfig_ConvHipConv_NONE, SerializeRoundTrip)
{
    Config stored;
    stored.descriptor = "direct[tile_size_k=256,tile_size_n=1,tile_size_h=16,tile_size_w=16]";

    Config loaded;
    ASSERT_TRUE(loaded.Deserialize(stored.ToString()));
    EXPECT_EQ(loaded.descriptor, stored.descriptor);
}

// A perf-db record selects its config only under the hipconv minor version that wrote it.
TEST(GPU_PerfConfig_ConvHipConv_FP16, VersionStamp)
{
    if(!IsTestSupportedByDevice(Gpu::gfx950 | Gpu::gfx125X))
        GTEST_SKIP();

    constexpr auto type   = miopenHalf;
    constexpr auto layout = miopenTensorNHWC;
    // clang-format off
    const auto test_case = TestCase{{type, layout, {4, 64, 8, 1}}, {type, layout, {64, 4, 3, 3}}, type, {{1, 1}, {1, 1}, {1, 1}, 16}};
    // clang-format on
    const auto problem = test_case.GetProblemDescription(miopen::conv::Direction::Forward);

    auto&& handle = get_handle();
    auto ctx      = miopen::ExecutionContext{&handle};
    problem.SetupFloats(ctx);

    const auto solver = miopen::solver::conv::ConvHipConv{};
    ASSERT_TRUE(solver.IsApplicable(ctx, problem));

    const auto record = solver.GetDefaultPerformanceConfig(ctx, problem).descriptor;
    const auto colon  = record.find(':');
    ASSERT_NE(colon, std::string::npos) << record;
    EXPECT_NE(record.substr(0, colon), "unknown");
    const auto body = record.substr(colon + 1);

    const auto loads = [&](const std::string& stored) {
        Config config;
        config.Deserialize(stored);
        return solver.IsValidPerformanceConfig(ctx, problem, config);
    };
    EXPECT_TRUE(loads(record));
    EXPECT_FALSE(loads("v999.999:" + body));
    EXPECT_FALSE(loads("unknown:" + body));
    EXPECT_FALSE(loads(body));
}

#endif // MIOPEN_USE_HIPCONV
