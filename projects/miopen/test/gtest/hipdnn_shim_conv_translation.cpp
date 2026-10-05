// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Problems that test how the forwarded path translates shapes into a hipDNN graph: 3-D,
// per-axis geometry and strided tensors. Each must be served in both modes.

#include "hipdnn_shim_helpers.hpp"

using namespace hipdnn_shim_test;

namespace {

struct TranslationCase
{
    const char* name;
    std::vector<std::size_t> x_lengths;
    // Empty for packed.
    std::vector<std::size_t> x_strides;
    std::vector<std::size_t> w_lengths;
    ConvGeometry geometry;
    bool forward;
    bool backward_data;
    bool backward_weights;
};

// Keeps ctest names stable. Without it gtest prints the struct's bytes, which include pointers.
void PrintTo(const TranslationCase& c, std::ostream* os) { *os << c.name; }

// generate() ignores strides, so it would leave part of a strided tensor zero.
tensor<float> MakeFilled(const std::vector<std::size_t>& lengths,
                         const std::vector<std::size_t>& strides,
                         std::uint64_t max_value)
{
    tensor<float> t = strides.empty() ? tensor<float>{lengths} : tensor<float>{lengths, strides};
    const tensor_elem_gen_integer gen{max_value};
    t.for_each([&](auto... i) { t(i...) = static_cast<float>(gen(i...)); });
    return t;
}

void CheckForward(tensor<float>& x, tensor<float>& w, const ConvGeometry& geometry)
{
    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, geometry));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<float> y{out_lengths};

    auto x_dev = handle_deref.Write(x.data);
    auto w_dev = handle_deref.Write(w.data);
    auto y_dev = handle_deref.Write(y.data);

    std::size_t workspace_size = 0;
    ASSERT_EQ(miopenConvolutionForwardGetWorkSpaceSize(
                  handle, &w.desc, &x.desc, conv.handle, &y.desc, &workspace_size),
              miopenStatusSuccess);
    Workspace wspace{workspace_size};

    int returned_algo_count = 0;
    miopenConvAlgoPerf_t perf{};
    ASSERT_EQ(miopenFindConvolutionForwardAlgorithm(handle,
                                                    &x.desc,
                                                    x_dev.get(),
                                                    &w.desc,
                                                    w_dev.get(),
                                                    conv.handle,
                                                    &y.desc,
                                                    y_dev.get(),
                                                    1,
                                                    &returned_algo_count,
                                                    &perf,
                                                    wspace.ptr(),
                                                    wspace.size(),
                                                    false),
              miopenStatusSuccess);
    ASSERT_GT(returned_algo_count, 0);

    ASSERT_EQ(miopenConvolutionForward(handle,
                                       &kOne,
                                       &x.desc,
                                       x_dev.get(),
                                       &w.desc,
                                       w_dev.get(),
                                       conv.handle,
                                       perf.fwd_algo,
                                       &kZero,
                                       &y.desc,
                                       y_dev.get(),
                                       wspace.ptr(),
                                       wspace.size()),
              miopenStatusSuccess);

    y.data = handle_deref.Read<float>(y_dev, y.data.size());
    CheckMatchesCpuReference(x, w, y, geometry);
}

// dx takes x's descriptor, so a strided x also tests a strided dx.
void CheckBackwardData(tensor<float>& x, tensor<float>& w, const ConvGeometry& geometry)
{
    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, geometry));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<float> dy = MakeFilled(out_lengths, {}, 13);
    tensor<float> dx{x.desc};

    auto dy_dev = handle_deref.Write(dy.data);
    auto w_dev  = handle_deref.Write(w.data);
    auto dx_dev = handle_deref.Write(dx.data);

    std::size_t workspace_size = 0;
    ASSERT_EQ(miopenConvolutionBackwardDataGetWorkSpaceSize(
                  handle, &dy.desc, &w.desc, conv.handle, &dx.desc, &workspace_size),
              miopenStatusSuccess);
    Workspace wspace{workspace_size};

    int returned_algo_count = 0;
    miopenConvAlgoPerf_t perf{};
    ASSERT_EQ(miopenFindConvolutionBackwardDataAlgorithm(handle,
                                                         &dy.desc,
                                                         dy_dev.get(),
                                                         &w.desc,
                                                         w_dev.get(),
                                                         conv.handle,
                                                         &dx.desc,
                                                         dx_dev.get(),
                                                         1,
                                                         &returned_algo_count,
                                                         &perf,
                                                         wspace.ptr(),
                                                         wspace.size(),
                                                         false),
              miopenStatusSuccess);
    ASSERT_GT(returned_algo_count, 0);

    ASSERT_EQ(miopenConvolutionBackwardData(handle,
                                            &kOne,
                                            &dy.desc,
                                            dy_dev.get(),
                                            &w.desc,
                                            w_dev.get(),
                                            conv.handle,
                                            perf.bwd_data_algo,
                                            &kZero,
                                            &dx.desc,
                                            dx_dev.get(),
                                            wspace.ptr(),
                                            wspace.size()),
              miopenStatusSuccess);

    dx.data = handle_deref.Read<float>(dx_dev, dx.data.size());
    CheckMatchesCpuBackwardData(dx, w, dy, geometry);
}

void CheckBackwardWeights(tensor<float>& x, tensor<float>& w, const ConvGeometry& geometry)
{
    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, geometry));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<float> dy = MakeFilled(out_lengths, {}, 13);
    tensor<float> dw{w.desc.GetLengths()};

    auto dy_dev = handle_deref.Write(dy.data);
    auto x_dev  = handle_deref.Write(x.data);
    auto dw_dev = handle_deref.Write(dw.data);

    std::size_t workspace_size = 0;
    ASSERT_EQ(miopenConvolutionBackwardWeightsGetWorkSpaceSize(
                  handle, &dy.desc, &x.desc, conv.handle, &dw.desc, &workspace_size),
              miopenStatusSuccess);
    Workspace wspace{workspace_size};

    int returned_algo_count = 0;
    miopenConvAlgoPerf_t perf{};
    ASSERT_EQ(miopenFindConvolutionBackwardWeightsAlgorithm(handle,
                                                            &dy.desc,
                                                            dy_dev.get(),
                                                            &x.desc,
                                                            x_dev.get(),
                                                            conv.handle,
                                                            &dw.desc,
                                                            dw_dev.get(),
                                                            1,
                                                            &returned_algo_count,
                                                            &perf,
                                                            wspace.ptr(),
                                                            wspace.size(),
                                                            false),
              miopenStatusSuccess);
    ASSERT_GT(returned_algo_count, 0);

    ASSERT_EQ(miopenConvolutionBackwardWeights(handle,
                                               &kOne,
                                               &dy.desc,
                                               dy_dev.get(),
                                               &x.desc,
                                               x_dev.get(),
                                               conv.handle,
                                               perf.bwd_weights_algo,
                                               &kZero,
                                               &dw.desc,
                                               dw_dev.get(),
                                               wspace.ptr(),
                                               wspace.size()),
              miopenStatusSuccess);

    dw.data = handle_deref.Read<float>(dw_dev, dw.data.size());
    CheckMatchesCpuBackwardWeights(x, dw, dy, geometry);
}

std::vector<TranslationCase> TranslationCases()
{
    return {
        {"Ncdhw",
         {1, 4, 6, 8, 8},
         {},
         {4, 4, 3, 3, 3},
         {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
         true,
         true,
         true},
        // The problem from conv_api_strided_tensors.cpp, which checks only the status.
        {"NcdhwStrided",
         {4, 4, 16, 9, 16},
         {10240, 2560, 160, 16, 1},
         {8, 4, 3, 3, 3},
         {{1, 0, 1}, {2, 2, 2}, {1, 1, 1}},
         true,
         false,
         false},
        // Each axis differs, so mixing up stride and dilation or the axis order changes the
        // output.
        {"PerAxisGeometry",
         {2, 4, 9, 13},
         {},
         {4, 4, 3, 5},
         {{1, 2}, {2, 1}, {1, 2}},
         true,
         true,
         true},
        // No stride is 1, yet the layout is plain NCHW, not vectorized.
        {"InnermostStride2", {2, 4, 8, 8}, {512, 128, 16, 2}, {4, 4, 3, 3}, {}, true, true, true},
    };
}

} // namespace

class GPU_HipdnnShimConvTranslation_FP32 : public ::testing::TestWithParam<TranslationCase>
{
};

TEST_P(GPU_HipdnnShimConvTranslation_FP32, MatchesCpuReference)
{
    const TranslationCase& config = GetParam();
    tensor<float> x               = MakeFilled(config.x_lengths, config.x_strides, 17);
    tensor<float> w               = MakeFilled(config.w_lengths, {}, 17);

    if(config.forward)
    {
        SCOPED_TRACE("forward");
        ASSERT_NO_FATAL_FAILURE(CheckForward(x, w, config.geometry));
    }
    if(config.backward_data)
    {
        SCOPED_TRACE("backward data");
        ASSERT_NO_FATAL_FAILURE(CheckBackwardData(x, w, config.geometry));
    }
    if(config.backward_weights)
    {
        SCOPED_TRACE("backward weights");
        ASSERT_NO_FATAL_FAILURE(CheckBackwardWeights(x, w, config.geometry));
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_HipdnnShimConvTranslation_FP32,
                         ::testing::ValuesIn(TranslationCases()),
                         [](const ::testing::TestParamInfo<TranslationCase>& info) {
                             return std::string(info.param.name);
                         });

// Same tensor shapes, different padding and dilation. A plan cache keyed without those would
// reuse the first plan for the second problem.
TEST(GPU_HipdnnShimConvPlanKey_FP32, SameShapeDifferentPadsAndDilations)
{
    tensor<float> x = MakeFilled({2, 4, 8, 8}, {}, 17);
    tensor<float> w = MakeFilled({4, 4, 3, 3}, {}, 17);

    for(const ConvGeometry& geometry :
        {ConvGeometry{{1, 1}, {1, 1}, {1, 1}}, ConvGeometry{{2, 2}, {1, 1}, {2, 2}}})
    {
        SCOPED_TRACE("pad and dilation " + std::to_string(geometry.pads[0]));
        ASSERT_NO_FATAL_FAILURE(CheckForward(x, w, geometry));
        ASSERT_NO_FATAL_FAILURE(CheckBackwardData(x, w, geometry));
        ASSERT_NO_FATAL_FAILURE(CheckBackwardWeights(x, w, geometry));
    }
}
