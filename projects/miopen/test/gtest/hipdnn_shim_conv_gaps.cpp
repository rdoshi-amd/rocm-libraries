// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// One test per problem MIOpen serves but the forwarded path cannot express yet. Each records
// what each mode did, so every gap is one known-divergence entry in the parity run.
//
// Every call runs Find first, because natively grouped, depthwise and transposed calls fail
// without it. Find's status is not asserted: Find is never forwarded, and for some of these
// problems it fails in both modes.

#include "hipdnn_shim_helpers.hpp"

#include <algorithm>
#include <array>

using namespace hipdnn_shim_test;

namespace {

enum class Direction
{
    Forward,
    BackwardData,
    BackwardWeights,
};

constexpr std::array<Direction, 3> kAllDirections = {
    Direction::Forward, Direction::BackwardData, Direction::BackwardWeights};

std::string CaseName(const std::string& problem, Direction direction)
{
    switch(direction)
    {
    case Direction::Forward: return problem + "-fwd";
    case Direction::BackwardData: return problem + "-bwd-data";
    case Direction::BackwardWeights: return problem + "-bwd-weights";
    }
    return problem;
}

// Filled through data because tensor_elem_gen_integer does not take six dimensions.
tensor<float> MakeFilled(const std::vector<std::size_t>& lengths)
{
    tensor<float> t{lengths};
    for(auto& value : t.data)
        value = static_cast<float>(prng::gen_0_to_B(17));
    return t;
}

struct CallResult
{
    miopenStatus_t status;
    // Read straight after the call, because a later call that MIOpen serves clears it.
    std::string message;
};

// x, w and y hold the problem in forward terms, whatever the direction. The tensor the direction
// writes holds the result if the call succeeded.
CallResult FindAndRun(Direction direction,
                      miopenConvolutionDescriptor_t conv,
                      tensor<float>& x,
                      tensor<float>& w,
                      tensor<float>& y,
                      const float* alpha = &kOne,
                      const float* beta  = &kZero)
{
    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    auto x_dev = handle_deref.Write(x.data);
    auto w_dev = handle_deref.Write(w.data);
    auto y_dev = handle_deref.Write(y.data);

    std::size_t workspace_size = 0;
    int returned_algo_count    = 0;
    miopenConvAlgoPerf_t perf{};
    Workspace wspace;
    CallResult result{miopenStatusSuccess, {}};

    switch(direction)
    {
    case Direction::Forward:
        miopenConvolutionForwardGetWorkSpaceSize(
            handle, &w.desc, &x.desc, conv, &y.desc, &workspace_size);
        wspace.resize(workspace_size);
        miopenFindConvolutionForwardAlgorithm(handle,
                                              &x.desc,
                                              x_dev.get(),
                                              &w.desc,
                                              w_dev.get(),
                                              conv,
                                              &y.desc,
                                              y_dev.get(),
                                              1,
                                              &returned_algo_count,
                                              &perf,
                                              wspace.ptr(),
                                              wspace.size(),
                                              false);
        wspace.resize(std::max(workspace_size, perf.memory));
        ResetAfterFind(y_dev, y);
        result.status = miopenConvolutionForward(handle,
                                                 alpha,
                                                 &x.desc,
                                                 x_dev.get(),
                                                 &w.desc,
                                                 w_dev.get(),
                                                 conv,
                                                 perf.fwd_algo,
                                                 beta,
                                                 &y.desc,
                                                 y_dev.get(),
                                                 wspace.ptr(),
                                                 wspace.size());
        break;
    case Direction::BackwardData:
        miopenConvolutionBackwardDataGetWorkSpaceSize(
            handle, &y.desc, &w.desc, conv, &x.desc, &workspace_size);
        wspace.resize(workspace_size);
        miopenFindConvolutionBackwardDataAlgorithm(handle,
                                                   &y.desc,
                                                   y_dev.get(),
                                                   &w.desc,
                                                   w_dev.get(),
                                                   conv,
                                                   &x.desc,
                                                   x_dev.get(),
                                                   1,
                                                   &returned_algo_count,
                                                   &perf,
                                                   wspace.ptr(),
                                                   wspace.size(),
                                                   false);
        wspace.resize(std::max(workspace_size, perf.memory));
        ResetAfterFind(x_dev, x);
        result.status = miopenConvolutionBackwardData(handle,
                                                      alpha,
                                                      &y.desc,
                                                      y_dev.get(),
                                                      &w.desc,
                                                      w_dev.get(),
                                                      conv,
                                                      perf.bwd_data_algo,
                                                      beta,
                                                      &x.desc,
                                                      x_dev.get(),
                                                      wspace.ptr(),
                                                      wspace.size());
        break;
    case Direction::BackwardWeights:
        miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            handle, &y.desc, &x.desc, conv, &w.desc, &workspace_size);
        wspace.resize(workspace_size);
        miopenFindConvolutionBackwardWeightsAlgorithm(handle,
                                                      &y.desc,
                                                      y_dev.get(),
                                                      &x.desc,
                                                      x_dev.get(),
                                                      conv,
                                                      &w.desc,
                                                      w_dev.get(),
                                                      1,
                                                      &returned_algo_count,
                                                      &perf,
                                                      wspace.ptr(),
                                                      wspace.size(),
                                                      false);
        wspace.resize(std::max(workspace_size, perf.memory));
        ResetAfterFind(w_dev, w);
        result.status = miopenConvolutionBackwardWeights(handle,
                                                         alpha,
                                                         &y.desc,
                                                         y_dev.get(),
                                                         &x.desc,
                                                         x_dev.get(),
                                                         conv,
                                                         perf.bwd_weights_algo,
                                                         beta,
                                                         &w.desc,
                                                         w_dev.get(),
                                                         wspace.ptr(),
                                                         wspace.size());
        break;
    }
    result.message = miopenGetErrorString(result.status);

    if(result.status == miopenStatusSuccess)
    {
        switch(direction)
        {
        case Direction::Forward: y.data = handle_deref.Read<float>(y_dev, y.data.size()); break;
        case Direction::BackwardData:
            x.data = handle_deref.Read<float>(x_dev, x.data.size());
            break;
        case Direction::BackwardWeights:
            w.data = handle_deref.Read<float>(w_dev, w.data.size());
            break;
        }
    }
    return result;
}

// Rules out a silent fall back to MIOpen.
void ExpectForwardedDecline(const CallResult& result)
{
    if(ForwardingEnabled())
    {
        EXPECT_NE(result.message.find("[hipDNN-forwarded]"), std::string::npos)
            << "decline did not come from hipDNN: " << result.message;
    }
}

// Natively, 1-D and 4-D calls fail with UnknownError, so unknown_error_declines counts that as a
// decline in that mode only. Returns whether the case was served.
bool RecordOutcome(const std::string& case_name,
                   const CallResult& result,
                   bool unknown_error_declines = false)
{
    const bool served   = result.status == miopenStatusSuccess;
    const bool declined = result.status == miopenStatusUnsupportedOp ||
                          result.status == miopenStatusNotImplemented ||
                          (unknown_error_declines && !ForwardingEnabled() &&
                           result.status == miopenStatusUnknownError);
    EXPECT_TRUE(served || declined)
        << case_name << ": unexpected status " << result.status << ": " << result.message;
    if(declined)
        ExpectForwardedDecline(result);
    RecordServed(case_name, served);
    return served;
}

// Transposed mode swaps the roles of forward and backward data, as conv_common.hpp does.
void CheckAgainstCpu(Direction direction,
                     miopenConvolutionMode_t mode,
                     const tensor<float>& x,
                     const tensor<float>& w,
                     const tensor<float>& y,
                     const tensor<float>& got,
                     const ConvGeometry& geometry)
{
    const auto spatial    = geometry.pads.size();
    const bool transposed = mode == miopenTranspose;
    tensor<float> ref{got.desc};
    switch(direction)
    {
    case Direction::Forward:
        if(transposed)
            cpu_convolution_backward_data(spatial,
                                          ref,
                                          w,
                                          x,
                                          geometry.pads,
                                          geometry.strides,
                                          geometry.dilations,
                                          geometry.groups);
        else
            cpu_convolution_forward(spatial,
                                    x,
                                    w,
                                    ref,
                                    geometry.pads,
                                    geometry.strides,
                                    geometry.dilations,
                                    geometry.groups);
        break;
    case Direction::BackwardData:
        if(transposed)
            cpu_convolution_forward(spatial,
                                    y,
                                    w,
                                    ref,
                                    geometry.pads,
                                    geometry.strides,
                                    geometry.dilations,
                                    geometry.groups);
        else
            cpu_convolution_backward_data(spatial,
                                          ref,
                                          w,
                                          y,
                                          geometry.pads,
                                          geometry.strides,
                                          geometry.dilations,
                                          geometry.groups);
        break;
    case Direction::BackwardWeights:
        if(transposed)
            cpu_convolution_backward_weight(spatial,
                                            y,
                                            ref,
                                            x,
                                            geometry.pads,
                                            geometry.strides,
                                            geometry.dilations,
                                            geometry.groups);
        else
            cpu_convolution_backward_weight(spatial,
                                            x,
                                            ref,
                                            y,
                                            geometry.pads,
                                            geometry.strides,
                                            geometry.dilations,
                                            geometry.groups);
        break;
    }
    CheckWithinTolerance(ref, got, "result");
}

// y_lengths is passed in rather than queried, because MIOpen's output-size query fails for 1-D.
void RunDirections(const std::string& problem,
                   miopenConvolutionDescriptor_t conv,
                   miopenConvolutionMode_t mode,
                   const std::vector<std::size_t>& x_lengths,
                   const std::vector<std::size_t>& w_lengths,
                   const std::vector<std::size_t>& y_lengths,
                   const ConvGeometry& geometry,
                   bool unknown_error_declines = false)
{
    const tensor<float> x = MakeFilled(x_lengths);
    const tensor<float> w = MakeFilled(w_lengths);
    const tensor<float> y = MakeFilled(y_lengths);

    for(const auto direction : kAllDirections)
    {
        const std::string case_name = CaseName(problem, direction);
        SCOPED_TRACE(case_name);

        // Zeroed, so a kernel that writes nothing cannot pass on what was there before.
        tensor<float> x_arg = direction == Direction::BackwardData ? tensor<float>{x.desc} : x;
        tensor<float> w_arg = direction == Direction::BackwardWeights ? tensor<float>{w.desc} : w;
        tensor<float> y_arg = direction == Direction::Forward ? tensor<float>{y.desc} : y;

        const auto result = FindAndRun(direction, conv, x_arg, w_arg, y_arg);
        if(!RecordOutcome(case_name, result, unknown_error_declines))
            continue;

        const tensor<float>& got = direction == Direction::Forward        ? y_arg
                                   : direction == Direction::BackwardData ? x_arg
                                                                          : w_arg;
        CheckAgainstCpu(direction, mode, x, w, y, got, geometry);
    }
}

void RunQueriedDirections(const std::string& problem,
                          miopenConvolutionMode_t mode,
                          const std::vector<std::size_t>& x_lengths,
                          const std::vector<std::size_t>& w_lengths,
                          const ConvGeometry& geometry,
                          const std::vector<int>& output_pads = {})
{
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, mode, geometry));
    if(!output_pads.empty())
    {
        ASSERT_EQ(miopenSetTransposeConvNdOutputPadding(
                      conv.handle, static_cast<int>(output_pads.size()), output_pads.data()),
                  miopenStatusSuccess);
    }

    tensor<float> x{x_lengths};
    tensor<float> w{w_lengths};
    std::vector<std::size_t> y_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, y_lengths));

    RunDirections(problem, conv.handle, mode, x_lengths, w_lengths, y_lengths, geometry);
}

} // namespace

// Gap: the forwarded path declines a group count other than 1.
TEST(GPU_HipdnnShimConvGrouped_FP32, MatchesCpuReferenceWhenServed)
{
    ConvGeometry grouped;
    grouped.groups = 2;
    ASSERT_NO_FATAL_FAILURE(
        RunQueriedDirections("grouped", miopenConvolution, {2, 4, 8, 8}, {4, 2, 3, 3}, grouped));

    ConvGeometry depthwise;
    depthwise.groups = 4;
    ASSERT_NO_FATAL_FAILURE(RunQueriedDirections(
        "depthwise", miopenConvolution, {2, 4, 8, 8}, {4, 1, 3, 3}, depthwise));
}

// Gap: the forwarded path declines transposed convolution. Output padding needs a stride above
// 1 to add anything.
TEST(GPU_HipdnnShimConvTransposed_FP32, MatchesCpuReferenceWhenServed)
{
    ASSERT_NO_FATAL_FAILURE(
        RunQueriedDirections("transposed", miopenTranspose, {2, 4, 8, 8}, {4, 4, 3, 3}, {}));

    const ConvGeometry strided{{1, 1}, {2, 2}, {1, 1}};
    ASSERT_NO_FATAL_FAILURE(RunQueriedDirections(
        "transposed-output-padding", miopenTranspose, {2, 4, 8, 8}, {4, 4, 3, 3}, strided, {1, 1}));
}

// Not a gap: MIOpen fails every 1-D and 4-D call and the forwarded path declines them. A pad of
// 1 around a 3-wide filter keeps each length.
TEST(GPU_HipdnnShimConvRank_FP32, MatchesCpuReferenceWhenServed)
{
    {
        const ConvGeometry geometry{{1}, {1}, {1}};
        OwnedConvDescriptor conv;
        ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, geometry));
        RunDirections(
            "1d", conv.handle, miopenConvolution, {2, 4, 8}, {4, 4, 3}, {2, 4, 8}, geometry, true);
    }
    {
        const ConvGeometry geometry{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}};
        OwnedConvDescriptor conv;
        ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, geometry));
        RunDirections("4d",
                      conv.handle,
                      miopenConvolution,
                      {1, 2, 4, 4, 4, 4},
                      {2, 2, 3, 3, 3, 3},
                      {1, 2, 4, 4, 4, 4},
                      geometry,
                      true);
    }
}

// Gap: neither mode serves a non-default alpha or beta, so this records the status. Natively 3-D
// returns UnknownError, where 2-D and the forwarded path return NotImplemented.
TEST(GPU_HipdnnShimConvAlphaBeta_FP32, StatusMatchesMiopen)
{
    struct Problem
    {
        const char* name;
        std::vector<std::size_t> x_lengths;
        std::vector<std::size_t> w_lengths;
        ConvGeometry geometry;
    };
    const std::array<Problem, 2> problems = {{
        {"2d", {2, 4, 8, 8}, {4, 4, 3, 3}, {}},
        {"3d", {1, 4, 6, 6, 6}, {4, 4, 3, 3, 3}, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}},
    }};
    struct Scale
    {
        const char* name;
        float alpha;
        float beta;
    };
    const std::array<Scale, 2> scales = {{{"alpha", 2.0f, 0.0f}, {"beta", 1.0f, 1.0f}}};

    for(const auto& problem : problems)
    {
        OwnedConvDescriptor conv;
        ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, problem.geometry));
        tensor<float> x = MakeFilled(problem.x_lengths);
        tensor<float> w = MakeFilled(problem.w_lengths);
        std::vector<std::size_t> y_lengths;
        ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, y_lengths));

        for(const auto& scale : scales)
        {
            const std::string case_name = std::string(problem.name) + "-" + scale.name;
            SCOPED_TRACE(case_name);

            // Filled so that beta has something to scale.
            const tensor<float> y_before = MakeFilled(y_lengths);
            tensor<float> y              = y_before;
            const auto result =
                FindAndRun(Direction::Forward, conv.handle, x, w, y, &scale.alpha, &scale.beta);
            RecordStatus(case_name, result.status);
            if(result.status != miopenStatusSuccess)
            {
                ExpectForwardedDecline(result);
                continue;
            }

            tensor<float> ref{y.desc};
            cpu_convolution_forward(problem.geometry.pads.size(),
                                    x,
                                    w,
                                    ref,
                                    problem.geometry.pads,
                                    problem.geometry.strides,
                                    problem.geometry.dilations,
                                    problem.geometry.groups);
            for(std::size_t i = 0; i < ref.data.size(); ++i)
                ref.data[i] = scale.alpha * ref.data[i] + scale.beta * y_before.data[i];
            CheckWithinTolerance(ref, y, "scaled result");
        }
    }
}
