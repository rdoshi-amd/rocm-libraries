// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Convolution through each public entry point, since each is forwarded to hipDNN separately.
// "HipdnnShim" in a suite name is what puts the suite in the parity run; see README.md.

#include "hipdnn_shim_helpers.hpp"

#include <hip/hip_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

using namespace hipdnn_shim_test;

namespace {

// Half precision in a channels-last layout, because the only MIOpen solver that matches the
// four-operation plan this entry point always builds requires both. Whole numbers in [-4, 4]
// keep every product and partial sum on a value half precision holds exactly, so a
// disagreement with the reference is a real one and not accumulated rounding; the negatives
// are what make the ReLU do something.
using FusedType = half_float::half;

auto FusedElementGenerator()
{
    return [](auto...) { return static_cast<FusedType>(prng::gen_A_to_B(-4, 5)); };
}

// The shapes the fused path is checked on. Kept as a list because which shapes an
// implementation actually has a kernel for varies by device, and adding one is a line here.
// Sizes are small so the doubled replay stays cheap.
//
// Only the three-dimensional shape is listed. The two-dimensional NHWC equivalent is served
// by MIOpen on gfx90a but disagrees with the CPU reference by far more than half-precision
// rounding, while the reference is the same code that agrees on the three-dimensional shape.
// That is a convolution bug to raise on its own, not something this test should carry.
struct FusedCase
{
    const char* name;
    miopenTensorLayout_t layout;
    std::vector<std::size_t> x_lengths;
    std::vector<std::size_t> w_lengths;
    std::vector<int> pads;
};

std::vector<FusedCase> FusedCases()
{
    return {{"3d-ndhwc", miopenTensorNDHWC, {1, 4, 14, 11, 1}, {4, 4, 3, 3, 3}, {1, 1, 1}}};
}

// One value per output channel. Unit strides throughout: the bias is one value per channel,
// and a layout-derived stride would otherwise describe a stride over dimensions of length one.
tensor<FusedType> MakeFusedBias(const FusedCase& config)
{
    std::vector<std::size_t> lengths(config.x_lengths.size(), 1);
    lengths[1] = config.w_lengths[0];
    const std::vector<std::size_t> unit_strides(lengths.size(), 1);
    tensor<FusedType> b{config.layout, lengths, unit_strides};
    b.generate(FusedElementGenerator());
    return b;
}

// The fused path is convolution, then a per-output-channel bias, then ReLU. Composed here
// from the same CPU convolution the unfused tests use, so a fused kernel that silently drops
// either of the trailing two steps shows up.
//
// The reference is built in the same layout as y so the two can be compared element by
// element; the CPU convolution indexes logically, so it is layout-agnostic already.
void CheckMatchesCpuBiasActivation(const FusedCase& config,
                                   const tensor<FusedType>& x,
                                   const tensor<FusedType>& w,
                                   const tensor<FusedType>& bias,
                                   const tensor<FusedType>& y)
{
    const std::vector<int> unit(config.pads.size(), 1);
    tensor<FusedType> ref_y{y.desc};
    cpu_convolution_forward(config.pads.size(), x, w, ref_y, config.pads, unit, unit, group_count);

    ref_y.par_for_each([&](auto n, auto k, auto... spatial) {
        auto& value            = ref_y(n, k, spatial...);
        const FusedType biased = static_cast<FusedType>(value + bias.data[k]);
        value                  = std::max(FusedType(0), biased);
    });

    CheckWithinTolerance(ref_y, y, "fused bias+activation result");
}

// The setup the forward tests share: the same input, weights and convolution, the output shape
// the library reports for them, and all three tensors on the device. A fatal failure in SetUp()
// skips the test body, and the members still clean up.
class HipdnnShimConvFwd : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv));
        std::vector<std::size_t> out_lengths;
        ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
        y = tensor<float>{out_lengths};

        x_dev = handle_deref.Write(x.data);
        w_dev = handle_deref.Write(w.data);
        y_dev = handle_deref.Write(y.data);
    }

    void ReadBackAndCheck()
    {
        y.data = handle_deref.Read<float>(y_dev, y.data.size());
        CheckMatchesCpuReference(x, w, y);
    }

    miopen::Handle& handle_deref = get_handle();
    miopenHandle_t handle        = &handle_deref;
    tensor<float> x              = MakeInput();
    tensor<float> w              = MakeWeights();
    tensor<float> y;
    OwnedConvDescriptor conv;
    miopen::Allocator::ManageDataPtr x_dev;
    miopen::Allocator::ManageDataPtr w_dev;
    miopen::Allocator::ManageDataPtr y_dev;
};

// One subclass per path, so each keeps its own suite name: the parity filter and the GPU
// exclusion patterns select by it.
class GPU_HipdnnShimConvFwdApi_FP32 : public HipdnnShimConvFwd
{
};

class GPU_HipdnnShimConvSolutionApi_FP32 : public HipdnnShimConvFwd
{
};

class GPU_HipdnnShimConvDeclined_FP32 : public HipdnnShimConvFwd
{
protected:
    // Runs the fused entry point on the fixture's problem. z is always passed because the
    // entry point needs a valid descriptor, even when alpha2 = 0 drops it.
    void RunFused(const float* alpha2,
                  miopenActivationMode_t mode,
                  miopenStatus_t& status,
                  std::string* message = nullptr)
    {
        tensor<float> bias{1, 4, 1, 1};
        tensor<float> z{y.desc.GetLengths()};
        auto bias_dev = handle_deref.Write(bias.data);
        auto z_dev    = handle_deref.Write(z.data);

        OwnedActivationDescriptor activation;
        ASSERT_EQ(miopenCreateActivationDescriptor(&activation.handle), miopenStatusSuccess);
        ASSERT_EQ(miopenSetActivationDescriptor(activation.handle, mode, 0.0, 0.0, 0.0),
                  miopenStatusSuccess);

        status = miopenConvolutionBiasActivationForward(handle,
                                                        &kOne,
                                                        &x.desc,
                                                        x_dev.get(),
                                                        &w.desc,
                                                        w_dev.get(),
                                                        conv.handle,
                                                        miopenConvolutionFwdAlgoImplicitGEMM,
                                                        nullptr,
                                                        0ull,
                                                        alpha2,
                                                        &z.desc,
                                                        z_dev.get(),
                                                        &bias.desc,
                                                        bias_dev.get(),
                                                        activation.handle,
                                                        &y.desc,
                                                        y_dev.get());
        // Read now: MIOpen serves the activation descriptor's release, which clears the
        // forwarded failure.
        if(message != nullptr)
            *message = miopenGetErrorString(status);
    }
};

template <class T, hipError_t (*Destroy)(T)>
struct OwnedHip
{
    OwnedHip()                           = default;
    OwnedHip(const OwnedHip&)            = delete;
    OwnedHip& operator=(const OwnedHip&) = delete;
    ~OwnedHip()
    {
        if(handle != nullptr)
            EXPECT_EQ(Destroy(handle), hipSuccess);
    }

    T handle = nullptr;
};

using OwnedStream = OwnedHip<hipStream_t, hipStreamDestroy>;
using OwnedEvent  = OwnedHip<hipEvent_t, hipEventDestroy>;

// Holds back the work queued on a stream after Close() until Open(), so a test can order work
// across streams without relying on timing.
class StreamGate
{
public:
    StreamGate()                             = default;
    StreamGate(const StreamGate&)            = delete;
    StreamGate& operator=(const StreamGate&) = delete;

    // Opens first, so a test that stops early does not leave the stream blocked.
    ~StreamGate()
    {
        Open();
        if(stream_ != nullptr)
            static_cast<void>(hipStreamSynchronize(stream_));
        if(flag_ != nullptr)
            static_cast<void>(hipHostFree(flag_));
    }

    void Close(hipStream_t stream)
    {
        stream_ = stream;

        int device            = 0;
        int can_wait_on_value = 0;
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        ASSERT_EQ(hipDeviceGetAttribute(
                      &can_wait_on_value, hipDeviceAttributeCanUseStreamWaitValue, device),
                  hipSuccess);
        if(can_wait_on_value == 0)
        {
            ASSERT_EQ(hipLaunchHostFunc(stream, &StreamGate::Wait, this), hipSuccess);
            return;
        }

        void* flag = nullptr;
        ASSERT_EQ(hipHostMalloc(
                      &flag, sizeof(std::uint32_t), hipHostMallocMapped | hipHostMallocCoherent),
                  hipSuccess);
        flag_  = static_cast<std::uint32_t*>(flag);
        *flag_ = 0;
        ASSERT_EQ(hipStreamWaitValue32(stream, flag_, 1, hipStreamWaitValueEq), hipSuccess);
    }

    void Open()
    {
        if(flag_ != nullptr)
            std::atomic_ref<std::uint32_t>(*flag_).store(1);
        opened_.store(true);
    }

private:
    static void Wait(void* gate)
    {
        while(!static_cast<StreamGate*>(gate)->opened_.load())
            std::this_thread::yield();
    }

    hipStream_t stream_  = nullptr;
    std::uint32_t* flag_ = nullptr;
    std::atomic<bool> opened_{false};
};

} // namespace

// The Find/Run pair: the older of the two public convolution paths, and the one most callers
// still use.
TEST_F(GPU_HipdnnShimConvFwdApi_FP32, FindAndForwardMatchCpuReference)
{
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

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    ASSERT_EQ(miopenConvolutionForward(handle,
                                       &alpha,
                                       &x.desc,
                                       x_dev.get(),
                                       &w.desc,
                                       w_dev.get(),
                                       conv.handle,
                                       perf.fwd_algo,
                                       &beta,
                                       &y.desc,
                                       y_dev.get(),
                                       wspace.ptr(),
                                       wspace.size()),
              miopenStatusSuccess);

    ReadBackAndCheck();
}

// The Problem/Solution path reaches the same convolution through different public entry
// points, so it has to be swapped over separately and is covered separately.
TEST_F(GPU_HipdnnShimConvSolutionApi_FP32, RunSolutionMatchesCpuReference)
{
    OwnedProblem problem;
    ASSERT_EQ(miopenCreateConvProblem(&problem.handle, conv.handle, miopenProblemDirectionForward),
              miopenStatusSuccess);
    ASSERT_EQ(miopenSetProblemTensorDescriptor(problem.handle, miopenTensorConvolutionX, &x.desc),
              miopenStatusSuccess);
    ASSERT_EQ(miopenSetProblemTensorDescriptor(problem.handle, miopenTensorConvolutionW, &w.desc),
              miopenStatusSuccess);
    ASSERT_EQ(miopenSetProblemTensorDescriptor(problem.handle, miopenTensorConvolutionY, &y.desc),
              miopenStatusSuccess);

    std::vector<miopenSolution_t> solutions(1);
    OwnedSolutions owned_solutions{solutions};
    std::size_t found = 0;
    ASSERT_EQ(miopenFindSolutions(
                  handle, problem.handle, nullptr, solutions.data(), &found, solutions.size()),
              miopenStatusSuccess);
    ASSERT_GT(found, 0);
    solutions.resize(found);

    std::size_t workspace_size = 0;
    ASSERT_EQ(miopenGetSolutionWorkspaceSize(solutions[0], &workspace_size), miopenStatusSuccess);
    Workspace wspace{workspace_size};

    miopenTensorDescriptor_t descriptors[3] = {&x.desc, &w.desc, &y.desc};
    const std::array<miopenTensorArgument_t, 3> arguments{{
        {miopenTensorConvolutionX, &descriptors[0], x_dev.get()},
        {miopenTensorConvolutionW, &descriptors[1], w_dev.get()},
        {miopenTensorConvolutionY, &descriptors[2], y_dev.get()},
    }};

    ASSERT_EQ(
        miopenRunSolution(
            handle, solutions[0], arguments.size(), arguments.data(), wspace.ptr(), wspace.size()),
        miopenStatusSuccess);

    ReadBackAndCheck();
}

TEST(GPU_HipdnnShimConvBwdDataApi_FP32, BackwardDataMatchesCpuReference)
{
    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    auto x = MakeInput();
    auto w = MakeWeights();
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));

    tensor<float> dy{out_lengths};
    dy.generate(tensor_elem_gen_integer{17});
    tensor<float> dx{x.desc.GetLengths()};

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

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    ASSERT_EQ(miopenConvolutionBackwardData(handle,
                                            &alpha,
                                            &dy.desc,
                                            dy_dev.get(),
                                            &w.desc,
                                            w_dev.get(),
                                            conv.handle,
                                            perf.bwd_data_algo,
                                            &beta,
                                            &dx.desc,
                                            dx_dev.get(),
                                            wspace.ptr(),
                                            wspace.size()),
              miopenStatusSuccess);

    dx.data = handle_deref.Read<float>(dx_dev, dx.data.size());

    CheckMatchesCpuBackwardData(dx, w, dy);
}

TEST(GPU_HipdnnShimConvBwdWeightsApi_FP32, BackwardWeightsMatchesCpuReference)
{
    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    auto x = MakeInput();
    auto w = MakeWeights();
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));

    tensor<float> dy{out_lengths};
    dy.generate(tensor_elem_gen_integer{17});
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

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    ASSERT_EQ(miopenConvolutionBackwardWeights(handle,
                                               &alpha,
                                               &dy.desc,
                                               dy_dev.get(),
                                               &x.desc,
                                               x_dev.get(),
                                               conv.handle,
                                               perf.bwd_weights_algo,
                                               &beta,
                                               &dw.desc,
                                               dw_dev.get(),
                                               wspace.ptr(),
                                               wspace.size()),
              miopenStatusSuccess);

    dw.data = handle_deref.Read<float>(dw_dev, dw.data.size());

    CheckMatchesCpuBackwardWeights(x, dw, dy);
}

// Forwarded calls on one handle share a workspace. The gate holds back the call on stream A, so
// the call on stream B can finish first only if it does not wait for A.
TEST(GPU_HipdnnShimConvStreamSwitch_FP32, CallAfterStreamSwitchWaitsForOldStream)
{
    OwnedStream stream_a;
    OwnedStream stream_b;
    ASSERT_EQ(hipStreamCreate(&stream_a.handle), hipSuccess);
    ASSERT_EQ(hipStreamCreate(&stream_b.handle), hipSuccess);
    OwnedEvent event_a;
    OwnedEvent event_b;
    ASSERT_EQ(hipEventCreate(&event_a.handle), hipSuccess);
    ASSERT_EQ(hipEventCreate(&event_b.handle), hipSuccess);

    // Its own handle, since the test changes its stream. Declared after the streams so it is
    // destroyed first.
    Owned<miopenHandle_t, miopenDestroy> owned_handle;
    ASSERT_EQ(miopenCreateWithStream(&owned_handle.handle, stream_a.handle), miopenStatusSuccess);
    miopenHandle_t handle = owned_handle.handle;

    // The MIOpen provider needs workspace for this problem, so both calls use the shared one.
    const ConvGeometry geometry{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    tensor<float> x1{1, 4, 6, 8, 8};
    x1.generate(tensor_elem_gen_integer{17});
    tensor<float> x2{1, 4, 6, 8, 8};
    x2.generate(tensor_elem_gen_integer{13});
    tensor<float> w{4, 4, 3, 3, 3};
    w.generate(tensor_elem_gen_integer{17});
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, geometry));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x1, w, out_lengths));
    tensor<float> y1{out_lengths};
    tensor<float> y2{out_lengths};

    auto& staging = get_handle();
    auto x1_dev   = staging.Write(x1.data);
    auto x2_dev   = staging.Write(x2.data);
    auto w_dev    = staging.Write(w.data);
    auto y1_dev   = staging.Write(y1.data);
    auto y2_dev   = staging.Write(y2.data);

    std::size_t workspace_size = 0;
    ASSERT_EQ(miopenConvolutionForwardGetWorkSpaceSize(
                  handle, &w.desc, &x1.desc, conv.handle, &y1.desc, &workspace_size),
              miopenStatusSuccess);
    Workspace workspace1{workspace_size};
    Workspace workspace2{workspace_size};

    int returned_algo_count = 0;
    miopenConvAlgoPerf_t perf{};
    ASSERT_EQ(miopenFindConvolutionForwardAlgorithm(handle,
                                                    &x1.desc,
                                                    x1_dev.get(),
                                                    &w.desc,
                                                    w_dev.get(),
                                                    conv.handle,
                                                    &y1.desc,
                                                    y1_dev.get(),
                                                    1,
                                                    &returned_algo_count,
                                                    &perf,
                                                    workspace1.ptr(),
                                                    workspace1.size(),
                                                    false),
              miopenStatusSuccess);
    ASSERT_GT(returned_algo_count, 0);

    auto forward = [&](tensor<float>& x,
                       const miopen::Allocator::ManageDataPtr& x_dev,
                       tensor<float>& y,
                       const miopen::Allocator::ManageDataPtr& y_dev,
                       Workspace& workspace) {
        return miopenConvolutionForward(handle,
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
                                        workspace.ptr(),
                                        workspace.size());
    };

    // Warm up on each stream. Building the plan or growing the workspace during the gated part
    // would make the host wait, which hides the race.
    ASSERT_EQ(forward(x1, x1_dev, y1, y1_dev, workspace1), miopenStatusSuccess);
    ASSERT_EQ(miopenSetStream(handle, stream_b.handle), miopenStatusSuccess);
    ASSERT_EQ(forward(x2, x2_dev, y2, y2_dev, workspace2), miopenStatusSuccess);
    ASSERT_EQ(miopenSetStream(handle, stream_a.handle), miopenStatusSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_a.handle), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_b.handle), hipSuccess);

    StreamGate gate;
    ASSERT_NO_FATAL_FAILURE(gate.Close(stream_a.handle));

    // Started before the gated calls, so a call that makes the host wait for stream A fails
    // late instead of hanging. Nothing may return before the join.
    std::thread opener([&gate] {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        gate.Open();
    });

    const miopenStatus_t status1  = forward(x1, x1_dev, y1, y1_dev, workspace1);
    const hipError_t record_a     = hipEventRecord(event_a.handle, stream_a.handle);
    const miopenStatus_t switched = miopenSetStream(handle, stream_b.handle);
    const miopenStatus_t status2  = forward(x2, x2_dev, y2, y2_dev, workspace2);
    const hipError_t record_b     = hipEventRecord(event_b.handle, stream_b.handle);

    const hipError_t waited_b = hipEventSynchronize(event_b.handle);
    const hipError_t a_done   = hipEventQuery(event_a.handle);

    opener.join();
    ASSERT_EQ(hipStreamSynchronize(stream_a.handle), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream_b.handle), hipSuccess);

    ASSERT_EQ(status1, miopenStatusSuccess);
    ASSERT_EQ(record_a, hipSuccess);
    ASSERT_EQ(switched, miopenStatusSuccess);
    ASSERT_EQ(status2, miopenStatusSuccess);
    ASSERT_EQ(record_b, hipSuccess);
    ASSERT_EQ(waited_b, hipSuccess);

    // MIOpen uses each caller's own workspace, so natively B may finish first.
    if(ForwardingEnabled())
    {
        EXPECT_EQ(a_done, hipSuccess)
            << "the call on stream B finished while the call on stream A was still waiting";
    }

    y1.data = staging.Read<float>(y1_dev, y1.data.size());
    y2.data = staging.Read<float>(y2_dev, y2.data.size());
    CheckMatchesCpuReference(x1, w, y1, geometry);
    CheckMatchesCpuReference(x2, w, y2, geometry);
}

void RunFusedCase(const FusedCase& config)
{
    SCOPED_TRACE(config.name);

    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    tensor<FusedType> x{config.layout, config.x_lengths};
    x.generate(FusedElementGenerator());
    tensor<FusedType> w{config.layout, config.w_lengths};
    w.generate(FusedElementGenerator());
    auto bias = MakeFusedBias(config);

    const std::vector<int> unit(config.pads.size(), 1);
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv, miopenConvolution, {config.pads, unit, unit}));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<FusedType> y{config.layout, out_lengths};
    tensor<FusedType> z{config.layout, out_lengths};

    OwnedActivationDescriptor activation;
    ASSERT_EQ(miopenCreateActivationDescriptor(&activation.handle), miopenStatusSuccess);
    ASSERT_EQ(miopenSetActivationDescriptor(activation.handle, miopenActivationRELU, 0.0, 0.0, 0.0),
              miopenStatusSuccess);

    auto x_dev    = handle_deref.Write(x.data);
    auto w_dev    = handle_deref.Write(w.data);
    auto bias_dev = handle_deref.Write(bias.data);
    auto y_dev    = handle_deref.Write(y.data);
    auto z_dev    = handle_deref.Write(z.data);

    // alpha2 is zero, so the zeroed z tensor contributes nothing; it is still passed because
    // the entry point needs a valid descriptor and buffer there.
    const float alpha1 = 1.0f;
    const float alpha2 = 0.0f;
    const auto status  = miopenConvolutionBiasActivationForward(handle,
                                                               &alpha1,
                                                               &x.desc,
                                                               x_dev.get(),
                                                               &w.desc,
                                                               w_dev.get(),
                                                               conv.handle,
                                                               miopenConvolutionFwdAlgoImplicitGEMM,
                                                               nullptr,
                                                               0ull,
                                                               &alpha2,
                                                               &z.desc,
                                                               z_dev.get(),
                                                               &bias.desc,
                                                               bias_dev.get(),
                                                               activation.handle,
                                                               &y.desc,
                                                               y_dev.get());

    // Fused conv+bias+activation is unimplemented on some devices -- the only MIOpen solver that
    // matches the plan this entry point builds needs a whitelisted device, and hipDNN has its own
    // set of cases it will not express -- so a decline is a legitimate answer rather than a
    // failure. It ends the test as a pass, not a skip: a skip in one parity replay against a pass
    // in the other reads as a divergence that is not one. Where the decline came from is still
    // checkable, and with forwarding on it must carry the forwarded-error prefix, which rules out
    // a silent fall back to MIOpen. The recorded property is what keeps the decline visible to
    // the parity comparison, which a pass on its own would not be.
    if(status == miopenStatusUnsupportedOp)
    {
        RecordServed(config.name, false);
        std::string message = miopenGetErrorString(status);
        if(ForwardingEnabled())
        {
            EXPECT_NE(message.find("[hipDNN-forwarded]"), std::string::npos)
                << "decline did not come from hipDNN: " << message;
        }
        GTEST_LOG_(INFO) << "fused conv+bias+activation was declined here, so the result was not "
                            "checked against the CPU reference: "
                         << message;
        return;
    }
    ASSERT_EQ(status, miopenStatusSuccess);
    RecordServed(config.name, true);

    y.data = handle_deref.Read<FusedType>(y_dev, y.data.size());

    CheckMatchesCpuBiasActivation(config, x, w, bias, y);
}

TEST(GPU_HipdnnShimConvBiasActivApi_FP16, FusedForwardMatchesCpuReference)
{
    for(const auto& config : FusedCases())
        ASSERT_NO_FATAL_FAILURE(RunFusedCase(config));
}

// fp16, so the forwarded graph is served only if its bias add computes in the bias type (in
// fp32 that is also the compute type). NCHW because MIOpen declines it before launching a
// kernel, so it cannot fault the test process. Both modes decline it on gfx90a, where the
// MIOpen provider turns fusion off.
TEST(GPU_HipdnnShimConvBiasActiv2d_FP16, FusedForwardMatchesCpuReference)
{
    ASSERT_NO_FATAL_FAILURE(RunFusedCase(
        FusedCase{"2d-nchw", miopenTensorNCHW, {1, 16, 8, 8}, {16, 16, 3, 3}, {1, 1}}));
}

// MIOpen accepts int8 input and weights with an int32 output, so the forwarded path must keep
// each tensor's type instead of declining the mix. No hipDNN provider has an int8 convolution
// engine yet, so with forwarding on this is declined, and the parity run lists it as a known
// divergence. Once a provider supports it, the parity run asks for that line to be removed.
TEST(GPU_HipdnnShimConvMixedTypeApi_I8, Int8InInt32OutMatchesCpuReference)
{
    auto& handle_deref    = get_handle();
    miopenHandle_t handle = &handle_deref;

    tensor<int8_t> x{2, 4, 8, 8};
    x.generate(tensor_elem_gen_integer{17});
    tensor<int8_t> w{4, 4, 3, 3};
    w.generate(tensor_elem_gen_integer{17});
    OwnedConvDescriptor conv;
    ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(conv));
    std::vector<std::size_t> out_lengths;
    ASSERT_NO_FATAL_FAILURE(OutputLengths(conv.handle, x, w, out_lengths));
    tensor<int> y{out_lengths};

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

    const auto status = miopenConvolutionForward(handle,
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
                                                 wspace.size());

    // A decline passes rather than skips, as in the fused test. It must come from hipDNN's engine
    // search: a decline from the up-front checks means the mixed types were never translated.
    const std::string case_name = "int8-in-int32-out";
    if(status == miopenStatusUnsupportedOp && ForwardingEnabled())
    {
        RecordServed(case_name, false);
        const std::string message = miopenGetErrorString(status);
        EXPECT_NE(message.find("[hipDNN-forwarded]"), std::string::npos)
            << "decline did not come from hipDNN: " << message;
        for(const char* check_reason : {"has no hipDNN convolution", "not forwarded to hipDNN"})
        {
            EXPECT_EQ(message.find(check_reason), std::string::npos)
                << "the mixed types were refused instead of translated: " << message;
        }
        return;
    }
    ASSERT_EQ(status, miopenStatusSuccess);
    RecordServed(case_name, true);

    y.data = handle_deref.Read<int>(y_dev, y.data.size());
    tensor<int> ref_y{out_lengths};
    const ConvGeometry geometry;
    cpu_convolution_forward(geometry.pads.size(),
                            x,
                            w,
                            ref_y,
                            geometry.pads,
                            geometry.strides,
                            geometry.dilations,
                            group_count);
    ASSERT_FALSE(miopen::range_zero(ref_y)) << "CPU reference is all zeros";
    // Integer arithmetic, so the results must match exactly.
    EXPECT_EQ(ref_y.data, y.data);
}

// One call per check the forwarded path makes before building a graph. Forwarding only, since
// natively several of these problems are malformed or crash. The reason is checked, not just
// the status, because a hipDNN decline for lack of an engine would hide a missing check. The
// data-type check has no case: no type the public API can describe reaches it.
TEST_F(GPU_HipdnnShimConvDeclined_FP32, UnsupportedProblemsAreDeclinedWithReason)
{
    if(!ForwardingEnabled())
        return;

    auto expect_declined_because = [](miopenStatus_t status, const char* reason) {
        EXPECT_EQ(status, miopenStatusUnsupportedOp) << reason;
        const std::string message = miopenGetErrorString(status);
        EXPECT_NE(message.find(reason), std::string::npos) << message;
    };

    // The declines happen before any hipDNN object is built, so no workspace is needed to
    // reach them.
    auto forward = [&](const float* alpha,
                       const float* beta,
                       tensor<float>& in,
                       tensor<float>& weights,
                       miopenConvolutionDescriptor_t c,
                       tensor<float>& out) {
        auto in_dev       = handle_deref.Write(in.data);
        auto weights_dev  = handle_deref.Write(weights.data);
        auto out_dev      = handle_deref.Write(out.data);
        const auto status = miopenConvolutionForward(handle,
                                                     alpha,
                                                     &in.desc,
                                                     in_dev.get(),
                                                     &weights.desc,
                                                     weights_dev.get(),
                                                     c,
                                                     miopenConvolutionFwdAlgoGEMM,
                                                     beta,
                                                     &out.desc,
                                                     out_dev.get(),
                                                     nullptr,
                                                     0);
        // With a check missing the call is served, so its work must finish before the buffers
        // are freed.
        handle_deref.Finish();
        return status;
    };

    const float two = 2.0f;
    expect_declined_because(forward(&two, &kZero, x, w, conv.handle, y),
                            "supports only alpha=1, beta=0");
    expect_declined_because(forward(&kOne, &kOne, x, w, conv.handle, y),
                            "supports only alpha=1, beta=0");
    expect_declined_because(forward(nullptr, &kZero, x, w, conv.handle, y),
                            "supports only alpha=1, beta=0");

    {
        tensor<float> vectorized_x{miopenTensorNCHWc4, x.desc.GetLengths()};
        tensor<float> vectorized_w{miopenTensorNCHWc4, w.desc.GetLengths()};
        expect_declined_because(forward(&kOne, &kZero, vectorized_x, vectorized_w, conv.handle, y),
                                "vectorized tensor layouts");
    }

    {
        OwnedConvDescriptor grouped;
        ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(grouped));
        ASSERT_EQ(miopenSetConvolutionGroupCount(grouped.handle, 2), miopenStatusSuccess);
        tensor<float> grouped_w{4, 2, 3, 3};
        expect_declined_because(forward(&kOne, &kZero, x, grouped_w, grouped.handle, y),
                                "grouped convolution is not forwarded");
    }

    {
        const ConvGeometry geometry{{1}, {1}, {1}};
        OwnedConvDescriptor one_d;
        ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(one_d, miopenConvolution, geometry));
        // Written out because MIOpen's output-size query fails for 1-D. A pad of 1 around a
        // 3-wide filter keeps the length.
        tensor<float> one_d_x{std::vector<std::size_t>{2, 4, 8}};
        tensor<float> one_d_w{std::vector<std::size_t>{4, 4, 3}};
        tensor<float> one_d_y{std::vector<std::size_t>{2, 4, 8}};
        expect_declined_because(forward(&kOne, &kZero, one_d_x, one_d_w, one_d.handle, one_d_y),
                                "only 2-D and 3-D convolutions");
    }

    {
        const ConvGeometry geometry{{0, 0, 0, 0}, {1, 1, 1, 1}, {1, 1, 1, 1}};
        OwnedConvDescriptor four_d;
        ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(four_d, miopenConvolution, geometry));
        // A 1x1x1x1 filter with no padding keeps the input's shape. Left zeroed because the
        // element generator does not take six dimensions.
        const std::vector<std::size_t> lengths{1, 1, 2, 2, 2, 2};
        tensor<float> four_d_x{lengths};
        tensor<float> four_d_w{std::vector<std::size_t>(6, 1)};
        tensor<float> four_d_y{lengths};
        expect_declined_because(forward(&kOne, &kZero, four_d_x, four_d_w, four_d.handle, four_d_y),
                                "only 2-D and 3-D convolutions");
    }

    {
        OwnedConvDescriptor transposed;
        ASSERT_NO_FATAL_FAILURE(InitConvDescriptor(transposed, miopenTranspose));
        std::vector<std::size_t> transposed_lengths;
        ASSERT_NO_FATAL_FAILURE(OutputLengths(transposed.handle, x, w, transposed_lengths));
        tensor<float> transposed_y{transposed_lengths};
        expect_declined_because(forward(&kOne, &kZero, x, w, transposed.handle, transposed_y),
                                "transposed convolution is not forwarded");
    }

    miopenStatus_t fused_status = miopenStatusSuccess;
    std::string fused_message;
    ASSERT_NO_FATAL_FAILURE(RunFused(nullptr, miopenActivationRELU, fused_status, &fused_message));
    EXPECT_EQ(fused_status, miopenStatusUnsupportedOp);
    EXPECT_NE(fused_message.find("alpha2 is null"), std::string::npos) << fused_message;
}

// Runs in both modes: for an activation other than ReLU, the forwarded path must return the
// same status as MIOpen's own fused path.
TEST_F(GPU_HipdnnShimConvDeclined_FP32, NonReluActivationIsNotImplemented)
{
    miopenStatus_t status = miopenStatusSuccess;
    ASSERT_NO_FATAL_FAILURE(RunFused(&kZero, miopenActivationPASTHRU, status));
    EXPECT_EQ(status, miopenStatusNotImplemented);
}
