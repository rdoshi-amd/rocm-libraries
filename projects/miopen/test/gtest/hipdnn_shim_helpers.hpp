// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Helpers shared by the hipDNN shim tests. Every call that selects or performs compute goes
// through a public miopen.h entry point, and results are checked against a CPU reference
// rather than another MIOpen run.

#pragma once

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../cpu_conv.hpp"
#include "../verify.hpp"
#include "../workspace.hpp"

#include <miopen/miopen.h>

// Reached by relative path because src/private is deliberately off the test include path.
#include "../../src/private/routing.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace hipdnn_shim_test {

inline constexpr std::size_t group_count = 1;

inline constexpr float kOne  = 1.0f;
inline constexpr float kZero = 0.0f;

// The default is the 2-D problem most tests share.
struct ConvGeometry
{
    std::vector<int> pads      = {1, 1};
    std::vector<int> strides   = {1, 1};
    std::vector<int> dilations = {1, 1};
    int groups                 = 1;
};

// Small because the parity run executes every test twice, but large enough that a wrong
// kernel cannot match the reference by chance.
inline tensor<float> MakeInput()
{
    tensor<float> x{2, 4, 8, 8};
    x.generate(tensor_elem_gen_integer{17});
    return x;
}

inline tensor<float> MakeWeights()
{
    tensor<float> w{4, 4, 3, 3};
    w.generate(tensor_elem_gen_integer{17});
    return w;
}

// So an ASSERT_* that ends a test early does not leak the handle.
template <class T, miopenStatus_t (*Destroy)(T)>
struct Owned
{
    Owned()                        = default;
    Owned(const Owned&)            = delete;
    Owned& operator=(const Owned&) = delete;
    ~Owned()
    {
        if(handle != nullptr)
            EXPECT_EQ(Destroy(handle), miopenStatusSuccess);
    }

    T handle = nullptr;
};

using OwnedConvDescriptor =
    Owned<miopenConvolutionDescriptor_t, miopenDestroyConvolutionDescriptor>;
using OwnedProblem = Owned<miopenProblem_t, miopenDestroyProblem>;
using OwnedActivationDescriptor =
    Owned<miopenActivationDescriptor_t, miopenDestroyActivationDescriptor>;

// Declare it after the vector it refers to, so it is destroyed first. Nulls are skipped
// because the vector can hold slots the find call did not fill.
struct OwnedSolutions
{
    explicit OwnedSolutions(const std::vector<miopenSolution_t>& s) : solutions(s) {}
    OwnedSolutions(const OwnedSolutions&)            = delete;
    OwnedSolutions& operator=(const OwnedSolutions&) = delete;
    ~OwnedSolutions()
    {
        for(auto* solution : solutions)
        {
            if(solution != nullptr)
                EXPECT_EQ(miopenDestroySolution(solution), miopenStatusSuccess);
        }
    }

    const std::vector<miopenSolution_t>& solutions;
};

// Fatal on failure, so later calls are never handed a null descriptor. Wrap calls in
// ASSERT_NO_FATAL_FAILURE, since a fatal failure only returns from this function.
inline void InitConvDescriptor(OwnedConvDescriptor& conv,
                               miopenConvolutionMode_t mode = miopenConvolution,
                               const ConvGeometry& geometry = {})
{
    ASSERT_EQ(miopenCreateConvolutionDescriptor(&conv.handle), miopenStatusSuccess);
    ASSERT_EQ(miopenInitConvolutionNdDescriptor(conv.handle,
                                                static_cast<int>(geometry.pads.size()),
                                                geometry.pads.data(),
                                                geometry.strides.data(),
                                                geometry.dilations.data(),
                                                mode),
              miopenStatusSuccess);
    if(geometry.groups != 1)
        ASSERT_EQ(miopenSetConvolutionGroupCount(conv.handle, geometry.groups),
                  miopenStatusSuccess);
}

// Asks the library rather than computing the shape here, so the shape is part of what the two
// modes must agree on. Fatal on failure, because an empty output matches an empty reference.
// The tensors are non-const because the C API takes `const miopenTensorDescriptor_t`, a const
// pointer to a non-const descriptor.
template <class T>
void OutputLengths(miopenConvolutionDescriptor_t conv_desc,
                   tensor<T>& x,
                   tensor<T>& w,
                   std::vector<std::size_t>& out_lengths)
{
    const auto expected_dims = x.desc.GetNumDims();
    int out_dim_count        = 0;
    std::vector<int> out_dims(expected_dims);
    ASSERT_EQ(miopenGetConvolutionNdForwardOutputDim(
                  conv_desc, &x.desc, &w.desc, &out_dim_count, out_dims.data()),
              miopenStatusSuccess);
    ASSERT_EQ(out_dim_count, static_cast<int>(expected_dims));
    out_lengths.assign(out_dims.begin(), out_dims.end());
}

// Find writes its own result into the output buffer, and Find is never forwarded, so without
// this a forwarded call that wrote nothing would pass on MIOpen's answer.
template <class T>
void ResetAfterFind(miopen::Allocator::ManageDataPtr& output_dev, const tensor<T>& output)
{
    output_dev = get_handle().Write(output.data);
}

// The tolerances ConvFwdSolverTestBase::ThresholdChecks() uses.
template <class T>
double Tolerance()
{
    return std::numeric_limits<T>::epsilon() * (std::is_same_v<T, bfloat16> ? 4 : 80);
}

// Internal MIOpen helpers are fine here: they only compare results, they do not produce them.
template <class T>
void CheckWithinTolerance(const tensor<T>& reference, const tensor<T>& got, const char* what)
{
    // rms_range() is 0 for two empty or two all-zero ranges, so without these a run that
    // produced nothing would pass.
    ASSERT_FALSE(miopen::range_zero(reference)) << what << ": CPU reference is all zeros";
    ASSERT_FALSE(miopen::range_zero(got)) << what << ": GPU result is all zeros";
    ASSERT_EQ(miopen::range_distance(reference), miopen::range_distance(got)) << what;
    ASSERT_LT(miopen::find_idx(reference, miopen::not_finite), 0)
        << what << ": non-finite value in the CPU reference";

    const double error = miopen::rms_range(reference, got);
    EXPECT_LT(error, Tolerance<T>()) << what << " beyond cross-implementation tolerance";
}

// Each reference takes the result's own descriptor, so a kernel that writes into the gaps of
// a strided result fails the check.
template <class T>
void CheckMatchesCpuReference(const tensor<T>& x,
                              const tensor<T>& w,
                              const tensor<T>& y,
                              const ConvGeometry& geometry = {})
{
    tensor<T> ref_y{y.desc};
    cpu_convolution_forward(geometry.pads.size(),
                            x,
                            w,
                            ref_y,
                            geometry.pads,
                            geometry.strides,
                            geometry.dilations,
                            geometry.groups);
    CheckWithinTolerance(ref_y, y, "convolution result");
}

template <class T>
void CheckMatchesCpuBackwardData(const tensor<T>& dx,
                                 const tensor<T>& w,
                                 const tensor<T>& dy,
                                 const ConvGeometry& geometry = {})
{
    tensor<T> ref_dx{dx.desc};
    cpu_convolution_backward_data(geometry.pads.size(),
                                  ref_dx,
                                  w,
                                  dy,
                                  geometry.pads,
                                  geometry.strides,
                                  geometry.dilations,
                                  geometry.groups);
    CheckWithinTolerance(ref_dx, dx, "backward-data result");
}

template <class T>
void CheckMatchesCpuBackwardWeights(const tensor<T>& x,
                                    const tensor<T>& dw,
                                    const tensor<T>& dy,
                                    const ConvGeometry& geometry = {})
{
    tensor<T> ref_dw{dw.desc};
    cpu_convolution_backward_weight(geometry.pads.size(),
                                    x,
                                    ref_dw,
                                    dy,
                                    geometry.pads,
                                    geometry.strides,
                                    geometry.dilations,
                                    geometry.groups);
    CheckWithinTolerance(ref_dw, dw, "backward-weights result");
}

// A served case and a declined case both pass, so this property is what lets the parity
// comparison tell them apart. Keyed by case name because gtest keeps only the last value
// recorded under a key.
inline void RecordServed(const std::string& case_name, bool served)
{
    ::testing::Test::RecordProperty("parity_served_" + case_name, served ? "true" : "false");
}

// For a case neither mode serves. Records the enum name, because miopenGetErrorString's text
// always differs between the modes.
inline void RecordStatus(const std::string& case_name, miopenStatus_t status)
{
    const char* name = "Unrecognized";
    switch(status)
    {
    case miopenStatusSuccess: name = "Success"; break;
    case miopenStatusNotInitialized: name = "NotInitialized"; break;
    case miopenStatusInvalidValue: name = "InvalidValue"; break;
    case miopenStatusBadParm: name = "BadParm"; break;
    case miopenStatusAllocFailed: name = "AllocFailed"; break;
    case miopenStatusInternalError: name = "InternalError"; break;
    case miopenStatusNotImplemented: name = "NotImplemented"; break;
    case miopenStatusUnknownError: name = "UnknownError"; break;
    case miopenStatusUnsupportedOp: name = "UnsupportedOp"; break;
    case miopenStatusGpuOperationsSkipped: name = "GpuOperationsSkipped"; break;
    case miopenStatusVersionMismatch: name = "VersionMismatch"; break;
    }
    ::testing::Test::RecordProperty("parity_status_" + case_name, name);
}

// For a case where one mode is known to return a wrong answer, so the test records the check
// instead of failing on it.
inline void RecordCorrect(const std::string& case_name, bool correct)
{
    ::testing::Test::RecordProperty("parity_correct_" + case_name, correct ? "true" : "false");
}

// Fixed for the life of the process; the parity run executes each binary once per mode.
inline bool ForwardingEnabled()
{
    return miopen::wrapper::GetForwardingMode() == miopen::wrapper::ForwardingMode::Enabled;
}

// The parity comparison matches known divergences against this device. Recorded only when a
// shim test is selected, because in a single-binary build this runs for every test selection,
// and most of those must not touch the GPU.
class RecordParityDevice : public ::testing::Environment
{
public:
    void SetUp() override
    {
        const ::testing::UnitTest& unit_test = *::testing::UnitTest::GetInstance();
        for(int i = 0; i < unit_test.total_test_suite_count(); ++i)
        {
            const ::testing::TestSuite& suite = *unit_test.GetTestSuite(i);
            if(suite.should_run() &&
               std::string(suite.name()).find("HipdnnShim") != std::string::npos)
            {
                ::testing::Test::RecordProperty("forwarding_parity_device",
                                                get_handle().GetDeviceName());
                return;
            }
        }
    }
};

// Inline, so a binary with several shim test files registers it once.
inline ::testing::Environment* const kRecordParityDevice =
    ::testing::AddGlobalTestEnvironment(new RecordParityDevice);

} // namespace hipdnn_shim_test
