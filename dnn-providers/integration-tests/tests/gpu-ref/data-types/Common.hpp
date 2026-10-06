// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefLaunch.hpp>
#include <hipdnn-gpu-ref/detail/HipRtcTypeName.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include <GpuRefCommonArgs.h>
#include <hip/hip_runtime.h>

namespace gpu_ref_device_data_type_test
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_gpu_ref::detail;

namespace detail
{

constexpr unsigned int BLOCK_SIZE = 256;

struct StateFlags
{
    bool nan = false;
    bool inf = false;
    bool signbit = false;
    bool finite = false;
};

template <typename InputType, typename OutputType>
void deviceDataCast(const std::vector<InputType>& input, std::vector<OutputType>& output)
{
    const auto elementCount = static_cast<int64_t>(input.size());
    if(elementCount == 0)
    {
        return;
    }

    if(elementCount != static_cast<int64_t>(output.size()))
    {
        throw std::invalid_argument("Source and destination vectors must have the same size.");
    }

    Tensor<InputType> inputTensor({elementCount});
    Tensor<OutputType> outputTensor({elementCount});
    std::memcpy(inputTensor.memory().hostData(),
                input.data(),
                static_cast<size_t>(elementCount) * sizeof(InputType));
    inputTensor.markHostModified();

    auto& compiler = GpuRefKernelCompiler::instance();
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DINPUT_TYPE=") + HipRtcTypeName<InputType>::VALUE);
    defines.emplace_back(std::string("-DOUTPUT_TYPE=") + HipRtcTypeName<OutputType>::VALUE);
    defines.emplace_back("-DCOMPUTE_TYPE=float");
    const auto& kernel = compiler.getOrCompile("GpuRefDataTypeUtils.cpp", defines, "DataCast");

    DataTypeUtilArgs args{};
    args.input = inputTensor.memory().deviceData();
    args.output = outputTensor.memory().deviceData();
    args.count = static_cast<long long>(elementCount);

    const auto gridSize = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipdnn_gpu_ref::detail::launchKernel1d(
        kernel.function(), gridSize, BLOCK_SIZE, &args, sizeof(args));

    outputTensor.markDeviceModified();
    std::memcpy(output.data(),
                outputTensor.memory().hostData(),
                static_cast<size_t>(elementCount) * sizeof(OutputType));
}

template <typename T>
std::vector<T> deviceNegate(const std::vector<T>& input)
{
    const auto elementCount = static_cast<int64_t>(input.size());
    if(elementCount == 0)
    {
        return {};
    }

    Tensor<T> inputTensor({elementCount});
    Tensor<T> outputTensor({elementCount});
    std::memcpy(inputTensor.memory().hostData(),
                input.data(),
                static_cast<size_t>(elementCount) * sizeof(T));
    inputTensor.markHostModified();

    auto& compiler = GpuRefKernelCompiler::instance();
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DINPUT_TYPE=") + HipRtcTypeName<T>::VALUE);
    defines.emplace_back(std::string("-DOUTPUT_TYPE=") + HipRtcTypeName<T>::VALUE);
    defines.emplace_back("-DCOMPUTE_TYPE=float");

    const auto& kernel = compiler.getOrCompile("GpuRefDataTypeUtils.cpp", defines, "Negate");

    DataTypeUtilArgs args{};
    args.input = inputTensor.memory().deviceData();
    args.output = outputTensor.memory().deviceData();
    args.count = static_cast<long long>(elementCount);

    const auto gridSize = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipdnn_gpu_ref::detail::launchKernel1d(
        kernel.function(), gridSize, BLOCK_SIZE, &args, sizeof(args));

    outputTensor.markDeviceModified();
    std::vector<T> output(static_cast<size_t>(elementCount));
    std::memcpy(output.data(),
                outputTensor.memory().hostData(),
                static_cast<size_t>(elementCount) * sizeof(T));

    return output;
}

template <typename T>
std::vector<StateFlags> deviceQueryState(const std::vector<T>& input)
{
    const auto elementCount = static_cast<int64_t>(input.size());
    if(elementCount == 0)
    {
        return {};
    }

    Tensor<T> dataTensor({elementCount});
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    Tensor<uint8_t> flagsTensor({elementCount});
    std::memcpy(dataTensor.memory().hostData(),
                input.data(),
                static_cast<size_t>(elementCount) * sizeof(T));
    dataTensor.markHostModified();

    auto& compiler = GpuRefKernelCompiler::instance();
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DINPUT_TYPE=") + HipRtcTypeName<T>::VALUE);
    defines.emplace_back(std::string("-DOUTPUT_TYPE=") + HipRtcTypeName<T>::VALUE);
    defines.emplace_back("-DCOMPUTE_TYPE=float");

    const auto& kernel = compiler.getOrCompile("GpuRefDataTypeUtils.cpp", defines, "QueryState");

    DataTypeUtilArgs args{};
    args.input = dataTensor.memory().deviceData();
    args.output = flagsTensor.memory().deviceData();
    args.count = static_cast<long long>(elementCount);

    const auto gridSize = (elementCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipdnn_gpu_ref::detail::launchKernel1d(
        kernel.function(), gridSize, BLOCK_SIZE, &args, sizeof(args));

    flagsTensor.markDeviceModified();
    const auto* rawFlags = static_cast<const uint8_t*>(flagsTensor.memory().hostData());
    std::vector<StateFlags> outStateFlags(static_cast<size_t>(elementCount));
    for(size_t i = 0; i < static_cast<size_t>(elementCount); ++i)
    {
        outStateFlags[i] = {(rawFlags[i] & 1) != 0,
                            (rawFlags[i] & 2) != 0,
                            (rawFlags[i] & 4) != 0,
                            (rawFlags[i] & 8) != 0};
    }

    return outStateFlags;
}

} // namespace detail

template <typename T>
std::vector<T> deviceEncode(const std::vector<float>& input)
{
    std::vector<T> output(input.size());
    detail::deviceDataCast(input, output);
    return output;
}

template <typename T>
std::vector<float> deviceDecode(const std::vector<T>& input)
{
    std::vector<float> output(input.size());
    detail::deviceDataCast(input, output);
    return output;
}

template <typename T>
std::vector<float> deviceRoundTrip(const std::vector<float>& input)
{
    return deviceDecode<T>(deviceEncode<T>(input));
}

template <typename T>
T deviceEncodeSingle(float val)
{
    return deviceEncode<T>({val})[0];
}

template <typename T>
float deviceDecodeSingle(T val)
{
    return deviceDecode<T>({val})[0];
}

template <typename T>
float deviceRoundTripSingle(float value)
{
    return deviceDecodeSingle<T>(deviceEncodeSingle<T>(value));
}

} // namespace gpu_ref_type_test
