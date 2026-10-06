// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <type_traits>

#include "GpuRefTypes.h"

template <typename T>
constexpr bool IS_STD_FLOAT_V = std::is_floating_point<T>::value || std::is_same<T, _Float16>::value
                                || std::is_same<T, __bf16>::value;

// --- deviceIsNan ---
template <typename T>
__device__ inline typename std::enable_if<!IS_STD_FLOAT_V<T>, bool>::type deviceIsNan(T x)
{
    return isnan(x); // Use ADL functions
}

template <typename T>
__device__ inline typename std::enable_if<IS_STD_FLOAT_V<T>, bool>::type deviceIsNan(T x)
{
    return isnan(static_cast<double>(x)); // Resolve ambiguity in hip_runtime.h
}

// --- deviceIsInf ---
template <typename T>
__device__ inline typename std::enable_if<!IS_STD_FLOAT_V<T>, bool>::type deviceIsInf(T x)
{
    return isinf(x); // Use ADL functions
}

template <typename T>
__device__ inline typename std::enable_if<IS_STD_FLOAT_V<T>, bool>::type deviceIsInf(T x)
{
    return isinf(static_cast<double>(x)); // Resolve ambiguity in hip_runtime.h
}

// --- deviceSignBit ---
template <typename T>
__device__ inline typename std::enable_if<!IS_STD_FLOAT_V<T>, bool>::type deviceSignBit(T x)
{
    return signbit(x); // Use ADL functions
}

template <typename T>
__device__ inline typename std::enable_if<IS_STD_FLOAT_V<T>, bool>::type deviceSignBit(T x)
{
    return signbit(static_cast<double>(x)); // Resolve ambiguity in hip_runtime.h
}

// --- deviceIsFinite ---
template <typename T>
__device__ inline typename std::enable_if<!IS_STD_FLOAT_V<T>, bool>::type deviceIsFinite(T x)
{
    return isfinite(x); // Use ADL functions
}

template <typename T>
__device__ inline typename std::enable_if<IS_STD_FLOAT_V<T>, bool>::type deviceIsFinite(T x)
{
    return isfinite(static_cast<double>(x)); // Resolve ambiguity in hip_runtime.h
}

extern "C" __global__ void DataCast(DataTypeUtilArgs args)
{
    auto* input = static_cast<const INPUT_TYPE*>(args.input);
    auto* output = static_cast<OUTPUT_TYPE*>(args.output);

    const auto idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                     + static_cast<long long>(threadIdx.x);
    if(idx < args.count)
    {
        output[idx] = static_cast<OUTPUT_TYPE>(input[idx]);
    }
}

extern "C" __global__ void Negate(DataTypeUtilArgs args)
{
    auto* input = static_cast<const INPUT_TYPE*>(args.input);
    auto* output = static_cast<INPUT_TYPE*>(args.output);

    const auto idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                     + static_cast<long long>(threadIdx.x);
    if(idx < args.count)
    {
        output[idx] = -input[idx];
    }
}

extern "C" __global__ void QueryState(DataTypeUtilArgs args)
{
    auto* data = static_cast<const INPUT_TYPE*>(args.input);
    auto* flags = static_cast<uint8_t*>(args.output);

    const auto idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                     + static_cast<long long>(threadIdx.x);
    if(idx >= args.count)
    {
        return;
    }

    const INPUT_TYPE x = data[idx];

    const uint8_t isNan = deviceIsNan(x) ? 1 : 0;
    const uint8_t isInf = deviceIsInf(x) ? 2 : 0;
    const uint8_t isSign = deviceSignBit(x) ? 4 : 0;
    const uint8_t isFinite = deviceIsFinite(x) ? 8 : 0;

    flags[idx] = static_cast<uint8_t>(isNan | isInf | isSign | isFinite);
}
