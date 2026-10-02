// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#if !defined(HOST_DEVICE)
#if defined(__HIPCC__) || defined(__HIP__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif
#endif

namespace hipdnn_gpu_ref::types::detail
{

HOST_DEVICE inline uint32_t floatToBits(float f) noexcept
{
#if defined(__has_builtin)
#if __has_builtin(__builtin_bit_cast)
    return __builtin_bit_cast(uint32_t, f);
#else
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
#endif
#else
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
#endif
}

HOST_DEVICE inline float bitsToFloat(uint32_t b) noexcept
{
#if defined(__has_builtin)
#if __has_builtin(__builtin_bit_cast)
    return __builtin_bit_cast(float, b);
#else
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
#endif
#else
    float f;
    std::memcpy(&f, &b, sizeof(f));
    return f;
#endif
}

} // namespace hipdnn_gpu_ref::types::detail
