// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "Common.hpp"

namespace hipdnn_gpu_ref::data_types
{

namespace detail
{

// ============================================================================
// FP6 E3M2 (OCP MX Format) Bit Layout Constants
// ============================================================================
// See fp6_e3m2 struct documentation for full format specification.
//
// Bit layout: [S|EEE|MM]
//              5 4 3 2 1 0
// ============================================================================

/// Sign bit mask (bit 5)
constexpr uint8_t FP6_E3M2_SIGN_MASK = 0x20;

/// Exponent field mask (bits 2-4)
constexpr uint8_t FP6_E3M2_EXP_MASK = 0x1C;

/// Mantissa field mask (bits 0-1)
constexpr uint8_t FP6_E3M2_MANT_MASK = 0x03;

/// Absolute value mask (bits 0-4)
constexpr uint8_t FP6_E3M2_ABS_MASK = 0x1F;

/// Exponent bias for E3M2
constexpr int FP6_E3M2_EXP_BIAS = 3;

/// Number of exponent bits
constexpr int FP6_E3M2_EXP_BITS = 3;

/// Number of mantissa bits
constexpr int FP6_E3M2_MANT_BITS = 2;

// ============================================================================
// FP6 E3M2 Special Values (bit patterns)
// ============================================================================

/// Positive zero
constexpr uint8_t FP6_E3M2_POS_ZERO = 0x00;

/// Negative zero
constexpr uint8_t FP6_E3M2_NEG_ZERO = 0x20;

/// Minimum positive subnormal: 0.0625
constexpr uint8_t FP6_E3M2_DENORM_MIN = 0x01;

/// Minimum positive normal: 0.25
constexpr uint8_t FP6_E3M2_MIN_NORMAL = 0x04;

/// Maximum positive value: 28.0
constexpr uint8_t FP6_E3M2_MAX = 0x1F;

/// Maximum negative value (lowest): -28.0
constexpr uint8_t FP6_E3M2_LOWEST = 0x3F;

/// Epsilon: smallest difference at 1.0 = 0.25
constexpr uint8_t FP6_E3M2_EPSILON = 0x04;

/// Round error (0.5) - maximum rounding error
constexpr uint8_t FP6_E3M2_ROUND_ERROR = 0x08;

/// Rounding threshold for round-to-nearest-even (midpoint of 21-bit remainder)
constexpr uint32_t FP6_E3M2_ROUND_THRESHOLD = 0x100000;

// Convert float to FP6 E3M2 bits (OCP MX format: 1 sign, 3 exponent, 2 mantissa)
// Range: +/- 28.0, no infinity (saturates to max), no NaN (returns zero)
// NOLINTNEXTLINE(readability-identifier-naming)
HOST_DEVICE inline uint8_t float_to_fp6_e3m2_bits(float f) noexcept
{
    const uint32_t bits = floatToBits(f);
    const uint32_t absBits = bits & 0x7FFFFFFFu;
    const uint32_t sign = (bits >> 26) & FP6_E3M2_SIGN_MASK; // float sign bit 31 -> bit 5

    // Per OCP MX Specification v1.0, conversion from NaN is implementation-defined.
    // This implementation returns zero.
    if(absBits > 0x7F800000u)
    {
        return static_cast<uint8_t>(FP6_E3M2_POS_ZERO);
    }

    // E3M2 has no infinity - saturate to max
    if(absBits == 0x7F800000u)
    {
        return static_cast<uint8_t>(sign ? FP6_E3M2_LOWEST : FP6_E3M2_MAX);
    }

    // Rebias from float (127) to E3M2 (3)
    const uint32_t fp32Exp = absBits >> 23;
    int32_t exp = static_cast<int32_t>(fp32Exp) - 127 + FP6_E3M2_EXP_BIAS;
    uint32_t mant = bits & 0x007FFFFFu;

    // Handle overflow (exp > 7 means value exceeds max representable)
    if(exp > 7)
    {
        return static_cast<uint8_t>(sign | FP6_E3M2_ABS_MASK); // Saturate to max finite value
    }

    // Handle zero
    if(fp32Exp == 0 && mant == 0)
    {
        return static_cast<uint8_t>(sign);
    }

    // Subnormal / underflow range (may round up into the smallest normal)
    if(exp <= 0)
    {
        mant |= 0x00800000u; // implicit 1
        // 23 - 2 = 21 bits to shift
        const auto shift = static_cast<uint32_t>(1 - exp + 21);
        if(shift > 24)
        {
            return static_cast<uint8_t>(sign); // too small -> zero
        }

        // Apply round-to-nearest-even for subnormal results
        const uint32_t halfPoint = 1u << (shift - 1);
        const uint32_t remainder = mant & ((1u << shift) - 1u);
        mant >>= shift;

        // Round to nearest even
        if(remainder > halfPoint || (remainder == halfPoint && ((mant & 1u) != 0u)))
        {
            mant++;
            if(mant > FP6_E3M2_MANT_MASK)
            {
                // Promoted to smallest normal: exp=1, mant=0 -> value 0.25
                return static_cast<uint8_t>(sign | FP6_E3M2_MIN_NORMAL);
            }
        }
        return static_cast<uint8_t>(sign | (mant & FP6_E3M2_MANT_MASK));
    }
    // Normal case: shift mantissa from 23 bits to 2 bits with rounding
    uint32_t fp6Mant = (mant >> 21) & FP6_E3M2_MANT_MASK;
    const uint32_t remainder = mant & 0x001FFFFFu;

    // Round to nearest even
    if(remainder > FP6_E3M2_ROUND_THRESHOLD
       || (remainder == FP6_E3M2_ROUND_THRESHOLD && ((fp6Mant & 1u) != 0u)))
    {
        fp6Mant++;
        if(fp6Mant > 3)
        {
            fp6Mant = 0;
            exp++;
            if(exp > 7)
            {
                return static_cast<uint8_t>(sign | FP6_E3M2_ABS_MASK);
            }
        }
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 2) | fp6Mant);
}

// Convert FP6 E3M2 bits to float
// NOLINTNEXTLINE(readability-identifier-naming)
HOST_DEVICE inline float fp6_e3m2_bits_to_float(uint8_t b) noexcept
{
    // Sign bit 5 -> float sign bit 31
    const uint32_t sign = static_cast<uint32_t>(b & FP6_E3M2_SIGN_MASK) << 26;
    const uint32_t exp = static_cast<uint32_t>(b & FP6_E3M2_EXP_MASK) >> FP6_E3M2_MANT_BITS;
    const uint32_t mant = static_cast<uint32_t>(b & FP6_E3M2_MANT_MASK);

    uint32_t fp32Bits;
    if(exp == 0)
    {
        if(mant == 0u)
        {
            fp32Bits = sign;
        }
        else if(mant == 1u)
        {
            fp32Bits = sign | 0x3d800000u; // 2^-4
        }
        else if(mant == 2u)
        {
            fp32Bits = sign | 0x3e000000u; // 2^-3
        }
        else
        {
            fp32Bits = sign | 0x3e400000u; // 1.5 * 2^-3
        }
    }
    else
    {
        // Normal: rebias exponent from 3 to 127 and left-align the 2-bit mantissa.
        fp32Bits = sign | ((exp + (127u - FP6_E3M2_EXP_BIAS)) << 23)
                   | (mant << (23 - FP6_E3M2_MANT_BITS));
    }
    return bitsToFloat(fp32Bits);
}

} // namespace detail

/**
 * @brief Custom storage-only FP6 E3M2 type  on device
 *
 * This type provides a portable OCP FP6 E3M2 (1 sign, 3 exponent, 2 mantissa)
 * implementation

 * This is a STORAGE-ONLY type intended for data representation and conversion,
 * not direct computation. Arithmetic operations and comparisons are
 * intentionally not provided. For computation, explicitly convert to float.
 *
 * - Binary layout: 1 sign bit (bit 5), 3 exponent bits (bits 2-4), 2 mantissa bits (bits 0-1)
 * - Exponent bias: 3
 * - No infinity representation
 * - No NaN representation
 * - Subnormals supported
 *
 * Range: -28.0 to +28.0
 * Rounding: roundTiesToEven (IEEE 754 default)
 * Overflow: Saturates to ±28.0 (maximum magnitude)
 * Underflow: Values with magnitude below 0.03125 round to zero
 */
// NOLINTNEXTLINE(readability-identifier-naming)
struct fp6_e3m2
{
    // Raw bits representation of the fp6_e3m2 value on device
    uint8_t data;

    constexpr fp6_e3m2() noexcept
        : data(0)
    {
    }

    fp6_e3m2(const fp6_e3m2&) = default;
    fp6_e3m2(fp6_e3m2&&) noexcept = default;
    fp6_e3m2& operator=(const fp6_e3m2&) = default;
    fp6_e3m2& operator=(fp6_e3m2&&) noexcept = default;

    HOST_DEVICE explicit fp6_e3m2(float f) noexcept
        : data(detail::float_to_fp6_e3m2_bits(f))
    {
    }

    HOST_DEVICE explicit fp6_e3m2(double d) noexcept
        : fp6_e3m2(static_cast<float>(d))
    {
    }

    HOST_DEVICE explicit fp6_e3m2(_Float16 h) noexcept
        : fp6_e3m2(static_cast<float>(h))
    {
    }

    HOST_DEVICE explicit fp6_e3m2(__bf16 b) noexcept
        : fp6_e3m2(static_cast<float>(b))
    {
    }

    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    HOST_DEVICE explicit fp6_e3m2(T value) noexcept
        : fp6_e3m2(static_cast<float>(value))
    {
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    HOST_DEVICE static constexpr fp6_e3m2 from_bits(uint8_t bits) noexcept
    {
        fp6_e3m2 val;
        val.data = bits;
        return val;
    }

    HOST_DEVICE explicit operator float() const noexcept
    {
        return detail::fp6_e3m2_bits_to_float(data);
    }

    HOST_DEVICE explicit operator double() const noexcept
    {
        return static_cast<double>(detail::fp6_e3m2_bits_to_float(data));
    }

    HOST_DEVICE explicit operator _Float16() const noexcept
    {
        return static_cast<_Float16>(detail::fp6_e3m2_bits_to_float(data));
    }

    HOST_DEVICE explicit operator __bf16() const noexcept
    {
        return static_cast<__bf16>(detail::fp6_e3m2_bits_to_float(data));
    }

    HOST_DEVICE fp6_e3m2 operator-() const noexcept
    {
        return from_bits(static_cast<uint8_t>(data ^ detail::FP6_E3M2_SIGN_MASK));
    }

    HOST_DEVICE fp6_e3m2 operator+() const noexcept
    {
        return *this;
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(fp6_e3m2) == sizeof(uint8_t), "fp6_e3m2 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp6_e3m2>, "fp6_e3m2 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp6_e3m2>, "fp6_e3m2 must be standard layout");
static_assert(std::is_default_constructible_v<fp6_e3m2>, "fp6_e3m2 must be default constructible");
static_assert(std::is_copy_constructible_v<fp6_e3m2>, "fp6_e3m2 must be copy constructible");
static_assert(std::is_move_constructible_v<fp6_e3m2>, "fp6_e3m2 must be move constructible");

// ============================================================================
// Math functions for fp6_e3m2
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: isnan(x), isinf(x), etc.
// ============================================================================

HOST_DEVICE inline bool isnan(fp6_e3m2 /*x*/)
{
    // FP6 E3M2 has no NaN representation
    return false;
}

HOST_DEVICE inline bool isinf(fp6_e3m2 /*x*/)
{
    // FP6 E3M2 has no infinity representation
    return false;
}

HOST_DEVICE inline bool signbit(fp6_e3m2 x)
{
    return (x.data & detail::FP6_E3M2_SIGN_MASK) != 0;
}

HOST_DEVICE inline bool isfinite(fp6_e3m2 /*x*/)
{
    // All FP6 E3M2 values are finite (no NaN or Inf)
    return true;
}

// ============================================================================
// Packed FP6 E3M2 Storage Type (2 values per 16 bits)
// ============================================================================

/**
 * @brief Packed storage type for 2 FP6 E3M2 values in 16 bits
 *
 * This type provides packed storage for FP6 E3M2 values,
 * storing 2 values per 16-bit word.
 *
 * Storage Model:
 * - Bits 0-5: lo element
 * - Bits 6-11: hi element
 * - Bits 12-15: unused (padding)
 *
 * This is a STORAGE-ONLY type. For value operations, access individual
 * elements via lo() and hi() methods.
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp6x2_e3m2
{
    /// Raw bit representation storing 2 packed FP6 values.
    /// Bits 0-5 = lo element, Bits 6-11 = hi element, Bits 12-15 = unused.
    uint16_t data;

    // Default constructor - both elements zero
    constexpr fp6x2_e3m2() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp6x2_e3m2(const fp6x2_e3m2&) = default;
    fp6x2_e3m2(fp6x2_e3m2&&) noexcept = default;
    fp6x2_e3m2& operator=(const fp6x2_e3m2&) = default;
    fp6x2_e3m2& operator=(fp6x2_e3m2&&) noexcept = default;

    // Single value constructor - lo = val, hi = zero
    constexpr explicit fp6x2_e3m2(fp6_e3m2 lo) noexcept
        : data(lo.data & 0x3F)
    {
    }

    // Two values constructor
    constexpr fp6x2_e3m2(fp6_e3m2 lo, fp6_e3m2 hi) noexcept
        : data(static_cast<uint16_t>((lo.data & 0x3F) | ((hi.data & 0x3F) << 6)))
    {
    }

    // Get lo element (bits 0-5)
    HOST_DEVICE constexpr fp6_e3m2 lo() const noexcept
    {
        return fp6_e3m2::from_bits(static_cast<uint8_t>(data & 0x3F));
    }

    // Get hi element (bits 6-11)
    HOST_DEVICE constexpr fp6_e3m2 hi() const noexcept
    {
        return fp6_e3m2::from_bits(static_cast<uint8_t>((data >> 6) & 0x3F));
    }
};

// Static assertions for packed type properties
static_assert(sizeof(fp6x2_e3m2) == sizeof(uint16_t), "fp6x2_e3m2 must be 2 bytes");
static_assert(std::is_trivially_copyable_v<fp6x2_e3m2>, "fp6x2_e3m2 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp6x2_e3m2>, "fp6x2_e3m2 must be standard layout");

// ============================================================================
// Packed FP6 E3M2 Storage Type (4 values in 3 bytes)
// ============================================================================

/**
 * @brief Packed storage type for 4 FP6 E3M2 values in 3 bytes (24 bits)
 *
 * This type provides efficient packed storage for FP6 E3M2 values,
 * storing 4 values in exactly 3 bytes with no padding.
 *
 * Storage Model (24 bits total):
 * - data[0] bits 0-5: element 0 (lo of lo_pair)
 * - data[0] bits 6-7 + data[1] bits 0-3: element 1 (hi of lo_pair)
 * - data[1] bits 4-7 + data[2] bits 0-1: element 2 (lo of hi_pair)
 * - data[2] bits 2-7: element 3 (hi of hi_pair)
 *
 * This is a STORAGE-ONLY type. For value operations, access pairs
 * via lo_pair() and hi_pair() methods.
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp6x4_e3m2
{
    /// Raw bit representation storing 4 packed FP6 values in 3 bytes.
    uint8_t data[3];

    // Default constructor - all elements zero
    constexpr fp6x4_e3m2() noexcept
        : data{0, 0, 0}
    {
    }

    // Copy/move constructors - implicit
    fp6x4_e3m2(const fp6x4_e3m2&) = default;
    fp6x4_e3m2(fp6x4_e3m2&&) noexcept = default;
    fp6x4_e3m2& operator=(const fp6x4_e3m2&) = default;
    fp6x4_e3m2& operator=(fp6x4_e3m2&&) noexcept = default;

    // Two pairs constructor
    constexpr fp6x4_e3m2(fp6x2_e3m2 loPair, fp6x2_e3m2 hiPair) noexcept
        : data{static_cast<uint8_t>(loPair.data),
               static_cast<uint8_t>((loPair.data >> 8) | (hiPair.data << 4)),
               static_cast<uint8_t>(hiPair.data >> 4)}
    {
    }

    // Get lo_pair (elements 0 and 1)
    // NOLINTNEXTLINE(readability-identifier-naming)
    HOST_DEVICE constexpr fp6x2_e3m2 lo_pair() const noexcept
    {
        fp6x2_e3m2 result;
        result.data = static_cast<uint16_t>(data[0] | ((data[1] & 0x0F) << 8));
        return result;
    }

    // Get hi_pair (elements 2 and 3)
    // NOLINTNEXTLINE(readability-identifier-naming)
    HOST_DEVICE constexpr fp6x2_e3m2 hi_pair() const noexcept
    {
        fp6x2_e3m2 result;
        result.data = static_cast<uint16_t>((data[1] >> 4) | (data[2] << 4));
        return result;
    }
};

// Static assertions for packed type properties
static_assert(sizeof(fp6x4_e3m2) == 3, "fp6x4_e3m2 must be 3 bytes");
static_assert(std::is_trivially_copyable_v<fp6x4_e3m2>, "fp6x4_e3m2 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp6x4_e3m2>, "fp6x4_e3m2 must be standard layout");

} // namespace hipdnn_gpu_ref::data_types
