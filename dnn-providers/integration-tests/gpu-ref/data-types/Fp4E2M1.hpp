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
// FP4 E2M1 (OCP MX Format) Bit Layout Constants
// ============================================================================
// See fp4_e2m1 struct documentation for full format specification.
//
// Bit layout: [S|EE|M]
//              3 2 1 0
// ============================================================================

/// Sign bit mask (bit 3)
constexpr uint8_t FP4_E2M1_SIGN_MASK = 0x08;

/// Exponent field mask (bits 1-2)
constexpr uint8_t FP4_E2M1_EXP_MASK = 0x06;

/// Mantissa field mask (bit 0)
constexpr uint8_t FP4_E2M1_MANT_MASK = 0x01;

/// Absolute value mask (bits 0-2)
constexpr uint8_t FP4_E2M1_ABS_MASK = 0x07;

/// Exponent bias for E2M1
constexpr int FP4_E2M1_EXP_BIAS = 1;

/// Number of exponent bits
constexpr int FP4_E2M1_EXP_BITS = 2;

/// Number of mantissa bits
constexpr int FP4_E2M1_MANT_BITS = 1;

// ============================================================================
// FP4 E2M1 Special Values (bit patterns)
// ============================================================================

/// Positive zero
constexpr uint8_t FP4_E2M1_POS_ZERO = 0x00;

/// Negative zero
constexpr uint8_t FP4_E2M1_NEG_ZERO = 0x08;

/// Minimum positive subnormal: 0.5
constexpr uint8_t FP4_E2M1_DENORM_MIN = 0x01;

/// Minimum positive normal: 1.0
constexpr uint8_t FP4_E2M1_MIN_NORMAL = 0x02;

/// Maximum positive value: 6.0
constexpr uint8_t FP4_E2M1_MAX = 0x07;

/// Maximum negative value (lowest): -6.0
constexpr uint8_t FP4_E2M1_LOWEST = 0x0F;

/// Epsilon: smallest difference at 1.0 = 0.5
constexpr uint8_t FP4_E2M1_EPSILON = 0x01;

/// Round error (0.5) - same as epsilon for E2M1 format
constexpr uint8_t FP4_E2M1_ROUND_ERROR = 0x01;

/// Rounding threshold for round-to-nearest-even (midpoint of 22-bit remainder)
constexpr uint32_t FP4_E2M1_ROUND_THRESHOLD = 0x200000;

// Convert float to FP4 E2M1 bits (OCP MX format: 1 sign, 2 exponent, 1 mantissa)
// Range: +/- 6, no infinity (saturates to max), no NaN (returns zero)
// NOLINTNEXTLINE(readability-identifier-naming)
HOST_DEVICE inline uint8_t float_to_fp4_e2m1_bits(float f) noexcept
{
    const uint32_t bits = floatToBits(f);
    const uint32_t absBits = bits & 0x7FFFFFFFu;
    const uint32_t sign = (bits >> 28) & FP4_E2M1_SIGN_MASK; // float sign bit 31 -> bit 3

    // Per OCP MX Specification v1.0, conversion from NaN is implementation-defined.
    // This implementation returns zero.
    if(absBits > 0x7F800000u)
    {
        return static_cast<uint8_t>(FP4_E2M1_POS_ZERO);
    }

    // E2M1 has no infinity - saturate to max
    if(absBits == 0x7F800000u)
    {
        return static_cast<uint8_t>(sign ? FP4_E2M1_LOWEST : FP4_E2M1_MAX);
    }

    // Rebias from float (127) to E2M1 (1)
    const uint32_t fp32Exp = absBits >> 23;
    int32_t exp = static_cast<int32_t>(fp32Exp) - 127 + FP4_E2M1_EXP_BIAS;
    uint32_t mant = bits & 0x007FFFFFu;

    // Handle overflow (exp > 3 means value exceeds max representable)
    if(exp > 3)
    {
        return static_cast<uint8_t>(sign | FP4_E2M1_ABS_MASK); // Saturate to max finite value
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
        // 23 - 1 = 22 bits to shift
        const auto shift = static_cast<uint32_t>(1 - exp + 22);
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
            if(mant > FP4_E2M1_MANT_MASK)
            {
                // Promoted to smallest normal: exp=1, mant=0 -> value 1.0
                return static_cast<uint8_t>(sign | FP4_E2M1_MIN_NORMAL);
            }
        }
        return static_cast<uint8_t>(sign | (mant & FP4_E2M1_MANT_MASK));
    }

    // Normal case: shift mantissa from 23 bits to 1 bit with rounding
    uint32_t fp4Mant = (mant >> 22) & FP4_E2M1_MANT_MASK;
    const uint32_t remainder = mant & 0x003FFFFF;

    // Round to nearest even
    if(remainder > FP4_E2M1_ROUND_THRESHOLD
       || (remainder == FP4_E2M1_ROUND_THRESHOLD && ((fp4Mant & 1) != 0)))
    {
        fp4Mant++;
        if(fp4Mant > 1)
        {
            fp4Mant = 0;
            exp++;
            if(exp > 3)
            {
                return static_cast<uint8_t>(sign | FP4_E2M1_ABS_MASK);
            }
        }
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 1) | fp4Mant);
}

// Convert FP4 E2M1 bits to float
// NOLINTNEXTLINE(readability-identifier-naming)
HOST_DEVICE inline float fp4_e2m1_bits_to_float(uint8_t b) noexcept
{
    const uint32_t sign = static_cast<uint32_t>(b & FP4_E2M1_SIGN_MASK) << 28;
    const uint32_t exp = static_cast<uint32_t>(b & FP4_E2M1_EXP_MASK) >> FP4_E2M1_MANT_BITS;
    const uint32_t mant = static_cast<uint32_t>(b & FP4_E2M1_MANT_MASK);

    uint32_t fp32Bits;
    if(exp == 0)
    {
        // 0 or 0.5 depending on whether the mantissa bit is setng on if mantissa bit is set
        fp32Bits = (mant == 0u) ? sign : (sign | 0x3f000000u);
    }
    else
    {
        // Normal: rebias exponent from 1 to 127 and left-align the 1-bit mantissa.
        fp32Bits = sign | ((exp + (127u - FP4_E2M1_EXP_BIAS)) << 23)
                   | (mant << (23 - FP4_E2M1_MANT_BITS));
    }
    return bitsToFloat(fp32Bits);
}

} // namespace detail

/**
 * @brief Custom storage-only FP4 E2M1 type  on device
 *
 * This type provides a portable OCP MX FP4 E2M1 (1 sign, 2 exponent, 1 mantissa)
 * implementation that does not require the __HIPCC__ macro.
 *
 * This is a STORAGE-ONLY type intended for data representation and conversion,
 * not direct computation. Arithmetic operations and comparisons are
 * intentionally not provided. For computation, explicitly convert to float.
 *
 * E2M1 Format Properties:
 * - Binary layout: 1 sign bit (bit 3), 2 exponent bits (bits 1-2), 1 mantissa bit (bit 0)
 * - Exponent bias: 1
 * - No infinity representation
 * - No NaN representation
 * - Subnormals supported
 *
 * Representable Values:
 * | Bits | Sign | Exp | Mant | Value  |
 * |------|------|-----|------|--------|
 * | 0000 |  +   |  0  |  0   |  +0    |
 * | 0001 |  +   |  0  |  1   |  +0.5  |
 * | 0010 |  +   |  1  |  0   |  +1.0  |
 * | 0011 |  +   |  1  |  1   |  +1.5  |
 * | 0100 |  +   |  2  |  0   |  +2.0  |
 * | 0101 |  +   |  2  |  1   |  +3.0  |
 * | 0110 |  +   |  3  |  0   |  +4.0  |
 * | 0111 |  +   |  3  |  1   |  +6.0  |
 * | 1000 |  -   |  0  |  0   |  -0    |
 * | 1001 |  -   |  0  |  1   |  -0.5  |
 * | 1010 |  -   |  1  |  0   |  -1.0  |
 * | 1011 |  -   |  1  |  1   |  -1.5  |
 * | 1100 |  -   |  2  |  0   |  -2.0  |
 * | 1101 |  -   |  2  |  1   |  -3.0  |
 * | 1110 |  -   |  3  |  0   |  -4.0  |
 * | 1111 |  -   |  3  |  1   |  -6.0  |
 *
 * Range: -6.0 to +6.0
 * Rounding: roundTiesToEven (IEEE 754 default)
 * Overflow: Saturates to ±6 (maximum magnitude)
 * Underflow: Values with magnitude below 0.25 round to zero
 */
// NOLINTNEXTLINE(readability-identifier-naming)
struct fp4_e2m1
{
    /// Raw bit representation of the FP4 E2M1 value.
    uint8_t data;

    // Default constructor - value-initialized to zero for constexpr support
    constexpr fp4_e2m1() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp4_e2m1(const fp4_e2m1&) = default;
    fp4_e2m1(fp4_e2m1&&) noexcept = default;
    fp4_e2m1& operator=(const fp4_e2m1&) = default;
    fp4_e2m1& operator=(fp4_e2m1&&) noexcept = default;

    // EXPLICIT constructor from float
    HOST_DEVICE explicit fp4_e2m1(float f) noexcept
        : data(detail::float_to_fp4_e2m1_bits(f))
    {
    }

    // EXPLICIT constructor from double (via float)
    HOST_DEVICE explicit fp4_e2m1(double d) noexcept
        : fp4_e2m1(static_cast<float>(d))
    {
    }

    HOST_DEVICE explicit fp4_e2m1(_Float16 h) noexcept
        : fp4_e2m1(static_cast<float>(h))
    {
    }

    HOST_DEVICE explicit fp4_e2m1(__bf16 b) noexcept
        : fp4_e2m1(static_cast<float>(b))
    {
    }

    // EXPLICIT constructor from integral types
    template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
    HOST_DEVICE explicit fp4_e2m1(T value) noexcept
        : fp4_e2m1(static_cast<float>(value))
    {
    }

    // Factory for raw bits
    // NOLINTNEXTLINE(readability-identifier-naming)
    HOST_DEVICE static constexpr fp4_e2m1 from_bits(uint8_t bits) noexcept
    {
        fp4_e2m1 val;
        val.data = bits & 0x0F;
        return val;
    }

    // EXPLICIT conversion to float
    HOST_DEVICE explicit operator float() const noexcept
    {
        return detail::fp4_e2m1_bits_to_float(data);
    }

    // EXPLICIT conversion to double
    HOST_DEVICE explicit operator double() const noexcept
    {
        return static_cast<double>(detail::fp4_e2m1_bits_to_float(data));
    }

    HOST_DEVICE explicit operator _Float16() const noexcept
    {
        return static_cast<_Float16>(detail::fp4_e2m1_bits_to_float(data));
    }

    HOST_DEVICE explicit operator __bf16() const noexcept
    {
        return static_cast<__bf16>(detail::fp4_e2m1_bits_to_float(data));
    }

    // Unary negation - flip sign bit
    HOST_DEVICE fp4_e2m1 operator-() const noexcept
    {
        return from_bits(data ^ detail::FP4_E2M1_SIGN_MASK);
    }

    // Unary plus
    HOST_DEVICE fp4_e2m1 operator+() const noexcept
    {
        return *this;
    }
};

// Static assertions for binary compatibility
static_assert(sizeof(fp4_e2m1) == sizeof(uint8_t), "fp4_e2m1 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp4_e2m1>, "fp4_e2m1 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp4_e2m1>, "fp4_e2m1 must be standard layout");
static_assert(std::is_default_constructible_v<fp4_e2m1>, "fp4_e2m1 must be default constructible");
static_assert(std::is_copy_constructible_v<fp4_e2m1>, "fp4_e2m1 must be copy constructible");
static_assert(std::is_move_constructible_v<fp4_e2m1>, "fp4_e2m1 must be move constructible");

// ============================================================================
// Math functions for fp4_e2m1
// ============================================================================
// These are defined in our namespace to enable ADL (Argument Dependent Lookup).
// Use unqualified calls like: isnan(x), isinf(x), etc.
// ============================================================================

HOST_DEVICE inline bool isnan(fp4_e2m1 x)
{
    // FP4 E2M1 has no NaN representation
    return false;
}

HOST_DEVICE inline bool isinf(fp4_e2m1 x)
{
    // FP4 E2M1 has no infinity representation
    return false;
}

HOST_DEVICE inline bool signbit(fp4_e2m1 x)
{
    return (x.data & detail::FP4_E2M1_SIGN_MASK) != 0;
}

HOST_DEVICE inline bool isfinite(fp4_e2m1 x)
{
    // All FP4 E2M1 values are finite (no NaN or Inf)
    return true;
}

// ============================================================================
// Packed FP4 E2M1 Storage Type (2 values per byte)
// ============================================================================

/**
 * @brief Packed storage type for 2 FP4 E2M1 values in a single byte
 *
 * This type provides efficient packed storage for FP4 E2M1 values,
 * storing 2 values per byte.
 *
 * Storage Model:
 * - Low nibble (bits 0-3): lo element
 * - High nibble (bits 4-7): hi element
 *
 * This is a STORAGE-ONLY type. For value operations, access individual
 * elements via lo() and hi() methods.
 */
// NOLINTNEXTLINE(readability-identifier-naming) - lowercase for consistency
struct fp4x2_e2m1
{
    /// Raw bit representation storing 2 packed FP4 values.
    uint8_t data;

    // Default constructor - both elements zero
    constexpr fp4x2_e2m1() noexcept
        : data(0)
    {
    }

    // Copy/move constructors - implicit
    fp4x2_e2m1(const fp4x2_e2m1&) = default;
    fp4x2_e2m1(fp4x2_e2m1&&) noexcept = default;
    fp4x2_e2m1& operator=(const fp4x2_e2m1&) = default;
    fp4x2_e2m1& operator=(fp4x2_e2m1&&) noexcept = default;

    // Single value constructor - lo = val, hi = zero
    constexpr explicit fp4x2_e2m1(fp4_e2m1 lo) noexcept
        : data(lo.data & 0x0F)
    {
    }

    // Two values constructor
    constexpr fp4x2_e2m1(fp4_e2m1 lo, fp4_e2m1 hi) noexcept
        : data(static_cast<uint8_t>((lo.data & 0x0F) | ((hi.data & 0x0F) << 4)))
    {
    }

    // Get lo element (low nibble)
    HOST_DEVICE constexpr fp4_e2m1 lo() const noexcept
    {
        return fp4_e2m1::from_bits(data & 0x0F);
    }

    // Get hi element (high nibble)
    HOST_DEVICE constexpr fp4_e2m1 hi() const noexcept
    {
        return fp4_e2m1::from_bits((data >> 4) & 0x0F);
    }
};

// Static assertions for packed type properties
static_assert(sizeof(fp4x2_e2m1) == sizeof(uint8_t), "fp4x2_e2m1 must be 1 byte");
static_assert(std::is_trivially_copyable_v<fp4x2_e2m1>, "fp4x2_e2m1 must be trivially copyable");
static_assert(std::is_standard_layout_v<fp4x2_e2m1>, "fp4x2_e2m1 must be standard layout");

} // namespace hipdnn_gpu_ref::data_types
