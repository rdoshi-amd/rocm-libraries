// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>

#include "Common.hpp"

using namespace hipdnn_data_sdk::types;
using namespace gpu_ref_device_data_type_test;
using namespace gpu_ref_device_data_type_test::detail;

namespace
{
constexpr size_t FP6_CODE_COUNT = 64;
} // namespace

// ============================================================================
// Exact-value round trip
// ============================================================================

TEST(TestFp6E3M2Device, RoundTripExactValues)
{
    // clang-format off
    const std::vector<float> exactValues = {
        // Positive zero / subnormals
        0.0f,    0.0625f, 0.125f,  0.1875f,
        // Positive normals
        0.25f,   0.3125f, 0.375f,  0.4375f,
        0.5f,    0.625f,  0.75f,   0.875f,
        1.0f,    1.25f,   1.5f,    1.75f,
        2.0f,    2.5f,    3.0f,    3.5f,
        4.0f,    5.0f,    6.0f,    7.0f,
        8.0f,    10.0f,   12.0f,   14.0f,
        16.0f,   20.0f,   24.0f,   28.0f,
        // Negative zero / subnormals
        -0.0f,   -0.0625f, -0.125f, -0.1875f,
        // Negative normals
        -0.25f,  -0.3125f, -0.375f, -0.4375f,
        -0.5f,   -0.625f,  -0.75f,  -0.875f,
        -1.0f,   -1.25f,   -1.5f,   -1.75f,
        -2.0f,   -2.5f,    -3.0f,   -3.5f,
        -4.0f,   -5.0f,    -6.0f,   -7.0f,
        -8.0f,   -10.0f,   -12.0f,  -14.0f,
        -16.0f,  -20.0f,   -24.0f,  -28.0f};
    // clang-format on

    const auto decoded = deviceRoundTrip<fp6_e3m2>(exactValues);
    ASSERT_EQ(decoded.size(), exactValues.size());
    for(size_t i = 0; i < exactValues.size(); ++i)
    {
        EXPECT_EQ(decoded[i], exactValues[i]) << "Round-trip failed for " << exactValues[i];
        EXPECT_EQ(std::signbit(decoded[i]), std::signbit(exactValues[i]))
            << "Sign lost for " << exactValues[i];
    }
}

// ============================================================================
// Rounding Tests
// ============================================================================

namespace
{

struct RoundingTestCase
{
    float input;
    float expected;

    friend std::ostream& operator<<(std::ostream& os, const RoundingTestCase& tc)
    {
        return os << tc.input << " -> " << tc.expected;
    }
};

} // namespace

class TestFp6E3M2DeviceRounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp6E3M2DeviceRounding, Rounding)
{
    auto [input, expected] = GetParam();
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(input), expected);
}

// Midpoints between two representable encodings: ties-to-even
INSTANTIATE_TEST_SUITE_P(MidpointRounding,
                         TestFp6E3M2DeviceRounding,
                         ::testing::Values(RoundingTestCase{0.09375f, 0.125f},
                                           RoundingTestCase{0.15625f, 0.125f},
                                           RoundingTestCase{1.125f, 1.0f},
                                           RoundingTestCase{1.375f, 1.5f},
                                           RoundingTestCase{2.25f, 2.0f},
                                           RoundingTestCase{2.75f, 3.0f},
                                           RoundingTestCase{4.5f, 4.0f},
                                           RoundingTestCase{5.5f, 6.0f},
                                           RoundingTestCase{9.0f, 8.0f},
                                           RoundingTestCase{18.0f, 16.0f}));

// Non-tie boundary cases: verify rounding direction
INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp6E3M2DeviceRounding,
                         ::testing::Values(RoundingTestCase{26.0f, 24.0f},
                                           RoundingTestCase{27.0f, 28.0f},
                                           RoundingTestCase{22.0f, 24.0f},
                                           RoundingTestCase{11.0f, 12.0f},
                                           RoundingTestCase{13.0f, 12.0f},
                                           RoundingTestCase{-26.0f, -24.0f}));

// Subnormal / underflow range. denorm_min = 0.0625; underflow threshold = 0.03125
INSTANTIATE_TEST_SUITE_P(SubnormalRounding,
                         TestFp6E3M2DeviceRounding,
                         ::testing::Values(RoundingTestCase{0.05f, 0.0625f},
                                           RoundingTestCase{0.07f, 0.0625f},
                                           RoundingTestCase{0.1f, 0.125f},
                                           RoundingTestCase{0.03f, 0.0f},
                                           RoundingTestCase{0.03125f, 0.0f}, // tie -> even (zero)
                                           RoundingTestCase{0.16f, 0.1875f},
                                           RoundingTestCase{0.21875f, 0.25f}, // promote to normal
                                           RoundingTestCase{-0.05f, -0.0625f},
                                           RoundingTestCase{-0.1f, -0.125f}));

// Mantissa overflow cascading into the exponent, and rounding into saturation
INSTANTIATE_TEST_SUITE_P(RoundingOverflow,
                         TestFp6E3M2DeviceRounding,
                         ::testing::Values(RoundingTestCase{1.875f, 2.0f}, // e=3 -> e=4
                                           RoundingTestCase{3.75f, 4.0f}, // e=4 -> e=5
                                           RoundingTestCase{7.5f, 8.0f}, // e=5 -> e=6
                                           RoundingTestCase{15.0f, 16.0f}, // e=6 -> e=7
                                           RoundingTestCase{30.0f, 28.0f}, // exp>7 -> saturate
                                           RoundingTestCase{-1.875f, -2.0f},
                                           RoundingTestCase{-7.5f, -8.0f},
                                           RoundingTestCase{-30.0f, -28.0f}));

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp6E3M2Device, SaturationPositive)
{
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(1e10f), 28.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(100.0f), 28.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(30.0f), 28.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(29.0f), 28.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(std::nextafter(28.0f, 1e9f)), 28.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(std::numeric_limits<float>::max()), 28.0f);
}

TEST(TestFp6E3M2Device, SaturationNegative)
{
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(-1e10f), -28.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(-100.0f), -28.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(-30.0f), -28.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(std::numeric_limits<float>::lowest()), -28.0f);
}

TEST(TestFp6E3M2Device, SaturationInfinity)
{
    // E3M2 has no infinity: +/-Inf saturate to +/-MAX
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(std::numeric_limits<float>::infinity()), 28.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(-std::numeric_limits<float>::infinity()), -28.0f);
    EXPECT_EQ(deviceEncodeSingle<fp6_e3m2>(std::numeric_limits<float>::infinity()).data, 0x1F);
    EXPECT_EQ(deviceEncodeSingle<fp6_e3m2>(-std::numeric_limits<float>::infinity()).data, 0x3F);
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp6E3M2Device, Underflow)
{
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(0.03f), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(1e-10f), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(0.03125f), 0.0f); // tie -> even (zero)

    // fp32 subnormal inputs (exercises the shift > 24 path with fp32Exp == 0)
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(std::numeric_limits<float>::denorm_min()), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(-std::numeric_limits<float>::denorm_min()), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp6_e3m2>(1e-40f), 0.0f);

    // Underflow preserves the sign
    EXPECT_EQ(deviceEncodeSingle<fp6_e3m2>(-0.03f).data, 0x20);
    EXPECT_TRUE(std::signbit(deviceRoundTripSingle<fp6_e3m2>(-0.03f)));

    EXPECT_EQ(deviceEncodeSingle<fp6_e3m2>(0.05f).data, static_cast<uint8_t>(0x01)); // denorm_min
    EXPECT_EQ(deviceEncodeSingle<fp6_e3m2>(-0.05f).data, static_cast<uint8_t>(0x21));
    EXPECT_EQ(deviceDecodeSingle<fp6_e3m2>(fp6_e3m2::from_bits(0x01)), 0.0625f);
    EXPECT_EQ(deviceDecodeSingle<fp6_e3m2>(fp6_e3m2::from_bits(0x21)), -0.0625f);
}

// ============================================================================
// Device / CPU equivalence
// ============================================================================

TEST(TestFp6E3M2Device, DecodeMatchesCpuAllPatterns)
{
    std::vector<fp6_e3m2> all(FP6_CODE_COUNT);
    for(size_t b = 0; b < FP6_CODE_COUNT; ++b)
    {
        all[b] = fp6_e3m2::from_bits(static_cast<uint8_t>(b));
    }

    const auto dev = deviceDecode<fp6_e3m2>(all);
    for(size_t b = 0; b < FP6_CODE_COUNT; ++b)
    {
        // Bit-exact comparison so that -0.0 vs +0.0 is caught
        EXPECT_EQ(__builtin_bit_cast(uint32_t, dev[b]),
                  __builtin_bit_cast(uint32_t, static_cast<float>(all[b])))
            << "pattern 0x" << std::hex << b;
    }
}

TEST(TestFp6E3M2Device, EncodeMatchesCpu)
{
    auto bits = [](float f) { return __builtin_bit_cast(uint32_t, f); };
    auto fromBits = [](uint32_t b) { return __builtin_bit_cast(float, b); };

    std::vector<float> in;
    auto addAround = [&](float x) {
        for(int d = -3; d <= 3; ++d)
        {
            const float v = fromBits(bits(x) + static_cast<uint32_t>(d));
            in.push_back(v);
            in.push_back(-v);
        }
    };

    // Every representable magnitude, plus every midpoint between neighbours
    for(int c = 0; c <= 0x1F; ++c)
    {
        const float v = static_cast<float>(fp6_e3m2::from_bits(static_cast<uint8_t>(c)));
        addAround(v);
        if(c < 0x1F)
        {
            addAround((v + static_cast<float>(fp6_e3m2::from_bits(static_cast<uint8_t>(c + 1))))
                      / 2);
        }
    }
    // Midpoint between MAX and the first unrepresentable value
    addAround(30.0f);

    for(const float s : {std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::quiet_NaN(),
                         -std::numeric_limits<float>::quiet_NaN(),
                         fromBits(0x7F800001u), // signalling NaN
                         1e30f,
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::lowest(),
                         std::numeric_limits<float>::denorm_min(),
                         1e-40f,
                         0.0f,
                         -0.0f})
    {
        in.push_back(s);
    }

    const auto dev = deviceEncode<fp6_e3m2>(in);
    ASSERT_EQ(dev.size(), in.size());
    for(size_t i = 0; i < in.size(); ++i)
    {
        EXPECT_EQ(dev[i].data, fp6_e3m2(in[i]).data) << "float bits 0x" << std::hex << bits(in[i]);
    }
}

TEST(TestFp6E3M2Device, KnownValues)
{
    const std::vector<std::pair<float, float>> cases
        = {{1.0f, 1.0f},
           {-28.0f, -28.0f},
           {0.0625f, 0.0625f}, // denorm_min
           {0.25f, 0.25f}, // min normal
           {1.125f, 1.0f}, // ties to even
           {1.375f, 1.5f}, // ties to even
           {1.875f, 2.0f}, // mantissa overflow -> exponent increment
           {30.0f, 28.0f}, // exponent overflow -> saturation
           {1e10f, 28.0f}, // saturation
           {std::numeric_limits<float>::infinity(), 28.0f},
           {-std::numeric_limits<float>::infinity(), -28.0f},
           {std::numeric_limits<float>::quiet_NaN(), 0.0f}, // NaN -> zero (OCP MX)
           {1e-10f, 0.0f},
           {0.03125f, 0.0f}, // underflow tie -> 0
           {std::numeric_limits<float>::denorm_min(), 0.0f}};

    std::vector<float> in(cases.size());
    size_t idx = 0;
    for(const auto& c : cases)
    {
        in[idx++] = c.first;
    }
    const auto out = deviceRoundTrip<fp6_e3m2>(in);
    for(size_t i = 0; i < cases.size(); ++i)
    {
        EXPECT_EQ(out[i], cases[i].second) << "input " << cases[i].first;
    }
    EXPECT_EQ(deviceEncode<fp6_e3m2>({-0.0f})[0].data, 0x20); // signed zero
}

// ============================================================================
// E3M2-OCP-Specific: no Infinity, no NaN, Signed Zero, Bit-Pattern Round-Trip
// ============================================================================

TEST(TestFp6E3M2Device, RoundTripAllPatterns)
{
    std::vector<fp6_e3m2> all(FP6_CODE_COUNT);
    for(size_t i = 0; i < FP6_CODE_COUNT; ++i)
    {
        all[i] = fp6_e3m2::from_bits(static_cast<uint8_t>(i));
    }

    const auto decoded = deviceDecode<fp6_e3m2>(all);
    const auto reencoded = deviceEncode<fp6_e3m2>(decoded);
    ASSERT_EQ(decoded.size(), FP6_CODE_COUNT);
    ASSERT_EQ(reencoded.size(), FP6_CODE_COUNT);

    for(size_t bits = 0; bits < FP6_CODE_COUNT; ++bits)
    {
        const auto pattern = static_cast<uint8_t>(bits);
        // E3M2 has neither NaN nor Inf: every code is finite and round-trips exactly
        EXPECT_TRUE(std::isfinite(decoded[bits]))
            << "Pattern 0x" << std::hex << bits << " should be finite";
        EXPECT_EQ(reencoded[bits].data, pattern)
            << "Round-trip failed for bit pattern 0x" << std::hex << bits;
    }
}

TEST(TestFp6E3M2Device, NanConvertsToZero)
{
    // Per OCP MX Specification v1.0, conversion from NaN is implementation-defined;
    // this implementation returns positive zero for both NaN signs.
    EXPECT_EQ(deviceEncodeSingle<fp6_e3m2>(std::numeric_limits<float>::quiet_NaN()).data, 0x00);
    EXPECT_EQ(deviceEncodeSingle<fp6_e3m2>(-std::numeric_limits<float>::quiet_NaN()).data, 0x00);
}

TEST(TestFp6E3M2Device, SignedZeroSupported)
{
    EXPECT_EQ(deviceEncodeSingle<fp6_e3m2>(0.0f).data, static_cast<uint8_t>(0x00));
    EXPECT_EQ(deviceEncodeSingle<fp6_e3m2>(-0.0f).data, static_cast<uint8_t>(0x20));
    const float negZero = deviceDecodeSingle<fp6_e3m2>(fp6_e3m2::from_bits(0x20));
    EXPECT_EQ(negZero, -0.0f);
    EXPECT_TRUE(std::signbit(negZero));
}

TEST(TestFp6E3M2Device, NegateZeroProducesNegativeZero)
{
    const auto neg = deviceNegate<fp6_e3m2>({fp6_e3m2::from_bits(0x00)});
    EXPECT_EQ(neg[0].data, 0x20);
    const auto flags = deviceQueryState<fp6_e3m2>(neg);
    EXPECT_FALSE(flags[0].nan);
    EXPECT_TRUE(flags[0].signbit);
}

TEST(TestFp6E3M2Device, NegateFlipsSignForAllPatterns)
{
    std::vector<fp6_e3m2> all(FP6_CODE_COUNT);
    for(size_t i = 0; i < FP6_CODE_COUNT; ++i)
    {
        all[i] = fp6_e3m2::from_bits(static_cast<uint8_t>(i));
    }

    const auto neg = deviceNegate<fp6_e3m2>(all);
    const auto flags = deviceQueryState<fp6_e3m2>(neg);
    for(size_t i = 0; i < FP6_CODE_COUNT; ++i)
    {
        EXPECT_EQ(neg[i].data, static_cast<uint8_t>(i ^ 0x20u))
            << "negate wrong for 0x" << std::hex << i;
        EXPECT_TRUE(flags[i].finite);
        EXPECT_EQ(flags[i].signbit, (i & 0x20u) == 0);
    }
}

TEST(TestFp6E3M2Device, NegateMaxSwapsWithLowest)
{
    const auto neg = deviceNegate<fp6_e3m2>({fp6_e3m2::from_bits(0x1F), fp6_e3m2::from_bits(0x3F)});
    EXPECT_EQ(neg[0].data, 0x3F);
    EXPECT_EQ(neg[1].data, 0x1F);
    const auto flags = deviceQueryState<fp6_e3m2>(neg);
    EXPECT_TRUE(flags[0].finite);
    EXPECT_TRUE(flags[0].signbit);
    EXPECT_TRUE(flags[1].finite);
    EXPECT_FALSE(flags[1].signbit);
}

TEST(TestFp6E3M2Device, ClassificationTruthTable)
{
    std::vector<fp6_e3m2> all(FP6_CODE_COUNT);
    for(size_t i = 0; i < FP6_CODE_COUNT; ++i)
    {
        all[i] = fp6_e3m2::from_bits(static_cast<uint8_t>(i));
    }

    const auto flags = deviceQueryState<fp6_e3m2>(all);
    for(size_t bits = 0; bits < FP6_CODE_COUNT; ++bits)
    {
        // E3M2 has no NaN and no infinity: every code is finite
        EXPECT_FALSE(flags[bits].nan) << "isnan wrong for 0x" << std::hex << bits;
        EXPECT_FALSE(flags[bits].inf) << "isinf wrong for 0x" << std::hex << bits;
        EXPECT_TRUE(flags[bits].finite) << "isfinite wrong for 0x" << std::hex << bits;
        EXPECT_EQ(flags[bits].signbit, (bits & 0x20u) != 0)
            << "signbit wrong for 0x" << std::hex << bits;
    }
}

// ============================================================================
// Fp16 and Bf16 casting tests
// ============================================================================

TEST(TestFp6E3M2Device, Fp16ToFp6E3M2RoundTrip)
{
    std::vector<half> input;
    input.reserve(65536);
    for(uint32_t b = 0; b < 65536; ++b)
    {
        input.push_back(half::from_bits(static_cast<uint16_t>(b)));
    }

    std::vector<fp6_e3m2> encoded(input.size());
    deviceDataCast(input, encoded);

    std::vector<half> decoded(encoded.size());
    deviceDataCast(encoded, decoded);

    for(size_t i = 0; i < input.size(); ++i)
    {
        const fp6_e3m2 cpuEncoded(static_cast<float>(input[i]));
        ASSERT_EQ(encoded[i].data, cpuEncoded.data) << "fp16 pattern 0x" << std::hex << i;

        // E3M2 never decodes to NaN, so a bit-exact comparison always applies
        ASSERT_EQ(__builtin_bit_cast(uint32_t, static_cast<float>(decoded[i])),
                  __builtin_bit_cast(uint32_t, static_cast<float>(cpuEncoded)))
            << "fp16 pattern 0x" << std::hex << i;
    }
}

TEST(TestFp6E3M2Device, Bf16ToFp6E3M2RoundTrip)
{
    std::vector<bfloat16> input;
    input.reserve(65536);
    for(uint32_t b = 0; b < 65536; ++b)
    {
        input.push_back(bfloat16::from_bits(static_cast<uint16_t>(b)));
    }

    std::vector<fp6_e3m2> encoded(input.size());
    deviceDataCast(input, encoded);

    std::vector<bfloat16> decoded(encoded.size());
    deviceDataCast(encoded, decoded);

    for(size_t i = 0; i < input.size(); ++i)
    {
        const fp6_e3m2 cpuEncoded(static_cast<float>(input[i]));
        ASSERT_EQ(encoded[i].data, cpuEncoded.data) << "bf16 pattern 0x" << std::hex << i;

        ASSERT_EQ(__builtin_bit_cast(uint32_t, static_cast<float>(decoded[i])),
                  __builtin_bit_cast(uint32_t, static_cast<float>(cpuEncoded)))
            << "bf16 pattern 0x" << std::hex << i;
    }
}

TEST(TestFp6E3M2Device, Fp6E3M2ToFp16AllPatterns)
{
    std::vector<fp6_e3m2> input;
    input.reserve(FP6_CODE_COUNT);
    for(uint32_t b = 0; b < FP6_CODE_COUNT; ++b)
    {
        input.push_back(fp6_e3m2::from_bits(static_cast<uint8_t>(b)));
    }

    std::vector<half> output(input.size());
    deviceDataCast(input, output);

    for(size_t i = 0; i < input.size(); ++i)
    {
        // Every E3M2 value is exactly representable in fp16
        ASSERT_EQ(__builtin_bit_cast(uint32_t, static_cast<float>(output[i])),
                  __builtin_bit_cast(uint32_t, static_cast<float>(input[i])))
            << "fp6 pattern 0x" << std::hex << i;
    }
}

TEST(TestFp6E3M2Device, Fp6E3M2ToBf16AllPatterns)
{
    std::vector<fp6_e3m2> input;
    input.reserve(FP6_CODE_COUNT);
    for(uint32_t b = 0; b < FP6_CODE_COUNT; ++b)
    {
        input.push_back(fp6_e3m2::from_bits(static_cast<uint8_t>(b)));
    }

    std::vector<bfloat16> output(input.size());
    deviceDataCast(input, output);

    for(size_t i = 0; i < input.size(); ++i)
    {
        // Every E3M2 value is exactly representable in bf16
        ASSERT_EQ(__builtin_bit_cast(uint32_t, static_cast<float>(output[i])),
                  __builtin_bit_cast(uint32_t, static_cast<float>(input[i])))
            << "fp6 pattern 0x" << std::hex << i;
    }
}
