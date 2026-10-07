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
constexpr size_t FP4_CODE_COUNT = 16;
} // namespace

// ============================================================================
// Exact-value round trip
// ============================================================================

TEST(TestFp4E2M1Device, RoundTripExactValues)
{
    // All 16 E2M1 codes are exactly representable in float
    const std::vector<float> exactValues = {0.0f,
                                            0.5f,
                                            1.0f,
                                            1.5f,
                                            2.0f,
                                            3.0f,
                                            4.0f,
                                            6.0f,
                                            -0.0f,
                                            -0.5f,
                                            -1.0f,
                                            -1.5f,
                                            -2.0f,
                                            -3.0f,
                                            -4.0f,
                                            -6.0f};

    const auto decoded = deviceRoundTrip<fp4_e2m1>(exactValues);
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

class TestFp4E2M1DeviceRounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp4E2M1DeviceRounding, Rounding)
{
    auto [input, expected] = GetParam();
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(input), expected);
}

// Midpoints between two representable encodings: ties-to-even
INSTANTIATE_TEST_SUITE_P(MidpointRounding,
                         TestFp4E2M1DeviceRounding,
                         ::testing::Values(RoundingTestCase{0.25f, 0.0f}, // 0 (even) vs 0.5
                                           RoundingTestCase{0.75f, 1.0f}, // 0.5 vs 1.0 (even)
                                           RoundingTestCase{1.25f, 1.0f}, // 1.0 (even) vs 1.5
                                           RoundingTestCase{2.5f, 2.0f}, // 2.0 (even) vs 3.0
                                           RoundingTestCase{5.0f, 4.0f})); // 4.0 (even) vs 6.0

// Non-tie boundary cases: verify rounding direction
INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp4E2M1DeviceRounding,
                         ::testing::Values(RoundingTestCase{5.5f, 6.0f},
                                           RoundingTestCase{5.1f, 6.0f},
                                           RoundingTestCase{3.5f, 4.0f},
                                           RoundingTestCase{2.25f, 2.0f},
                                           RoundingTestCase{2.75f, 3.0f},
                                           RoundingTestCase{-5.5f, -6.0f}));

// Subnormal / underflow range. denorm_min = 0.5; underflow threshold = 0.25
INSTANTIATE_TEST_SUITE_P(SubnormalRounding,
                         TestFp4E2M1DeviceRounding,
                         ::testing::Values(RoundingTestCase{0.3f, 0.5f},
                                           RoundingTestCase{0.4f, 0.5f},
                                           RoundingTestCase{0.24f, 0.0f},
                                           RoundingTestCase{0.6f, 0.5f},
                                           RoundingTestCase{-0.3f, -0.5f},
                                           RoundingTestCase{-0.4f, -0.5f}));

// Mantissa overflow cascading into the exponent, and rounding into saturation
INSTANTIATE_TEST_SUITE_P(RoundingOverflow,
                         TestFp4E2M1DeviceRounding,
                         ::testing::Values(RoundingTestCase{1.75f, 2.0f},
                                           RoundingTestCase{-1.75f, -2.0f},
                                           RoundingTestCase{7.0f, 6.0f}, // exp overflow -> MAX
                                           RoundingTestCase{-7.0f, -6.0f}));

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp4E2M1Device, SaturationPositive)
{
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(1e10f), 6.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(100.0f), 6.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(7.0f), 6.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(6.5f), 6.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(std::nextafter(6.0f, 1e9f)), 6.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(std::numeric_limits<float>::max()), 6.0f);
}

TEST(TestFp4E2M1Device, SaturationNegative)
{
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(-1e10f), -6.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(-100.0f), -6.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(-7.0f), -6.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(std::numeric_limits<float>::lowest()), -6.0f);
}

TEST(TestFp4E2M1Device, SaturationInfinity)
{
    // E2M1 has no infinity: +/-Inf saturate to +/-MAX
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(std::numeric_limits<float>::infinity()), 6.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(-std::numeric_limits<float>::infinity()), -6.0f);
    EXPECT_EQ(deviceEncodeSingle<fp4_e2m1>(std::numeric_limits<float>::infinity()).data, 0x07);
    EXPECT_EQ(deviceEncodeSingle<fp4_e2m1>(-std::numeric_limits<float>::infinity()).data, 0x0F);
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp4E2M1Device, Underflow)
{
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(0.1f), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(1e-10f), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(0.25f), 0.0f); // tie -> even (zero)

    // fp32 subnormal inputs (exercises the shift > 24 path with fp32Exp == 0)
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(std::numeric_limits<float>::denorm_min()), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(-std::numeric_limits<float>::denorm_min()), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp4_e2m1>(1e-40f), 0.0f);

    // Underflow preserves the sign
    EXPECT_EQ(deviceEncodeSingle<fp4_e2m1>(-0.1f).data, 0x08);
    EXPECT_TRUE(std::signbit(deviceRoundTripSingle<fp4_e2m1>(-0.1f)));

    EXPECT_EQ(deviceEncodeSingle<fp4_e2m1>(0.3f).data, 0x01); // denorm_min
    EXPECT_EQ(deviceEncodeSingle<fp4_e2m1>(-0.3f).data, 0x09);
    EXPECT_EQ(deviceDecodeSingle<fp4_e2m1>(fp4_e2m1::from_bits(0x01)), 0.5f);
    EXPECT_EQ(deviceDecodeSingle<fp4_e2m1>(fp4_e2m1::from_bits(0x09)), -0.5f);
}

// ============================================================================
// Device / CPU equivalence
// ============================================================================

TEST(TestFp4E2M1Device, DecodeMatchesCpuAllPatterns)
{
    std::vector<fp4_e2m1> all(FP4_CODE_COUNT);
    for(size_t b = 0; b < FP4_CODE_COUNT; ++b)
    {
        all[b] = fp4_e2m1::from_bits(static_cast<uint8_t>(b));
    }

    const auto dev = deviceDecode<fp4_e2m1>(all);
    for(size_t b = 0; b < FP4_CODE_COUNT; ++b)
    {
        // Bit-exact comparison so that -0.0 vs +0.0 is caught
        EXPECT_EQ(__builtin_bit_cast(uint32_t, dev[b]),
                  __builtin_bit_cast(uint32_t, static_cast<float>(all[b])))
            << "pattern 0x" << std::hex << b;
    }
}

TEST(TestFp4E2M1Device, EncodeMatchesCpu)
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
    for(int c = 0; c <= 0x07; ++c)
    {
        const float v = static_cast<float>(fp4_e2m1::from_bits(static_cast<uint8_t>(c)));
        addAround(v);
        if(c < 0x07)
        {
            addAround((v + static_cast<float>(fp4_e2m1::from_bits(static_cast<uint8_t>(c + 1))))
                      / 2);
        }
    }
    // Midpoint between MAX and the first unrepresentable power of two
    addAround(7.0f);

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

    const auto dev = deviceEncode<fp4_e2m1>(in);
    ASSERT_EQ(dev.size(), in.size());
    for(size_t i = 0; i < in.size(); ++i)
    {
        EXPECT_EQ(dev[i].data, fp4_e2m1(in[i]).data) << "float bits 0x" << std::hex << bits(in[i]);
    }
}

TEST(TestFp4E2M1Device, KnownValues)
{
    const std::vector<std::pair<float, float>> cases
        = {{1.0f, 1.0f},
           {-6.0f, -6.0f},
           {0.5f, 0.5f}, // denorm_min
           {1.25f, 1.0f}, // ties to even
           {2.5f, 2.0f}, // ties to even
           {1.75f, 2.0f}, // mantissa overflow -> exponent increment
           {7.0f, 6.0f}, // exponent overflow -> saturation
           {1e10f, 6.0f}, // saturation
           {std::numeric_limits<float>::infinity(), 6.0f},
           {-std::numeric_limits<float>::infinity(), -6.0f},
           {std::numeric_limits<float>::quiet_NaN(), 0.0f}, // NaN -> zero (OCP MX)
           {1e-10f, 0.0f},
           {0.25f, 0.0f}, // underflow tie -> 0
           {std::numeric_limits<float>::denorm_min(), 0.0f}};

    std::vector<float> in(cases.size());
    size_t idx = 0;
    for(const auto& c : cases)
    {
        in[idx++] = c.first;
    }
    const auto out = deviceRoundTrip<fp4_e2m1>(in);
    for(size_t i = 0; i < cases.size(); ++i)
    {
        EXPECT_EQ(out[i], cases[i].second) << "input " << cases[i].first;
    }
    EXPECT_EQ(deviceEncode<fp4_e2m1>({-0.0f})[0].data, 0x08); // signed zero
}

// ============================================================================
// E2M1-OCP-Specific: no Infinity, no NaN, Signed Zero, Bit-Pattern Round-Trip
// ============================================================================

TEST(TestFp4E2M1Device, RoundTripAllPatterns)
{
    std::vector<fp4_e2m1> all(FP4_CODE_COUNT);
    for(size_t i = 0; i < FP4_CODE_COUNT; ++i)
    {
        all[i] = fp4_e2m1::from_bits(static_cast<uint8_t>(i));
    }

    const auto decoded = deviceDecode<fp4_e2m1>(all);
    const auto reencoded = deviceEncode<fp4_e2m1>(decoded);
    ASSERT_EQ(decoded.size(), FP4_CODE_COUNT);
    ASSERT_EQ(reencoded.size(), FP4_CODE_COUNT);

    for(size_t bits = 0; bits < FP4_CODE_COUNT; ++bits)
    {
        const auto pattern = static_cast<uint8_t>(bits);
        // E2M1 has neither NaN nor Inf: every code is finite and round-trips exactly
        EXPECT_TRUE(std::isfinite(decoded[bits]))
            << "Pattern 0x" << std::hex << bits << " should be finite";
        EXPECT_EQ(reencoded[bits].data, pattern)
            << "Round-trip failed for bit pattern 0x" << std::hex << bits;
    }
}

TEST(TestFp4E2M1Device, NanConvertsToZero)
{
    // Per OCP MX Specification v1.0, conversion from NaN is implementation-defined;
    // this implementation returns positive zero for both NaN signs.
    EXPECT_EQ(deviceEncodeSingle<fp4_e2m1>(std::numeric_limits<float>::quiet_NaN()).data, 0x00);
    EXPECT_EQ(deviceEncodeSingle<fp4_e2m1>(-std::numeric_limits<float>::quiet_NaN()).data, 0x00);
}

TEST(TestFp4E2M1Device, SignedZeroSupported)
{
    EXPECT_EQ(deviceEncodeSingle<fp4_e2m1>(0.0f).data, static_cast<uint8_t>(0x00));
    EXPECT_EQ(deviceEncodeSingle<fp4_e2m1>(-0.0f).data, static_cast<uint8_t>(0x08));
    const float negZero = deviceDecodeSingle<fp4_e2m1>(fp4_e2m1::from_bits(0x08));
    EXPECT_EQ(negZero, -0.0f);
    EXPECT_TRUE(std::signbit(negZero));
}

TEST(TestFp4E2M1Device, NegateZeroProducesNegativeZero)
{
    const auto neg = deviceNegate<fp4_e2m1>({fp4_e2m1::from_bits(0x00)});
    EXPECT_EQ(neg[0].data, 0x08);
    const auto flags = deviceQueryState<fp4_e2m1>(neg);
    EXPECT_FALSE(flags[0].nan);
    EXPECT_TRUE(flags[0].signbit);
}

TEST(TestFp4E2M1Device, NegateFlipsSignForAllPatterns)
{
    std::vector<fp4_e2m1> all(FP4_CODE_COUNT);
    for(size_t i = 0; i < FP4_CODE_COUNT; ++i)
    {
        all[i] = fp4_e2m1::from_bits(static_cast<uint8_t>(i));
    }

    const auto neg = deviceNegate<fp4_e2m1>(all);
    const auto flags = deviceQueryState<fp4_e2m1>(neg);
    for(size_t i = 0; i < FP4_CODE_COUNT; ++i)
    {
        EXPECT_EQ(neg[i].data, static_cast<uint8_t>(i ^ 0x08u))
            << "negate wrong for 0x" << std::hex << i;
        EXPECT_TRUE(flags[i].finite);
        EXPECT_EQ(flags[i].signbit, (i & 0x08u) == 0);
    }
}

TEST(TestFp4E2M1Device, NegateMaxSwapsWithLowest)
{
    const auto neg = deviceNegate<fp4_e2m1>({fp4_e2m1::from_bits(0x07), fp4_e2m1::from_bits(0x0F)});
    EXPECT_EQ(neg[0].data, 0x0F);
    EXPECT_EQ(neg[1].data, 0x07);
    const auto flags = deviceQueryState<fp4_e2m1>(neg);
    EXPECT_TRUE(flags[0].finite);
    EXPECT_TRUE(flags[0].signbit);
    EXPECT_TRUE(flags[1].finite);
    EXPECT_FALSE(flags[1].signbit);
}

TEST(TestFp4E2M1Device, ClassificationTruthTable)
{
    std::vector<fp4_e2m1> all(FP4_CODE_COUNT);
    for(size_t i = 0; i < FP4_CODE_COUNT; ++i)
    {
        all[i] = fp4_e2m1::from_bits(static_cast<uint8_t>(i));
    }

    const auto flags = deviceQueryState<fp4_e2m1>(all);
    for(size_t bits = 0; bits < FP4_CODE_COUNT; ++bits)
    {
        // E2M1 has no NaN and no infinity: every code is finite
        EXPECT_FALSE(flags[bits].nan) << "isnan wrong for 0x" << std::hex << bits;
        EXPECT_FALSE(flags[bits].inf) << "isinf wrong for 0x" << std::hex << bits;
        EXPECT_TRUE(flags[bits].finite) << "isfinite wrong for 0x" << std::hex << bits;
        EXPECT_EQ(flags[bits].signbit, (bits & 0x08u) != 0)
            << "signbit wrong for 0x" << std::hex << bits;
    }
}

// ============================================================================
// Fp16 and Bf16 casting tests
// ============================================================================

TEST(TestFp4E2M1Device, Fp16ToFp4E2M1RoundTrip)
{
    std::vector<half> input;
    input.reserve(65536);
    for(uint32_t b = 0; b < 65536; ++b)
    {
        input.push_back(half::from_bits(static_cast<uint16_t>(b)));
    }

    std::vector<fp4_e2m1> encoded(input.size());
    deviceDataCast(input, encoded);

    std::vector<half> decoded(encoded.size());
    deviceDataCast(encoded, decoded);

    for(size_t i = 0; i < input.size(); ++i)
    {
        const fp4_e2m1 cpuEncoded(static_cast<float>(input[i]));
        ASSERT_EQ(encoded[i].data, cpuEncoded.data) << "fp16 pattern 0x" << std::hex << i;

        // E2M1 never decodes to NaN, so a bit-exact comparison always applies
        ASSERT_EQ(__builtin_bit_cast(uint32_t, static_cast<float>(decoded[i])),
                  __builtin_bit_cast(uint32_t, static_cast<float>(cpuEncoded)))
            << "fp16 pattern 0x" << std::hex << i;
    }
}

TEST(TestFp4E2M1Device, Bf16ToFp4E2M1RoundTrip)
{
    std::vector<bfloat16> input;
    input.reserve(65536);
    for(uint32_t b = 0; b < 65536; ++b)
    {
        input.push_back(bfloat16::from_bits(static_cast<uint16_t>(b)));
    }

    std::vector<fp4_e2m1> encoded(input.size());
    deviceDataCast(input, encoded);

    std::vector<bfloat16> decoded(encoded.size());
    deviceDataCast(encoded, decoded);

    for(size_t i = 0; i < input.size(); ++i)
    {
        const fp4_e2m1 cpuEncoded(static_cast<float>(input[i]));
        ASSERT_EQ(encoded[i].data, cpuEncoded.data) << "bf16 pattern 0x" << std::hex << i;

        ASSERT_EQ(__builtin_bit_cast(uint32_t, static_cast<float>(decoded[i])),
                  __builtin_bit_cast(uint32_t, static_cast<float>(cpuEncoded)))
            << "bf16 pattern 0x" << std::hex << i;
    }
}

TEST(TestFp4E2M1Device, Fp4E2M1ToFp16AllPatterns)
{
    std::vector<fp4_e2m1> input;
    input.reserve(FP4_CODE_COUNT);
    for(uint32_t b = 0; b < FP4_CODE_COUNT; ++b)
    {
        input.push_back(fp4_e2m1::from_bits(static_cast<uint8_t>(b)));
    }

    std::vector<half> output(input.size());
    deviceDataCast(input, output);

    for(size_t i = 0; i < input.size(); ++i)
    {
        // Every E2M1 value is exactly representable in fp16
        ASSERT_EQ(__builtin_bit_cast(uint32_t, static_cast<float>(output[i])),
                  __builtin_bit_cast(uint32_t, static_cast<float>(input[i])))
            << "fp4 pattern 0x" << std::hex << i;
    }
}

TEST(TestFp4E2M1Device, Fp4E2M1ToBf16AllPatterns)
{
    std::vector<fp4_e2m1> input;
    input.reserve(FP4_CODE_COUNT);
    for(uint32_t b = 0; b < FP4_CODE_COUNT; ++b)
    {
        input.push_back(fp4_e2m1::from_bits(static_cast<uint8_t>(b)));
    }

    std::vector<bfloat16> output(input.size());
    deviceDataCast(input, output);

    for(size_t i = 0; i < input.size(); ++i)
    {
        // Every E2M1 value is exactly representable in bf16
        ASSERT_EQ(__builtin_bit_cast(uint32_t, static_cast<float>(output[i])),
                  __builtin_bit_cast(uint32_t, static_cast<float>(input[i])))
            << "fp4 pattern 0x" << std::hex << i;
    }
}
