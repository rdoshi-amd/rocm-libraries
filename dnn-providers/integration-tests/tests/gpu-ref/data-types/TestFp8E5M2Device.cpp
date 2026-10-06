// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>

#include "Common.hpp"

using namespace hipdnn_data_sdk::types;
using namespace gpu_ref_device_data_type_test;
using namespace gpu_ref_device_data_type_test::detail;

// ============================================================================
// Exact-value round trip
// ============================================================================

TEST(TestFp8E5M2Device, RoundTripExactValues)
{
    const std::vector<float> exactValues = {
        0.0f,
        1.0f,
        -1.0f,
        2.0f,
        -2.0f,
        4.0f,
        -4.0f,
        8.0f,
        -8.0f,
        0.5f,
        -0.5f,
        0.25f,
        -0.25f,
        57344.0f, // MAX
        -57344.0f, // LOWEST
        6.103515625e-5f, // MIN_NORMAL = 2^-14
        -6.103515625e-5f,
        1.52587890625e-5f, // denorm_min = 2^-16
        -1.52587890625e-5f,
    };

    const auto decoded = deviceRoundTrip<fp8_e5m2>(exactValues);
    ASSERT_EQ(decoded.size(), exactValues.size());
    for(size_t i = 0; i < exactValues.size(); ++i)
    {
        EXPECT_EQ(decoded[i], exactValues[i]) << "Round-trip failed for " << exactValues[i];
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

class TestFp8E5M2DeviceRounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp8E5M2DeviceRounding, Rounding)
{
    auto [input, expected] = GetParam();
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(input), expected);
}

// Midpoints between two representable encodings: ties-to-even
INSTANTIATE_TEST_SUITE_P(MidpointRounding,
                         TestFp8E5M2DeviceRounding,
                         ::testing::Values(
                             // 0x40 (mant=0, even) vs 0x41 (odd): midpoint 1.125 -> 1.0
                             RoundingTestCase{1.125f, 1.0f},
                             // 0x41 (odd) vs 0x42 (even): midpoint 1.375 -> 1.5
                             RoundingTestCase{1.375f, 1.5f},
                             // 0x7A (even) vs 0x7B (odd): midpoint 53248 -> 49152
                             RoundingTestCase{53248.0f, 49152.0f}));

// Subnormal / underflow range. denorm_min = 2^-16; underflow threshold = 2^-17
INSTANTIATE_TEST_SUITE_P(
    SubnormalRounding,
    TestFp8E5M2DeviceRounding,
    ::testing::Values(RoundingTestCase{1e-6f, 0.0f},
                      RoundingTestCase{7.62939453125e-6f, 0.0f},
                      RoundingTestCase{8e-6f, 1.52587890625e-5f},
                      RoundingTestCase{2.288818359375e-5f, 3.0517578125e-5f},
                      RoundingTestCase{3.814697265625e-5f, 3.0517578125e-5f},
                      RoundingTestCase{4.96246337890625e-5f, 4.57763671875e-5f},
                      RoundingTestCase{5.340576171875e-5f, 6.103515625e-5f},
                      RoundingTestCase{5.7220458984375e-5f, 6.103515625e-5f},
                      RoundingTestCase{-5.7220458984375e-5f, -6.103515625e-5f}));

// Non-tie boundary cases: verify rounding direction
INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp8E5M2DeviceRounding,
                         ::testing::Values(RoundingTestCase{1.1f, 1.0f},
                                           RoundingTestCase{1.15f, 1.25f},
                                           RoundingTestCase{50000.0f, 49152.0f},
                                           RoundingTestCase{56000.0f, 57344.0f},
                                           RoundingTestCase{-1.1f, -1.0f}));

// Mantissa overflow cascading into the exponent, and rounding into saturation
INSTANTIATE_TEST_SUITE_P(RoundingOverflow,
                         TestFp8E5M2DeviceRounding,
                         ::testing::Values(RoundingTestCase{1.875f, 2.0f},
                                           RoundingTestCase{61440.0f, 57344.0f},
                                           RoundingTestCase{-1.875f, -2.0f},
                                           RoundingTestCase{-61440.0f, -57344.0f}));

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp8E5M2Device, SaturationPositive)
{
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(1e10f), 57344.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(60000.0f), 57344.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(57345.0f), 57344.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(std::nextafter(57344.0f, 1e9f)), 57344.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(61440.0f), 57344.0f);
}

TEST(TestFp8E5M2Device, SaturationNegative)
{
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(-1e10f), -57344.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(-60000.0f), -57344.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(-57345.0f), -57344.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(-61440.0f), -57344.0f);
}

TEST(TestFp8E5M2Device, SaturationInfinity)
{
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(std::numeric_limits<float>::infinity()), 57344.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(-std::numeric_limits<float>::infinity()), -57344.0f);
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp8E5M2Device, Underflow)
{
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(1e-10f), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(-1e-10f), 0.0f);

    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(std::numeric_limits<float>::denorm_min()), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(-std::numeric_limits<float>::denorm_min()), 0.0f);

    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(1e-6f), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e5m2>(7.62939453125e-6f), 0.0f); // tie -> even (zero)

    EXPECT_EQ(deviceEncodeSingle<fp8_e5m2>(1.5f * 7.62939453125e-6f).data,
              static_cast<uint8_t>(0x01));
    EXPECT_EQ(deviceEncodeSingle<fp8_e5m2>(-1.5f * 7.62939453125e-6f).data,
              static_cast<uint8_t>(0x81));
    EXPECT_EQ(deviceDecodeSingle<fp8_e5m2>(fp8_e5m2::from_bits(0x01)), 1.52587890625e-5f);
    EXPECT_EQ(deviceDecodeSingle<fp8_e5m2>(fp8_e5m2::from_bits(0x81)), -1.52587890625e-5f);
}

TEST(TestFp8E5M2Device, DecodeMatchesCpuAllPatterns)
{
    std::vector<fp8_e5m2> all(256);
    for(size_t b = 0; b < 256; ++b)
    {
        all[b] = fp8_e5m2::from_bits(static_cast<uint8_t>(b));
    }

    const auto dev = deviceDecode<fp8_e5m2>(all);
    for(size_t b = 0; b < 256; ++b)
    {
        EXPECT_EQ(__builtin_bit_cast(uint32_t, dev[b]),
                  __builtin_bit_cast(uint32_t, static_cast<float>(all[b])))
            << "pattern 0x" << std::hex << b;
    }
}

TEST(TestFp8E5M2Device, EncodeMatchesCpu)
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
    for(int c = 0; c < 0x7C; ++c)
    {
        const float v = static_cast<float>(fp8_e5m2::from_bits(static_cast<uint8_t>(c)));
        addAround(v);
        if(c + 1 < 0x7C)
        {
            addAround((v + static_cast<float>(fp8_e5m2::from_bits(static_cast<uint8_t>(c + 1))))
                      / 2);
        }
    }
    for(const float s : {std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::quiet_NaN(),
                         -std::numeric_limits<float>::quiet_NaN(),
                         fromBits(0x7F800001u),
                         1e30f,
                         std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::denorm_min(),
                         61440.0f,
                         0.0f,
                         -0.0f})
    {
        in.push_back(s);
    }

    const auto dev = deviceEncode<fp8_e5m2>(in);
    for(size_t i = 0; i < in.size(); ++i)
    {
        EXPECT_EQ(dev[i].data, fp8_e5m2(in[i]).data) << "float bits 0x" << std::hex << bits(in[i]);
    }
}

TEST(TestFp8E5M2Device, KnownValues)
{
    const std::vector<std::pair<float, float>> cases
        = {{1.0f, 1.0f},
           {-57344.0f, -57344.0f},
           {1.52587890625e-5f, 1.52587890625e-5f},
           {1.125f, 1.0f},
           {1.375f, 1.5f}, // ties to even
           {61440.0f, 57344.0f},
           {1e10f, 57344.0f}, // saturation
           {std::numeric_limits<float>::infinity(), 57344.0f},
           {-std::numeric_limits<float>::infinity(), -57344.0f},
           {1e-10f, 0.0f},
           {7.62939453125e-6f, 0.0f}, // underflow, tie -> 0
           {std::numeric_limits<float>::denorm_min(), 0.0f}};

    std::vector<float> in(cases.size());
    size_t idx = 0;
    for(const auto& c : cases)
    {
        in[idx++] = c.first;
    }
    const auto out = deviceRoundTrip<fp8_e5m2>(in);
    for(size_t i = 0; i < cases.size(); ++i)
    {
        EXPECT_EQ(out[i], cases[i].second) << "input " << cases[i].first;
    }
    EXPECT_EQ(deviceEncode<fp8_e5m2>({-0.0f})[0].data, 0x80); // signed zero
}

// ============================================================================
// E5M2-OCP-Specific: Infinity, NaN, Signed Zero, Bit-Pattern Round-Trip
// ============================================================================

TEST(TestFp8E5M2Device, RoundTripAllPatterns)
{
    std::vector<fp8_e5m2> all(256);
    for(size_t i = 0; i < 256; ++i)
    {
        all[i].data = static_cast<uint8_t>(i);
    }

    const auto decoded = deviceDecode<fp8_e5m2>(all);
    const auto reencoded = deviceEncode<fp8_e5m2>(decoded);
    ASSERT_EQ(decoded.size(), 256u);
    ASSERT_EQ(reencoded.size(), 256u);

    for(size_t bits = 0; bits <= 0xFF; ++bits)
    {
        const auto pattern = static_cast<uint8_t>(bits);
        const float f = decoded[bits];

        if(((pattern & 0x7C) == 0x7C) && ((pattern & 0x03) != 0))
        {
            EXPECT_TRUE(std::isnan(f)) << "Pattern 0x" << std::hex << bits << " should be NaN";
            EXPECT_EQ(reencoded[bits].data & 0x7F, 0x7F);
            continue;
        }

        if((pattern & 0x7F) == 0x7C)
        {
            EXPECT_TRUE(std::isinf(f)) << "Pattern 0x" << std::hex << bits << " should be Inf";

            const uint8_t expectedSaturated = (pattern == 0x7C) ? 0x7B : 0xFB;
            EXPECT_EQ(reencoded[bits].data, expectedSaturated)
                << "Inf saturation round-trip failed for pattern 0x" << std::hex << bits;
            continue;
        }

        EXPECT_EQ(reencoded[bits].data, pattern)
            << "Round-trip failed for bit pattern 0x" << std::hex << bits;
    }
}

TEST(TestFp8E5M2Device, SignedZeroSupported)
{
    EXPECT_EQ(deviceEncodeSingle<fp8_e5m2>(0.0f).data, static_cast<uint8_t>(0x00));
    EXPECT_EQ(deviceEncodeSingle<fp8_e5m2>(-0.0f).data, static_cast<uint8_t>(0x80));
    const float negZero = deviceDecodeSingle<fp8_e5m2>(fp8_e5m2::from_bits(0x80));
    EXPECT_EQ(negZero, -0.0f);
    EXPECT_TRUE(std::signbit(negZero));
}

TEST(TestFp8E5M2Device, NegateZeroProducesNegativeZero)
{
    const auto neg = deviceNegate<fp8_e5m2>({fp8_e5m2::from_bits(0x00)});
    EXPECT_EQ(neg[0].data, 0x80);
    EXPECT_FALSE(deviceQueryState<fp8_e5m2>(neg)[0].nan);
}

TEST(TestFp8E5M2Device, NegateNanProducesNan)
{
    const auto neg = deviceNegate<fp8_e5m2>({fp8_e5m2::from_bits(0x7F), fp8_e5m2::from_bits(0x7D)});
    EXPECT_EQ(neg[0].data, 0xFF);
    EXPECT_EQ(neg[1].data, 0xFD);
    for(const auto& f : deviceQueryState<fp8_e5m2>(neg))
    {
        EXPECT_TRUE(f.nan);
    }
}

TEST(TestFp8E5M2Device, NegateInfinityProducesInfinity)
{
    const auto neg = deviceNegate<fp8_e5m2>({fp8_e5m2::from_bits(0x7C), fp8_e5m2::from_bits(0xFC)});
    EXPECT_EQ(neg[0].data, 0xFC);
    EXPECT_EQ(neg[1].data, 0x7C);
    const auto flags = deviceQueryState<fp8_e5m2>(neg);
    EXPECT_TRUE(flags[0].inf);
    EXPECT_TRUE(flags[0].signbit);
    EXPECT_TRUE(flags[1].inf);
    EXPECT_FALSE(flags[1].signbit);
}

TEST(TestFp8E5M2Device, InfinityTruthTable)
{
    std::vector<fp8_e5m2> all(256);
    for(size_t i = 0; i < 256; ++i)
    {
        all[i] = fp8_e5m2::from_bits(static_cast<uint8_t>(i));
    }
    const auto flags = deviceQueryState<fp8_e5m2>(all);
    for(size_t bits = 0; bits <= 0xFF; ++bits)
    {
        EXPECT_EQ(flags[bits].inf, bits == 0x7C || bits == 0xFC)
            << "isinf wrong for bit pattern 0x" << std::hex << bits;
    }
}

TEST(TestFp8E5M2Device, NanTruthTable)
{
    std::vector<fp8_e5m2> all(256);
    for(size_t i = 0; i < 256; ++i)
    {
        all[i] = fp8_e5m2::from_bits(static_cast<uint8_t>(i));
    }
    const auto flags = deviceQueryState<fp8_e5m2>(all);
    for(size_t bits = 0; bits <= 0xFF; ++bits)
    {
        const bool expectedNan = ((bits & 0x7C) == 0x7C) && ((bits & 0x03) != 0);
        EXPECT_EQ(flags[bits].nan, expectedNan)
            << "isnan wrong for bit pattern 0x" << std::hex << bits;
    }
}

TEST(TestFp8E5M2Device, ClassificationTruthTable)
{
    std::vector<fp8_e5m2> all(256);
    for(size_t i = 0; i < 256; ++i)
    {
        all[i] = fp8_e5m2::from_bits(static_cast<uint8_t>(i));
    }
    const auto flags = deviceQueryState<fp8_e5m2>(all);
    for(size_t bits = 0; bits <= 0xFF; ++bits)
    {
        EXPECT_EQ(flags[bits].finite, !flags[bits].nan && !flags[bits].inf)
            << "isfinite wrong for 0x" << std::hex << bits;
        EXPECT_EQ(flags[bits].signbit, (bits & 0x80) != 0)
            << "signbit wrong for 0x" << std::hex << bits;
    }
}

TEST(TestFp8E5M2Device, MathFunctions)
{
    const auto enc = deviceEncode<fp8_e5m2>({1.0f, 0.0f, 57344.0f, -1.0f, -0.0f});
    const std::vector<fp8_e5m2> p = {enc[0],
                                     enc[1],
                                     enc[2],
                                     fp8_e5m2::from_bits(0x7C),
                                     fp8_e5m2::from_bits(0x7F),
                                     fp8_e5m2::from_bits(0xFC),
                                     enc[3],
                                     enc[4]};
    const auto f = deviceQueryState<fp8_e5m2>(p);

    EXPECT_TRUE(f[0].finite && f[1].finite && f[2].finite);
    EXPECT_FALSE(f[3].finite); // +Inf
    EXPECT_FALSE(f[4].finite); // NaN
    EXPECT_TRUE(f[3].inf && f[5].inf);
    EXPECT_FALSE(f[0].inf);
    EXPECT_FALSE(f[4].inf); // NaN is not Inf
    EXPECT_TRUE(f[4].nan);
    EXPECT_FALSE(f[0].nan);
    EXPECT_FALSE(f[3].nan); // Inf is not NaN
    EXPECT_FALSE(f[0].signbit);
    EXPECT_TRUE(f[6].signbit);
    EXPECT_FALSE(f[1].signbit);
    EXPECT_TRUE(f[7].signbit); // -0
    EXPECT_FALSE(f[3].signbit);
    EXPECT_TRUE(f[5].signbit);
}

// ============================================================================
// Fp16 and Bfp16 casting tests
// ============================================================================

TEST(TestFp8E5M2Device, Fp16ToFp8E5M2RoundTrip)
{
    std::vector<half> input;
    input.reserve(65536);
    for(uint32_t b = 0; b < 65536; ++b)
    {
        input.push_back(half::from_bits(static_cast<uint16_t>(b)));
    }

    std::vector<fp8_e5m2> encoded(input.size());
    deviceDataCast(input, encoded);

    std::vector<half> decoded(encoded.size());
    deviceDataCast(encoded, decoded);

    for(size_t i = 0; i < input.size(); ++i)
    {
        const fp8_e5m2 cpuEncoded(static_cast<float>(input[i]));
        ASSERT_EQ(encoded[i].data, cpuEncoded.data) << "fp16 pattern 0x" << std::hex << i;

        const auto cpuDecoded = static_cast<float>(cpuEncoded);
        const auto deviceDecoded = static_cast<float>(decoded[i]);
        if(std::isnan(cpuDecoded))
        {
            ASSERT_TRUE(std::isnan(deviceDecoded)) << "fp16 pattern 0x" << std::hex << i;
        }
        else
        {
            ASSERT_EQ(__builtin_bit_cast(uint32_t, deviceDecoded),
                      __builtin_bit_cast(uint32_t, cpuDecoded))
                << "fp16 pattern 0x" << std::hex << i;
        }
    }
}

TEST(TestFp8E5M2Device, Bf16ToFp8E5M2RoundTrip)
{
    std::vector<bfloat16> input;
    input.reserve(65536);
    for(uint32_t b = 0; b < 65536; ++b)
    {
        input.push_back(bfloat16::from_bits(static_cast<uint16_t>(b)));
    }

    std::vector<fp8_e5m2> encoded(input.size());
    deviceDataCast(input, encoded);

    std::vector<bfloat16> decoded(encoded.size());
    deviceDataCast(encoded, decoded);

    for(size_t i = 0; i < input.size(); ++i)
    {
        const fp8_e5m2 cpuEncoded(static_cast<float>(input[i]));
        ASSERT_EQ(encoded[i].data, cpuEncoded.data) << "bf16 pattern 0x" << std::hex << i;

        const auto cpuDecoded = static_cast<float>(cpuEncoded);
        const auto deviceDecoded = static_cast<float>(decoded[i]);
        if(std::isnan(cpuDecoded))
        {
            ASSERT_TRUE(std::isnan(deviceDecoded)) << "bf16 pattern 0x" << std::hex << i;
        }
        else
        {
            ASSERT_EQ(__builtin_bit_cast(uint32_t, deviceDecoded),
                      __builtin_bit_cast(uint32_t, cpuDecoded))
                << "bf16 pattern 0x" << std::hex << i;
        }
    }
}

TEST(TestFp8E5M2Device, Fp8E5M2ToFp16AllPatterns)
{
    std::vector<fp8_e5m2> input;
    input.reserve(256);
    for(uint32_t b = 0; b < 256; ++b)
    {
        input.push_back(fp8_e5m2::from_bits(static_cast<uint8_t>(b)));
    }

    std::vector<half> output(input.size());
    deviceDataCast(input, output);

    for(size_t i = 0; i < input.size(); ++i)
    {
        const auto expected = static_cast<float>(input[i]);
        const auto actual = static_cast<float>(output[i]);
        if(std::isnan(expected))
        {
            ASSERT_TRUE(std::isnan(actual)) << "fp8 pattern 0x" << std::hex << i;
            ASSERT_EQ(std::signbit(actual), std::signbit(expected))
                << "NaN sign, fp8 pattern 0x" << std::hex << i;
        }
        else
        {
            ASSERT_EQ(__builtin_bit_cast(uint32_t, actual), __builtin_bit_cast(uint32_t, expected))
                << "fp8 pattern 0x" << std::hex << i;
        }
    }
}

TEST(TestFp8E5M2Device, Fp8E5M2ToBf16AllPatterns)
{
    std::vector<fp8_e5m2> input;
    input.reserve(256);
    for(uint32_t b = 0; b < 256; ++b)
    {
        input.push_back(fp8_e5m2::from_bits(static_cast<uint8_t>(b)));
    }

    std::vector<bfloat16> output(input.size());
    deviceDataCast(input, output);

    for(size_t i = 0; i < input.size(); ++i)
    {
        const auto expected = static_cast<float>(input[i]);
        const auto actual = static_cast<float>(output[i]);
        if(std::isnan(expected))
        {
            ASSERT_TRUE(std::isnan(actual)) << "fp8 pattern 0x" << std::hex << i;
            ASSERT_EQ(std::signbit(actual), std::signbit(expected))
                << "NaN sign, fp8 pattern 0x" << std::hex << i;
        }
        else
        {
            ASSERT_EQ(__builtin_bit_cast(uint32_t, actual), __builtin_bit_cast(uint32_t, expected))
                << "fp8 pattern 0x" << std::hex << i;
        }
    }
}
