// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>

#include "Common.hpp"

using namespace hipdnn_data_sdk::types;
using namespace gpu_ref_device_type_test;
using namespace gpu_ref_device_type_test::detail;

// ============================================================================
// Exact-value round trip
// ============================================================================

TEST(TestFp8E4M3Device, RoundTripExactValues)
{
    const std::vector<float> exactValues = {
        0.0f,          1.0f,  -1.0f, 2.0f,   -2.0f,  4.0f,   -4.0f,  8.0f,   -8.0f,
        0.5f,          -0.5f, 0.25f, -0.25f, 1.125f, 1.875f, 240.0f, 416.0f,
        448.0f, // MAX
        -448.0f, // LOWEST
        0.015625f, // MIN_NORMAL = 2^-6
        -0.015625f,
        0.013671875f, // largest subnormal = 7 * 2^-9
        0.001953125f, // denorm_min = 2^-9
        -0.001953125f,
    };

    const auto decoded = deviceRoundTrip<fp8_e4m3>(exactValues);
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

class TestFp8E4M3DeviceRounding : public ::testing::TestWithParam<RoundingTestCase>
{
};

TEST_P(TestFp8E4M3DeviceRounding, Rounding)
{
    auto [input, expected] = GetParam();
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(input), expected);
}

// Midpoints between two representable encodings: ties-to-even
INSTANTIATE_TEST_SUITE_P(MidpointRounding,
                         TestFp8E4M3DeviceRounding,
                         ::testing::Values(
                             // 0x38 (mant=0, even) vs 0x39 (odd): midpoint 1.0625 -> 1.0
                             RoundingTestCase{1.0625f, 1.0f},
                             // 0x39 (odd) vs 0x3A (even): midpoint 1.1875 -> 1.25
                             RoundingTestCase{1.1875f, 1.25f},
                             // 0x7C=384 (even) vs 0x7D=416 (odd): midpoint 400 -> 384
                             RoundingTestCase{400.0f, 384.0f},
                             // 0x7D=416 (odd) vs 0x7E=448 (even): midpoint 432 -> 448
                             RoundingTestCase{432.0f, 448.0f}));

// Non-tie boundary cases: verify rounding direction
INSTANTIATE_TEST_SUITE_P(BoundaryRounding,
                         TestFp8E4M3DeviceRounding,
                         ::testing::Values(RoundingTestCase{1.03f, 1.0f},
                                           RoundingTestCase{1.1f, 1.125f},
                                           RoundingTestCase{390.0f, 384.0f},
                                           RoundingTestCase{410.0f, 416.0f},
                                           RoundingTestCase{-1.03f, -1.0f}));

// Subnormal / underflow range. denorm_min = 2^-9; underflow threshold = 2^-10
INSTANTIATE_TEST_SUITE_P(SubnormalRounding,
                         TestFp8E4M3DeviceRounding,
                         ::testing::Values(RoundingTestCase{1e-4f, 0.0f},
                                           RoundingTestCase{0.0009765625f, 0.0f},
                                           RoundingTestCase{0.001f, 0.001953125f},
                                           RoundingTestCase{0.0029296875f, 0.00390625f},
                                           RoundingTestCase{0.0048828125f, 0.00390625f},
                                           RoundingTestCase{0.01416015625f, 0.013671875f},
                                           RoundingTestCase{0.0146484375f, 0.015625f},
                                           RoundingTestCase{0.01513671875f, 0.015625f},
                                           RoundingTestCase{-0.01513671875f, -0.015625f}));

// Mantissa overflow cascading into the exponent, and rounding into saturation
INSTANTIATE_TEST_SUITE_P(RoundingOverflow,
                         TestFp8E4M3DeviceRounding,
                         ::testing::Values(RoundingTestCase{1.9375f, 2.0f},
                                           RoundingTestCase{464.0f, 448.0f},
                                           RoundingTestCase{470.0f, 448.0f},
                                           RoundingTestCase{-1.9375f, -2.0f},
                                           RoundingTestCase{-464.0f, -448.0f},
                                           RoundingTestCase{-470.0f, -448.0f}));

// ============================================================================
// Saturation Tests
// ============================================================================

TEST(TestFp8E4M3Device, SaturationPositive)
{
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(1e10f), 448.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(500.0f), 448.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(480.0f), 448.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(464.0f), 448.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(449.0f), 448.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(std::nextafter(448.0f, 1e9f)), 448.0f);
}

TEST(TestFp8E4M3Device, SaturationNegative)
{
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(-1e10f), -448.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(-500.0f), -448.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(-480.0f), -448.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(-464.0f), -448.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(-449.0f), -448.0f);
}

TEST(TestFp8E4M3Device, SaturationInfinity)
{
    // E4M3 has no infinity: the saturating constructor maps +/-Inf to +/-MAX
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(std::numeric_limits<float>::infinity()), 448.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(-std::numeric_limits<float>::infinity()), -448.0f);
}

// ============================================================================
// Underflow Tests
// ============================================================================

TEST(TestFp8E4M3Device, Underflow)
{
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(1e-10f), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(-1e-10f), 0.0f);

    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(std::numeric_limits<float>::denorm_min()), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(-std::numeric_limits<float>::denorm_min()), 0.0f);

    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(1e-4f), 0.0f);
    EXPECT_EQ(deviceRoundTripSingle<fp8_e4m3>(0.0009765625f), 0.0f);

    EXPECT_EQ(deviceEncodeSingle<fp8_e4m3>(1.5f * 0.0009765625f).data, static_cast<uint8_t>(0x01));
    EXPECT_EQ(deviceEncodeSingle<fp8_e4m3>(-1.5f * 0.0009765625f).data, static_cast<uint8_t>(0x81));
    EXPECT_EQ(deviceDecodeSingle<fp8_e4m3>(fp8_e4m3::from_bits(0x01)), 0.001953125f);
    EXPECT_EQ(deviceDecodeSingle<fp8_e4m3>(fp8_e4m3::from_bits(0x81)), -0.001953125f);
}

TEST(TestFp8E4M3Device, DecodeMatchesCpuAllPatterns)
{
    std::vector<fp8_e4m3> all(256);
    for(size_t b = 0; b < 256; ++b)
    {
        all[b] = fp8_e4m3::from_bits(static_cast<uint8_t>(b));
    }

    const auto dev = deviceDecode<fp8_e4m3>(all);
    for(size_t b = 0; b < 256; ++b)
    {
        EXPECT_EQ(__builtin_bit_cast(uint32_t, dev[b]),
                  __builtin_bit_cast(uint32_t, static_cast<float>(all[b])))
            << "pattern 0x" << std::hex << b;
    }
}

TEST(TestFp8E4M3Device, EncodeMatchesCpu)
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

    for(int c = 0; c <= 0x7E; ++c)
    {
        const float v = static_cast<float>(fp8_e4m3::from_bits(static_cast<uint8_t>(c)));
        addAround(v);
        if(c < 0x7E)
        {
            addAround((v + static_cast<float>(fp8_e4m3::from_bits(static_cast<uint8_t>(c + 1))))
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
                         464.0f, // midpoint between MAX and the NaN pattern
                         480.0f,
                         0.0f,
                         -0.0f})
    {
        in.push_back(s);
    }
    addAround(464.0f);

    const auto dev = deviceEncode<fp8_e4m3>(in);
    for(size_t i = 0; i < in.size(); ++i)
    {
        EXPECT_EQ(dev[i].data, fp8_e4m3(in[i]).data) << "float bits 0x" << std::hex << bits(in[i]);
    }
}

TEST(TestFp8E4M3Device, KnownValues)
{
    const std::vector<std::pair<float, float>> cases
        = {{1.0f, 1.0f},
           {-448.0f, -448.0f},
           {0.001953125f, 0.001953125f},
           {1.0625f, 1.0f},
           {1.1875f, 1.25f}, // ties to even
           {464.0f, 448.0f},
           {1e10f, 448.0f}, // saturation
           {std::numeric_limits<float>::infinity(), 448.0f},
           {-std::numeric_limits<float>::infinity(), -448.0f},
           {1e-10f, 0.0f},
           {0.0009765625f, 0.0f}, // underflow, tie -> 0
           {std::numeric_limits<float>::denorm_min(), 0.0f}};

    std::vector<float> in(cases.size());
    size_t idx = 0;
    for(const auto& c : cases)
    {
        in[idx++] = c.first;
    }
    const auto out = deviceRoundTrip<fp8_e4m3>(in);
    for(size_t i = 0; i < cases.size(); ++i)
    {
        EXPECT_EQ(out[i], cases[i].second) << "input " << cases[i].first;
    }
    EXPECT_EQ(deviceEncode<fp8_e4m3>({-0.0f})[0].data, 0x80); // signed zero
}

// ============================================================================
// E4M3-OCP-Specific: no Infinity, NaN, Signed Zero, Bit-Pattern Round-Trip
// ============================================================================

TEST(TestFp8E4M3Device, RoundTripAllPatterns)
{
    std::vector<fp8_e4m3> all(256);
    for(size_t i = 0; i < 256; ++i)
    {
        all[i].data = static_cast<uint8_t>(i);
    }

    const auto decoded = deviceDecode<fp8_e4m3>(all);
    const auto reencoded = deviceEncode<fp8_e4m3>(decoded);
    ASSERT_EQ(decoded.size(), 256u);
    ASSERT_EQ(reencoded.size(), 256u);

    for(size_t bits = 0; bits <= 0xFF; ++bits)
    {
        const auto pattern = static_cast<uint8_t>(bits);
        const float f = decoded[bits];

        if((pattern & 0x7F) == 0x7F)
        {
            EXPECT_TRUE(std::isnan(f)) << "Pattern 0x" << std::hex << bits << " should be NaN";
            EXPECT_EQ(reencoded[bits].data, pattern)
                << "NaN round-trip failed for pattern 0x" << std::hex << bits;
            continue;
        }

        // There is no Inf: every other pattern is finite and round-trips exactly
        EXPECT_TRUE(std::isfinite(f)) << "Pattern 0x" << std::hex << bits << " should be finite";
        EXPECT_EQ(reencoded[bits].data, pattern)
            << "Round-trip failed for bit pattern 0x" << std::hex << bits;
    }
}

TEST(TestFp8E4M3Device, SignedZeroSupported)
{
    EXPECT_EQ(deviceEncodeSingle<fp8_e4m3>(0.0f).data, static_cast<uint8_t>(0x00));
    EXPECT_EQ(deviceEncodeSingle<fp8_e4m3>(-0.0f).data, static_cast<uint8_t>(0x80));
    const float negZero = deviceDecodeSingle<fp8_e4m3>(fp8_e4m3::from_bits(0x80));
    EXPECT_EQ(negZero, -0.0f);
    EXPECT_TRUE(std::signbit(negZero));
}

TEST(TestFp8E4M3Device, NegateZeroProducesNegativeZero)
{
    const auto neg = deviceNegate<fp8_e4m3>({fp8_e4m3::from_bits(0x00)});
    EXPECT_EQ(neg[0].data, 0x80);
    EXPECT_FALSE(deviceQueryState<fp8_e4m3>(neg)[0].nan);
}

TEST(TestFp8E4M3Device, NegateNanProducesNan)
{
    // E4M3 has a single NaN encoding per sign: 0x7F and 0xFF
    const auto neg = deviceNegate<fp8_e4m3>({fp8_e4m3::from_bits(0x7F), fp8_e4m3::from_bits(0xFF)});
    EXPECT_EQ(neg[0].data, 0xFF);
    EXPECT_EQ(neg[1].data, 0x7F);
    for(const auto& f : deviceQueryState<fp8_e4m3>(neg))
    {
        EXPECT_TRUE(f.nan);
    }
}

TEST(TestFp8E4M3Device, NegateFiniteFlipsSign)
{
    // E4M3 has no infinity; negating MAX / LOWEST swaps them and stays finite
    const auto neg = deviceNegate<fp8_e4m3>({fp8_e4m3::from_bits(0x7E), fp8_e4m3::from_bits(0xFE)});
    EXPECT_EQ(neg[0].data, 0xFE);
    EXPECT_EQ(neg[1].data, 0x7E);
    const auto flags = deviceQueryState<fp8_e4m3>(neg);
    EXPECT_TRUE(flags[0].finite);
    EXPECT_TRUE(flags[0].signbit);
    EXPECT_TRUE(flags[1].finite);
    EXPECT_FALSE(flags[1].signbit);
}

TEST(TestFp8E4M3Device, InfinityTruthTable)
{
    // E4M3 has no infinity representation: isinf must be false for every pattern
    std::vector<fp8_e4m3> all(256);
    for(size_t i = 0; i < 256; ++i)
    {
        all[i] = fp8_e4m3::from_bits(static_cast<uint8_t>(i));
    }
    const auto flags = deviceQueryState<fp8_e4m3>(all);
    for(size_t bits = 0; bits <= 0xFF; ++bits)
    {
        EXPECT_FALSE(flags[bits].inf) << "isinf wrong for bit pattern 0x" << std::hex << bits;
    }
}

TEST(TestFp8E4M3Device, NanTruthTable)
{
    std::vector<fp8_e4m3> all(256);
    for(size_t i = 0; i < 256; ++i)
    {
        all[i] = fp8_e4m3::from_bits(static_cast<uint8_t>(i));
    }
    const auto flags = deviceQueryState<fp8_e4m3>(all);
    for(size_t bits = 0; bits <= 0xFF; ++bits)
    {
        const bool expectedNan = (bits & 0x7F) == 0x7F;
        EXPECT_EQ(flags[bits].nan, expectedNan)
            << "isnan wrong for bit pattern 0x" << std::hex << bits;
    }
}

TEST(TestFp8E4M3Device, ClassificationTruthTable)
{
    std::vector<fp8_e4m3> all(256);
    for(size_t i = 0; i < 256; ++i)
    {
        all[i] = fp8_e4m3::from_bits(static_cast<uint8_t>(i));
    }
    const auto flags = deviceQueryState<fp8_e4m3>(all);
    for(size_t bits = 0; bits <= 0xFF; ++bits)
    {
        // No infinity, so finite == !nan
        EXPECT_EQ(flags[bits].finite, !flags[bits].nan)
            << "isfinite wrong for 0x" << std::hex << bits;
        EXPECT_EQ(flags[bits].signbit, (bits & 0x80) != 0)
            << "signbit wrong for 0x" << std::hex << bits;
    }
}

TEST(TestFp8E4M3Device, MathFunctions)
{
    const auto enc = deviceEncode<fp8_e4m3>({1.0f, 0.0f, 448.0f, -1.0f, -0.0f});
    const std::vector<fp8_e4m3> p = {enc[0],
                                     enc[1],
                                     enc[2],
                                     fp8_e4m3::from_bits(0x7F),
                                     fp8_e4m3::from_bits(0xFF),
                                     enc[3],
                                     enc[4]};
    const auto f = deviceQueryState<fp8_e4m3>(p);

    EXPECT_TRUE(f[0].finite && f[1].finite && f[2].finite);
    EXPECT_FALSE(f[3].finite); // +NaN
    EXPECT_FALSE(f[4].finite); // -NaN
    for(const auto& flag : f)
    {
        EXPECT_FALSE(flag.inf); // E4M3 has no infinity
    }
    EXPECT_TRUE(f[3].nan && f[4].nan);
    EXPECT_FALSE(f[0].nan);
    EXPECT_FALSE(f[2].nan);
    EXPECT_FALSE(f[0].signbit);
    EXPECT_TRUE(f[5].signbit);
    EXPECT_FALSE(f[1].signbit);
    EXPECT_TRUE(f[6].signbit); // -0
    EXPECT_FALSE(f[3].signbit);
    EXPECT_TRUE(f[4].signbit);
}

// ============================================================================
// Fp16 and Bfp16 casting tests
// ============================================================================

TEST(TestFp8E4M3Device, Fp16ToFp8E4M3RoundTrip)
{
    std::vector<half> input;
    input.reserve(65536);
    for(uint32_t b = 0; b < 65536; ++b)
    {
        input.push_back(half::from_bits(static_cast<uint16_t>(b)));
    }

    std::vector<fp8_e4m3> encoded(input.size());
    deviceDataCast(input, encoded);

    std::vector<half> decoded(encoded.size());
    deviceDataCast(encoded, decoded);

    for(size_t i = 0; i < input.size(); ++i)
    {
        const fp8_e4m3 cpuEncoded(static_cast<float>(input[i]));
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

TEST(TestFp8E4M3Device, Bf16ToFp8E4M3RoundTrip)
{
    std::vector<bfloat16> input;
    input.reserve(65536);
    for(uint32_t b = 0; b < 65536; ++b)
    {
        input.push_back(bfloat16::from_bits(static_cast<uint16_t>(b)));
    }

    std::vector<fp8_e4m3> encoded(input.size());
    deviceDataCast(input, encoded);

    std::vector<bfloat16> decoded(encoded.size());
    deviceDataCast(encoded, decoded);

    for(size_t i = 0; i < input.size(); ++i)
    {
        const fp8_e4m3 cpuEncoded(static_cast<float>(input[i]));
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

TEST(TestFp8E4M3Device, Fp8E4M3ToFp16AllPatterns)
{
    std::vector<fp8_e4m3> input;
    input.reserve(256);
    for(uint32_t b = 0; b < 256; ++b)
    {
        input.push_back(fp8_e4m3::from_bits(static_cast<uint8_t>(b)));
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

TEST(TestFp8E4M3Device, Fp8E4M3ToBf16AllPatterns)
{
    std::vector<fp8_e4m3> input;
    input.reserve(256);
    for(uint32_t b = 0; b < 256; ++b)
    {
        input.push_back(fp8_e4m3::from_bits(static_cast<uint8_t>(b)));
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
