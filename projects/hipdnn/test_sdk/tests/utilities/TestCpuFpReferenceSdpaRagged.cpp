// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ============================================================================
// Validates the ragged CPU reference (CpuFpReferenceSdpaRagged) against the
// trusted dense CpuFpReferenceSdpa. Inputs are RFC-0014 ragged tensors
// (ShallowRaggedTensor over packed buffers + per-primary ragged_offset aux);
// each batch is extracted into a dense [1,H,seqlen_b,D] tensor, the dense
// reference is run on it, and its output (and LSE) is compared with the ragged
// reference. This is the middle link of the validation chain: dense CPU
// (trusted) -> CPU ragged (here) -> GPU ragged (TestGpuFpReferenceSdpaRagged).
//
// This suite deliberately covers the fp8/descale, LSE, and fully-masked branches
// on a CPU-only path (the GPU-vs-CPU suite that also exercises them is
// SKIP_IF_NO_DEVICES and does not run in the coverage lane), and gives fp8 +
// descale an independent dense oracle via dequantize-to-float. CPU-only.
// ============================================================================

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using namespace hipdnn_test_sdk::utilities;

namespace
{

int64_t sum(const std::vector<int64_t>& v)
{
    return std::accumulate(v.begin(), v.end(), int64_t{0});
}

int64_t maxOf(const std::vector<int64_t>& v)
{
    return *std::max_element(v.begin(), v.end());
}

// Deterministic fill of a packed backing buffer in [-1, 1); avoids <random> cross-TU drift.
template <typename T>
void fillPacked(std::vector<T>& buf, unsigned int seed)
{
    uint32_t state = seed;
    for(auto& x : buf)
    {
        state = state * 1664525U + 1013904223U;
        const float u = static_cast<float>(state >> 8) / static_cast<float>(1U << 24); // [0,1)
        x = static_cast<T>(2.0f * u - 1.0f);
    }
}

// Wrap a borrowed packed backing buffer as an RFC-0014 ragged tensor ([B,H,S,D], seqAxis=2, BSHD).
template <typename T>
ShallowRaggedTensor<T> wrapRagged(T* buf,
                                  const std::vector<int64_t>& dims,
                                  int64_t seqStride,
                                  const std::vector<int64_t>& cum)
{
    return ShallowRaggedTensor<T>(
        buf, dims, bshd(dims), SEQ_AXIS, makeRaggedOffsetAux(cum, seqStride));
}

// Build a valid rank-4 ragged tensor over a caller-owned backing buffer, for the negative-validation
// tests below where forward() is expected to throw in validateInput() before any ragged addressing.
// seqStride is H*D in int64_t (avoids the implicit-widening tidy warning); the backing is sized to
// the packed token count so construction always succeeds.
ShallowRaggedTensor<float> makeValidRagged(std::vector<float>& backing,
                                           const std::vector<int64_t>& dims,
                                           const std::vector<int64_t>& seqLens)
{
    const int64_t seqStride = dims[1] * dims[3]; // H * D
    backing.assign(static_cast<size_t>(sum(seqLens) * seqStride), 0.0f);
    return wrapRagged(backing.data(), dims, seqStride, cumTokens(seqLens));
}

Tensor<float> makeScalarDescale(float value)
{
    Tensor<float> d({1});
    d.memory().hostData()[0] = value;
    d.memory().markHostModified();
    return d;
}

// Per-KV-head descale [B, heads, 1, 1] with a distinct value per (b, head).
Tensor<float> makePerHeadDescale(int64_t batch, int64_t heads, float base)
{
    Tensor<float> d({batch, heads, 1, 1});
    auto* p = d.memory().hostData();
    for(int64_t i = 0; i < batch * heads; ++i)
    {
        p[i] = base + 0.1f * static_cast<float>(i);
    }
    d.memory().markHostModified();
    return d;
}

// Descale value for (batch, head): scalar [1] or per-head [B, heads, 1, 1].
float descaleValue(TensorBase<float>& descale, int64_t b, int64_t head)
{
    if(descale.elementCount() == 1)
    {
        return descale.getHostValue(std::vector<int64_t>{0});
    }
    return descale.getHostValue(std::vector<int64_t>{b, head, 0, 0});
}

// Extract a dense [1, heads, seqLen, dim] slice for batch b from a ragged tensor (batch-relative
// seq index; ragged addressing handles the packing).
template <typename T>
Tensor<float> extractDenseSlice(TensorBase<T>& ragged, int64_t b, int64_t seqLen)
{
    const auto heads = ragged.dims()[1];
    const auto dim = ragged.dims()[3];
    Tensor<float> dense({1, heads, seqLen, dim});
    for(int64_t h = 0; h < heads; ++h)
    {
        for(int64_t s = 0; s < seqLen; ++s)
        {
            for(int64_t d = 0; d < dim; ++d)
            {
                dense(0, h, s, d)
                    = static_cast<float>(ragged.getHostValue(std::vector<int64_t>{b, h, s, d}));
            }
        }
    }
    dense.memory().markHostModified();
    return dense;
}

// Like extractDenseSlice but dequantizes fp8 -> float and folds in the per-(batch, KV-head) descale.
// Folding descale into the inputs is algebraically the reference's score*=dQ*dK; out*=dV. Descales
// are per KV head, so tensor head h reads descale head h / headsPerDescaleHead (H_q / H_kv for Q,
// 1 for K/V).
template <typename FP8>
Tensor<float> dequantDenseSlice(TensorBase<FP8>& ragged,
                                int64_t b,
                                int64_t seqLen,
                                TensorBase<float>& descale,
                                int64_t headsPerDescaleHead)
{
    const auto heads = ragged.dims()[1];
    const auto dim = ragged.dims()[3];
    Tensor<float> dense({1, heads, seqLen, dim});
    for(int64_t h = 0; h < heads; ++h)
    {
        const float dsc = descaleValue(descale, b, h / headsPerDescaleHead);
        for(int64_t s = 0; s < seqLen; ++s)
        {
            for(int64_t d = 0; d < dim; ++d)
            {
                dense(0, h, s, d)
                    = static_cast<float>(ragged.getHostValue(std::vector<int64_t>{b, h, s, d}))
                      * dsc;
            }
        }
    }
    dense.memory().markHostModified();
    return dense;
}

// Build packed ragged float inputs (ShallowRaggedTensor), run the ragged CPU reference (with LSE),
// then validate each batch's output and LSE against the dense CPU reference on [1,H,seqlen_b,D]
// slices.
void checkRaggedVsDense(const std::vector<int64_t>& seqQ,
                        const std::vector<int64_t>& seqKv,
                        int64_t numHeads,
                        int64_t numHeadsKv,
                        int64_t headDim,
                        int64_t headDimV,
                        int64_t leftBound = -1,
                        int64_t rightBound = -1,
                        bool topLeftAlignment = true,
                        std::optional<float> attnScale = std::nullopt)
{
    ASSERT_EQ(seqQ.size(), seqKv.size());
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto sMaxQ = maxOf(seqQ);
    const auto sMaxKv = maxOf(seqKv);
    const auto totalQ = sum(seqQ);
    const auto totalKv = sum(seqKv);
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);

    const std::vector<int64_t> qDims = {batch, numHeads, sMaxQ, headDim};
    const std::vector<int64_t> kDims = {batch, numHeadsKv, sMaxKv, headDim};
    const std::vector<int64_t> vDims = {batch, numHeadsKv, sMaxKv, headDimV};
    const std::vector<int64_t> oDims = {batch, numHeads, sMaxQ, headDimV};
    const std::vector<int64_t> lseDims = {batch, numHeads, sMaxQ, 1};

    std::vector<float> qBack(static_cast<size_t>(totalQ * numHeads * headDim));
    std::vector<float> kBack(static_cast<size_t>(totalKv * numHeadsKv * headDim));
    std::vector<float> vBack(static_cast<size_t>(totalKv * numHeadsKv * headDimV));
    std::vector<float> oBack(static_cast<size_t>(totalQ * numHeads * headDimV), 0.0f);
    std::vector<float> lseBack(static_cast<size_t>(totalQ * numHeads), 0.0f);
    fillPacked(qBack, 11);
    fillPacked(kBack, 22);
    fillPacked(vBack, 33);

    auto q = wrapRagged(qBack.data(), qDims, numHeads * headDim, cumQ);
    auto k = wrapRagged(kBack.data(), kDims, numHeadsKv * headDim, cumKv);
    auto v = wrapRagged(vBack.data(), vDims, numHeadsKv * headDimV, cumKv);
    auto o = wrapRagged(oBack.data(), oDims, numHeads * headDimV, cumQ);
    auto lse = wrapRagged(lseBack.data(), lseDims, numHeads, cumQ);

    CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
        q, k, v, o, attnScale, leftBound, rightBound, topLeftAlignment, &lse);

    for(int64_t b = 0; b < batch; ++b)
    {
        const auto sQ = seqQ[static_cast<size_t>(b)];
        const auto sKv = seqKv[static_cast<size_t>(b)];
        if(sQ == 0)
        {
            continue; // no rows to check, and the dense oracle rejects an empty slice
        }
        auto qd = extractDenseSlice(q, b, sQ);
        auto kd = extractDenseSlice(k, b, sKv);
        auto vd = extractDenseSlice(v, b, sKv);
        Tensor<float> oDense({1, numHeads, sQ, headDimV});
        Tensor<float> lseDense({1, numHeads, sQ, 1});

        CpuFpReferenceSdpa::forward<float, float, float, float, float>(qd,
                                                                       kd,
                                                                       vd,
                                                                       oDense,
                                                                       attnScale,
                                                                       /*attnMask=*/nullptr,
                                                                       leftBound,
                                                                       rightBound,
                                                                       topLeftAlignment,
                                                                       &lseDense);

        for(int64_t s = 0; s < sQ; ++s)
        {
            for(int64_t h = 0; h < numHeads; ++h)
            {
                EXPECT_NEAR(
                    lse.getHostValue(std::vector<int64_t>{b, h, s, 0}), lseDense(0, h, s, 0), 1e-4f)
                    << "LSE mismatch batch " << b << " token " << s << " head " << h;
                for(int64_t dv = 0; dv < headDimV; ++dv)
                {
                    EXPECT_NEAR(o.getHostValue(std::vector<int64_t>{b, h, s, dv}),
                                oDense(0, h, s, dv),
                                1e-4f)
                        << "output mismatch batch " << b << " token " << s << " head " << h
                        << " dv " << dv;
                }
            }
        }
    }
}

// fp8 (E4M3) + descale vs an independent dense oracle (dequantize-to-float). bf16 output.
void checkRaggedFp8VsDense(const std::vector<int64_t>& seqQ,
                           const std::vector<int64_t>& seqKv,
                           int64_t numHeads,
                           int64_t numHeadsKv,
                           int64_t headDim,
                           TensorBase<float>& descaleQ,
                           TensorBase<float>& descaleK,
                           TensorBase<float>& descaleV,
                           int64_t leftBound,
                           int64_t rightBound,
                           bool topLeftAlignment)
{
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto sMaxQ = maxOf(seqQ);
    const auto sMaxKv = maxOf(seqKv);
    const auto totalQ = sum(seqQ);
    const auto totalKv = sum(seqKv);
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);

    const std::vector<int64_t> qDims = {batch, numHeads, sMaxQ, headDim};
    const std::vector<int64_t> kvDims = {batch, numHeadsKv, sMaxKv, headDim};
    const std::vector<int64_t> oDims = {batch, numHeads, sMaxQ, headDim};

    std::vector<fp8_e4m3> qBack(static_cast<size_t>(totalQ * numHeads * headDim));
    std::vector<fp8_e4m3> kBack(static_cast<size_t>(totalKv * numHeadsKv * headDim));
    std::vector<fp8_e4m3> vBack(static_cast<size_t>(totalKv * numHeadsKv * headDim));
    std::vector<bfloat16> oBack(static_cast<size_t>(totalQ * numHeads * headDim), bfloat16(0.0f));
    fillPacked(qBack, 11);
    fillPacked(kBack, 22);
    fillPacked(vBack, 33);

    auto q = wrapRagged(qBack.data(), qDims, numHeads * headDim, cumQ);
    auto k = wrapRagged(kBack.data(), kvDims, numHeadsKv * headDim, cumKv);
    auto v = wrapRagged(vBack.data(), kvDims, numHeadsKv * headDim, cumKv);
    auto o = wrapRagged(oBack.data(), oDims, numHeads * headDim, cumQ);

    CpuFpReferenceSdpaRagged::forward<fp8_e4m3, fp8_e4m3, fp8_e4m3, bfloat16, float>(
        q,
        k,
        v,
        o,
        std::nullopt,
        leftBound,
        rightBound,
        topLeftAlignment,
        nullptr,
        &descaleQ,
        &descaleK,
        &descaleV);

    for(int64_t b = 0; b < batch; ++b)
    {
        const auto sQ = seqQ[static_cast<size_t>(b)];
        const auto sKv = seqKv[static_cast<size_t>(b)];
        auto qd = dequantDenseSlice(q, b, sQ, descaleQ, numHeads / numHeadsKv);
        auto kd = dequantDenseSlice(k, b, sKv, descaleK, 1);
        auto vd = dequantDenseSlice(v, b, sKv, descaleV, 1);
        Tensor<bfloat16> oDense({1, numHeads, sQ, headDim});
        CpuFpReferenceSdpa::forward<float, float, float, bfloat16, float>(qd,
                                                                          kd,
                                                                          vd,
                                                                          oDense,
                                                                          std::nullopt,
                                                                          /*attnMask=*/nullptr,
                                                                          leftBound,
                                                                          rightBound,
                                                                          topLeftAlignment);

        for(int64_t s = 0; s < sQ; ++s)
        {
            for(int64_t h = 0; h < numHeads; ++h)
            {
                for(int64_t dv = 0; dv < headDim; ++dv)
                {
                    EXPECT_NEAR(
                        static_cast<float>(o.getHostValue(std::vector<int64_t>{b, h, s, dv})),
                        static_cast<float>(oDense(0, h, s, dv)),
                        2e-2f)
                        << "fp8 output mismatch batch " << b << " token " << s << " head " << h
                        << " dv " << dv;
                }
            }
        }
    }
}

} // namespace

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedBasicMha)
{
    checkRaggedVsDense({3, 5, 1}, {3, 5, 1}, 4, 4, 16, 16);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedCrossAttention)
{
    checkRaggedVsDense({2, 4, 3}, {5, 1, 6}, 2, 2, 16, 16);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedCausalTopLeft)
{
    checkRaggedVsDense({4, 7}, {4, 7}, 2, 2, 16, 16, -1, 0, true);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedCausalBottomRight)
{
    checkRaggedVsDense({3, 5}, {6, 8}, 2, 2, 16, 16, -1, 0, false);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedSlidingWindow)
{
    checkRaggedVsDense({8, 6}, {8, 6}, 2, 2, 16, 16, 2, 2, true);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedGqa)
{
    checkRaggedVsDense({5, 3}, {5, 3}, 8, 2, 16, 16);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedAsymmetricHeadDim)
{
    // hdim_q = 192, hdim_v = 128 (asymmetric head dims, as on the ASM v3 path).
    checkRaggedVsDense({3, 5}, {3, 5}, 2, 2, 192, 128);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedExplicitAttnScale)
{
    // Caller-provided attnScale (not the default 1/sqrt(headDim)); the dense oracle uses the same
    // value, so a match confirms the explicit-scale path is honored.
    checkRaggedVsDense({4, 6}, {4, 6}, 2, 2, 16, 16, -1, -1, true, /*attnScale=*/0.125f);
}

// --- fp8 (E4M3) + descale vs the dense reference (dequantize-to-float oracle) ---

TEST(TestCpuFpReferenceSdpaRaggedFp8, RaggedPerTensorDescale)
{
    auto descaleQ = makeScalarDescale(0.5f);
    auto descaleK = makeScalarDescale(0.25f);
    auto descaleV = makeScalarDescale(2.0f);
    checkRaggedFp8VsDense({3, 5}, {3, 5}, 2, 2, 128, descaleQ, descaleK, descaleV, -1, -1, true);
}

TEST(TestCpuFpReferenceSdpaRaggedFp8, RaggedCausalGqaPerKvHeadDescale)
{
    const int64_t batch = 2;
    const int64_t numHeadsKv = 2; // GQA (numHeads = 4)
    auto descaleQ = makeScalarDescale(0.5f);
    auto descaleK = makePerHeadDescale(batch, numHeadsKv, 0.2f);
    auto descaleV = makePerHeadDescale(batch, numHeadsKv, 0.3f);
    checkRaggedFp8VsDense(
        {4, 6}, {4, 6}, 4, numHeadsKv, 128, descaleQ, descaleK, descaleV, -1, 0, true);
}

// GQA with per-KV-head descales on all of Q/K/V (AITER's [B, H_kv] shape). Distinct values per
// (batch, KV head) on Q catch a Q descale indexed by the query head instead of its KV head.
TEST(TestCpuFpReferenceSdpaRaggedFp8, RaggedGqaPerKvHeadDescaleQkv)
{
    const int64_t batch = 2;
    const int64_t numHeadsKv = 2; // GQA (numHeads = 4)
    auto descaleQ = makePerHeadDescale(batch, numHeadsKv, 0.4f);
    auto descaleK = makePerHeadDescale(batch, numHeadsKv, 0.2f);
    auto descaleV = makePerHeadDescale(batch, numHeadsKv, 0.3f);
    checkRaggedFp8VsDense(
        {4, 6}, {5, 3}, 4, numHeadsKv, 128, descaleQ, descaleK, descaleV, -1, -1, true);
}

// --- Dense LSE (the frontend's default [B, H, Sq_max, 1] stats layout) ---
// The ragged LSE path is validated against the dense reference in checkRaggedVsDense; a dense LSE
// written through the same forward() must carry the identical per-(b, h, sq) values, with padding
// rows (sq >= seqQ[b]) left untouched. This is the contract the GPU reference mirrors.
TEST(TestCpuFpReferenceSdpaRaggedFp32, DenseLseMatchesRaggedLse)
{
    const std::vector<int64_t> seqQ = {3, 5, 1};
    const std::vector<int64_t> seqKv = {4, 2, 6};
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto sMaxQ = maxOf(seqQ);
    const auto sMaxKv = maxOf(seqKv);
    const auto totalQ = sum(seqQ);
    const auto totalKv = sum(seqKv);
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);

    const std::vector<int64_t> qDims = {batch, numHeads, sMaxQ, headDim};
    const std::vector<int64_t> kvDims = {batch, numHeads, sMaxKv, headDim};
    const std::vector<int64_t> lseDims = {batch, numHeads, sMaxQ, 1};

    std::vector<float> qBack(static_cast<size_t>(totalQ * numHeads * headDim));
    std::vector<float> kBack(static_cast<size_t>(totalKv * numHeads * headDim));
    std::vector<float> vBack(static_cast<size_t>(totalKv * numHeads * headDim));
    std::vector<float> oBack(qBack.size(), 0.0f);
    std::vector<float> lseRaggedBack(static_cast<size_t>(totalQ * numHeads), 0.0f);
    fillPacked(qBack, 11);
    fillPacked(kBack, 22);
    fillPacked(vBack, 33);

    auto q = wrapRagged(qBack.data(), qDims, numHeads * headDim, cumQ);
    auto k = wrapRagged(kBack.data(), kvDims, numHeads * headDim, cumKv);
    auto v = wrapRagged(vBack.data(), kvDims, numHeads * headDim, cumKv);
    auto o = wrapRagged(oBack.data(), qDims, numHeads * headDim, cumQ);
    auto lseRagged = wrapRagged(lseRaggedBack.data(), lseDims, numHeads, cumQ);

    constexpr float SENTINEL = -99.0f;
    Tensor<float> lseDense(lseDims);
    lseDense.fillWithValue(SENTINEL);

    CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
        q, k, v, o, std::nullopt, -1, -1, true, &lseRagged);
    CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
        q, k, v, o, std::nullopt, -1, -1, true, &lseDense);

    for(int64_t b = 0; b < batch; ++b)
    {
        for(int64_t h = 0; h < numHeads; ++h)
        {
            for(int64_t s = 0; s < sMaxQ; ++s)
            {
                const float dense = lseDense(b, h, s, 0);
                if(s < seqQ[static_cast<size_t>(b)])
                {
                    EXPECT_EQ(dense, lseRagged.getHostValue(std::vector<int64_t>{b, h, s, 0}))
                        << "dense LSE mismatch batch " << b << " head " << h << " token " << s;
                }
                else
                {
                    EXPECT_EQ(dense, SENTINEL)
                        << "padding row written batch " << b << " head " << h << " token " << s;
                }
            }
        }
    }
}

// --- Fully-masked branch: a zero-length-KV batch yields zero output and LSE = -inf ---

TEST(TestCpuFpReferenceSdpaRaggedFp32, ZeroLengthKvFullyMasked)
{
    const std::vector<int64_t> seqQ = {3, 2};
    const std::vector<int64_t> seqKv = {3, 0}; // batch 1: queries but no keys -> fully masked
    const int64_t numHeads = 2;
    const int64_t headDim = 16;
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto sMaxQ = maxOf(seqQ);
    const auto sMaxKv = maxOf(seqKv);
    const auto totalQ = sum(seqQ);
    const auto totalKv = sum(seqKv);
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);

    const std::vector<int64_t> qDims = {batch, numHeads, sMaxQ, headDim};
    const std::vector<int64_t> kvDims = {batch, numHeads, sMaxKv, headDim};
    const std::vector<int64_t> lseDims = {batch, numHeads, sMaxQ, 1};

    std::vector<float> qBack(static_cast<size_t>(totalQ * numHeads * headDim));
    std::vector<float> kBack(static_cast<size_t>(totalKv * numHeads * headDim));
    std::vector<float> vBack(static_cast<size_t>(totalKv * numHeads * headDim));
    std::vector<float> oBack(static_cast<size_t>(totalQ * numHeads * headDim), -1.0f); // sentinel
    std::vector<float> lseBack(static_cast<size_t>(totalQ * numHeads), 123.0f); // sentinel
    fillPacked(qBack, 11);
    fillPacked(kBack, 22);
    fillPacked(vBack, 33);

    auto q = wrapRagged(qBack.data(), qDims, numHeads * headDim, cumQ);
    auto k = wrapRagged(kBack.data(), kvDims, numHeads * headDim, cumKv);
    auto v = wrapRagged(vBack.data(), kvDims, numHeads * headDim, cumKv);
    auto o = wrapRagged(oBack.data(), qDims, numHeads * headDim, cumQ);
    auto lse = wrapRagged(lseBack.data(), lseDims, numHeads, cumQ);

    CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
        q, k, v, o, std::nullopt, -1, -1, true, &lse);

    // Batch 1 has seqKv == 0: every query is fully masked -> output 0, LSE -inf.
    const int64_t b = 1;
    for(int64_t s = 0; s < seqQ[static_cast<size_t>(b)]; ++s)
    {
        for(int64_t h = 0; h < numHeads; ++h)
        {
            const float lseVal = lse.getHostValue(std::vector<int64_t>{b, h, s, 0});
            EXPECT_TRUE(std::isinf(lseVal) && lseVal < 0.0f)
                << "expected -inf LSE at fully-masked batch " << b << " token " << s << " head "
                << h;
            for(int64_t dv = 0; dv < headDim; ++dv)
            {
                EXPECT_EQ(o.getHostValue(std::vector<int64_t>{b, h, s, dv}), 0.0f)
                    << "expected zero output at fully-masked batch " << b << " token " << s;
            }
        }
    }
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ZeroLengthQBatch)
{
    checkRaggedVsDense({3, 0, 4}, {3, 2, 4}, 2, 2, 16, 16);
}

// No query tokens at all: nothing is written.
TEST(TestCpuFpReferenceSdpaRaggedFp32, AllQueriesEmpty)
{
    constexpr float SENTINEL = -99.0f;
    const std::vector<int64_t> dims = {2, 1, 2, 16};
    std::vector<float> qB(32, 1.0f);
    std::vector<float> kB(48, 1.0f);
    std::vector<float> vB(48, 1.0f);
    std::vector<float> oB(32, SENTINEL);
    auto q = wrapRagged(qB.data(), dims, 16, {0, 0, 0});
    auto k = wrapRagged(kB.data(), dims, 16, {0, 2, 3});
    auto v = wrapRagged(vB.data(), dims, 16, {0, 2, 3});
    auto o = wrapRagged(oB.data(), dims, 16, {0, 0, 0});
    Tensor<float> lse({2, 1, 2, 1});
    lse.fillWithValue(SENTINEL);

    EXPECT_NO_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
        q, k, v, o, std::nullopt, -1, -1, true, &lse)));

    for(size_t i = 0; i < oB.size(); ++i)
    {
        EXPECT_EQ(oB[i], SENTINEL) << "output written at element " << i;
    }
    const auto* lp = lse.memory().hostData();
    for(size_t i = 0; i < lse.elementCount(); ++i)
    {
        EXPECT_EQ(lp[i], SENTINEL) << "LSE written at element " << i;
    }
}

// --- Validation (negative) cases ---

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnNonRaggedInput)
{
    // Plain (non-ragged) tensors: raggedIterationInfo() is nullopt -> reference rejects them.
    Tensor<float> q({1, 2, 4, 16});
    Tensor<float> k({1, 2, 4, 16});
    Tensor<float> v({1, 2, 4, 16});
    Tensor<float> o({1, 2, 4, 16});
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnBadLseShape)
{
    const std::vector<int64_t> dims = {1, 2, 4, 16};
    const auto cum = cumTokens({4});
    std::vector<float> qB(static_cast<size_t>(2 * 4 * 16));
    std::vector<float> kB(qB.size());
    std::vector<float> vB(qB.size());
    std::vector<float> oB(qB.size());
    auto q = wrapRagged(qB.data(), dims, int64_t{2} * 16, cum);
    auto k = wrapRagged(kB.data(), dims, int64_t{2} * 16, cum);
    auto v = wrapRagged(vB.data(), dims, int64_t{2} * 16, cum);
    auto o = wrapRagged(oB.data(), dims, int64_t{2} * 16, cum);

    Tensor<float> badLse({1, 2, 4, 2}); // last dim must be 1
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                     q, k, v, o, std::nullopt, -1, -1, true, &badLse)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnBadDescaleShape)
{
    const std::vector<int64_t> dims = {1, 2, 4, 16};
    const auto cum = cumTokens({4});
    std::vector<float> qB(static_cast<size_t>(2 * 4 * 16));
    std::vector<float> kB(qB.size());
    std::vector<float> vB(qB.size());
    std::vector<float> oB(qB.size());
    auto q = wrapRagged(qB.data(), dims, int64_t{2} * 16, cum);
    auto k = wrapRagged(kB.data(), dims, int64_t{2} * 16, cum);
    auto v = wrapRagged(vB.data(), dims, int64_t{2} * 16, cum);
    auto o = wrapRagged(oB.data(), dims, int64_t{2} * 16, cum);

    Tensor<float> badDescale({1, 3, 1, 1}); // heads (3) != H_kv (2)
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                     q, k, v, o, std::nullopt, -1, -1, true, nullptr, &badDescale)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnPerQueryHeadQDescaleUnderGqa)
{
    // Q descale is per KV head: under GQA (H_q = 4, H_kv = 2) a [B, H_q, 1, 1] Q descale is invalid.
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, {1, 4, 4, 16}, {4});
    auto k = makeValidRagged(kB, {1, 2, 4, 16}, {4});
    auto v = makeValidRagged(vB, {1, 2, 4, 16}, {4});
    auto o = makeValidRagged(oB, {1, 4, 4, 16}, {4});

    auto perQueryHead = makePerHeadDescale(1, 4, 0.5f);
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                     q, k, v, o, std::nullopt, -1, -1, true, nullptr, &perQueryHead)),
                 std::invalid_argument);
}

// --- validateInput() negative cases: each builds valid rank-4 ragged q/k/v/o with exactly one
//     dimension wrong so forward() throws in validateInput() before any addressing. ---

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnBatchMismatch)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, {1, 2, 4, 16}, {4});
    auto k = makeValidRagged(kB, {2, 2, 4, 16}, {4, 4}); // batch 2 != q batch 1
    auto v = makeValidRagged(vB, {1, 2, 4, 16}, {4});
    auto o = makeValidRagged(oB, {1, 2, 4, 16}, {4});
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnQkHeadDimMismatch)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, {1, 2, 4, 16}, {4});
    auto k = makeValidRagged(kB, {1, 2, 4, 32}, {4}); // K head_dim 32 != Q head_dim 16
    auto v = makeValidRagged(vB, {1, 2, 4, 16}, {4});
    auto o = makeValidRagged(oB, {1, 2, 4, 16}, {4});
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnKvSeqExtentMismatch)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, {1, 2, 4, 16}, {4});
    auto k = makeValidRagged(kB, {1, 2, 4, 16}, {4});
    auto v = makeValidRagged(vB, {1, 2, 6, 16}, {6}); // V S_max 6 != K S_max 4
    auto o = makeValidRagged(oB, {1, 2, 4, 16}, {4});
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

// --- Tensors sharing a packing must describe the same per-batch sequence lengths ---

// Reviewer repro: K lengths {2, 1} and V lengths {1, 2} with the same S_max, zero Q/K, packed V
// {10, 20, 30}. Unchecked, batch 0 averaged in batch 1's V row (output {15, 20}).
TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnKvSequenceLengthMismatch)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, {2, 1, 1, 1}, {1, 1});
    auto k = makeValidRagged(kB, {2, 1, 2, 1}, {2, 1});
    auto v = makeValidRagged(vB, {2, 1, 2, 1}, {1, 2});
    auto o = makeValidRagged(oB, {2, 1, 1, 1}, {1, 1});
    vB[0] = 10.0f;
    vB[1] = 20.0f;
    vB[2] = 30.0f;
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnQoSequenceLengthMismatch)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, {2, 2, 2, 16}, {2, 1});
    auto k = makeValidRagged(kB, {2, 2, 2, 16}, {2, 2});
    auto v = makeValidRagged(vB, {2, 2, 2, 16}, {2, 2});
    auto o = makeValidRagged(oB, {2, 2, 2, 16}, {1, 2}); // O lengths != Q lengths
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnRaggedLseSequenceLengthMismatch)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    std::vector<float> lseB;
    auto q = makeValidRagged(qB, {2, 1, 2, 16}, {2, 1});
    auto k = makeValidRagged(kB, {2, 1, 2, 16}, {2, 2});
    auto v = makeValidRagged(vB, {2, 1, 2, 16}, {2, 2});
    auto o = makeValidRagged(oB, {2, 1, 2, 16}, {2, 1});
    auto lse = makeValidRagged(lseB, {2, 1, 2, 1}, {1, 2}); // LSE lengths != Q lengths
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                     q, k, v, o, std::nullopt, -1, -1, true, &lse)),
                 std::invalid_argument);
}

// A dense (non-ragged) output cannot follow the packed Q layout.
TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnNonRaggedOutput)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    auto q = makeValidRagged(qB, {1, 2, 4, 16}, {4});
    auto k = makeValidRagged(kB, {1, 2, 4, 16}, {4});
    auto v = makeValidRagged(vB, {1, 2, 4, 16}, {4});
    Tensor<float> o({1, 2, 4, 16});
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

namespace
{

// Overwrites a ragged tensor's offset table after construction. The tensor checked only the
// original table, so the reference has to catch a bad edited one itself.
void setTokenOffsets(ITensor& aux, const std::vector<int64_t>& tokens, int64_t seqStride)
{
    auto& offsets = static_cast<Tensor<int32_t>&>(aux);
    for(size_t i = 0; i < tokens.size(); ++i)
    {
        offsets.setHostValue(
            static_cast<int32_t>(tokens[i] * seqStride), static_cast<int64_t>(i), 0, 0, 0);
    }
}

// B = 2, H = 1, D = 16, S_max = 2 over 4-token buffers, built valid with lengths {2, 2}. Q's and
// O's offsets are then replaced by qTokens. Returns whether forward() threw std::invalid_argument.
bool throwsOnEditedQTokens(const std::vector<int64_t>& qTokens)
{
    const std::vector<int64_t> dims = {2, 1, 2, 16};
    const std::vector<int64_t> valid = {0, 2, 4};
    std::vector<float> qB(64, 0.0f);
    std::vector<float> kB(64, 0.0f);
    std::vector<float> vB(64, 1.0f);
    std::vector<float> oB(64, 0.0f);
    auto qAux = makeRaggedOffsetAux(valid, 16);
    auto oAux = makeRaggedOffsetAux(valid, 16);
    ShallowRaggedTensor<float> q(qB.data(), dims, bshd(dims), SEQ_AXIS, qAux);
    auto k = wrapRagged(kB.data(), dims, 16, valid);
    auto v = wrapRagged(vB.data(), dims, 16, valid);
    ShallowRaggedTensor<float> o(oB.data(), dims, bshd(dims), SEQ_AXIS, oAux);
    setTokenOffsets(*qAux, qTokens, 16);
    setTokenOffsets(*oAux, qTokens, 16);
    try
    {
        CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o);
    }
    catch(const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

} // namespace

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnBadOffsetTable)
{
    EXPECT_TRUE(throwsOnEditedQTokens({1, 2, 4})) << "ragged_offset[0] != 0 accepted";
    EXPECT_TRUE(throwsOnEditedQTokens({0, 3, 4})) << "batch longer than S_max accepted";
    EXPECT_FALSE(throwsOnEditedQTokens({0, 2, 3}));
}

// A dense LSE with a smaller Sq than Q would take rows from the next batch.
TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnLseShorterThanQ)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, {2, 1, 2, 16}, {2, 1});
    auto k = makeValidRagged(kB, {2, 1, 2, 16}, {2, 2});
    auto v = makeValidRagged(vB, {2, 1, 2, 16}, {2, 2});
    auto o = makeValidRagged(oB, {2, 1, 2, 16}, {2, 1});
    Tensor<float> lse({2, 1, 1, 1});
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
                     q, k, v, o, std::nullopt, -1, -1, true, &lse)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnNonDivisibleHeads)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, {1, 4, 4, 16}, {4});
    auto k = makeValidRagged(kB, {1, 3, 4, 16}, {4}); // 4 % 3 != 0
    auto v = makeValidRagged(vB, {1, 3, 4, 16}, {4});
    auto o = makeValidRagged(oB, {1, 4, 4, 16}, {4});
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, ThrowsOnBadOutputShape)
{
    std::vector<float> qB;
    std::vector<float> kB;
    std::vector<float> vB;
    std::vector<float> oB;
    auto q = makeValidRagged(qB, {1, 2, 4, 16}, {4});
    auto k = makeValidRagged(kB, {1, 2, 4, 16}, {4});
    auto v = makeValidRagged(vB, {1, 2, 4, 16}, {4});
    auto o = makeValidRagged(oB, {1, 2, 4, 32}, {4}); // O head_dim 32 != V head_dim 16
    EXPECT_THROW((CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(q, k, v, o)),
                 std::invalid_argument);
}
