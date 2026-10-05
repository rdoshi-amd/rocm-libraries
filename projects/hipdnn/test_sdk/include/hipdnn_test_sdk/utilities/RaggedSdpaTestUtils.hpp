// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace hipdnn_test_sdk::utilities
{

// Geometry helpers for ragged SDPA tests (RFC-0014), shared by the CPU test_sdk suite and the
// gpu-ref integration suite.

// SDPA dims are [B, H, S, D] with the sequence at axis 2. The packed BSHD layout lives only in the
// strides, so use 2 here, not the SDK's BSHD_SEQ_AXIS = 1 (which assumes dims [B, S, H, D]).
inline constexpr int SEQ_AXIS = 2;

// Dims of a ragged SDPA tensor from its batch, sequence extent (S_max), heads and head dim.
// Ragged tests build every ragged shape, stride and element index through these helpers, so the
// axis order lives here only.
inline std::vector<int64_t> raggedDims(int64_t batch, int64_t seq, int64_t heads, int64_t dim)
{
    return {batch, heads, seq, dim};
}

// Logical index of element (b, s, h, d) in a tensor shaped by raggedDims.
inline std::vector<int64_t> raggedIndex(int64_t b, int64_t s, int64_t h, int64_t d)
{
    return {b, h, s, d};
}

inline int64_t raggedSeqExtent(const std::vector<int64_t>& dims)
{
    return dims[SEQ_AXIS];
}

inline int64_t raggedHeads(const std::vector<int64_t>& dims)
{
    return dims[1];
}

// Strides of packed BSHD memory (token-major: token, then head, then dim) for raggedDims dims.
// The seq stride is H * D.
inline std::vector<int64_t> raggedStrides(const std::vector<int64_t>& dims)
{
    const auto heads = raggedHeads(dims);
    const auto seq = raggedSeqExtent(dims);
    const auto dim = dims[3];
    return {seq * heads * dim, dim, heads * dim, 1};
}

// Exclusive prefix sum: cum[0] = 0, cum[b + 1] = cum[b] + lengths[b].
inline std::vector<int64_t> cumTokens(const std::vector<int64_t>& lengths)
{
    std::vector<int64_t> cum(lengths.size() + 1, 0);
    for(size_t i = 0; i < lengths.size(); ++i)
    {
        cum[i + 1] = cum[i] + lengths[i];
    }
    return cum;
}

// RFC-0014 ragged_offset aux: int32 [B + 1, 1, 1, 1] holding cum * seqStride in element units.
inline std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
    makeRaggedOffsetAux(const std::vector<int64_t>& cum, int64_t seqStride)
{
    auto aux = std::make_shared<hipdnn_data_sdk::utilities::Tensor<int32_t>>(
        std::vector<int64_t>{static_cast<int64_t>(cum.size()), 1, 1, 1});
    for(size_t i = 0; i < cum.size(); ++i)
    {
        aux->setHostValue(
            static_cast<int32_t>(cum[i] * seqStride), static_cast<int64_t>(i), 0, 0, 0);
    }
    return aux;
}

} // namespace hipdnn_test_sdk::utilities
