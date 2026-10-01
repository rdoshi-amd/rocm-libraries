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

// Strides for packed BSHD memory under [B, H, S, D] dims. The seq stride is H * D.
inline std::vector<int64_t> bshd(const std::vector<int64_t>& dims)
{
    return {dims[1] * dims[2] * dims[3], dims[3], dims[1] * dims[3], 1};
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
