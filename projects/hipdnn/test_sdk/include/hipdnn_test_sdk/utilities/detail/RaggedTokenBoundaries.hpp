// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_test_sdk::detail
{

// Converts an RFC-0014 ragged_offset table (B+1 cumulative ELEMENT offsets) into token boundaries,
// tokenBoundary[b] = offset[b] / seqStride. Tensors of different per-token widths (e.g. K with
// H_k*D and V with H_v*D_v) have different element offsets for the same packing, so consistency
// between tensors is checked on token boundaries, never on raw offsets. Throws std::invalid_argument
// unless every offset is a whole number of tokens and the offsets are non-decreasing.
inline std::vector<int64_t> raggedTokenBoundaries(const std::vector<int64_t>& elementOffsets,
                                                  int64_t seqStride,
                                                  const std::string& who,
                                                  const char* name)
{
    if(seqStride <= 0)
    {
        throw std::invalid_argument(who + ": " + name + " sequence stride must be positive");
    }
    std::vector<int64_t> tokens;
    tokens.reserve(elementOffsets.size());
    for(size_t b = 0; b < elementOffsets.size(); ++b)
    {
        const auto offset = elementOffsets[b];
        if(offset % seqStride != 0)
        {
            throw std::invalid_argument(who + ": " + name + " ragged_offset[" + std::to_string(b)
                                        + "] = " + std::to_string(offset)
                                        + " is not a whole number of tokens (seq stride "
                                        + std::to_string(seqStride) + ")");
        }
        if(b > 0 && offset < elementOffsets[b - 1])
        {
            throw std::invalid_argument(who + ": " + name
                                        + " ragged_offset must be non-decreasing");
        }
        tokens.push_back(offset / seqStride);
    }
    return tokens;
}

// Throws std::invalid_argument unless two tensors that must share a packing (Q/O, K/V, Q/LSE) have
// identical token boundaries, i.e. the same per-batch sequence lengths at the same positions.
inline void requireMatchingTokenBoundaries(const std::vector<int64_t>& a,
                                           const char* aName,
                                           const std::vector<int64_t>& b,
                                           const char* bName,
                                           const std::string& who)
{
    if(a.size() != b.size())
    {
        throw std::invalid_argument(who + ": " + aName + " and " + bName
                                    + " ragged_offset tables differ in length");
    }
    for(size_t i = 0; i < a.size(); ++i)
    {
        if(a[i] != b[i])
        {
            throw std::invalid_argument(who + ": " + aName + " and " + bName
                                        + " per-batch sequence lengths differ (token boundary "
                                        + std::to_string(i) + ": " + std::to_string(a[i]) + " vs "
                                        + std::to_string(b[i]) + ")");
        }
    }
}

} // namespace hipdnn_test_sdk::detail
