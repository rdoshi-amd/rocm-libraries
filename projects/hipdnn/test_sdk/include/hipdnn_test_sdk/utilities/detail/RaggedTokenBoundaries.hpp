// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_test_sdk::detail
{

// Turns an RFC-0014 ragged_offset table (B+1 cumulative element offsets) into token boundaries:
// offset[b] / seqStride. Tensors with different per-token widths share a packing but not their
// element offsets, so cross-tensor checks compare token boundaries, never raw offsets.
// Throws std::invalid_argument unless offset[0] == 0, every offset is a whole number of tokens,
// offsets never decrease, and no batch is longer than sMax (the tensor's dims()[2]).
inline std::vector<int64_t> raggedTokenBoundaries(const std::vector<int64_t>& elementOffsets,
                                                  int64_t seqStride,
                                                  int64_t sMax,
                                                  const std::string& who,
                                                  const char* name)
{
    if(seqStride <= 0)
    {
        throw std::invalid_argument(who + ": " + name + " sequence stride must be positive");
    }
    if(!elementOffsets.empty() && elementOffsets.front() != 0)
    {
        throw std::invalid_argument(who + ": " + name + " ragged_offset[0] must be 0 (got "
                                    + std::to_string(elementOffsets.front()) + ")");
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
        tokens.push_back(offset / seqStride);
        if(b == 0)
        {
            continue;
        }
        const auto length = tokens[b] - tokens[b - 1];
        if(length < 0)
        {
            throw std::invalid_argument(who + ": " + name
                                        + " ragged_offset must be non-decreasing");
        }
        if(length > sMax)
        {
            throw std::invalid_argument(who + ": " + name + " batch " + std::to_string(b - 1)
                                        + " has " + std::to_string(length)
                                        + " tokens, more than S_max = " + std::to_string(sMax));
        }
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
