// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/PassManager.hpp"

namespace stinkytofu {
struct HazardRule;
struct StinkyInstruction;

/// One producer->consumer pair of a hazard rule: positions in the instruction sequence
/// and the cycles of the instructions strictly between them.
struct HazardGapPair {
    int producer;
    int consumer;
    int gap;
};

/// Every pair of \p rule in \p instrs, one basic block's order, the way this analysis
/// pairs them. Only write-then-read rules counted in cycles are measured; for any other
/// rule the result is empty.
STINKYTOFU_EXPORT std::vector<HazardGapPair> findHazardGapPairs(
    const std::vector<const StinkyInstruction*>& instrs, const HazardRule& rule);

/// Scan each basic block for kCdna5HazardRules producer->consumer pairs and
/// report the cycle gap between them (using real issueCycles/latencyCycles, not
/// instruction count). Prints a per-rule summary and, per consumer, the tightest
/// producer gap. Exits with a non-zero status if any gap is below the rule threshold.
///
/// Usage:  stinkytofu-opt --arch gfx1250 kernel.s --HazardGapAnalysisPass
///
/// Optional args (comma-separated after '='):
///   verbose   — print every producer->consumer pair, not just violations
STINKYTOFU_EXPORT std::unique_ptr<Pass> createHazardGapAnalysisPass(bool verbose = false);

}  // namespace stinkytofu
