// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/pipeline/CloneSpec.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"

namespace stinkytofu {
class Pass;

/// Bits of CoexecSimRepairOptions::patternMask.
enum CoexecRepairPattern : unsigned {
    kCoexecScalarShadow = 1u << 0,  ///< P1: scalar writes out of a VALU's interlock shadow
    kCoexecVccPair = 1u << 1,       ///< P2: a VCC producer away from its reader
    kCoexecOverfull = 1u << 2,      ///< P3: work out of a window too full for the wave's lead
    kCoexecFallback = 1u << 3,      ///< any filler of a blamed window, by one to three windows
    kCoexecAllPatterns = 0xFu,
};

struct CoexecSimRepairOptions {
    /// The options InsertWaitAlu runs with, so the s_wait_alu it emits can be predicted.
    InsertWaitAluOptions waitAlu;
    /// False when InsertWaitAlu does not run (no expert schedule mode 2).
    bool predictWaitAlu = true;
    /// RegionClonePass jobs; nothing moves across their split points.
    std::vector<CloneSpec> clones;
    unsigned patternMask = kCoexecAllPatterns;
    /// A block is rewritten only if its predicted idle drops by at least
    /// max(marginCycles, marginFraction * its predicted idle).
    int marginCycles = 8;
    float marginFraction = 0.02f;
    int maxMovesPerWindow = 2;
};

/// Matrix co-execution repair driven by a machine model (coexec/CoexecModel).
///
/// For each block it simulates two iterations of the order, picks the window with the
/// most matrix-pipe idle that a pattern explains, simulates the moves the patterns
/// propose and applies the best, until no move gains a cycle. Only SALU and VALU move,
/// within the ranges the register, counter and prefetch-pin dependences allow, never
/// across a hard boundary or a RegionClonePass split point; v_wmma, ds_load, waits,
/// barriers, tensor loads and prefetches keep their places. The result is kept only if
/// its predicted idle beats the input by the margin and it passes the dependence and
/// hazard checks. Every move is reported with an optimization remark.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createCoexecSimRepairPass(
    CoexecSimRepairOptions options = {});

}  // namespace stinkytofu
