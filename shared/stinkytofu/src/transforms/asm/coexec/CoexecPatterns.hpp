// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>
#include <vector>

#include "CoexecModel.hpp"
#include "stinkytofu/hardware/HWModel.hpp"

namespace stinkytofu {
namespace coexec {

/// A proposed move: `unit` is taken out of the order and put back, in the same relative
/// order, before the instruction at `slot` of what is left.
struct Move {
    std::vector<int> unit;
    int slot = 0;
    int toWindow = 0;
    unsigned pattern = 0;
};

/// Where the fillers of one block may go. Built from the input order, whose positions
/// are the ids; queried with the current order.
class MoveSpace {
   public:
    struct Inputs {
        std::vector<int> segment;          ///< per id; -1 for a hard boundary
        std::vector<bool> filler;          ///< per id: a SALU or VALU that may move
        std::vector<bool> wmma;            ///< per id: closes a window
        std::vector<bool> noInsertBefore;  ///< per id: a wait after the first of its group,
                                           ///< or the v_wmma a wait group guards
        std::vector<std::vector<int>> preds;
        std::vector<std::vector<int>> succs;
        /// per id: the fillers that move with it, itself included, in input order
        std::vector<std::vector<int>> units;
    };

    explicit MoveSpace(Inputs inputs);

    int size() const {
        return static_cast<int>(in_.segment.size());
    }
    bool isFiller(int id) const {
        return in_.filler[id];
    }
    const std::vector<int>& unitOf(int id) const {
        return in_.units[id];
    }
    /// \p later depends on \p earlier through a chain of dependences.
    bool dependsOn(int later, int earlier) const;

    /// Every legal slot of \p unit in \p order (\p pos: id -> position), as (slot,
    /// window) pairs in ascending order. Slots count positions of the order without the
    /// unit; the window of a slot is the number of v_wmma before it.
    std::vector<std::pair<int, int>> legalSlots(const std::vector<int>& order,
                                                const std::vector<int>& pos,
                                                const std::vector<int>& unit) const;

    std::vector<int> apply(const std::vector<int>& order, const Move& move) const;

    /// Every dependence points forward, and every segment and fixed instruction is where
    /// the input had it.
    bool respects(const std::vector<int>& order) const;

   private:
    Inputs in_;
};

/// What the patterns look at: the block in its current order and its simulation.
struct PatternContext {
    const CoexecModel& model;
    const MoveSpace& space;
    const std::vector<int>& order;
    const std::vector<int>& pos;
    const SimResult& sim;
    const HWModel::CoexecTiming& timing;
};

/// The patterns in \p mask (CoexecRepairPattern bits) that explain the idle of \p window.
unsigned explainingPatterns(const PatternContext& ctx, int window, unsigned mask);

/// The moves the patterns in \p mask propose for \p window, in a fixed order.
std::vector<Move> proposeMoves(const PatternContext& ctx, int window, unsigned mask);

const char* patternName(unsigned pattern);

}  // namespace coexec
}  // namespace stinkytofu
