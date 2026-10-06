// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/Types.hpp"

namespace stinkytofu {
struct StinkyInstruction;

/// One s_set_vgpr_msb the InsertVgprMsbPass walk places.
struct PlannedMsbSwitch {
    /// The switch, and its s_nop when withNop, go immediately before this instruction.
    const StinkyInstruction* insertBefore;
    /// The s_set_vgpr_msb immediate. In Msb16 mode bits [15:8] hold the previous state.
    int value;
    /// The MSB state after the switch.
    int state;
    /// First switch after a label: an s_nop 0 goes before it.
    bool withNop;
};

/// The s_set_vgpr_msb placement walk of InsertVgprMsbPass without the IR edits, so a
/// scheduler can predict the switches an instruction order costs and the pass can
/// materialize them.
///
/// Feed every StinkyInstruction of a basic block in program order, labels and pseudo
/// instructions included: the deferred insertion anchor is the instruction observed right
/// after the last VALU / SALU / matrix instruction.
class STINKYTOFU_EXPORT VgprMsbPlanner {
   public:
    /// No MSB state is known in this block yet.
    static constexpr int kNotRequired = -1;
    /// A label was just passed: the next switch needs an s_nop before it.
    static constexpr int kLabelBegin = -2;

    explicit VgprMsbPlanner(VgprMsbMode mode) : mode_(mode) {}

    /// Start a basic block whose entry state is \p entryState (kNotRequired, kLabelBegin or
    /// a known MSB state).
    void beginBlock(int entryState = kNotRequired);

    /// Account for \p inst; returns the switch to emit for it, if any.
    std::optional<PlannedMsbSwitch> observe(const StinkyInstruction& inst);

    /// Switch to \p state immediately before \p insertBefore unless already in it.
    std::optional<PlannedMsbSwitch> require(int state, const StinkyInstruction& insertBefore);

    int state() const {
        return state_;
    }

   private:
    std::optional<PlannedMsbSwitch> switchTo(int state, const StinkyInstruction& insertBefore);

    VgprMsbMode mode_;
    int state_ = kNotRequired;
    const StinkyInstruction* anchor_ = nullptr;
    bool anchorOnNext_ = false;
};

}  // namespace stinkytofu
