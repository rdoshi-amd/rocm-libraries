// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/VgprMsbPlanner.hpp"

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"

namespace stinkytofu {
namespace {

bool isMsbComputableClass(const StinkyInstruction& inst) {
    return !(inst.is(InstFlag::IF_SALU) || inst.is(InstFlag::IF_SMemLoad) ||
             inst.is(InstFlag::IF_SMemStore) || inst.is(InstFlag::IF_SMemAtomic) ||
             inst.is(InstFlag::IF_Branch) || inst.is(InstFlag::IF_Call) ||
             inst.is(InstFlag::IF_Barrier) || inst.is(InstFlag::IF_WaitCnt) ||
             inst.is(InstFlag::IF_HasSideEffect));
}

bool preferInsertAfter(const StinkyInstruction& inst) {
    return isVectorALU(inst) || (isScalarALU(inst) && !isBarrier(inst)) ||
           isMatrixInstruction(inst);
}

}  // namespace

void VgprMsbPlanner::beginBlock(int entryState) {
    state_ = entryState;
    anchor_ = nullptr;
    anchorOnNext_ = false;
}

std::optional<PlannedMsbSwitch> VgprMsbPlanner::observe(const StinkyInstruction& inst) {
    if (anchorOnNext_) {
        anchor_ = &inst;
        anchorOnNext_ = false;
    }

    if (isLabel(inst)) {
        state_ = kLabelBegin;
        anchor_ = nullptr;
        return std::nullopt;
    }

    if (isPseudoInst(&inst)) return std::nullopt;

    // A call (e.g. s_swappc_b64) transfers to a callee that may leave the VGPR MSB
    // hardware register in an unknown state, so the next VGPR op re-establishes it. A
    // deferred anchor never crosses the call: post-call rebuilds must stay post-call.
    if (isCall(inst)) {
        state_ = kNotRequired;
        anchor_ = nullptr;
        return std::nullopt;
    }

    const StinkyInstruction& insertBefore = anchor_ ? *anchor_ : inst;
    auto [required, hasVgpr] = computeRequiredMsb(&inst);
    std::optional<PlannedMsbSwitch> planned;
    if (hasVgpr)
        planned = switchTo(required, insertBefore);
    else if (state_ == kLabelBegin)
        state_ = kNotRequired;

    if (planned || isMsbComputableClass(inst)) anchor_ = nullptr;
    if (preferInsertAfter(inst)) anchorOnNext_ = true;
    return planned;
}

std::optional<PlannedMsbSwitch> VgprMsbPlanner::require(int state,
                                                        const StinkyInstruction& insertBefore) {
    std::optional<PlannedMsbSwitch> planned = switchTo(state, insertBefore);
    if (planned) anchor_ = nullptr;
    return planned;
}

std::optional<PlannedMsbSwitch> VgprMsbPlanner::switchTo(int state,
                                                         const StinkyInstruction& insertBefore) {
    if (state == state_) return std::nullopt;

    PlannedMsbSwitch planned{&insertBefore, state, state, state_ == kLabelBegin};
    if (mode_ == VgprMsbMode::Msb16 && state_ >= 0) planned.value += state_ << 8;
    state_ = state;
    return planned;
}

}  // namespace stinkytofu
