// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// CDNA5 hardware hazard rules shared between CDNA5ReadyQueue (DAG scheduler)
// and HazardGapAnalysisPass. Keep this header free of CDNA5ReadyQueue internals.

#include <vector>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {

// Which operand of the producer opens the gap, and which of the consumer closes it.
enum class HazardDir {
    WriteThenRead,  // producer writes a reg, consumer reads it (RAW-shaped)
    ReadThenWrite,  // producer reads a reg, consumer overwrites it (WAR-shaped)
};

// What the gap is counted in, which decides whether elapsing time can pay it.
enum class HazardUnit {
    Cycles,   // decays in advanceTime; reports a wait the scheduler may pay
    PipeOps,  // advances only when an isPipeOp instruction issues; not payable by time
};

// Which registers a gap is tracked on.
enum class HazardScope {
    // The consumer reads a register the producer wrote: one gate per register.
    SameRegister,
    // The consumer reads any register of the scalar family after the producer wrote any
    // of them: one gate for the whole family (kAnyRegisterHazardKey). The hardware keeps
    // such a gap on a counter of writes in flight, not per register.
    AnyRegister,
};

// Gate key of an AnyRegister rule; never a valid per-register key.
inline constexpr int kAnyRegisterHazardKey = -1;

struct HazardRule {
    const char* name;
    bool (*isProducer)(const StinkyInstruction&);
    bool (*isConsumer)(const StinkyInstruction&);
    RegType regType;
    // Cycles rules: the gap itself. PipeOps rules: 0 means the arch policy supplies it.
    int distance;
    HazardDir dir;
    HazardUnit unit;
    // PipeOps rules only: what advances the counter. Null for Cycles rules.
    bool (*isPipeOp)(const StinkyInstruction&);
    // AnyRegister ignores regType: producer and consumer test the scalar family.
    HazardScope scope;
};

// SALU sgpr -> any SMEM/tensor_load/VMEM address consumer.
inline bool isSaluHazardConsumer(const StinkyInstruction& inst) {
    return isGlobalMemLoad(inst) || isTensorLoad(inst);
}

// VALU vgpr -> VMEM address consumer (global_read / MUBUF / FLAT / global_prefetch).
inline bool isVmemAddrHazardConsumer(const StinkyInstruction& inst) {
    return isBufferMemLoad(inst) || isGlobalPrefetch(inst);
}

// Registers the SALU writes and a VALU can read as a uniform operand. SCC is not one:
// a VALU never reads it.
inline bool isScalarFamilyReg(RegType type) {
    switch (type) {
        case RegType::S:
        case RegType::VCC:
        case RegType::VCC_LO:
        case RegType::VCC_HI:
        case RegType::EXEC:
        case RegType::EXEC_LO:
        case RegType::EXEC_HI:
            return true;
        default:
            return false;
    }
}

inline bool hasScalarFamilyReg(const std::vector<StinkyRegister>& regs) {
    for (const StinkyRegister& r : regs)
        if (r.isRegister() && isScalarFamilyReg(r.reg.type)) return true;
    return false;
}

// A SALU that writes a scalar register; an SCC-only writer such as s_cmp does not count.
inline bool isScalarWritingSalu(const StinkyInstruction& inst) {
    return isScalarALU(inst) && hasScalarFamilyReg(inst.getDestRegs());
}

inline bool isNonMatrixValu(const StinkyInstruction& inst) {
    return isVectorALU(inst) && !isMatrixInstruction(inst);
}

// A VALU with a scalar or VCC source operand.
inline bool isValuWithScalarSrc(const StinkyInstruction& inst) {
    return isNonMatrixValu(inst) && hasScalarFamilyReg(inst.getSrcRegs());
}

inline constexpr HazardRule kCdna5HazardRules[] = {
    {"SaluSgprToMemAddr", isScalarALU, isSaluHazardConsumer, RegType::S, 8,
     HazardDir::WriteThenRead, HazardUnit::Cycles, nullptr, HazardScope::SameRegister},
    {"ValuVgprToVmemAddr", isVectorALU, isVmemAddrHazardConsumer, RegType::V, 32,
     HazardDir::WriteThenRead, HazardUnit::Cycles, nullptr, HazardScope::SameRegister},
    // mode2 WAR: a WMMA reads a vgpr, a later ds_load overwrites it. The gap is counted in
    // issued matrix ops, not cycles, so elapsing time cannot pay it -- hence PipeOps.
    {"WmmaVgprSrcToDsWrite", isMatrixInstruction, isDSRead, RegType::V, 0, HazardDir::ReadThenWrite,
     HazardUnit::PipeOps, isMatrixInstruction, HazardScope::SameRegister},
};
inline constexpr int kNumCdna5HazardRules =
    static_cast<int>(sizeof(kCdna5HazardRules) / sizeof(kCdna5HazardRules[0]));

// Interlocks the hardware enforces by holding the consumer, with no instruction in the
// code. Breaking one costs a stall, never a wrong result, so they are performance rules:
// the scheduler applies them under dagFeatures.scalarInterlocks and HazardGapAnalysisPass
// reports them without failing.
inline constexpr HazardRule kCdna5InterlockRules[] = {
    // A VALU reading any scalar register waits on the latest SALU write to any of them.
    {"SaluSgprToValuScalarSrc", isScalarWritingSalu, isValuWithScalarSrc, RegType::S, 9,
     HazardDir::WriteThenRead, HazardUnit::Cycles, nullptr, HazardScope::AnyRegister},
    // The carry a VALU writes to VCC (v_add_co_u32) as read by the next (v_add_co_ci_u32).
    {"ValuVccToValuVccSrc", isNonMatrixValu, isNonMatrixValu, RegType::VCC_LO, 8,
     HazardDir::WriteThenRead, HazardUnit::Cycles, nullptr, HazardScope::SameRegister},
};
inline constexpr int kNumCdna5InterlockRules =
    static_cast<int>(sizeof(kCdna5InterlockRules) / sizeof(kCdna5InterlockRules[0]));

// The dir/unit combinations a consumer implements. Anything else is skipped by both
// HazardGapAnalysisPass and the pipe-op lanes, so it would be a rule that does nothing.
constexpr bool hazardRuleImplemented(const HazardRule& r) {
    if (r.scope == HazardScope::AnyRegister)
        return r.dir == HazardDir::WriteThenRead && r.unit == HazardUnit::Cycles &&
               r.isPipeOp == nullptr;
    if (r.dir == HazardDir::WriteThenRead && r.unit == HazardUnit::Cycles)
        return r.isPipeOp == nullptr;
    if (r.dir == HazardDir::ReadThenWrite && r.unit == HazardUnit::PipeOps)
        return r.isPipeOp != nullptr;
    return false;
}

constexpr bool hazardRulesWellFormed() {
    for (const HazardRule& r : kCdna5HazardRules)
        if (!hazardRuleImplemented(r)) return false;
    for (const HazardRule& r : kCdna5InterlockRules)
        if (!hazardRuleImplemented(r)) return false;
    return true;
}

static_assert(hazardRulesWellFormed(),
              "hazard rule combination has no consumer -- implement it or drop the rule");

}  // namespace stinkytofu
