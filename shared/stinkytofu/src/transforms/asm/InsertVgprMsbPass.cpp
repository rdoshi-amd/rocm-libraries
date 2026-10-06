/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"

#include <cassert>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/transforms/asm/VgprMsbPlanner.hpp"

namespace stinkytofu {
namespace {

// Set offset = -msb*256 on each VGPR operand so the emitter prints byte form
// (`v[idx + offset]` evaluates to idx ≤ 255).
void encodeVgprOperands(StinkyInstruction* inst) {
    auto rewrite = [](StinkyRegister& reg) {
        if (reg.dataType != StinkyRegister::Type::Register) return;
        if (reg.reg.type != RegType::V) return;
        int msb = static_cast<int>(reg.reg.idx) / 256;
        if (msb == 0) return;  // already byte-form; nothing to do
        int wantOffset = -msb * 256;
        if (reg.reg.offset == wantOffset) return;  // already encoded (rocisa path)
        reg.reg.offset = static_cast<int16_t>(wantOffset);
    };
    for (auto& src : const_cast<std::vector<StinkyRegister>&>(inst->getSrcRegs())) rewrite(src);
    for (auto& dst : const_cast<std::vector<StinkyRegister>&>(inst->getDestRegs())) rewrite(dst);
}

void emitVgprMsb(const PlannedMsbSwitch& planned, AsmIRBuilder& irBuilder, GfxArchID archId) {
    IRBase* insertBefore = const_cast<StinkyInstruction*>(planned.insertBefore);
    if (planned.withNop) {
        StinkyInstruction* nopInst =
            irBuilder.create(getMCIDByUOp(GFX::s_nop, archId), insertBefore);
        nopInst->addSrcReg(StinkyRegister(0));
    }

    const HwInstDesc* desc = getMCIDByUOp(GFX::s_set_vgpr_msb, archId);
    assert(desc != nullptr && "s_set_vgpr_msb is not supported on this architecture");
    StinkyInstruction* msbInst = irBuilder.create(desc, insertBefore);
    msbInst->addSrcReg(StinkyRegister(planned.value));

    std::string msbComment = "src0: " + std::to_string(decodeVgprMsbForSlot(planned.state, 0)) +
                             ", src1: " + std::to_string(decodeVgprMsbForSlot(planned.state, 1)) +
                             ", src2: " + std::to_string(decodeVgprMsbForSlot(planned.state, 2)) +
                             ", dst: " + std::to_string(decodeVgprMsbForSlot(planned.state, 3));
    msbInst->addModifier<CommentData>(CommentData{msbComment});
}

std::vector<StinkyInstruction*> stinkyInstructions(BasicBlock& bb) {
    std::vector<StinkyInstruction*> insts;
    for (IRBase& ir : bb)
        if (auto* inst = dyn_cast<StinkyInstruction>(&ir)) insts.push_back(inst);
    return insts;
}

/// How many instructions name each label, as a branch target or as an operand.
std::unordered_map<std::string, int> countLabelReferences(Function& func) {
    std::unordered_map<std::string, int> refs;
    for (BasicBlock& bb : func) {
        for (StinkyInstruction* inst : stinkyInstructions(bb)) {
            if (isLabel(*inst)) continue;
            std::unordered_set<std::string> named;
            if (const auto* label = inst->getModifier<LabelData>()) named.insert(label->label);
            for (const StinkyRegister& src : inst->getSrcRegs())
                if (src.dataType == StinkyRegister::Type::LiteralString)
                    named.insert(src.getLiteralString());
            for (const std::string& name : named) ++refs[name];
        }
    }
    return refs;
}

/// A diamond `head: ...; s_cmp; s_cbranch join` / `middle` / `join: label; ...` laid out in
/// that order, where only head's branch names join's label and middle has no label, branch,
/// call or VGPR operand. Both paths into join leave the state head exits in, so join can
/// start in the state its first VGPR instruction needs once head switches to it.
struct LabelJoin {
    const StinkyInstruction* hoistBefore;
    const BasicBlock* join;
    int joinState;
};

bool isVgprFreeFallThrough(BasicBlock& bb) {
    for (StinkyInstruction* inst : stinkyInstructions(bb)) {
        if (isLabel(*inst) || isBranch(*inst) || isCall(*inst)) return false;
        if (computeRequiredMsb(inst).second) return false;
    }
    return true;
}

std::optional<int> firstRequiredState(const std::vector<StinkyInstruction*>& joinInsts) {
    for (size_t i = 1; i < joinInsts.size(); ++i) {
        const StinkyInstruction& inst = *joinInsts[i];
        if (isLabel(inst) || isCall(inst)) return std::nullopt;
        auto [required, hasVgpr] = computeRequiredMsb(&inst);
        if (hasVgpr) return required;
    }
    return std::nullopt;
}

std::unordered_map<const BasicBlock*, LabelJoin> findLabelJoins(Function& func) {
    const std::unordered_map<std::string, int> refs = countLabelReferences(func);
    std::vector<BasicBlock*> blocks;
    for (BasicBlock& bb : func) blocks.push_back(&bb);

    std::unordered_map<const BasicBlock*, LabelJoin> joins;
    for (size_t i = 0; i + 2 < blocks.size(); ++i) {
        const std::vector<StinkyInstruction*> head = stinkyInstructions(*blocks[i]);
        if (head.empty() || !isConditionalBranch(*head.back())) continue;

        const std::vector<StinkyInstruction*> join = stinkyInstructions(*blocks[i + 2]);
        if (join.empty() || !isLabel(*join.front())) continue;
        const auto* joinLabel = join.front()->getModifier<LabelData>();
        if (!joinLabel || getBranchTarget(*head.back()) != joinLabel->label) continue;
        auto ref = refs.find(joinLabel->label);
        if (ref == refs.end() || ref->second != 1) continue;

        if (!isVgprFreeFallThrough(*blocks[i + 1])) continue;
        std::optional<int> joinState = firstRequiredState(join);
        if (!joinState) continue;

        const StinkyInstruction* hoistBefore = head.back();
        if (head.size() >= 2) {
            const StinkyInstruction& cmp = *head[head.size() - 2];
            if (isScalarALU(cmp) && cmp.is(InstFlag::IF_ImplicitWriteSCC)) hoistBefore = &cmp;
        }
        joins.emplace(blocks[i], LabelJoin{hoistBefore, blocks[i + 2], *joinState});
    }
    return joins;
}

class InsertVgprMsbPassImpl : public Pass {
   public:
    static char ID;

    explicit InsertVgprMsbPassImpl(InsertVgprMsbOptions options) : options_(options) {}

    const char* getName() const override {
        return "Insert VGPR MSB";
    }

    Pass::ID getPassID() const override {
        return &InsertVgprMsbPassImpl::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        auto arch = passCtx.getGemmTileConfig().arch;
        GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);

        VgprMsbMode msbMode = passCtx.getAsmCapsConfig().vgprMsbMode;
        if (msbMode == VgprMsbMode::None) return preserveCFGAnalyses();

        runOnFunction(func, archId, msbMode);
        return preserveCFGAnalyses();
    }

   private:
    void runOnFunction(Function& func, GfxArchID archId, VgprMsbMode msbMode) const {
        std::unordered_map<const BasicBlock*, LabelJoin> joinsByHead;
        std::unordered_map<const BasicBlock*, int> joinEntryStates;
        if (options_.labelJoin) {
            joinsByHead = findLabelJoins(func);
            for (const auto& [head, join] : joinsByHead)
                joinEntryStates[join.join] = join.joinState;
        }

        VgprMsbPlanner planner(msbMode);
        for (BasicBlock& bb : func) {
            AsmIRBuilder irBuilder(bb, archId);
            auto entry = joinEntryStates.find(&bb);
            bool atJoinLabel = entry != joinEntryStates.end();
            planner.beginBlock(atJoinLabel ? entry->second : VgprMsbPlanner::kNotRequired);
            auto head = joinsByHead.find(&bb);
            const LabelJoin* join = head != joinsByHead.end() ? &head->second : nullptr;

            for (IRBase& ir : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&ir);
                if (!inst) continue;
                if (atJoinLabel) {
                    atJoinLabel = false;
                    if (isLabel(*inst)) continue;
                }
                if (join && inst == join->hoistBefore) {
                    if (std::optional<PlannedMsbSwitch> planned =
                            planner.require(join->joinState, *inst))
                        emitVgprMsb(*planned, irBuilder, archId);
                }
                if (std::optional<PlannedMsbSwitch> planned = planner.observe(*inst))
                    emitVgprMsb(*planned, irBuilder, archId);
                if (!isPseudoInst(inst) && !isCall(*inst)) encodeVgprOperands(inst);
            }
        }
    }

    InsertVgprMsbOptions options_;
};

char InsertVgprMsbPassImpl::ID = 0;

}  // anonymous namespace

std::unique_ptr<Pass> createInsertVgprMsbPass(InsertVgprMsbOptions options) {
    return std::make_unique<InsertVgprMsbPassImpl>(options);
}

}  // namespace stinkytofu
