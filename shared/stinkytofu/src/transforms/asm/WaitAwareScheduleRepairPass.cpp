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

#include "stinkytofu/transforms/asm/WaitAwareScheduleRepairPass.hpp"

#include <cassert>
#include <unordered_set>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"

// Before dag/*.hpp so PASS_DEBUG inside those headers uses this pass name.
#define DEBUG_TYPE "WaitAwareScheduleRepairPass"

#include "dag/RegionDAG.hpp"
#include "dag/WaitAnchorUtils.hpp"
#include "dag/WaitAnchoredReadyQueue.hpp"

namespace {
using namespace stinkytofu;
using namespace stinkytofu::dag;
using namespace stinkytofu::waitcnt;

std::vector<StinkyInstruction*> repairSegment(const std::vector<StinkyInstruction*>& instructions,
                                              const WaitAnchorMap& anchors,
                                              const PassContext& passCtx,
                                              unsigned slotsToMovePastAnchor, bool pinPrefetches) {
    if (instructions.empty()) return {};

    RegionDAG dag = buildRegisterDependencyDAG(instructions);
    addCounterOrderEdges(dag, instructions, anchors);
    // Prefetches placed by the DAG's prefetch lead stay where it put them.
    if (pinPrefetches) addPrefetchPinEdges(dag, instructions);

    WaitAnchoredReadyQueue queue(passCtx, anchors, dag, slotsToMovePastAnchor);
    std::vector<StinkyInstruction*> scheduled = scheduleWithWaitAnchoredReadyQueue(dag, queue);
    assert(scheduled.size() == instructions.size() &&
           "Repair schedule must include every segment instruction exactly once");
    return scheduled;
}

void emitInstWithWaits(std::vector<IRBase*>& output, StinkyInstruction* inst,
                       const WaitAnchorMap& anchors) {
    auto it = anchors.find(inst);
    if (it != anchors.end()) {
        for (StinkyInstruction* wait : it->second.waits) output.push_back(wait);
    }
    output.push_back(inst);
}

void repairBlock(BasicBlock& bb, const PassContext& passCtx, unsigned slotsToMovePastAnchor) {
    const WaitAnchorMap anchors = discoverWaitAnchors(bb);
    // Without a wait-anchored WMMA there is nothing for this pass to repair.
    if (anchors.empty()) return;

    const std::unordered_set<StinkyInstruction*> attachedWaits = collectAttachedWaits(anchors);

    std::vector<IRBase*> output;
    output.reserve(bb.size());

    std::vector<StinkyInstruction*> segment;
    segment.reserve(bb.size());

    // Same per-block lead as the DAG scheduler (StageWmmaCounter).
    StageWmmaCounter stage;
    for (IRBase& ir : bb)
        if (ir.getType() == IRBase::IRType::StinkyTofu) stage.add(*cast<StinkyInstruction>(&ir));
    const bool pinPrefetches =
        stage.effectiveLead(passCtx.getPassFeatureConfig().dagFeatures.prefetchLeadWmmas,
                            passCtx.getPassFeatureConfig().dagFeatures.prefetchLeadMinStageWmmas) >
        0;

    auto flushSegment = [&]() {
        if (segment.empty()) return;
        const std::vector<StinkyInstruction*> repaired =
            repairSegment(segment, anchors, passCtx, slotsToMovePastAnchor, pinPrefetches);
        for (StinkyInstruction* inst : repaired) emitInstWithWaits(output, inst, anchors);
        segment.clear();
    };

    for (IRBase& ir : bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) {
            flushSegment();
            output.push_back(&ir);
            continue;
        }

        auto* inst = cast<StinkyInstruction>(&ir);
        if (attachedWaits.count(inst) != 0) continue;

        if (isHardBoundary(*inst, attachedWaits)) {
            flushSegment();
            output.push_back(inst);
            continue;
        }

        segment.push_back(inst);
    }
    flushSegment();

    assert(output.size() == bb.size() && "Repair must preserve instruction count");

    for (IRBase* ir : output) {
        bb.removeIR(ir);
        bb.appendIR(ir);
    }
}

class WaitAwareScheduleRepairPass : public StinkyInstPass {
   public:
    static char ID;

    explicit WaitAwareScheduleRepairPass(int kSlotsToMovePastAnchor)
        : kSlotsToMovePastAnchor_(kSlotsToMovePastAnchor) {}

    const char* getName() const override {
        return "WaitAwareScheduleRepairPass";
    }

    PassID getPassID() const override {
        return &WaitAwareScheduleRepairPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        (void)AM;
        if (kSlotsToMovePastAnchor_ <= 0) return PreservedAnalyses::all();

        const GfxArchID archId =
            getGfxArchID(passCtx.getGemmTileConfig().arch[0], passCtx.getGemmTileConfig().arch[1],
                         passCtx.getGemmTileConfig().arch[2]);
        const uint32_t wavefrontSize = passCtx.getWavefrontSize();

        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;

            AsmIRBuilder builder(bb, archId);
            collapseExecMaskedRegions(bb, builder, wavefrontSize);
            repairBlock(bb, passCtx, static_cast<unsigned>(kSlotsToMovePastAnchor_));
            expandExecMaskedGroups(bb);
        }
        return PreservedAnalyses::none();
    }

   private:
    int kSlotsToMovePastAnchor_;
};

char WaitAwareScheduleRepairPass::ID = 0;

}  // namespace

namespace stinkytofu {

std::unique_ptr<Pass> createWaitAwareScheduleRepairPass(int kSlotsToMovePastAnchor) {
    return std::make_unique<WaitAwareScheduleRepairPass>(kSlotsToMovePastAnchor);
}

}  // namespace stinkytofu
