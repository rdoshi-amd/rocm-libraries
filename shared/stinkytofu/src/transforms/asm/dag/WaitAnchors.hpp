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
#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ReadyQueue.hpp"
#include "RegionDAG.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitDataflow.hpp"
#include "stinkytofu/transforms/asm/waitcnt/WaitPlan.hpp"

namespace stinkytofu {
namespace dag {

// ---------------------------------------------------------------------------
// Wait anchors, segment boundaries, and the ordering a repair pass has to add
// to a register DAG itself.
//
// Shared by every pass that re-schedules already-waited IR: the wait contract
// and the segment rules are properties of that problem, not of any one pick
// policy, so they outlive the policy that first needed them.
// ---------------------------------------------------------------------------

struct WaitAnchorInfo {
    StinkyInstruction* anchor = nullptr;
    std::vector<StinkyInstruction*> waits;
    waitcnt::WaitCountSpec spec;
};

using WaitAnchorMap = std::unordered_map<StinkyInstruction*, WaitAnchorInfo>;

inline bool counterApplies(waitcnt::CounterKind kind, const waitcnt::WaitCountSpec& spec) {
    using waitcnt::WaitCountSpec;
    switch (kind) {
        case waitcnt::CK_DS:
            return spec.dsCount != waitcnt::WaitCountSpec::kUnused;
        case waitcnt::CK_Load:
            return spec.loadCount != waitcnt::WaitCountSpec::kUnused;
        case waitcnt::CK_KM:
            return spec.kmCount != waitcnt::WaitCountSpec::kUnused;
        case waitcnt::CK_Tensor:
            return spec.tensorCount != waitcnt::WaitCountSpec::kUnused;
        case waitcnt::CK_Async:
            return spec.asyncCount != waitcnt::WaitCountSpec::kUnused;
        default:
            return false;
    }
}

/// For each instruction \p shouldPin accepts, look forward for the next
/// instruction \p isBarrier accepts and add an edge between them. The edge
/// forces the first instruction to be scheduled before the second.
///
/// Both of the orderings this pass adds by itself, on top of the ones it gets
/// from register dependencies, are built out of this one step.
///
/// Two useful things follow from only ever looking forward. Edges always run
/// from a lower index to a higher one, so they can never form a cycle. And an
/// instruction with no barrier after it gets no edge at all, which is right:
/// there is nothing left for it to stay ahead of.
///
/// The predicates take a pointer instead of a reference because one of them
/// looks the instruction up in a \ref WaitAnchorMap, which is keyed on pointers.
///
/// On cost: when both predicates accept the same instructions, each forward scan
/// stops at the next accepted one, so no two scans cover the same ground and the
/// total work is linear. When they accept different instructions, several scans
/// can cross the same stretch, so keep the pinned set small.
template <typename PinPred, typename BarrierPred>
inline void addEdgesToFirstFollowing(RegionDAG& dag,
                                     const std::vector<StinkyInstruction*>& instructions,
                                     PinPred shouldPin, BarrierPred isBarrier) {
    for (unsigned i = 0; i < instructions.size(); ++i) {
        if (!shouldPin(instructions[i])) continue;
        for (unsigned j = i + 1; j < instructions.size(); ++j) {
            if (!isBarrier(instructions[j])) continue;
            addEdgeById(&dag.nodes[i], &dag.nodes[j], dag.graph);
            break;
        }
    }
}

/// Preserve the meaning of final wait immediates by ordering each counter's
/// producers and wait anchors in their original sequence.
///
/// Every event of a counter is both pinned and a barrier for that counter, so
/// the edges chain each event to the next one and the whole chain stays in input
/// order.
inline void addCounterOrderEdges(RegionDAG& dag,
                                 const std::vector<StinkyInstruction*>& instructions,
                                 const WaitAnchorMap& anchors) {
    using waitcnt::CK_Count;
    using waitcnt::CounterKind;

    for (int ck = 0; ck < CK_Count; ++ck) {
        const auto kind = static_cast<CounterKind>(ck);
        auto isCounterEvent = [&](StinkyInstruction* inst) {
            if (waitcnt::classifyMemOp(*inst) == kind) return true;
            auto anchor = anchors.find(inst);
            return anchor != anchors.end() && counterApplies(kind, anchor->second.spec);
        };
        addEdgesToFirstFollowing(dag, instructions, isCounterEvent, isCounterEvent);
    }
}

/// Hints whose only value is the lead time they get: no waitcnt counter and no
/// destination register, so nothing else in the DAG orders them.
///
/// The null check matters here. StinkyInstruction::is() dereferences its
/// descriptor unchecked, which is why hasSideEffect() tests the pointer first.
inline bool isPrefetchHint(const StinkyInstruction& inst) {
    if (inst.getHwInstDesc() == nullptr) return false;
    return isGlobalPrefetch(inst);
}

/// Keep memory operations, prefetch hints and matrix instructions in the order
/// the scheduler left them, so only ALU work floats.
///
/// The counter chain above already pins each memory op against the waits that
/// count it, which is what keeps the wait immediates honest. This is the
/// separate and stronger statement that the memory/matrix skeleton is not this
/// pass's to rearrange.
///
/// The scheduler places DS reads against the matrix ops that consume them using
/// a model this pass does not reproduce -- queue depth, drain latency, throttle
/// transitions and per-matrix-op affinity -- derived with the whole region in
/// view and before any wait existed. Replaying a segment re-decides that
/// placement from a narrower view: on the production kernel it moved 267 of 462
/// DS reads across a matrix op while moving only 20 of 818 VALU ops, so nearly
/// all of the churn was in the one thing the repair is not for. Refilling a
/// co-execution window only ever needs the work that can fill a slot, which is
/// ALU work (see fillsCoexecSlot), and that work is left free here.
///
/// Prefetch hints join the chain even though no counter tracks them, which is
/// what puts them outside the memory-op test. Nothing else orders them, and the
/// queue does not put a free hint back where the scheduler did: the scheduler
/// placed it against the stage barrier and tensor_load of its whole region,
/// which a segment rarely contains. Left free, hints moved by up to 244 matrix
/// ops on the gfx1250 tile kernels, and a hint moved either way loses the lead
/// time it was placed for.
///
/// One chain over all three classes, the same shape as the counter chain: every
/// member is both pinned and a barrier, so consecutive members are linked and
/// the whole sequence holds its input order.
inline void addMatrixMemoryOrderEdges(RegionDAG& dag,
                                      const std::vector<StinkyInstruction*>& instructions) {
    auto isSkeleton = [](StinkyInstruction* inst) {
        return isMatrixInstruction(*inst) || isPrefetchHint(*inst) ||
               waitcnt::classifyMemOp(*inst) != waitcnt::CK_Count;
    };
    addEdgesToFirstFollowing(dag, instructions, isSkeleton, isSkeleton);
}

/// All ordering this pass adds that register dependencies did not give it.
///
/// The two rules stay separate because they answer different questions. Counter
/// order is a correctness constraint: it keeps a wait immediate counting the
/// operations it was computed for. Skeleton order is a scope constraint: it
/// limits the pass to the work a co-issue slot can hold, and holds prefetch
/// hints, which nothing else orders, where the scheduler put them.
inline void addSyntheticOrderEdges(RegionDAG& dag,
                                   const std::vector<StinkyInstruction*>& instructions,
                                   const WaitAnchorMap& anchors) {
    addCounterOrderEdges(dag, instructions, anchors);
    addMatrixMemoryOrderEdges(dag, instructions);
}

inline bool isAnyWaitCnt(const StinkyInstruction& inst) {
    return isWaitCnt(inst) || inst.is(InstFlag::IF_WaitTensorCnt);
}

inline waitcnt::WaitCountSpec decodeWaitSpec(const StinkyInstruction& wait) {
    waitcnt::WaitCountSpec spec;
    if (const auto* data = wait.getModifier<SWaitCntData>()) {
        if (data->dlcnt >= 0) spec.dsCount = data->dlcnt;
        if (data->vlcnt >= 0) spec.loadCount = data->vlcnt;
        if (data->kmcnt >= 0) spec.kmCount = data->kmcnt;
    }
    if (const auto* tdata = wait.getModifier<SWaitTensorCntData>()) {
        if (tdata->tlcnt >= 0) spec.tensorCount = static_cast<unsigned char>(tdata->tlcnt);
    }
    if (const auto* adata = wait.getModifier<SWaitAsyncCntData>()) {
        if (adata->asynccnt >= 0) spec.asyncCount = static_cast<unsigned char>(adata->asynccnt);
    }
    return spec;
}

inline waitcnt::WaitCountSpec mergeWaitSpecs(const waitcnt::WaitCountSpec& a,
                                             const waitcnt::WaitCountSpec& b) {
    waitcnt::WaitCountSpec out = a;
    if (b.dsCount != waitcnt::WaitCountSpec::kUnused) out.dsCount = b.dsCount;
    if (b.loadCount != waitcnt::WaitCountSpec::kUnused) out.loadCount = b.loadCount;
    if (b.kmCount != waitcnt::WaitCountSpec::kUnused) out.kmCount = b.kmCount;
    if (b.tensorCount != waitcnt::WaitCountSpec::kUnused) out.tensorCount = b.tensorCount;
    if (b.asyncCount != waitcnt::WaitCountSpec::kUnused) out.asyncCount = b.asyncCount;
    return out;
}

inline void discoverWaitAnchorsInRun(const std::vector<StinkyInstruction*>& seq,
                                     WaitAnchorMap& anchors) {
    for (size_t i = 0; i < seq.size(); ++i) {
        if (!isAnyWaitCnt(*seq[i])) continue;

        size_t waitStart = i;
        size_t waitEnd = waitStart + 1;
        while (waitEnd < seq.size() && isAnyWaitCnt(*seq[waitEnd])) ++waitEnd;

        if (waitEnd >= seq.size() || !isMatrixInstruction(*seq[waitEnd])) {
            i = waitEnd;
            continue;
        }

        WaitAnchorInfo info;
        info.anchor = seq[waitEnd];
        waitcnt::WaitCountSpec combined;
        for (size_t w = waitStart; w < waitEnd; ++w) {
            info.waits.push_back(seq[w]);
            combined = mergeWaitSpecs(combined, decodeWaitSpec(*seq[w]));
        }
        info.spec = combined;
        anchors[info.anchor] = std::move(info);
        i = waitEnd;
    }
}

/// Discover wait anchors per run of consecutive StinkyTofu instructions.
/// Runs are split exactly where repairBlock() splits segments, so a wait group
/// can never be anchored to a WMMA that the rewrite places past a boundary.
inline WaitAnchorMap discoverWaitAnchors(BasicBlock& bb) {
    WaitAnchorMap anchors;
    std::vector<StinkyInstruction*> run;
    run.reserve(bb.size());

    for (IRBase& ir : bb) {
        if (ir.getType() != IRBase::IRType::StinkyTofu) {
            discoverWaitAnchorsInRun(run, anchors);
            run.clear();
            continue;
        }
        run.push_back(cast<StinkyInstruction>(&ir));
    }
    discoverWaitAnchorsInRun(run, anchors);
    return anchors;
}

inline std::unordered_set<StinkyInstruction*> collectAttachedWaits(const WaitAnchorMap& anchors) {
    std::unordered_set<StinkyInstruction*> attached;
    for (const auto& [anchor, info] : anchors) {
        (void)anchor;
        for (StinkyInstruction* wait : info.waits) attached.insert(wait);
    }
    return attached;
}

inline bool isHardBoundary(const StinkyInstruction& inst,
                           const std::unordered_set<StinkyInstruction*>& attachedWaits) {
    if (isLabel(inst)) return true;
    if (isAnyWaitCnt(inst) && attachedWaits.count(const_cast<StinkyInstruction*>(&inst)) == 0)
        return true;
    if (hasSideEffect(inst)) return true;
    if (isExecMaskGroup(inst)) return true;
    return false;
}

}  // namespace dag
}  // namespace stinkytofu
