// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <unordered_set>
#include <vector>

#include "WaitAnchoredReadyQueue.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace dag {

/// Wait-anchor discovery shared by the schedule repair passes: a run of waits
/// directly before a WMMA is attached to that WMMA (its anchor) and moves with it.

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
    using waitcnt::WaitCountSpec;
    WaitCountSpec out = a;
    if (b.dsCount != WaitCountSpec::kUnused) out.dsCount = b.dsCount;
    if (b.loadCount != WaitCountSpec::kUnused) out.loadCount = b.loadCount;
    if (b.kmCount != WaitCountSpec::kUnused) out.kmCount = b.kmCount;
    if (b.tensorCount != WaitCountSpec::kUnused) out.tensorCount = b.tensorCount;
    if (b.asyncCount != WaitCountSpec::kUnused) out.asyncCount = b.asyncCount;
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
/// Runs are split exactly where the repair passes split segments, so a wait group
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
