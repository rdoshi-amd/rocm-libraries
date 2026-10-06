// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/CoexecSimRepairPass.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/analysis/asm/HazardGapAnalysisPass.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"
#include "stinkytofu/transforms/asm/RegionClonePass.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

// Before dag/*.hpp so PASS_DEBUG inside those headers uses this pass name.
#define DEBUG_TYPE "CoexecSimRepairPass"

#include "coexec/CoexecModel.hpp"
#include "coexec/CoexecPatterns.hpp"
#include "dag/RegionDAG.hpp"
#include "dag/WaitAnchorUtils.hpp"
#include "dag/WaitAnchoredReadyQueue.hpp"

namespace {
using namespace stinkytofu;
using namespace stinkytofu::coexec;
using namespace stinkytofu::dag;

constexpr const char* kPassName = "CoexecSimRepairPass";
// Gaining candidates per round re-simulated with their own s_wait_alu, best first.
constexpr size_t kExactChecks = 4;
// Search rounds per block; each round restarts the per-window move counts.
constexpr int kMaxRounds = 8;

bool hasRegType(const std::vector<StinkyRegister>& regs, std::initializer_list<RegType> types) {
    for (const StinkyRegister& r : regs)
        if (r.isRegister() && std::find(types.begin(), types.end(), r.reg.type) != types.end())
            return true;
    return false;
}

// Every VALU reads EXEC, and some instructions read M0, without naming it, so the
// dependence graph cannot order a write of either; nothing moves across one.
bool writesImplicitlyReadReg(const StinkyInstruction& inst) {
    return hasRegType(inst.getDestRegs(),
                      {RegType::EXEC, RegType::EXEC_LO, RegType::EXEC_HI, RegType::M});
}

// A SALU or VALU the repair may move.
bool isFillerClass(const StinkyInstruction& inst) {
    if (inst.getHwInstDesc() == nullptr || isPseudoInst(&inst) || isMatrixInstruction(inst))
        return false;
    const auto op = inst.getUnifiedOpcode();
    if (op == GFX::s_nop || op == GFX::v_nop || op == GFX::s_set_vgpr_msb || op == GFX::s_wait_alu)
        return false;
    const bool valu = isVectorALU(inst) || isTranscendental(inst);
    const bool salu =
        isScalarALU(inst) && !isBarrier(inst) && !isControlTransfer(inst) && !isAnyWaitCnt(inst);
    return valu || salu;
}

bool isRegisterLoad(const StinkyInstruction& inst) {
    return isDSRead(inst) || isDSAtomic(inst) || isGlobalMemLoad(inst) || isReturningAtomic(inst);
}

using RegKey = long long;

void addRegKeys(const std::vector<StinkyRegister>& regs, std::unordered_set<RegKey>& out) {
    for (const StinkyRegister& r : regs) {
        if (!r.isRegister() || isPseudoReg(r)) continue;
        for (unsigned i = 0; i < r.reg.num; ++i)
            out.insert((static_cast<RegKey>(r.reg.type) << 32) | (r.reg.idx + i));
    }
}

bool touchesAny(const StinkyInstruction& inst, const std::unordered_set<RegKey>& keys) {
    std::unordered_set<RegKey> own;
    addRegKeys(inst.getSrcRegs(), own);
    addRegKeys(inst.getDestRegs(), own);
    for (RegKey k : own)
        if (keys.count(k)) return true;
    return false;
}

const char* mnemonicOf(const StinkyInstruction& inst) {
    const HwInstDesc* desc = inst.getHwInstDesc();
    return desc && desc->mnemonic ? desc->mnemonic : "?";
}

// A window holding a ds_load may take on issue time only up to the spare time it had:
// the model cannot see the LDS contention that shifting the loads would change.
bool dsWindowsKeepTiming(const SimResult& before, const SimResult& after) {
    for (size_t k = 0; k < before.windows.size(); ++k) {
        const WindowTiming& b = before.windows[k];
        if (b.hasDsLoad && after.windows[k].ownCycles > b.ownCycles + std::max(0, b.lead))
            return false;
    }
    return true;
}

struct AppliedMove {
    Move move;
    int fromWindow;
    int gain;
};

struct SearchState {
    std::vector<int> order;  ///< the current order, by id
    std::vector<int> pos;    ///< id -> position in order
    std::vector<WaitAluNeed> waits;
    SimResult cur;
    std::vector<AppliedMove> applied;
};

class BlockRepair {
   public:
    BlockRepair(BasicBlock& bb, const PassContext& passCtx, const CoexecSimRepairOptions& opts,
                const std::unordered_set<const StinkyInstruction*>& cloneSplits)
        : bb_(bb),
          passCtx_(passCtx),
          opts_(opts),
          cloneSplits_(cloneSplits),
          model_(passCtx, CoexecModel::Options{opts.predictWaitAlu, opts.waitAlu}) {}

    bool run();

   private:
    bool collectBlock();
    MoveSpace buildMoveSpace() const;
    void searchRound(const MoveSpace& space, SearchState& st) const;
    bool hazardsHold(const std::vector<int>& output) const;
    std::string describe(const AppliedMove& applied) const;
    void remark(OptimizationRemark::Kind kind, const std::string& message) const {
        emitRemark(passCtx_, {kind, kPassName, "CoexecSimRepair", message});
    }

    BasicBlock& bb_;
    const PassContext& passCtx_;
    const CoexecSimRepairOptions& opts_;
    const std::unordered_set<const StinkyInstruction*>& cloneSplits_;
    CoexecModel model_;
    std::vector<StinkyInstruction*> block_;
};

bool BlockRepair::collectBlock() {
    bool hasWmma = false;
    for (IRBase& ir : bb_) {
        // Only blocks made of instructions are modeled.
        if (ir.getType() != IRBase::IRType::StinkyTofu) return false;
        auto* inst = cast<StinkyInstruction>(&ir);
        block_.push_back(inst);
        if (inst->getHwInstDesc() != nullptr && isMatrixInstruction(*inst)) hasWmma = true;
    }
    return hasWmma;
}

MoveSpace BlockRepair::buildMoveSpace() const {
    const int n = static_cast<int>(block_.size());
    MoveSpace::Inputs in;
    in.segment.assign(n, -1);
    in.filler.assign(n, false);
    in.wmma.assign(n, false);
    in.noInsertBefore.assign(n, false);
    in.preds.assign(n, {});
    in.succs.assign(n, {});
    in.units.assign(n, {});

    std::unordered_map<const StinkyInstruction*, int> idOf;
    for (int id = 0; id < n; ++id) idOf[block_[id]] = id;

    // A wait group stays in front of the v_wmma it guards: a filler may go before the
    // group, never inside it.
    const WaitAnchorMap anchors = discoverWaitAnchors(bb_);
    const std::unordered_set<StinkyInstruction*> attached = collectAttachedWaits(anchors);
    for (const auto& [anchor, info] : anchors) {
        if (info.waits.empty()) continue;
        in.noInsertBefore[idOf.at(anchor)] = true;
        for (size_t i = 1; i < info.waits.size(); ++i)
            in.noInsertBefore[idOf.at(info.waits[i])] = true;
    }

    // Segments end at hard boundaries and right after each clone split point.
    int segment = 0;
    for (int id = 0; id < n; ++id) {
        StinkyInstruction* inst = block_[id];
        in.wmma[id] = inst->getHwInstDesc() != nullptr && isMatrixInstruction(*inst);
        if (!attached.count(inst) &&
            (isHardBoundary(*inst, attached) || writesImplicitlyReadReg(*inst))) {
            ++segment;
            continue;
        }
        in.segment[id] = segment;
        in.filler[id] = isFillerClass(*inst);
        if (cloneSplits_.count(inst)) ++segment;
    }

    std::unordered_set<RegKey> loadedRegs;
    for (StinkyInstruction* inst : block_)
        if (inst->getHwInstDesc() != nullptr && isRegisterLoad(*inst))
            addRegKeys(inst->getDestRegs(), loadedRegs);

    // Same per-block prefetch lead as the DAG scheduler and WaitAwareScheduleRepairPass.
    StageWmmaCounter stage;
    for (StinkyInstruction* inst : block_) stage.add(*inst);
    const auto& dagFeatures = passCtx_.getPassFeatureConfig().dagFeatures;
    const bool pinPrefetches = stage.effectiveLead(dagFeatures.prefetchLeadWmmas,
                                                   dagFeatures.prefetchLeadMinStageWmmas) > 0;

    auto addEdge = [&](int from, int to) {
        in.succs[from].push_back(to);
        in.preds[to].push_back(from);
    };
    for (int first = 0; first < n;) {
        if (in.segment[first] < 0) {
            ++first;
            continue;
        }
        int last = first;
        while (last < n && in.segment[last] == in.segment[first]) ++last;
        std::vector<StinkyInstruction*> insts(block_.begin() + first, block_.begin() + last);

        RegionDAG dag = buildRegisterDependencyDAG(insts);
        addCounterOrderEdges(dag, insts, anchors);
        if (pinPrefetches) addPrefetchPinEdges(dag, insts);
        for (unsigned i = 0; i < dag.graph.size(); ++i)
            for (unsigned j : dag.graph[i])
                addEdge(first + static_cast<int>(i), first + static_cast<int>(j));

        // A filler that reads or writes a loaded register stays behind the wait before
        // it: that wait is what makes the load's value safe to use or overwrite.
        int lastWait = -1;
        for (int id = first; id < last; ++id) {
            if (isAnyWaitCnt(*block_[id])) {
                lastWait = id;
            } else if (in.filler[id] && lastWait >= 0 && touchesAny(*block_[id], loadedRegs)) {
                addEdge(lastWait, id);
            }
        }

        // An SCC producer moves with the fillers that read its SCC.
        std::vector<int> group(last - first);
        std::iota(group.begin(), group.end(), first);
        auto root = [&](int id) {
            while (group[id - first] != id) id = group[id - first];
            return id;
        };
        int sccWriter = -1;
        for (int id = first; id < last; ++id) {
            const StinkyInstruction& inst = *block_[id];
            if (sccWriter >= 0 && hasRegType(inst.getSrcRegs(), {RegType::SCC}) && in.filler[id] &&
                in.filler[sccWriter])
                group[root(id) - first] = root(sccWriter);
            if (hasRegType(inst.getDestRegs(), {RegType::SCC})) sccWriter = id;
        }
        std::unordered_map<int, std::vector<int>> members;
        for (int id = first; id < last; ++id)
            if (in.filler[id]) members[root(id)].push_back(id);
        for (int id = first; id < last; ++id) {
            if (!in.filler[id]) continue;
            in.units[id] = members[root(id)];
        }
        first = last;
    }
    for (int id = 0; id < n; ++id) {
        if (in.units[id].empty()) in.units[id] = {id};
        for (auto* edges : {&in.preds[id], &in.succs[id]}) {
            std::sort(edges->begin(), edges->end());
            edges->erase(std::unique(edges->begin(), edges->end()), edges->end());
        }
    }
    return MoveSpace(std::move(in));
}

// No hazard pair falls below its distance unless it already did in the input, and no
// more interlock pairs fall below theirs.
bool BlockRepair::hazardsHold(const std::vector<int>& output) const {
    std::vector<const StinkyInstruction*> input(block_.begin(), block_.end());
    std::vector<const StinkyInstruction*> repaired;
    for (int id : output) repaired.push_back(block_[id]);
    auto below = [](const std::vector<const StinkyInstruction*>& seq, const HazardRule& rule) {
        std::set<std::pair<const StinkyInstruction*, const StinkyInstruction*>> pairs;
        for (const HazardGapPair& pair : findHazardGapPairs(seq, rule))
            if (pair.gap < rule.distance) pairs.insert({seq[pair.producer], seq[pair.consumer]});
        return pairs;
    };
    const HWModel& hw = passCtx_.getHWModel();
    for (int i = 0; i < hw.hazards.numRules; ++i) {
        const auto before = below(input, hw.hazards.rules[i]);
        const auto after = below(repaired, hw.hazards.rules[i]);
        if (!std::includes(before.begin(), before.end(), after.begin(), after.end())) return false;
    }
    for (const HazardRule& rule : kCdna5InterlockRules)
        if (below(repaired, rule).size() > below(input, rule).size()) return false;
    return true;
}

std::string BlockRepair::describe(const AppliedMove& applied) const {
    std::ostringstream os;
    os << bb_.getLabel() << ": " << patternName(applied.move.pattern) << ": moved ";
    for (size_t i = 0; i < applied.move.unit.size(); ++i)
        os << (i ? " + " : "") << mnemonicOf(*block_[applied.move.unit[i]]);
    if (applied.fromWindow == applied.move.toWindow)
        os << " within window " << applied.fromWindow;
    else
        os << " from window " << applied.fromWindow << " to window " << applied.move.toWindow;
    os << ", predicted gain " << applied.gain << " cycles";
    return os.str();
}

// One round of the search: repair the most-blamed window a pattern explains, until no
// window gains or every window has used its moves.
void BlockRepair::searchRound(const MoveSpace& space, SearchState& st) const {
    const HWModel::CoexecTiming& timing = passCtx_.getHWModel().coexecTiming;
    const int n = static_cast<int>(block_.size());
    const int numWindows = st.cur.numWmma + 1;
    const int maxMoves = std::max(0, opts_.maxMovesPerWindow);
    std::vector<int> movesFrom(numWindows, 0), movesInto(numWindows, 0);
    std::vector<char> done(numWindows, 0);

    while (true) {
        const PatternContext ctx{model_, space, st.order, st.pos, st.cur, timing};
        int window = -1;
        int windowIdle = 0;
        for (int k = 0; k < numWindows; ++k) {
            if (done[k] || movesFrom[k] >= maxMoves || st.cur.windows[k].idle <= windowIdle)
                continue;
            if (!explainingPatterns(ctx, k, opts_.patternMask)) continue;
            window = k;
            windowIdle = st.cur.windows[k].idle;
        }
        if (window < 0) return;

        // Screen every proposal with the current s_wait_alu, then confirm the best with
        // their own. Ties go to the lowest instruction id, then the earliest slot.
        const std::vector<Move> moves = proposeMoves(ctx, window, opts_.patternMask);
        struct Gaining {
            int idle;
            int firstId;
            int slot;
            size_t move;
            std::vector<int> order;
        };
        std::vector<Gaining> gaining;
        std::set<std::vector<int>> seen;
        for (size_t i = 0; i < moves.size(); ++i) {
            const Move& move = moves[i];
            if (move.toWindow < 0 || move.toWindow >= numWindows ||
                movesInto[move.toWindow] >= maxMoves)
                continue;
            std::vector<int> candidate = space.apply(st.order, move);
            if (candidate == st.order || !seen.insert(candidate).second) continue;
            const int idle = model_.simulate(candidate, st.waits).idle;
            if (idle <= st.cur.idle - 1)
                gaining.push_back({idle, move.unit.front(), move.slot, i, std::move(candidate)});
        }
        std::sort(gaining.begin(), gaining.end(), [](const Gaining& a, const Gaining& b) {
            if (a.idle != b.idle) return a.idle < b.idle;
            if (a.firstId != b.firstId) return a.firstId < b.firstId;
            return a.slot < b.slot;
        });
        DEBUG_WITH_TYPE("CoexecSimRepairSearch",
                        std::cerr << "[" << kPassName << "] " << bb_.getLabel() << " window "
                                  << window << " (idle " << st.cur.windows[window].idle
                                  << ", patterns 0x" << std::hex
                                  << explainingPatterns(ctx, window, opts_.patternMask) << std::dec
                                  << "): " << moves.size() << " proposed, " << seen.size()
                                  << " new orders, " << gaining.size() << " gaining"
                                  << (gaining.empty() ? std::string()
                                                      : ", best " + std::to_string(gaining[0].idle))
                                  << " vs " << st.cur.idle << "\n");

        bool accepted = false;
        for (size_t c = 0; c < gaining.size() && c < kExactChecks && !accepted; ++c) {
            std::vector<WaitAluNeed> candidateWaits = model_.predictWaitAlu(gaining[c].order);
            SimResult exact = model_.simulate(gaining[c].order, candidateWaits);
            if (exact.idle > st.cur.idle - 1) continue;
            // No s_wait_alu where the current order needs none.
            if (!std::includes(st.cur.waitAluIds.begin(), st.cur.waitAluIds.end(),
                               exact.waitAluIds.begin(), exact.waitAluIds.end()))
                continue;
            if (!dsWindowsKeepTiming(st.cur, exact)) continue;

            const Move& move = moves[gaining[c].move];
            st.applied.push_back(
                {move, st.cur.inst[st.pos[move.unit.front()]].window, st.cur.idle - exact.idle});
            ++movesFrom[window];
            ++movesInto[move.toWindow];
            st.order = std::move(gaining[c].order);
            for (int p = 0; p < n; ++p) st.pos[st.order[p]] = p;
            st.waits = std::move(candidateWaits);
            st.cur = std::move(exact);
            std::fill(done.begin(), done.end(), 0);
            accepted = true;
        }
        if (!accepted) done[window] = 1;
    }
}

bool BlockRepair::run() {
    if (!collectBlock()) return false;
    const MoveSpace space = buildMoveSpace();
    const int n = static_cast<int>(block_.size());
    bool anyFiller = false;
    for (int id = 0; id < n; ++id) anyFiller = anyFiller || space.isFiller(id);
    if (!anyFiller) return false;

    model_.setBlock(block_);
    SearchState st;
    st.order.resize(n);
    std::iota(st.order.begin(), st.order.end(), 0);
    st.pos = st.order;
    st.waits = model_.predictWaitAlu(st.order);
    st.cur = model_.simulate(st.order, st.waits);
    const int baseIdle = st.cur.idle;
    if (baseIdle <= 0) return false;

    // The move limit holds per round. Rounds repeat until one makes no move, so the
    // result is a fixed point of the search and a second run finds nothing to do. Each
    // accepted move lowers the predicted idle by at least a cycle, so this ends.
    for (int round = 0; round < kMaxRounds; ++round) {
        const size_t before = st.applied.size();
        searchRound(space, st);
        if (st.applied.size() == before) break;
    }
    const std::vector<int>& order = st.order;
    const std::vector<AppliedMove>& applied = st.applied;
    const SimResult& cur = st.cur;

    PASS_DEBUG({
        std::cerr << "[" << kPassName << "] " << bb_.getLabel() << ": idle " << baseIdle << " -> "
                  << cur.idle << " with " << applied.size() << " move(s)\n";
        for (int k = 0; k <= cur.numWmma; ++k) {
            if (cur.windows[k].idle <= 0) continue;
            std::cerr << "  window " << k << ": idle " << cur.windows[k].idle << ", lead in "
                      << cur.windows[k].leadIn << ":";
            for (int p = 0; p < n; ++p) {
                const InstTiming& t = cur.inst[p];
                if (t.window != k || t.idle() <= 0) continue;
                const Cause cause = t.idleStall > 0 ? t.stallCause
                                    : t.idleGap > 0 ? t.gapCause
                                                    : Cause::Issue;
                std::cerr << " " << mnemonicOf(*block_[order[p]]) << "=" << t.idle() << "("
                          << causeName(cause) << ")";
            }
            std::cerr << "\n";
        }
    });
    if (applied.empty()) return false;

    const int gain = baseIdle - cur.idle;
    const int margin =
        std::max(opts_.marginCycles, static_cast<int>(std::ceil(opts_.marginFraction * baseIdle)));
    if (gain < margin) {
        remark(OptimizationRemark::Kind::Missed,
               bb_.getLabel() + ": kept the input: predicted gain " + std::to_string(gain) +
                   " of " + std::to_string(baseIdle) + " idle cycles is below the margin " +
                   std::to_string(margin));
        return false;
    }
    if (!space.respects(order) || !hazardsHold(order)) {
        remark(OptimizationRemark::Kind::Missed,
               bb_.getLabel() + ": kept the input: the repaired order failed verification");
        return false;
    }

    for (const AppliedMove& a : applied) remark(OptimizationRemark::Kind::Passed, describe(a));
    for (int id : order) {
        bb_.removeIR(block_[id]);
        bb_.appendIR(block_[id]);
    }
    return true;
}

class CoexecSimRepairPass : public StinkyInstPass {
   public:
    static char ID;

    explicit CoexecSimRepairPass(CoexecSimRepairOptions options) : options_(std::move(options)) {}

    const char* getName() const override {
        return kPassName;
    }

    PassID getPassID() const override {
        return &CoexecSimRepairPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        (void)AM;
        const auto& arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
        const uint32_t wavefrontSize = passCtx.getWavefrontSize();

        const std::vector<StinkyInstruction*> splits = findRegionCloneSplits(func, options_.clones);
        const std::unordered_set<const StinkyInstruction*> cloneSplits(splits.begin(),
                                                                       splits.end());

        for (BasicBlock& bb : func) {
            if (!passCtx.shouldProcessBasicBlock(bb)) continue;
            AsmIRBuilder builder(bb, archId);
            collapseExecMaskedRegions(bb, builder, wavefrontSize);
            BlockRepair(bb, passCtx, options_, cloneSplits).run();
            expandExecMaskedGroups(bb);
        }
        return PreservedAnalyses::none();
    }

   private:
    CoexecSimRepairOptions options_;
};

char CoexecSimRepairPass::ID = 0;

}  // namespace

namespace stinkytofu {

std::unique_ptr<Pass> createCoexecSimRepairPass(CoexecSimRepairOptions options) {
    return std::make_unique<CoexecSimRepairPass>(std::move(options));
}

}  // namespace stinkytofu
