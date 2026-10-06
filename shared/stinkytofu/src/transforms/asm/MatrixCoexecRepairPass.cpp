// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/MatrixCoexecRepairPass.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"
#include "stinkytofu/support/LoopDetection.hpp"
#include "stinkytofu/support/OptimizationRemark.hpp"
#include "stinkytofu/transforms/asm/ExecMaskGrouping.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"

// Before dag/*.hpp so PASS_DEBUG inside those headers uses this pass name.
#define DEBUG_TYPE "MatrixCoexecRepairPass"

#include "coexec/CoexecSimulator.hpp"
#include "dag/RegionDAG.hpp"

namespace stinkytofu {
namespace {
using coexec::CoexecSimulator;
using coexec::SimConfig;
using coexec::SimResult;
using coexec::SimState;
using dag::buildRegisterDependencyDAG;
using dag::RegionDAG;

constexpr const char* kPassName = "MatrixCoexecRepairPass";
/// Later WMMAs tried as the new slot of one filler.
constexpr int kSlotsPerFiller = 16;
/// Committed moves allowed per filler of a block.
constexpr int kMovesPerFiller = 4;
/// Instructions before a block that seed its s_wait_alu prediction; the counters it
/// tracks saturate well within this.
constexpr size_t kWaitAluHistory = 128;

bool isAnyWait(const StinkyInstruction& inst) {
    return isWaitCnt(inst) || inst.is(InstFlag::IF_WaitTensorCnt);
}

/// Fillers never cross these. Counter waits are not among them: a filler reaches other
/// instructions only through registers, which the dependency DAG covers.
bool isSegmentBoundary(const IRBase& ir) {
    const auto* inst = dyn_cast<StinkyInstruction>(&ir);
    if (!inst) return true;
    if (isAnyWait(*inst)) return false;
    const auto op = inst->getUnifiedOpcode();
    return isPseudoInst(inst) || isBranch(*inst) || isCall(*inst) || isBarrier(*inst) ||
           isExecMaskGroup(*inst) || isHasSideEffect(*inst) || isGlobalMemStore(*inst) ||
           isGlobalMemAtomic(*inst) || op == GFX::s_wait_alu || op == GFX::s_delay_alu ||
           op == GFX::s_set_vgpr_msb;
}

bool isFillerDest(const StinkyRegister& reg) {
    if (reg.dataType != StinkyRegister::Type::Register) return false;
    switch (reg.reg.type) {
        case RegType::V:
        case RegType::S:
        case RegType::SCC:
        case RegType::VCC:
        case RegType::VCC_LO:
        case RegType::VCC_HI:
            return true;
        default:
            return false;
    }
}

/// VALU / SALU work that reaches other instructions only through V / S / SCC / VCC.
bool isFiller(const StinkyInstruction& inst) {
    if (!isVectorALU(inst) && !isScalarALU(inst)) return false;
    if (isMatrixInstruction(inst) || isAnyWait(inst) || isSegmentBoundary(inst) ||
        hasLdsPseudoRegs(inst))
        return false;
    const std::vector<StinkyRegister>& dests = inst.getDestRegs();
    return !dests.empty() && std::all_of(dests.begin(), dests.end(), isFillerDest);
}

std::vector<StinkyInstruction*> stinkyOf(const std::vector<IRBase*>& items) {
    std::vector<StinkyInstruction*> insts;
    insts.reserve(items.size());
    for (IRBase* ir : items)
        if (auto* inst = dyn_cast<StinkyInstruction>(ir)) insts.push_back(inst);
    return insts;
}

bool hasMatrixWork(const BasicBlock& bb) {
    for (const IRBase& ir : bb)
        if (const auto* inst = dyn_cast<StinkyInstruction>(&ir); inst && isMatrixInstruction(*inst))
            return true;
    return false;
}

struct Move {
    const StinkyInstruction* filler;
    /// Positions among the block's StinkyInstructions.
    size_t from;
    size_t to;
    int64_t gain;
};

struct BlockReport {
    std::string label;
    std::vector<StinkyInstruction*> beforeOrder;
    SimResult before;
    std::vector<StinkyInstruction*> afterOrder;
    SimResult after;
    std::vector<Move> moves;
};

struct LoopReport {
    std::string header;
    int64_t iterationBefore = 0;
    int64_t iterationAfter = 0;
    std::vector<BlockReport> blocks;
};

/// One loop body, laid out as its blocks appear in the function.
class LoopRepairer {
   public:
    LoopRepairer(std::vector<BasicBlock*> blocks, const CoexecSimulator& sim)
        : blocks_(std::move(blocks)), sim_(sim) {
        for (BasicBlock* bb : blocks_) {
            std::vector<IRBase*> items;
            for (IRBase& ir : *bb) items.push_back(&ir);
            items_.push_back(std::move(items));
        }
    }

    LoopReport run(bool analyzeOnly) {
        LoopReport report;
        std::vector<SimState> entries = steadyEntries();
        report.iterationBefore = iterationCycles();

        for (size_t k = 0; k < blocks_.size(); ++k) {
            BlockReport block;
            block.label = blocks_[k]->getLabel();
            block.beforeOrder = stinkyOf(items_[k]);
            block.before = simulate(k, items_[k], entries[k]);
            if (!analyzeOnly && hasMatrixWork(*blocks_[k])) repairBlock(k, entries[k], block);
            block.afterOrder = stinkyOf(items_[k]);
            block.after = simulate(k, items_[k], entries[k]);
            if (k + 1 < blocks_.size()) entries[k + 1] = block.after.exit;
            report.blocks.push_back(std::move(block));
        }

        report.iterationAfter = iterationCycles();
        return report;
    }

    /// Lay the blocks out in their repaired order.
    void apply() {
        for (size_t k = 0; k < blocks_.size(); ++k) {
            if (!changed_[k]) continue;
            for (IRBase* ir : items_[k]) {
                blocks_[k]->removeIR(ir);
                blocks_[k]->appendIR(ir);
            }
        }
    }

    bool changed() const {
        return std::any_of(changed_.begin(), changed_.end(), [](bool c) { return c; });
    }

   private:
    /// The last kWaitAluHistory instructions the body ran before block k (it wraps into the
    /// previous iteration).
    std::vector<StinkyInstruction*> historyBefore(size_t k) const {
        std::vector<StinkyInstruction*> history;
        for (size_t back = 1; back <= blocks_.size() && history.size() < kWaitAluHistory; ++back) {
            const size_t j = (k + blocks_.size() - back) % blocks_.size();
            const std::vector<StinkyInstruction*> insts = stinkyOf(items_[j]);
            const size_t take = std::min(insts.size(), kWaitAluHistory - history.size());
            history.insert(history.begin(), insts.end() - static_cast<std::ptrdiff_t>(take),
                           insts.end());
        }
        return history;
    }

    SimResult simulate(size_t k, const std::vector<IRBase*>& items, const SimState& entry) const {
        return sim_.run(stinkyOf(items), entry, historyBefore(k));
    }

    /// The entry state of each block in the second iteration of the body.
    std::vector<SimState> steadyEntries() const {
        std::vector<SimState> entries(blocks_.size());
        SimState state;
        for (int iteration = 0; iteration < 2; ++iteration) {
            for (size_t k = 0; k < blocks_.size(); ++k) {
                if (iteration == 1) entries[k] = state;
                state = simulate(k, items_[k], state).exit;
            }
        }
        return entries;
    }

    static int64_t finishOf(const SimState& state) {
        return std::max(state.nextIssue, state.matrixFreeAt);
    }

    /// Steady-state cycles of one iteration of the body as currently laid out.
    int64_t iterationCycles() const {
        SimState state;
        int64_t previous = 0;
        for (int iteration = 0; iteration < 3; ++iteration) {
            previous = finishOf(state);
            for (size_t k = 0; k < blocks_.size(); ++k) state = simulate(k, items_[k], state).exit;
        }
        return finishOf(state) - previous;
    }

    /// Finish of the iteration from block k's entry, with block k laid out as \p candidate.
    int64_t tailFinish(size_t k, const std::vector<IRBase*>& candidate,
                       const SimState& entry) const {
        SimState state = simulate(k, candidate, entry).exit;
        for (size_t j = k + 1; j < blocks_.size(); ++j) state = simulate(j, items_[j], state).exit;
        return finishOf(state);
    }

    void repairBlock(size_t k, const SimState& entry, BlockReport& block) {
        std::vector<IRBase*>& items = items_[k];
        size_t fillers = 0;
        for (StinkyInstruction* inst : stinkyOf(items)) fillers += isFiller(*inst) ? 1 : 0;
        const size_t budget = kMovesPerFiller * fillers;

        int64_t current = tailFinish(k, items, entry);
        SimResult simulated = simulate(k, items, entry);
        while (block.moves.size() < budget) {
            const std::vector<StinkyInstruction*> insts = stinkyOf(items);
            bool committed = false;
            for (size_t i = insts.size(); i-- > 0 && !committed;) {
                if (!isFiller(*insts[i]) || simulated.records[i].exposed <= 0) continue;
                committed = trySink(k, entry, insts[i], current, block);
            }
            if (!committed) break;
            changed_[k] = true;
            simulated = simulate(k, items, entry);
        }
    }

    /// Move \p filler right after the later WMMA of its segment that shortens the iteration
    /// most, if any does.
    bool trySink(size_t k, const SimState& entry, StinkyInstruction* filler, int64_t& current,
                 BlockReport& block) {
        std::vector<IRBase*>& items = items_[k];
        const size_t pos = static_cast<size_t>(
            std::find(items.begin(), items.end(), static_cast<IRBase*>(filler)) - items.begin());
        size_t first = pos;
        size_t last = pos + 1;
        while (first > 0 && !isSegmentBoundary(*items[first - 1])) --first;
        while (last < items.size() && !isSegmentBoundary(*items[last])) ++last;

        // Between an s_barrier_signal and its s_barrier_wait the timing is set by the other
        // waves, which a one-wave replay cannot see; reordering there measured slower.
        auto isBarrierOp = [&](size_t i, bool (*is)(const StinkyInstruction&)) {
            const auto* inst = i < items.size() ? dyn_cast<StinkyInstruction>(items[i]) : nullptr;
            return inst && is(*inst);
        };
        if (first > 0 && isBarrierOp(first - 1, isBarrierSignal) &&
            isBarrierOp(last, isBarrierWait))
            return false;

        // The filler stays ahead of every consumer, and of every later writer / reader the
        // DAG orders after it.
        std::vector<StinkyInstruction*> segment;
        for (size_t i = first; i < last; ++i) segment.push_back(cast<StinkyInstruction>(items[i]));
        const RegionDAG dag = buildRegisterDependencyDAG(segment);
        size_t limit = last;
        for (unsigned successor : dag.graph[pos - first])
            limit = std::min(limit, first + successor);

        int64_t bestFinish = current;
        std::optional<std::vector<IRBase*>> bestItems;
        size_t bestSlot = 0;
        int tried = 0;
        for (size_t slot = pos + 1; slot < limit && tried < kSlotsPerFiller; ++slot) {
            if (!isMatrixInstruction(*cast<StinkyInstruction>(items[slot]))) continue;
            ++tried;
            std::vector<IRBase*> candidate = items;
            candidate.erase(candidate.begin() + static_cast<std::ptrdiff_t>(pos));
            candidate.insert(candidate.begin() + static_cast<std::ptrdiff_t>(slot), filler);
            const int64_t finish = tailFinish(k, candidate, entry);
            if (finish < bestFinish) {
                bestFinish = finish;
                bestItems = std::move(candidate);
                bestSlot = slot;
            }
        }
        if (!bestItems) return false;

        auto stinkyIndex = [](const std::vector<IRBase*>& order, size_t itemPos) {
            size_t index = 0;
            for (size_t i = 0; i < itemPos; ++i) index += isa<StinkyInstruction>(order[i]) ? 1 : 0;
            return index;
        };
        block.moves.push_back({filler, stinkyIndex(items, pos), stinkyIndex(*bestItems, bestSlot),
                               current - bestFinish});
        PASS_DEBUG(std::cerr << "[" << kPassName << "] " << blocks_[k]->getLabel() << ": "
                             << filler->getHwInstDesc()->mnemonic << " " << block.moves.back().from
                             << " -> " << block.moves.back().to << " (-" << block.moves.back().gain
                             << " cycles)\n");
        items = std::move(*bestItems);
        current = bestFinish;
        return true;
    }

    std::vector<BasicBlock*> blocks_;
    std::vector<std::vector<IRBase*>> items_;
    std::vector<bool> changed_ = std::vector<bool>(blocks_.size(), false);
    const CoexecSimulator& sim_;
};

std::string jsonString(const std::string& text) {
    std::string out = "\"";
    for (char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (static_cast<unsigned char>(c) < 0x20) {
            out += ' ';
        } else {
            out += c;
        }
    }
    return out + "\"";
}

std::string opName(const StinkyInstruction& inst) {
    if (isLabel(inst))
        if (const auto* label = inst.getModifier<LabelData>()) return label->label + ":";
    const char* mnemonic = inst.getHwInstDesc()->mnemonic;
    return mnemonic ? mnemonic : "?";
}

void writeTimeline(std::ostream& os, const std::vector<StinkyInstruction*>& order,
                   const SimResult& sim) {
    os << "{\"finish\":" << sim.finish() << ",\"matrixIdle\":" << sim.matrixIdle
       << ",\"instructions\":[";
    for (size_t i = 0; i < order.size(); ++i) {
        const coexec::IssueRecord& r = sim.records[i];
        os << (i ? "," : "") << "{\"op\":" << jsonString(opName(*order[i]))
           << ",\"issue\":" << r.issue << ",\"stall\":" << r.stall << ",\"exposed\":" << r.exposed
           << ",\"msb\":" << (r.msbSwitch ? "true" : "false")
           << ",\"waitAlu\":" << (r.waitAlu ? "true" : "false")
           << ",\"filler\":" << (isFiller(*order[i]) ? "true" : "false") << "}";
    }
    os << "]}";
}

void writeReport(const std::string& path, const Function& func, const HWModel::MatrixIssue& mi,
                 bool analyzeOnly, const std::vector<LoopReport>& loops) {
    const std::filesystem::path file(path);
    std::error_code ec;
    if (file.has_parent_path()) std::filesystem::create_directories(file.parent_path(), ec);
    std::ofstream os(file);
    if (!os) {
        std::cerr << "[" << kPassName << "] cannot write " << path << "\n";
        return;
    }
    os << "{\"kernel\":" << jsonString(func.getName())
       << ",\"mode\":" << jsonString(analyzeOnly ? "analyze" : "repair") << ",\"model\":{"
       << "\"queueCapacity\":" << mi.queueCapacity << ",\"wmmaIssueCycles\":" << mi.wmmaIssueCycles
       << ",\"saluSgprToValu\":" << mi.saluSgprToValu << ",\"valuVgprToValu\":" << mi.valuVgprToValu
       << ",\"sccToBranch\":" << mi.sccToBranch << ",\"msbAfterMemOrWait\":" << mi.msbAfterMemOrWait
       << ",\"msbAfterSaluBeforeValu\":" << mi.msbAfterSaluBeforeValu
       << ",\"syncAfterMatrixCycles\":" << mi.syncAfterMatrixCycles
       << ",\"matrixVaVdstTailCycles\":" << mi.matrixVaVdstTailCycles
       << ",\"dsReturnIntervalCycles\":" << mi.dsReturnIntervalCycles << "},\"loops\":[";
    for (size_t l = 0; l < loops.size(); ++l) {
        const LoopReport& loop = loops[l];
        os << (l ? "," : "") << "{\"header\":" << jsonString(loop.header)
           << ",\"iterationCycles\":{\"before\":" << loop.iterationBefore
           << ",\"after\":" << loop.iterationAfter << "},\"blocks\":[";
        for (size_t b = 0; b < loop.blocks.size(); ++b) {
            const BlockReport& block = loop.blocks[b];
            os << (b ? "," : "") << "{\"label\":" << jsonString(block.label) << ",\"before\":";
            writeTimeline(os, block.beforeOrder, block.before);
            os << ",\"after\":";
            writeTimeline(os, block.afterOrder, block.after);
            os << ",\"moves\":[";
            for (size_t m = 0; m < block.moves.size(); ++m) {
                const Move& move = block.moves[m];
                os << (m ? "," : "") << "{\"op\":" << jsonString(opName(*move.filler))
                   << ",\"from\":" << move.from << ",\"to\":" << move.to
                   << ",\"gain\":" << move.gain << "}";
            }
            os << "]}";
        }
        os << "]}";
    }
    os << "]}\n";
}

class MatrixCoexecRepairPass : public StinkyInstPass {
   public:
    static char ID;

    explicit MatrixCoexecRepairPass(MatrixCoexecRepairOptions options)
        : options_(std::move(options)) {}

    const char* getName() const override {
        return kPassName;
    }

    PassID getPassID() const override {
        return &MatrixCoexecRepairPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& AM) override {
        (void)AM;
        const HWModel& hw = passCtx.getHWModel();
        if (hw.matrixIssue.queueCapacity <= 0) return PreservedAnalyses::all();

        SimConfig config;
        config.matrix = hw.matrixIssue;
        config.barrierWaitCycles = hw.barrier.signalToWaitLatency;
        if (hw.hazards.rules)
            config.hazards = {hw.hazards.rules, static_cast<size_t>(hw.hazards.numRules)};
        config.msbMode = passCtx.getAsmCapsConfig().vgprMsbMode;
        if (options_.predictWaitAlu) {
            config.waitAluContext = &passCtx;
            config.waitAluOptions = gfx1250InsertWaitAluOptions(options_.waitAluTrackValuVsrc);
        }
        const CoexecSimulator sim(config);

        const auto& arch = passCtx.getGemmTileConfig().arch;
        const GfxArchID archId = getGfxArchID(arch[0], arch[1], arch[2]);
        const uint32_t wavefrontSize = passCtx.getWavefrontSize();

        std::vector<LoopReport> reports;
        std::unordered_set<const BasicBlock*> visited;
        bool changed = false;
        for (const Loop& loop : detectLoops(func)) {
            std::vector<BasicBlock*> blocks;
            bool matrixWork = false;
            for (BasicBlock& bb : func) {
                if (!loop.contains(&bb) || visited.contains(&bb) ||
                    !passCtx.shouldProcessBasicBlock(bb))
                    continue;
                blocks.push_back(&bb);
                matrixWork |= hasMatrixWork(bb);
            }
            if (!matrixWork) continue;

            for (BasicBlock* bb : blocks) {
                AsmIRBuilder builder(*bb, archId);
                collapseExecMaskedRegions(*bb, builder, wavefrontSize);
            }
            LoopRepairer repairer(blocks, sim);
            LoopReport report = repairer.run(options_.analyzeOnly);
            if (!options_.analyzeOnly) repairer.apply();
            changed |= !options_.analyzeOnly && repairer.changed();
            for (BasicBlock* bb : blocks) {
                expandExecMaskedGroups(*bb);
                visited.insert(bb);
            }

            report.header = loop.headerBB->getLabel();
            size_t moves = 0;
            for (const BlockReport& block : report.blocks) moves += block.moves.size();
            std::ostringstream message;
            message << "loop '" << report.header << "': " << moves << " filler move(s), "
                    << "predicted iteration " << report.iterationBefore << " -> "
                    << report.iterationAfter << " cycles";
            emitRemark(passCtx, {OptimizationRemark::Kind::Analysis, kPassName, "LoopSummary",
                                 message.str()});
            reports.push_back(std::move(report));
        }

        if (!options_.reportPath.empty())
            writeReport(options_.reportPath, func, hw.matrixIssue, options_.analyzeOnly, reports);
        return changed ? preserveCFGAnalyses() : PreservedAnalyses::all();
    }

   private:
    MatrixCoexecRepairOptions options_;
};

char MatrixCoexecRepairPass::ID = 0;

}  // namespace

std::unique_ptr<Pass> createMatrixCoexecRepairPass(MatrixCoexecRepairOptions options) {
    return std::make_unique<MatrixCoexecRepairPass>(std::move(options));
}

}  // namespace stinkytofu
