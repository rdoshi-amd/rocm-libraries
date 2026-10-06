// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "stinkytofu/analysis/asm/ClusterBarrierHandoffVerifier.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"

namespace stinkytofu {
namespace {

bool barrier(const StinkyInstruction& inst, bool signal, int id) {
    if (signal ? !isBarrierSignal(inst) : !isBarrierWait(inst)) return false;
    const auto& srcs = inst.getSrcRegs();
    return !srcs.empty() && srcs[0].dataType == StinkyRegister::Type::LiteralInt &&
           srcs[0].getLiteralInt() == id;
}

// Reconstruct instruction successors from final branch metadata. Do not rely
// on BB edges: flattening, prefetch insertion and instruction removal run after
// the last CFGBuilderPass. Labels on blocks and inline labels are both supported.
struct FinalControlFlow {
    std::vector<StinkyInstruction*> instructions;
    std::unordered_map<std::string, size_t> labels;
    bool ambiguous = false;

    explicit FinalControlFlow(Function& func) {
        auto label = [&](const std::string& name, size_t index) {
            if (name.empty()) return;
            const auto [it, inserted] = labels.emplace(name, index);
            ambiguous |= !inserted && it->second != index;
        };
        std::unordered_map<std::string, size_t> blockLabels;
        for (BasicBlock& bb : func) {
            blockLabels.emplace(bb.getLabel(), instructions.size());
            for (IRBase& ir : bb) {
                auto* inst = dyn_cast<StinkyInstruction>(&ir);
                if (inst == nullptr) continue;
                if (isLabel(*inst)) {
                    if (const auto* data = inst->getModifier<LabelData>())
                        label(data->label, instructions.size());
                }
                instructions.push_back(inst);
            }
        }
        // Inline labels are the actual branch destinations. A late pass may
        // insert instructions before the label that originally named its BB.
        for (const auto& [name, index] : blockLabels) labels.emplace(name, index);
    }

    bool targets(const std::vector<std::string>& names, std::vector<size_t>& out) const {
        if (names.empty()) return false;
        for (const std::string& name : names) {
            const auto found = labels.find(name);
            if (found == labels.end()) return false;
            out.push_back(found->second);
        }
        return true;
    }

    // A call cannot discharge the handoff merely by returning. Crossing a known
    // barrier-free helper is safe; a synchronization-bearing or unknown helper
    // requires interprocedural phase reasoning and is deliberately rejected.
    bool barrierFreeCall(const StinkyInstruction& call) const {
        std::vector<size_t> work;
        if (!targets(getCallTargets(call), work)) return false;
        std::unordered_set<size_t> seen;
        while (!work.empty()) {
            size_t index = work.back();
            work.pop_back();
            for (; index < instructions.size(); ++index) {
                if (!seen.insert(index).second) break;
                const auto& inst = *instructions[index];
                if (isBarrierSignal(inst) || isBarrierWait(inst)) return false;
                if (isEndOfFunction(inst)) break;
                if (isCall(inst) && !targets(getCallTargets(inst), work)) return false;
                if (!isBranch(inst)) continue;
                if (!targets(getBranchTargets(inst), work)) return false;
                if (isUnconditionalBranch(inst)) break;
            }
        }
        return true;
    }

    std::string location(size_t index) const {
        if (index >= instructions.size()) return "function end";
        return instructions[index]->getParent()->getLabel() + " instruction " +
               std::to_string(index);
    }
};

// A small, conservative predicate domain: only equality facts established by
// scalar eq/ne-u32 branches. This is enough to preserve a zero-trip guard across
// shadow initialization and a later re-test of the same, unmodified counter.
// Unsupported operations lose facts; they never justify pruning a CFG edge.
struct ScalarFacts {
    struct Predicate {
        uint32_t reg;
        uint32_t value;
        bool equal;
        bool operator==(const Predicate&) const = default;
    };
    std::unordered_map<uint32_t, uint32_t> equalities;
    std::optional<Predicate> predicate;
    std::optional<bool> scc;

    bool merge(const ScalarFacts& other) {
        bool changed = false;
        for (auto it = equalities.begin(); it != equalities.end();) {
            auto found = other.equalities.find(it->first);
            if (found == other.equalities.end() || found->second != it->second) {
                it = equalities.erase(it);
                changed = true;
            } else {
                ++it;
            }
        }
        if (predicate && predicate != other.predicate) {
            predicate.reset();
            changed = true;
        }
        if (scc && scc != other.scc) {
            scc.reset();
            changed = true;
        }
        return changed;
    }

    void transfer(const StinkyInstruction& inst) {
        bool writesScc = inst.is(InstFlag::IF_ImplicitWriteSCC);
        for (const auto& dst : inst.getDestRegs()) {
            if (!dst.isRegister()) continue;
            writesScc |= dst.reg.type == RegType::SCC;
            if (dst.reg.type != RegType::S) continue;
            if (dst.reg.offset != 0) {
                // Do not reason about symbolic/offset destination aliases.
                equalities.clear();
                predicate.reset();
                continue;
            }
            for (uint32_t i = dst.reg.idx; i < dst.reg.idx + dst.reg.num; ++i) {
                equalities.erase(i);
                if (predicate && predicate->reg == i) predicate.reset();
            }
        }
        if (writesScc) {
            predicate.reset();
            scc.reset();
        }
        const auto opcode = inst.getUnifiedOpcode();
        if (opcode != GFX::s_cmp_eq_u32 && opcode != GFX::s_cmp_lg_u32) return;
        // Also handle minimal IR whose implicit SCC destination is not explicit.
        predicate.reset();
        scc.reset();
        const auto& srcs = inst.getSrcRegs();
        if (srcs.size() < 2) return;
        const StinkyRegister* reg = &srcs[0];
        const StinkyRegister* value = &srcs[1];
        if (reg->dataType == StinkyRegister::Type::LiteralInt) std::swap(reg, value);
        if (!reg->isRegister() || reg->reg.type != RegType::S || reg->reg.num != 1 ||
            reg->reg.offset != 0 || reg->reg.isMinus || reg->reg.isAbs ||
            value->dataType != StinkyRegister::Type::LiteralInt) return;
        predicate = Predicate{reg->reg.idx, static_cast<uint32_t>(value->getLiteralInt()),
                              opcode == GFX::s_cmp_eq_u32};
        if (auto found = equalities.find(predicate->reg); found != equalities.end())
            scc = (found->second == predicate->value) == predicate->equal;
    }

    bool assumeScc(bool value) {
        if (scc && *scc != value) return false;
        if (predicate && value == predicate->equal)
            equalities[predicate->reg] = predicate->value;
        scc = value;
        return true;
    }
};

class ClusterBarrierHandoffVerifierPass : public Pass {
   public:
    static char ID;
    PassID getPassID() const override { return &ID; }
    const char* getName() const override { return "ClusterBarrierHandoffVerifier"; }
    PreservedAnalyses run(Function& func, PassContext&, AnalysisManager&) override {
        const std::string error = verifyClusterBarrierHandoffs(func);
        if (!error.empty()) report_fatal_error(error.c_str());
        return PreservedAnalyses::all();
    }
};
char ClusterBarrierHandoffVerifierPass::ID = 0;

}  // namespace

std::string verifyClusterBarrierHandoffs(Function& func) {
    FinalControlFlow cfg(func);
    bool hasClusterWait = false;
    for (const auto* inst : cfg.instructions) hasClusterWait |= barrier(*inst, false, -3);
    if (!hasClusterWait) return {};
    if (cfg.ambiguous) return "cluster handoff: ambiguous labels in " + func.getName();

    enum Phase { Joined, Pending, LocalArrival };
    struct State {
        Phase phase = Joined;
        size_t clusterWait = 0;
        ScalarFacts facts;
    };
    // Keep joined paths separate from pending paths. Merging them would lose
    // the zero-iteration predicate of a wait that only executes on the exit path.
    // At each (instruction, phase), intersection only removes facts, so loops
    // converge without an iteration limit or an unsafe 'give up and accept'.
    std::vector<std::array<std::optional<State>, 3>> incoming(cfg.instructions.size());
    std::deque<std::pair<size_t, Phase>> work;
    auto enqueue = [&](size_t index, const State& state) {
        if (index >= cfg.instructions.size()) return;
        auto& old = incoming[index][state.phase];
        if (!old) {
            old = state;
            work.emplace_back(index, state.phase);
        } else if (old->facts.merge(state.facts)) {
            work.emplace_back(index, state.phase);
        }
    };
    enqueue(0, State{});
    while (!work.empty()) {
        auto [index, phase] = work.front();
        work.pop_front();
        State state = *incoming[index][phase];
        const auto& inst = *cfg.instructions[index];
        auto error = [&](const std::string& reason) {
            const std::string origin = state.phase == Joined ? "entry" :
                "wait at " + cfg.location(state.clusterWait);
            return "cluster handoff: " + func.getName() + ": " + origin + " reaches " +
                   cfg.location(index) + " " + reason;
        };
        if (barrier(inst, true, -3) && state.phase != Joined)
            return error("without a workgroup signal/wait pair");
        if (barrier(inst, false, -3)) {
            state.clusterWait = index;
            state.phase = Pending;
        } else if (barrier(inst, true, -1) && state.phase == Pending) {
            state.phase = LocalArrival;
        } else if (barrier(inst, false, -1) && state.phase == LocalArrival) {
            state.phase = Joined;
        }
        if (isEndOfFunction(inst)) continue;
        if (isCall(inst)) {
            if (!cfg.barrierFreeCall(inst))
                return error("through a call whose barrier behavior is not proven");
            // Even a barrier-free helper may overwrite scalar registers/SCC.
            state.facts = {};
        }
        state.facts.transfer(inst);
        if (!isBranch(inst)) {
            enqueue(index + 1, state);
            continue;
        }
        std::vector<size_t> successors;
        if (!cfg.targets(getBranchTargets(inst), successors))
            return error("through an unresolved branch");
        const auto opcode = inst.getUnifiedOpcode();
        const bool sccBranch = opcode == GFX::s_cbranch_scc0 || opcode == GFX::s_cbranch_scc1;
        State taken = state;
        if (!sccBranch || taken.facts.assumeScc(opcode == GFX::s_cbranch_scc1)) {
            for (size_t successor : successors) enqueue(successor, taken);
        }
        if (!isUnconditionalBranch(inst) &&
            (!sccBranch || state.facts.assumeScc(opcode == GFX::s_cbranch_scc0)))
            enqueue(index + 1, state);
    }
    return {};
}

std::unique_ptr<Pass> createClusterBarrierHandoffVerifierPass() {
    return std::make_unique<ClusterBarrierHandoffVerifierPass>();
}
}  // namespace stinkytofu
