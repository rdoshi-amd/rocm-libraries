// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// HazardGapAnalysisPass — scan a post-schedule ASM kernel for CDNA5 hardware hazard
// violations. For each rule in kCdna5HazardRules, finds every consumer instruction,
// walks backward to the nearest preceding producer of each source register, sums the
// real cycle cost (isMatrixInstruction ? latencyCycles : issueCycles) of instructions
// strictly between them, and flags any gap below the rule's required cycle count.
//
// Intra-block only: the DAG scheduler's hazard gate is also intra-region, and region
// boundaries are side-effect cuts that the scheduler cannot reorder across. Cross-region
// pairs already have fixed ordering enforced by barriers/waits.

#include "stinkytofu/analysis/asm/HazardGapAnalysisPass.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

namespace stinkytofu {
namespace {

int instCycles(const StinkyInstruction& inst) {
    return isMatrixInstruction(inst) ? inst.latencyCycles : inst.issueCycles;
}

const char* instMnemonic(const StinkyInstruction& inst) {
    return inst.hwInstDesc && inst.hwInstDesc->mnemonic ? inst.hwInstDesc->mnemonic : "?";
}

// Collect physical register keys (encoded as type*100000 + idx) for source or dest
// operands of the given regType.
void collectRegKeys(const std::vector<StinkyRegister>& regs, RegType regType,
                    std::vector<int>& out) {
    for (const auto& r : regs) {
        if (r.dataType != StinkyRegister::Type::Register) continue;
        if (r.reg.type != regType) continue;
        for (int i = 0; i < r.reg.num; ++i) out.push_back(static_cast<int>(r.reg.idx) + i);
    }
}

struct Violation {
    std::string ruleName;
    int required;
    const char* producerMnemonic;
    int producerIdx;
    const char* consumerMnemonic;
    int consumerIdx;
    int gap;
};

struct RuleStat {
    int pairs = 0;
    int violations = 0;
};

// Every producer/consumer pair of \p rule in \p bb, with the cycle gap between them.
std::vector<Violation> analyzeBlock(const BasicBlock& bb, const HazardRule& rule) {
    std::vector<const StinkyInstruction*> instrs;
    for (const IRBase& ir : bb) {
        const auto* inst = dyn_cast<StinkyInstruction>(&ir);
        if (inst) instrs.push_back(inst);
    }

    std::vector<Violation> results;
    for (const HazardGapPair& pair : findHazardGapPairs(instrs, rule)) {
        Violation v;
        v.ruleName = rule.name;
        v.required = rule.distance;
        v.producerMnemonic = instMnemonic(*instrs[pair.producer]);
        v.producerIdx = pair.producer;
        v.consumerMnemonic = instMnemonic(*instrs[pair.consumer]);
        v.consumerIdx = pair.consumer;
        v.gap = pair.gap;
        results.push_back(v);
    }
    return results;
}

class HazardGapAnalysisPass : public StinkyInstPass {
   public:
    static char ID;
    explicit HazardGapAnalysisPass(bool verbose) : verbose_(verbose) {}

    const char* getName() const override {
        return "HazardGapAnalysisPass";
    }
    PassID getPassID() const override {
        return &HazardGapAnalysisPass::ID;
    }

    PreservedAnalyses run(Function& func, PassContext& passCtx, AnalysisManager& /*AM*/) override {
        std::map<std::string, RuleStat> stats;
        for (int i = 0; i < kNumCdna5HazardRules; ++i) stats[kCdna5HazardRules[i].name] = {};
        for (int i = 0; i < kNumCdna5InterlockRules; ++i) stats[kCdna5InterlockRules[i].name] = {};

        bool anyViolation = false;

        // An interlock below its distance is a stall, not a wrong result: it is
        // reported as BELOW and never fails the analysis.
        auto check = [&](const BasicBlock& bb, const HazardRule& rule, bool interlock) {
            for (const auto& v : analyzeBlock(bb, rule)) {
                stats[v.ruleName].pairs++;
                bool bad = v.gap < v.required;
                if (bad) stats[v.ruleName].violations++;
                if (bad && !interlock) anyViolation = true;
                if (!bad && !verbose_) continue;
                if (bad && interlock && !verbose_) continue;
                const char* tag = !bad ? "OK" : interlock ? "BELOW" : "VIOLATION";
                std::cerr << "[HazardGapAnalysisPass] " << tag << " rule=" << v.ruleName
                          << " bb=" << bb.getLabel() << " producer[" << v.producerIdx
                          << "]=" << v.producerMnemonic << " consumer[" << v.consumerIdx
                          << "]=" << v.consumerMnemonic << " gap=" << v.gap
                          << " required>=" << v.required << "\n";
            }
        };
        for (const BasicBlock& bb : func) {
            for (int i = 0; i < kNumCdna5HazardRules; ++i)
                check(bb, kCdna5HazardRules[i], /*interlock=*/false);
            for (int i = 0; i < kNumCdna5InterlockRules; ++i)
                check(bb, kCdna5InterlockRules[i], /*interlock=*/true);
        }

        std::cerr << "\n[HazardGapAnalysisPass] Summary for " << func.getName() << ":\n";
        for (int i = 0; i < kNumCdna5HazardRules; ++i) {
            const auto& rule = kCdna5HazardRules[i];
            const auto& s = stats[rule.name];
            std::cerr << "  " << rule.name << " (>=" << rule.distance << " cyc):" << "  " << s.pairs
                      << " pair(s) checked," << "  " << s.violations << " VIOLATION(s)\n";
        }
        for (int i = 0; i < kNumCdna5InterlockRules; ++i) {
            const auto& rule = kCdna5InterlockRules[i];
            const auto& s = stats[rule.name];
            std::cerr << "  " << rule.name << " (interlock, >=" << rule.distance << " cyc):" << "  "
                      << s.pairs << " pair(s) checked," << "  " << s.violations
                      << " below the distance\n";
        }

        // Report failure through the pass framework rather than aborting the process,
        // so the driver controls the exit code and any teardown still runs.
        if (anyViolation) passCtx.setAnalysisFailed();
        return preserveCFGAnalyses();
    }

   private:
    bool verbose_;
};

char HazardGapAnalysisPass::ID = 0;

}  // namespace

std::unique_ptr<Pass> createHazardGapAnalysisPass(bool verbose) {
    return std::make_unique<HazardGapAnalysisPass>(verbose);
}

std::vector<HazardGapPair> findHazardGapPairs(const std::vector<const StinkyInstruction*>& instrs,
                                              const HazardRule& rule) {
    std::vector<HazardGapPair> results;
    // Only write->read cycle gaps are measurable here; other directions and units are
    // enforced by the scheduler.
    if (rule.dir != HazardDir::WriteThenRead || rule.unit != HazardUnit::Cycles) return results;

    // Prefix-sum of cycle costs for O(1) range sum.
    std::vector<int> cumCycles(instrs.size() + 1, 0);
    for (int i = 0; i < (int)instrs.size(); ++i)
        cumCycles[i + 1] = cumCycles[i] + instCycles(*instrs[i]);
    // Gap = sum of cycles of instructions strictly between producer and consumer.
    auto record = [&](int producer, int consumer) {
        results.push_back({producer, consumer, cumCycles[consumer] - cumCycles[producer + 1]});
    };

    if (rule.scope == HazardScope::AnyRegister) {
        // Any producer write stays in flight whatever later writes the family sees.
        int lastProducer = -1;
        for (int ci = 0; ci < (int)instrs.size(); ++ci) {
            const StinkyInstruction& inst = *instrs[ci];
            if (rule.isConsumer(inst) && lastProducer >= 0) record(lastProducer, ci);
            if (rule.isProducer(inst)) lastProducer = ci;
        }
        return results;
    }

    // lastWriter[regKey] = the most recent instruction that wrote that reg, plus
    // whether that writer is a hazard producer. A non-producer write (e.g. a load
    // dest) still overwrites the register's value, so it must overwrite the entry
    // too — otherwise a stale earlier producer would be paired against a consumer
    // that no longer reads the producer's value.
    struct Writer {
        int idx;
        bool isProducer;
    };
    std::map<int, Writer> lastWriter;

    for (int ci = 0; ci < (int)instrs.size(); ++ci) {
        const StinkyInstruction& inst = *instrs[ci];

        // Find the latest producer across all matching source regs. A source whose
        // most recent writer is not a producer contributes no hazard pair. Reads are
        // matched before this instruction's own writes are recorded, so one that reads
        // and writes the same register (v_add_co_ci_u32 and vcc) pairs with the writer
        // before it, not with itself.
        if (rule.isConsumer(inst)) {
            std::vector<int> srcKeys;
            collectRegKeys(inst.getSrcRegs(), rule.regType, srcKeys);
            int latestProducer = -1;
            for (int k : srcKeys) {
                auto it = lastWriter.find(k);
                if (it != lastWriter.end() && it->second.isProducer)
                    latestProducer = std::max(latestProducer, it->second.idx);
            }
            if (latestProducer >= 0) record(latestProducer, ci);
        }

        // Record the latest writer for every dest reg, producer or not.
        std::vector<int> destKeys;
        collectRegKeys(inst.getDestRegs(), rule.regType, destKeys);
        if (!destKeys.empty()) {
            bool isProd = rule.isProducer(inst);
            for (int k : destKeys) lastWriter[k] = {ci, isProd};
        }
    }
    return results;
}

}  // namespace stinkytofu
