// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// One wave's in-order issue next to the gfx1250 matrix pipe, replayed from
// HWModel::MatrixIssue. MatrixCoexecRepairPass asks it what an instruction order of a
// basic block costs: when each instruction issues, how long it stalls, and how much of
// that stall leaves the matrix pipe idle.
//
// Modeled: the matrix queue (queueCapacity WMMAs held, executed one at a time for their
// latency), operand readiness by producer / consumer class (implicit SCC / VCC included),
// the s_set_vgpr_msb switches InsertVgprMsbPass will place (VgprMsbPlanner) and what they
// cost, the s_wait_alu InsertWaitAlu will place (WaitAluTracker, SCHED_MODE 2 only), and
// s_wait_dscnt / s_barrier_wait, which issue at once and hold the next instruction until
// the DS returns (rate-limited, in order) or the signal's latency. Not modeled:
// s_wait_tensorcnt / loadcnt / kmcnt (no stall), other waves, and taken-branch redirects.

#include <cstdint>
#include <deque>
#include <span>
#include <vector>

#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

namespace stinkytofu {
class PassContext;
struct StinkyInstruction;

namespace coexec {

/// What an instruction occupies, for operand latencies and switch costs.
enum class IssueClass : uint8_t { Salu, Valu, Matrix, Lds, Memory, Branch, Other };

IssueClass classify(const StinkyInstruction& inst);

/// One wave's machine state between two instructions. Copyable, so a candidate order can be
/// replayed from the same entry state.
struct SimState {
    struct Producer {
        int64_t issue;
        /// Data return for matrix and LDS producers.
        int64_t ready;
        IssueClass cls;
        const StinkyInstruction* inst;
    };

    /// The issue slot the last instruction leaves free.
    int64_t nextIssue = 0;
    /// An s_wait_* / s_barrier_wait holds the instruction after it until this cycle.
    int64_t holdUntil = 0;
    /// Issue cycle and class of the last instruction issued.
    int64_t lastIssue = -1;
    IssueClass lastClass = IssueClass::Other;
    /// Execution end of each WMMA still in the matrix pipe, ascending.
    std::deque<int64_t> matrixEnds;
    /// When the matrix pipe finishes the WMMAs issued so far.
    int64_t matrixFreeAt = 0;
    /// Data return of each DS op still outstanding, in issue order.
    std::deque<int64_t> dsReturns;
    int64_t lastDsReturn = -1;
    /// VA_VDST retirement of each VGPR-writing VALU / matrix sub-issue still counted, in
    /// issue order: the counter retires in order, so an op leaves no earlier than the one
    /// before it.
    std::deque<int64_t> vaRetire;
    int64_t lastVaRetire = -1;
    int64_t lastBarrierSignal = -1;
    RegKeyMap<Producer> producers;
};

struct SimConfig {
    HWModel::MatrixIssue matrix{};
    int barrierWaitCycles = 0;
    /// Producer -> consumer gaps the DAG scheduler keeps (HWModel::Hazards); the cycle
    /// write-then-read rules are charged as minimum issue distances.
    std::span<const HazardRule> hazards;
    VgprMsbMode msbMode = VgprMsbMode::None;
    /// Non-null: predict the s_wait_alu InsertWaitAlu emits (SCHED_MODE 2 kernels).
    const PassContext* waitAluContext = nullptr;
    InsertWaitAluOptions waitAluOptions{};
};

/// The prediction for one input instruction (exec-mask groups: summed over the children).
struct IssueRecord {
    int64_t issue = -1;
    /// Cycles past the earliest slot the previous instruction left.
    int64_t stall = 0;
    /// Part of the stall with the matrix pipe idle.
    int64_t exposed = 0;
    bool msbSwitch = false;
    bool waitAlu = false;
};

struct SimResult {
    std::vector<IssueRecord> records;
    SimState exit;
    /// Matrix-pipe idle cycles between WMMAs of the run, counted from the entry state.
    int64_t matrixIdle = 0;

    int64_t finish() const {
        return exit.nextIssue > exit.matrixFreeAt ? exit.nextIssue : exit.matrixFreeAt;
    }
};

class CoexecSimulator {
   public:
    explicit CoexecSimulator(const SimConfig& config) : config_(config) {}

    /// Replay one basic block (every StinkyInstruction in program order, labels included)
    /// from \p entry. records[i] belongs to block[i]. \p waitAluHistory is what ran before
    /// the block (e.g. the rest of a loop body); it seeds the s_wait_alu prediction the way
    /// InsertWaitAlu seeds a block from its predecessors, and is not timed.
    SimResult run(const std::vector<StinkyInstruction*>& block, const SimState& entry,
                  const std::vector<StinkyInstruction*>& waitAluHistory = {}) const;

   private:
    SimConfig config_;
};

}  // namespace coexec
}  // namespace stinkytofu
