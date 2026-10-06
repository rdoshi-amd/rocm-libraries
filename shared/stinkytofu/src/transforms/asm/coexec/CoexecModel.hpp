// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"

namespace stinkytofu {
class PassContext;
struct HWModel;
struct StinkyInstruction;

namespace coexec {

/// What a cycle of the wave's issue timeline is spent on. Every cycle from one issue to
/// the next belongs to a piece of the later instruction, in this order: its gap (cycles
/// before it can issue at all), its stall (it could issue but is held) and its issue.
enum class Cause : uint8_t {
    None,
    Issue,            ///< the instruction's own issue cycles
    AfterWmma,        ///< the second cycle between a v_wmma and the next instruction
    BankSwitch,       ///< a predicted s_set_vgpr_msb right before it
    WaitAlu,          ///< a predicted s_wait_alu right before it
    ScalarInterlock,  ///< a VALU held after the latest SALU write of a scalar register
    VccInterlock,     ///< a VALU held after the VALU that wrote VCC
    WmmaSpacing,      ///< a v_wmma held by the minimum spacing between two v_wmma
    MatrixQueue,      ///< a v_wmma held because the matrix queue is full
    LdsData,          ///< an s_wait_dscnt: its own cycles and the LDS data it waits for
    Sync,             ///< a barrier, tensor or memory wait; the matrix pipe drains first
};

const char* causeName(Cause cause);

/// One instruction in the scored iteration.
struct InstTiming {
    int window = 0;  ///< v_wmma before it in the order; the tail window is numWmma
    int reach = 0;   ///< cycle it could issue, once its gap is over
    int issue = 0;   ///< cycle it issues
    int gap = 0;     ///< length of its gap piece
    Cause gapCause = Cause::None;
    Cause stallCause = Cause::None;  ///< what held it when issue > reach
    int ownCycles = 0;               ///< gap plus issue cycles: the time it costs by itself
    bool bankSwitchBefore = false;
    bool waitAluBefore = false;
    /// Matrix-pipe idle cycles charged to its gap, stall and issue pieces.
    int idleGap = 0;
    int idleStall = 0;
    int idleIssue = 0;
    int idle() const {
        return idleGap + idleStall + idleIssue;
    }
};

/// Window k holds the instructions between v_wmma k-1 and v_wmma k, and ends with v_wmma
/// k. Window 0 continues the previous iteration's tail (window numWmma).
struct WindowTiming {
    int wmmaPos = -1;  ///< position of v_wmma k; -1 for the tail
    int idle = 0;      ///< idle cycles charged to the window's instructions
    int leadIn = 0;    ///< cycles v_wmma k-1 waited in the matrix queue
    int lead = 0;      ///< cycles v_wmma k waited in the matrix queue
    /// Cycles from v_wmma k-1's issue until the pipe would run dry: its queue wait plus
    /// its pipe time. A window whose instructions take longer leaves the pipe idle.
    int room = 0;
    int ownCycles = 0;  ///< sum of the window's InstTiming::ownCycles
    bool hasDsLoad = false;
};

/// The model's classification of one instruction.
struct InstFacts {
    bool wmma = false;
    bool salu = false;
    bool valu = false;           ///< matrix instructions excluded
    bool scalarWrite = false;    ///< a SALU that writes a scalar register
    bool scalarSrcValu = false;  ///< a VALU with a scalar or VCC source
    bool vccWrite = false;
    bool vccRead = false;
    bool dsLoad = false;
};

struct SimResult {
    int idle = 0;    ///< matrix-pipe idle cycles in the scored iteration
    int cycles = 0;  ///< length of the scored iteration
    int numWmma = 0;
    int bankSwitches = 0;               ///< predicted s_set_vgpr_msb per iteration
    std::vector<int> waitAluIds;        ///< ids a predicted s_wait_alu precedes, ascending
    std::vector<InstTiming> inst;       ///< by position in the simulated order
    std::vector<WindowTiming> windows;  ///< numWmma + 1 entries, the tail last
};

/// Machine model of one basic block of a matrix loop: a wave issuing in order into the
/// matrix pipe and the queue in front of it, with the issue spacing, bank-switch,
/// interlock and LDS-wait costs of HWModel::CoexecTiming. The block is run twice as a
/// loop body and the second iteration is scored, so it starts with the queue, the
/// in-flight LDS loads and the scalar writes the first one leaves behind.
///
/// It predicts what later passes add instead of ignoring it: the s_set_vgpr_msb that
/// InsertVgprMsbPass places (computeRequiredMsb and its placement rule) and the s_wait_alu
/// InsertWaitAlu emits (its WaitAluTracker). Barrier, tensor and memory waits are sync
/// points: the wave resumes after the matrix pipe has drained, so its lead is gone.
class CoexecModel {
   public:
    struct Options {
        /// InsertWaitAlu runs (expert schedule mode 2), so its s_wait_alu are predicted.
        bool predictWaitAlu = true;
        InsertWaitAluOptions waitAlu;
    };

    CoexecModel(const PassContext& passCtx, Options options);
    ~CoexecModel();
    CoexecModel(const CoexecModel&) = delete;
    CoexecModel& operator=(const CoexecModel&) = delete;

    /// The block to model. An instruction's id is its index in \p block.
    void setBlock(const std::vector<StinkyInstruction*>& block);
    int size() const;
    StinkyInstruction* inst(int id) const;
    InstFacts facts(int id) const;

    /// The s_wait_alu InsertWaitAlu would emit before each id in \p order. Replays its
    /// scoreboard over two iterations, which costs far more than simulate().
    std::vector<WaitAluNeed> predictWaitAlu(const std::vector<int>& order) const;

    /// Simulate \p order, a permutation of the ids, with the s_wait_alu of \p waitAluById
    /// (from predictWaitAlu, possibly of a nearby order; empty for none).
    SimResult simulate(const std::vector<int>& order,
                       const std::vector<WaitAluNeed>& waitAluById) const;

    /// simulate() with the s_wait_alu predicted for \p order itself.
    SimResult simulateExact(const std::vector<int>& order) const;

   private:
    struct Info;
    std::vector<bool> predictBankSwitches(const std::vector<int>& order,
                                          std::vector<bool>& nopBefore) const;

    const PassContext& passCtx_;
    const HWModel& hw_;
    Options options_;
    int numWaves_ = 1;
    bool modelBankSwitches_ = true;
    std::vector<Info> info_;
};

}  // namespace coexec
}  // namespace stinkytofu
