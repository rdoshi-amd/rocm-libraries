// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// MatrixCoexecRepairPass — gfx1250 main loops: sink a VALU / SALU "filler" whose
// predicted stall leaves the matrix pipe idle to right after a later WMMA, where its
// latency co-executes with the queued matrix work.
//
// Every other instruction (WMMAs, memory ops, counter waits, barriers, branches, labels,
// exec-mask groups) is a fixed skeleton: its relative order never changes. A filler moves
// only later, only inside its segment (no label, branch, barrier, call or side effect in
// between), and never past a register consumer. A move is kept only when the coexec
// simulator (HWModel::MatrixIssue, with the s_set_vgpr_msb / s_wait_alu the later passes
// will add) predicts the loop body at least one cycle shorter.
//
// Run after the CFG is built and before InsertVgprMsbPass / InsertWaitAlu. An arch without
// a MatrixIssue model (queueCapacity 0) leaves the function untouched.

#include <memory>
#include <string>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

struct MatrixCoexecRepairOptions {
    /// Simulate and report only; keep the instruction order.
    bool analyzeOnly = false;
    /// Non-empty: write the per-loop prediction to this JSON file.
    std::string reportPath;
    /// Predict the s_wait_alu InsertWaitAlu emits (SCHED_MODE 2 kernels).
    bool predictWaitAlu = false;
    bool waitAluTrackValuVsrc = false;
};

STINKYTOFU_EXPORT std::unique_ptr<Pass> createMatrixCoexecRepairPass(
    MatrixCoexecRepairOptions options = {});

}  // namespace stinkytofu
