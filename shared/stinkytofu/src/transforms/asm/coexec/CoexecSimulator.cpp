// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CoexecSimulator.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <unordered_map>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/transforms/asm/VgprMsbPlanner.hpp"

namespace stinkytofu::coexec {
namespace {

bool isAnyWait(const StinkyInstruction& inst) {
    return isWaitCnt(inst) || inst.is(InstFlag::IF_WaitTensorCnt);
}

bool isLdsOp(const StinkyInstruction& inst) {
    return isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst);
}

/// The dscnt an s_wait_* holds out for, or -1.
int dsWaitCount(const StinkyInstruction& inst) {
    if (!isWaitCnt(inst)) return -1;
    if (const auto* data = inst.getModifier<SWaitCntData>()) return data->dlcnt;
    if (inst.getUnifiedOpcode() == GFX::s_wait_dscnt && !inst.getSrcRegs().empty() &&
        inst.getSrcReg(0).dataType == StinkyRegister::Type::LiteralInt)
        return inst.getSrcReg(0).getLiteralInt();
    return -1;
}

/// The cycle, not before \p from, at which at most \p allowed of the in-order \p retire
/// times are still ahead. Drops what has already retired.
int64_t releaseAt(std::deque<int64_t>& retire, int64_t from, int allowed) {
    while (!retire.empty() && retire.front() <= from) retire.pop_front();
    if (static_cast<int>(retire.size()) <= allowed) return from;
    return retire[retire.size() - static_cast<size_t>(allowed) - 1];
}

bool writesVgpr(const StinkyInstruction& inst) {
    for (const StinkyRegister& dst : inst.getDestRegs())
        if (dst.isRegister() && dst.reg.type == RegType::V) return true;
    return false;
}

/// VOP3PX2 / VOP3PX3 are an LD_SCALE + WMMA pair: two VA_VDST sub-issues.
bool isMatrixScalePair(const StinkyInstruction& inst) {
    const MicrocodeFormat mc = inst.getHwInstDesc()->microcode;
    return mc == MicrocodeFormat::MC_VOP3PX2 || mc == MicrocodeFormat::MC_VOP3PX3;
}

struct StreamEntry {
    const StinkyInstruction* inst;
    size_t owner;
};

/// The instructions as InsertVgprMsbPass and InsertWaitAlu see them: exec-mask groups
/// contribute their children.
std::vector<StreamEntry> flatten(const std::vector<StinkyInstruction*>& block) {
    std::vector<StreamEntry> stream;
    stream.reserve(block.size());
    for (size_t i = 0; i < block.size(); ++i) {
        const StinkyInstruction* inst = block[i];
        if (isExecMaskGroup(*inst)) {
            if (const auto* group = inst->getModifier<ExecGroupData>())
                for (const StinkyInstruction* child : group->children) stream.push_back({child, i});
            continue;
        }
        stream.push_back({inst, i});
    }
    return stream;
}

struct SwitchSlot {
    bool present = false;
    bool withNop = false;
};

/// The s_set_vgpr_msb InsertVgprMsbPass places right before each stream entry.
std::vector<SwitchSlot> planSwitches(const std::vector<StreamEntry>& stream, VgprMsbMode mode) {
    std::vector<SwitchSlot> slots(stream.size());
    if (mode == VgprMsbMode::None) return slots;

    std::unordered_map<const StinkyInstruction*, size_t> position;
    for (size_t i = 0; i < stream.size(); ++i) position.emplace(stream[i].inst, i);

    VgprMsbPlanner planner(mode);
    planner.beginBlock();
    for (const StreamEntry& entry : stream) {
        std::optional<PlannedMsbSwitch> planned = planner.observe(*entry.inst);
        if (!planned) continue;
        auto at = position.find(planned->insertBefore);
        if (at == position.end()) continue;
        slots[at->second].present = true;
        slots[at->second].withNop |= planned->withNop;
    }
    return slots;
}

int64_t operandReady(const SimState::Producer& producer, IssueClass consumer, RegType type,
                     const HWModel::MatrixIssue& mi) {
    switch (producer.cls) {
        case IssueClass::Salu:
            if (consumer == IssueClass::Branch && type == RegType::SCC)
                return producer.issue + mi.sccToBranch;
            if (consumer == IssueClass::Salu || consumer == IssueClass::Branch ||
                consumer == IssueClass::Other)
                return producer.issue + 1;
            return producer.issue + mi.saluSgprToValu;
        case IssueClass::Valu:
            return producer.issue + mi.valuVgprToValu;
        case IssueClass::Matrix:
            // D -> C accumulation is forwarded; the serial pipe already orders execution.
            if (consumer == IssueClass::Matrix) return producer.issue + mi.wmmaIssueCycles;
            return producer.ready;
        case IssueClass::Lds:
            return producer.ready;
        default:
            return producer.issue + 1;
    }
}

int issueCost(const StinkyInstruction& inst, IssueClass cls, const HWModel::MatrixIssue& mi) {
    if (cls == IssueClass::Matrix) return std::max(1, mi.wmmaIssueCycles);
    return std::max(1, inst.issueCycles);
}

}  // namespace

IssueClass classify(const StinkyInstruction& inst) {
    if (isMatrixInstruction(inst)) return IssueClass::Matrix;
    if (isLdsOp(inst)) return IssueClass::Lds;
    if (isBranch(inst) || isCall(inst)) return IssueClass::Branch;
    if (isAnyWait(inst) || isBarrier(inst)) return IssueClass::Other;
    if (inst.is(InstFlag::IF_VALU)) return IssueClass::Valu;
    if (isVmemTex(inst) || isGlobalMemLoad(inst) || isGlobalMemStore(inst) ||
        isGlobalMemAtomic(inst))
        return IssueClass::Memory;
    if (isScalarALU(inst)) return IssueClass::Salu;
    return IssueClass::Other;
}

SimResult CoexecSimulator::run(const std::vector<StinkyInstruction*>& block, const SimState& entry,
                               const std::vector<StinkyInstruction*>& waitAluHistory) const {
    const HWModel::MatrixIssue& mi = config_.matrix;
    SimResult result;
    result.records.resize(block.size());
    result.exit = entry;
    SimState& s = result.exit;

    const std::vector<StreamEntry> stream = flatten(block);
    const std::vector<SwitchSlot> switches = planSwitches(stream, config_.msbMode);
    std::unique_ptr<WaitAluTracker> waitAlu;
    if (config_.waitAluContext) {
        waitAlu = std::make_unique<WaitAluTracker>(*config_.waitAluContext, config_.waitAluOptions);
        for (const StreamEntry& past : flatten(waitAluHistory))
            if (!isPseudoInst(past.inst)) waitAlu->commit(*past.inst);
    }

    for (size_t i = 0; i < stream.size(); ++i) {
        const StinkyInstruction& inst = *stream[i].inst;
        if (isPseudoInst(&inst)) continue;
        IssueRecord& record = result.records[stream[i].owner];
        const IssueClass cls = classify(inst);
        const int64_t slot = s.nextIssue;
        int64_t earliest = std::max(slot, s.holdUntil);

        if (switches[i].present) {
            record.msbSwitch = true;
            if (switches[i].withNop) earliest += 1;
            if (s.lastIssue >= 0) {
                switch (s.lastClass) {
                    case IssueClass::Lds:
                    case IssueClass::Memory:
                    case IssueClass::Other:
                        earliest = std::max(earliest, s.lastIssue + 1 + mi.msbAfterMemOrWait);
                        break;
                    case IssueClass::Salu:
                        if (cls == IssueClass::Valu)
                            earliest = std::max(earliest, s.lastIssue + mi.msbAfterSaluBeforeValu);
                        break;
                    default:
                        break;
                }
            }
        }

        const bool isSync = isAnyWait(inst) || isBarrier(inst);
        if (isSync && s.lastClass == IssueClass::Matrix)
            earliest = std::max(earliest, s.lastIssue + mi.syncAfterMatrixCycles);

        if (waitAlu) {
            const WaitAluNeed need = waitAlu->query(inst);
            if (need.any()) {
                record.waitAlu = true;
                earliest += 1;
                if (need.vaVdst >= 0) earliest = releaseAt(s.vaRetire, earliest, need.vaVdst);
            }
            waitAlu->commit(inst);
        }

        auto readKey = [&](const RegKey& key) {
            auto producer = s.producers.find(key);
            if (producer == s.producers.end()) return;
            const SimState::Producer& p = producer->second;
            earliest = std::max(earliest, operandReady(p, cls, key.type, mi));
            for (const HazardRule& rule : config_.hazards) {
                if (rule.unit != HazardUnit::Cycles || rule.dir != HazardDir::WriteThenRead ||
                    rule.regType != key.type || !rule.isProducer(*p.inst) || !rule.isConsumer(inst))
                    continue;
                earliest = std::max(earliest, p.issue + rule.distance);
            }
        };
        for (const StinkyRegister& src : inst.getSrcRegs())
            if (!isPseudoReg(src)) forEachRegUnit(src, readKey);
        if (inst.is(InstFlag::IF_ImplicitReadSCC)) readKey({RegType::SCC, 0});
        if (inst.is(InstFlag::IF_ImplicitReadVCC)) {
            readKey({RegType::VCC_LO, 0});
            readKey({RegType::VCC, 0});
        }

        if (cls == IssueClass::Matrix && mi.queueCapacity > 0) {
            std::deque<int64_t>& held = s.matrixEnds;
            while (!held.empty() && held.front() <= earliest) held.pop_front();
            if (static_cast<int>(held.size()) >= mi.queueCapacity) {
                earliest = std::max(earliest, held[held.size() - mi.queueCapacity]);
                while (!held.empty() && held.front() <= earliest) held.pop_front();
            }
        }

        const int64_t issue = earliest;
        if (const int dsAllowed = dsWaitCount(inst); dsAllowed >= 0) {
            std::deque<int64_t>& outstanding = s.dsReturns;
            while (!outstanding.empty() && outstanding.front() <= issue) outstanding.pop_front();
            if (static_cast<int>(outstanding.size()) > dsAllowed) {
                const int64_t release = outstanding[outstanding.size() - dsAllowed - 1];
                s.holdUntil = std::max(s.holdUntil, release);
                while (!outstanding.empty() && outstanding.front() <= release)
                    outstanding.pop_front();
            }
        } else if (isBarrierWait(inst) && s.lastBarrierSignal >= 0) {
            s.holdUntil = std::max(s.holdUntil, s.lastBarrierSignal + config_.barrierWaitCycles);
        }
        if (record.issue < 0) record.issue = issue;
        record.stall += issue - slot;
        record.exposed += std::max<int64_t>(0, issue - std::max(slot, s.matrixFreeAt));

        auto countVa = [&](int64_t done) {
            s.lastVaRetire = std::max(done, s.lastVaRetire);
            s.vaRetire.push_back(s.lastVaRetire);
        };
        int64_t ready = issue + 1;
        if (cls == IssueClass::Matrix) {
            const int64_t start = std::max(issue, s.matrixFreeAt);
            result.matrixIdle += start - s.matrixFreeAt;
            ready = start + std::max(1, inst.latencyCycles);
            s.matrixFreeAt = ready;
            s.matrixEnds.push_back(ready);
            if (isMatrixScalePair(inst)) countVa(start);
            countVa(ready + mi.matrixVaVdstTailCycles);
        } else if (cls == IssueClass::Lds) {
            ready = issue + std::max(1, inst.latencyCycles);
            if (s.lastDsReturn >= 0)
                ready = std::max(ready, s.lastDsReturn + mi.dsReturnIntervalCycles);
            s.lastDsReturn = ready;
            s.dsReturns.push_back(ready);
        } else if (cls == IssueClass::Valu && writesVgpr(inst)) {
            countVa(issue + mi.valuVgprToValu);
        }
        while (s.vaRetire.size() > 256 && s.vaRetire.front() <= issue) s.vaRetire.pop_front();
        if (isBarrierSignal(inst)) s.lastBarrierSignal = issue;

        auto writeKey = [&](const RegKey& key) { s.producers[key] = {issue, ready, cls, &inst}; };
        for (const StinkyRegister& dst : inst.getDestRegs())
            if (!isPseudoReg(dst)) forEachRegUnit(dst, writeKey);
        if (inst.is(InstFlag::IF_ImplicitWriteSCC)) writeKey({RegType::SCC, 0});

        s.lastIssue = issue;
        s.lastClass = cls;
        s.nextIssue = issue + issueCost(inst, cls, mi);
    }
    return result;
}

}  // namespace stinkytofu::coexec
