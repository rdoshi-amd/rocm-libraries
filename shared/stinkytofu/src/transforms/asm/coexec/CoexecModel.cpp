// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "CoexecModel.hpp"

#include <algorithm>
#include <climits>
#include <deque>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/transforms/asm/dag/HazardRules.hpp"

// Before dag/*.hpp so PASS_DEBUG inside those headers uses this name.
#define DEBUG_TYPE "CoexecModel"

#include "../dag/WaitAnchorUtils.hpp"

namespace stinkytofu {
namespace coexec {

namespace {

enum class Kind : uint8_t {
    Pseudo,  // label, phi, fence, marker: no issue
    Wmma,
    Valu,
    Salu,
    Ds,
    Vmem,    // global, buffer, flat, prefetch, tensor load
    WaitDs,  // s_wait_dscnt
    Sync,    // barrier wait, tensor/load/async wait
    Signal,  // barrier signal
    Other,
};

constexpr int kNever = INT_MIN / 4;

// InsertVgprMsbPass's state for "no MSB set yet" and "just after a label".
constexpr int kMsbNotRequired = -1;
constexpr int kMsbLabelBegin = -2;

enum PieceKind : uint8_t { kGap, kStall, kIssue };

bool hasRegOfType(const std::vector<StinkyRegister>& regs, bool (*pred)(RegType)) {
    for (const StinkyRegister& r : regs)
        if (r.isRegister() && pred(r.reg.type)) return true;
    return false;
}

bool isVccType(RegType t) {
    return t == RegType::VCC || t == RegType::VCC_LO || t == RegType::VCC_HI;
}

bool isVgprType(RegType t) {
    return t == RegType::V;
}

int nopCycles(const StinkyInstruction& inst) {
    const auto& srcs = inst.getSrcRegs();
    if (srcs.empty() || srcs[0].dataType != StinkyRegister::Type::LiteralInt) return 1;
    return std::max(0, srcs[0].getLiteralInt()) + 1;
}

// Cycles an s_set_vgpr_msb adds between `prev` and `next` (REPORT section 3.2).
int bankSwitchCost(Kind prev, Kind next, const HWModel::CoexecTiming& ct) {
    if (prev == Kind::Ds || prev == Kind::Vmem) return ct.msbAfterMemory;
    if (prev == Kind::Wmma && next == Kind::Valu) return ct.msbWmmaToValu;
    if (prev == Kind::Valu && next == Kind::Valu) return ct.msbValuToValu;
    return 0;
}

// Earliest cycle from `from` at which at most `allowed` of `done` are still in flight.
int releaseTime(const std::deque<int>& done, int from, int allowed) {
    std::vector<int> pending;
    for (int d : done)
        if (d > from) pending.push_back(d);
    if (static_cast<int>(pending.size()) <= allowed) return from;
    std::sort(pending.begin(), pending.end());
    return pending[pending.size() - allowed - 1];
}

}  // namespace

struct CoexecModel::Info {
    StinkyInstruction* inst = nullptr;
    Kind kind = Kind::Other;
    int issueCycles = 1;
    int latency = 0;
    bool label = false;
    bool call = false;
    bool scalarWrite = false;    // SALU that writes a scalar register (counted, not keyed)
    bool scalarSrcValu = false;  // VALU with a scalar or VCC source
    bool vccWrite = false;
    bool vccRead = false;
    bool vgprProducer = false;  // VALU-class instruction with a VGPR destination (va_vdst)
    bool barrierWait = false;
    bool dsLoad = false;
    int dsCount = -1;
    DsLoadDrainEntry drain;
    int msbReq = -1;
    bool hasVgpr = false;
    bool msbComputable = false;
    bool preferAfter = false;
};

const char* causeName(Cause cause) {
    switch (cause) {
        case Cause::None:
            return "none";
        case Cause::Issue:
            return "issue";
        case Cause::AfterWmma:
            return "after-wmma";
        case Cause::BankSwitch:
            return "bank switch";
        case Cause::WaitAlu:
            return "s_wait_alu";
        case Cause::ScalarInterlock:
            return "scalar interlock";
        case Cause::VccInterlock:
            return "vcc interlock";
        case Cause::WmmaSpacing:
            return "wmma spacing";
        case Cause::MatrixQueue:
            return "matrix queue";
        case Cause::LdsData:
            return "lds data";
        case Cause::Sync:
            return "sync";
    }
    return "?";
}

CoexecModel::CoexecModel(const PassContext& passCtx, Options options)
    : passCtx_(passCtx), hw_(passCtx.getHWModel()), options_(options) {
    numWaves_ = std::clamp(static_cast<int>(passCtx.getGemmTileConfig().NumWaves), 1, 4);
    modelBankSwitches_ = passCtx.getAsmCapsConfig().vgprMsbMode != VgprMsbMode::None;
}

CoexecModel::~CoexecModel() = default;

void CoexecModel::setBlock(const std::vector<StinkyInstruction*>& block) {
    info_.assign(block.size(), Info{});
    for (size_t id = 0; id < block.size(); ++id) {
        StinkyInstruction& inst = *block[id];
        Info& I = info_[id];
        I.inst = &inst;
        I.latency = inst.latencyCycles;
        I.issueCycles = std::max(1, inst.issueCycles);
        I.label = isLabel(inst);
        I.call = isCall(inst);

        if (isPseudoInst(&inst)) {
            I.kind = Kind::Pseudo;
            I.issueCycles = 0;
            continue;
        }
        if (isExecMaskGroup(inst)) {
            const auto* group = inst.getModifier<ExecGroupData>();
            I.issueCycles = group ? std::max<int>(1, static_cast<int>(group->children.size())) : 1;
        } else if (isMatrixInstruction(inst)) {
            I.kind = Kind::Wmma;
            I.issueCycles = 1;
            I.latency = std::max(1, inst.latencyCycles);
        } else if (isWaitCnt(inst) || inst.is(InstFlag::IF_WaitTensorCnt)) {
            using waitcnt::WaitCountSpec;
            const WaitCountSpec spec = dag::decodeWaitSpec(inst);
            I.issueCycles = 1;
            if (spec.tensorCount != WaitCountSpec::kUnused ||
                spec.loadCount != WaitCountSpec::kUnused ||
                spec.asyncCount != WaitCountSpec::kUnused) {
                I.kind = Kind::Sync;
            } else if (spec.dsCount != WaitCountSpec::kUnused) {
                I.kind = Kind::WaitDs;
                I.dsCount = spec.dsCount;
            }
        } else if (isBarrier(inst)) {
            I.issueCycles = 1;
            if (isBarrierWait(inst) || inst.getUnifiedOpcode() == GFX::s_barrier) {
                I.kind = Kind::Sync;
                I.barrierWait = true;
            } else {
                I.kind = Kind::Signal;
            }
        } else if (isDSRead(inst) || isDSWrite(inst) || isDSAtomic(inst)) {
            I.kind = Kind::Ds;
            I.dsLoad = isDSRead(inst);
            const HwInstDesc* desc = inst.getHwInstDesc();
            I.drain = makeDsLoadDrainEntry(hw_, inst.latencyCycles, desc ? desc->dsThroughput : 0,
                                           desc ? desc->dsMaxDrain : 0);
        } else if (isVmemTex(inst) || isFLATLoad(inst) || isFLATStore(inst) || isFLATAtomic(inst)) {
            I.kind = Kind::Vmem;
        } else if (isVectorALU(inst) || isTranscendental(inst)) {
            I.kind = Kind::Valu;
            I.scalarSrcValu = isValuWithScalarSrc(inst);
            I.vccWrite = isNonMatrixValu(inst) && hasRegOfType(inst.getDestRegs(), isVccType);
            I.vccRead = isNonMatrixValu(inst) && hasRegOfType(inst.getSrcRegs(), isVccType);
        } else if (inst.getUnifiedOpcode() == GFX::s_nop) {
            I.issueCycles = nopCycles(inst);
        } else if (isScalarALU(inst) && !isControlTransfer(inst)) {
            I.kind = Kind::Salu;
            I.scalarWrite = isScalarWritingSalu(inst);
        }
        I.vgprProducer = (I.kind == Kind::Valu || I.kind == Kind::Wmma) &&
                         hasRegOfType(inst.getDestRegs(), isVgprType);

        const auto [msbReq, hasVgpr] = computeRequiredMsb(&inst);
        I.msbReq = msbReq;
        I.hasVgpr = hasVgpr;
        I.msbComputable = isMsbComputableClass(inst);
        I.preferAfter = preferInsertAfter(inst);
    }
}

int CoexecModel::size() const {
    return static_cast<int>(info_.size());
}

StinkyInstruction* CoexecModel::inst(int id) const {
    return info_[id].inst;
}

InstFacts CoexecModel::facts(int id) const {
    const Info& I = info_[id];
    InstFacts f;
    f.wmma = I.kind == Kind::Wmma;
    f.salu = I.kind == Kind::Salu;
    f.valu = I.kind == Kind::Valu;
    f.scalarWrite = I.scalarWrite;
    f.scalarSrcValu = I.scalarSrcValu;
    f.vccWrite = I.vccWrite;
    f.vccRead = I.vccRead;
    f.dsLoad = I.dsLoad;
    return f;
}

// InsertVgprMsbPass's walk over one block: true at the position an s_set_vgpr_msb is
// placed before, and in \p nopBefore where the s_nop 0 that follows a label goes with it.
std::vector<bool> CoexecModel::predictBankSwitches(const std::vector<int>& order,
                                                   std::vector<bool>& nopBefore) const {
    const int n = static_cast<int>(order.size());
    std::vector<bool> before(n, false);
    nopBefore.assign(n, false);
    if (!modelBankSwitches_) return before;

    int currentMsb = kMsbNotRequired;
    int preferred = -1;
    for (int p = 0; p < n; ++p) {
        const Info& I = info_[order[p]];
        if (I.label) {
            currentMsb = kMsbLabelBegin;
            preferred = -1;
            continue;
        }
        if (I.kind == Kind::Pseudo) continue;
        if (I.call) {
            currentMsb = kMsbNotRequired;
            preferred = -1;
            continue;
        }
        const int insertAt = preferred >= 0 ? preferred : p;
        bool emitted = false;
        if (!I.hasVgpr || I.msbReq == currentMsb) {
            if (currentMsb == kMsbLabelBegin) currentMsb = kMsbNotRequired;
        } else {
            if (currentMsb == kMsbLabelBegin) nopBefore[insertAt] = true;
            before[insertAt] = true;
            currentMsb = I.msbReq;
            emitted = true;
        }
        if (emitted || I.msbComputable) preferred = -1;
        if (I.preferAfter) preferred = p + 1 < n ? p + 1 : -1;
    }
    return before;
}

std::vector<WaitAluNeed> CoexecModel::predictWaitAlu(const std::vector<int>& order) const {
    std::vector<WaitAluNeed> byId(info_.size());
    if (!options_.predictWaitAlu) return byId;

    auto tighter = [](int a, int b) { return a < 0 ? b : b < 0 ? a : std::min(a, b); };
    WaitAluTracker tracker(passCtx_, options_.waitAlu);
    for (int iter = 0; iter < 2; ++iter) {
        for (int id : order) {
            const StinkyInstruction& inst = *info_[id].inst;
            // InsertWaitAlu sees the group's instructions; a wait one of them needs is
            // charged before the group.
            if (isExecMaskGroup(inst)) {
                if (const auto* group = inst.getModifier<ExecGroupData>()) {
                    for (const StinkyInstruction* child : group->children) {
                        if (iter == 1) {
                            const WaitAluNeed need = tracker.query(*child);
                            byId[id].vaVdst = tighter(byId[id].vaVdst, need.vaVdst);
                            byId[id].vmVsrc = tighter(byId[id].vmVsrc, need.vmVsrc);
                        }
                        tracker.commit(*child);
                    }
                }
                continue;
            }
            if (iter == 1) byId[id] = tracker.query(inst);
            tracker.commit(inst);
        }
    }
    return byId;
}

SimResult CoexecModel::simulate(const std::vector<int>& order,
                                const std::vector<WaitAluNeed>& waitAluById) const {
    const HWModel::CoexecTiming& ct = hw_.coexecTiming;
    const int n = static_cast<int>(order.size());
    SimResult r;
    r.inst.resize(n);

    int numWmma = 0;
    for (int p = 0; p < n; ++p) {
        r.inst[p].window = numWmma;
        if (info_[order[p]].kind == Kind::Wmma) ++numWmma;
    }
    r.numWmma = numWmma;
    r.windows.assign(numWmma + 1, WindowTiming{});
    for (int p = 0; p < n; ++p) {
        const Info& I = info_[order[p]];
        if (I.kind == Kind::Wmma) r.windows[r.inst[p].window].wmmaPos = p;
        if (I.dsLoad) r.windows[r.inst[p].window].hasDsLoad = true;
    }
    const WaitAluNeed noWait;
    auto waitAluBefore = [&](int id) -> const WaitAluNeed& {
        return waitAluById.empty() ? noWait : waitAluById[id];
    };
    for (int id = 0; id < size(); ++id)
        if (waitAluBefore(id).any()) r.waitAluIds.push_back(id);
    if (numWmma == 0) return r;

    std::vector<bool> nopBefore;
    const std::vector<bool> msbBefore = predictBankSwitches(order, nopBefore);
    r.bankSwitches = static_cast<int>(std::count(msbBefore.begin(), msbBefore.end(), true));

    struct Piece {
        int lo, hi, pos;
        PieceKind kind;
    };
    std::vector<Piece> pending;  // pieces since the latest v_wmma issued
    std::deque<int> slotFree;    // per v_wmma: when its waiting slot frees
    std::vector<int> dsDone;     // per ds op, in issue order: when it completes
    std::deque<int> vgprDone;    // recent VALU-class VGPR writes: when they complete
    constexpr size_t kTrackedVgprWrites = 64;
    std::vector<int> wmmaLead(numWmma, 0);
    std::vector<int> wmmaRoom(numWmma, 0);  // queue wait plus pipe time
    int lastLeadFirstIter = 0;
    int lastRoomFirstIter = 0;

    int t = 0;
    int pipeEnd = 0;
    int lastWmmaIssue = kNever;
    int lastScalarWrite = kNever;
    int lastVccWrite = kNever;
    int lastSignal = kNever;
    int scoredStart = 0;
    Kind prevKind = Kind::Pseudo;

    for (int iter = 0; iter < 2; ++iter) {
        const bool scored = iter == 1;
        if (scored) scoredStart = t;
        int k = 0;
        for (int p = 0; p < n; ++p) {
            const int id = order[p];
            const Info& I = info_[id];
            InstTiming& out = r.inst[p];
            if (I.kind == Kind::Pseudo) {
                if (scored) out.reach = out.issue = t;
                continue;
            }

            // Gap: cycles the instruction cannot issue at all.
            const int gapStart = t;
            int gapCycles = 0;
            int gapLargest = 0;
            Cause gapCause = Cause::None;
            auto addGap = [&](int cycles, Cause cause) {
                if (cycles <= 0) return;
                gapCycles += cycles;
                if (cycles > gapLargest) {
                    gapLargest = cycles;
                    gapCause = cause;
                }
            };
            if (prevKind == Kind::Wmma) addGap(ct.afterWmmaIssue - 1, Cause::AfterWmma);
            if (msbBefore[p])
                addGap(bankSwitchCost(prevKind, I.kind, ct) + (nopBefore[p] ? 1 : 0),
                       Cause::BankSwitch);
            const WaitAluNeed& need = waitAluBefore(id);
            if (need.any()) {
                // Its own issue cycle, then va_vdst until enough VGPR writes have landed.
                // vm_vsrc waits are charged their issue cycle only.
                const int waitStart = gapStart + gapCycles;
                int done = waitStart + 1;
                if (need.vaVdst >= 0)
                    done = std::max(done, releaseTime(vgprDone, done, need.vaVdst));
                addGap(done - waitStart, Cause::WaitAlu);
            }
            const int reach = gapStart + gapCycles;

            // Stall: it could issue but is held.
            int issue = reach;
            Cause stallCause = Cause::None;
            auto hold = [&](int until, Cause cause) {
                if (until > issue) {
                    issue = until;
                    stallCause = cause;
                }
            };
            if (I.scalarSrcValu)
                hold(lastScalarWrite + ct.saluScalarToValu, Cause::ScalarInterlock);
            if (I.vccRead) hold(lastVccWrite + ct.vccWriteToRead, Cause::VccInterlock);
            if (I.kind == Kind::Wmma) {
                hold(lastWmmaIssue + ct.wmmaMinSpacing, Cause::WmmaSpacing);
                while (!slotFree.empty() && slotFree.front() <= issue) slotFree.pop_front();
                const int depth = ct.wmmaQueueDepth;
                if (depth > 0 && static_cast<int>(slotFree.size()) >= depth) {
                    hold(slotFree[slotFree.size() - depth], Cause::MatrixQueue);
                    while (!slotFree.empty() && slotFree.front() <= issue) slotFree.pop_front();
                }
            }
            if (I.kind == Kind::WaitDs) {
                int until = reach + std::max(0, ct.waitDscntIssue - 1);
                const int issued = static_cast<int>(dsDone.size());
                if (issued > I.dsCount) until = std::max(until, dsDone[issued - I.dsCount - 1]);
                hold(until, Cause::LdsData);
            }
            if (I.kind == Kind::Sync) {
                int until = std::max(reach, pipeEnd);
                if (I.barrierWait)
                    until = std::max(until, lastSignal + hw_.barrier.signalToWaitLatency);
                hold(until, Cause::Sync);
            }

            if (scored) {
                out.reach = reach;
                out.issue = issue;
                out.gap = gapCycles;
                out.gapCause = gapCause;
                out.stallCause = stallCause;
                out.ownCycles = gapCycles + I.issueCycles;
                out.bankSwitchBefore = msbBefore[p];
                out.waitAluBefore = need.any();
            }
            if (gapCycles > 0) pending.push_back({gapStart, reach, p, kGap});
            if (issue > reach) pending.push_back({reach, issue, p, kStall});
            pending.push_back({issue, issue + I.issueCycles, p, kIssue});

            if (I.kind == Kind::Wmma) {
                const int start = std::max(issue, pipeEnd);
                // The pipe idles from pipeEnd until this v_wmma reaches it; charge every
                // cycle to the piece of the wave's timeline it overlaps.
                if (scored && issue > pipeEnd) {
                    for (const Piece& piece : pending) {
                        const int overlap = std::min(piece.hi, issue) - std::max(piece.lo, pipeEnd);
                        if (overlap <= 0) continue;
                        InstTiming& blamed = r.inst[piece.pos];
                        if (piece.kind == kGap)
                            blamed.idleGap += overlap;
                        else if (piece.kind == kStall)
                            blamed.idleStall += overlap;
                        else
                            blamed.idleIssue += overlap;
                    }
                }
                pending.clear();
                if (scored) {
                    wmmaLead[k] = start - issue;
                    wmmaRoom[k] = start - issue + I.latency;
                } else {
                    lastLeadFirstIter = start - issue;
                    lastRoomFirstIter = start - issue + I.latency;
                }
                ++k;
                pipeEnd = start + I.latency;
                slotFree.push_back(start + ct.queueSlotRelease);
                lastWmmaIssue = issue;
                if (I.vgprProducer) vgprDone.push_back(pipeEnd);
            } else if (I.vgprProducer) {
                vgprDone.push_back(issue + I.latency);
            }
            while (vgprDone.size() > kTrackedVgprWrites) vgprDone.pop_front();
            if (I.scalarWrite) lastScalarWrite = issue;
            if (I.vccWrite) lastVccWrite = issue;
            if (I.kind == Kind::Signal) lastSignal = issue;
            if (I.kind == Kind::Ds) {
                // LDS returns in order. Within the queue depth one returns every
                // numWaves cycles (the drain model's spacing), past it by throughput.
                const int prev = dsDone.empty() ? kNever : dsDone.back();
                int inFlight = 0;
                for (int j = static_cast<int>(dsDone.size()) - 1; j >= 0 && dsDone[j] > issue; --j)
                    ++inFlight;
                const int depth = hw_.lds.readQueueDepth;
                const int spacing = depth <= 0 || inFlight < depth
                                        ? numWaves_
                                        : std::max(1, numWaves_ / std::max(1, I.drain.throughput));
                int done = std::max(issue + I.drain.latency, prev + spacing);
                if (I.drain.maxDrain > 0) done = std::min(done, issue + I.drain.maxDrain);
                dsDone.push_back(std::max(done, prev));
            }
            t = issue + I.issueCycles;
            prevKind = I.kind;
        }
    }
    r.cycles = t - scoredStart;

    for (int w = 0; w < numWmma; ++w) {
        r.windows[w].lead = wmmaLead[w];
        r.windows[w].leadIn = w == 0 ? lastLeadFirstIter : wmmaLead[w - 1];
        r.windows[w].room = w == 0 ? lastRoomFirstIter : wmmaRoom[w - 1];
    }
    r.windows[numWmma].leadIn = wmmaLead[numWmma - 1];
    r.windows[numWmma].room = wmmaRoom[numWmma - 1];
    for (int p = 0; p < n; ++p) {
        const InstTiming& out = r.inst[p];
        r.windows[out.window].idle += out.idle();
        r.windows[out.window].ownCycles += out.ownCycles;
        r.idle += out.idle();
    }
    return r;
}

SimResult CoexecModel::simulateExact(const std::vector<int>& order) const {
    return simulate(order, predictWaitAlu(order));
}

}  // namespace coexec
}  // namespace stinkytofu
