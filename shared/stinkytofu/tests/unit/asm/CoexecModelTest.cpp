// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// One hand-built block per CoexecModel rule. Each block is simulated as a loop body and
// the assertions read the second iteration, as the repair pass does.
#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <string>
#include <vector>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/IRConverter.hpp"
#include "transforms/asm/coexec/CoexecModel.hpp"

using namespace stinkytofu;
using namespace stinkytofu::coexec;

namespace {

const std::string kWmma =
    R"("st.v_wmma_scale_f32_16x16x128_f8f6f4"(v[100:115], v[116:131], %ACC%) { issueCycles = 1, latencyCycles = 8, mod.matrix_fmt = { fmtA = "MATRIX_FMT_FP8", fmtB = "MATRIX_FMT_FP8" } })";

// A v_wmma accumulating into v[8k:8k+7].
std::string wmma(int k) {
    const std::string acc = "v[" + std::to_string(8 * k) + ":" + std::to_string(8 * k + 7) + "]";
    std::string body = kWmma;
    body.replace(body.find("%ACC%"), 5, acc);
    return "  " + acc + " = " + body + "\n";
}

std::string smov(int sgpr) {
    return "  s" + std::to_string(sgpr) +
           " = \"st.s_mov_b32\"(1) { issueCycles = 1, latencyCycles = 2 }\n";
}

class CoexecModelTest : public ::testing::Test {
   protected:
    std::array<int, 3> arch{12, 5, 0};
    StinkyIRConverter converter{arch};
    PassContext passCtx;
    std::vector<StinkyInstruction*> block;

    void SetUp() override {
        GemmTileConfig config;
        config.arch = arch;
        passCtx.setGemmTileConfig(config);
        AsmCapsConfig caps;
        caps.vgprMsbMode = VgprMsbMode::Msb16;
        passCtx.setAsmCapsConfig(caps);
    }

    const HWModel::CoexecTiming& timing() const {
        return passCtx.getHWModel().coexecTiming;
    }

    // Simulate the body of a one-block function, in program order, without s_wait_alu.
    SimResult run(const std::string& body) {
        Function* func = converter.convertToFunction("st.func @t() {\n^entry:\n" + body + "}\n");
        EXPECT_NE(func, nullptr);
        block.clear();
        for (IRBase& ir : *func->begin())
            if (auto* inst = dyn_cast<StinkyInstruction>(&ir)) block.push_back(inst);
        CoexecModel::Options options;
        options.predictWaitAlu = false;
        model = std::make_unique<CoexecModel>(passCtx, options);
        model->setBlock(block);
        std::vector<int> order(block.size());
        std::iota(order.begin(), order.end(), 0);
        return model->simulate(order, {});
    }

    // Position of the n-th instruction whose mnemonic starts with `prefix`.
    int find(const std::string& prefix, int nth = 0) const {
        for (size_t p = 0; p < block.size(); ++p) {
            const char* name = block[p]->getHwInstDesc()->mnemonic;
            if (std::string(name).rfind(prefix, 0) == 0 && nth-- == 0) return static_cast<int>(p);
        }
        ADD_FAILURE() << "no " << prefix;
        return -1;
    }

    std::unique_ptr<CoexecModel> model;
};

TEST_F(CoexecModelTest, NextInstructionIssuesTwoCyclesAfterWmma) {
    const SimResult r = run(wmma(0) + smov(5) + wmma(1));
    const int w = find("v_wmma"), s = find("s_mov");
    EXPECT_EQ(r.inst[s].issue - r.inst[w].issue, timing().afterWmmaIssue);
    EXPECT_EQ(r.inst[s].gapCause, Cause::AfterWmma);
}

TEST_F(CoexecModelTest, BankSwitchAfterDsLoadCostsThree) {
    // The load's destination is in bank 3 and the v_wmma's operands in bank 0, so a
    // switch is needed between them. Right after the load it costs 3 cycles.
    const std::string load =
        "  v[768:771] = \"st.ds_load_b128\"(v400, LDS0) { issueCycles = 1, latencyCycles = 56, "
        "mod.ds = { na = 1, offset = 0, gds = false }, mod.memtoken = { tokens = [0] } }\n";
    SimResult r = run(load + wmma(0));
    int d = find("ds_load"), w = find("v_wmma");
    EXPECT_TRUE(r.inst[w].bankSwitchBefore);
    EXPECT_EQ(r.inst[w].gapCause, Cause::BankSwitch);
    EXPECT_EQ(r.inst[w].reach - r.inst[d].issue, 1 + timing().msbAfterMemory);

    // A SALU between them takes the switch: it is placed right after the SALU, for free.
    r = run(load + smov(5) + wmma(0));
    d = find("ds_load");
    w = find("v_wmma");
    EXPECT_TRUE(r.inst[w].bankSwitchBefore);
    EXPECT_EQ(r.inst[w].reach - r.inst[d].issue, 2);
}

TEST_F(CoexecModelTest, ScalarInterlockFiresForAnyScalarRegister) {
    // v_add_co_u32 reads s83; the SALU before it writes the unrelated s6.
    const std::string add =
        "  v1006, vcc_lo0 = \"st.v_add_co_u32\"(v1006, s83) { issueCycles = 1, latencyCycles = 5 "
        "}\n";
    SimResult r = run(wmma(0) + smov(6) + add + wmma(1));
    int s = find("s_mov"), v = find("v_add_co_u32");
    EXPECT_EQ(r.inst[v].issue - r.inst[s].issue, timing().saluScalarToValu);
    EXPECT_EQ(r.inst[v].stallCause, Cause::ScalarInterlock);

    // A VALU with only vector operands is not held.
    const std::string vecOnly =
        "  v1006 = \"st.v_add_nc_u32\"(v1006, v1007) { issueCycles = 1, latencyCycles = 5 }\n";
    r = run(wmma(0) + smov(6) + vecOnly + wmma(1));
    s = find("s_mov");
    v = find("v_add_nc_u32");
    EXPECT_EQ(r.inst[v].issue - r.inst[s].issue, 1);
    EXPECT_EQ(r.inst[v].stallCause, Cause::None);
}

TEST_F(CoexecModelTest, VccInterlockIsEightCycles) {
    const std::string pair =
        "  v1006, vcc_lo0 = \"st.v_add_co_u32\"(v1006, v1000) { issueCycles = 1, latencyCycles = 5 "
        "}\n"
        "  v1007, vcc_lo0 = \"st.v_add_co_ci_u32\"(v1007, 0, vcc_lo0) { issueCycles = 1, "
        "latencyCycles = 5 }\n";
    const SimResult r = run(wmma(0) + pair + wmma(1));
    const int lo = find("v_add_co_u32"), hi = find("v_add_co_ci_u32");
    EXPECT_EQ(r.inst[hi].issue - r.inst[lo].issue, timing().vccWriteToRead);
    EXPECT_EQ(r.inst[hi].stallCause, Cause::VccInterlock);
}

TEST_F(CoexecModelTest, QueueDepthAndSlotRelease) {
    // Back-to-back v_wmma keep the pipe busy. With the queue full, each one issues
    // queueSlotRelease cycles after the v_wmma wmmaQueueDepth places ahead starts.
    std::string body;
    for (int k = 0; k < 8; ++k) body += wmma(k);
    const SimResult r = run(body);
    const HWModel::CoexecTiming& ct = timing();
    const int pipe = 8;
    EXPECT_EQ(r.idle, 0);
    for (int k = 1; k < 8; ++k) {
        const InstTiming& cur = r.inst[find("v_wmma", k)];
        EXPECT_EQ(cur.issue - r.inst[find("v_wmma", k - 1)].issue, pipe);
        EXPECT_EQ(cur.stallCause, Cause::MatrixQueue);
        EXPECT_EQ(r.windows[k].lead, ct.wmmaQueueDepth * pipe - ct.queueSlotRelease);
    }
}

TEST_F(CoexecModelTest, LeadAndIdleAccounting) {
    // Window 1 holds 12 SALU. The wave reaches v_wmma 0 only 4 cycles ahead of the pipe
    // (its spacing after v_wmma 1 of the previous iteration), so the window's 14 cycles
    // against 8 + 4 of room leave the pipe idle for 2, charged to the last two SALU.
    std::string body = wmma(0);
    for (int i = 0; i < 12; ++i) body += smov(10 + i);
    body += wmma(1);
    const SimResult r = run(body);
    EXPECT_EQ(r.idle, 2);
    EXPECT_EQ(r.windows[1].idle, 2);
    EXPECT_EQ(r.windows[1].leadIn, timing().wmmaMinSpacing);
    EXPECT_EQ(r.windows[1].ownCycles, timing().afterWmmaIssue - 1 + 12 + 1);
    for (int i = 0; i < 12; ++i)
        EXPECT_EQ(r.inst[find("s_mov", i)].idleIssue, i >= 10 ? 1 : 0) << "s_mov " << i;
}

TEST_F(CoexecModelTest, PredictsInsertWaitAluFromItsScoreboard) {
    // A prefetch reading the address a VALU just wrote needs a va_vdst wait, which holds
    // it until every VALU-class write in flight has landed, the v_wmma's included. The
    // block runs as a loop, so the VALU also needs a vm_vsrc wait: it overwrites the
    // address the previous iteration's prefetch reads.
    const std::string body = wmma(0) +
                             "  v1006 = \"st.v_add_nc_u32\"(v1006, v1000) { issueCycles = 1, "
                             "latencyCycles = 5 }\n"
                             "  \"st.global_prefetch_b8\"(v[1006:1007], off) { issueCycles = 1, "
                             "latencyCycles = 1, mod.global = { offset = 0, th = \"TH_LOAD_NT\", "
                             "scope = \"SCOPE_SE\" } }\n" +
                             wmma(1);
    run(body);
    CoexecModel::Options options;
    options.waitAlu = gfx1250InsertWaitAluOptions(/*enableESM2TrackValuVsrc=*/true);
    CoexecModel exact(passCtx, options);
    exact.setBlock(block);
    std::vector<int> order(block.size());
    std::iota(order.begin(), order.end(), 0);
    const std::vector<WaitAluNeed> waits = exact.predictWaitAlu(order);
    const SimResult r = exact.simulate(order, waits);
    const int w = find("v_wmma"), v = find("v_add_nc_u32"), pf = find("global_prefetch");
    EXPECT_EQ(r.waitAluIds, (std::vector<int>{v, pf}));
    EXPECT_GE(waits[pf].vaVdst, 0);
    EXPECT_LT(waits[v].vaVdst, 0);
    EXPECT_GE(waits[v].vmVsrc, 0);
    EXPECT_EQ(r.inst[pf].gapCause, Cause::WaitAlu);
    const int wmmaDone = r.inst[w].issue + r.windows[0].lead + 8;
    EXPECT_EQ(r.inst[pf].reach, std::max(r.inst[v].issue + 5, wmmaDone));
}

TEST_F(CoexecModelTest, WaitDscntCostsThreeCyclesWhenItsCountIsMet) {
    const std::string wait =
        "  \"st.s_wait_dscnt\"(5) { issueCycles = 1, latencyCycles = 1, mod.swaitcnt = { vlcnt = "
        "-1, vscnt = -1, dlcnt = 5, dscnt = -1, kmcnt = -1 } }\n";
    const SimResult r = run(wmma(0) + smov(5) + wait + smov(6) + wmma(1));
    const int s0 = find("s_mov", 0), w = find("s_wait_dscnt"), s1 = find("s_mov", 1);
    EXPECT_EQ(r.inst[s1].issue - r.inst[w].reach, timing().waitDscntIssue);
    EXPECT_EQ(r.inst[w].reach - r.inst[s0].issue, 1);
    EXPECT_EQ(r.inst[w].stallCause, Cause::LdsData);
}

}  // namespace
