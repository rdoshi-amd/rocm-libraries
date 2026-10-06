// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/HWModel.hpp"
#include "stinkytofu/transforms/asm/InsertWaitAluPass.hpp"
#include "transforms/asm/coexec/CoexecSimulator.hpp"

using namespace stinkytofu;
using namespace stinkytofu::coexec;
using namespace stinkytofu::test;

namespace {

class CoexecSimulatorTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;
    const HWModel::MatrixIssue& mi = hwModelForArch({12, 5, 0}).matrixIssue;

    void SetUp() override {
        func = std::make_unique<Function>("coexec_simulator_test");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
    }

    StinkyInstruction* create(GFX op) {
        AsmIRBuilder builder(*bb, arch);
        return builder.create(getMCIDByUOp(op, arch));
    }

    StinkyInstruction* wmma(int dst) {
        StinkyInstruction* inst = create(GFX::v_wmma_f32_16x16x32_bf16);
        inst->addDestReg(vgpr(dst, 8));
        inst->addSrcReg(vgpr(100, 8));
        inst->addSrcReg(vgpr(108, 8));
        inst->addSrcReg(vgpr(dst, 8));
        inst->latencyCycles = 8;
        return inst;
    }

    StinkyInstruction* saluWrite(int sdst) {
        StinkyInstruction* inst = create(GFX::s_mov_b32);
        inst->addDestReg(sgpr(sdst));
        inst->addSrcReg(StinkyRegister(1));
        return inst;
    }

    StinkyInstruction* valu(int dst, StinkyRegister src0, StinkyRegister src1) {
        StinkyInstruction* inst = create(GFX::v_add_f32);
        inst->addDestReg(vgpr(dst));
        inst->addSrcReg(src0);
        inst->addSrcReg(src1);
        return inst;
    }

    static constexpr int kDsLatency = 56;

    StinkyInstruction* dsLoad(int dst, int addr) {
        StinkyInstruction* inst = createDsReadB128InBlock(bb, arch, dst, addr);
        inst->issueCycles = 1;
        inst->latencyCycles = kDsLatency;
        return inst;
    }

    StinkyInstruction* waitDscnt(int count) {
        StinkyInstruction* inst = create(GFX::s_wait_dscnt);
        inst->addSrcReg(StinkyRegister(count));
        inst->addModifier<SWaitCntData>(SWaitCntData(-1, -1, count, -1, -1));
        return inst;
    }

    std::vector<StinkyInstruction*> block() {
        std::vector<StinkyInstruction*> insts;
        for (IRBase& ir : *bb)
            if (auto* inst = dyn_cast<StinkyInstruction>(&ir)) insts.push_back(inst);
        return insts;
    }

    SimResult simulate(VgprMsbMode msbMode = VgprMsbMode::None) {
        SimConfig config;
        config.matrix = mi;
        config.barrierWaitCycles = hwModelForArch({12, 5, 0}).barrier.signalToWaitLatency;
        config.msbMode = msbMode;
        return CoexecSimulator(config).run(block(), SimState{});
    }
};

TEST_F(CoexecSimulatorTest, MatrixQueueHoldsFourWmmas) {
    for (int i = 0; i < 6; ++i) wmma(8 * i);
    SimResult result = simulate();

    // Four WMMAs issue back to back; then each waits for the oldest to leave the pipe.
    const std::vector<int64_t> expected = {0, 2, 4, 6, 8, 16};
    for (size_t i = 0; i < expected.size(); ++i) EXPECT_EQ(result.records[i].issue, expected[i]);
    EXPECT_EQ(result.exit.matrixFreeAt, 48);
    EXPECT_EQ(result.matrixIdle, 0);
    EXPECT_EQ(result.records[5].exposed, 0) << "a full queue keeps the pipe busy";
}

TEST_F(CoexecSimulatorTest, SaluToValuStallIsExposedOnceTheBacklogDrains) {
    wmma(0);
    saluWrite(5);
    valu(40, sgpr(5), vgpr(41));
    SimResult result = simulate();

    EXPECT_EQ(result.records[1].issue, 2);
    EXPECT_EQ(result.records[2].issue, 2 + mi.saluSgprToValu);
    EXPECT_EQ(result.records[2].stall, mi.saluSgprToValu - 1);
    EXPECT_EQ(result.records[2].exposed, 2 + mi.saluSgprToValu - 8)
        << "the pipe idles from the WMMA's end (8) to the VALU's issue";
}

TEST_F(CoexecSimulatorTest, BacklogHidesTheStall) {
    for (int i = 0; i < 4; ++i) wmma(8 * i);
    saluWrite(5);
    valu(40, sgpr(5), vgpr(41));
    SimResult result = simulate();

    EXPECT_EQ(result.records[5].stall, mi.saluSgprToValu - 1);
    EXPECT_EQ(result.records[5].exposed, 0);
    EXPECT_EQ(result.matrixIdle, 0);
}

TEST_F(CoexecSimulatorTest, MsbSwitchRightAfterDsLoadDelaysTheNextIssue) {
    dsLoad(300, 1);
    valu(2, vgpr(3), vgpr(4));
    SimResult result = simulate(VgprMsbMode::Msb16);

    ASSERT_TRUE(result.records[0].msbSwitch);
    ASSERT_TRUE(result.records[1].msbSwitch);
    EXPECT_EQ(result.records[1].issue - result.records[0].issue, 1 + mi.msbAfterMemOrWait);
}

TEST_F(CoexecSimulatorTest, MsbSwitchAfterValuIsFree) {
    valu(300, vgpr(1), vgpr(2));
    valu(2, vgpr(3), vgpr(4));
    SimResult result = simulate(VgprMsbMode::Msb16);

    ASSERT_TRUE(result.records[1].msbSwitch);
    EXPECT_EQ(result.records[1].issue, result.records[0].issue + 1);
}

TEST_F(CoexecSimulatorTest, MsbSwitchBetweenSaluAndValu) {
    saluWrite(5);
    valu(300, vgpr(1), vgpr(2));
    SimResult result = simulate(VgprMsbMode::Msb16);

    ASSERT_TRUE(result.records[1].msbSwitch);
    EXPECT_EQ(result.records[1].issue, mi.msbAfterSaluBeforeValu);
}

TEST_F(CoexecSimulatorTest, NoMsbModeMeansNoSwitchCost) {
    saluWrite(5);
    valu(300, vgpr(1), vgpr(2));
    SimResult result = simulate();

    EXPECT_FALSE(result.records[1].msbSwitch);
    EXPECT_EQ(result.records[1].issue, 1);
}

// The wait issues at once and holds the instruction after it.
TEST_F(CoexecSimulatorTest, DsWaitHoldsTheNextInstructionUntilTheLoadReturns) {
    dsLoad(10, 1);
    dsLoad(14, 2);
    waitDscnt(1);
    saluWrite(5);
    SimResult result = simulate();

    EXPECT_EQ(result.records[2].issue, 2);
    EXPECT_EQ(result.records[3].issue, kDsLatency) << "dscnt <= 1 needs the first load back";
    EXPECT_EQ(result.records[3].stall, kDsLatency - 3);
}

TEST_F(CoexecSimulatorTest, DsReturnsAreRateLimited) {
    dsLoad(10, 1);
    dsLoad(14, 2);
    waitDscnt(0);
    saluWrite(5);
    SimResult result = simulate();

    EXPECT_EQ(result.records[3].issue, kDsLatency + mi.dsReturnIntervalCycles)
        << "the second load returns one interval after the first";
}

TEST_F(CoexecSimulatorTest, SyncRightAfterAWmmaWaitsForIt) {
    wmma(0);
    waitDscnt(0);
    SimResult result = simulate();

    EXPECT_EQ(result.records[1].issue, mi.syncAfterMatrixCycles);
}

TEST_F(CoexecSimulatorTest, ImplicitSccFlagsCarryTheDependency) {
    StinkyInstruction* cmp = create(GFX::s_cmp_eq_u32);
    ASSERT_TRUE(cmp->is(InstFlag::IF_ImplicitWriteSCC));
    cmp->addSrcReg(sgpr(1));
    cmp->addSrcReg(StinkyRegister(0));
    StinkyInstruction* branch = create(GFX::s_cbranch_scc0);
    ASSERT_TRUE(branch->is(InstFlag::IF_ImplicitReadSCC));
    branch->addModifier<LabelData>(LabelData{"label_skip"});
    SimResult result = simulate();

    EXPECT_EQ(result.records[1].issue, mi.sccToBranch);
}

TEST_F(CoexecSimulatorTest, SccToBranchLatency) {
    StinkyInstruction* cmp = create(GFX::s_cmp_eq_u32);
    cmp->addDestReg(StinkyRegister(RegType::SCC, 0, 1));
    cmp->addSrcReg(sgpr(1));
    cmp->addSrcReg(StinkyRegister(0));
    StinkyInstruction* branch = create(GFX::s_cbranch_scc0);
    branch->addSrcReg(StinkyRegister(RegType::SCC, 0, 1));
    branch->addModifier<LabelData>(LabelData{"label_skip"});
    SimResult result = simulate();

    EXPECT_EQ(result.records[1].issue, mi.sccToBranch);
}

TEST_F(CoexecSimulatorTest, BarrierWaitHoldsTheNextInstructionForTheSignalLatency) {
    create(GFX::s_barrier_signal)->addSrcReg(StinkyRegister(-1));
    create(GFX::s_barrier_wait)->addSrcReg(StinkyRegister(-1));
    saluWrite(5);
    SimResult result = simulate();

    EXPECT_EQ(result.records[1].issue, 1);
    EXPECT_EQ(result.records[2].issue, hwModelForArch({12, 5, 0}).barrier.signalToWaitLatency);
}

TEST_F(CoexecSimulatorTest, HazardRuleDistancesAreCharged) {
    valu(6, vgpr(6), vgpr(8));
    StinkyInstruction* prefetch = create(GFX::global_prefetch_b8);
    prefetch->addSrcReg(vgpr(6, 2));
    ASSERT_TRUE(isGlobalPrefetch(*prefetch));

    const HWModel& hw = hwModelForArch({12, 5, 0});
    SimConfig config;
    config.matrix = mi;
    EXPECT_EQ(CoexecSimulator(config).run(block(), SimState{}).records[1].issue, mi.valuVgprToValu);

    config.hazards = {hw.hazards.rules, static_cast<size_t>(hw.hazards.numRules)};
    EXPECT_EQ(CoexecSimulator(config).run(block(), SimState{}).records[1].issue, 32)
        << "ValuVgprToVmemAddr";
}

// With a context the simulator charges exactly the s_wait_alu WaitAluTracker predicts.
TEST_F(CoexecSimulatorTest, WaitAluPredictionsFollowTheTracker) {
    valu(1, vgpr(2), vgpr(3));
    wmma(8);
    valu(4, vgpr(1), vgpr(9));
    dsLoad(20, 1);
    valu(1, vgpr(5), vgpr(6));

    PassContext ctx;
    ctx.setGemmTileConfig(func->getGemmTileConfig());
    const InsertWaitAluOptions opts = gfx1250InsertWaitAluOptions(true);

    std::vector<bool> expected;
    WaitAluTracker tracker(ctx, opts);
    for (StinkyInstruction* inst : block()) {
        expected.push_back(tracker.query(*inst).any());
        tracker.commit(*inst);
    }
    ASSERT_TRUE(expected.back()) << "overwriting the ds_load's address VGPR needs vm_vsrc";

    SimConfig config;
    config.matrix = mi;
    config.waitAluContext = &ctx;
    config.waitAluOptions = opts;
    SimResult result = CoexecSimulator(config).run(block(), SimState{});
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_EQ(result.records[i].waitAlu, expected[i]) << "instruction " << i;
}

}  // namespace
