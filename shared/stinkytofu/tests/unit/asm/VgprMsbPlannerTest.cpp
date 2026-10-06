// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/ir/asm/VgprMsbEncoding.hpp"
#include "stinkytofu/transforms/asm/InsertVgprMsbPass.hpp"
#include "stinkytofu/transforms/asm/VgprMsbPlanner.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

constexpr int kDstHigh = 0x40;   // dst slot in bank 1
constexpr int kSrc0High = 0x01;  // src0 slot in bank 1

class VgprMsbPlannerTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    std::unique_ptr<Function> func;
    BasicBlock* bb = nullptr;

    void SetUp() override {
        func = std::make_unique<Function>("vgpr_msb_planner_test");
        setFunctionArch(*func, arch);
        bb = func->createBasicBlock("entry");
    }

    StinkyInstruction* salu() {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_add_u32, arch));
        inst->addDestReg(sgpr(0));
        inst->addSrcReg(sgpr(1));
        inst->addSrcReg(sgpr(2));
        return inst;
    }

    StinkyInstruction* waitDscnt() {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_wait_dscnt, arch));
        inst->addSrcReg(StinkyRegister(0));
        return inst;
    }

    StinkyInstruction* label(const char* name) {
        AsmIRBuilder builder(*bb, arch);
        return builder.createLabel(name);
    }

    StinkyInstruction* call() {
        AsmIRBuilder builder(*bb, arch);
        StinkyInstruction* inst = builder.create(getMCIDByUOp(GFX::s_swappc_b64, arch));
        inst->addDestReg(sgpr(30, 2));
        inst->addSrcReg(sgpr(32, 2));
        return inst;
    }

    std::vector<StinkyInstruction*> blockInsts() {
        std::vector<StinkyInstruction*> insts;
        for (IRBase& ir : *bb)
            if (auto* inst = dyn_cast<StinkyInstruction>(&ir)) insts.push_back(inst);
        return insts;
    }

    static std::vector<std::optional<PlannedMsbSwitch>> plan(
        VgprMsbPlanner& planner, const std::vector<StinkyInstruction*>& insts) {
        std::vector<std::optional<PlannedMsbSwitch>> out;
        for (StinkyInstruction* inst : insts) out.push_back(planner.observe(*inst));
        return out;
    }
};

TEST_F(VgprMsbPlannerTest, FirstVgprOpInBlockSwitchesEvenInBankZero) {
    StinkyInstruction* add = createVAddInBlock(bb, arch, 0, 1, 2);
    VgprMsbPlanner planner(VgprMsbMode::Msb16);
    planner.beginBlock();

    std::optional<PlannedMsbSwitch> planned = planner.observe(*add);
    ASSERT_TRUE(planned.has_value());
    EXPECT_EQ(planned->insertBefore, add);
    EXPECT_EQ(planned->value, 0);
    EXPECT_EQ(planned->state, 0);
    EXPECT_FALSE(planned->withNop);
    EXPECT_EQ(planner.state(), 0);
}

TEST_F(VgprMsbPlannerTest, SameStateNeedsNoSwitch) {
    createVAddInBlock(bb, arch, 300, 1, 2);
    createVAddInBlock(bb, arch, 301, 3, 4);
    VgprMsbPlanner planner(VgprMsbMode::Msb16);
    planner.beginBlock();

    auto planned = plan(planner, blockInsts());
    ASSERT_TRUE(planned[0].has_value());
    EXPECT_EQ(planned[0]->state, kDstHigh);
    EXPECT_FALSE(planned[1].has_value());
}

TEST_F(VgprMsbPlannerTest, Msb16PacksPreviousStateInHighByte) {
    createVAddInBlock(bb, arch, 300, 1, 2);
    StinkyInstruction* second = createVAddInBlock(bb, arch, 3, 260, 4);

    VgprMsbPlanner msb16(VgprMsbMode::Msb16);
    msb16.beginBlock();
    auto planned16 = plan(msb16, blockInsts());
    ASSERT_TRUE(planned16[1].has_value());
    EXPECT_EQ(planned16[1]->insertBefore, second);
    EXPECT_EQ(planned16[1]->state, kSrc0High);
    EXPECT_EQ(planned16[1]->value, kSrc0High | (kDstHigh << 8));

    VgprMsbPlanner msb8(VgprMsbMode::Msb8);
    msb8.beginBlock();
    auto planned8 = plan(msb8, blockInsts());
    ASSERT_TRUE(planned8[1].has_value());
    EXPECT_EQ(planned8[1]->value, kSrc0High);
}

TEST_F(VgprMsbPlannerTest, SwitchRightAfterLabelNeedsNop) {
    label("L");
    StinkyInstruction* add = createVAddInBlock(bb, arch, 300, 1, 2);
    VgprMsbPlanner planner(VgprMsbMode::Msb16);
    planner.beginBlock(kDstHigh);

    auto planned = plan(planner, blockInsts());
    ASSERT_TRUE(planned[1].has_value());
    EXPECT_EQ(planned[1]->insertBefore, add);
    EXPECT_TRUE(planned[1]->withNop);
    EXPECT_EQ(planned[1]->value, kDstHigh) << "the state is unknown after a label";
}

TEST_F(VgprMsbPlannerTest, InstructionWithoutVgprAfterLabelDropsTheNop) {
    label("L");
    salu();
    createVAddInBlock(bb, arch, 300, 1, 2);
    VgprMsbPlanner planner(VgprMsbMode::Msb16);
    planner.beginBlock();

    auto planned = plan(planner, blockInsts());
    ASSERT_TRUE(planned[2].has_value());
    EXPECT_FALSE(planned[2]->withNop);
}

TEST_F(VgprMsbPlannerTest, SwitchAnchorsAfterLastPreferredInstruction) {
    createVAddInBlock(bb, arch, 300, 1, 2);
    StinkyInstruction* wait = waitDscnt();
    StinkyInstruction* add = createVAddInBlock(bb, arch, 3, 4, 5);
    VgprMsbPlanner planner(VgprMsbMode::Msb16);
    planner.beginBlock();

    auto planned = plan(planner, blockInsts());
    ASSERT_TRUE(planned[2].has_value());
    EXPECT_EQ(planned[2]->insertBefore, wait) << "switch goes right after the VALU, before "
                                                 "the wait";
    EXPECT_NE(planned[2]->insertBefore, add);
}

TEST_F(VgprMsbPlannerTest, ComputableInstructionResetsTheAnchor) {
    createVAddInBlock(bb, arch, 300, 1, 2);
    createTensorLoadInBlock(bb, arch, 4, 8);
    StinkyInstruction* add = createVAddInBlock(bb, arch, 3, 4, 5);
    VgprMsbPlanner planner(VgprMsbMode::Msb16);
    planner.beginBlock();

    auto planned = plan(planner, blockInsts());
    ASSERT_TRUE(planned[2].has_value());
    EXPECT_EQ(planned[2]->insertBefore, add);
}

TEST_F(VgprMsbPlannerTest, KnownEntryStateSkipsTheFirstSwitch) {
    createVAddInBlock(bb, arch, 300, 1, 2);
    VgprMsbPlanner planner(VgprMsbMode::Msb16);
    planner.beginBlock(kDstHigh);

    auto planned = plan(planner, blockInsts());
    EXPECT_FALSE(planned[0].has_value());
}

TEST_F(VgprMsbPlannerTest, CallResetsTheState) {
    createVAddInBlock(bb, arch, 300, 1, 2);
    call();
    StinkyInstruction* add = createVAddInBlock(bb, arch, 301, 3, 4);
    VgprMsbPlanner planner(VgprMsbMode::Msb16);
    planner.beginBlock();

    auto planned = plan(planner, blockInsts());
    ASSERT_TRUE(planned[2].has_value());
    EXPECT_EQ(planned[2]->insertBefore, add);
    EXPECT_EQ(planned[2]->value, kDstHigh) << "no previous state to pack after a call";
}

TEST_F(VgprMsbPlannerTest, RequireSwitchesOnlyOnChange) {
    StinkyInstruction* add = createVAddInBlock(bb, arch, 300, 1, 2);
    VgprMsbPlanner planner(VgprMsbMode::Msb16);
    planner.beginBlock(kDstHigh);

    EXPECT_FALSE(planner.require(kDstHigh, *add).has_value());
    std::optional<PlannedMsbSwitch> planned = planner.require(kSrc0High, *add);
    ASSERT_TRUE(planned.has_value());
    EXPECT_EQ(planned->value, kSrc0High | (kDstHigh << 8));
    EXPECT_EQ(planner.state(), kSrc0High);
}

// The pass materializes exactly the switches the planner predicts, so a scheduler can
// cost an order with the planner.
TEST_F(VgprMsbPlannerTest, PassEmitsThePlannedSwitches) {
    createVAddInBlock(bb, arch, 300, 1, 2);
    salu();
    createVAddInBlock(bb, arch, 3, 260, 4);
    waitDscnt();
    createVAddInBlock(bb, arch, 5, 6, 7);
    label("L");
    createVAddInBlock(bb, arch, 8, 9, 270);

    struct Expected {
        const StinkyInstruction* before;
        int value;
        bool withNop;
    };
    std::vector<Expected> expected;
    VgprMsbPlanner planner(VgprMsbMode::Msb16);
    planner.beginBlock();
    for (StinkyInstruction* inst : blockInsts())
        if (auto planned = planner.observe(*inst))
            expected.push_back({planned->insertBefore, planned->value, planned->withNop});
    ASSERT_EQ(expected.size(), 4u);

    PassContext ctx;
    ctx.setGemmTileConfig(func->getGemmTileConfig());
    AsmCapsConfig caps;
    caps.vgprMsbMode = VgprMsbMode::Msb16;
    ctx.setAsmCapsConfig(caps);
    AnalysisManager am;
    registerAllAnalyses(am);
    createInsertVgprMsbPass()->run(*func, ctx, am);

    std::vector<StinkyInstruction*> after = blockInsts();
    size_t matched = 0;
    for (size_t i = 0; i < after.size(); ++i) {
        if (after[i]->getUnifiedOpcode() != GFX::s_set_vgpr_msb) continue;
        ASSERT_LT(matched, expected.size());
        const Expected& e = expected[matched++];
        ASSERT_LT(i + 1, after.size());
        EXPECT_EQ(after[i + 1], e.before);
        EXPECT_EQ(after[i]->getSrcReg(0).getLiteralInt(), e.value);
        const bool hasNop = i > 0 && after[i - 1]->getUnifiedOpcode() == GFX::s_nop;
        EXPECT_EQ(hasNop, e.withNop);
    }
    EXPECT_EQ(matched, expected.size());
}

}  // namespace
