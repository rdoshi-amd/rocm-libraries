// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "TestHelpers.hpp"
#include "stinkytofu/analysis/AnalysisRegistration.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/transforms/asm/MatrixCoexecRepairPass.hpp"

using namespace stinkytofu;
using namespace stinkytofu::test;

namespace {

class MatrixCoexecRepairPassTest : public ::testing::Test {
   protected:
    GfxArchID arch = GfxArchID::Gfx1250;
    std::unique_ptr<Function> func;
    BasicBlock* loop = nullptr;
    AnalysisManager am;

    void SetUp() override {
        func = std::make_unique<Function>("matrix_coexec_repair_test");
        setFunctionArch(*func, arch);
        registerAllAnalyses(am);
    }

    StinkyInstruction* create(BasicBlock* bb, GFX op) {
        AsmIRBuilder builder(*bb, arch);
        return builder.create(getMCIDByUOp(op, arch));
    }

    void wmma(int dst, int src) {
        StinkyInstruction* inst = create(loop, GFX::v_wmma_f32_16x16x32_bf16);
        inst->addDestReg(vgpr(dst, 8));
        inst->addSrcReg(vgpr(src, 8));
        inst->addSrcReg(vgpr(src + 8, 8));
        inst->addSrcReg(vgpr(dst, 8));
        inst->latencyCycles = 8;
    }

    void stalledValu(int sdst, int vdst) {
        StinkyInstruction* mov = create(loop, GFX::s_mov_b32);
        mov->addDestReg(sgpr(sdst));
        mov->addSrcReg(StinkyRegister(1));
        StinkyInstruction* add = create(loop, GFX::v_add_f32);
        add->addDestReg(vgpr(vdst));
        add->addSrcReg(sgpr(sdst));
        add->addSrcReg(vgpr(vdst + 1));
    }

    /// The issue-bound loop of matrix_coexec_repair_sink_test.stir: three SALU -> VALU
    /// stalls behind one WMMA, four more WMMAs after them.
    void buildIssueBoundLoop() {
        BasicBlock* entry = func->createBasicBlock("entry");
        loop = func->createBasicBlock("loop");
        BasicBlock* exit = func->createBasicBlock("exit");
        entry->addSuccessor(loop);
        loop->addSuccessor(loop);
        loop->addSuccessor(exit);

        wmma(0, 100);
        stalledValu(5, 40);
        stalledValu(6, 42);
        stalledValu(8, 44);
        for (int i = 1; i < 5; ++i) wmma(8 * i, 100 + 16 * i);

        StinkyInstruction* cmp = create(loop, GFX::s_cmp_lt_u32);
        cmp->addDestReg(StinkyRegister(RegType::SCC, 0, 1));
        cmp->addSrcReg(sgpr(7));
        cmp->addSrcReg(StinkyRegister(16));
        StinkyInstruction* branch = create(loop, GFX::s_cbranch_scc1);
        branch->addSrcReg(StinkyRegister(RegType::SCC, 0, 1));
        branch->addModifier<LabelData>(LabelData{"loop"});
        create(exit, GFX::s_endpgm);
    }

    std::vector<StinkyInstruction*> loopOrder() const {
        std::vector<StinkyInstruction*> order;
        for (IRBase& ir : *loop)
            if (auto* inst = dyn_cast<StinkyInstruction>(&ir)) order.push_back(inst);
        return order;
    }

    PreservedAnalyses runPass(MatrixCoexecRepairOptions options) {
        PassContext ctx;
        ctx.setGemmTileConfig(func->getGemmTileConfig());
        return createMatrixCoexecRepairPass(std::move(options))->run(*func, ctx, am);
    }

    static std::string tempReport(const char* name) {
        return (std::filesystem::temp_directory_path() / name).string();
    }

    static std::string readFile(const std::string& path) {
        std::ifstream in(path);
        std::stringstream text;
        text << in.rdbuf();
        return text.str();
    }

    static std::pair<long, long> iterationCycles(const std::string& report) {
        std::smatch match;
        const std::regex re("\"iterationCycles\":\\{\"before\":(\\d+),\"after\":(\\d+)\\}");
        if (!std::regex_search(report, match, re)) return {-1, -1};
        return {std::stol(match[1]), std::stol(match[2])};
    }
};

TEST_F(MatrixCoexecRepairPassTest, AnalyzeOnlyReportsAndKeepsTheOrder) {
    buildIssueBoundLoop();
    const std::vector<StinkyInstruction*> before = loopOrder();
    const std::string path = tempReport("matrix_coexec_repair_analyze.json");
    std::filesystem::remove(path);

    PreservedAnalyses preserved = runPass({.analyzeOnly = true, .reportPath = path});
    EXPECT_TRUE(preserved.areAllPreserved());
    EXPECT_EQ(loopOrder(), before);

    const std::string report = readFile(path);
    EXPECT_NE(report.find("\"mode\":\"analyze\""), std::string::npos);
    EXPECT_NE(report.find("\"header\":\"loop\""), std::string::npos);
    EXPECT_NE(report.find("\"exposed\":"), std::string::npos);
    auto [cyclesBefore, cyclesAfter] = iterationCycles(report);
    EXPECT_GT(cyclesBefore, 0);
    EXPECT_EQ(cyclesAfter, cyclesBefore);
}

TEST_F(MatrixCoexecRepairPassTest, RepairShortensThePredictedIteration) {
    buildIssueBoundLoop();
    const std::string path = tempReport("matrix_coexec_repair_repair.json");
    std::filesystem::remove(path);

    runPass({.reportPath = path});

    const std::string report = readFile(path);
    EXPECT_NE(report.find("\"mode\":\"repair\""), std::string::npos);
    auto [cyclesBefore, cyclesAfter] = iterationCycles(report);
    EXPECT_GT(cyclesBefore, 0);
    EXPECT_LT(cyclesAfter, cyclesBefore);
}

// Only fillers move, only later; everything else keeps its relative order.
TEST_F(MatrixCoexecRepairPassTest, SkeletonOrderIsPreservedAndFillersOnlySink) {
    buildIssueBoundLoop();
    const std::vector<StinkyInstruction*> before = loopOrder();

    runPass({});
    const std::vector<StinkyInstruction*> after = loopOrder();
    ASSERT_EQ(after.size(), before.size());
    ASSERT_NE(after, before) << "the issue-bound loop should be repaired";

    auto skeleton = [](const std::vector<StinkyInstruction*>& order) {
        std::vector<StinkyInstruction*> kept;
        for (StinkyInstruction* inst : order)
            if (inst->getUnifiedOpcode() != GFX::v_add_f32) kept.push_back(inst);
        return kept;
    };
    EXPECT_EQ(skeleton(after), skeleton(before));

    for (size_t i = 0; i < before.size(); ++i) {
        if (before[i]->getUnifiedOpcode() != GFX::v_add_f32) continue;
        const size_t now =
            static_cast<size_t>(std::find(after.begin(), after.end(), before[i]) - after.begin());
        EXPECT_GE(now, i) << "a filler only moves later";
    }
}

}  // namespace
