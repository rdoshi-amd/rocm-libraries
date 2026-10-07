// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/GpuGenericTargets.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>

/**
 * @file TestGenericTargets.cpp
 * @brief The generic GPU target table and the tier algebra over it, as C++ evaluates
 *        them, on the generated constexpr table.
 */
namespace
{

using namespace hipdnn_plugin_sdk;
using namespace hipdnn_plugin_sdk::ingestor;

using Arch = std::vector<std::string>;

TEST(TestGenericTargets, FindsAGenericOnlyByItsExactName)
{
    const auto* row = findGenericTarget("gfx11-generic");
    ASSERT_NE(row, nullptr);
    EXPECT_EQ(row->name, "gfx11-generic");

    EXPECT_EQ(findGenericTarget("gfx11"), nullptr);
    EXPECT_EQ(findGenericTarget("gfx11-generic "), nullptr);
    EXPECT_EQ(findGenericTarget("GFX11-GENERIC"), nullptr);
    EXPECT_EQ(findGenericTarget("gfx9-4-generic"), nullptr);
    EXPECT_EQ(findGenericTarget(""), nullptr);

    EXPECT_TRUE(isGenericShapedArchName("gfx9-4-generic"));
    EXPECT_FALSE(isGenericShapedArchName("gfx942"));
    EXPECT_FALSE(isGenericShapedArchName("generic"));
}

TEST(TestGenericTargets, ContainsReportsMembership)
{
    EXPECT_TRUE(genericTargetContains("gfx11-generic", "gfx1100"));
    EXPECT_TRUE(genericTargetContains("gfx11-generic", "gfx1153"));
    EXPECT_FALSE(genericTargetContains("gfx11-generic", "gfx1154"));
    EXPECT_FALSE(genericTargetContains("gfx11-generic", "gfx1200"));
    EXPECT_TRUE(genericTargetContains("gfx12-generic", "gfx1201"));
    EXPECT_FALSE(genericTargetContains("gfx12-generic", "gfx1250"));
    EXPECT_FALSE(genericTargetContains("gfx9-4-generic", "gfx942"));
}

TEST(TestGenericTargets, EntryTierRanksExplicitAboveGeneric)
{
    EXPECT_EQ(entryTier("gfx1151", "gfx1151"), ArchTier::EXPLICIT);
    EXPECT_EQ(entryTier("gfx1151", "gfx1151:sramecc+:xnack-"), ArchTier::EXPLICIT);
    EXPECT_EQ(entryTier("gfx11-generic", "gfx1151"), ArchTier::GENERIC);
    EXPECT_EQ(entryTier("gfx11-generic", "gfx1151:sramecc+:xnack-"), ArchTier::GENERIC);
    EXPECT_EQ(entryTier("gfx1150", "gfx1151"), std::nullopt);
    EXPECT_EQ(entryTier("gfx11-generic", "gfx1154"), std::nullopt);
    EXPECT_EQ(entryTier("gfx9-4-generic", "gfx9-4-generic"), std::nullopt);
    EXPECT_LT(ArchTier::EXPLICIT, ArchTier::GENERIC);
    EXPECT_LT(ArchTier::GENERIC, ArchTier::UNRESTRICTED);
}

TEST(TestGenericTargets, ListTierIsTheBestOfItsEntries)
{
    const Arch both{"gfx11-generic", "gfx1151"};
    EXPECT_EQ(archTier(both, "gfx1151"), ArchTier::EXPLICIT);
    EXPECT_EQ(archTier(both, "gfx1151:sramecc+:xnack-"), ArchTier::EXPLICIT);
    EXPECT_EQ(archTier(both, "gfx1100"), ArchTier::GENERIC);
    EXPECT_EQ(archTier(both, "gfx1154"), std::nullopt);
    EXPECT_EQ(archTier({"gfx11-generic"}, "gfx1151:sramecc+"), ArchTier::GENERIC);
}

TEST(TestGenericTargets, EmptyListIsUnrestricted)
{
    EXPECT_EQ(archTier({}, "gfx1151"), ArchTier::UNRESTRICTED);
    EXPECT_EQ(archTier({}, ""), ArchTier::UNRESTRICTED);
    EXPECT_TRUE(archSupports({}, "gfx1151"));
}

TEST(TestGenericTargets, OverlapIsOverExpandedMemberSets)
{
    EXPECT_TRUE(archOverlaps({"gfx11-generic"}, {"gfx1151"}));
    EXPECT_TRUE(archOverlaps({"gfx1151"}, {"gfx11-generic"}));
    EXPECT_TRUE(archOverlaps({"gfx11-generic"}, {"gfx11-generic"}));
    EXPECT_TRUE(archOverlaps({}, {"gfx11-generic"}));
    EXPECT_FALSE(archOverlaps({"gfx11-generic"}, {"gfx12-generic"}));
    EXPECT_FALSE(archOverlaps({"gfx11-generic"}, {"gfx1154"}));
    EXPECT_FALSE(archOverlaps({"gfx1250"}, {"gfx12-generic"}));
}

TEST(TestGenericTargets, CoversIsOverExpandedMemberSets)
{
    EXPECT_TRUE(archCovers({"gfx11-generic"}, {"gfx1151"}));
    EXPECT_TRUE(archCovers({"gfx11-generic"}, {"gfx1100", "gfx1153"}));
    EXPECT_FALSE(archCovers({"gfx11-generic"}, {"gfx1154"}));
    EXPECT_FALSE(archCovers({"gfx1151"}, {"gfx11-generic"}));
    EXPECT_TRUE(archCovers({"gfx11-generic"}, {"gfx11-generic"}));
    EXPECT_FALSE(archCovers({"gfx11-generic"}, {"gfx12-generic"}));
    EXPECT_TRUE(archCovers({}, {"gfx11-generic"}));
    EXPECT_TRUE(archCovers({"gfx11-generic"}, {}));
}

TEST(TestGenericTargets, CompeteIsSameTierOverlap)
{
    EXPECT_TRUE(archesCompete({}, {}));
    EXPECT_TRUE(archesCompete({"gfx942"}, {"gfx942"}));
    EXPECT_TRUE(archesCompete({"gfx11-generic"}, {"gfx11-generic"}));
    EXPECT_TRUE(archesCompete({"gfx1151", "gfx11-generic"}, {"gfx11-generic"}));
    EXPECT_FALSE(archesCompete({}, {"gfx942"}));
    EXPECT_FALSE(archesCompete({"gfx11-generic"}, {"gfx1151"}));
    EXPECT_FALSE(archesCompete({"gfx11-generic"}, {}));
    EXPECT_FALSE(archesCompete({"gfx11-generic"}, {"gfx12-generic"}));
}

TEST(TestGenericTargets, UnknownGenericExpandsToNothing)
{
    const Arch unknown{"gfx9-4-generic"};
    EXPECT_FALSE(archSupports(unknown, "gfx942"));
    EXPECT_FALSE(archSupports(unknown, "gfx9-4-generic"));
    EXPECT_FALSE(archOverlaps(unknown, unknown));
    EXPECT_TRUE(archOverlaps(unknown, {}));
    EXPECT_TRUE(archCovers({"gfx11-generic"}, unknown));
    EXPECT_FALSE(archCovers(unknown, {"gfx1151"}));
    EXPECT_FALSE(archesCompete(unknown, unknown));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
