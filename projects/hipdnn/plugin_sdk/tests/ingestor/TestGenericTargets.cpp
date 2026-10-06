// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/GpuGenericTargets.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>

/**
 * @file TestGenericTargets.cpp
 * @brief The generic GPU target table and the tier algebra over it, as C++ evaluates
 *        them. The table is the generated constexpr form of data/gpu_generic_targets.json
 *        and the golden vectors are shared with the Python implementation.
 */
namespace
{

using namespace hipdnn_plugin_sdk;
using namespace hipdnn_plugin_sdk::ingestor;

using Arch = std::vector<std::string>;

nlohmann::json readJson(const char* path)
{
    std::ifstream stream(path);
    EXPECT_TRUE(stream.is_open()) << "cannot open " << path;
    return nlohmann::json::parse(stream);
}

Arch toArch(const nlohmann::json& list)
{
    return list.get<Arch>();
}

const char* tierName(const std::optional<ArchTier>& tier)
{
    if(!tier)
    {
        return "none";
    }
    if(*tier == ArchTier::EXPLICIT)
    {
        return "explicit";
    }
    if(*tier == ArchTier::GENERIC)
    {
        return "generic";
    }
    return "unrestricted";
}

TEST(TestGenericTargets, GeneratedTableMatchesTheJsonMemberForMember)
{
    const auto json = readJson(HIPDNN_GPU_GENERIC_TARGETS_JSON_PATH);
    const auto& generics = json.at("generics");

    ASSERT_EQ(generics.size(), generated::GENERIC_TARGET_ROW_COUNT);
    // Rows are matched by name: object key order is not part of the contract, member
    // array order is.
    for(const auto& [name, members] : generics.items())
    {
        const auto* row = findGenericTarget(name);
        ASSERT_NE(row, nullptr) << name;
        ASSERT_EQ(row->memberCount, members.size()) << name;
        for(std::size_t i = 0; i < members.size(); ++i)
        {
            EXPECT_EQ(row->members[i], members[i].get<std::string>()) << name << "[" << i << "]";
        }
    }
}

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
    EXPECT_EQ(archEntryTier("gfx1151", "gfx1151"), ArchTier::EXPLICIT);
    EXPECT_EQ(archEntryTier("gfx11-generic", "gfx1151"), ArchTier::GENERIC);
    EXPECT_EQ(archEntryTier("gfx1150", "gfx1151"), std::nullopt);
    EXPECT_EQ(archEntryTier("gfx11-generic", "gfx1154"), std::nullopt);
    EXPECT_LT(static_cast<int>(ArchTier::EXPLICIT), static_cast<int>(ArchTier::GENERIC));
    EXPECT_LT(static_cast<int>(ArchTier::GENERIC), static_cast<int>(ArchTier::UNRESTRICTED));
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

TEST(TestGenericTargets, MatchesGoldenTierVectors)
{
    const auto vectors = readJson(HIPDNN_ARCH_TIER_VECTORS_JSON_PATH);

    ASSERT_FALSE(vectors.at("tier").empty());
    for(const auto& v : vectors.at("tier"))
    {
        EXPECT_STREQ(tierName(archTier(toArch(v.at("arch")), v.at("device").get<std::string>())),
                     v.at("expect").get<std::string>().c_str())
            << v.dump();
    }
    ASSERT_FALSE(vectors.at("overlap").empty());
    for(const auto& v : vectors.at("overlap"))
    {
        EXPECT_EQ(archOverlaps(toArch(v.at("a")), toArch(v.at("b"))), v.at("expect").get<bool>())
            << v.dump();
    }
    ASSERT_FALSE(vectors.at("covers").empty());
    for(const auto& v : vectors.at("covers"))
    {
        EXPECT_EQ(archCovers(toArch(v.at("outer")), toArch(v.at("inner"))),
                  v.at("expect").get<bool>())
            << v.dump();
    }
    ASSERT_FALSE(vectors.at("compete").empty());
    for(const auto& v : vectors.at("compete"))
    {
        EXPECT_EQ(archesCompete(toArch(v.at("a")), toArch(v.at("b"))), v.at("expect").get<bool>())
            << v.dump();
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
