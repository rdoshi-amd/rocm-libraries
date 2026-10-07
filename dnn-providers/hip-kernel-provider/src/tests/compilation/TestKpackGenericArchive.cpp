// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/GpuGenericTargets.hpp>

#include "PackedKernelSource.hpp"
#include "TestDescriptorRoot.hpp"
#include "compilation/KpackArchive.hpp"
#include "compilation/KpackModuleCache.hpp"

namespace hip_kernel_provider::compilation
{
namespace
{

using hip_kernel_provider::testing::PackedKernelSource;
using hip_kernel_provider::testing::readPackedKernelSource;
using hip_kernel_provider::testing::unitKpackRoot;

/// The generic named by test_descriptors/shared/conv_fwd/conv_fwd_generic.kdp.json.
constexpr const char* GENERIC = "gfx11-generic";
constexpr const char* GENERIC_KDP = "conv_fwd_generic.kdp.json";

std::vector<std::string> archivesOf(const std::filesystem::path& kpackFile)
{
    KpackArchive archive;
    KpackError error;
    std::vector<std::string> arches;
    EXPECT_TRUE(archive.open(kpackFile, error)) << kpackFile << " (" << error.codeName << ")";
    EXPECT_TRUE(archive.architectures(arches, error)) << kpackFile << " (" << error.codeName << ")";
    return arches;
}

/// Packed arch shards present in this tree: every directory of the packed set holding a
/// kpack/ child. Enumerated from the filesystem so the test needs no device.
std::vector<std::filesystem::path> packedShards()
{
    std::vector<std::filesystem::path> shards;
    std::error_code ec;
    for(const auto& entry : std::filesystem::directory_iterator(unitKpackRoot(), ec))
    {
        if(entry.is_directory(ec) && std::filesystem::is_directory(entry.path() / "kpack", ec))
        {
            shards.push_back(entry.path());
        }
    }
    std::sort(shards.begin(), shards.end());
    return shards;
}

} // namespace

TEST(TestPackedGenericArchive, ShardsCarryTheGenericArchiveExactlyWhenTheirArchIsAMember)
{
    const auto shards = packedShards();
    ASSERT_FALSE(shards.empty()) << "no packed arch shard under " << unitKpackRoot();

    for(const auto& shard : shards)
    {
        const std::string arch = shard.filename().string();
        SCOPED_TRACE(arch);

        std::vector<std::filesystem::path> kpackFiles;
        for(const auto& entry : std::filesystem::directory_iterator(shard / "kpack"))
        {
            kpackFiles.push_back(entry.path());
        }
        ASSERT_FALSE(kpackFiles.empty());

        const bool member = hipdnn_plugin_sdk::genericTargetContains(GENERIC, arch);

        if(!member)
        {
            for(const auto& file : kpackFiles)
            {
                const auto arches = archivesOf(file);
                EXPECT_EQ(std::count(arches.begin(), arches.end(), GENERIC), 0) << file;
            }
            EXPECT_FALSE(std::filesystem::exists(shard / "conv_fwd" / GENERIC_KDP));
            continue;
        }

        PackedKernelSource packed;
        ASSERT_NO_FATAL_FAILURE(readPackedKernelSource(shard, GENERIC_KDP, packed));

        // The generic descriptor points at an archive keyed by the generic alone.
        const std::vector<std::string> keys = archivesOf(packed.archive);
        ASSERT_EQ(keys, std::vector<std::string>{GENERIC});

        // KpackModuleCache serves this shard's device from that archive's generic key.
        const std::string* selected = KpackModuleCache::selectArch(keys, arch);
        ASSERT_NE(selected, nullptr);
        EXPECT_EQ(*selected, GENERIC);

        // The device's own archive keeps winning when both are visible.
        std::vector<std::string> both;
        for(const auto& file : kpackFiles)
        {
            for(const auto& key : archivesOf(file))
            {
                both.push_back(key);
            }
        }
        ASSERT_NE(std::count(both.begin(), both.end(), arch), 0);
        const std::string* preferred = KpackModuleCache::selectArch(both, arch);
        ASSERT_NE(preferred, nullptr);
        EXPECT_EQ(*preferred, arch);

        // The generic key resolves the descriptor's own toc_key to a code object.
        KpackArchive archive;
        KpackError error;
        ASSERT_TRUE(archive.open(packed.archive, error)) << error.codeName;
        KpackCodeObject object;
        ASSERT_TRUE(archive.codeObject(packed.tocKey, *selected, object, error)) << error.codeName;
        EXPECT_FALSE(object.empty());
    }
}

} // namespace hip_kernel_provider::compilation

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
