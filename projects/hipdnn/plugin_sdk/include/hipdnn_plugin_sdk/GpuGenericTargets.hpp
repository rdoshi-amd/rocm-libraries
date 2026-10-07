// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file GpuGenericTargets.hpp
 * @brief LLVM generic GPU targets (`gfx11-generic`, ...) as data: which concrete
 * processors each one supports, and how an `arch` entry ranks for a device.
 *
 * Membership comes from the table generated out of data/gpu_generic_targets.json; a
 * name's shape never implies membership, so a generic-shaped name absent from the table
 * matches no device. Nothing here allocates.
 */

#include <cstddef>
#include <optional>
#include <string_view>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/GpuGenericTargetsTable.hpp>

namespace hipdnn_plugin_sdk
{

/// Does @p name have the shape of a generic target (`...-generic`)? Shape only; use
/// findGenericTarget for membership in the table.
inline bool isGenericShapedArchName(std::string_view name)
{
    constexpr std::string_view SUFFIX = "-generic";
    return name.size() >= SUFFIX.size()
           && name.compare(name.size() - SUFFIX.size(), std::string_view::npos, SUFFIX) == 0;
}

/// The table row for the generic named exactly @p name, or nullptr when absent.
inline const generated::GenericTargetRow* findGenericTarget(std::string_view name)
{
    for(const auto& row : generated::GENERIC_TARGET_ROWS)
    {
        if(row.name == name)
        {
            return &row;
        }
    }
    return nullptr;
}

/// Is @p baseDeviceId (features already stripped) a member of the table generic
/// @p generic? False when @p generic is not in the table.
inline bool genericTargetContains(std::string_view generic, std::string_view baseDeviceId)
{
    const auto* row = findGenericTarget(generic);
    if(row == nullptr)
    {
        return false;
    }
    for(std::size_t i = 0; i < row->memberCount; ++i)
    {
        if(row->members[i] == baseDeviceId)
        {
            return true;
        }
    }
    return false;
}

/// How specifically an `arch` entry names a device; a lower value wins.
enum class ArchTier : int
{
    EXPLICIT = 0, ///< The entry is the device's own base id.
    GENERIC = 1, ///< A table generic whose members contain the device.
    UNRESTRICTED = 2, ///< An empty list: any device.
};

/// Rank of one `arch` list entry for the device @p rawDeviceArch as the device reports it
/// (features included): EXPLICIT when the entry is the device's own base id (PREFIX
/// match, so `gfx942` is EXPLICIT for `gfx942:sramecc+:xnack-` and never matches
/// `gfx950`), GENERIC when a table generic containing the device's base id, else nullopt.
/// A generic-shaped entry is never EXPLICIT, so an unknown generic matches no device.
inline std::optional<ArchTier> entryTier(std::string_view entry, std::string_view rawDeviceArch)
{
    if(isGenericShapedArchName(entry))
    {
        return genericTargetContains(entry, stripArchFeatures(rawDeviceArch))
                   ? std::optional<ArchTier>(ArchTier::GENERIC)
                   : std::nullopt;
    }
    return archMatches(rawDeviceArch, entry, ArchMatchMode::PREFIX)
               ? std::optional<ArchTier>(ArchTier::EXPLICIT)
               : std::nullopt;
}

} // namespace hipdnn_plugin_sdk
