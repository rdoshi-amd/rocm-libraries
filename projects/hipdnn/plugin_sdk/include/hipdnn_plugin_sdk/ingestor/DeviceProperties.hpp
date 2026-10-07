// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/GpuGenericTargets.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// The device facts matching and dispatch may read, as the `$device.*` namespace.
/// Deliberately not `hipDeviceProp_t`: named fields keep this a closed, reviewable
/// vocabulary instead of every consumer taking a HIP dependency.
struct DeviceProperties
{
    /// Raw GFX target, suffix intact (e.g. `"gfx942:sramecc+:xnack-"`). Compare with
    /// `hipdnn_plugin_sdk::archMatches`, not `==`.
    std::string gcnArchName;
    int warpSize = 0; ///< Threads per wavefront; 0 if unresolved.
    int multiProcessorCount = 0; ///< Compute units; 0 if unresolved.
};

/// How @p arch (a KDP's supported-target list) ranks for @p deviceArch: the best tier of
/// any entry (see entryTier), or nullopt when no entry admits the device. An empty list is
/// UNRESTRICTED.
inline std::optional<ArchTier> archTier(const std::vector<std::string>& arch,
                                        std::string_view deviceArch)
{
    if(arch.empty())
    {
        return ArchTier::UNRESTRICTED;
    }
    std::optional<ArchTier> best;
    for(const auto& entry : arch)
    {
        const auto tier = entryTier(entry, deviceArch);
        if(tier && (!best || *tier < *best))
        {
            best = tier;
        }
    }
    return best;
}

/// Does @p arch (a KDP's supported-target list; empty admits everything) admit
/// @p deviceArch, by any tier?
inline bool archSupports(const std::vector<std::string>& arch, std::string_view deviceArch)
{
    return archTier(arch, deviceArch).has_value();
}

namespace detail
{

inline bool listAdmits(const std::vector<std::string>& list, std::string_view device)
{
    return std::any_of(list.begin(), list.end(), [device](const std::string& entry) {
        return entryTier(entry, device).has_value();
    });
}

/// Calls @p visit with every device id in @p entry's expansion (a generic's members, an
/// explicit entry itself, nothing for an unknown generic) until it returns true.
template <typename Visit>
inline bool anyExpandedMember(std::string_view entry, Visit&& visit)
{
    if(!isGenericShapedArchName(entry))
    {
        return visit(entry);
    }
    const auto* row = findGenericTarget(entry);
    if(row == nullptr)
    {
        return false;
    }
    for(std::size_t i = 0; i < row->memberCount; ++i)
    {
        if(visit(row->members[i]))
        {
            return true;
        }
    }
    return false;
}

/// Does some device in the expansion of @p from get admitted by @p into?
inline bool anyExpandedMemberAdmittedBy(const std::vector<std::string>& from,
                                        const std::vector<std::string>& into)
{
    return std::any_of(from.begin(), from.end(), [&into](const std::string& entry) {
        return anyExpandedMember(
            entry, [&into](std::string_view device) { return listAdmits(into, device); });
    });
}

} // namespace detail

/// Can one device satisfy both @p a and @p b? Compared over expanded member sets: a
/// generic stands for its table members. Empty means "every arch", so it overlaps
/// everything; an unknown generic expands to nothing and overlaps nothing.
inline bool archOverlaps(const std::vector<std::string>& a, const std::vector<std::string>& b)
{
    if(a.empty() || b.empty())
    {
        return true;
    }
    return detail::anyExpandedMemberAdmittedBy(a, b) || detail::anyExpandedMemberAdmittedBy(b, a);
}

/// Is every device @p inner admits also admitted by @p outer? The asymmetric counterpart
/// to archOverlaps, for asking whether a kernel stays within the pack that binds it.
/// Compared over expanded member sets. Empty @p outer admits every device, so it covers
/// anything; empty @p inner declares no restriction of its own and is covered by anything;
/// an unknown generic in @p inner expands to nothing and is covered vacuously.
inline bool archCovers(const std::vector<std::string>& outer, const std::vector<std::string>& inner)
{
    if(outer.empty())
    {
        return true;
    }
    return std::all_of(inner.begin(), inner.end(), [&outer](const std::string& entry) {
        return !detail::anyExpandedMember(entry, [&outer](std::string_view device) {
            return !detail::listAdmits(outer, device);
        });
    });
}

/// Do @p a and @p b tie? True when some candidate device is matched by both at the same
/// tier, where the candidates are every explicit id in either list and every member of
/// every table generic in either list. Two empty lists compete (both unrestricted on
/// every device). Lists whose best tiers always differ per device do not: the better
/// tier shadows the other, so they may coexist.
inline bool archesCompete(const std::vector<std::string>& a, const std::vector<std::string>& b)
{
    if(a.empty() && b.empty())
    {
        return true;
    }
    const auto tiesOn = [&a, &b](std::string_view device) {
        const auto tierA = archTier(a, device);
        return tierA && tierA == archTier(b, device);
    };
    const auto anyTie = [&tiesOn](const std::vector<std::string>& list) {
        return std::any_of(list.begin(), list.end(), [&tiesOn](const std::string& entry) {
            return detail::anyExpandedMember(entry, tiesOn);
        });
    };
    return anyTie(a) || anyTie(b);
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
