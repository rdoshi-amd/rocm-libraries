// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/input-init/InputFillRecipes.hpp"

namespace hipdnn_integration_tests
{

using InputTensorMap
    = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

struct FillResult
{
    bool filled = false;
    std::string reason;

    static FillResult ok()
    {
        return {true, {}};
    }
    static FillResult unsupported(std::string why)
    {
        return {false, std::move(why)};
    }
};

/// Generates large FREE inputs straight into device memory with rocRAND, instead of by
/// the host's serial RNG, which costs seconds per tensor at the largest shapes. A
/// tensor filled this way is device-resident afterwards, so the first non-const host
/// access migrates it; a const access cannot.
///
/// Owns the rocRAND generator, created on the first fill that needs one, so its
/// lifetime is the owner's rather than the process's. Not thread-safe. Without rocRAND
/// it never accepts a fill, and every tensor is filled on the host.
class DeviceInputFiller
{
public:
    DeviceInputFiller();
    ~DeviceInputFiller();

    DeviceInputFiller(const DeviceInputFiller&) = delete;
    DeviceInputFiller& operator=(const DeviceInputFiller&) = delete;
    DeviceInputFiller(DeviceInputFiller&&) = delete;
    DeviceInputFiller& operator=(DeviceInputFiller&&) = delete;

    /// Starts a FREE fill of `tensor` on the device and returns true, or returns false
    /// and leaves the tensor untouched when it is too small to be worth it or of a
    /// type rocRAND does not fill. The fill is left in flight: nothing may read the
    /// tensor until waitForFills() returns.
    bool tryFill(hipdnn_data_sdk::utilities::ITensor& tensor,
                 const FillRecipe& recipe,
                 unsigned int seed);

    /// Blocks until every fill started by tryFill() has finished. Throws if one failed.
    void waitForFills();

    /// Whether this build can generate on the device at all, which is whether rocRAND
    /// was available when FillInputs.cpp was compiled.
    static bool isSupported();

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

/// Fills every owned input of `graph`. With a `device` filler, large FREE inputs are
/// generated on the device; with nullptr, which is what the unit tests rely on since
/// they have no device, everything is filled on the host. All device fills have
/// finished by the time this returns or throws.
FillResult fillInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                      InputTensorMap& inputs,
                      const std::vector<int64_t>& ownedUids,
                      InputFillRecipes& recipes,
                      DeviceInputFiller* device);

} // namespace hipdnn_integration_tests
