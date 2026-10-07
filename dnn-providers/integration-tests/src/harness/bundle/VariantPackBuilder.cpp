// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "harness/bundle/VariantPackBuilder.hpp"

#include <set>
#include <stdexcept>
#include <string>

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/VariantPackUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_integration_tests::bundle::detail
{

VariantPack buildVariantPack(
    TensorMap& inputs,
    OutputTensors& outputs,
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    const std::vector<int64_t>& outputTensorUids,
    bool useDevice)
{
    VariantPack variantPack;
    const std::set<int64_t> outputUids(outputTensorUids.begin(), outputTensorUids.end());

    for(auto& [uid, tensor] : inputs)
    {
        if(outputUids.count(uid) != 0)
        {
            continue;
        }

        const auto attrIt = tensorAttributes.find(uid);
        const bool isRuntimePassByValue
            = attrIt != tensorAttributes.end() && attrIt->second->is_runtime_pass_by_value();
        variantPack[uid] = hipdnn_test_sdk::utilities::selectVariantPackPointer(
            *tensor, useDevice, isRuntimePassByValue);
    }

    for(auto& [uid, tensor] : outputs)
    {
        variantPack[uid] = hipdnn_test_sdk::utilities::selectVariantPackPointer(
            *tensor, useDevice, /*isRuntimePassByValue=*/false);
    }

    return variantPack;
}

OutputTensors allocateSentinelOutputs(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorAttributes,
    const std::vector<int64_t>& outputTensorUids,
    const TensorMap& loadedTensors)
{
    OutputTensors outputs;
    for(const int64_t uid : outputTensorUids)
    {
        const auto& attributes = *tensorAttributes.at(uid);
        const auto raggedOffsetUid = attributes.ragged_offset_tensor_uid();
        if(raggedOffsetUid.has_value())
        {
            const auto offsetIt = loadedTensors.find(raggedOffsetUid.value());
            if(offsetIt == loadedTensors.end())
            {
                throw std::invalid_argument(
                    "ragged output " + std::to_string(uid) + " references offset tensor "
                    + std::to_string(raggedOffsetUid.value()) + ", which is not loaded");
            }
            outputs[uid] = hipdnn_test_sdk::detail::createRaggedTensorFromAttributeAndOffset(
                attributes, offsetIt->second);
        }
        else
        {
            outputs[uid] = hipdnn_test_sdk::detail::createTensorFromAttribute(attributes);
        }
        outputs[uid]->fillWithSentinelValue();
    }
    return outputs;
}

void markOutputsModified(OutputTensors& outputs, bool device)
{
    for(auto& [uid, tensor] : outputs)
    {
        if(device)
        {
            tensor->markDeviceModified();
        }
        else
        {
            tensor->markHostModified();
        }
    }
}

} // namespace hipdnn_integration_tests::bundle::detail
