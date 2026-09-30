// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedTokenBoundaries.hpp>

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{

// Ragged (RFC-0014: packed [B,H,S,D] + ragged_offset) forward SDPA GPU reference, parallel to
// GpuFpReferenceSdpa. Logical rank-4 dims with BSHD-layout strides (seq stride = strides[2] = H*D):
//   q = [B, H,  Sq,  D ]   k = [B, Hk, Skv, D ]
//   v = [B, Hv, Skv, Dv]   o = [B, H,  Sq,  Dv]
// The physical buffer is packed (no per-batch padding); batch b begins at element ragged_offset[b].
// raggedOffsetQ/K/V/O are the cumulative ELEMENT offsets (RFC-0014), rank-4 [B+1,1,1,1] INT32, of
// the q/k/v/o tensors. They are read back to the host and validated: o must share Q's token
// boundaries and v must share K's (per-batch sequence lengths), while element offsets may differ
// with the per-token width (e.g. D != Dv). The kernel then addresses o/v through Q/K token
// boundaries. Per-batch sequence lengths are (ragged_offset[b+1] - ragged_offset[b]) / seqStride.
// Numerics (fp32 softmax, provider P storage) and the SdpaSoftmaxProbabilityMode enum are shared
// with the dense reference. No additive bias/alibi/dropout (gated off on the ASM v3 path);
// per-batch causal/sliding-window and GQA/MQA are supported. The optional LSE is logical
// [B, H, Sq_max, 1]: ragged (packed) when raggedOffsetLse is given, else dense and addressed
// through its strides.
class GpuFpReferenceSdpaRagged
{
public:
    // Takes non-const references because deviceData() may trigger host→device sync.
    template <class QDataType,
              class KDataType = QDataType,
              class VDataType = QDataType,
              class ODataType = QDataType,
              class ComputeDataType = float>
    static void
        fpropRagged(hipdnn_data_sdk::utilities::TensorBase<QDataType>& q,
                    hipdnn_data_sdk::utilities::TensorBase<KDataType>& k,
                    hipdnn_data_sdk::utilities::TensorBase<VDataType>& v,
                    hipdnn_data_sdk::utilities::TensorBase<ODataType>& o,
                    hipdnn_data_sdk::utilities::TensorBase<int32_t>& raggedOffsetQ,
                    hipdnn_data_sdk::utilities::TensorBase<int32_t>& raggedOffsetK,
                    hipdnn_data_sdk::utilities::TensorBase<int32_t>& raggedOffsetV,
                    hipdnn_data_sdk::utilities::TensorBase<int32_t>& raggedOffsetO,
                    std::optional<float> attnScaleValue = std::nullopt,
                    int64_t leftBound = -1,
                    int64_t rightBound = -1,
                    bool topLeftAlignment = true,
                    hipdnn_data_sdk::utilities::TensorBase<float>* lse = nullptr,
                    hipdnn_data_sdk::utilities::TensorBase<int32_t>* raggedOffsetLse = nullptr,
                    SdpaSoftmaxProbabilityMode probabilityMode = SdpaSoftmaxProbabilityMode::FLOAT,
                    hipdnn_data_sdk::utilities::TensorBase<float>* descaleQ = nullptr,
                    hipdnn_data_sdk::utilities::TensorBase<float>* descaleK = nullptr,
                    hipdnn_data_sdk::utilities::TensorBase<float>* descaleV = nullptr)
    {
        validateInput(q.dims(),
                      k.dims(),
                      v.dims(),
                      o.dims(),
                      {raggedOffsetQ.dims(),
                       raggedOffsetK.dims(),
                       raggedOffsetV.dims(),
                       raggedOffsetO.dims()});

        const auto batch = q.dims()[0];
        const auto numHeads = q.dims()[1];
        const auto headDim = q.dims()[3];
        const auto numHeadsK = k.dims()[1];
        const auto numHeadsV = v.dims()[1];
        const auto headDimV = v.dims()[3];

        // BSHD-layout sequence strides (elements per token): H*D for Q, Hk*D for K.
        const auto seqStrideQ = q.strides()[2];
        const auto seqStrideKv = k.strides()[2];

        // Token boundaries of each tensor, read back from the (possibly device-only) offset aux.
        // Tensors sharing a packing must agree batch by batch; total_q is Q's last boundary.
        const std::string who = "GpuFpReferenceSdpaRagged";
        const auto tokenBoundaries = [&](hipdnn_data_sdk::utilities::TensorBase<int32_t>& offsets,
                                         int64_t seqStride,
                                         const char* name) {
            if(offsets.strides()[0] != 1)
            {
                throw std::invalid_argument(who + ": " + name
                                            + " ragged_offset must be contiguous");
            }
            return hipdnn_test_sdk::detail::raggedTokenBoundaries(
                readRaggedOffsets(offsets.memory().deviceData(), batch + 1), seqStride, who, name);
        };
        const auto qTokens = tokenBoundaries(raggedOffsetQ, seqStrideQ, "Q");
        const auto kTokens = tokenBoundaries(raggedOffsetK, seqStrideKv, "K");
        hipdnn_test_sdk::detail::requireMatchingTokenBoundaries(
            qTokens, "Q", tokenBoundaries(raggedOffsetO, o.strides()[2], "O"), "O", who);
        hipdnn_test_sdk::detail::requireMatchingTokenBoundaries(
            kTokens, "K", tokenBoundaries(raggedOffsetV, v.strides()[2], "V"), "V", who);
        const int64_t totalQ = qTokens.back();

        const float scale = attnScaleValue.has_value()
                                ? attnScaleValue.value()
                                : (1.0F / std::sqrt(static_cast<float>(headDim)));

        auto defines
            = detail::buildSdpaDefines<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
                probabilityMode);

        void* lsePtr = nullptr;
        void* raggedOffsetLsePtr = nullptr;
        std::vector<int64_t> lseStrides;
        if(lse != nullptr)
        {
            // LSE is one value per query token; logical [B, H, Sq, 1], dense or ragged.
            if(lse->dims().size() != 4 || lse->dims()[0] != batch || lse->dims()[1] != numHeads
               || lse->dims()[3] != 1)
            {
                throw std::invalid_argument(
                    "GpuFpReferenceSdpaRagged: lse must be rank-4 [B, H, Sq, 1]");
            }
            lsePtr = lse->memory().deviceData();
            lseStrides = lse->strides();
            if(raggedOffsetLse != nullptr)
            {
                if(raggedOffsetLse->dims() != raggedOffsetQ.dims())
                {
                    throw std::invalid_argument(
                        "GpuFpReferenceSdpaRagged: raggedOffsetLse must be rank-4 [B+1, 1, 1, 1]");
                }
                raggedOffsetLsePtr = raggedOffsetLse->memory().deviceData();
                hipdnn_test_sdk::detail::requireMatchingTokenBoundaries(
                    qTokens,
                    "Q",
                    tokenBoundaries(*raggedOffsetLse, lseStrides[2], "LSE"),
                    "LSE",
                    who);
            }
        }
        else if(raggedOffsetLse != nullptr)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: raggedOffsetLse given without an lse tensor");
        }

        // Optional fp8 Q/K/V descale: scalar [1] or per-KV-head [B, H_kv, 1, 1]. The Q descale is
        // per K head (query head h reads the descale of the K head it attends to).
        const DescaleBinding dq = bindDescale(descaleQ, batch, numHeadsK, "Q");
        const DescaleBinding dk = bindDescale(descaleK, batch, numHeadsK, "K");
        const DescaleBinding dv = bindDescale(descaleV, batch, numHeadsV, "V");

        launchSdpaRaggedFwd(q.memory().deviceData(),
                            k.memory().deviceData(),
                            v.memory().deviceData(),
                            o.memory().deviceData(),
                            lsePtr,
                            raggedOffsetLsePtr,
                            raggedOffsetQ.memory().deviceData(),
                            raggedOffsetK.memory().deviceData(),
                            seqStrideQ,
                            seqStrideKv,
                            dq.ptr,
                            dq.batchStride,
                            dq.headStride,
                            dk.ptr,
                            dk.batchStride,
                            dk.headStride,
                            dv.ptr,
                            dv.batchStride,
                            dv.headStride,
                            q.strides(),
                            k.strides(),
                            v.strides(),
                            o.strides(),
                            lseStrides,
                            batch,
                            totalQ,
                            numHeads,
                            numHeadsK,
                            numHeadsV,
                            headDim,
                            headDimV,
                            scale,
                            leftBound,
                            rightBound,
                            topLeftAlignment,
                            defines);

        o.memory().markDeviceModified();
        if(lse != nullptr)
        {
            lse->memory().markDeviceModified();
        }
    }

private:
    // Resolved fp8 descale binding: device pointer + (batch, head) index strides. A null pointer
    // (no descale) or a scalar descale uses zero strides.
    struct DescaleBinding
    {
        const void* ptr = nullptr;
        long long batchStride = 0;
        long long headStride = 0;
    };

    // Validate a descale tensor's shape (scalar [1] / (1,1,1,1), or per-KV-head [B, heads, 1, 1])
    // and resolve its device pointer + strides. `heads` is H_k for Q and K, H_v for V.
    static DescaleBinding bindDescale(hipdnn_data_sdk::utilities::TensorBase<float>* descale,
                                      int64_t batch,
                                      int64_t heads,
                                      const char* name)
    {
        DescaleBinding binding;
        if(descale == nullptr)
        {
            return binding;
        }
        if(descale->elementCount() == 1)
        {
            binding.ptr = descale->memory().deviceData(); // scalar; zero strides
            return binding;
        }
        const auto& dims = descale->dims();
        if(dims.size() == 4 && dims[0] == batch && dims[1] == heads && dims[2] == 1 && dims[3] == 1)
        {
            binding.ptr = descale->memory().deviceData();
            binding.batchStride = static_cast<long long>(descale->strides()[0]);
            binding.headStride = static_cast<long long>(descale->strides()[1]);
            return binding;
        }
        throw std::invalid_argument(std::string("GpuFpReferenceSdpaRagged: ") + name
                                    + " descale must be scalar [1] or per-KV-head [B, H_kv, 1, 1]");
    }

    static void validateInput(const std::vector<int64_t>& qDims,
                              const std::vector<int64_t>& kDims,
                              const std::vector<int64_t>& vDims,
                              const std::vector<int64_t>& oDims,
                              const std::vector<std::vector<int64_t>>& raggedOffsetDims)
    {
        if(qDims.size() != 4 || kDims.size() != 4 || vDims.size() != 4 || oDims.size() != 4)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: q/k/v/o must all be rank-4 [B, H, S, D] tensors");
        }
        // ragged_offset aux is rank-4 [B+1, 1, 1, 1] INT32 (RFC-0014 structural contract).
        for(const auto& d : raggedOffsetDims)
        {
            if(d.size() != 4 || d[0] != qDims[0] + 1 || d[1] != 1 || d[2] != 1 || d[3] != 1)
            {
                throw std::invalid_argument(
                    "GpuFpReferenceSdpaRagged: raggedOffsetQ/K/V/O must be rank-4 [B+1, 1, 1, 1]");
            }
        }

        const auto batch = qDims[0];
        const auto numHeads = qDims[1];
        const auto headDim = qDims[3];
        const auto numHeadsK = kDims[1];
        const auto numHeadsV = vDims[1];
        const auto headDimV = vDims[3];

        if(batch <= 0 || numHeads <= 0 || headDim <= 0 || numHeadsK <= 0 || numHeadsV <= 0
           || headDimV <= 0)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: all dimensions must be positive");
        }
        if(kDims[0] != batch || vDims[0] != batch || oDims[0] != batch)
        {
            throw std::invalid_argument("GpuFpReferenceSdpaRagged: batch dimension mismatch");
        }
        if(vDims[2] != kDims[2])
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: K and V sequence extents (S_max) must match");
        }
        if(kDims[3] != headDim)
        {
            throw std::invalid_argument("GpuFpReferenceSdpaRagged: Q head_dim != K head_dim");
        }
        if(numHeads % numHeadsK != 0 || numHeads % numHeadsV != 0)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: numHeads must be divisible by numHeadsK and numHeadsV");
        }
        if(oDims[1] != numHeads || oDims[2] != qDims[2] || oDims[3] != headDimV)
        {
            throw std::invalid_argument(
                "GpuFpReferenceSdpaRagged: output shape must be [B, H, Sq, Dv]");
        }
    }

    // Reads a contiguous ragged_offset table (count INT32 element offsets) from device memory.
    // Defined in GpuFpReferenceSdpaRagged.cpp.
    static std::vector<int64_t> readRaggedOffsets(const void* raggedOffsetPtr, int64_t count);

    // --- Kernel launcher (defined in GpuFpReferenceSdpaRagged.cpp) ---

    static void launchSdpaRaggedFwd(const void* qPtr,
                                    const void* kPtr,
                                    const void* vPtr,
                                    void* oPtr,
                                    void* lsePtr,
                                    const void* raggedOffsetLsePtr,
                                    const void* raggedOffsetQPtr,
                                    const void* raggedOffsetKvPtr,
                                    int64_t seqStrideQ,
                                    int64_t seqStrideKv,
                                    const void* descaleQPtr,
                                    int64_t descaleQBatchStride,
                                    int64_t descaleQHeadStride,
                                    const void* descaleKPtr,
                                    int64_t descaleKBatchStride,
                                    int64_t descaleKHeadStride,
                                    const void* descaleVPtr,
                                    int64_t descaleVBatchStride,
                                    int64_t descaleVHeadStride,
                                    const std::vector<int64_t>& qTensorStrides,
                                    const std::vector<int64_t>& kTensorStrides,
                                    const std::vector<int64_t>& vTensorStrides,
                                    const std::vector<int64_t>& oTensorStrides,
                                    const std::vector<int64_t>& lseTensorStrides,
                                    int64_t batch,
                                    int64_t totalQ,
                                    int64_t numHeads,
                                    int64_t numHeadsK,
                                    int64_t numHeadsV,
                                    int64_t headDim,
                                    int64_t headDimV,
                                    float scale,
                                    int64_t leftBound,
                                    int64_t rightBound,
                                    bool topLeftAlignment,
                                    const std::vector<std::string>& defines);
};

} // namespace hipdnn_gpu_ref
