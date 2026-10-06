// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

namespace TensileLite
{
    class ContractionProblemGemm;
}

namespace hipblaslt_ext::experimental::jit::detail
{
    struct GemmRequest;
}

namespace hipblaslt_jit
{
    // The TensileLite problem hipBLASLt solves for the request.
    TensileLite::ContractionProblemGemm
        lowerForJit(const hipblaslt_ext::experimental::jit::detail::GemmRequest& request);
}
