// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// --- Common kernel argument structs ---
// Shared between device kernels and host launch code for ABI compatibility.

struct ScaleUniformArgs
{
    const void* src;
    void* dst;
    long long count;
    double minValue;
    double maxValue;
};

struct DataTypeUtilArgs
{
    const void* input;
    void* output;
    long long count;
};
