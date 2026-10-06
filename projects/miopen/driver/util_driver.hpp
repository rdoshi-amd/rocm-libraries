/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_MIOPEN_UTIL_DRIVER_HPP
#define GUARD_MIOPEN_UTIL_DRIVER_HPP

#include <miopen/config.h>
#include <miopen/miopen.h>

#include <cstring>
#include <iostream>
#include <vector>

#define STATUS_SUCCESS 0
typedef int status_t;
typedef uint32_t context_t;
#define DEFINE_CONTEXT(name) context_t name = 0
typedef hipStream_t stream;

/// Restores the default number formatting of std::cout.
/// Tensile (PropertyMatching.hpp, in both hipBLASLt and rocBLAS) sets std::fixed and
/// setprecision(2) on std::cout and never restores them, which rounds every number printed
/// afterwards to 2 decimals.
inline void ResetCoutFormat()
{
    std::cout.flags(std::ios_base::dec | std::ios_base::skipws);
    std::cout.precision(6);
}

#endif // GUARD_MIOPEN_UTIL_DRIVER_HPP
