// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <Tensile/Contractions.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>

namespace hipblaslt_jit
{
    using TensileLibrary = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;

    // A one-solution TensileLite library entry and, once loaded, its code objects.
    struct TensileBundle
    {
        std::shared_ptr<TensileLite::Hardware>             hardware;
        std::shared_ptr<TensileLibrary>                    library;
        std::shared_ptr<TensileLite::hip::SolutionAdapter> adapter; // null until loaded
        std::string                                        kernel;
    };

    // Reads a TensileLite source bundle by directory convention: its entry, every
    // sources/*.s as a main unit, and sources/Kernels.cpp as the helper unit.
    GeneratedSolution readTensileSourceBundle(const std::filesystem::path& bundle);

    // Reads the entry of a built solution for hardware and loads no code. Throws
    // unless the entry holds only local solution 0, named after the built kernel.
    std::shared_ptr<TensileBundle> parseTensileBundle(const BuiltSolution&                   built,
                                                      std::shared_ptr<TensileLite::Hardware> hardware);

    // Loads the built code objects into a new adapter and resolves the main
    // kernel. Throws on failure.
    void loadTensileBundle(TensileBundle& bundle, const BuiltSolution& built);

    // A Tensile bundle as the GEMM API runs it.
    struct TensileGemmBundle final : KernelBundle
    {
        using Diagnostics = hipblaslt_ext::experimental::jit::Diagnostics;

        std::shared_ptr<const TensileBundle> tensile;

        std::string_view operationKind() const noexcept override;
        std::string      name() const override
        {
            return tensile->library->solutions.at(0)->solutionName;
        }
        std::string kernelNames() const override
        {
            return tensile->kernel;
        }
        hipblasStatus_t
            support(const OperationRequest&, size_t, size_t&, Diagnostics&) const override;
    };

    // Loads CustomKernel entries from any backend into TensileGemmBundles.
    std::shared_ptr<const SolutionLoader> makeTensileLoader();
}
