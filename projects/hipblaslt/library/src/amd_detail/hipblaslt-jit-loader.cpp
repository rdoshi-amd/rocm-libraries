// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-source-bundle.hpp"
#include <Tensile/Tensile.hpp>
#include <stdexcept>

namespace hipblaslt_jit
{
    namespace
    {
        void require(bool condition, const std::string& message)
        {
            if(!condition)
                throw std::runtime_error(message);
        }

        void checkHip(hipError_t status, const char* operation)
        {
            if(status != hipSuccess)
                throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
        }

        std::shared_ptr<TensileLibrary> readEntry(const std::vector<uint8_t>& entry)
        {
            auto library = std::dynamic_pointer_cast<TensileLibrary>(
                TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(entry));
            require(library && library->solutions.size() == 1 && library->solutions.count(0)
                        && library->solutions.at(0),
                    "Expected a non-lazy library containing only local solution 0");
            return library;
        }
    }

    GeneratedSolution readTensileSourceBundle(const std::filesystem::path& bundle)
    {
        namespace artifacts = hipblaslt_jit::source_bundle;
        auto              sources = artifacts::readSourceBundle(bundle);
        GeneratedSolution result;
        result.entry      = std::move(sources.library);
        result.kernelName = readEntry(result.entry)->solutions.at(0)->kernelName;
        for(auto& file : sources.assembly)
            result.units.push_back({BuildUnit::Role::Main,
                                    std::move(file.name),
                                    std::move(file.bytes),
                                    BuildUnit::Kind::Assembly,
                                    {}});
        std::vector<IncludeFile> includes;
        for(auto& header : sources.headers)
            includes.push_back({std::move(header.name), std::move(header.bytes)});
        for(auto& file : sources.helpers)
            result.units.push_back({BuildUnit::Role::Helper,
                                    std::move(file.name),
                                    std::move(file.bytes),
                                    BuildUnit::Kind::Hip,
                                    includes});
        return result;
    }

    std::shared_ptr<TensileBundle> parseTensileBundle(const BuiltSolution&                   built,
                                                      std::shared_ptr<TensileLite::Hardware> hardware)
    {
        require(hardware != nullptr, "Tensile hardware is required");
        auto bundle      = std::make_shared<TensileBundle>();
        bundle->hardware = std::move(hardware);
        bundle->kernel   = built.generated.kernelName;
        bundle->library  = readEntry(built.generated.entry);
        const auto solution = bundle->library->solutions.at(0);
        require(solution->index == 0 && solution->kernelName == bundle->kernel,
                "Library solution identity does not match the generated kernel");
        return bundle;
    }

    void loadTensileBundle(TensileBundle& bundle, const BuiltSolution& built)
    {
        bundle.adapter = std::make_shared<TensileLite::hip::SolutionAdapter>(false, "jit-gemm");
        checkHip(bundle.adapter->loadCodeObjectBytes(built.object.bytes),
                 "Load generated code object");
        for(const auto& helper : built.helpers)
            checkHip(bundle.adapter->loadCodeObjectBytes(helper.bytes),
                     "Load generated code object");
        checkHip(bundle.adapter->initKernel(bundle.kernel), "Resolve generated kernel symbol");
    }
}
