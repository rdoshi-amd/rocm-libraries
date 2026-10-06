// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-source-bundle.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

// Builds a committed bundle with comgr for the target its manifest names, plus
// a HIP helper unit, and checks that the code object defines both kernels.
// Needs no GPU.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;

    const char* const helperName = "hipblaslt_jit_builder_test_scale";
    const char* const helper     = R"(
#include <hip/hip_runtime.h>
extern "C" __global__ void hipblaslt_jit_builder_test_scale(float* x, float a, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        x[i] *= a;
}
)";

    void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }

    // The string value of key in the manifest's object named owner.
    std::string field(const std::string& manifest, const std::string& owner, const std::string& key)
    {
        std::smatch      match;
        const std::regex pattern("\"" + owner + "\": *\\{[^}]*\"" + key + "\": *\"([^\"]*)\"");
        require(std::regex_search(manifest, match, pattern), "The manifest has no " + owner + "." + key);
        return match[1].str();
    }
}

int main(int argc, char** argv)
try
{
    require(argc == 3, "Usage: hipblaslt-jit-builder-test BUNDLE SCRATCH");
    const auto bundle  = fs::u8path(argv[1]);
    const auto scratch = fs::u8path(argv[2]);
    fs::remove_all(scratch);
    fs::create_directories(scratch);

    const auto bytes    = hj::source_bundle::readArtifact(bundle / "manifest.json");
    const auto manifest = std::string(bytes.begin(), bytes.end());
    auto       sources  = hj::source_bundle::readSourceBundle(bundle);
    require(sources.assembly.size() == 1 && sources.helpers.empty(),
            "Expected one main kernel and no helpers in " + bundle.u8string());

    hj::GeneratedSolution solution;
    solution.entry      = std::move(sources.library);
    solution.kernelName = field(manifest, "main_kernel", "name");
    solution.units.push_back({hj::BuildUnit::Role::Main,
                              std::move(sources.assembly.front().name),
                              std::move(sources.assembly.front().bytes),
                              hj::BuildUnit::Kind::Assembly,
                              {}});
    solution.units.push_back({hj::BuildUnit::Role::Helper,
                              "scale.cpp",
                              std::vector<uint8_t>(helper, helper + std::char_traits<char>::length(helper)),
                              hj::BuildUnit::Kind::Hip,
                              {}});

    const hj::BuildRequest request{field(manifest, "architecture", "compiler_target"),
                                   hj::jitCodeObjectVersion,
                                   scratch};
    hj::BuiltSolution      built;
    const auto             status = hj::makeComgrBuilder()->build(solution, request, built);
    require(status.ok(), "The build for " + request.targetId + " failed: " + status.message);

    const auto metadata
        = hj::code_object::readMetadata(built.object.bytes.data(), built.object.bytes.size());
    require(metadata.ok(), "Cannot read the built code object: " + metadata.log);
    std::vector<std::string> kernels;
    for(const auto& kernel : metadata.metadata.kernels)
        kernels.push_back(kernel.name);
    for(const auto& name : {solution.kernelName, std::string(helperName)})
        require(std::find(kernels.begin(), kernels.end(), name) != kernels.end(),
                "The code object does not define " + name);
    require(metadata.metadata.codeObjectVersion == hj::jitCodeObjectVersion,
            "The code object is not version " + std::to_string(hj::jitCodeObjectVersion));
    std::cout << "PASS comgr built " << solution.kernelName.substr(0, 40) << "... and a HIP helper for "
              << request.targetId << '\n';
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
