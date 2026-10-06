// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-loader.hpp"

#include <Tensile/hip/HipHardware.hpp>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// Builds the committed bundle with comgr for the current device, loads it
// through the Tensile loader and checks that its library selects the solution
// for the GEMM it was generated for and for no other problem type. Launches
// nothing.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;

    void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }

    // FP16 A, B, C and D with FP32 alpha, beta and accumulation, as the bundle's entry expects.
    TensileLite::ContractionProblemGemm gemm(bool transA)
    {
        constexpr size_t M = 256, N = 128, K = 512;
        const auto       half = rocisa::DataType::Half;
        const size_t     lda  = transA ? K : M;
        auto problem = TensileLite::ContractionProblemGemm::GEMM_Strides(
            transA, false, half, half, half, half, M, N, K, 1, lda, lda * (transA ? M : K), K,
            K * N, M, M * N, M, M * N, 0.5);
        problem.setComputeInputTypeA(half);
        problem.setComputeInputTypeB(half);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        problem.setHighPrecisionAccumulate(true);
        problem.setStridedBatched(true);
        problem.setUseDeviceUserArguments(false);
        problem.setAlphaRestriction(TensileLite::toScalarValueEnum(1.25));
        problem.setBetaRestriction(TensileLite::toScalarValueEnum(0.5));
        problem.setCEqualsD(false);
        return problem;
    }

    std::string whyNot(const hj::TensileBundle& bundle, const TensileLite::ContractionProblemGemm& problem)
    {
        std::ostringstream reason;
        const auto&        solution = *bundle.library->solutions.at(0);
        solution.hardwarePredicate->debugEval(*bundle.hardware, reason);
        solution.problemPredicate->debugEval(problem, reason);
        return reason.str();
    }
}

int main(int argc, char** argv)
try
{
    require(argc == 3, "Usage: hipblaslt-jit-loader-test BUNDLE SCRATCH");
    const auto bundlePath = fs::u8path(argv[1]);
    const auto scratch    = fs::u8path(argv[2]);
    fs::remove_all(scratch);
    fs::create_directories(scratch);

    int             device = 0;
    hipDeviceProp_t properties{};
    require(hipGetDevice(&device) == hipSuccess
                && hipGetDeviceProperties(&properties, device) == hipSuccess,
            "Cannot query the current HIP device");

    const auto        solution = hj::readTensileSourceBundle(bundlePath);
    hj::BuiltSolution built;
    const auto        status = hj::makeComgrBuilder()->build(
        solution, {properties.gcnArchName, hj::jitCodeObjectVersion, scratch}, built);
    require(status.ok(), std::string("The build for ") + properties.gcnArchName
                             + " failed: " + status.message);

    auto bundle = hj::parseTensileBundle(built, TensileLite::hip::GetDevice(properties, device));
    hj::loadTensileBundle(*bundle, built);
    std::cout << "PASS loaded " << bundle->kernel.substr(0, 40) << "... on "
              << properties.gcnArchName << '\n';

    const auto expected = bundle->library->solutions.at(0);
    const auto plain    = gemm(false);
    require(bundle->library->findBestSolution(plain, *bundle->hardware) == expected,
            "The library does not select its solution for the GEMM: " + whyNot(*bundle, plain));
    std::cout << "PASS the library selects its solution for the GEMM it was generated for\n";

    const auto transposed = gemm(true);
    require(bundle->library->findBestSolution(transposed, *bundle->hardware) == nullptr,
            "The library selects its solution for a transposed A");
    std::cout << "PASS the library selects nothing for a transposed A\n";
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
