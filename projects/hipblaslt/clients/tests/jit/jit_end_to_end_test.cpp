// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "solution_entry.hpp"
#include <hip/hip_fp16.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

// Stages the plain-pair sources of the current device's architecture and replays
// them through Jit, which publishes each solution into the JIT solution library.
// getJitAlgo
// returns its first solution for K=512 and its second for K=256 as library
// indices from 2^30; each runs through hipblasLtMatmul and hipblaslt_ext::Gemm,
// and D is checked against the host. A second lookup returns the same index
// without generating.
namespace jit = hipblaslt_ext::experimental::jit;

namespace
{
    namespace fs = std::filesystem;
    using hipblaslt_jit_test::require;

    // data is the committed assembly tree. A path that already holds plain-pair
    // is a staged replay directory, which the second process receives.
    fs::path replayRoot(const std::string& data, const std::string& tag)
    {
        const auto path = fs::u8path(data);
        if(hipblaslt_jit_test::isDescribedReplayRoot(path))
            return path;
        const auto stage = fs::u8path(HIPBLASLT_JIT_REPLAY_ROOT) / tag;
        hipblaslt_jit_test::stageDescribedSources(path, stage);
        return stage;
    }
    using hipblaslt_jit_test::Device;
    using hipblaslt_jit_test::Fp16Gemm;

#define HIP(expression) hipblaslt_jit_test::checkHip((expression), #expression)
#define BLAS(expression)                                                         \
    do                                                                           \
    {                                                                            \
        const auto status_ = (expression);                                       \
        require(status_ == HIPBLAS_STATUS_SUCCESS,                               \
                std::string(#expression) + ": status " + std::to_string(status_) \
                    + diagnostics.message);                                      \
    } while(false)

    // FP16 NN GEMMs with FP32 compute, which the plain kernel solves.
    constexpr int M = Fp16Gemm::rows, N = Fp16Gemm::cols;

    // Generates the solution for an M x N x K GEMM, requires that its name ends
    // with suffix, and runs it through both GEMM APIs.
    void run(hipblasLtHandle_t     handle,
             hipblasLtMatmulDesc_t desc,
             hipStream_t           stream,
             const jit::Backend&   backend,
             int                   K,
             const std::string&    suffix)
    {
        int device = -1;
        HIP(hipGetDevice(&device));
        jit::Diagnostics diagnostics;
        const auto       label = "K=" + std::to_string(K);
        Fp16Gemm         matrices(K);

        // Generate (replay), build with comgr, then publish a JIT library index.
        jit::Request  request;
        jit::Solution solution;
        float         alpha = 1.25f, beta = 0.5f;
        BLAS(jit::makeGemmRequest(handle,
                                  desc,
                                  &alpha,
                                  matrices.A.pointer,
                                  matrices.layoutA,
                                  matrices.B.pointer,
                                  matrices.layoutB,
                                  &beta,
                                  matrices.C.pointer,
                                  matrices.layoutC,
                                  matrices.D.pointer,
                                  matrices.layoutD,
                                  request,
                                  diagnostics));
        BLAS(jit::getJitAlgo(device, request, backend, 64 << 20, solution, diagnostics));
        require(diagnostics.message != "1 of 1 solutions came from the JIT solution library",
                label + " did not build a solution: " + diagnostics.message);
        hipblasLtMatmulHeuristicResult_t result{};
        BLAS(jit::getGemmAlgo(solution, result, diagnostics));
        const int index = hipblaslt_ext::getIndexFromAlgo(result.algo);
        require(index >= (1 << 30),
                label + " returned " + std::to_string(index) + ", not a JIT library index");
        jit::Solution again;
        BLAS(jit::getJitAlgo(device, request, backend, 64 << 20, again, diagnostics));
        require(diagnostics.message == "1 of 1 solutions came from the JIT solution library",
                label + " generated on the second lookup: " + diagnostics.message);
        hipblasLtMatmulHeuristicResult_t repeat{};
        BLAS(jit::getGemmAlgo(again, repeat, diagnostics));
        require(hipblaslt_ext::getIndexFromAlgo(repeat.algo) == index,
                label + " published a different index on the second lookup");
        Device     workspace(result.workspaceSize);
        const auto name   = hipblaslt_ext::getSolutionNameFromAlgo(handle, result.algo);
        const auto kernel = hipblaslt_ext::getKernelNameFromAlgo(handle, result.algo);
        require(hipblaslt_jit_test::endsWith(name, suffix),
                label + " selected the solution '" + name + "', not the one ending in " + suffix);
        require(kernel.rfind("Cijk_", 0) == 0, "Unexpected kernel name '" + kernel + "'");
        std::cout << "PASS " << label << " published index " << index
                  << " for the solution ending in " << suffix
                  << " and a second lookup reused it\n";

        matrices.matmul(handle,
                        desc,
                        stream,
                        &result.algo,
                        workspace.pointer,
                        result.workspaceSize,
                        alpha,
                        beta,
                        "hipblasLtMatmul, " + label);

        hipblaslt_ext::Gemm gemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F,
                                 HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F);
        float gemmAlpha = -0.75f, gemmBeta = 0.25f;
        BLAS(gemm.setProblem(desc,
                             &gemmAlpha,
                             matrices.A.pointer,
                             matrices.layoutA,
                             matrices.B.pointer,
                             matrices.layoutB,
                             &gemmBeta,
                             matrices.C.pointer,
                             matrices.layoutC,
                             matrices.D.pointer,
                             matrices.layoutD));
        size_t needed = 0;
        BLAS(gemm.isAlgoSupported(result.algo, needed));
        require(needed == result.workspaceSize, "isAlgoSupported returned another workspace size");
        HIP(hipMemset(matrices.D.pointer, 0xff, static_cast<size_t>(M) * N * sizeof(__half)));
        BLAS(gemm.initialize(result.algo, workspace.pointer, true, stream));
        BLAS(gemm.run(stream));
        HIP(hipStreamSynchronize(stream));
        hipblaslt_jit_test::checkFp16("hipblaslt_ext::Gemm, " + label,
                                      M,
                                      N,
                                      K,
                                      matrices.hostA,
                                      matrices.hostB,
                                      matrices.hostC,
                                      matrices.D.pointer,
                                      gemmAlpha,
                                      gemmBeta);
    }

    void test(const std::string& root)
    {
        const char* libraryRoot = std::getenv("HIPBLASLT_JIT_LIBRARY_PATH");
        require(libraryRoot && *libraryRoot,
                "Set HIPBLASLT_JIT_LIBRARY_PATH to a scratch directory");
        std::filesystem::remove_all(std::filesystem::u8path(libraryRoot));
        hipblaslt_jit_test::ReplayFixture fixture(root);
        jit::Diagnostics                  diagnostics;
        auto&                             handle = fixture.handle;
        auto&                             desc   = fixture.desc;
        const int                         device = fixture.device;

        run(handle, desc, fixture.stream, fixture.backend, 512, "_K512_WGM8");
        run(handle, desc, fixture.stream, fixture.backend, 256, "_WGM1");

        // A transposed A is another problem type, which neither solution solves.
        constexpr int            K = 512;
        Device                   A(M * K * 2), B(K * N * 2), C(M * N * 2), D(M * N * 2);
        hipblasLtMatmulDesc_t    tn{};
        hipblasLtMatrixLayout_t  lt{}, lb{}, lc{};
        const hipblasOperation_t transpose = HIPBLAS_OP_T;
        float                    alpha = 1.25f, beta = 0.5f;
        BLAS(hipblasLtMatmulDescCreate(&tn, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        BLAS(hipblasLtMatmulDescSetAttribute(
            tn, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)));
        BLAS(hipblasLtMatrixLayoutCreate(&lt, HIP_R_16F, K, M, K));
        BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, K, N, K));
        BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, M, N, M));
        jit::Request  transposed;
        jit::Solution unsolved;
        BLAS(jit::makeGemmRequest(handle, tn, &alpha, A.pointer, lt, B.pointer, lb, &beta,
                                  C.pointer, lc, D.pointer, lc, transposed, diagnostics));
        require(jit::getJitAlgo(device, transposed, fixture.backend, 64 << 20, unsolved, diagnostics)
                    == HIPBLAS_STATUS_NOT_SUPPORTED,
                "The replay backend accepted a problem its bundle does not solve");
        std::cout << "PASS a problem the bundle does not solve is not supported\n";
        hipblasLtMatrixLayoutDestroy(lt);
        hipblasLtMatrixLayoutDestroy(lb);
        hipblasLtMatrixLayoutDestroy(lc);
        hipblasLtMatmulDescDestroy(tn);
    }

    std::string text(const std::vector<int32_t>& indices)
    {
        std::string result;
        for(const auto index : indices)
            result += (result.empty() ? "" : ",") + std::to_string(index);
        return result;
    }

    std::vector<int32_t> libraryAlgos(hipblasLtHandle_t     handle,
                                      hipblasLtMatmulDesc_t desc,
                                      const jit::Backend&   backend,
                                      int                   K,
                                      size_t                count,
                                      jit::Diagnostics&     diagnostics)
    {
        int device = -1;
        HIP(hipGetDevice(&device));
        Fp16Gemm     matrices(K);
        jit::Request request;
        float        alpha = 1.25f, beta = 0.5f;
        BLAS(jit::makeGemmRequest(handle,
                                  desc,
                                  &alpha,
                                  matrices.A.pointer,
                                  matrices.layoutA,
                                  matrices.B.pointer,
                                  matrices.layoutB,
                                  &beta,
                                  matrices.C.pointer,
                                  matrices.layoutC,
                                  matrices.D.pointer,
                                  matrices.layoutD,
                                  request,
                                  diagnostics));
        std::vector<int32_t> indices;
        BLAS(jit::getLibraryAlgos(
            device, request, backend, count, 64 << 20, indices, diagnostics));
        for(const auto index : indices)
            require(index >= (1 << 30),
                    "Index " + std::to_string(index) + " is outside the reserved JIT range");
        return indices;
    }

    void runIndex(hipblasLtHandle_t     handle,
                  hipblasLtMatmulDesc_t desc,
                  hipStream_t           stream,
                  int                   K,
                  int                   index,
                  const std::string&    suffix)
    {
        jit::Diagnostics diagnostics;
        const auto       label = "index " + std::to_string(index) + ", K=" + std::to_string(K);
        std::vector<int> wanted{index};
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        BLAS(hipblaslt_ext::getAlgosFromIndex(handle, wanted, results));
        require(results.size() == 1 && hipblaslt_ext::getIndexFromAlgo(results[0].algo) == index,
                label + " did not resolve");
        const auto name = hipblaslt_ext::getSolutionNameFromAlgo(handle, results[0].algo);
        require(hipblaslt_jit_test::endsWith(name, suffix),
                label + " is the solution '" + name + "', not the one ending in " + suffix);
        Fp16Gemm matrices(K);
        float    alpha = 1.25f, beta = 0.5f;
        size_t   bytes = 0;
        BLAS(hipblaslt_ext::matmulIsAlgoSupported(handle,
                                                  desc,
                                                  &alpha,
                                                  matrices.layoutA,
                                                  matrices.layoutB,
                                                  &beta,
                                                  matrices.layoutC,
                                                  matrices.layoutD,
                                                  results[0].algo,
                                                  bytes));
        Device workspace(std::max<size_t>(bytes, 1));
        matrices.matmul(handle,
                        desc,
                        stream,
                        &results[0].algo,
                        workspace.pointer,
                        bytes,
                        alpha,
                        beta,
                        "hipblasLtMatmul, " + label);
    }

    // Publishes plain-pair's first solution for K=512 and its second for K=256,
    // runs each index, and checks that later queries only look them up. With
    // indices from another process, runs those before any query instead.
    void library(const std::string& root, const std::vector<int32_t>& published)
    {
        const char* libraryRoot = std::getenv("HIPBLASLT_JIT_LIBRARY_PATH");
        require(libraryRoot && *libraryRoot,
                "Set HIPBLASLT_JIT_LIBRARY_PATH to a scratch directory");
        jit::Diagnostics diagnostics;
        std::vector<int32_t> indices = published;
        {
            hipblaslt_jit_test::ReplayFixture fixture(root);
            if(published.empty())
            {
                std::filesystem::remove_all(std::filesystem::u8path(libraryRoot));
                const auto first
                    = libraryAlgos(fixture.handle, fixture.desc, fixture.backend, 512, 1, diagnostics);
                require(first.size() == 1,
                        "K=512 published " + text(first) + ": " + diagnostics.message);
                const auto second
                    = libraryAlgos(fixture.handle, fixture.desc, fixture.backend, 256, 1, diagnostics);
                require(second.size() == 1 && second[0] != first[0],
                        "K=256 published " + text(second) + ": " + diagnostics.message);
                indices = {first[0], second[0]};
                std::cout << "PASS getLibraryAlgos published " << text(indices) << '\n';
            }
            runIndex(fixture.handle, fixture.desc, fixture.stream, 512, indices[0], "_K512_WGM8");
            runIndex(fixture.handle, fixture.desc, fixture.stream, 256, indices[1], "_WGM1");
            std::cout << "PASS each index ran through getAlgosFromIndex and hipblasLtMatmul\n";

            require(libraryAlgos(fixture.handle, fixture.desc, fixture.backend, 512, 1, diagnostics)
                            == std::vector<int32_t>{indices[0]}
                        && diagnostics.message
                               == "1 of 1 solutions came from the JIT solution library",
                    "K=512 did not come from the library: " + diagnostics.message);
            require(libraryAlgos(fixture.handle, fixture.desc, fixture.backend, 256, 1, diagnostics)
                            == std::vector<int32_t>{indices[1]}
                        && diagnostics.message
                               == "1 of 1 solutions came from the JIT solution library",
                    "K=256 did not come from the library: " + diagnostics.message);
            std::cout << "PASS later queries found the published solutions without generating\n";
        }
        if(!published.empty())
            return;

        std::cout.flush();
        std::vector<std::string> arguments{
            "hipblaslt-jit-end-to-end-test", root, "--library-reader"};
        for(const auto index : indices)
            arguments.push_back(std::to_string(index));
        const auto child = fork();
        require(child >= 0, "fork failed");
        if(child == 0)
        {
            std::vector<char*> argv;
            for(auto& argument : arguments)
                argv.push_back(argument.data());
            argv.push_back(nullptr);
            execv("/proc/self/exe", argv.data());
            _exit(127);
        }
        int status = 0;
        require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                "The second process failed (wait status " + std::to_string(status) + ")");
        std::cout << "PASS a second process ran the indices before any query and found them\n";
    }
}

int main(int argc, char** argv)
{
    const std::string mode = argc > 2 ? argv[2] : "";
    if(!(argc == 2 || (argc == 3 && mode == "--library")
         || (argc == 5 && mode == "--library-reader")))
    {
        std::cerr << "Usage: " << argv[0] << " DATA [--library]\n";
        return 2;
    }
    try
    {
        if(argc == 2)
            test(replayRoot(argv[1], "end-to-end").u8string());
        else if(argc == 3)
            library(replayRoot(argv[1], "end-to-end-library").u8string(), {});
        else
            library(replayRoot(argv[1], "end-to-end-library").u8string(),
                    {std::stoi(argv[3]), std::stoi(argv[4])});
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
