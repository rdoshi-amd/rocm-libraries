// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "solution_entry.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

// Heuristic queries under HIPBLASLT_JIT. The mode is a process environment
// variable, so each CTest entry runs one mode. The test stages the device's
// plain-pair sources and lists that directory in HIPBLASLT_JIT_TEST_REPLAY
// before the first query.
namespace
{
    namespace fs = std::filesystem;
    using hipblaslt_jit_test::Device;
    using hipblaslt_jit_test::endsWith;

#define HIP(expression) hipblaslt_jit_test::checkHip((expression), #expression)

    constexpr int M = hipblaslt_jit_test::Fp16Gemm::rows;
    constexpr int N = hipblaslt_jit_test::Fp16Gemm::cols;
    constexpr uint64_t workspaceLimit = 64ull << 20;

    struct Layouts
    {
        hipblasLtMatrixLayout_t a{}, b{}, c{}, d{};
        Layouts(int rowsA, int colsA, int ldA, int rowsB, int colsB, int ldB)
        {
            {
                INFO(("layout A"));
                REQUIRE((hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, rowsA, colsA, ldA)
                    == HIPBLAS_STATUS_SUCCESS));
            }
            {
                INFO(("layout B"));
                REQUIRE((hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, rowsB, colsB, ldB)
                    == HIPBLAS_STATUS_SUCCESS));
            }
            {
                INFO(("layout C"));
                REQUIRE((
                    hipblasLtMatrixLayoutCreate(&c, HIP_R_16F, M, N, M) == HIPBLAS_STATUS_SUCCESS));
            }
            {
                INFO(("layout D"));
                REQUIRE((
                    hipblasLtMatrixLayoutCreate(&d, HIP_R_16F, M, N, M) == HIPBLAS_STATUS_SUCCESS));
            }
        }
        ~Layouts()
        {
            hipblasLtMatrixLayoutDestroy(a);
            hipblasLtMatrixLayoutDestroy(b);
            hipblasLtMatrixLayoutDestroy(c);
            hipblasLtMatrixLayoutDestroy(d);
        }
        Layouts(const Layouts&)            = delete;
        Layouts& operator=(const Layouts&) = delete;
    };

    struct Listed
    {
        hipblasStatus_t                              status{};
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
    };

    bool isJit(hipblasLtMatmulAlgo_t algo)
    {
        // Heuristic results are JIT solution library indices, from 2^30.
        return hipblaslt_ext::getIndexFromAlgo(algo) >= (1 << 30);
    }

    Listed queryC(hipblasLtHandle_t            handle,
                  hipblasLtMatmulDesc_t        desc,
                  hipblasLtMatmulPreference_t  pref,
                  const Layouts&               layouts,
                  int                          requested)
    {
        Listed listed;
        listed.results.resize(requested);
        int count = 0;
        listed.status = hipblasLtMatmulAlgoGetHeuristic(handle,
                                                        desc,
                                                        layouts.a,
                                                        layouts.b,
                                                        layouts.c,
                                                        layouts.d,
                                                        pref,
                                                        requested,
                                                        listed.results.data(),
                                                        &count);
        {
            INFO(("C heuristic returned a count outside the request"));
            REQUIRE((count >= 0 && count <= requested));
        }
        listed.results.resize(count);
        return listed;
    }

    Listed queryCpp(hipblasLtHandle_t           handle,
                    hipblasLtMatmulDesc_t       desc,
                    hipblasOperation_t          opA,
                    hipblasOperation_t          opB,
                    const Layouts&              layouts,
                    void*                       A,
                    void*                       B,
                    void*                       C,
                    void*                       D,
                    int                         requested)
    {
        hipblaslt_ext::Gemm gemm(handle,
                                 opA,
                                 opB,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIPBLAS_COMPUTE_32F);
        float               alpha = 1.25f, beta = 0.5f;
        {
            INFO(("Gemm::setProblem"));
            REQUIRE((gemm.setProblem(desc, &alpha, A, layouts.a, B, layouts.b, &beta, C, layouts.c, D,
                            layouts.d)
                == HIPBLAS_STATUS_SUCCESS));
        }
        hipblaslt_ext::GemmPreference pref;
        pref.setMaxWorkspaceBytes(workspaceLimit);
        Listed listed;
        listed.status = gemm.algoGetHeuristic(requested, pref, listed.results);
        {
            INFO(("C++ heuristic returned more results than requested"));
            REQUIRE((static_cast<int>(listed.results.size()) <= requested));
        }
        return listed;
    }

    void requireNoJit(const std::string& label, const Listed& listed)
    {
        for(const auto& result : listed.results)
        {
            INFO((label + " returned a JIT algorithm"));
            REQUIRE((!isJit(result.algo)));
        }
        if(listed.results.empty())
        {
            INFO((label + " with no results returned status " + std::to_string(listed.status)));
            REQUIRE((listed.status == HIPBLAS_STATUS_INVALID_VALUE));
        }
        else
        {
            INFO((label + " returned status " + std::to_string(listed.status)));
            REQUIRE((listed.status == HIPBLAS_STATUS_SUCCESS));
        }
    }

    void requireOnlyJit(const std::string& label, const Listed& listed, const std::string& suffix,
                        hipblasLtHandle_t handle)
    {
        {
            INFO((label + " returned status " + std::to_string(listed.status)));
            REQUIRE((listed.status == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO((label + " returned no algorithm"));
            REQUIRE((!listed.results.empty()));
        }
        for(const auto& result : listed.results)
        {
            auto algo = result.algo;
            {
                INFO((label + " returned an algorithm that is not JIT"));
                REQUIRE((isJit(algo)));
            }
            const auto name = hipblaslt_ext::getSolutionNameFromAlgo(handle, algo);
            {
                INFO((label + " selected '" + name + "', not a solution ending in " + suffix));
                REQUIRE((endsWith(name, suffix)));
            }
        }
    }

    void requireUniqueKernels(hipblasLtHandle_t handle, const Listed& listed)
    {
        std::set<std::string> names;
        for(const auto& result : listed.results)
        {
            auto algo = result.algo;
            const auto name = hipblaslt_ext::getKernelNameFromAlgo(handle, algo);
            if(name.empty())
                continue;
            {
                INFO(("repeated kernel " + name));
                REQUIRE((names.insert(name).second));
            }
        }
    }

    // Equality results, then JIT, then the other providers. A full list must
    // end with a non-JIT result; a short list may end on JIT.
    void requireProviderOrder(const Listed& listed, int requested)
    {
        {
            INFO(("an Equality size returned no algorithm"));
            REQUIRE((!listed.results.empty()));
        }
        {
            INFO(("an Equality size did not start with an Equality result"));
            REQUIRE((!isJit(listed.results.front().algo)));
        }
        int  phase  = 0;
        bool sawJit = false;
        for(auto& result : listed.results)
        {
            const bool jit = isJit(result.algo);
            if(phase == 0 && jit)
                phase = 1;
            else if(phase == 1 && !jit)
                phase = 2;
            else if(phase == 2 && jit)
            {
                FAIL(("a JIT result followed the other providers"));
            }
            if(jit)
                sawJit = true;
        }
        {
            INFO(("an Equality size returned no JIT result after the Equality results"));
            REQUIRE((sawJit));
        }
        if(static_cast<int>(listed.results.size()) == requested)
        {
            INFO(("a full Equality-size list did not end with another provider"));
            REQUIRE((!isJit(listed.results.back().algo)));
        }
    }

    void runFirst(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc, hipStream_t stream, int K,
                  const hipblasLtMatmulHeuristicResult_t& result)
    {
        hipblaslt_jit_test::Fp16Gemm matrices(K);
        Device                       workspace(result.workspaceSize);
        float                        alpha = 1.25f, beta = 0.5f;
        matrices.matmul(handle,
                        desc,
                        stream,
                        &result.algo,
                        workspace.pointer,
                        result.workspaceSize,
                        alpha,
                        beta,
                        "");
    }

    bool equalitySize(hipblasLtHandle_t handle, hipblasLtMatmulDesc_t desc, int m, int n, int k)
    {
        hipblasLtMatrixLayout_t a{}, b{}, c{};
        {
            INFO(("equality layout A"));
            REQUIRE((
                hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, m, k, m) == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("equality layout B"));
            REQUIRE((
                hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, k, n, k) == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("equality layout C"));
            REQUIRE((
                hipblasLtMatrixLayoutCreate(&c, HIP_R_16F, m, n, m) == HIPBLAS_STATUS_SUCCESS));
        }
        const int tuned = hipblaslt_ext::matmulIsTuned(handle, desc, a, b, c, c);
        hipblasLtMatrixLayoutDestroy(a);
        hipblasLtMatrixLayoutDestroy(b);
        hipblasLtMatrixLayoutDestroy(c);
        return tuned == 1;
    }

    void checkPair(const std::string& label, const Listed& c, const Listed& cpp, bool jit,
                   const std::string& suffix, hipblasLtHandle_t handle)
    {
        if(jit)
        {
            requireOnlyJit(label + " C", c, suffix, handle);
            requireOnlyJit(label + " C++", cpp, suffix, handle);
        }
        else
        {
            requireNoJit(label + " C", c);
            requireNoJit(label + " C++", cpp);
        }
        requireUniqueKernels(handle, c);
        requireUniqueKernels(handle, cpp);
    }

    // A fresh process querying the same K=512 problem. Prints INDEX <n> and nothing else.
    void reuse(const std::string& root)
    {
        int             device = -1;
        hipDeviceProp_t properties{};
        HIP(hipGetDevice(&device));
        HIP(hipGetDeviceProperties(&properties, device));
        const auto bundles = hipblaslt_jit_test::deviceBundles(root, properties.gcnArchName);
        {
            INFO(("setenv"));
            REQUIRE((setenv("HIPBLASLT_JIT_TEST_REPLAY", (bundles / "plain-pair").u8string().c_str(), 1)
                == 0));
        }

        hipblasLtHandle_t           handle{};
        hipblasLtMatmulDesc_t       desc{};
        hipblasLtMatmulPreference_t pref{};
        {
            INFO(("hipblasLtCreate"));
            REQUIRE((hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("hipblasLtMatmulDescCreate"));
            REQUIRE((hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("preference"));
            REQUIRE((hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("workspace preference"));
            REQUIRE((hipblasLtMatmulPreferenceSetAttribute(pref,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspaceLimit,
                                                  sizeof(workspaceLimit))
                == HIPBLAS_STATUS_SUCCESS));
        }
        Layouts    layouts(M, 512, M, 512, N, 512);
        const auto listed = queryC(handle, desc, pref, layouts, 1);
        {
            INFO(("second process query"));
            REQUIRE((listed.status == HIPBLAS_STATUS_SUCCESS && listed.results.size() == 1));
        }
        auto      algo  = listed.results.front().algo;
        const int index = hipblaslt_ext::getIndexFromAlgo(algo);
        std::cout << "INDEX " << index << '\n' << std::flush;
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(handle);
    }

    void requireSameIndexInChild(const std::string& root, int expected)
    {
        int pipes[2];
        {
            INFO(("pipe"));
            REQUIRE((pipe(pipes) == 0));
        }
        const auto child = fork();
        {
            INFO(("fork"));
            REQUIRE((child >= 0));
        }
        if(child == 0)
        {
            if(dup2(pipes[1], STDOUT_FILENO) < 0)
                _exit(127);
            close(pipes[0]);
            close(pipes[1]);
            if(setenv("HIPBLASLT_JIT_TEST_MODE", "reuse", 1) != 0
               || setenv("HIPBLASLT_JIT_REUSE_ROOT", root.c_str(), 1) != 0)
                _exit(127);
            // Catch2's reporter must not share the pipe that carries INDEX.
            execl("/proc/self/exe",
                  "hipblaslt-jit-heuristic-test",
                  "--out",
                  "/dev/null",
                  static_cast<char*>(nullptr));
            _exit(127);
        }
        close(pipes[1]);
        std::string output;
        char        buffer[256];
        ssize_t     count = 0;
        while((count = read(pipes[0], buffer, sizeof(buffer))) > 0)
            output.append(buffer, buffer + count);
        close(pipes[0]);
        int status = 0;
        {
            INFO(("The second process failed (wait status " + std::to_string(status) + ")"));
            REQUIRE((
                waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0));
        }
        {
            INFO(("The second process reported '" + output + "', not index "
                + std::to_string(expected)));
            REQUIRE((output == "INDEX " + std::to_string(expected) + "\n"));
        }
        std::cout << "PASS a second process reused JIT library index " << expected << '\n';
    }

    void test(const std::string& mode, const std::string& root)
    {
        int             device = -1;
        hipDeviceProp_t properties{};
        HIP(hipGetDevice(&device));
        HIP(hipGetDeviceProperties(&properties, device));
        std::string replayRoot = root;
        if(mode == "ignored")
        {
            INFO(("setenv"));
            REQUIRE((setenv("HIPBLASLT_JIT_TEST_REPLAY", "ignored", 1) == 0));
        }
        else
        {
            const auto stage = fs::u8path(HIPBLASLT_JIT_REPLAY_ROOT) / mode;
            hipblaslt_jit_test::stageDescribedSources(fs::u8path(root), stage);
            replayRoot       = stage.u8string();
            const auto bundles
                = hipblaslt_jit_test::deviceBundles(stage, properties.gcnArchName);
            {
                INFO(("setenv"));
                REQUIRE((setenv("HIPBLASLT_JIT_TEST_REPLAY", (bundles / "plain-pair").u8string().c_str(), 1)
                    == 0));
            }
        }

        hipblasLtHandle_t           handle{};
        hipblasLtMatmulDesc_t       desc{};
        hipblasLtMatmulPreference_t pref{};
        hipStream_t                 stream{};
        {
            INFO(("hipblasLtCreate"));
            REQUIRE((hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("hipblasLtMatmulDescCreate"));
            REQUIRE((hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("preference"));
            REQUIRE((hipblasLtMatmulPreferenceCreate(&pref) == HIPBLAS_STATUS_SUCCESS));
        }
        {
            INFO(("workspace preference"));
            REQUIRE((hipblasLtMatmulPreferenceSetAttribute(pref,
                                                  HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspaceLimit,
                                                  sizeof(workspaceLimit))
                == HIPBLAS_STATUS_SUCCESS));
        }
        HIP(hipStreamCreate(&stream));

        constexpr int K = 512;
        Device        A(M * K * 2), B(K * N * 2), C(M * N * 2), D(M * N * 2);
        Layouts       layouts(M, K, M, K, N, K);
        const int equalitySizes[][3]
            = {{1024, 4096, 20}, {2048, 128, 16}, {864, 512, 432}, {128, 5120, 1024}};
        bool tuned = false;
        if(mode == "fallback")
        {
            for(const auto& size : equalitySizes)
                tuned = tuned || equalitySize(handle, desc, size[0], size[1], size[2]);
        }

        const auto c512 = queryC(handle, desc, pref, layouts, 4);
        const auto cpp512
            = queryCpp(handle, desc, HIPBLAS_OP_N, HIPBLAS_OP_N, layouts, A.pointer, B.pointer,
                       C.pointer, D.pointer, 4);
        Layouts    k256(M, 256, M, 256, N, 256);
        const auto c256 = queryC(handle, desc, pref, k256, 4);
        const auto cpp256 = queryCpp(handle, desc, HIPBLAS_OP_N, HIPBLAS_OP_N, k256, A.pointer,
                                     B.pointer, C.pointer, D.pointer, 4);

        if(mode == "off" || mode == "ignored")
        {
            checkPair("K=512", c512, cpp512, false, "", handle);
            checkPair("K=256", c256, cpp256, false, "", handle);
        }
        else if(mode == "forced" || !tuned)
        {
            checkPair("K=512", c512, cpp512, true, "_K512_WGM8", handle);
            checkPair("K=256", c256, cpp256, true, "_WGM1", handle);
            runFirst(handle, desc, stream, K, c512.results.front());
            std::cout << "PASS K=512 first result\n";
            auto      algo  = c512.results.front().algo;
            const int index = hipblaslt_ext::getIndexFromAlgo(algo);
            requireSameIndexInChild(replayRoot, index);
        }
        else
        {
            {
                INFO(("K=512"));
                REQUIRE((c512.status == HIPBLAS_STATUS_SUCCESS && !c512.results.empty()));
            }
            {
                INFO(("an untuned size with library results did not start with JIT"));
                REQUIRE((isJit(c512.results.front().algo)));
            }
            requireUniqueKernels(handle, c512);
            requireUniqueKernels(handle, cpp512);
            runFirst(handle, desc, stream, K, c512.results.front());
            std::cout << "PASS K=512 first result\n";
        }

        if(mode == "forced" || mode == "fallback")
        {
            hipblasLtMatmulDesc_t    tn{};
            const hipblasOperation_t transpose = HIPBLAS_OP_T;
            {
                INFO(("transposed descriptor"));
                REQUIRE((hipblasLtMatmulDescCreate(&tn, HIPBLAS_COMPUTE_32F, HIP_R_32F)
                    == HIPBLAS_STATUS_SUCCESS));
            }
            {
                INFO(("transpose A"));
                REQUIRE((hipblasLtMatmulDescSetAttribute(tn, HIPBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                                sizeof(transpose))
                    == HIPBLAS_STATUS_SUCCESS));
            }
            Layouts    transposed(K, M, K, K, N, K);
            const auto cT = queryC(handle, tn, pref, transposed, 4);
            const auto cppT = queryCpp(handle, tn, HIPBLAS_OP_T, HIPBLAS_OP_N, transposed, A.pointer,
                                       B.pointer, C.pointer, D.pointer, 4);
            if(mode == "forced")
            {
                {
                    INFO(("forced mode returned a solution for transposed A"));
                    REQUIRE((cT.status == HIPBLAS_STATUS_SUCCESS && cT.results.empty()));
                }
                {
                    INFO(("forced mode C++ returned a solution for transposed A"));
                    REQUIRE((cppT.status == HIPBLAS_STATUS_SUCCESS && cppT.results.empty()));
                }
                std::cout << "PASS transposed A returns no solution\n";
            }
            hipblasLtMatmulDescDestroy(tn);
        }

        if(mode == "fallback")
        {
            if(!tuned)
                std::cout << "SKIP heuristic-provider-order: the build has no device library with "
                             "an Equality size\n";
            else
            {
                for(const auto& size : equalitySizes)
                {
                    if(!equalitySize(handle, desc, size[0], size[1], size[2]))
                        continue;
                    hipblasLtMatrixLayout_t a{}, b{}, c{}, d{};
                    {
                        INFO(("order layout A"));
                        REQUIRE((hipblasLtMatrixLayoutCreate(&a, HIP_R_16F, size[0], size[2], size[0])
                            == HIPBLAS_STATUS_SUCCESS));
                    }
                    {
                        INFO(("order layout B"));
                        REQUIRE((hipblasLtMatrixLayoutCreate(&b, HIP_R_16F, size[2], size[1], size[2])
                            == HIPBLAS_STATUS_SUCCESS));
                    }
                    {
                        INFO(("order layout C"));
                        REQUIRE((hipblasLtMatrixLayoutCreate(&c, HIP_R_16F, size[0], size[1], size[0])
                            == HIPBLAS_STATUS_SUCCESS));
                    }
                    {
                        INFO(("order layout D"));
                        REQUIRE((hipblasLtMatrixLayoutCreate(&d, HIP_R_16F, size[0], size[1], size[0])
                            == HIPBLAS_STATUS_SUCCESS));
                    }
                    std::vector<hipblasLtMatmulHeuristicResult_t> results(8);
                    int                                           count = 0;
                    const auto                                    status
                        = hipblasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, d, pref, 8,
                                                          results.data(), &count);
                    results.resize(count);
                    {
                        INFO(("Equality-size heuristic"));
                        REQUIRE((status == HIPBLAS_STATUS_SUCCESS));
                    }
                    Listed listed{status, std::move(results)};
                    requireProviderOrder(listed, 8);
                    requireUniqueKernels(handle, listed);
                    hipblasLtMatrixLayoutDestroy(a);
                    hipblasLtMatrixLayoutDestroy(b);
                    hipblasLtMatrixLayoutDestroy(c);
                    hipblasLtMatrixLayoutDestroy(d);
                    std::cout << "PASS Equality size " << size[0] << "x" << size[1] << "x"
                              << size[2] << " is Equality, then JIT, then the other providers\n";
                    break;
                }
            }
        }

        std::cout << "PASS heuristic mode " << mode << '\n';
        static_cast<void>(hipStreamDestroy(stream));
        hipblasLtMatmulPreferenceDestroy(pref);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(handle);
    }
}

TEST_CASE("heuristic queries under HIPBLASLT_JIT", "[jit-gpu]")
{
    const char* mode = std::getenv("HIPBLASLT_JIT_TEST_MODE");
    {
        INFO(("Set HIPBLASLT_JIT_TEST_MODE"));
        REQUIRE((mode && *mode));
    }
    const std::string which = mode;
    if(which == "reuse")
    {
        const char* root = std::getenv("HIPBLASLT_JIT_REUSE_ROOT");
        {
            INFO(("reuse is missing its data directory"));
            REQUIRE((root && *root));
        }
        reuse(root);
    }
    else if(which == "ignored")
        test(which, "");
    else if(which == "off" || which == "fallback" || which == "forced")
        test(which, HIPBLASLT_JIT_DATA);
    else
    {
        FAIL(("Unknown mode " + which));
    }
}
