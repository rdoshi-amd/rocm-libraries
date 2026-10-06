// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-replay.hpp"
#include "test_helpers.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// Replays the plain-pair bundle of the current device's architecture through
// Jit, builds it with comgr and loads its two solutions as one TensileLite
// library. getJitAlgo returns its first solution for K=512 and its second for
// K=256; each runs through hipblasLtMatmul and hipblaslt_ext::Gemm, and D is
// checked against the host.
namespace jit    = hipblaslt_ext::experimental::jit;
namespace replay = hipblaslt_ext::experimental::jit::replay;

namespace
{
    using hipblaslt_jit_test::require;

    void hip(hipError_t status, const char* expression)
    {
        require(status == hipSuccess, std::string(expression) + ": " + hipGetErrorString(status));
    }
#define HIP(expression) hip((expression), #expression)
#define BLAS(expression)                                                         \
    do                                                                           \
    {                                                                            \
        const auto status_ = (expression);                                       \
        require(status_ == HIPBLAS_STATUS_SUCCESS,                               \
                std::string(#expression) + ": status " + std::to_string(status_) \
                    + diagnostics.message);                                      \
    } while(false)

    // FP16 NN GEMMs with FP32 compute, which the plain kernel solves.
    constexpr int M = 256, N = 128;

    struct Device
    {
        void* pointer{};
        explicit Device(size_t bytes)
        {
            HIP(hipMalloc(&pointer, bytes));
        }
        ~Device()
        {
            static_cast<void>(hipFree(pointer));
        }
        Device(const Device&)            = delete;
        Device& operator=(const Device&) = delete;
    };

    // A and B are small multiples of 1/8 and C of 1/4, so every product and sum is exact.
    std::vector<__half> fill(size_t count, int a, int b, int mod, float scale)
    {
        std::vector<__half> values(count);
        for(size_t i = 0; i < count; ++i)
            values[i] = __float2half(float(int(i * a + i / 7 * b) % mod - mod / 2) * scale);
        return values;
    }

    void verify(const std::string&         label,
                int                        K,
                const std::vector<__half>& a,
                const std::vector<__half>& b,
                const std::vector<__half>& c,
                const void*                d,
                float                      alpha,
                float                      beta)
    {
        std::vector<__half> out(M * N);
        HIP(hipMemcpy(out.data(), d, out.size() * sizeof(__half), hipMemcpyDeviceToHost));
        for(int col = 0; col < N; ++col)
            for(int row = 0; row < M; ++row)
            {
                float sum = 0;
                for(int k = 0; k < K; ++k)
                    sum += __half2float(a[row + k * M]) * __half2float(b[k + col * K]);
                const auto i        = row + col * M;
                const auto expected = __half2float(
                    __float2half(alpha * sum + beta * __half2float(c[i])));
                const auto actual = __half2float(out[i]);
                require(std::isfinite(actual)
                            && std::abs(actual - expected) <= 0.0005f + 0.001f * std::abs(expected),
                        label + ": D[" + std::to_string(i) + "] is " + std::to_string(actual)
                            + ", expected " + std::to_string(expected));
            }
        std::cout << "PASS " << label << '\n';
    }

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

        const auto hostA = fill(size_t(M) * K, 3, 5, 13, 1 / 8.0f);
        const auto hostB = fill(size_t(K) * N, 7, 2, 11, 1 / 8.0f);
        const auto hostC = fill(size_t(M) * N, 1, 3, 7, 1 / 4.0f);
        Device     A(hostA.size() * 2), B(hostB.size() * 2), C(hostC.size() * 2), D(M * N * 2);
        HIP(hipMemcpy(A.pointer, hostA.data(), hostA.size() * 2, hipMemcpyHostToDevice));
        HIP(hipMemcpy(B.pointer, hostB.data(), hostB.size() * 2, hipMemcpyHostToDevice));
        HIP(hipMemcpy(C.pointer, hostC.data(), hostC.size() * 2, hipMemcpyHostToDevice));
        hipblasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};
        BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, M, K, M));
        BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, K, N, K));
        BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, M, N, M));
        BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16F, M, N, M));

        // Generate (replay), Build (comgr), Support, then Load into a TensileLite library.
        jit::Request  request;
        jit::Solution solution;
        float         alpha = 1.25f, beta = 0.5f;
        BLAS(jit::makeGemmRequest(handle, desc, &alpha, A.pointer, la, B.pointer, lb, &beta,
                                  C.pointer, lc, D.pointer, ld, request, diagnostics));
        BLAS(jit::getJitAlgo(device, request, backend, 64 << 20, solution, diagnostics));
        hipblasLtMatmulHeuristicResult_t result{};
        BLAS(jit::getGemmAlgo(solution, result, diagnostics));
        Device     workspace(result.workspaceSize);
        const auto name   = hipblaslt_ext::getSolutionNameFromAlgo(handle, result.algo);
        const auto kernel = hipblaslt_ext::getKernelNameFromAlgo(handle, result.algo);
        require(name.size() > suffix.size()
                    && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0,
                label + " selected the solution '" + name + "', not the one ending in " + suffix);
        require(kernel.rfind("Cijk_", 0) == 0, "Unexpected kernel name '" + kernel + "'");
        std::cout << "PASS " << label << " replayed, built and loaded the solution ending in "
                  << suffix << '\n';

        HIP(hipMemset(D.pointer, 0xff, M * N * 2));
        BLAS(hipblasLtMatmul(handle, desc, &alpha, A.pointer, la, B.pointer, lb, &beta, C.pointer,
                             lc, D.pointer, ld, &result.algo, workspace.pointer,
                             result.workspaceSize, stream));
        HIP(hipStreamSynchronize(stream));
        verify("hipblasLtMatmul, " + label, K, hostA, hostB, hostC, D.pointer, alpha, beta);

        hipblaslt_ext::Gemm gemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F,
                                 HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F);
        float gemmAlpha = -0.75f, gemmBeta = 0.25f;
        BLAS(gemm.setProblem(desc, &gemmAlpha, A.pointer, la, B.pointer, lb, &gemmBeta,
                             C.pointer, lc, D.pointer, ld));
        size_t needed = 0;
        BLAS(gemm.isAlgoSupported(result.algo, needed));
        require(needed == result.workspaceSize, "isAlgoSupported returned another workspace size");
        HIP(hipMemset(D.pointer, 0xff, M * N * 2));
        BLAS(gemm.initialize(result.algo, workspace.pointer, true, stream));
        BLAS(gemm.run(stream));
        HIP(hipStreamSynchronize(stream));
        verify("hipblaslt_ext::Gemm, " + label, K, hostA, hostB, hostC, D.pointer, gemmAlpha,
               gemmBeta);

        hipblasLtMatrixLayoutDestroy(la);
        hipblasLtMatrixLayoutDestroy(lb);
        hipblasLtMatrixLayoutDestroy(lc);
        hipblasLtMatrixLayoutDestroy(ld);
    }

    void test(const std::string& root)
    {
        int             device = -1;
        hipDeviceProp_t properties{};
        HIP(hipGetDevice(&device));
        HIP(hipGetDeviceProperties(&properties, device));
        const auto bundles = hipblaslt_jit_test::deviceBundles(root, properties.gcnArchName);
        jit::Diagnostics diagnostics;

        hipblasLtHandle_t     handle{};
        hipblasLtMatmulDesc_t desc{};
        hipStream_t           stream{};
        BLAS(hipblasLtCreate(&handle));
        BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        HIP(hipStreamCreate(&stream));
        jit::Backend backend;
        BLAS(replay::createBackend({{(bundles / "plain-pair").u8string()}}, backend, diagnostics));

        run(handle, desc, stream, backend, 512, "_K512_WGM8");
        run(handle, desc, stream, backend, 256, "_WGM1");

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
        require(jit::getJitAlgo(device, transposed, backend, 64 << 20, unsolved, diagnostics)
                    == HIPBLAS_STATUS_NOT_SUPPORTED,
                "The replay backend accepted a problem its bundle does not solve");
        std::cout << "PASS a problem the bundle does not solve is not supported\n";
        hipblasLtMatrixLayoutDestroy(lt);
        hipblasLtMatrixLayoutDestroy(lb);
        hipblasLtMatrixLayoutDestroy(lc);
        hipblasLtMatmulDescDestroy(tn);

        static_cast<void>(hipStreamDestroy(stream));
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(handle);
    }
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " BUNDLES\n";
        return 2;
    }
    try
    {
        test(argv[1]);
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
