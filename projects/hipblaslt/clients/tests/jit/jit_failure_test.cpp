// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-replay.hpp"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

// Failures of the real JIT stages: the library's replay backend, comgr builder
// and TensileLite loader, run through Jit and through the GEMM entry points on
// damaged copies of the committed bundles.
namespace hj     = hipblaslt_jit;
namespace jit    = hipblaslt_ext::experimental::jit;
namespace abi    = hipblaslt_ext::experimental::jit::detail;
namespace replay = hipblaslt_ext::experimental::jit::replay;
namespace fs     = std::filesystem;
using Code       = hj::Status::Code;
using hj::Stage;

namespace
{
    void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }
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

    // FP16 NN GEMM with FP32 compute, which both bundles solve.
    constexpr int           M = 256, N = 128, K = 512;
    constexpr unsigned char poisonD = 0xff, poisonWorkspace = 0xa5;

    struct Device
    {
        void*  pointer{};
        size_t bytes{};
        explicit Device(size_t size)
            : bytes(size)
        {
            HIP(hipMalloc(&pointer, bytes ? bytes : 1));
        }
        ~Device()
        {
            static_cast<void>(hipFree(pointer));
        }
        Device(const Device&)                               = delete;
        Device&                    operator=(const Device&) = delete;
        std::vector<unsigned char> read() const
        {
            std::vector<unsigned char> host(bytes);
            HIP(hipMemcpy(host.data(), pointer, bytes, hipMemcpyDeviceToHost));
            return host;
        }
    };

    struct Problem
    {
        jit::Diagnostics        diagnostics;
        hipblasLtHandle_t       handle{};
        hipblasLtMatmulDesc_t   desc{};
        hipblasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};
        hipStream_t             stream{};
        Device                  A{size_t(M) * K * 2}, B{size_t(K) * N * 2}, C{size_t(M) * N * 2},
            D{size_t(M) * N * 2};
        float alpha = 1.25f, beta = 0.5f;

        Problem()
        {
            BLAS(hipblasLtCreate(&handle));
            BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
            BLAS(hipblasLtMatrixLayoutCreate(&la, HIP_R_16F, M, K, M));
            BLAS(hipblasLtMatrixLayoutCreate(&lb, HIP_R_16F, K, N, K));
            BLAS(hipblasLtMatrixLayoutCreate(&lc, HIP_R_16F, M, N, M));
            BLAS(hipblasLtMatrixLayoutCreate(&ld, HIP_R_16F, M, N, M));
            HIP(hipStreamCreate(&stream));
            // Every FP16 input is 0x3030, about 0.13.
            for(auto* input : {&A, &B, &C})
                HIP(hipMemset(input->pointer, 0x30, input->bytes));
        }
        ~Problem()
        {
            static_cast<void>(hipStreamDestroy(stream));
            hipblasLtMatrixLayoutDestroy(la);
            hipblasLtMatrixLayoutDestroy(lb);
            hipblasLtMatrixLayoutDestroy(lc);
            hipblasLtMatrixLayoutDestroy(ld);
            hipblasLtMatmulDescDestroy(desc);
            hipblasLtDestroy(handle);
        }
        jit::Request request()
        {
            jit::Request result;
            BLAS(jit::makeGemmRequest(handle,
                                      desc,
                                      &alpha,
                                      A.pointer,
                                      la,
                                      B.pointer,
                                      lb,
                                      &beta,
                                      C.pointer,
                                      lc,
                                      D.pointer,
                                      ld,
                                      result,
                                      diagnostics));
            return result;
        }
        hipblasStatus_t matmul(const hipblasLtMatmulAlgo_t& algo, const Device& workspace)
        {
            return hipblasLtMatmul(handle,
                                   desc,
                                   &alpha,
                                   A.pointer,
                                   la,
                                   B.pointer,
                                   lb,
                                   &beta,
                                   C.pointer,
                                   lc,
                                   D.pointer,
                                   ld,
                                   &algo,
                                   workspace.pointer,
                                   workspace.bytes,
                                   stream);
        }
    };

    jit::Backend backend(const replay::Options& options)
    {
        jit::Backend     result;
        jit::Diagnostics diagnostics;
        BLAS(replay::createBackend(options, result, diagnostics));
        return result;
    }

    std::vector<std::string> kernels(const hj::Jit::Outcome& outcome)
    {
        std::vector<std::string> names;
        for(const auto& bundle : outcome.bundles)
            names.push_back(bundle->kernelNames());
        return names;
    }

    void failed(const hj::Jit::Outcome& outcome,
                Code                    code,
                Stage                   stage,
                const std::string&      message,
                const std::string&      label)
    {
        require(!outcome.failures.empty(), label + ": no failure recorded");
        const auto& status = outcome.failures.front();
        require(status.code == code && status.stage == stage
                    && status.message.find(message) != std::string::npos,
                label + ": unexpected " + hj::toString(status.stage) + " failure '" + status.message
                    + "'");
    }

    // The Jit scratch directories that exist under TMPDIR.
    std::vector<fs::path> scratches()
    {
        std::vector<fs::path> paths;
        for(const auto& entry : fs::directory_iterator(fs::temp_directory_path()))
            if(entry.path().filename().string().rfind("hipblaslt-jit-", 0) == 0)
                paths.push_back(entry.path());
        return paths;
    }

    std::string readText(const fs::path& path)
    {
        std::ifstream file(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(file), {}};
    }
    void writeText(const fs::path& path, const std::string& text)
    {
        std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
    }
    void replaceAll(const fs::path& path, const std::string& from, const std::string& to)
    {
        auto text = readText(path);
        require(text.find(from) != std::string::npos, "No '" + from + "' in " + path.u8string());
        for(auto at = text.find(from); at != std::string::npos;
            at      = text.find(from, at + to.size()))
            text.replace(at, from.size(), to);
        writeText(path, text);
    }
    fs::path mainAssembly(const fs::path& bundle)
    {
        for(const auto& entry : fs::directory_iterator(bundle / "sources"))
            if(entry.path().extension() == ".s")
                return entry.path();
        throw std::runtime_error("No main assembly in " + bundle.u8string());
    }

    // Copies bundle to scratch/name and lets damage edit the copy.
    fs::path damaged(const fs::path&                             bundle,
                     const fs::path&                             scratch,
                     const std::string&                          name,
                     const std::function<void(const fs::path&)>& damage)
    {
        const auto copy = scratch / "bundles" / name;
        fs::create_directories(copy.parent_path());
        fs::copy(bundle, copy, fs::copy_options::recursive);
        damage(copy);
        return copy;
    }

    // Jit over the library's real stages, with the device target and request
    // that getJitAlgo used.
    void stages(Problem& p, const fs::path& plain, const fs::path& splitk)
    {
        int device = -1;
        HIP(hipGetDevice(&device));
        auto&         diagnostics = p.diagnostics;
        auto          request     = p.request();
        jit::Solution solution;
        BLAS(jit::getJitAlgo(
            device, request, backend({{plain.u8string()}}), 64 << 20, solution, diagnostics));
        const auto& target    = abi::SolutionAccess::get(solution)->target;
        const auto& operation = *abi::RequestAccess::get(request);
        const auto  run       = [&](const replay::Options&          options,
                             size_t                          count,
                             size_t                          workspaceLimit,
                             const std::vector<std::string>& exclude = {},
                             const hj::DeviceTarget*         other   = nullptr) {
            return abi::BackendAccess::get(backend(options))
                ->generate(operation, other ? *other : target, count, workspaceLimit, exclude);
        };
        const replay::Options both{{plain.u8string(), splitk.u8string()}};

        auto       outcome = run(both, 2, 64 << 20);
        const auto names   = kernels(outcome);
        require(names.size() == 2 && names[0] != names[1] && outcome.failures.empty()
                    && outcome.summary == "Replayed 2 bundles",
                "Two replayed bundles were not both built and loaded in order");
        require(kernels(run(both, 1, 64 << 20)) == std::vector<std::string>{names[0]},
                "A count of 1 did not return only the first bundle");
        require(kernels(run(both, 2, 64 << 20, {names[0]})) == std::vector<std::string>{names[1]},
                "An excluded kernel was returned");
        outcome = run(both, 2, 64 << 20, names);
        require(outcome.bundles.empty() && outcome.failures.empty(),
                "Excluding every kernel did not return an empty outcome");
        require(scratches().empty(), "A successful generation left its scratch directory");
        std::cout << "PASS two bundles: count, order and excluded kernels\n";

        outcome = run({{splitk.u8string(), plain.u8string()}}, 2, 0);
        failed(outcome, Code::NotSupported, Stage::Support, "TensileLite solution", "Support");
        require(kernels(outcome) == std::vector<std::string>{names[0]}
                    && outcome.failures.size() == 1,
                "A support failure dropped the other bundle");
        require(scratches().empty(), "A failure that wrote nothing kept its scratch directory");
        std::cout << "PASS a workspace limit fails split-K at support, keeps the other bundle, and "
                     "removes the empty scratch: "
                  << outcome.failures[0].message.substr(0, 80) << '\n';

        const std::pair<replay::Options::Fault, Stage> faults[]
            = {{replay::Options::Fault::Generate, Stage::Generate},
               {replay::Options::Fault::Build, Stage::Build}};
        for(const auto& [fault, stage] : faults)
        {
            outcome = run({{splitk.u8string()}, fault}, 1, 64 << 20);
            failed(outcome,
                   Code::Failed,
                   stage,
                   stage == Stage::Generate ? "Replay generation fault"
                                            : "comgr could not assemble",
                   hj::toString(stage));
            const fs::path log  = outcome.failures[0].logPath;
            const auto     kept = scratches();
            require(outcome.bundles.empty() && kept.size() == 1 && log.parent_path() == kept[0]
                        && fs::is_regular_file(log),
                    std::string(hj::toString(stage)) + " failure did not keep its log in scratch");
            if(stage == Stage::Generate)
                require(readText(log).find(" sizes=256,128,1,512 count=1 exclude=")
                            != std::string::npos,
                        "The generation log does not describe the request: " + readText(log));
            fs::remove_all(kept[0]);
        }
        std::cout << "PASS generate and build faults fail at their stage and keep their log\n";

        // The device has the opposite XNACK setting, so the code object built
        // for this target ID does not load.
        auto other   = target;
        auto feature = other.targetId.find("xnack-");
        if(feature == std::string::npos)
            feature = other.targetId.find("xnack+");
        require(feature != std::string::npos, "No XNACK setting in " + target.targetId);
        auto& setting = other.targetId[feature + 5];
        setting       = setting == '-' ? '+' : '-';
        outcome       = run({{plain.u8string()}}, 1, 64 << 20, {}, &other);
        failed(outcome, Code::Failed, Stage::Load, "Load generated code object", "Load");
        for(const auto& path : scratches())
            fs::remove_all(path);
        std::cout << "PASS a code object for " << other.targetId << " fails at load on "
                  << target.targetId << '\n';
    }

    // Damaged copies of the split-K bundle, through the GEMM entry points.
    void bundles(Problem& p, const fs::path& splitk, const fs::path& scratch)
    {
        int device = -1;
        HIP(hipGetDevice(&device));
        auto&      diagnostics = p.diagnostics;
        const auto request     = p.request();
        const auto escape      = scratch / "outside.h";
        writeText(escape, "#pragma once\n");
        struct Case
        {
            std::string                          name;
            std::function<void(const fs::path&)> damage;
            hipblasStatus_t                      status; // from createBackend, else getJitAlgo
            std::string                          message;
        };
        const auto library = [](const fs::path& b) { return b / "library/TensileLibrary.dat"; };
        const Case cases[] = {
            {"corrupt-library",
             [&](const fs::path& b) { writeText(library(b), "bad library"); },
             HIPBLAS_STATUS_INVALID_VALUE,
             "Expected a non-lazy library"},
            {"truncated-library",
             [&](const fs::path& b) {
                 const auto entry = readText(library(b));
                 writeText(library(b), entry.substr(0, entry.size() / 2));
             },
             HIPBLAS_STATUS_INVALID_VALUE,
             "Expected a non-lazy library"},
            // MsgPack false (0xc2) becomes true (0xc3) in the solution's problem
            // type and in its output-amax predicate.
            {"mismatched-amax",
             [&](const fs::path& b) {
                 const std::string type  = "\xab"
                                           "outputAmaxD",
                                   check = "\xaa"
                                           "AmaxDCheck\xa5"
                                           "value";
                 replaceAll(library(b), type + "\xc2", type + "\xc3");
                 replaceAll(library(b), check + "\xc2", check + "\xc3");
             },
             HIPBLAS_STATUS_NOT_SUPPORTED,
             "No replayed solution solves this problem"},
            {"missing-sources",
             [](const fs::path& b) { fs::remove_all(b / "sources"); },
             HIPBLAS_STATUS_INVALID_VALUE,
             "Missing source directory"},
            {"missing-main",
             [](const fs::path& b) { fs::remove(mainAssembly(b)); },
             HIPBLAS_STATUS_INVALID_VALUE,
             "No main kernel assembly"},
            {"path-escape",
             [&](const fs::path& b) { fs::create_symlink(escape, b / "sources/escape.h"); },
             HIPBLAS_STATUS_INVALID_VALUE,
             "Artifact symlink escapes bundle"},
            {"other-architecture",
             [](const fs::path& b) { replaceAll(mainAssembly(b), "--gfx950\"", "--gfx942\""); },
             HIPBLAS_STATUS_INTERNAL_ERROR,
             "targets gfx942"},
            {"corrupt-assembly",
             [](const fs::path& b) {
                 const auto s = mainAssembly(b);
                 writeText(s, readText(s) + "\n  s_not_an_instruction v0\n");
             },
             HIPBLAS_STATUS_INTERNAL_ERROR,
             "comgr could not assemble"},
            {"broken-helper",
             [](const fs::path& b) {
                 const auto s = b / "sources/Kernels.cpp";
                 writeText(s, readText(s) + "\nthis is not C++;\n");
             },
             HIPBLAS_STATUS_INTERNAL_ERROR,
             "comgr could not compile HIP"},
            {"renamed-main-kernel",
             [](const fs::path& b) { replaceAll(mainAssembly(b), "Cijk_", "Xijk_"); },
             HIPBLAS_STATUS_INTERNAL_ERROR,
             "does not define kernel"},
        };
        for(const auto& c : cases)
        {
            const auto    copy = damaged(splitk, scratch, c.name, c.damage);
            jit::Backend  provider;
            jit::Solution solution;
            auto status = replay::createBackend({{copy.u8string()}}, provider, diagnostics);
            if(status == HIPBLAS_STATUS_SUCCESS)
                status
                    = jit::getJitAlgo(device, request, provider, 64 << 20, solution, diagnostics);
            require(status == c.status && !abi::SolutionAccess::get(solution)
                        && diagnostics.message.find(c.message) != std::string::npos,
                    c.name + ": status " + std::to_string(status) + ", '" + diagnostics.message
                        + "'");
            if(c.status == HIPBLAS_STATUS_INTERNAL_ERROR && c.message.rfind("comgr", 0) == 0)
                require(diagnostics.message.find("comgr.log") != std::string::npos,
                        c.name + ": the message does not name comgr.log");
            std::cout << "PASS " << c.name << ": " << c.message << '\n';
        }
        for(const auto& path : scratches())
            fs::remove_all(path);
    }

    // A split-K solution whose code object lacks its helper kernels builds and
    // loads, but no GEMM call may submit work with it.
    void helpers(Problem& p, const fs::path& splitk, const fs::path& scratch)
    {
        int device = -1;
        HIP(hipGetDevice(&device));
        auto&      diagnostics = p.diagnostics;
        const auto request     = p.request();
        const auto algorithm   = [&](const fs::path& bundle) {
            jit::Solution                    solution;
            hipblasLtMatmulHeuristicResult_t result{};
            BLAS(jit::getJitAlgo(
                device, request, backend({{bundle.u8string()}}), 64 << 20, solution, diagnostics));
            BLAS(jit::getGemmAlgo(solution, result, diagnostics));
            return result;
        };
        const auto good = algorithm(splitk);
        require(good.workspaceSize > 0, "The split-K solution needs no workspace");
        Device workspace(good.workspaceSize);

        hipblaslt_ext::Gemm gemm(p.handle,
                                 HIPBLAS_OP_N,
                                 HIPBLAS_OP_N,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIP_R_16F,
                                 HIPBLAS_COMPUTE_32F);
        BLAS(gemm.setProblem(p.desc,
                             &p.alpha,
                             p.A.pointer,
                             p.la,
                             p.B.pointer,
                             p.lb,
                             &p.beta,
                             p.C.pointer,
                             p.lc,
                             p.D.pointer,
                             p.ld));
        BLAS(gemm.initialize(good.algo, workspace.pointer, false, p.stream));
        BLAS(gemm.run(p.stream));
        HIP(hipStreamSynchronize(p.stream));
        const auto expected = p.D.read();
        const auto kernel   = gemm.getKernelName();

        const auto poison = [&] {
            HIP(hipMemset(p.D.pointer, poisonD, p.D.bytes));
            HIP(hipMemset(workspace.pointer, poisonWorkspace, workspace.bytes));
        };
        const auto untouched = [&](const std::string& label) {
            HIP(hipStreamSynchronize(p.stream));
            const auto d = p.D.read(), w = workspace.read();
            require(std::all_of(d.begin(), d.end(), [](auto b) { return b == poisonD; }),
                    label + " wrote D");
            require(std::all_of(w.begin(), w.end(), [](auto b) { return b == poisonWorkspace; }),
                    label + " wrote the workspace");
        };
        const std::pair<std::string, std::function<void(const fs::path&)>> cases[] = {
            {"removed-helper-source",
             [](const fs::path& b) { fs::remove(b / "sources/Kernels.cpp"); }},
            // Every Cijk_ name in the helper sources is a helper kernel.
            {"renamed-helper-kernels",
             [](const fs::path& b) {
                 for(const auto* file : {"sources/Kernels.cpp", "sources/Kernels.h"})
                     replaceAll(b / file, "Cijk_", "Xijk_");
             }},
        };
        for(const auto& [name, damage] : cases)
        {
            const auto bad = algorithm(damaged(splitk, scratch, name, damage));
            poison();
            require(p.matmul(bad.algo, workspace) != HIPBLAS_STATUS_SUCCESS,
                    name + ": hipblasLtMatmul accepted a solution without its helper kernels");
            untouched(name + ": hipblasLtMatmul");
            require(gemm.initialize(bad.algo, workspace.pointer, false, p.stream)
                        != HIPBLAS_STATUS_SUCCESS,
                    name + ": Gemm::initialize accepted a solution without its helper kernels");
            untouched(name + ": Gemm::initialize");
            require(gemm.getKernelName() == kernel,
                    name + ": a failed Gemm::initialize changed the prepared kernel");
            BLAS(gemm.run(p.stream));
            HIP(hipStreamSynchronize(p.stream));
            require(p.D.read() == expected,
                    name + ": the earlier Gemm::initialize did not stay runnable");
            std::cout << "PASS " << name
                      << ": nothing submitted, and the earlier Gemm::initialize still runs\n";
        }
    }
}

int main(int argc, char** argv)
{
    if(argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " PLAIN_BUNDLE SPLITK_BUNDLE SCRATCH\n";
        return 2;
    }
    try
    {
        const fs::path plain = fs::absolute(argv[1]), splitk = fs::absolute(argv[2]),
                       scratch = fs::absolute(argv[3]);
        fs::remove_all(scratch);
        fs::create_directories(scratch / "tmp");
#ifdef _WIN32
        require(_putenv_s("TMP", (scratch / "tmp").string().c_str()) == 0, "Cannot set TMP");
#else
        require(setenv("TMPDIR", (scratch / "tmp").c_str(), 1) == 0, "Cannot set TMPDIR");
#endif
        Problem p;
        stages(p, plain, splitk);
        bundles(p, splitk, scratch);
        helpers(p, splitk, scratch);
        std::cout << "ALL JIT FAILURE CHECKS PASSED\n";
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
