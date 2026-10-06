// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-hash.hpp"
#include "hipblaslt-jit-heuristic.hpp"
#include "hipblaslt-jit-mode.hpp"
#include "hipblaslt_internal.hpp"
#include "rocblaslt_secure_env.hpp"
#include "hipblaslt-jit-replay.hpp"
#include "rocblaslt.h"
#include "rocblaslt_arch_revision.hpp"
#include <Tensile/hip/HipHardware.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace hipblaslt_jit
{
    Status DeviceTarget::make(int device, DeviceTarget& target)
    {
        target        = {};
        target.device = device;
        const auto error = hipGetDeviceProperties(&target.properties, device);
        if(error != hipSuccess)
            return {Status::Code::Failed,
                    Stage::Configure,
                    std::string("Cannot query HIP device properties: ") + hipGetErrorString(error)};
        target.targetId      = target.properties.gcnArchName;
        target.isa           = target.targetId.substr(0, target.targetId.find(':'));
        target.libraryArch   = rocblaslt_revisioned_arch_name(target.isa,
                                                            target.properties.asicRevision);
        target.wavefrontSize = target.properties.warpSize;
        target.cuCount       = target.properties.multiProcessorCount;
        target.hardware      = TensileLite::hip::GetDevice(target.properties, device);
        return {};
    }
}

namespace hipblaslt_ext::experimental
{
    namespace jit::detail
    {
        namespace
        {
            uint64_t processId()
            {
#ifdef _WIN32
                return _getpid();
#else
                return getpid();
#endif
            }
            struct Registry
            {
                std::mutex                                                            mutex;
                std::unordered_map<uint64_t, std::shared_ptr<const CompiledSolution>> entries;
                std::unordered_map<const CompiledSolution*, uint64_t>                 tokens;
                std::mt19937_64 random{std::random_device{}()};
            };
            Registry& registry()
            {
                // Copied algorithms and captured graphs can outlive their creator.
                static auto* instance = new Registry;
                return *instance;
            }

            hipblasStatus_t toHipStatus(hipblaslt_jit::Status::Code code)
            {
                using Code = hipblaslt_jit::Status::Code;
                switch(code)
                {
                case Code::Success:
                    return HIPBLAS_STATUS_SUCCESS;
                case Code::NotSupported:
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                case Code::TargetMismatch:
                    return HIPBLAS_STATUS_ARCH_MISMATCH;
                default:
                    return HIPBLAS_STATUS_INTERNAL_ERROR;
                }
            }

            // support() plus the workspace bound supportJit applies to a stored algorithm.
            hipblasStatus_t acceptBundle(const KernelBundle&     bundle,
                                         const OperationRequest& request,
                                         size_t                  workspaceLimit,
                                         size_t&                 workspaceBytes,
                                         Diagnostics&            diagnostics)
            {
                workspaceBytes = 0;
                size_t     required = 0;
                const auto status = bundle.support(request, workspaceLimit, required, diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                    return status;
                if(required > workspaceLimit)
                    return HIPBLAS_STATUS_INVALID_VALUE;
                workspaceBytes = required;
                return HIPBLAS_STATUS_SUCCESS;
            }
        }

        uint64_t registerBundle(std::shared_ptr<const CompiledSolution> bundle)
        {
            auto&                       r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            auto                        existing = r.tokens.find(bundle.get());
            if(existing != r.tokens.end())
                return existing->second;
            uint64_t token;
            do
                token = r.random() & ((uint64_t{1} << 56) - 1);
            while(token == 0 || r.entries.count(token));
            r.entries.emplace(token, bundle);
            try
            {
                r.tokens.emplace(bundle.get(), token);
            }
            catch(...)
            {
                r.entries.erase(token);
                throw;
            }
            return token;
        }

        std::shared_ptr<const CompiledSolution> resolveJitAlgo(const rocblaslt_matmul_algo& algo,
                                                               int                          device)
        {
            if(!experimental::detail::isJitAlgo(algo))
                return {};
            int index;
            std::memcpy(&index, algo.data, sizeof(index));
            uint64_t token = 0;
            std::memcpy(&token, algo.data_pad, sizeof(algo.data_pad));
            std::shared_ptr<const CompiledSolution> entry;
            {
                auto&                       r = registry();
                std::lock_guard<std::mutex> lock(r.mutex);
                auto                        found = r.entries.find(token);
                if(algo.fallback || found == r.entries.end()
                   || index != found->second->bundle->solutionIndex())
                    throw std::invalid_argument("Unknown process-local JIT algorithm");
                entry = found->second;
            }
            int current = -1;
            if(entry->process != processId() || entry->target.device != device
               || hipGetDevice(&current) != hipSuccess || current != device)
                throw std::invalid_argument("JIT algorithm belongs to another process or device");
            return entry;
        }

        rocblaslt_status toRocStatus(hipblasStatus_t status)
        {
            switch(status)
            {
            case HIPBLAS_STATUS_SUCCESS:
                return rocblaslt_status_success;
            case HIPBLAS_STATUS_NOT_INITIALIZED:
                return rocblaslt_status_not_initialized;
            case HIPBLAS_STATUS_ALLOC_FAILED:
                return rocblaslt_status_memory_error;
            case HIPBLAS_STATUS_INVALID_VALUE:
                return rocblaslt_status_invalid_value;
            case HIPBLAS_STATUS_ARCH_MISMATCH:
                return rocblaslt_status_arch_mismatch;
            case HIPBLAS_STATUS_NOT_SUPPORTED:
                return rocblaslt_status_not_supported;
            case HIPBLAS_STATUS_EXECUTION_FAILED:
                return rocblaslt_status_execution_failed;
            default:
                return rocblaslt_status_internal_error;
            }
        }

        template <class F>
        rocblaslt_status invoke(F&& f)
        {
            try
            {
                return toRocStatus(f());
            }
            catch(const std::bad_alloc&)
            {
                return rocblaslt_status_memory_error;
            }
            catch(const std::invalid_argument&)
            {
                return rocblaslt_status_invalid_value;
            }
            catch(...)
            {
                return rocblaslt_status_internal_error;
            }
        }

        hipblasStatus_t
            compiledFromBundle(std::shared_ptr<const hipblaslt_jit::Jit> jit,
                               std::shared_ptr<const OperationRequest>   request,
                               hipblaslt_jit::DeviceTarget               target,
                               std::shared_ptr<const KernelBundle>       bundle,
                               size_t                                    workspaceLimit,
                               std::shared_ptr<const CompiledSolution>&  compiled,
                               Diagnostics&                              diagnostics)
        {
            compiled = nullptr;
            size_t     workspace = 0;
            const auto status
                = acceptBundle(*bundle, *request, workspaceLimit, workspace, diagnostics);
            if(status != HIPBLAS_STATUS_SUCCESS)
                return status;
            auto solution            = std::make_shared<CompiledSolution>();
            solution->target         = std::move(target);
            solution->request        = std::move(request);
            solution->jit            = std::move(jit);
            solution->bundle         = std::move(bundle);
            solution->process        = processId();
            solution->workspaceLimit = workspaceLimit;
            solution->workspaceBytes = workspace;
            compiled                 = std::move(solution);
            return HIPBLAS_STATUS_SUCCESS;
        }

        rocblaslt_status supportJit(rocblaslt_handle             handle,
                                    const rocblaslt_matmul_algo& algo,
                                    const GemmRequest&           request,
                                    size_t&                      workspaceBytes)
        {
            workspaceBytes = 0;
            return invoke([&] {
                auto        entry = resolveJitAlgo(algo, handle->device);
                Diagnostics diagnostics;
                return acceptBundle(
                    *entry->bundle, request, algo.max_workspace_bytes, workspaceBytes, diagnostics);
            });
        }
    }

    namespace jit
    {
        hipblasStatus_t makeGemmRequest(hipblasLtHandle_t       handle,
                                        hipblasLtMatmulDesc_t   desc,
                                        const void*             alpha,
                                        const void*             A,
                                        hipblasLtMatrixLayout_t matA,
                                        const void*             B,
                                        hipblasLtMatrixLayout_t matB,
                                        const void*             beta,
                                        const void*             C,
                                        hipblasLtMatrixLayout_t matC,
                                        void*                   D,
                                        hipblasLtMatrixLayout_t matD,
                                        Request&                request,
                                        Diagnostics&            diagnostics)
        {
            request     = {};
            diagnostics = {};
            try
            {
                std::shared_ptr<const detail::GemmRequest> operation;
                auto                                       status = RocBlasLtStatusToHIPStatus(
                    detail::createGemmRequest(reinterpret_cast<rocblaslt_handle>(handle),
                                              reinterpret_cast<rocblaslt_matmul_desc>(desc),
                                              alpha,
                                              A,
                                              reinterpret_cast<rocblaslt_matrix_layout>(matA),
                                              B,
                                              reinterpret_cast<rocblaslt_matrix_layout>(matB),
                                              beta,
                                              C,
                                              reinterpret_cast<rocblaslt_matrix_layout>(matC),
                                              D,
                                              reinterpret_cast<rocblaslt_matrix_layout>(matD),
                                              operation));
                if(status != HIPBLAS_STATUS_SUCCESS)
                {
                    diagnostics.message = "Invalid GEMM descriptors";
                    return status;
                }
                if(!operation || !operation->problem.m || !operation->problem.n)
                {
                    diagnostics.message = "Empty output needs no GEMM algorithm";
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                }
                request = detail::RequestAccess::make(std::move(operation));
                return HIPBLAS_STATUS_SUCCESS;
            }
            catch(const std::bad_alloc&)
            {
                return HIPBLAS_STATUS_ALLOC_FAILED;
            }
            catch(const std::exception& e)
            {
                diagnostics.message = e.what();
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
        }

        hipblasStatus_t getJitAlgo(int            device,
                                   const Request& request,
                                   const Backend& backend,
                                   size_t         workspaceLimit,
                                   Solution&      solution,
                                   Diagnostics&   diagnostics)
        {
            solution    = {};
            diagnostics = {};
            try
            {
                auto jit       = detail::BackendAccess::get(backend);
                auto operation = detail::RequestAccess::get(request);
                if(!operation || !jit)
                {
                    diagnostics.message = "A valid request and backend are required";
                    return HIPBLAS_STATUS_INVALID_VALUE;
                }
                diagnostics.backend = jit->components().backend->info().name;
                int current         = -1;
                if(hipGetDevice(&current) != hipSuccess)
                    return HIPBLAS_STATUS_INTERNAL_ERROR;
                if(current != device)
                {
                    diagnostics.message = "Compile on the requested HIP device";
                    return HIPBLAS_STATUS_INVALID_VALUE;
                }
                hipblaslt_jit::DeviceTarget target;
                hipblaslt_jit::Jit::Outcome outcome;
                auto status = hipblaslt_jit::DeviceTarget::make(device, target);
                if(status.ok())
                    outcome = jit->generate(*operation, target, 1, workspaceLimit, {});
                if(status.ok() && outcome.bundles.empty())
                {
                    if(outcome.failures.empty())
                        return HIPBLAS_STATUS_INTERNAL_ERROR;
                    status = std::move(outcome.failures.front());
                }
                if(!status.ok())
                {
                    diagnostics.message = std::move(status.message);
                    return detail::toHipStatus(status.code);
                }
                auto bundle         = std::move(outcome.bundles.front());
                diagnostics.message = std::move(outcome.summary);
                if(bundle->operationKind() != operation->kind())
                {
                    diagnostics.message = "Bundle does not implement the requested operation";
                    return HIPBLAS_STATUS_NOT_SUPPORTED;
                }
                std::shared_ptr<const detail::CompiledSolution> compiled;
                const auto supported = detail::compiledFromBundle(std::move(jit),
                                                                  std::move(operation),
                                                                  std::move(target),
                                                                  std::move(bundle),
                                                                  workspaceLimit,
                                                                  compiled,
                                                                  diagnostics);
                if(supported != HIPBLAS_STATUS_SUCCESS)
                    return supported;
                solution = detail::SolutionAccess::make(std::move(compiled));
                return HIPBLAS_STATUS_SUCCESS;
            }
            catch(const std::bad_alloc&)
            {
                return HIPBLAS_STATUS_ALLOC_FAILED;
            }
            catch(const std::exception& e)
            {
                diagnostics.message = e.what();
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            }
            catch(...)
            {
                return HIPBLAS_STATUS_INTERNAL_ERROR;
            }
        }

        hipblasStatus_t getGemmAlgo(const Solution&                   solution,
                                    hipblasLtMatmulHeuristicResult_t& result,
                                    Diagnostics&                      diagnostics)
        {
            result       = {};
            result.state = HIPBLAS_STATUS_INVALID_VALUE;
            diagnostics  = {};
            auto finish  = [&](hipblasStatus_t status) {
                result.state = status;
                return status;
            };
            try
            {
                auto compiled = detail::SolutionAccess::get(solution);
                if(!compiled)
                    return finish(HIPBLAS_STATUS_INVALID_VALUE);
                diagnostics.backend = compiled->jit->components().backend->info().name;
                int device          = -1;
                if(compiled->process != detail::processId() || hipGetDevice(&device) != hipSuccess
                   || device != compiled->target.device)
                    return finish(HIPBLAS_STATUS_INVALID_VALUE);
                if(compiled->bundle->operationKind() != detail::GemmRequest::operation)
                    return finish(HIPBLAS_STATUS_NOT_SUPPORTED);
                const auto            token = detail::registerBundle(compiled);
                const int32_t         index = compiled->bundle->solutionIndex();
                rocblaslt_matmul_algo algo{};
                std::memcpy(algo.data, &index, sizeof(index));
                std::memcpy(algo.data + sizeof(int32_t),
                            &experimental::detail::jitAlgoTag,
                            sizeof(experimental::detail::jitAlgoTag));
                std::memcpy(algo.data_pad, &token, sizeof(algo.data_pad));
                algo.max_workspace_bytes = compiled->workspaceLimit;
                std::memcpy(&result.algo, &algo, sizeof(algo));
                result.workspaceSize = compiled->workspaceBytes;
                return finish(HIPBLAS_STATUS_SUCCESS);
            }
            catch(const std::bad_alloc&)
            {
                return finish(HIPBLAS_STATUS_ALLOC_FAILED);
            }
            catch(...)
            {
                return finish(HIPBLAS_STATUS_INTERNAL_ERROR);
            }
        }
    }
}

namespace hipblaslt_jit
{
    namespace
    {
        using GemmRequest = hipblaslt_ext::experimental::jit::detail::GemmRequest;
        using Compiled    = hipblaslt_ext::experimental::jit::detail::CompiledSolution;

        // Device, workspace limit and the GEMM problem the request already carries.
        // Buffer addresses are not part of it: two queries of one problem share a build.
        std::string requestKey(int device, size_t workspace, const GemmRequest& request)
        {
            const auto& problem = request.problem;
            Fnv1a       hash;
            auto        add = [&](auto value) { hash.add(std::to_string(value)); };
            add(device);
            add(workspace);
            add(problem.m);
            add(problem.n);
            add(problem.k);
            add(problem.batch_count);
            add(static_cast<int>(problem.trans_a));
            add(static_cast<int>(problem.trans_b));
            add(static_cast<int>(problem.a_type));
            add(static_cast<int>(problem.b_type));
            add(static_cast<int>(problem.c_type));
            add(static_cast<int>(problem.d_type));
            add(static_cast<int>(problem.compute_type));
            add(static_cast<int>(problem.scale_type));
            add(static_cast<int>(problem.epilogue));
            add(static_cast<int>(problem.bias_type));
            add(static_cast<int>(problem.aux_type));
            add(static_cast<int>(problem.scaleAType));
            add(static_cast<int>(problem.scaleBType));
            add(problem.row_stride_a);
            add(problem.col_stride_a);
            add(problem.batch_stride_a);
            add(problem.row_stride_b);
            add(problem.col_stride_b);
            add(problem.batch_stride_b);
            add(problem.row_stride_c);
            add(problem.col_stride_c);
            add(problem.batch_stride_c);
            add(problem.row_stride_d);
            add(problem.col_stride_d);
            add(problem.batch_stride_d);
            add(problem.strided_batch);
            add(problem.grouped_gemm);
            add(problem.gradient);
            add(problem.swizzleA);
            add(problem.swizzleB);
            add(problem.act0);
            add(problem.act1);
            add(problem.streamk_tile_scheduling_ext);
            add(problem.sm_count_target);
            add(problem.uniform_summation_order);
            add(problem.bias != nullptr);
            add(problem.scaleA != nullptr);
            add(problem.scaleB != nullptr);
            add(problem.scaleC != nullptr);
            add(problem.scaleD != nullptr);
            add(problem.scaleE != nullptr);
            add(problem.scaleAlphaVec != nullptr);
            add(problem.amaxD != nullptr);
            add(static_cast<int>(problem.batchMode));
            add(problem.bias_stride);
            hash.add(std::string_view(reinterpret_cast<const char*>(request.alpha.data()),
                                      request.alpha.size()));
            hash.add(std::string_view(reinterpret_cast<const char*>(request.beta.data()),
                                      request.beta.size()));
            return hash.hex();
        }

        struct CacheEntry
        {
            std::vector<std::shared_ptr<const Compiled>> solutions;
            bool                                         complete = false;
        };

        // True when one compiled solution's kernel is name. An empty name matches nothing.
        bool hasKernel(const std::vector<std::shared_ptr<const Compiled>>& solutions,
                       const std::string&                                  name)
        {
            return !name.empty()
                   && std::any_of(solutions.begin(),
                                  solutions.end(),
                                  [&](const std::shared_ptr<const Compiled>& solution) {
                                      return solution->bundle->kernelNames() == name;
                                  });
        }

        // The process-wide Jit: replay backend, comgr builder, TensileLite loader.
        // Null when HIPBLASLT_JIT_TEST_REPLAY names no bundle.
        std::shared_ptr<const Jit> replayProcess()
        {
            static std::once_flag                 once;
            static std::shared_ptr<const Jit>     jit;
            std::call_once(once, [] {
                const char* value = rocblaslt_secure_getenv("HIPBLASLT_JIT_TEST_REPLAY");
                if(!value || !*value)
                    return;
                hipblaslt_ext::experimental::jit::replay::Options options;
                std::istringstream                                paths(value);
                std::string                                       path;
                while(paths >> path)
                    options.replay.push_back(std::move(path));
                if(options.replay.empty())
                    return;
                hipblaslt_ext::experimental::jit::Backend     backend;
                hipblaslt_ext::experimental::jit::Diagnostics diagnostics;
                const auto status = hipblaslt_ext::experimental::jit::replay::createBackend(
                    options, backend, diagnostics);
                if(status != HIPBLAS_STATUS_SUCCESS)
                {
                    std::cerr << "hipblaslt warning: HIPBLASLT_JIT_TEST_REPLAY could not be read: "
                              << diagnostics.message << std::endl;
                    return;
                }
                jit = hipblaslt_ext::experimental::jit::detail::BackendAccess::get(backend);
            });
            return jit;
        }
    }

    bool jitHeuristicLibrary()
    {
        return static_cast<bool>(replayProcess());
    }

    void warnJitHeuristicUnavailable()
    {
        static std::once_flag once;
        std::call_once(once, [] {
            const char* value = rocblaslt_secure_getenv("HIPBLASLT_JIT");
            std::cerr << "hipblaslt warning: HIPBLASLT_JIT=" << (value ? value : "");
            if(mode() == Mode::Forced)
                std::cerr << " has no JIT library; heuristic queries return no solutions"
                          << std::endl;
            else
                std::cerr << " has no JIT library; heuristic queries are unchanged" << std::endl;
        });
    }

    int appendJitHeuristic(rocblaslt_handle                   handle,
                           const RocblasltContractionProblem& problem,
                           size_t                             workspaceLimit,
                           const std::vector<std::string>&    excludeKernels,
                           rocblaslt_matmul_heuristic_result* results,
                           int                                room)
    {
        if(room <= 0 || results == nullptr || handle == nullptr)
            return 0;
        const auto jit = replayProcess();
        if(!jit)
        {
            warnJitHeuristicUnavailable();
            return 0;
        }
        try
        {
            int current = -1;
            if(hipGetDevice(&current) != hipSuccess || current != handle->device)
                return 0;
            std::shared_ptr<const GemmRequest> owned = std::make_shared<GemmRequest>(problem);
            DeviceTarget target;
            if(!DeviceTarget::make(handle->device, target).ok() || !target.hardware)
                return 0;

            static auto*                    cache = new std::map<std::string, CacheEntry>;
            static std::mutex               guard;
            const auto                      key = requestKey(handle->device, workspaceLimit, *owned);
            std::lock_guard<std::mutex>     lock(guard);
            auto&                           entry = (*cache)[key];
            std::vector<std::shared_ptr<const Compiled>> chosen;
            for(const auto& solution : entry.solutions)
            {
                const auto name = solution->bundle->kernelNames();
                if(excludedKernel(excludeKernels, name) || hasKernel(chosen, name))
                    continue;
                chosen.push_back(solution);
                if(static_cast<int>(chosen.size()) == room)
                    break;
            }
            if(static_cast<int>(chosen.size()) < room && !entry.complete)
            {
                bool callerExcludesOnlyCached = true;
                for(const auto& name : excludeKernels)
                {
                    if(!hasKernel(entry.solutions, name))
                        callerExcludesOnlyCached = false;
                }
                std::vector<std::string> exclude = excludeKernels;
                for(const auto& solution : entry.solutions)
                {
                    auto name = solution->bundle->kernelNames();
                    if(!name.empty() && !excludedKernel(exclude, name))
                        exclude.push_back(std::move(name));
                }
                const auto need    = static_cast<size_t>(room) - chosen.size();
                auto       outcome = jit->generate(*owned, target, need, workspaceLimit, exclude);
                namespace detail = hipblaslt_ext::experimental::jit::detail;
                for(auto& bundle : outcome.bundles)
                {
                    const auto name = bundle->kernelNames();
                    if(excludedKernel(excludeKernels, name) || hasKernel(chosen, name)
                       || hasKernel(entry.solutions, name))
                        continue;
                    hipblaslt_ext::experimental::jit::Diagnostics diagnostics;
                    std::shared_ptr<const Compiled>               compiled;
                    if(detail::compiledFromBundle(jit,
                                                  owned,
                                                  target,
                                                  std::move(bundle),
                                                  workspaceLimit,
                                                  compiled,
                                                  diagnostics)
                       != HIPBLAS_STATUS_SUCCESS)
                        continue;
                    entry.solutions.push_back(compiled);
                    if(static_cast<int>(chosen.size()) < room)
                        chosen.push_back(compiled);
                }
                if(outcome.bundles.size() < need && callerExcludesOnlyCached)
                    entry.complete = true;
                for(const auto& failure : outcome.failures)
                {
                    if(failure.code == Status::Code::NotSupported
                       || failure.code == Status::Code::TargetMismatch)
                        continue;
                    std::cerr << "hipblaslt warning: JIT heuristic " << failure.message << std::endl;
                    break;
                }
            }

            namespace detail = hipblaslt_ext::experimental::jit::detail;
            namespace jitapi = hipblaslt_ext::experimental::jit;
            int written = 0;
            for(const auto& compiled : chosen)
            {
                jitapi::Solution  solution = detail::SolutionAccess::make(compiled);
                jitapi::Diagnostics diagnostics;
                hipblasLtMatmulHeuristicResult_t hipResult{};
                if(jitapi::getGemmAlgo(solution, hipResult, diagnostics) != HIPBLAS_STATUS_SUCCESS)
                    continue;
                auto& result = results[written];
                std::memset(&result, 0, sizeof(result));
                static_assert(sizeof(hipResult.algo) == sizeof(result.algo),
                              "JIT heuristic results share the matmul algorithm layout");
                std::memcpy(&result.algo, &hipResult.algo, sizeof(result.algo));
                result.workspaceSize = hipResult.workspaceSize;
                result.state         = rocblaslt_status_success;
                result.wavesCount    = 1.0f;
                ++written;
            }
            return written;
        }
        catch(const std::exception& error)
        {
            std::cerr << "hipblaslt warning: JIT heuristic " << error.what() << std::endl;
            return 0;
        }
        catch(...)
        {
            return 0;
        }
    }
}
