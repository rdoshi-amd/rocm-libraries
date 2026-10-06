// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-component.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

// Jit against stages that break its contract in ways the library's real
// stages cannot; jit_failure_test covers the real stages' failures.
namespace hj  = hipblaslt_jit;
namespace abi = hipblaslt_ext::experimental::jit::detail;
namespace fs  = std::filesystem;
using Code    = hj::Status::Code;
using hj::Stage;

namespace
{
    void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }

    // Jit is operation-agnostic; none of these stages sees a GEMM.
    struct ProbeRequest final : abi::OperationRequest
    {
        std::string_view kind() const noexcept override
        {
            return "test.jit.probe.v1";
        }
    };

    struct ProbeBundle final : abi::KernelBundle
    {
        std::string kernel;
        explicit ProbeBundle(std::string name)
            : kernel(std::move(name))
        {
        }
        std::string_view operationKind() const noexcept override
        {
            return "test.jit.probe.v1";
        }
        std::string name() const override
        {
            return kernel;
        }
        std::string kernelNames() const override
        {
            return kernel;
        }
        hipblasStatus_t support(const abi::OperationRequest&,
                                size_t,
                                size_t&,
                                hipblaslt_ext::experimental::jit::Diagnostics&) const override
        {
            return HIPBLAS_STATUS_NOT_SUPPORTED;
        }
    };

    // Returns every kernel whatever the requested count, and statuses with
    // whatever stage the test sets.
    struct Backend final : hj::Backend
    {
        hj::BackendInfo          information{"fake-backend", "Fake"};
        std::vector<std::string> kernels;
        hj::Status               result{Code::Success, Stage::Generate, "generated"};
        bool                     throws = false, allocationFails = false;
        mutable size_t           calls = 0;

        explicit Backend(std::vector<std::string> k)
            : kernels(std::move(k))
        {
        }
        const hj::BackendInfo& info() const noexcept override
        {
            return information;
        }
        hj::Status generate(const hj::GenerationRequest&,
                            std::vector<hj::GeneratedSolution>& solutions) const override
        {
            ++calls;
            if(allocationFails)
                throw std::bad_alloc();
            if(throws)
                throw std::runtime_error("deliberate generator exception");
            if(!result.ok())
                return result;
            for(const auto& kernel : kernels)
                solutions.push_back(
                    {{1, 2, 3}, kernel, {{hj::BuildUnit::Role::Main, kernel, {4}}}});
            return result;
        }
    };

    struct Builder final : hj::CodeObjectBuilder
    {
        mutable std::vector<std::string> built; // in build order
        hj::Status                       build(const hj::GeneratedSolution& solution,
                                               const hj::BuildRequest&,
                                               hj::BuiltSolution& result) const override
        {
            built.push_back(solution.kernelName);
            result.generated    = solution;
            result.object.bytes = solution.units.at(0).bytes;
            return {};
        }
    };

    struct Loader final : hj::SolutionLoader
    {
        std::set<std::string> throwing, empty;
        hj::Status            support(const hj::BuiltSolution& built,
                                      const hj::OperationRequest&,
                                      const hj::DeviceTarget&,
                                      size_t) const override
        {
            if(throwing.count(built.generated.kernelName))
                throw std::runtime_error("support threw " + built.generated.kernelName);
            return {};
        }
        hj::Status load(const hj::BuiltSolution& built,
                        const hj::OperationRequest&,
                        const hj::DeviceTarget&,
                        size_t,
                        std::shared_ptr<const hj::KernelBundle>& bundle) const override
        {
            if(!empty.count(built.generated.kernelName))
                bundle = std::make_shared<ProbeBundle>(built.generated.kernelName);
            return {};
        }
    };

    // Throws, or returns one index fewer than it was given solutions.
    struct Store final : hj::SolutionStore
    {
        bool           throws = false;
        mutable size_t calls  = 0;
        hj::Status     lookup(const hj::OperationRequest&,
                              const hj::DeviceTarget&,
                              size_t,
                              size_t,
                              const std::vector<std::string>&,
                              std::vector<int32_t>& indices) const override
        {
            indices.clear();
            return {};
        }
        hj::Status publish(const hj::OperationRequest&,
                           const hj::DeviceTarget&,
                           const std::vector<hj::BuiltSolution>& solutions,
                           std::vector<int32_t>&                 indices) const override
        {
            ++calls;
            if(throws)
                throw std::runtime_error("deliberate store exception");
            for(size_t i = 1; i < solutions.size(); ++i)
                indices.push_back(100 + static_cast<int32_t>(i));
            return {};
        }
    };

    // A Jit over fakes; tests adjust the fakes before calling run().
    struct Fixture
    {
        std::shared_ptr<Backend> backend;
        std::shared_ptr<Builder> builder = std::make_shared<Builder>();
        std::shared_ptr<Loader>  loader  = std::make_shared<Loader>();
        std::shared_ptr<Store>   store;
        ProbeRequest             request;
        hj::DeviceTarget         target;

        explicit Fixture(std::vector<std::string> kernels)
            : backend(std::make_shared<Backend>(std::move(kernels)))
        {
            target.device = 0;
            target.isa    = "gfx950";
        }
        hj::Jit::Outcome run(size_t count = 1) const
        {
            return hj::Jit({backend, builder, loader, store})
                .generate(request, target, count, 4096, {});
        }
    };

    std::vector<std::string> names(const hj::Jit::Outcome& outcome)
    {
        std::vector<std::string> result;
        for(const auto& bundle : outcome.bundles)
            result.push_back(bundle->name());
        return result;
    }

    void failure(const hj::Jit::Outcome& outcome,
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

    void construction()
    {
        auto backend  = std::make_shared<Backend>(std::vector<std::string>{});
        auto builder  = std::make_shared<Builder>();
        auto loader   = std::make_shared<Loader>();
        auto rejected = [](hj::Jit::Components components) {
            try
            {
                hj::Jit jit(std::move(components));
            }
            catch(const std::invalid_argument&)
            {
                return true;
            }
            return false;
        };
        require(rejected({nullptr, builder, loader}) && rejected({backend, nullptr, loader})
                    && rejected({backend, builder, nullptr}),
                "Jit accepted a missing backend, builder or loader");
        hj::Jit jit({backend, builder, loader});
        require(jit.components().backend == backend, "Jit did not keep its components");
        std::cout << "PASS Jit construction validates components\n";
    }

    void counts()
    {
        {
            Fixture f({"a"});
            require(f.run(0).bundles.empty() && f.backend->calls == 0,
                    "A zero count reached the backend");
        }
        {
            Fixture    f({"a", "b", "c"});
            const auto outcome = f.run(2);
            require(names(outcome) == std::vector<std::string>{"a", "b"}
                        && f.builder->built == std::vector<std::string>{"a", "b"},
                    "Jit built or returned more solutions than requested");
        }
        std::cout << "PASS a zero count generates nothing; extra solutions are not built\n";
    }

    void failures()
    {
        // The stage is the one Jit was running, whatever the backend reported.
        const std::pair<hj::Status, Stage> generated[]
            = {{{Code::TargetMismatch, Stage::Build, "other target"}, Stage::Configure},
               {{Code::NotSupported, Stage::Build, "not mine"}, Stage::Generate},
               {{Code::Failed, Stage::Load, "generator failed"}, Stage::Generate}};
        for(const auto& [status, stage] : generated)
        {
            Fixture f({"a"});
            f.backend->result  = status;
            const auto outcome = f.run();
            failure(outcome, status.code, stage, status.message, "Generate");
            require(f.builder->built.empty(), "Jit built after a failed generation");
        }
        {
            Fixture f({"a"});
            f.backend->throws = true;
            failure(f.run(), Code::Failed, Stage::Generate, "deliberate generator", "Throw");
        }
        {
            Fixture f({"a", "b"});
            f.loader->throwing = {"a"};
            const auto outcome = f.run();
            failure(outcome, Code::Failed, Stage::Support, "support threw a", "Support");
            require(names(outcome) == std::vector<std::string>{"b"},
                    "A throwing support check dropped the other solutions");
        }
        {
            Fixture f({"a"});
            f.loader->empty    = {"a"};
            const auto outcome = f.run();
            failure(outcome, Code::Failed, Stage::Load, "Loader returned no bundle", "Load");
            require(outcome.bundles.empty(), "A missing bundle was returned");
        }
        {
            Fixture f({"a"});
            f.backend->allocationFails = true;
            bool thrown                = false;
            try
            {
                f.run();
            }
            catch(const std::bad_alloc&)
            {
                thrown = true;
            }
            require(thrown, "Allocation failure did not propagate");
        }
        std::cout << "PASS misreported stages, exceptions and missing bundles fail at the stage "
                     "Jit was running\n";
    }

    void stores()
    {
        for(const bool throws : {false, true})
        {
            Fixture f({"a", "b"});
            f.store            = std::make_shared<Store>();
            f.store->throws    = throws;
            const auto outcome = f.run(2);
            failure(outcome,
                    Code::Failed,
                    Stage::Publish,
                    throws ? "deliberate store exception" : "returned 1 indices for 2 solutions",
                    throws ? "Throwing store" : "Short publish");
            require(f.store->calls == 1 && outcome.indices.empty()
                        && names(outcome) == std::vector<std::string>{"a", "b"},
                    "A failed publish did not fall back to loading");
        }
        {
            Fixture f({"a"});
            f.store            = std::make_shared<Store>();
            f.loader->throwing = {"a"};
            const auto outcome = f.run();
            require(f.store->calls == 0 && outcome.bundles.empty(),
                    "Nothing was supported, yet Jit published or loaded");
        }
        std::cout << "PASS a store that throws or returns too few indices fails at publish, "
                     "and the solutions load instead\n";
    }
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " SCRATCH_PARENT\n";
        return 2;
    }
    try
    {
        const fs::path parent = fs::absolute(argv[1]);
        fs::remove_all(parent);
        fs::create_directories(parent);
#ifdef _WIN32
        require(_putenv_s("TMP", parent.string().c_str()) == 0, "Cannot set TMP");
#else
        require(setenv("TMPDIR", parent.c_str(), 1) == 0, "Cannot set TMPDIR");
#endif
        construction();
        counts();
        failures();
        stores();
        require(fs::is_empty(parent), "Jit left a scratch directory");
        std::cout << "ALL JIT COMPONENT CHECKS PASSED\n";
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
