// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-heuristic.hpp"
#include "hipblaslt-jit-prediction.hpp"
#include "hipblaslt-jit-replay.hpp"
#include "rocblaslt_secure_env.hpp"
#include <sstream>
#include <string_view>

// The provider of HIPBLASLT_JIT_TESTING builds: heuristic queries replay, after
// an Origami prediction, the bundles HIPBLASLT_JIT_TEST_REPLAY lists through
// the replay backend. HIPBLASLT_JIT_TEST_FAULT injects a replay fault;
// the record fault appends each request to HIPBLASLT_JIT_TEST_RECORD.
namespace hipblaslt_jit
{
    Status makeDefaultProcessBackend(ProcessBackend& made)
    {
        namespace replay     = hipblaslt_ext::experimental::jit::replay;
        const auto configure = [](std::string message) {
            return Status{Status::Code::Failed, Stage::Configure, std::move(message)};
        };
        const auto variable = [](const char* name) {
            const char* value = rocblaslt_secure_getenv(name);
            return std::string(value ? value : "");
        };
        replay::Options options;
#ifdef _WIN32
        constexpr char separator = ';';
#else
        constexpr char separator = ':';
#endif
        std::istringstream paths(variable("HIPBLASLT_JIT_TEST_REPLAY"));
        for(std::string path; std::getline(paths, path, separator);)
            if(!path.empty())
                options.replay.push_back(path);
        if(options.replay.empty())
            return configure(
                "The JIT test backend has no bundle to replay; set HIPBLASLT_JIT_TEST_REPLAY");

        const auto fault = variable("HIPBLASLT_JIT_TEST_FAULT");
        using Fault      = replay::Options::Fault;
        const std::pair<std::string_view, Fault> faults[] = {{"", Fault::None},
                                                             {"generate", Fault::Generate},
                                                             {"build", Fault::Build},
                                                             {"record", Fault::Record},
                                                             {"trap", Fault::Trap}};
        bool known = false;
        for(const auto& [name, value] : faults)
            if(fault == name)
            {
                options.fault = value;
                known         = true;
            }
        if(!known)
            return configure("HIPBLASLT_JIT_TEST_FAULT=" + fault
                             + " is not one of generate, build, record or trap");
        options.record = variable("HIPBLASLT_JIT_TEST_RECORD");
        if(options.fault == Fault::Record && options.record.empty())
            return configure(
                "HIPBLASLT_JIT_TEST_FAULT=record requires a file in HIPBLASLT_JIT_TEST_RECORD");

        made.predictor = makeOrigamiPredictor();
        made.knowledge = makeCatalogKnowledge();
        // The replay backend replays bundles; it cannot carry a tuned set to a generator.
        options.contracts = {"origami.gemm.dp.v1"};
        made.backend      = replay::makeBackend(options);
        return {};
    }
}
