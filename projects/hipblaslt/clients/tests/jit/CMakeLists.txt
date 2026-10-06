# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# JIT tests, staged in clients/staging. CTest labels them jit-cpu, which needs
# no GPU, or jit-gpu. Each test empties its own scratch directory.
enable_testing()
find_package(Threads REQUIRED)

set(hipblaslt_jit_source "${PROJECT_SOURCE_DIR}/library/src/amd_detail")
set(hipblaslt_jit_scratch "${CMAKE_CURRENT_BINARY_DIR}/scratch")
# The committed bundles; data/README.md lists how they were generated.
set(hipblaslt_jit_data "${CMAKE_CURRENT_SOURCE_DIR}/data")
set(hipblaslt_jit_bundle "${hipblaslt_jit_data}/gfx950/plain")
set(hipblaslt_jit_splitk "${hipblaslt_jit_data}/gfx950/splitk")
string(REGEX MATCH "(^|;)gfx950(:[^;]*)?(;|$)" hipblaslt_jit_gfx950 "${GPU_TARGETS}")

# hipblaslt_jit_test(name label target args...) stages target and runs it.
function(hipblaslt_jit_test name label test_target)
    target_compile_features(${test_target} PRIVATE cxx_std_17)
    set_target_properties(${test_target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${test_target}> ${ARGN})
    set_tests_properties(${name} PROPERTIES LABELS ${label})
endfunction()

# hipblaslt_jit_debug(target) adds HIPBLASLT_JIT_DEBUG to a target that compiles
# the library's JIT sources directly.
function(hipblaslt_jit_debug test_target)
    target_sources(${test_target} PRIVATE
        "${hipblaslt_jit_source}/hipblaslt-jit-debug.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-mode.cpp")
    target_include_directories(${test_target} PRIVATE
        "${hipblaslt_jit_source}/rocblaslt/src/include")
    target_compile_definitions(${test_target} PRIVATE HIPBLASLT_ENABLE_JIT)
    target_link_libraries(${test_target} PRIVATE Threads::Threads)
endfunction()

# hipblaslt_jit_python_test(name label script args...) runs a script here.
function(hipblaslt_jit_python_test name label script)
    add_test(NAME ${name}
        COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/${script}" ${ARGN})
    set_tests_properties(${name} PROPERTIES
        LABELS ${label} ENVIRONMENT PYTHONDONTWRITEBYTECODE=1)
endfunction()

# Public API only, so it also builds without JIT; the heuristic scripts run it.
add_executable(hipblaslt-jit-heuristic-test heuristic_test.cpp)
target_compile_features(hipblaslt-jit-heuristic-test PRIVATE cxx_std_17)
set_target_properties(hipblaslt-jit-heuristic-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
target_link_libraries(hipblaslt-jit-heuristic-test
    PRIVATE roc::hipblaslt hip::device Threads::Threads)

add_executable(hipblaslt-jit-source-bundle-test source_bundle_test.cpp)
target_include_directories(hipblaslt-jit-source-bundle-test PRIVATE "${hipblaslt_jit_source}")
hipblaslt_jit_test(jit-source-bundle jit-cpu hipblaslt-jit-source-bundle-test
    "${hipblaslt_jit_scratch}/source-bundle")

if(NOT HIPBLASLT_ENABLE_JIT)
    add_executable(hipblaslt-jit-disabled-test disabled_api_test.cpp)
    target_link_libraries(hipblaslt-jit-disabled-test PRIVATE roc::hipblaslt hip::host)
    hipblaslt_jit_test(jit-disabled jit-cpu hipblaslt-jit-disabled-test)

    # HIPBLASLT_JIT is ignored with one warning.
    hipblaslt_jit_python_test(jit-heuristic-jit-off jit-gpu test_heuristic.py
        $<TARGET_FILE:hipblaslt-jit-heuristic-test> jit-off
        "${hipblaslt_jit_scratch}/heuristic-jit-off")
    if(HIPBLASLT_ENABLE_CLIENT)
        hipblaslt_jit_python_test(jit-bench-smoke-jit-off jit-gpu test_bench_smoke.py
            $<TARGET_FILE:hipblaslt-bench> "${hipblaslt_jit_scratch}/bench-smoke-jit-off"
            --jit-off)
    endif()
    return()
endif()

# Compiles the comgr builder directly and builds the committed bundle for its
# target. Needs no GPU, so every architecture runs it.
add_executable(hipblaslt-jit-builder-test
    builder_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp")
target_include_directories(hipblaslt-jit-builder-test PRIVATE
    "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
target_link_libraries(hipblaslt-jit-builder-test PRIVATE
    roc::hipblaslt amd_comgr hip::host)
hipblaslt_jit_debug(hipblaslt-jit-builder-test)
hipblaslt_jit_test(jit-builder jit-cpu hipblaslt-jit-builder-test
    "${hipblaslt_jit_bundle}" "${hipblaslt_jit_scratch}/builder")

# Drives the Jit stages with fakes that break the stage contracts.
add_executable(hipblaslt-jit-component-test
    jit_component_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-component.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-catalog-knowledge.cpp")
target_include_directories(hipblaslt-jit-component-test PRIVATE "${hipblaslt_jit_source}")
target_link_libraries(hipblaslt-jit-component-test PRIVATE roc::hipblaslt hip::host)
hipblaslt_jit_debug(hipblaslt-jit-component-test)
hipblaslt_jit_test(jit-component jit-cpu hipblaslt-jit-component-test
    "${hipblaslt_jit_scratch}/component")

# Compiles HIPBLASLT_JIT_DEBUG parsing and the line sink directly.
add_executable(hipblaslt-jit-debug-test jit_debug_test.cpp)
target_include_directories(hipblaslt-jit-debug-test PRIVATE "${hipblaslt_jit_source}")
hipblaslt_jit_debug(hipblaslt-jit-debug-test)
hipblaslt_jit_test(jit-debug jit-cpu hipblaslt-jit-debug-test "${hipblaslt_jit_scratch}/debug")

# Builds and inspects code objects with comgr; --gpu also launches them on the
# current device, which can run the gfx950 bundle only in a gfx950 build.
add_executable(hipblaslt-jit-code-object-test
    code_object_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp")
target_include_directories(hipblaslt-jit-code-object-test PRIVATE
    "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
target_link_libraries(hipblaslt-jit-code-object-test PRIVATE amd_comgr hip::host Threads::Threads)
hipblaslt_jit_test(jit-code-object jit-cpu hipblaslt-jit-code-object-test
    --target gfx950 --out "${hipblaslt_jit_scratch}/code-object" --bundle "${hipblaslt_jit_splitk}")
set(gpu_bundle)
if(hipblaslt_jit_gfx950)
    set(gpu_bundle --bundle "${hipblaslt_jit_splitk}")
endif()
hipblaslt_jit_test(jit-code-object-gpu jit-gpu hipblaslt-jit-code-object-test
    --gpu --out "${hipblaslt_jit_scratch}/code-object-gpu" ${gpu_bundle})

# The library entries are MsgPack, which a YAML build cannot read. TensileLite
# queries the current device when it reads one, so the library and freshness
# tests need a GPU.
if(NOT HIPBLASLT_ENABLE_YAML)
    # Compiles the JIT solution library directly.
    add_executable(hipblaslt-jit-library-test
        jit_library_test.cpp
        "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-fs.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-library.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-msgpack.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-problem-type.cpp")
    target_include_directories(hipblaslt-jit-library-test PRIVATE
        "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
    target_compile_definitions(hipblaslt-jit-library-test PRIVATE
        HIPBLASLT_ENABLE_JIT HIPBLASLT_JIT_LIBRARY_TESTING)
    target_link_libraries(hipblaslt-jit-library-test PRIVATE
        roc::hipblaslt roc::tensilelite-host ${hipblaslt_msgpack_target} amd_comgr
        hip::device Threads::Threads)
    hipblaslt_jit_debug(hipblaslt-jit-library-test)
    hipblaslt_jit_test(jit-library jit-gpu hipblaslt-jit-library-test
        "${hipblaslt_jit_splitk}" "${hipblaslt_jit_scratch}/library")
    hipblaslt_jit_test(jit-library-concurrency jit-gpu hipblaslt-jit-library-test
        "${hipblaslt_jit_splitk}" "${hipblaslt_jit_scratch}/library-concurrency"
        --writers 8 --per-writer 4)

    # Fails when the committed bundles no longer match the layout versions in
    # GlobalParameters.py, the builder's code-object version or the host library.
    set(parameters "${PROJECT_SOURCE_DIR}/tensilelite/Tensile/Common/GlobalParameters.py")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${parameters}")
    file(READ "${parameters}" parameters)
    foreach(key KernArgsVersion PersistentLoopArgsVersion)
        if(NOT parameters MATCHES "\"${key}\": *([0-9]+)")
            message(FATAL_ERROR "No ${key} in GlobalParameters.py")
        endif()
        set(${key} ${CMAKE_MATCH_1})
    endforeach()
    add_executable(hipblaslt-jit-bundle-freshness-test
        jit_bundle_freshness_test.cpp
        "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
        "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp")
    target_include_directories(hipblaslt-jit-bundle-freshness-test PRIVATE
        "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
    target_compile_definitions(hipblaslt-jit-bundle-freshness-test PRIVATE
        HIPBLASLT_JIT_KERNARGS_VERSION=${KernArgsVersion}
        HIPBLASLT_JIT_PERSISTENT_LOOP_ARGS_VERSION=${PersistentLoopArgsVersion})
    target_link_libraries(hipblaslt-jit-bundle-freshness-test PRIVATE
        roc::hipblaslt roc::tensilelite-host amd_comgr hip::device Threads::Threads)
    hipblaslt_jit_debug(hipblaslt-jit-bundle-freshness-test)
    hipblaslt_jit_test(jit-bundle-freshness jit-gpu hipblaslt-jit-bundle-freshness-test
        "${hipblaslt_jit_data}" "${hipblaslt_jit_scratch}/bundle-freshness")
endif()

# The bundles hold gfx950 code and MsgPack library entries, which a YAML build
# cannot read.
if(NOT hipblaslt_jit_gfx950 OR HIPBLASLT_ENABLE_YAML)
    return()
endif()

# Compiles the comgr builder and the Tensile loader directly.
add_executable(hipblaslt-jit-loader-test
    loader_test.cpp
    "${hipblaslt_jit_source}/hipblaslt-jit-builder.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-code-object.cpp"
    "${hipblaslt_jit_source}/hipblaslt-jit-loader.cpp")
target_include_directories(hipblaslt-jit-loader-test PRIVATE
    "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/src/include")
target_link_libraries(hipblaslt-jit-loader-test PRIVATE
    roc::hipblaslt roc::tensilelite-host amd_comgr hip::device)
hipblaslt_jit_debug(hipblaslt-jit-loader-test)
hipblaslt_jit_test(jit-loader jit-gpu hipblaslt-jit-loader-test
    "${hipblaslt_jit_bundle}" "${hipblaslt_jit_scratch}/loader")

if(HIPBLASLT_JIT_TESTING)
    add_executable(hipblaslt-jit-end-to-end-test jit_end_to_end_test.cpp)
    target_include_directories(hipblaslt-jit-end-to-end-test PRIVATE "${hipblaslt_jit_source}")
    target_link_libraries(hipblaslt-jit-end-to-end-test PRIVATE roc::hipblaslt hip::device)
    hipblaslt_jit_test(jit-end-to-end jit-gpu hipblaslt-jit-end-to-end-test
        "${hipblaslt_jit_bundle}")
    hipblaslt_jit_test(jit-end-to-end-splitk jit-gpu hipblaslt-jit-end-to-end-test
        "${hipblaslt_jit_splitk}")

    # Failures of the real stages, on damaged copies of the bundles.
    add_executable(hipblaslt-jit-failure-test
        jit_failure_test.cpp
        "${hipblaslt_jit_source}/hipblaslt-jit-component.cpp")
    target_include_directories(hipblaslt-jit-failure-test PRIVATE "${hipblaslt_jit_source}")
    target_link_libraries(hipblaslt-jit-failure-test PRIVATE roc::hipblaslt hip::device)
    hipblaslt_jit_debug(hipblaslt-jit-failure-test)
    hipblaslt_jit_test(jit-failure jit-gpu hipblaslt-jit-failure-test
        "${hipblaslt_jit_bundle}" "${hipblaslt_jit_splitk}" "${hipblaslt_jit_scratch}/failure")

    # The C and C++ GEMM APIs on replayed solutions; each case names its bundle.
    add_executable(hipblaslt-jit-api-test jit_api_test.cpp)
    target_include_directories(hipblaslt-jit-api-test PRIVATE "${hipblaslt_jit_source}")
    target_link_libraries(hipblaslt-jit-api-test PRIVATE roc::hipblaslt hip::device)
    foreach(case "splitk;splitk;--k;512"
                 "streamk;streamk;--k;4096;--workspace-fallback;1"
                 "amax;amax;--amax;1"
                 "alpha-zero;amax;--amax;1;--alpha-zero;1")
        list(POP_FRONT case name bundle)
        hipblaslt_jit_test(jit-api-${name} jit-gpu hipblaslt-jit-api-test
            --replay "${hipblaslt_jit_data}/gfx950/${bundle}" ${case})
    endforeach()
    set_tests_properties(jit-api-streamk PROPERTIES ENVIRONMENT
        "TENSILE_PERSISTENT_FIXED_GRID=16;TENSILE_PERSISTENT_DYNAMIC_GRID=0")

    # Algorithm ownership and the replay faults, on the split-K bundle.
    add_executable(hipblaslt-jit-replay-backend-test replay_backend_test.cpp)
    target_include_directories(hipblaslt-jit-replay-backend-test PRIVATE
        "${hipblaslt_jit_source}" "${hipblaslt_jit_source}/rocblaslt/include")
    target_link_libraries(hipblaslt-jit-replay-backend-test PRIVATE roc::hipblaslt hip::device)
    hipblaslt_jit_test(jit-replay-backend jit-gpu hipblaslt-jit-replay-backend-test
        "${hipblaslt_jit_splitk}")
    # Publishes in one process and resolves the index in a second.
    hipblaslt_jit_test(jit-replay-backend-library jit-gpu hipblaslt-jit-replay-backend-test
        "${hipblaslt_jit_splitk}" --library)
    set_tests_properties(jit-replay-backend-library PROPERTIES ENVIRONMENT
        HIPBLASLT_JIT_LIBRARY_PATH=${hipblaslt_jit_scratch}/replay-backend-library)

    # The heuristic routes, on what the test backend replays.
    if(hipblaslt_jit_process_backend STREQUAL "hipblaslt-jit-test-backend.cpp")
        set(replay --replay "${hipblaslt_jit_data}/gfx950/rank-1"
                   --replay "${hipblaslt_jit_data}/gfx950/rank-2"
                   --replay "${hipblaslt_jit_splitk}")
        foreach(route fallback-c fallback-cpp forced cache-hit distinct unsupported
                      concurrent null-algo capture capture-query report partial-fill override
                      provider-order debug-timing debug-progress debug-off debug-file)
            hipblaslt_jit_python_test(jit-heuristic-${route} jit-gpu test_heuristic.py
                $<TARGET_FILE:hipblaslt-jit-heuristic-test> ${route}
                "${hipblaslt_jit_scratch}/heuristic-${route}" ${replay})
        endforeach()
        if(HIPBLASLT_ENABLE_CLIENT)
            hipblaslt_jit_python_test(jit-bench-smoke jit-gpu test_bench_smoke.py
                $<TARGET_FILE:hipblaslt-bench> "${hipblaslt_jit_scratch}/bench-smoke" ${replay})
        endif()
    endif()
endif()
