# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# JIT tests, staged in clients/staging. CTest labels them jit-cpu, which needs
# no GPU, or jit-gpu. Each test empties its own scratch directory.
enable_testing()

set(hipblaslt_jit_source "${PROJECT_SOURCE_DIR}/library/src/amd_detail")
set(hipblaslt_jit_scratch "${CMAKE_CURRENT_BINARY_DIR}/scratch")
# The committed bundle; data/README.md lists how it was generated.
set(hipblaslt_jit_bundle "${CMAKE_CURRENT_SOURCE_DIR}/data/gfx950/plain")
string(REGEX MATCH "(^|;)gfx950(:[^;]*)?(;|$)" hipblaslt_jit_gfx950 "${GPU_TARGETS}")

# hipblaslt_jit_test(name label target args...) stages target and runs it.
function(hipblaslt_jit_test name label test_target)
    target_compile_features(${test_target} PRIVATE cxx_std_17)
    set_target_properties(${test_target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/clients/staging")
    add_test(NAME ${name} COMMAND $<TARGET_FILE:${test_target}> ${ARGN})
    set_tests_properties(${name} PROPERTIES LABELS ${label})
endfunction()

add_executable(hipblaslt-jit-source-bundle-test source_bundle_test.cpp)
target_include_directories(hipblaslt-jit-source-bundle-test PRIVATE "${hipblaslt_jit_source}")
hipblaslt_jit_test(jit-source-bundle jit-cpu hipblaslt-jit-source-bundle-test
    "${hipblaslt_jit_scratch}/source-bundle")

if(NOT HIPBLASLT_ENABLE_JIT)
    add_executable(hipblaslt-jit-disabled-test disabled_api_test.cpp)
    target_link_libraries(hipblaslt-jit-disabled-test PRIVATE roc::hipblaslt hip::host)
    hipblaslt_jit_test(jit-disabled jit-cpu hipblaslt-jit-disabled-test)
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
hipblaslt_jit_test(jit-builder jit-cpu hipblaslt-jit-builder-test
    "${hipblaslt_jit_bundle}" "${hipblaslt_jit_scratch}/builder")

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
hipblaslt_jit_test(jit-loader jit-gpu hipblaslt-jit-loader-test
    "${hipblaslt_jit_bundle}" "${hipblaslt_jit_scratch}/loader")

if(HIPBLASLT_JIT_TESTING)
    add_executable(hipblaslt-jit-end-to-end-test jit_end_to_end_test.cpp)
    target_include_directories(hipblaslt-jit-end-to-end-test PRIVATE "${hipblaslt_jit_source}")
    target_link_libraries(hipblaslt-jit-end-to-end-test PRIVATE roc::hipblaslt hip::device)
    hipblaslt_jit_test(jit-end-to-end jit-gpu hipblaslt-jit-end-to-end-test
        "${hipblaslt_jit_bundle}")

    # The C and C++ GEMM APIs on replayed solutions; each case names its bundle.
    add_executable(hipblaslt-jit-api-test jit_api_test.cpp)
    target_include_directories(hipblaslt-jit-api-test PRIVATE "${hipblaslt_jit_source}")
    target_link_libraries(hipblaslt-jit-api-test PRIVATE roc::hipblaslt hip::device)
    foreach(case "streamk;streamk;--k;4096;--workspace-fallback;1"
                 "amax;amax;--amax;1"
                 "alpha-zero;amax;--amax;1;--alpha-zero;1")
        list(POP_FRONT case name bundle)
        hipblaslt_jit_test(jit-api-${name} jit-gpu hipblaslt-jit-api-test
            --replay "${CMAKE_CURRENT_SOURCE_DIR}/data/gfx950/${bundle}" ${case})
    endforeach()
    set_tests_properties(jit-api-streamk PROPERTIES ENVIRONMENT
        "TENSILE_PERSISTENT_FIXED_GRID=16;TENSILE_PERSISTENT_DYNAMIC_GRID=0")
endif()
