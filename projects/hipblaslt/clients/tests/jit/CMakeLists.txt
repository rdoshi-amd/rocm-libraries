# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# JIT tests, staged in clients/staging. CTest labels them jit-cpu, which needs
# no GPU, or jit-gpu. Each test empties its own scratch directory.
enable_testing()

set(hipblaslt_jit_source "${PROJECT_SOURCE_DIR}/library/src/amd_detail")
set(hipblaslt_jit_scratch "${CMAKE_CURRENT_BINARY_DIR}/scratch")
# The committed bundle; data/README.md lists how it was generated.
set(hipblaslt_jit_bundle "${CMAKE_CURRENT_SOURCE_DIR}/data/gfx950/plain")

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
