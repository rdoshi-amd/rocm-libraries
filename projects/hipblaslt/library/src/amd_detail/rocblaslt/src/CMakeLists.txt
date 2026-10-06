# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT



target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/Debug.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/handle.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_auxiliary.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_mat.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/status.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/tensile_host.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/UserDrivenTuningParser.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/utility.cpp"
)

target_sources(hipblaslt
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/rocblaslt_transform.cpp"
)
target_include_directories(hipblaslt PRIVATE "${PROJECT_SOURCE_DIR}/device-library")
# hipblaslt itself is compiled as HIP. The rocRoller ranker includes C++20
# headers, so it is a separate C++ library and hipblaslt only calls it.
add_library(hipblaslt-rocroller-selection STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/rocroller_selected.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rocroller/solution_selection.cpp")
target_compile_features(hipblaslt-rocroller-selection PRIVATE cxx_std_20)
set_target_properties(hipblaslt-rocroller-selection PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(hipblaslt-rocroller-selection
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/rocroller/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../include"
        "${PROJECT_SOURCE_DIR}/library/include"
        "${PROJECT_BINARY_DIR}/library/include"
        "${PROJECT_SOURCE_DIR}/../../shared/rocroller/lib/include")
find_package(fmt 11.1.3 QUIET)
if(NOT TARGET fmt::fmt-header-only)
    include(FetchContent)
    set(FMT_SYSTEM_HEADERS ON)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 11.1.3)
    FetchContent_MakeAvailable(fmt)
endif()
# Same host half type and fmt headers the rocRoller library compiles with.
target_compile_definitions(hipblaslt-rocroller-selection PRIVATE
    ROCROLLER_USE_HIP
    __HIP_PLATFORM_AMD__)
target_link_libraries(hipblaslt-rocroller-selection PRIVATE
    roc::origami-headers
    hip::host
    fmt::fmt-header-only)
target_link_libraries(hipblaslt PRIVATE hipblaslt-rocroller-selection roc::origami)

if(HIPBLASLT_ENABLE_ROCROLLER)
    add_subdirectory(rocroller)
endif()
