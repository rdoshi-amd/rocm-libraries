# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# FlyDSL kernel set.
#
# The checked-in AOT objects and their descriptors are a bundle of the
# provider's production descriptor root, at <root>/FlyDSL/, beside rocKE's. The
# shared packer packs them like every other bundle there: it resolves each
# authored `hsaco` UKD's object, packs it into the per-arch
# hip_kernel_provider_<arch>.kpack, rewrites the UKD to kpack form, and stages
# the shard into arch_content/hip-kernel-provider/, which is installed and found
# beside the plugin with no further configuration. The FlyDSL family folder is
# switched on and off by HIPKERNELPROVIDER_ENABLE_FLYDSL through
# HKP_DESCRIPTOR_FAMILIES (descriptor-packaging/cmake/HkpPackaging.cmake).
#
# Nothing here compiles, packs or installs. This file answers one question --
# does this build ship FlyDSL kernels at all -- for the engine and test gates,
# and checks before the pack that the descriptors still agree with the objects.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(FLYDSL_GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/gen_descriptors.py")

# Read from the root the packer is actually packing. A build that points the
# production root elsewhere -- the superbuild CI lane packs the examples tree --
# ships no FlyDSL descriptors, and must not compile an engine whose kernels it
# does not ship.
set(FLYDSL_CONTENT_DIR "${HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT}/FlyDSL")

# ============================================================================
# Architecture selection
# ============================================================================
# The checked-in object set is the ceiling and GPU_TARGETS is the request; the
# intersection is what ships. An empty GPU_TARGETS states no preference, so it
# takes everything checked in. The packer prunes by each pack's own `arch`, so
# this list does not decide what is packed; it decides whether the native seams
# and the tests are compiled, which must agree with it.
#
# GPU_TARGETS entries may carry target features (gfx942:sramecc+:xnack-). The
# content tree keys on the bare arch, so match on the part before the first colon.
file(GLOB _flydsl_arch_dirs LIST_DIRECTORIES true RELATIVE "${FLYDSL_CONTENT_DIR}"
     CONFIGURE_DEPENDS "${FLYDSL_CONTENT_DIR}/*/gfx*")
set(FLYDSL_AVAILABLE_ARCHS "")
foreach(_entry IN LISTS _flydsl_arch_dirs)
    if(IS_DIRECTORY "${FLYDSL_CONTENT_DIR}/${_entry}")
        get_filename_component(_arch "${_entry}" NAME)
        list(APPEND FLYDSL_AVAILABLE_ARCHS "${_arch}")
    endif()
endforeach()
list(REMOVE_DUPLICATES FLYDSL_AVAILABLE_ARCHS)

set(_flydsl_archs "${FLYDSL_AVAILABLE_ARCHS}")
if(GPU_TARGETS)
    set(_flydsl_requested "")
    foreach(_target IN LISTS GPU_TARGETS)
        string(REGEX REPLACE ":.*$" "" _bare "${_target}")
        list(APPEND _flydsl_requested "${_bare}")
    endforeach()
    set(_flydsl_archs "")
    foreach(_arch IN LISTS FLYDSL_AVAILABLE_ARCHS)
        if("${_arch}" IN_LIST _flydsl_requested)
            list(APPEND _flydsl_archs "${_arch}")
        endif()
    endforeach()
endif()

if(NOT _flydsl_archs OR NOT TARGET hkp_packaging_product)
    string(REPLACE ";" ", " _flydsl_have "${FLYDSL_AVAILABLE_ARCHS}")
    string(REPLACE ";" ", " _flydsl_want "${GPU_TARGETS}")
    message(STATUS
        "flydsl: dormant -- no checked-in kernels for the requested targets in "
        "the production descriptor root (have: ${_flydsl_have}; GPU_TARGETS: "
        "${_flydsl_want}).")
    return()
endif()

# Past the dormant return, so the product pack ships FlyDSL kernels for every
# arch in the intersection. src/ and src/tests/ read these rather than the
# option, which says only that FlyDSL was asked for. INTERNAL implies FORCE, and
# the root cleared ACTIVE to FALSE before adding this directory.
set(HIPKERNELPROVIDER_FLYDSL_ACTIVE TRUE CACHE INTERNAL
    "The product pack ships FlyDSL kernels for at least one requested architecture")
set(HIPKERNELPROVIDER_FLYDSL_ARCHS "${_flydsl_archs}" CACHE INTERNAL
    "Architectures the product pack ships FlyDSL kernels for")

# ============================================================================
# Descriptors agree with the objects, before the pack reads either
# ============================================================================
# The packer verifies each object against the SHA256 its descriptor records, so
# an object regenerated without its descriptors already fails the pack. This
# catches the rest: a descriptor whose metadata, priority or symbol no longer
# matches what the manifest says the object is. Runs whenever any file in the
# FlyDSL folder changes, which is exactly when the two can start disagreeing.
file(GLOB_RECURSE _flydsl_content CONFIGURE_DEPENDS "${FLYDSL_CONTENT_DIR}/*")
set(_flydsl_check_commands "")
foreach(_arch IN LISTS _flydsl_archs)
    list(APPEND _flydsl_check_commands
        COMMAND ${Python3_EXECUTABLE} "${FLYDSL_GEN_SCRIPT}"
                --content-dir "${FLYDSL_CONTENT_DIR}" --arch ${_arch} --check)
endforeach()

set(_flydsl_check_stamp "${CMAKE_CURRENT_BINARY_DIR}/flydsl-descriptors.checked")
add_custom_command(
    OUTPUT "${_flydsl_check_stamp}"
    ${_flydsl_check_commands}
    COMMAND ${CMAKE_COMMAND} -E touch "${_flydsl_check_stamp}"
    DEPENDS ${_flydsl_content} "${FLYDSL_GEN_SCRIPT}"
    COMMENT "flydsl: checking descriptors against the checked-in objects"
    VERBATIM
)
add_custom_target(flydsl_descriptor_check
    DEPENDS "${_flydsl_check_stamp}"
    COMMENT "Check FlyDSL descriptors against the checked-in objects")
add_dependencies(hkp_packaging_product flydsl_descriptor_check)

string(REPLACE ";" ", " _flydsl_arch_csv "${_flydsl_archs}")
message(STATUS "flydsl: shipping through the product pack for ${_flydsl_arch_csv}")
