# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# FlyDSL kernel set.
#
# Packs the checked-in AOT .hsaco objects under kernels/<arch>/ into a .kpack
# archive and stages a descriptor shard the kernel ingestor engine loads at
# runtime. Nothing here compiles: the objects are produced out of band (see
# REGEN.md) because generating them needs FlyDSL, and this build must not.
#
# WHY ITS OWN OUTPUT ROOT. The ingestor's own packer opens every pack command
# with `rm -rf ${OUT_ROOT}` (descriptor-packaging/cmake/HkpPackaging.cmake:291)
# and states the contract at :127-129 -- "one output folder, wiped and filled by
# this invocation alone. No two invocations may share a destination." The live
# root it wipes is HIPKERNELPROVIDER_DESCRIPTOR_BUILD_DIR
# (arch_content/hip-kernel-provider). Staging there would lose this tree on every
# build, not merely on a dormant configure, so ordering add_subdirectory()
# correctly would not save it. Hence a sibling root of arch_content/, named the
# way test_arch_content/ is: <root>/<provider>/<arch>, provider-qualified so two
# providers cannot collide on a set name, and so the install(DIRECTORY) below
# cannot ship another provider's content.
#
# The archive is load-bearing at runtime -- a descriptor that cannot reach it is
# a broken shard, not a degraded one -- so a missing packer is FATAL here. The
# ASM SDPA engine skips with a STATUS instead, and says why
# (src/engines/asm_sdpa_engine/CMakeLists.txt:95-100): nothing reads its archives
# yet. That comment describes exactly the case this set is not in.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(FLYDSL_KERNEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
set(FLYDSL_PACK_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/pack.py")
set(FLYDSL_DESCRIPTOR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/descriptors")
set(FLYDSL_GEN_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/gen_descriptors.py")

# Sibling of arch_content/ rather than a folder inside it; see the header.
set(FLYDSL_DESCRIPTOR_SUBDIR "flydsl_arch_content/hip-kernel-provider")
set(FLYDSL_DESCRIPTOR_BUILD_DIR
    "${HIPDNN_BUILD_PLUGIN_ENGINE_DIR}/${FLYDSL_DESCRIPTOR_SUBDIR}")

# ============================================================================
# Architecture selection
# ============================================================================
# The checked-in object set is the ceiling and GPU_TARGETS is the request; ship
# the intersection. Deriving the arch list rather than hardcoding it is the same
# lesson the Flash2 block records at CMakeLists.txt:119-127 -- hardcoding both
# the arch flag and the arch-encoding filename silently produces mislabelled
# objects when someone points the build at another arch. An empty GPU_TARGETS
# states no preference, so it takes everything checked in.
#
# GPU_TARGETS entries may carry target features (gfx942:sramecc+:xnack-). The
# kernels/ tree and the .kpack filename both key on the bare arch, so match on
# the part before the first colon.
file(GLOB _flydsl_arch_dirs RELATIVE "${FLYDSL_KERNEL_SOURCE_DIR}"
     CONFIGURE_DEPENDS "${FLYDSL_KERNEL_SOURCE_DIR}/gfx*")
set(FLYDSL_AVAILABLE_ARCHS "")
foreach(_entry IN LISTS _flydsl_arch_dirs)
    if(IS_DIRECTORY "${FLYDSL_KERNEL_SOURCE_DIR}/${_entry}")
        list(APPEND FLYDSL_AVAILABLE_ARCHS "${_entry}")
    endif()
endforeach()

set(FLYDSL_GPU_ARCHS "${FLYDSL_AVAILABLE_ARCHS}")
if(GPU_TARGETS)
    set(_flydsl_requested "")
    foreach(_target IN LISTS GPU_TARGETS)
        string(REGEX REPLACE ":.*$" "" _bare "${_target}")
        list(APPEND _flydsl_requested "${_bare}")
    endforeach()
    set(FLYDSL_GPU_ARCHS "")
    foreach(_arch IN LISTS FLYDSL_AVAILABLE_ARCHS)
        if("${_arch}" IN_LIST _flydsl_requested)
            list(APPEND FLYDSL_GPU_ARCHS "${_arch}")
        endif()
    endforeach()
endif()

if(NOT FLYDSL_GPU_ARCHS)
    string(REPLACE ";" ", " _flydsl_have "${FLYDSL_AVAILABLE_ARCHS}")
    string(REPLACE ";" ", " _flydsl_want "${GPU_TARGETS}")
    message(STATUS
        "flydsl: dormant -- no checked-in kernels for the requested targets "
        "(have: ${_flydsl_have}; GPU_TARGETS: ${_flydsl_want}).")
    # Recorded even though nothing is staged: the census helper tells a dormant
    # label from an unknown one, and skipping this leaves a later census
    # declaration reading as a typo rather than as a set that stood down.
    hkp_register_external_pack(NAME flydsl DORMANT)
    return()
endif()

# Past the dormant return, so a shard will be staged for every arch in the
# intersection. src/ and src/tests/ read this rather than the option, which says
# only that FlyDSL was asked for. INTERNAL implies FORCE, and the root cleared it
# to FALSE before adding this directory.
set(HIPKERNELPROVIDER_FLYDSL_ACTIVE TRUE CACHE INTERNAL
    "FlyDSL staged a descriptor shard for at least one requested architecture")

# ============================================================================
# Packer resolution
# ============================================================================
# CMAKE_MODULE_PATH already carries hip-kernel-provider/cmake (CMakeLists.txt:44)
# and the module carries include_guard(GLOBAL), so this needs no plumbing and
# costs nothing when the ASM SDPA engine has already included it.
include(KpackPython)

kpack_resolve_python_dir(_FLYDSL_KPACK_PYTHON_DIR)
if("${_FLYDSL_KPACK_PYTHON_DIR}" STREQUAL "")
    kpack_unset_reason(_flydsl_kpack_reason)
    message(FATAL_ERROR
        "flydsl: cannot pack kernels -- ${_flydsl_kpack_reason}. The runtime "
        "loads these kernels from the .kpack archive, so an unpacked set ships "
        "descriptors pointing at a file that does not exist. Configure with "
        "-DHIPKERNELPROVIDER_ENABLE_FLYDSL=OFF to drop the set instead.")
endif()

kpack_check_python_deps("${Python3_EXECUTABLE}" "${_FLYDSL_KPACK_PYTHON_DIR}"
                        _FLYDSL_KPACK_MISSING_DEPS)
if(_FLYDSL_KPACK_MISSING_DEPS)
    string(REPLACE ";" ", " _flydsl_missing_csv "${_FLYDSL_KPACK_MISSING_DEPS}")
    message(FATAL_ERROR
        "flydsl: ${Python3_EXECUTABLE} cannot import ${_flydsl_missing_csv} from "
        "${_FLYDSL_KPACK_PYTHON_DIR} (rocm_kpack needs zstandard>=0.20.0 and "
        "msgpack). Install them into that interpreter, or configure with "
        "-DHIPKERNELPROVIDER_ENABLE_FLYDSL=OFF.")
endif()

# ============================================================================
# Per-architecture shard
# ============================================================================
# Two steps per arch, in this order:
#
#   1. stage  -- wipe the shard and refill it from descriptors/<arch>/
#   2. pack   -- write <shard>/kpack/<group>_<arch>.kpack
#
# The wipe is what makes a renamed or deleted descriptor disappear from the
# shard. Without it a stale .ukd.json keeps being loaded, which is a runtime
# correctness bug rather than a cosmetic one. It has to precede the pack for the
# obvious reason, so the pack command depends on the stage stamp. The stamp
# lives outside the shard, since the shard is what gets wiped.
#
# The archive path and the descriptors' `kernel_source.library` string are two
# halves of one contract: library is
# "kpack/<group>_<arch>.kpack" for a descriptor at the shard root, and the
# runtime resolves it against the descriptor file's own directory and then
# checks the result stays inside the descriptor tree.
set(_flydsl_stamp_dir "${CMAKE_CURRENT_BINARY_DIR}/stamps")
set(FLYDSL_KPACK_OUTPUTS "")

foreach(_arch IN LISTS FLYDSL_GPU_ARCHS)
    set(_shard "${FLYDSL_DESCRIPTOR_BUILD_DIR}/${_arch}")
    set(_kpack "${_shard}/kpack/hip_kernel_provider_flydsl_${_arch}.kpack")
    set(_stage_stamp "${_flydsl_stamp_dir}/${_arch}.staged")
    set(_desc_src "${CMAKE_CURRENT_SOURCE_DIR}/descriptors/${_arch}")

    if(NOT IS_DIRECTORY "${_desc_src}")
        # A shard holding an archive no descriptor names is unreachable at
        # runtime: nothing selects it, so the build passes and proves nothing.
        # An arch reaches this loop only because kernels/<arch>/ exists, so the
        # two directories are one unit and half of it is a packaging bug.
        message(FATAL_ERROR
            "flydsl: kernels/${_arch}/ exists but descriptors/${_arch}/ does not. "
            "Regenerate with:\n"
            "  python3 ${FLYDSL_GEN_SCRIPT} --kernel-dir ${FLYDSL_KERNEL_SOURCE_DIR} "
            "--descriptor-dir ${FLYDSL_DESCRIPTOR_SOURCE_DIR} --arch ${_arch}\n"
            "or configure with -DHIPKERNELPROVIDER_ENABLE_FLYDSL=OFF.")
    endif()
    file(GLOB_RECURSE _desc_files CONFIGURE_DEPENDS "${_desc_src}/*.json")

    # Validate before staging, so a descriptor set that disagrees with the
    # objects fails here rather than at load on a machine with no way to
    # diagnose it. --check writes nothing and derives no signatures, so it needs
    # no msgpack and costs a single stdlib-only Python run per arch. It re-runs
    # whenever a descriptor or a code object changes, which is exactly when the
    # two can start disagreeing.
    set(_stage_commands
        COMMAND ${Python3_EXECUTABLE} "${FLYDSL_GEN_SCRIPT}"
                --kernel-dir "${FLYDSL_KERNEL_SOURCE_DIR}"
                --descriptor-dir "${FLYDSL_DESCRIPTOR_SOURCE_DIR}"
                --arch ${_arch} --check
        COMMAND ${CMAKE_COMMAND} -E rm -rf "${_shard}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_shard}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${_desc_src}" "${_shard}")

    # Depending on ${_desc_files} alone tracks descriptor *contents*, not the
    # descriptor *set*, and the wipe above exists for the set: it is what makes a
    # deleted or renamed .ukd.json leave the shard instead of lingering and being
    # loaded. Deleting one was measured to leave it staged. CONFIGURE_DEPENDS does
    # notice the glob change and re-runs configure, but that is not enough on its
    # own -- the regenerated rule has the same command list, and its stamp is still
    # newer than every surviving descriptor, so the build considers it up to date
    # and the wipe never runs.
    #
    # So make the set itself a file the stamp depends on. file(GENERATE) rewrites
    # only when the content differs, so this file's timestamp moves exactly when a
    # descriptor is added, renamed or removed, and not on a no-op reconfigure.
    list(SORT _desc_files)
    string(REPLACE ";" "\n" _desc_manifest_content "${_desc_files}")
    set(_desc_manifest "${_flydsl_stamp_dir}/${_arch}.descriptor-set")
    file(GENERATE OUTPUT "${_desc_manifest}" CONTENT "${_desc_manifest_content}\n")

    file(GLOB_RECURSE _flydsl_objs CONFIGURE_DEPENDS
         "${FLYDSL_KERNEL_SOURCE_DIR}/${_arch}/*.hsaco")
    file(GLOB_RECURSE _flydsl_manifests CONFIGURE_DEPENDS
         "${FLYDSL_KERNEL_SOURCE_DIR}/${_arch}/*/manifest.json")

    # The stage step now validates as well as copies, so it depends on both
    # sides of what it compares: the descriptors and the objects they claim.
    add_custom_command(
        OUTPUT "${_stage_stamp}"
        ${_stage_commands}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_flydsl_stamp_dir}"
        COMMAND ${CMAKE_COMMAND} -E touch "${_stage_stamp}"
        DEPENDS ${_desc_files} "${_desc_manifest}" "${FLYDSL_GEN_SCRIPT}"
                "${FLYDSL_PACK_SCRIPT}" ${_flydsl_objs} ${_flydsl_manifests}
        COMMENT "flydsl: validating and staging ${_arch} descriptor shard"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${_kpack}"
        COMMAND ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${_FLYDSL_KPACK_PYTHON_DIR}"
                "PYTHONDONTWRITEBYTECODE=1"
                --
                ${Python3_EXECUTABLE} "${FLYDSL_PACK_SCRIPT}"
                --kernel-dir "${FLYDSL_KERNEL_SOURCE_DIR}"
                --arch ${_arch}
                --out-dir "${_shard}/kpack"
        DEPENDS "${FLYDSL_PACK_SCRIPT}" "${_stage_stamp}"
                ${_flydsl_objs} ${_flydsl_manifests}
        COMMENT "flydsl: packing ${_arch} code objects into hip_kernel_provider_flydsl_${_arch}.kpack"
        VERBATIM
    )
    list(APPEND FLYDSL_KPACK_OUTPUTS "${_kpack}")

    message(STATUS "flydsl: ${_arch} shard -> ${_shard}")
endforeach()

add_custom_target(flydsl_kernel_kpack ALL
    DEPENDS ${FLYDSL_KPACK_OUTPUTS}
    COMMENT "Stage FlyDSL descriptor shards and pack their code objects"
)

# Tell the packaging registry where these shards landed, so src/tests/ can census
# them through the same helper the lowered packs use. The arch list is the
# intersection, not GPU_TARGETS: a census entry for an arch this loop skipped
# would address a shard directory the rules above never wrote.
hkp_register_external_pack(
    NAME flydsl
    OUT_ROOT "${FLYDSL_DESCRIPTOR_BUILD_DIR}"
    ARCHES ${FLYDSL_GPU_ARCHS}
)

# Trailing slash: install the shards themselves, not a directory named after the
# build root. The build tree already mirrors the install tree here, so the path
# a developer points HIPDNN_DESCRIPTOR_DIR at in the build tree is the same
# relative path it has once installed.
install(DIRECTORY "${FLYDSL_DESCRIPTOR_BUILD_DIR}/"
        DESTINATION "${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}/${FLYDSL_DESCRIPTOR_SUBDIR}")
