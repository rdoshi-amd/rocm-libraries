# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Validates data/gpu_generic_targets.json: the LLVM provenance block and the shape of the
# generic-target -> member-processor table. Every violation is a FATAL_ERROR that names
# the offending key or member, so a failing configure points at the line to fix.
#
# This is the only implementation of the provenance rules; the Python readers validate
# shape only.
#
# cmake-lint does not model foreach(... RANGE ...) and reports E1120 on its arguments.
# cmake-lint: disable=E1120

# Reads the string at <key path...> of <json> into <out>; FATAL_ERROR if it is absent,
# not a string, or empty.
function(_hipdnn_gpu_generic_targets_get_string out json file key_path)
    string(JSON value ERROR_VARIABLE err GET "${json}" ${ARGN})
    string(JSON type ERROR_VARIABLE type_err TYPE "${json}" ${ARGN})
    if(err OR type_err)
        message(FATAL_ERROR "gpu_generic_targets: ${file}: missing required key '${key_path}'")
    endif()
    if(NOT type STREQUAL "STRING")
        message(FATAL_ERROR "gpu_generic_targets: ${file}: key '${key_path}' must be a string")
    endif()
    if(value STREQUAL "")
        message(FATAL_ERROR "gpu_generic_targets: ${file}: key '${key_path}' must be non-empty")
    endif()
    set(${out} "${value}" PARENT_SCOPE)
endfunction()

# Checks the provenance block ('source') of the table.
function(_hipdnn_gpu_generic_targets_validate_source json json_path)
    string(JSON source_type ERROR_VARIABLE err TYPE "${json}" source)
    if(err OR NOT source_type STREQUAL "OBJECT")
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: missing required object 'source'")
    endif()

    foreach(key description document section repository revision permalink retrievedOn license
                licenseNote statement pinnedBy)
        _hipdnn_gpu_generic_targets_get_string(source_${key} "${json}" "${json_path}" "source.${key}"
                                               source ${key})
    endforeach()

    if(NOT source_document MATCHES "AMDGPUUsage\\.rst$")
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: 'source.document' must name AMDGPUUsage.rst")
    endif()
    if(NOT source_section MATCHES "^AMDGPU Generic Processors")
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: 'source.section' must name the 'AMDGPU Generic Processors' table")
    endif()
    if(NOT source_repository STREQUAL "https://github.com/ROCm/llvm-project")
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: 'source.repository' must be https://github.com/ROCm/llvm-project")
    endif()
    string(LENGTH "${source_revision}" revision_length)
    if(NOT source_revision MATCHES "^[0-9a-f]+$" OR NOT revision_length EQUAL 40)
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: 'source.revision' must be a 40-digit lowercase hex commit id, got '${source_revision}'")
    endif()
    if(NOT source_permalink MATCHES "^https://")
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: 'source.permalink' must be an https:// URL")
    endif()
    string(FIND "${source_permalink}" "${source_revision}" revision_at)
    if(revision_at EQUAL -1)
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: 'source.permalink' must contain 'source.revision' (${source_revision}) so the link is pinned")
    endif()
    if(NOT source_retrievedOn MATCHES "^[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]$")
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: 'source.retrievedOn' must be YYYY-MM-DD, got '${source_retrievedOn}'")
    endif()
    if(NOT source_license STREQUAL "Apache-2.0 WITH LLVM-exception")
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: 'source.license' must be 'Apache-2.0 WITH LLVM-exception', got '${source_license}'")
    endif()
    if(NOT source_statement MATCHES "[Oo]nly factual processor-name membership")
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: 'source.statement' must state that only factual processor-name membership is recorded")
    endif()

endfunction()

# Checks the 'generics' map: names, member shape, duplicates.
function(_hipdnn_gpu_generic_targets_validate_generics json json_path)
    string(JSON generics_type ERROR_VARIABLE err TYPE "${json}" generics)
    if(err OR NOT generics_type STREQUAL "OBJECT")
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: missing required object 'generics'")
    endif()
    string(JSON generic_count LENGTH "${json}" generics)
    if(generic_count EQUAL 0)
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: 'generics' must be non-empty")
    endif()

    math(EXPR last_generic "${generic_count} - 1")
    foreach(g RANGE 0 ${last_generic})
        string(JSON generic MEMBER "${json}" generics ${g})
        if(NOT generic MATCHES "^gfx[0-9]+(-[0-9]+)?-generic$")
            message(FATAL_ERROR "gpu_generic_targets: ${json_path}: generic name '${generic}' does not match ^gfx[0-9]+(-[0-9]+)?-generic$")
        endif()
        string(JSON members_type TYPE "${json}" generics "${generic}")
        if(NOT members_type STREQUAL "ARRAY")
            message(FATAL_ERROR "gpu_generic_targets: ${json_path}: generics.${generic} must be an array")
        endif()
        string(JSON member_count LENGTH "${json}" generics "${generic}")
        if(member_count EQUAL 0)
            message(FATAL_ERROR "gpu_generic_targets: ${json_path}: generics.${generic} must list at least one member")
        endif()

        set(seen "")
        math(EXPR last_member "${member_count} - 1")
        foreach(m RANGE 0 ${last_member})
            string(JSON member_type TYPE "${json}" generics "${generic}" ${m})
            if(NOT member_type STREQUAL "STRING")
                message(FATAL_ERROR "gpu_generic_targets: ${json_path}: generics.${generic}[${m}] must be a string")
            endif()
            string(JSON member GET "${json}" generics "${generic}" ${m})
            if(NOT member MATCHES "^gfx[0-9a-f]+$")
                message(FATAL_ERROR "gpu_generic_targets: ${json_path}: member '${member}' of ${generic} does not match ^gfx[0-9a-f]+$")
            endif()
            list(FIND seen "${member}" seen_at)
            if(NOT seen_at EQUAL -1)
                message(FATAL_ERROR "gpu_generic_targets: ${json_path}: member '${member}' is listed twice in ${generic}")
            endif()
            list(APPEND seen "${member}")
        endforeach()
    endforeach()
endfunction()

# Validates the generic target table at <json_path>: the comment and schema version, the
# provenance block, and every generic's member list. FATAL_ERROR naming the offending
# key or member on the first violation.
function(hipdnn_validate_gpu_generic_targets json_path)
    if(NOT EXISTS "${json_path}")
        message(FATAL_ERROR "gpu_generic_targets: cannot read '${json_path}'")
    endif()
    file(READ "${json_path}" json)

    string(JSON _root_type ERROR_VARIABLE parse_error TYPE "${json}")
    if(parse_error)
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: not valid JSON: ${parse_error}")
    endif()

    _hipdnn_gpu_generic_targets_get_string(_ "${json}" "${json_path}" "\$comment" "\$comment")

    string(JSON version ERROR_VARIABLE err GET "${json}" schemaVersion)
    if(err)
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: missing required key 'schemaVersion'")
    endif()
    string(JSON version_type TYPE "${json}" schemaVersion)
    if(NOT version_type STREQUAL "NUMBER" OR NOT version STREQUAL "1")
        message(FATAL_ERROR "gpu_generic_targets: ${json_path}: 'schemaVersion' must be 1, got '${version}'")
    endif()

    _hipdnn_gpu_generic_targets_validate_source("${json}" "${json_path}")
    _hipdnn_gpu_generic_targets_validate_generics("${json}" "${json_path}")
endfunction()
