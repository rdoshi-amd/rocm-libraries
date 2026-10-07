# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Script-mode driver for hipdnn_validate_gpu_generic_targets (run with `cmake -P`):
#   cmake -DGENERIC_TARGETS_JSON=<table.json> [-DMUTATION_OP=<op> ...] -P CheckGpuGenericTargets.cmake
# Without MUTATION_OP the table is validated as is. With it, the table is mutated in
# memory, written to MUTATED_JSON, and that file is validated. MUTATION_PATH is a
# '.'-separated key/index path into the table. MUTATION_OP is one of:
#   REMOVE      delete the value at MUTATION_PATH
#   SET_STRING  set MUTATION_PATH to the string MUTATION_VALUE
#   SET_JSON    set MUTATION_PATH to the JSON text MUTATION_VALUE
#   RAW         replace the whole document with the text MUTATION_VALUE
# Prints "gpu_generic_targets: valid" only after the table passed every rule, so a ctest
# can require the validator's FATAL_ERROR text for a mutation and forbid that line. A
# crash or a missing input matches neither.

if(NOT GENERIC_TARGETS_JSON)
    message(FATAL_ERROR "CheckGpuGenericTargets: pass -DGENERIC_TARGETS_JSON=<table.json>")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/ValidateGpuGenericTargets.cmake")

set(table_path "${GENERIC_TARGETS_JSON}")
if(MUTATION_OP)
    if(NOT MUTATED_JSON)
        message(FATAL_ERROR "CheckGpuGenericTargets: MUTATION_OP needs -DMUTATED_JSON=<temp file>")
    endif()
    file(READ "${GENERIC_TARGETS_JSON}" json)
    string(REPLACE "." ";" path "${MUTATION_PATH}")
    if(MUTATION_OP STREQUAL "REMOVE")
        string(JSON json REMOVE "${json}" ${path})
    elseif(MUTATION_OP STREQUAL "SET_STRING")
        string(JSON json SET "${json}" ${path} "\"${MUTATION_VALUE}\"")
    elseif(MUTATION_OP STREQUAL "SET_JSON")
        string(JSON json SET "${json}" ${path} "${MUTATION_VALUE}")
    elseif(MUTATION_OP STREQUAL "RAW")
        set(json "${MUTATION_VALUE}")
    else()
        message(FATAL_ERROR "CheckGpuGenericTargets: unknown MUTATION_OP '${MUTATION_OP}'")
    endif()
    file(WRITE "${MUTATED_JSON}" "${json}")
    set(table_path "${MUTATED_JSON}")
endif()

hipdnn_validate_gpu_generic_targets("${table_path}")
message(STATUS "gpu_generic_targets: valid")
