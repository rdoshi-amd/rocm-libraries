# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

# Script-mode driver for hipdnn_validate_gpu_generic_targets (run with `cmake -P`):
#   cmake -DGENERIC_TARGETS_JSON=<table.json> -P CheckGpuGenericTargets.cmake
# Prints "gpu_generic_targets: valid" only after the table passed every rule, so a ctest
# can require that line for the real table and the validator's FATAL_ERROR text for each
# invalid fixture. A crash or a missing input matches neither.

if(NOT GENERIC_TARGETS_JSON)
    message(FATAL_ERROR "CheckGpuGenericTargets: pass -DGENERIC_TARGETS_JSON=<table.json>")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/ValidateGpuGenericTargets.cmake")

hipdnn_validate_gpu_generic_targets("${GENERIC_TARGETS_JSON}")
message(STATUS "gpu_generic_targets: valid")
