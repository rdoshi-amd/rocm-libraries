# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

target_sources(hip_kernel_provider_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/TestHandleDeviceResolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIngestorKernelCode.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TestKernelIngestorEngine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddDispatchHandler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwiseAddMatchers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestPointwisePacks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestConvFwdPack.cpp
)

# The FlyDSL suite rides FlyDSL's gate, not the ingestor's: every case reads the staged
# shard through loadedSet(), which throws rather than skipping when no shard is there.
# The sources above are the generic ingestor suites and stay ungated.
if(HIPKERNELPROVIDER_FLYDSL_ACTIVE)
    target_sources(hip_kernel_provider_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/packs/TestFlydslRmsNormPack.cpp
    )
endif()
