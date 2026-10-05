# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Packaging probes, one line per architecture. Loaded by hkp_load_packaging_probes()
# under HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES; see HkpPackagingProbes.cmake for the
# arguments.
#
# A probe packs the production descriptors for ARCH, one instance per compile group,
# independent of the build's GPU_TARGETS. Probe an architecture the lane does not build
# itself: the lane's own architectures are covered in full by the normal build. A new
# architecture is covered by adding one line here.

hkp_add_packaging_probe(ARCH gfx950)
