# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
#
# Packaging probes. Loaded by hkp_load_packaging_probes() under
# HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES; see HkpPackagingProbes.cmake for the
# arguments and descriptor-packaging/README.md, "Packaging probes", for usage.
#
# A probe packs a descriptor root for ARCH, independent of the build's GPU_TARGETS; ARCH
# may be any architecture. Without UKDS it packs one UKD per compile group of every KDP
# shipping for ARCH, so a new pack under ARCH needs no new line. An integration adds its
# own line, hkp_add_packaging_probe(ARCH <gfx> NAME <integration> UKDS <ukd-name>...),
# naming a representative set of UKDs that is packed on every PR.

hkp_add_packaging_probe(ARCH gfx950)
