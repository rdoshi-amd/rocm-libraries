# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wiring tests for the U-03 SDPA unified binding.

Pins: fp16 d=256 is declined on all arches (ATT-34); bf16 d=256 and all
fp16/bf16 d=64/128 shapes are admitted; the "f16" dtype alias normalizes
correctly and does not block non-d256 fp16.

These are CPU-only dispatch tests — no GPU, no torch required.
"""
from __future__ import annotations

import unittest

from dispatch.sdpa.common import SdpaRequest
from dispatch.sdpa.bind_unified import sdpa_unified_admits


def _req(arch: str, dtype: str, head_dim: int) -> SdpaRequest:
    return SdpaRequest(arch=arch, dtype=dtype, head_dim=head_dim)


class TestFp16D256Declined(unittest.TestCase):
    """fp16 d=256 must be declined on every arch (ATT-34)."""

    def test_fp16_d256_gfx942_declined(self):
        ok, why = sdpa_unified_admits(_req("gfx942", "fp16", 256))
        self.assertFalse(ok)
        self.assertIn("ATT-34", why)

    def test_fp16_d256_gfx950_declined(self):
        ok, why = sdpa_unified_admits(_req("gfx950", "fp16", 256))
        self.assertFalse(ok)
        self.assertIn("ATT-34", why)

    def test_fp16_d256_f16_alias_declined(self):
        # "f16" normalizes to "fp16" in __post_init__; the decline must still fire.
        ok, why = sdpa_unified_admits(_req("gfx942", "f16", 256))
        self.assertFalse(ok)
        self.assertIn("ATT-34", why)


class TestBf16D256Admitted(unittest.TestCase):
    """bf16 d=256 has dedicated fast-path candidates on gfx942 and gfx950."""

    def test_bf16_d256_gfx942_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "bf16", 256))
        self.assertTrue(ok)

    def test_bf16_d256_gfx950_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx950", "bf16", 256))
        self.assertTrue(ok)


class TestSmallerHeadDimsAdmitted(unittest.TestCase):
    """fp16 and bf16 at d=64 and d=128 must be admitted on gfx942 and gfx950."""

    def test_fp16_d128_gfx942_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "fp16", 128))
        self.assertTrue(ok)

    def test_fp16_d64_gfx942_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "fp16", 64))
        self.assertTrue(ok)

    def test_fp16_d128_gfx950_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx950", "fp16", 128))
        self.assertTrue(ok)

    def test_fp16_d64_gfx950_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx950", "fp16", 64))
        self.assertTrue(ok)


class TestDtypeAliasNormalization(unittest.TestCase):
    """The "f16" alias must not block non-d256 fp16 requests."""

    def test_f16_alias_d128_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "f16", 128))
        self.assertTrue(ok)

    def test_bad_dtype_raises(self):
        with self.assertRaises(ValueError):
            SdpaRequest(arch="gfx942", dtype="badtype", head_dim=128)


if __name__ == "__main__":
    unittest.main()
