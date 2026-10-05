# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wiring tests for the U-03 SDPA unified binding.

Pins: fp16 d=256 is declined on all arches (ATT-34); gfx1151 bf16 is declined
until RD-01 (ST8); bf16 d=256 and all fp16 d=64/128 shapes are admitted *by the
candidate that actually serves them*; the "f16" dtype alias normalizes
correctly and does not block non-d256 fp16.

These are CPU-only dispatch tests — no GPU, no torch required.
"""
from __future__ import annotations

import unittest

from dispatch.sdpa.common import SdpaRequest
from dispatch.sdpa.bind_unified import sdpa_unified_admits, sdpa_unified_candidates


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

    def test_declined_request_offers_no_candidates(self):
        self.assertEqual(sdpa_unified_candidates(_req("gfx942", "fp16", 256)), ())


class TestGfx1151Bf16Declined(unittest.TestCase):
    """gfx1151 attention is WMMA fp16-only until RD-01 (ST8).

    The attention registry returns the same verdict for gfx1151 fp16 and bf16,
    so nothing downstream catches this — the decline has to live in the binding.
    RD-01 removes this rule and this test together, deliberately.
    """

    def test_gfx1151_bf16_d128_declined(self):
        ok, why = sdpa_unified_admits(_req("gfx1151", "bf16", 128))
        self.assertFalse(ok)
        self.assertIn("RD-01", why)

    def test_gfx1151_bf16_d256_declined(self):
        ok, why = sdpa_unified_admits(_req("gfx1151", "bf16", 256))
        self.assertFalse(ok)
        self.assertIn("RD-01", why)

    def test_gfx1151_fp16_still_admitted(self):
        # The bf16 decline must not take fp16 down with it.
        ok, _ = sdpa_unified_admits(_req("gfx1151", "fp16", 128))
        self.assertTrue(ok)


class TestBf16D256Admitted(unittest.TestCase):
    """bf16 d=256 is admitted, and by the candidate that really serves it."""

    def test_bf16_d256_gfx942_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "bf16", 256))
        self.assertTrue(ok)

    def test_bf16_d256_gfx950_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx950", "bf16", 256))
        self.assertTrue(ok)

    def test_bf16_d256_gfx950_uses_dedicated_prefill_candidate(self):
        # Regression guard. A degenerate probe shape (seqlen 1x1, or nhead_q=1)
        # routes to the generic fallback and silently drops this candidate, so
        # the boolean alone would stay green while the fast path went unseen.
        names = sdpa_unified_candidates(_req("gfx950", "bf16", 256))
        self.assertIn("attention_gfx950_d256", names)

    def test_bf16_d256_reaches_the_decode_candidate(self):
        # The decode probe is what surfaces this one; pins that probe's presence.
        names = sdpa_unified_candidates(_req("gfx950", "bf16", 256))
        self.assertIn("attention_d256_decode", names)

    def test_bf16_d256_gfx942_has_no_dedicated_prefill_candidate(self):
        # gfx942's d256 fast path is a cohort *inside* the unified kernel, not a
        # separate candidate -- so a generic verdict is correct here, not a gap.
        names = sdpa_unified_candidates(_req("gfx942", "bf16", 256))
        self.assertIn("attention_unified_2d", names)
        self.assertNotIn("attention_gfx950_d256", names)


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

    def test_bf16_d128_gfx942_admitted(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "bf16", 128))
        self.assertTrue(ok)


class TestUnsupportedInputsDeclined(unittest.TestCase):
    """Head dims off the unified vocabulary, and unknown arches, are declined."""

    def test_non_vocabulary_head_dim_declined(self):
        ok, why = sdpa_unified_admits(_req("gfx942", "bf16", 48))
        self.assertFalse(ok)
        self.assertIn("not a unified head size", why)

    def test_zero_head_dim_declined(self):
        ok, _ = sdpa_unified_admits(_req("gfx942", "bf16", 0))
        self.assertFalse(ok)

    def test_unknown_arch_declined(self):
        ok, _ = sdpa_unified_admits(_req("not-a-gpu", "bf16", 128))
        self.assertFalse(ok)

    def test_unserved_known_arch_declined(self):
        # gfx906 is a known target with no unified attention candidate.
        ok, _ = sdpa_unified_admits(_req("gfx906", "bf16", 128))
        self.assertFalse(ok)


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
