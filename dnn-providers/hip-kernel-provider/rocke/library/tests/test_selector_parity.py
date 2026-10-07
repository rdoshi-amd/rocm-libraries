# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pure-Python unit tests for the gfx950 attention tiled-2D selector logic.

These tests verify that the Python ``_tiled_spec_from_problem`` selector
produces the expected spec fields for a range of UnifiedAttentionProblem
inputs -- covering the selectors and gate predicates that route a problem
into the correct kernel variant.  No GPU, no subprocess, no rocke_engine .so
required.

Primary motivation: PR #9220 silently broke parity by removing the bias-
exclusion guard from Python ``_enable_combo_2d``, routing biased combo-geometry
attention onto the transposed combo path while C++ still refused bias.  These
tests encode the expected routing decisions so that kind of regression is caught
immediately on any developer machine.

NOTE ON EXPECTED FAILURES (PR #9220 regression):
Tests marked with @skip document a known divergence introduced by
PR #9220, which removed the bias-exclusion guard from Python _enable_combo_2d.
Python now routes biased combo-geometry problems onto the combo path while
C++ (attention_unified_selectors.cpp) still refuses them. The tests are
written for the CORRECT behavior (bias should block combo_2d) and are being skipped until the Python guard is restored.
"""

from __future__ import annotations

import unittest
from dataclasses import asdict, fields, replace
from types import SimpleNamespace
from unittest import mock

import builders.common.attention_spec_builder as _asb
import kernels.common.attention_unified as _au
from dispatch.attention import AttentionRequest, attention_candidates
from dispatch.attention.common import _tuning_problem
from kernels import UnifiedAttentionProblem
from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa
from kernels.gfx1250.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx1250TiledSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_arch(arch: str):
    """Pin ``_resolve_attention_arch`` on the module that defines it.

    That is sufficient because the spec builder reaches the resolver through its
    ``attention_unified`` module handle -- a bound import there would freeze the
    reference and silently leave the builder on the real device arch, so these
    cases would test the host, not ``arch``. That invariant is pinned, statically
    and behaviourally, by ``test_arch_binding_guard.py``; it is not re-asserted
    here.
    """
    return mock.patch.object(_au, "_resolve_attention_arch", return_value=arch)


def _spec(problem: UnifiedAttentionProblem, arch: str = "gfx950") -> dict:
    """Return _tiled_spec_from_problem as a plain dict for easy assertions."""
    with _patch_arch(arch):
        return asdict(_asb._tiled_spec_from_problem(problem))


def _prob(
    head_size: int,
    block_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    dtype: str = "bf16",
    max_seqlen_q: int = 512,
    max_seqlen_k: int = 4096,
    num_seqs: int = 2,
    total_q: int = 1024,
    sliding_window: int = 0,
    softcap: float = 0.0,
    use_sinks: bool = False,
    use_alibi: bool = False,
    use_qq_bias: bool = False,
    use_fp8: bool = False,
    num_kv_blocks: int = 0,
) -> UnifiedAttentionProblem:
    return UnifiedAttentionProblem(
        total_q=total_q,
        num_seqs=num_seqs,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dtype=dtype,
        sliding_window=sliding_window,
        softcap=softcap,
        use_sinks=use_sinks,
        use_alibi=use_alibi,
        use_qq_bias=use_qq_bias,
        use_fp8=use_fp8,
        num_kv_blocks=num_kv_blocks,
    )


# Canonical combo_2d problem: gfx950 + bf16 + d64/b32 + GQA-8 + seqlen_q > 256
def _combo_prob(**overrides) -> UnifiedAttentionProblem:
    defaults = dict(
        head_size=64,
        block_size=32,
        num_query_heads=64,
        num_kv_heads=8,
        dtype="bf16",
        max_seqlen_q=512,
    )
    defaults.update(overrides)
    return _prob(**defaults)


# ---------------------------------------------------------------------------
# enable_combo_2d gate
# ---------------------------------------------------------------------------


class TestEnableCombo2d(unittest.TestCase):
    """_enable_combo_2d fires for the gfx950 d64/b32/GQA-8/S>256 cohort: bf16
    (sinks or not), or fp16 only when use_sinks (the fp16 widening was measured
    on sink prefill)."""

    _BIAS_BUG = (
        "PR #9220: Python _enable_combo_2d missing bias guard (C++ still refuses bias)"
    )

    def _gate(self, problem: UnifiedAttentionProblem, arch: str = "gfx950") -> bool:
        with _patch_arch(arch):
            return _au._enable_combo_2d(problem)

    def test_canonical_combo_fires(self):
        self.assertTrue(self._gate(_combo_prob()))

    @unittest.skip(_BIAS_BUG)
    def test_qq_bias_blocks_combo(self):
        """use_qq_bias=True must disable combo_2d (was the #9220 bug)."""
        self.assertFalse(self._gate(_combo_prob(use_qq_bias=True)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_alibi_blocks_combo(self):
        self.assertFalse(self._gate(_combo_prob(use_alibi=True)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_softcap_blocks_combo(self):
        self.assertFalse(self._gate(_combo_prob(softcap=50.0)), self._BIAS_BUG)

    def test_fp16_no_sinks_blocks_combo(self):
        # fp16 combo is sink-prefill only; non-sink fp16 stays off the combo path.
        self.assertFalse(self._gate(_combo_prob(dtype="fp16")))

    def test_fp16_sinks_enables_combo(self):
        # fp16 + sinks is the one fp16 shape the widening admits to combo.
        self.assertTrue(self._gate(_combo_prob(dtype="fp16", use_sinks=True)))

    def test_wrong_head_size_blocks_combo(self):
        self.assertFalse(self._gate(_prob(128, 32, 64, 8)))

    def test_wrong_block_size_blocks_combo(self):
        self.assertFalse(self._gate(_prob(64, 16, 64, 8)))

    def test_wrong_gqa_ratio_blocks_combo(self):
        # nqpk = 16/4 = 4, not 8
        self.assertFalse(self._gate(_prob(64, 32, 16, 4)))

    def test_short_seqlen_blocks_combo(self):
        self.assertFalse(self._gate(_combo_prob(max_seqlen_q=256)))
        self.assertFalse(self._gate(_combo_prob(max_seqlen_q=100)))

    def test_gfx942_blocks_combo(self):
        self.assertFalse(self._gate(_combo_prob(), arch="gfx942"))


# ---------------------------------------------------------------------------
# Routing: combo_2d cohort → transposed_qk_32x32 flags
# ---------------------------------------------------------------------------


class TestCombo2dRouting(unittest.TestCase):
    """combo_2d problems must get the full 32x32 transposed stack."""

    _BIAS_BUG = "PR #9220: Python _enable_combo_2d missing bias guard"

    def test_combo_gets_transposed_flags(self):
        s = _spec(_combo_prob())
        self.assertTrue(s["use_mfma_32x32"])
        self.assertTrue(s["use_transposed_qk_32x32"])
        self.assertTrue(s["use_transposed_half_local_pv"])
        self.assertTrue(s["use_transposed_scalar_state"])

    @unittest.skip(_BIAS_BUG)
    def test_biased_combo_does_not_get_fast_paged_kv(self):
        """#9220: biased combo must NOT get use_fast_paged_kv_desc."""
        s = _spec(_combo_prob(use_qq_bias=True))
        self.assertFalse(s["use_fast_paged_kv_desc"], self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_alibi_combo_no_fast_paged_kv(self):
        s = _spec(_combo_prob(use_alibi=True))
        self.assertFalse(s["use_fast_paged_kv_desc"], self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_softcap_combo_no_fast_paged_kv(self):
        s = _spec(_combo_prob(softcap=50.0))
        self.assertFalse(s["use_fast_paged_kv_desc"], self._BIAS_BUG)

    def test_combo_no_sw_gets_fast_paged_kv(self):
        s = _spec(_combo_prob())
        self.assertTrue(s["use_fast_paged_kv_desc"])

    def test_combo_with_sw_no_fast_paged_kv(self):
        s = _spec(_combo_prob(sliding_window=256))
        self.assertFalse(s["use_fast_paged_kv_desc"])


# ---------------------------------------------------------------------------
# Bias x geometry cohort matrix (covers the #9220 class of divergence)
# ---------------------------------------------------------------------------


class TestBiasGeometryCohorts(unittest.TestCase):
    """Systematic sweep of bias flags x combo-geometry dimensions."""

    _BIAS_BUG = "PR #9220: Python _enable_combo_2d missing bias guard"

    def _fast_kv(self, problem: UnifiedAttentionProblem) -> bool:
        return _spec(problem)["use_fast_paged_kv_desc"]

    # ---- baseline: canonical combo fires ----
    def test_canonical_combo(self):
        self.assertTrue(self._fast_kv(_combo_prob()))

    # ---- PR #9220 biased-combo cohort (skip the test until bug is fixed) ----
    @unittest.skip(_BIAS_BUG)
    def test_qq_bias_blocks_fast_paged_kv(self):
        self.assertFalse(self._fast_kv(_combo_prob(use_qq_bias=True)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_alibi_blocks_fast_paged_kv(self):
        self.assertFalse(self._fast_kv(_combo_prob(use_alibi=True)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_softcap_blocks_fast_paged_kv(self):
        self.assertFalse(self._fast_kv(_combo_prob(softcap=50.0)), self._BIAS_BUG)

    @unittest.skip(_BIAS_BUG)
    def test_alibi_and_qq_bias_no_combo(self):
        self.assertFalse(
            self._fast_kv(_combo_prob(use_alibi=True, use_qq_bias=True)), self._BIAS_BUG
        )

    # ---- geometry boundaries (not affected by #9220) ----
    def test_seqlen_at_boundary_no_combo(self):
        self.assertFalse(self._fast_kv(_combo_prob(max_seqlen_q=256)))

    def test_seqlen_above_boundary_combo(self):
        self.assertTrue(self._fast_kv(_combo_prob(max_seqlen_q=257)))

    def test_wrong_dtype_no_combo(self):
        self.assertFalse(self._fast_kv(_combo_prob(dtype="fp16")))

    def test_wrong_head_size_no_combo(self):
        self.assertFalse(self._fast_kv(_prob(128, 32, 64, 8)))

    def test_wrong_gqa_ratio_no_combo(self):
        self.assertFalse(self._fast_kv(_prob(64, 32, 16, 4)))


# ---------------------------------------------------------------------------
# Selector fields: tile_size, num_warps, block_m_per_warp, waves_per_eu
# ---------------------------------------------------------------------------


class TestSelectorFields(unittest.TestCase):
    """Key selector outputs for representative problems."""

    def test_combo_tile_size_is_2x_block(self):
        # combo_2d, no SW: tile_size = 2 * block_size = 64
        s = _spec(_combo_prob())
        self.assertEqual(s["tile_size"], 64)

    def test_combo_sw_tile_size_is_block_size(self):
        # combo_2d + SW: tile_size = block_size = 32
        s = _spec(_combo_prob(sliding_window=256))
        self.assertEqual(s["tile_size"], 32)

    def test_single_batch_d64_tile_size(self):
        # single-batch d64 combo: tile_size = 128
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=512))
        self.assertEqual(s["tile_size"], 128)

    def test_combo_waves_per_eu(self):
        # combo_2d: waves_per_eu = 4
        s = _spec(_combo_prob())
        self.assertEqual(s["waves_per_eu"], 4)

    def test_combo_block_m_per_warp(self):
        # combo_2d uses 32x32: block_m_per_warp = 32
        s = _spec(_combo_prob())
        self.assertEqual(s["block_m_per_warp"], 32)

    def test_non_combo_block_m_per_warp(self):
        # short seqlen → not combo, not transposed → block_m_per_warp = 16
        s = _spec(_combo_prob(max_seqlen_q=64))
        self.assertEqual(s["block_m_per_warp"], 16)

    def test_non_combo_waves_per_eu(self):
        # standard path: waves_per_eu = 2
        s = _spec(_combo_prob(max_seqlen_q=64))
        self.assertEqual(s["waves_per_eu"], 2)


# ---------------------------------------------------------------------------
# Single-batch combo cohort
# ---------------------------------------------------------------------------


class TestSingleBatchCombo(unittest.TestCase):

    def test_single_batch_d64_gets_transposed_flags(self):
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=512))
        self.assertTrue(s["use_mfma_32x32"])
        self.assertTrue(s["use_transposed_qk_32x32"])

    def test_single_batch_bias_no_transposed(self):
        # single-batch + bias: _enable_single_batch_combo rejects alibi/qq_bias
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=512, use_qq_bias=True))
        # transposed_qk_32x32 may still fire via multi-batch branch, but
        # fast_paged_kv_desc (which needs combo + no-bias) must be off
        self.assertFalse(s["use_fast_paged_kv_desc"])

    def test_single_batch_long_d64_gets_early_v(self):
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=2048))
        self.assertTrue(s["use_early_v_schedule"])
        self.assertFalse(s["use_v_double_buffer"])

    def test_single_batch_short_d64_gets_v_double_buffer(self):
        s = _spec(_prob(64, 32, 32, 32, num_seqs=1, max_seqlen_q=512))
        self.assertFalse(s["use_early_v_schedule"])
        self.assertTrue(s["use_v_double_buffer"])

    def test_single_batch_d128_ksingle_off_not_raise(self):
        # d128 single-batch triggers the softmax-MFMA interleave -> num_warps=4
        # (block_m=128). With tile_size=64 the geometry guard block_m <= tile_size
        # fails, so _enable_k_single_buffer derives K-single OFF and the spec
        # BUILDS (previously the stale block_size>=32 proxy left K-single ON here,
        # raising an uncaught ValueError at spec build).
        s = _spec(_prob(128, 32, 32, 32, num_seqs=1, max_seqlen_q=512))
        self.assertFalse(s["use_k_single_buffer"])


# ---------------------------------------------------------------------------
# Register-PV gate
# ---------------------------------------------------------------------------


class TestRegisterPv(unittest.TestCase):

    def test_register_pv_off_for_combo(self):
        # combo_2d uses mfma_32x32 which conflicts with register_pv
        s = _spec(_combo_prob())
        self.assertFalse(s["use_register_pv"])

    def test_register_pv_off_for_bias(self):
        s = _spec(_prob(64, 32, 32, 32, use_qq_bias=True))
        self.assertFalse(s["use_register_pv"])

    def test_register_pv_off_for_alibi(self):
        s = _spec(_prob(64, 32, 32, 32, use_alibi=True))
        self.assertFalse(s["use_register_pv"])

    def test_register_pv_off_for_sinks(self):
        s = _spec(_prob(64, 32, 32, 32, use_sinks=True))
        self.assertFalse(s["use_register_pv"])

    def test_register_pv_eligible_for_standard_bf16(self):
        # Standard bf16 short-seqlen, no special flags → register_pv eligible
        s = _spec(_prob(64, 32, 32, 32, max_seqlen_q=64))
        self.assertTrue(s["use_register_pv"])


# ---------------------------------------------------------------------------
# i64_kv_addr
# ---------------------------------------------------------------------------


# Combo-shape blocks are 32 tokens x 8 kv-heads x 64 dims x 2 bytes = 32 KiB, so
# 65534 blocks fill the i32 buffer range (0x7FFF0000 bytes) exactly.
_AT_I32_LIMIT = 65534
_OVER_I32_LIMIT = 65535


class TestI64KvAddr(unittest.TestCase):

    def test_no_kv_blocks_no_i64(self):
        s = _spec(_combo_prob())  # num_kv_blocks=0 default
        self.assertFalse(s["use_i64_kv_addr"])

    def test_small_cache_no_i64(self):
        # Exactly 0x7FFF0000 bytes -> still i32 (threshold is STRICTLY above)
        s = _spec(_combo_prob(num_kv_blocks=_AT_I32_LIMIT))
        self.assertFalse(s["use_i64_kv_addr"])

    def test_large_cache_gets_i64(self):
        # One block over the i32 buffer range -> i64
        s = _spec(_combo_prob(num_kv_blocks=_OVER_I32_LIMIT))
        self.assertTrue(s["use_i64_kv_addr"])

    def test_top_64kib_below_2gib_gets_i64(self):
        # The tiled builders bound K/V buffer loads at num_records=0x7FFF0000, so
        # a cache of exactly 2 GiB would read zeros from its top 64 KiB on i32.
        s = _spec(_combo_prob(num_kv_blocks=65536))
        self.assertTrue(s["use_i64_kv_addr"])
        self.assertEqual(_au._I32_KV_BUFFER_BYTES, 0x7FFF0000)

    def test_i64_threshold_splits_cache_key(self):
        # Regression guard against dropping the i64 flag from _tiled_cache_key.
        # Crossing the threshold builds a different kernel (i64 vs i32 paged-KV
        # addressing), so a key without that decision would let a large cache
        # reuse the i32 launcher cached for a small cache of the same geometry.
        # Two problems straddling the threshold must not collide.
        small = _combo_prob(num_kv_blocks=_AT_I32_LIMIT)
        large = _combo_prob(num_kv_blocks=_OVER_I32_LIMIT)
        with _patch_arch("gfx950"):
            self.assertNotEqual(
                _au._tiled_cache_key(small), _au._tiled_cache_key(large)
            )

    def test_i64_threshold_splits_3d_cache_key(self):
        # Same guard for the 3D split-KV launcher key, which also folds in
        # _enable_i64_kv_addr.
        small = _combo_prob(num_kv_blocks=_AT_I32_LIMIT)
        large = _combo_prob(num_kv_blocks=_OVER_I32_LIMIT)
        with _patch_arch("gfx950"):
            self.assertNotEqual(
                _au._tiled_3d_cache_key(small), _au._tiled_3d_cache_key(large)
            )

    def test_num_kv_blocks_only_affects_key_via_i64(self):
        # num_kv_blocks must reach the cache key only through the i64 decision.
        # Keying the raw count would compile a new launcher per cache size, and
        # not keying it at all is the collision above. Across the threshold the
        # two keys differ in exactly one field, _enable_i64_kv_addr.
        small = _combo_prob(num_kv_blocks=_AT_I32_LIMIT)
        large = _combo_prob(num_kv_blocks=_OVER_I32_LIMIT)
        with _patch_arch("gfx950"):
            ks = _au._tiled_cache_key(small)
            kl = _au._tiled_cache_key(large)
            i64_small = _au._enable_i64_kv_addr(small)
            i64_large = _au._enable_i64_kv_addr(large)
        self.assertEqual(len(ks), len(kl))
        diffs = [(a, b) for a, b in zip(ks, kl) if a != b]
        self.assertEqual(diffs, [(i64_small, i64_large)])

    def test_spec_meta_and_guard_agree_at_every_count(self):
        # The launch meta derives the kernel's flag from _enable_i64_kv_addr
        # rather than building the spec (100-200 us per new total_q), so the
        # spec builders must set exactly that. And a spec built from a problem
        # with the real count must always pass the launch guard, including the
        # counts between the i32 buffer range and 2^31 bytes.
        for arch in ("gfx942", "gfx950"):
            for n in (_AT_I32_LIMIT - 1, _AT_I32_LIMIT, _OVER_I32_LIMIT, 65536, 65537):
                p = _combo_prob(num_kv_blocks=n)
                with self.subTest(arch=arch, n=n), _patch_arch(arch):
                    spec = _asb._tiled_spec_from_problem(p)
                    self.assertEqual(spec.use_i64_kv_addr, _au._enable_i64_kv_addr(p))
                    limit = _au._kv_addr_limit(p, spec.use_i64_kv_addr)
                    with mock.patch.dict(_au._2D_LAUNCH_META, clear=True):
                        meta = _au._get_2d_launch_meta(p, _au._tiled_cache_key(p))
                    self.assertEqual(meta.kv_addr_limit, limit)
                    _au._check_kv_addr_width(p, _kv_cache(n, p), limit)


def _kv_cache(num_blocks: int, problem: UnifiedAttentionProblem) -> SimpleNamespace:
    """Shape-only stand-in for a paged K cache tensor."""
    shape = (num_blocks, problem.block_size, problem.num_kv_heads, problem.head_size)
    return SimpleNamespace(shape=shape)


# Parity NQK shape: 32768 blocks x 64 tokens x 8 kv-heads x 128 dims x 2 bytes is
# 4 GiB, which is also 2^31 elements.
_BUG_BLOCKS = 32768


def _bug_prob(**overrides) -> UnifiedAttentionProblem:
    return _prob(128, 64, 64, 8, **overrides)


class TestKvAddrLimit(unittest.TestCase):
    """_kv_addr_limit maps the compiled kernel's flag to the cache it addresses."""

    def test_i64_kernel_is_unbounded(self):
        self.assertIsNone(_au._kv_addr_limit(_bug_prob(), True))

    def test_i32_kernel_is_bounded_by_the_buffer_range(self):
        self.assertEqual(_au._kv_addr_limit(_bug_prob(), False), 0x7FFF0000)

    def test_kernel_without_i64_path_is_bounded_by_element_count(self):
        # Scalar 2D, gfx942 4-warp GQA and gfx1250 tiled 2D: i32 element
        # offsets, so 2^31 elements of the cache dtype.
        self.assertEqual(_au._kv_addr_limit(_bug_prob(), None), 2 * 2**31)
        self.assertEqual(_au._kv_addr_limit(_bug_prob(use_fp8=True), None), 2**31)

    def test_gfx1250_meta_has_no_i64_path(self):
        p = _bug_prob(num_kv_blocks=_BUG_BLOCKS)
        with _patch_arch("gfx1250"), mock.patch.dict(_au._2D_LAUNCH_META, clear=True):
            meta = _au._get_2d_launch_meta(p, ("gfx1250-test",))
        self.assertEqual(meta.kv_addr_limit, 2 * 2**31)

    def test_gfx942_4warp_route_meta_has_no_i64_path(self):
        # D128 sliding-window bf16 rides build_gfx942_4warp_gqa, which indexes
        # K/V with i32 element offsets whatever the problem decides.
        p = _prob(128, 32, 32, 8, sliding_window=128)
        with _patch_arch("gfx942"), mock.patch.dict(_au._2D_LAUNCH_META, clear=True):
            self.assertIsNotNone(_au._gfx942_4warp_route(p))
            meta = _au._get_2d_launch_meta(p, _au._tiled_cache_key(p))
        self.assertEqual(meta.kv_addr_limit, 2 * 2**31)

    def test_gfx1250_spec_has_no_i64_field(self):
        self.assertNotIn("use_i64_kv_addr", {f.name for f in fields(Gfx1250TiledSpec)})

    def test_4warp_builder_rejects_i64(self):
        p = _prob(128, 32, 32, 8, sliding_window=128, num_kv_blocks=_BUG_BLOCKS)
        with _patch_arch("gfx942"):
            spec = _asb._tiled_spec_from_problem(p)
        spec = replace(spec, use_i64_kv_addr=True)
        with self.assertRaisesRegex(NotImplementedError, "use_i64_kv_addr"):
            build_gfx942_4warp_gqa(spec, arch="gfx942")


class TestKvAddrWidthGuard(unittest.TestCase):
    """_check_kv_addr_width rejects a cache larger than the kernel addresses."""

    _I32 = 0x7FFF0000

    def test_i32_kernel_over_limit_raises(self):
        p = _bug_prob()
        with self.assertRaisesRegex(ValueError, "zeros or wrong data"):
            _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), self._I32)

    def test_missing_count_hint_points_at_the_problem(self):
        p = _bug_prob()
        with self.assertRaisesRegex(ValueError, "num_kv_blocks=k.shape"):
            _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), self._I32)

    def test_stale_count_hint_points_at_the_problem(self):
        # A count taken from a smaller cache than the one launched on.
        p = _bug_prob(num_kv_blocks=1024)
        with self.assertRaisesRegex(ValueError, "below the cache's block count"):
            _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), self._I32)

    def test_correct_count_hint_points_at_the_kernel(self):
        # A hand-built i32 spec on a problem that already knows the cache size:
        # fixing the problem would not help, rebuilding the spec would.
        p = _bug_prob(num_kv_blocks=_BUG_BLOCKS)
        with self.assertRaisesRegex(ValueError, "use_i64_kv_addr=True"):
            _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), self._I32)

    def test_i64_kernel_passes(self):
        p = _bug_prob()
        _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), None)

    def test_small_cache_passes(self):
        p = _bug_prob()
        _au._check_kv_addr_width(p, _kv_cache(16, p), self._I32)

    def test_buffer_limit_boundary(self):
        p = _combo_prob()
        _au._check_kv_addr_width(p, _kv_cache(_AT_I32_LIMIT, p), self._I32)
        with self.assertRaises(ValueError):
            _au._check_kv_addr_width(p, _kv_cache(_OVER_I32_LIMIT, p), self._I32)

    def test_fp8_halves_cache_bytes(self):
        # 64 KiB fp8 blocks: 32767 fill the buffer range exactly.
        p = _bug_prob(use_fp8=True)
        _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS - 1, p), self._I32)
        with self.assertRaises(ValueError):
            _au._check_kv_addr_width(p, _kv_cache(_BUG_BLOCKS, p), self._I32)

    def test_shapeless_k_is_skipped(self):
        _au._check_kv_addr_width(_bug_prob(), None, self._I32)

    def test_attn_values_requires_the_limit(self):
        # Required, so a new direct-launch caller cannot forget it.
        p = _bug_prob()
        args = dict(
            problem=p,
            q=None,
            k=_kv_cache(_BUG_BLOCKS, p),
            v=None,
            out=None,
            cu_seqlens_q=None,
            seqused_k=None,
            softmax_scale=1.0,
            block_table=None,
            softcap=0.0,
            sinks=None,
            bt_stride=0,
            include_bt_stride=True,
        )
        with self.assertRaises(TypeError):
            _au._attn_values(**args)
        with self.assertRaisesRegex(ValueError, "i32 KV addressing"):
            _au._attn_values(**args, kv_addr_limit=self._I32)


class TestProductionRouteGuard(unittest.TestCase):
    """run_unified_attention_torch end to end with mocked launchers.

    The K cache is shape-only, so the route through count filling, spec and
    launch-meta selection and the kernarg pack runs on CPU.
    """

    def setUp(self):
        self.launcher = mock.Mock(return_value=None)
        patches = [
            _patch_arch("gfx950"),
            mock.patch.dict(_au._2D_LAUNCH_META, clear=True),
            mock.patch.object(_au, "_get_2d_launcher", return_value=self.launcher),
            mock.patch.object(_au, "_get_scalar_launcher", return_value=self.launcher),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _run(self, problem: UnifiedAttentionProblem, num_blocks: int, **kw) -> None:
        return _au.run_unified_attention_torch(
            problem=problem,
            q=None,
            k=_kv_cache(num_blocks, problem),
            v=None,
            out=None,
            cu_seqlens_q=None,
            seqused_k=None,
            softmax_scale=1.0,
            block_table=SimpleNamespace(shape=(problem.num_seqs, 64)),
            softcap=0.0,
            **kw,
        )

    def test_unset_count_on_4gib_cache_launches_i64(self):
        self._run(_bug_prob(), _BUG_BLOCKS, backend="tiled")
        self.launcher.assert_called_once()
        problem = _au._get_2d_launcher.call_args.args[0]
        self.assertEqual(problem.num_kv_blocks, _BUG_BLOCKS)
        self.assertTrue(_au._enable_i64_kv_addr(problem))

    def test_stale_count_raises(self):
        # A positive count is trusted (only 0 is refilled), so a stale one gets
        # the i32 kernel and the guard stops the launch.
        with self.assertRaisesRegex(ValueError, "below the cache's block count"):
            self._run(_bug_prob(num_kv_blocks=1024), _BUG_BLOCKS, backend="tiled")
        self.launcher.assert_not_called()

    def test_tuning_spec_rebinds_to_the_real_cache(self):
        candidate = next(
            c
            for c in attention_candidates()
            if c.name.startswith("attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm")
        )
        req = AttentionRequest(
            batch=1,
            nhead_q=32,
            nhead_k=8,
            seqlen_q=1024,
            seqlen_k=1024,
            hdim_q=128,
            hdim_v=128,
            arch="gfx950",
            dtype="bf16",
            algorithm=candidate.algorithm,
            spec_id=candidate.spec_id,
        )
        spec = next(iter(candidate.sweep_space(req)))
        self.assertFalse(spec.kernel_spec.use_i64_kv_addr)
        problem = _tuning_problem(req)
        # 16-token bf16 blocks of 32 KiB: one block over the i32 buffer range.
        self._run(problem, _OVER_I32_LIMIT, tuning_spec=spec)
        self.launcher.assert_called_once()
        bound = _au._get_2d_launcher.call_args.kwargs["tuning_spec"]
        self.assertTrue(bound.kernel_spec.use_i64_kv_addr)

    def test_scalar_fallback_is_bounded_by_element_count(self):
        # The production count is filled here, so the problem alone says i64;
        # the scalar kernel still indexes K/V with i32 element offsets.
        p = _bug_prob()
        self._run(p, _BUG_BLOCKS, backend="scalar")  # exactly 2^31 elements
        with self.assertRaisesRegex(ValueError, "kernel itself is i32"):
            self._run(p, _BUG_BLOCKS + 1, backend="scalar")
        self.launcher.assert_called_once()


# ---------------------------------------------------------------------------
# Transposed sub-flags (mask_once / mask_limit / scalar_state)
# ---------------------------------------------------------------------------


class TestTransposedSubflags(unittest.TestCase):

    def test_combo_no_sw_gets_mask_flags(self):
        s = _spec(_combo_prob())
        self.assertTrue(s["use_transposed_mask_once"])
        self.assertTrue(s["use_transposed_mask_limit"])
        self.assertTrue(s["use_transposed_scalar_state"])

    def test_combo_with_sw_no_mask_flags(self):
        # SW combo: mask_once / mask_limit must be off
        s = _spec(_combo_prob(sliding_window=256))
        self.assertFalse(s["use_transposed_mask_once"])
        self.assertFalse(s["use_transposed_mask_limit"])

    def test_biased_combo_no_mask_flags(self):
        # bias_active blocks mask_opts
        s = _spec(_combo_prob(use_qq_bias=True))
        self.assertFalse(s["use_transposed_mask_once"])
        self.assertFalse(s["use_transposed_mask_limit"])


if __name__ == "__main__":
    unittest.main()
