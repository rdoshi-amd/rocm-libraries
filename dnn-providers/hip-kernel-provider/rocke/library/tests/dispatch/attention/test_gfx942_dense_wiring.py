# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the gfx942 ``attention_dense`` dispatcher wiring.

Required by ``library/dispatch/AGENTS.md`` step 4. Covers:
  - the candidate is registered and discoverable, with the right spec_id/algorithm
  - OPT-IN ONLY: ``algorithm="auto"`` never selects it, despite priority 3 outranking
    every other attention candidate
  - ``spec_id`` is an equivalent opt-in door
  - routing on gfx942, and rejection of every out-of-scope request
  - ``dense_persistent``: 'auto' turns the persistent grid on once there is enough
    work; an explicit 'on' is accepted
  - non-persistent gfx942 dense reads ``batch`` / ``seqlen_q`` / ``seqlen_kv`` as
    runtime kernel params, so those fields drop out of ``kernel_name()`` and the
    dispatched signature includes them. The persistent grid still bakes batch.

The priority-3 tests are the load-bearing ones: the arm sorts ahead of every other
candidate, so the opt-in check is the ONLY thing keeping a correctness-first P0 kernel
off the default gfx942 path.
"""

from __future__ import annotations

import unittest

import kernels.common.attention_unified as au
from dispatch.attention import (
    AttentionMaskType,
    AttentionRequest,
    attention_candidates,
    dispatch_attention,
    registered_attention_combos,
)

# gfx942's own spec factory. NOT the package-level ``dense_spec_for_request``,
# which is gfx950's and would hand back an untuned spec for a gfx942 request.
from dispatch.attention.gfx942 import _dense_spec
from kernels.common.attention_dense_spec import AttentionDenseSpec
from kernels.gfx942.attention_dense import (
    Gfx942AttentionDenseSpec,
    build_attention_dense,
    supports_attention_dense,
)

_NAME = "attention_gfx942_dense"
_SPEC_ID = "gfx942_attention_dense"


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=128,
        nhead_k=8,
        seqlen_q=2048,
        seqlen_k=2048,
        hdim_q=128,
        hdim_v=128,
        arch="gfx942",
        dtype="bf16",
        mask_type=1,
        algorithm="attention_dense",
    )
    base.update(kw)
    return AttentionRequest(**base)


def _candidate():
    return next(c for c in attention_candidates() if c.name == _NAME)


class _Gfx942Arch:
    """Pin _RESOLVED_ATTENTION_ARCH so routing does not depend on the host GPU."""

    def __enter__(self):
        self._old = au._RESOLVED_ATTENTION_ARCH
        au._RESOLVED_ATTENTION_ARCH = "gfx942"
        return self

    def __exit__(self, *_):
        au._RESOLVED_ATTENTION_ARCH = self._old


class TestGfx942DenseRegistration(unittest.TestCase):
    def test_candidate_is_registered(self):
        self.assertIn(_NAME, [c.name for c in attention_candidates()])

    def test_spec_id_and_algorithm(self):
        c = _candidate()
        self.assertEqual(c.spec_id, _SPEC_ID)
        self.assertEqual(c.algorithm, "attention_dense")

    def test_priority_outranks_every_other_candidate(self):
        """Documents WHY the opt-in gate matters: nothing else holds this arm back."""
        c = _candidate()
        others = [o for o in attention_candidates() if o.name != _NAME]
        self.assertTrue(all(c.priority <= o.priority for o in others))


class TestGfx942DenseOptIn(unittest.TestCase):
    def test_auto_algorithm_never_selects_it(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(algorithm="auto", spec_id="auto"))
            self.assertFalse(ok, "attention_dense must never be auto-selected")
            self.assertIn("opt-in", why)
            routed = dispatch_attention(_req(algorithm="auto", spec_id="auto"))
            self.assertNotEqual(routed.candidate.name, _NAME)

    def test_spec_id_is_an_equivalent_opt_in(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(algorithm="auto", spec_id=_SPEC_ID))
            self.assertTrue(ok, why)

    def test_routes_on_explicit_algorithm(self):
        with _Gfx942Arch():
            r = dispatch_attention(_req())
            self.assertEqual(r.candidate.name, _NAME)
            from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec

            self.assertIsInstance(r.spec, Gfx942AttentionDenseSpec)
            self.assertIn("gfx942", r.spec.kernel_name())
            self.assertNotEqual(r.grid, (0, 0, 0))


class TestGfx942DenseSupportGates(unittest.TestCase):
    """Arch, dtype and feature rejections are the declared ``Capability``'s job;
    only what capability cannot express as data stays in the predicate. Each test
    below asserts which of the two turned the request down, so a gate silently
    migrating between them is a failure rather than a rename."""

    def test_rejects_non_gfx942_arch(self):
        ok, why = _candidate().admits(_req(arch="gfx950"))
        self.assertFalse(ok)
        self.assertIn("capability", why)
        self.assertIn("gfx942", why)

    def test_rejects_unsupported_dtype(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(dtype="fp8"))
            self.assertFalse(ok)
            self.assertIn("capability", why)
            self.assertIn("fp8", why)

    def test_admits_sliding_window(self):
        with _Gfx942Arch():
            ok, _ = _candidate().admits(_req(sliding_window=64))
            self.assertTrue(ok)

    def test_rejects_sinks(self):
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(use_sinks=True))
            self.assertFalse(ok)
            self.assertIn("capability", why)
            self.assertIn("sinks", why)

    def test_rejects_ragged_sequence_length(self):
        """_dense_spec sets ragged=True for any non-256-multiple self-attention
        length -- most real serving shapes. The kernel must decline, not
        select-then-fail. Capability cannot see this one: it is a property of the
        BUILT spec, so it stays in the predicate."""
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(seqlen_q=1000, seqlen_k=1000))
            self.assertFalse(ok)
            self.assertNotIn("capability", why)
            self.assertIn("ragged", why)


class TestGfx942BottomRight(unittest.TestCase):
    """Moving bottom-right diagonal (seqlen_q != seqlen_kv), mirroring gfx950's
    ``TestDenseBottomRightWiring``. Unlike gfx950, gfx942 serves it on BOTH grids:
    the persistent and default grids share one work-item body, so the persistent
    policy is not overridden, and the default grid keeps runtime shape params."""

    _MOVING = (AttentionMaskType.BOTTOM_RIGHT_CAUSAL, 2)

    def test_capability_declares_bottom_right(self):
        self.assertIn("causal_bottom_right", _candidate().capability.supports_features)

    def test_moving_bottom_right_is_admitted_on_both_grids(self):
        with _Gfx942Arch():
            for mask_type in self._MOVING:
                for persistent in ("off", "on"):
                    with self.subTest(mask_type=mask_type, persistent=persistent):
                        req = _req(
                            seqlen_q=2048,
                            seqlen_k=4096,
                            mask_type=mask_type,
                            dense_persistent=persistent,
                        )
                        ok, why = _candidate().admits(req)
                        self.assertTrue(ok, why)
                        spec = _dense_spec(req)
                        self.assertTrue(spec.causal)
                        self.assertTrue(spec.causal_bottom_right)
                        self.assertEqual(spec.persistent, persistent == "on")
                        self.assertIn("br", spec.kernel_name().split("_"))
                        self.assertEqual(dispatch_attention(req).candidate.name, _NAME)

    def test_auto_persistent_policy_is_unchanged_for_moving_bottom_right(self):
        with _Gfx942Arch():
            spec = _dense_spec(
                _req(
                    seqlen_q=8192,
                    seqlen_k=12288,
                    mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
                    dense_persistent="auto",
                )
            )
            self.assertTrue(spec.causal_bottom_right)
            self.assertTrue(spec.persistent)

    def test_default_grid_serves_every_bottom_right_shape_with_one_symbol(self):
        """The default-grid body reads the diagonal offset from its shape params,
        so bottom-right stays in the runtime-shape cohort: one symbol per config,
        distinct from the top-left symbol."""
        shapes = ((2048, 4096), (1024, 4096), (256, 320))
        with _Gfx942Arch():
            specs = [
                _dense_spec(
                    _req(
                        seqlen_q=sq,
                        seqlen_k=sk,
                        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
                        dense_persistent="off",
                    )
                )
                for sq, sk in shapes
            ]
            top_left = _dense_spec(
                _req(seqlen_q=2048, seqlen_k=4096, dense_persistent="off")
            )
        self.assertTrue(all(s.runtime_shape for s in specs))
        self.assertEqual(len({s.kernel_name() for s in specs}), 1)
        self.assertNotRegex(specs[0].kernel_name(), r"_sq\d+|_sk\d+")
        self.assertNotEqual(specs[0].kernel_name(), top_left.kernel_name())

    def test_persistent_grid_bakes_the_offset_into_the_symbol(self):
        with _Gfx942Arch():
            names = {
                _dense_spec(
                    _req(
                        seqlen_q=2048,
                        seqlen_k=sk,
                        mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
                        dense_persistent="on",
                    )
                ).kernel_name()
                for sk in (2304, 4096)
            }
        self.assertEqual(len(names), 2)

    def test_concrete_support_admits_shared_bottom_right_spec(self):
        common = dict(
            batch=1,
            seqlen_q=2048,
            seqlen_kv=4096,
            num_query_heads=128,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            causal_bottom_right=True,
            dtype="bf16",
        )
        ok, why = supports_attention_dense(AttentionDenseSpec(**common), arch="gfx942")
        self.assertTrue(ok, why)
        self.assertTrue(Gfx942AttentionDenseSpec(**common).causal_bottom_right)

    def test_bottom_right_with_sliding_window_is_rejected(self):
        """A spec-level rejection the feature set cannot express, so it is turned
        down by the predicate, not by capability."""
        with _Gfx942Arch():
            ok, why = _candidate().admits(
                _req(
                    seqlen_q=2048,
                    seqlen_k=4096,
                    mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
                    sliding_window=128,
                )
            )
        self.assertFalse(ok)
        self.assertNotIn("capability", why)
        self.assertIn("sliding_window", why)
        common = dict(
            batch=1,
            seqlen_q=2048,
            seqlen_kv=4096,
            num_query_heads=128,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            causal_bottom_right=True,
            sliding_window=128,
        )
        ok, why = supports_attention_dense(AttentionDenseSpec(**common), arch="gfx942")
        self.assertFalse(ok)
        self.assertIn("sliding_window", why)

    def test_query_longer_than_kv_is_rejected(self):
        """Bottom-right with seqlen_q > seqlen_kv would leave fully masked rows."""
        with _Gfx942Arch():
            ok, why = _candidate().admits(
                _req(
                    seqlen_q=4096,
                    seqlen_k=2048,
                    mask_type=AttentionMaskType.BOTTOM_RIGHT_CAUSAL,
                )
            )
        self.assertFalse(ok)
        self.assertNotIn("capability", why)
        self.assertIn("seqlen_q <= seqlen_kv", why)

    def test_equal_length_bottom_right_preserves_persistent_policy(self):
        common = dict(
            seqlen_q=8192,
            seqlen_k=8192,
            dense_persistent="auto",
        )
        mask_pairs = (
            (AttentionMaskType.TOP_LEFT_CAUSAL, 2),
            (1, AttentionMaskType.BOTTOM_RIGHT_CAUSAL),
        )
        with _Gfx942Arch():
            for top_left, bottom_right in mask_pairs:
                with self.subTest(top_left=top_left, bottom_right=bottom_right):
                    top_left_spec = _dense_spec(_req(mask_type=top_left, **common))
                    bottom_right_req = _req(mask_type=bottom_right, **common)
                    bottom_right_spec = _dense_spec(bottom_right_req)
                    self.assertEqual(bottom_right_spec, top_left_spec)
                    self.assertFalse(bottom_right_spec.causal_bottom_right)
                    self.assertTrue(bottom_right_spec.persistent)
                    ok, why = _candidate().admits(bottom_right_req)
                    self.assertTrue(ok, why)


class TestGfx942DensePersistent(unittest.TestCase):
    def test_auto_persistent_turns_on_for_large_sq(self):
        """Post-P4 (ledger row 16): 'auto' turns the persistent grid-stride variant
        ON once there is enough work to fill the grid -- the large-Sq prefill
        regime -- and the request is accepted."""
        with _Gfx942Arch():
            req = _req(seqlen_q=8192, seqlen_k=8192, dense_persistent="auto")
            ok, why = _candidate().admits(req)
            self.assertTrue(ok, why)
            self.assertTrue(_dense_spec(req).persistent)

    def test_explicit_persistent_on_is_accepted_and_builds_persistent(self):
        """Post-P4 the persistent variant ships, so an explicit 'on' is accepted
        and yields a genuinely persistent spec -- never silently downgraded to a
        default-grid kernel."""
        with _Gfx942Arch():
            req = _req(dense_persistent="on")
            ok, why = _candidate().admits(req)
            self.assertTrue(ok, why)
            self.assertTrue(_dense_spec(req).persistent)


class TestGfx942DenseWavesPerEu(unittest.TestCase):
    def test_shipped_policy_and_explicit_override(self):
        self.assertEqual(_dense_spec(_req()).waves_per_eu, 2)
        self.assertEqual(
            _dense_spec(_req(hdim_q=64, hdim_v=64)).waves_per_eu,
            4,
        )
        overridden = _dense_spec(_req(dense_waves_per_eu=3))
        self.assertEqual(overridden.waves_per_eu, 3)
        self.assertIn("wpe3", overridden.kernel_name())
        self.assertEqual(
            build_attention_dense(overridden, arch="gfx942").attrs["waves_per_eu"],
            3,
        )

    def test_invalid_override_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dense_waves_per_eu"):
            _dense_spec(_req(dense_waves_per_eu=9))

    def _swept_waves(self, level: str, *, pin: int = 0) -> set[int]:
        return {
            spec.waves_per_eu
            for _candidate, spec in registered_attention_combos(
                _req(dense_waves_per_eu=pin),
                candidate_prefix=_NAME,
                sweep_level=level,
            )
        }

    def test_production_and_full_sweeps_expand_wpe(self):
        self.assertEqual(self._swept_waves("production"), {2, 4})
        self.assertEqual(self._swept_waves("full"), {1, 2, 3, 4})

    def test_explicit_override_pins_sweep(self):
        self.assertEqual(self._swept_waves("production", pin=3), {3})
        self.assertEqual(self._swept_waves("full", pin=3), {3})


class TestGfx942DenseSpecIdentity(unittest.TestCase):
    def test_kernel_name_follows_the_runtime_shape_contract(self):
        """Non-persistent gfx942 dense takes batch and both seqlens as kernel
        params, so one name covers every batch. The persistent grid still bakes
        batch into the symbol, and the dispatched signature matches that split."""
        from kernels.gfx942.attention_dense import attention_dense_signature

        with _Gfx942Arch():
            runtime = [
                dispatch_attention(_req(batch=b, dense_persistent="off")).spec
                for b in (1, 2, 4)
            ]
            self.assertTrue(all(s.runtime_shape for s in runtime))
            self.assertEqual(len({s.kernel_name() for s in runtime}), 1)
            self.assertNotRegex(runtime[0].kernel_name(), r"_b\d+")
            names = [p["name"] for p in attention_dense_signature(runtime[0])]
            self.assertEqual(
                names,
                [
                    "q_ptr",
                    "k_ptr",
                    "v_ptr",
                    "o_ptr",
                    "scale",
                    "batch",
                    "seqlen_q",
                    "seqlen_kv",
                ],
            )

            baked = [
                dispatch_attention(_req(batch=b, dense_persistent="on")).spec
                for b in (1, 2, 4)
            ]
            self.assertTrue(all(not s.runtime_shape for s in baked))
            self.assertEqual(len({s.kernel_name() for s in baked}), 3)
            baked_names = [p["name"] for p in attention_dense_signature(baked[0])]
            self.assertEqual(baked_names, ["q_ptr", "k_ptr", "v_ptr", "o_ptr", "scale"])

    def test_support_implies_the_dispatched_spec_builds(self):
        """The dispatch-level half of the supports/build contract: the spec the
        dispatcher actually selects (persistent auto-on for this large-Sq shape,
        post-P4) is exactly what the builder emits."""
        with _Gfx942Arch():
            req = _req()
            self.assertTrue(_candidate().admits(req)[0])
            spec = _dense_spec(req)
            kd = build_attention_dense(spec, arch="gfx942")
            self.assertEqual(kd.name, dispatch_attention(req).spec.kernel_name())


class TestGfx942SlidingWindow(unittest.TestCase):
    """Sliding-window pass-through and capability tests, mirroring gfx950's suite."""

    def test_sliding_window_zero_by_default(self):
        with _Gfx942Arch():
            spec = _dense_spec(_req())
            self.assertEqual(spec.sliding_window, 0)

    def test_sliding_window_passes_through_to_spec(self):
        with _Gfx942Arch():
            spec = _dense_spec(_req(sliding_window=128))
            self.assertEqual(spec.sliding_window, 128)

    def test_sliding_window_appears_in_kernel_name(self):
        with _Gfx942Arch():
            spec = _dense_spec(_req(sliding_window=256))
            self.assertIn("swa256", spec.kernel_name())

    def test_different_window_sizes(self):
        with _Gfx942Arch():
            for window in (64, 128, 256):
                spec = _dense_spec(_req(sliding_window=window))
                self.assertEqual(spec.sliding_window, window)

    def test_sliding_window_in_supports_features(self):
        self.assertIn("sliding_window", _candidate().capability.supports_features)

    def test_sliding_window_requires_causal(self):
        """sliding_window without causal is rejected by _dense_spec (spec validates it)."""
        with _Gfx942Arch():
            ok, why = _candidate().admits(_req(sliding_window=128, mask_type=0))
            self.assertFalse(ok)
            self.assertNotIn("capability", why)


if __name__ == "__main__":
    unittest.main()
