# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Benchmark lifecycle: validation-before-isolation, stable shards, FLOPs."""

from __future__ import annotations

import ast
import inspect
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from dispatch.attention import (
    ATTENTION_EXECUTION_REGISTRY,
    AttentionRequest,
    attention_dispatch_result,
)
from benchmarks.common.attention_flops import attention_flops
from benchmarks.common import attention_combo_sweep as sweep
from benchmarks.gfx950.attention.decode import decode_table_sweep
from benchmarks.gfx950.attention.prefill import dense_prefill_table_sweep


def _req(**kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1024,
        seqlen_k=1024,
        hdim_q=128,
        hdim_v=128,
        arch="gfx950",
        dtype="bf16",
        mask_type=1,
    )
    base.update(kw)
    return AttentionRequest(**base)


def _args(**kw):
    base = dict(
        arch="gfx950",
        dtype="bf16",
        batch=1,
        heads=32,
        kv_heads=8,
        head_dim=[128],
        seqlen_q=[1024],
        seqlen_k=[1024],
        kv_block_size=16,
        sliding_window=0,
        num_cus=0,
        dense_waves_per_eu=0,
        causal=True,
        candidate_prefix="attention_gfx950_dense",
        tuning_id_prefix="",
        offset=0,
        limit=0,
        isolate=True,
        output_jsonl="",
        progress=False,
        top=0,
        verbose_errors=False,
        warmup=3,
        iters=10,
        benchmark_iterations=5,
        seed=7,
        tolerance=0.03,
        no_check=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestComboSweepLifecycle(unittest.TestCase):
    def test_module_does_not_mutate_sys_path(self):
        source = inspect.getsource(sweep)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "sys"
                and node.attr == "path"
            ):
                self.fail("attention_combo_sweep mutates sys.path")
        self.assertNotIn("sys.path.insert", source)
        self.assertNotIn("ROCKE_ROOT", source)
        self.assertNotIn("PYTHONPATH", source)

    def test_theoretical_flops_are_not_the_padded_rectangle(self):
        req = _req()
        causal = sweep._flops(req)
        full = attention_flops(
            req.batch,
            req.nhead_q,
            req.hdim_q,
            req.seqlen_q,
            req.seqlen_k,
            causal=False,
            sliding_window=0,
        )
        self.assertEqual(
            causal,
            attention_flops(
                req.batch,
                req.nhead_q,
                req.hdim_q,
                req.seqlen_q,
                req.seqlen_k,
                causal=True,
                sliding_window=0,
            ),
        )
        self.assertLess(causal, full)

    def test_repeated_timing_prepares_once_checks_once_and_uses_median(self):
        class Tensor:
            def reshape_as(self, _other):
                return self

            def float(self):
                return self

            def __sub__(self, _other):
                return self

            def abs(self):
                return self

            def max(self):
                return self

            def item(self):
                return 0.01

        tensor = Tensor()
        tensors = {
            "q": tensor,
            "k": tensor,
            "v": tensor,
            "out": tensor,
            "_dense_q": tensor,
            "_dense_k": tensor,
            "_dense_v": tensor,
        }
        binding = SimpleNamespace(launch=mock.Mock())
        result = SimpleNamespace(
            candidate=object(),
            spec=object(),
            bind_torch=mock.Mock(return_value=binding),
        )
        values = [0.10, 0.05, 0.07, 0.06, 0.04]
        cuda = SimpleNamespace(
            current_stream=mock.Mock(return_value=SimpleNamespace(cuda_stream=7)),
            synchronize=mock.Mock(),
        )
        with (
            mock.patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}),
            mock.patch.object(sweep, "_row_skeleton", return_value={"kind": "dense"}),
            mock.patch.object(sweep, "_dense_tensors", return_value=tensors) as prepare,
            mock.patch.object(sweep, "_reference", return_value=tensor) as reference,
            mock.patch("rocke.runtime.time_launches", side_effect=values) as timing,
            mock.patch("rocke.runtime.synchronize_and_release"),
        ):
            row = sweep._run_result(
                _req(),
                result,
                _args(warmup=15, iters=50, benchmark_iterations=5),
                0,
            )
        prepare.assert_called_once()
        result.bind_torch.assert_called_once()
        reference.assert_called_once()
        self.assertEqual(timing.call_count, 5)
        self.assertEqual(binding.launch.call_count, 1)
        self.assertEqual(row["ms"], 0.06)
        self.assertEqual(row["timing"]["excluded_initial_iterations"], 1)
        self.assertEqual(row["timing"]["warmup_executions_per_iteration"], 15)
        self.assertEqual(row["timing"]["timed_executions_per_iteration"], 50)

    def test_offset_limit_preserve_absolute_indices(self):
        idxs = [i for i, _req, _res in sweep.iter_shard(_args(offset=1, limit=2))]
        self.assertEqual(idxs, [1, 2])

    def test_isolated_child_keeps_outer_timing_count(self):
        result = SimpleNamespace(candidate=SimpleNamespace(name="candidate"))
        argv = sweep._child_argv(
            _args(benchmark_iterations=7),
            _req(),
            result,
        )
        self.assertEqual(argv[argv.index("--benchmark-iterations") + 1], "7")

    def test_diagonal_reaches_request_and_isolated_child(self):
        """Bottom-right by default (the corner unified always masks); top-left on
        request. The isolated child must be told the same corner, or it checks
        against a different reference than the parent validated."""
        result = SimpleNamespace(candidate=SimpleNamespace(name="candidate"))
        cases = (
            (_args(), 2, "bottom_right"),
            (_args(diagonal="top_left"), 1, "top_left"),
            (_args(diagonal="bottom_right"), 2, "bottom_right"),
            (_args(causal=False), 0, "bottom_right"),
        )
        for args, mask_type, diagonal in cases:
            with self.subTest(mask_type=mask_type):
                req = next(sweep._requests(args))
                self.assertEqual(req.mask_type, mask_type)
                argv = sweep._child_argv(args, req, result)
                self.assertEqual(argv[argv.index("--diagonal") + 1], diagonal)

    def test_reference_anchors_the_diagonal_at_the_requested_corner(self):
        torch = __import__("pytest").importorskip("torch")
        sq, sk, h, d = 2, 4, 1, 8
        q = torch.zeros(1, sq, h, d)
        k = torch.zeros(1, sk, h, d)
        # V row j holds the value j, so with uniform scores each output is the mean
        # of the attended key indices.
        v = torch.arange(sk, dtype=torch.float32).view(1, sk, 1, 1).expand(1, sk, h, d)
        top_left = sweep._reference(
            q, k, v, causal=True, sliding_window=0, bottom_right=False
        )
        bottom_right = sweep._reference(
            q, k, v, causal=True, sliding_window=0, bottom_right=True
        )
        # Top-left: row 0 sees {0}, row 1 sees {0, 1}.
        self.assertEqual(top_left[0, :, 0, 0].tolist(), [0.0, 0.5])
        # Bottom-right (offset 2): row 0 sees {0, 1, 2}, row 1 sees all 4 keys.
        self.assertEqual(bottom_right[0, :, 0, 0].tolist(), [1.0, 1.5])

    def test_dense_wpe_reaches_request_and_isolated_child(self):
        args = _args(dense_waves_per_eu=3)
        req = next(sweep._requests(args))
        self.assertEqual(req.dense_waves_per_eu, 3)
        result = SimpleNamespace(candidate=SimpleNamespace(name="candidate"))
        argv = sweep._child_argv(args, req, result)
        self.assertEqual(argv[argv.index("--dense-waves-per-eu") + 1], "3")

    def test_invalid_host_validation_does_not_isolate_or_init_torch(self):
        req = _req(algorithm="attention_dense")
        candidate = ATTENTION_EXECUTION_REGISTRY.get("attention_gfx950_dense")
        spec = candidate.select_spec(req)
        result = attention_dispatch_result(req, candidate, spec)
        args = _args()
        with (
            mock.patch.object(sweep, "iter_shard", return_value=[(0, req, result)]),
            mock.patch.object(
                sweep,
                "validate_config",
                return_value=sweep.Validation("IR verification failed"),
            ),
            mock.patch.object(sweep, "init_torch_first") as init_torch,
            mock.patch.object(sweep.subprocess, "run") as run,
        ):
            rc = sweep.sweep(args)
        self.assertEqual(rc, 1)
        run.assert_not_called()
        init_torch.assert_not_called()

    def test_no_admitted_candidate_is_explicitly_unsupported(self):
        emitted = []
        args = _args()
        with (
            mock.patch.object(sweep, "_iter_results", return_value=()),
            mock.patch.object(
                sweep,
                "_emit",
                side_effect=lambda row, *_args: emitted.append(row),
            ),
            mock.patch.object(sweep, "init_torch_first") as init_torch,
            mock.patch.object(sweep.subprocess, "run") as run,
        ):
            rc = sweep.sweep(args)
        self.assertEqual(rc, 0)
        self.assertEqual([row["status"] for row in emitted], ["unsupported"])
        run.assert_not_called()
        init_torch.assert_not_called()

    def test_host_validate_reports_support_failures(self):
        req = _req(algorithm="attention_dense")
        candidate = ATTENTION_EXECUTION_REGISTRY.get("attention_gfx950_dense")
        spec = candidate.select_spec(req)
        result = attention_dispatch_result(
            replace(req, arch="gfx1250"), candidate, spec
        )
        reason = sweep.host_validate(result)
        self.assertIsNotNone(reason)

    def test_host_validate_probes_opt_in_candidates(self):
        req = _req(algorithm="auto")
        candidate = next(
            c
            for c in ATTENTION_EXECUTION_REGISTRY.candidates()
            if c.name.startswith("attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm")
        )
        spec = candidate.select_spec(
            replace(req, algorithm=candidate.algorithm, spec_id=candidate.spec_id)
        )
        result = attention_dispatch_result(req, candidate, spec)
        with (
            mock.patch.object(
                type(result), "build", return_value=SimpleNamespace(name="k")
            ),
            mock.patch("rocke.core.verify.verify_or_raise"),
            mock.patch.object(sweep, "_lower_kernel", side_effect=RuntimeError("boom")),
        ):
            reason = sweep.host_validate(result)
        self.assertIsNotNone(reason)
        self.assertIn("boom", reason)

    def test_offset_limit_window_is_per_shape(self):
        shapes = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]

        def results(req, _args):
            return [SimpleNamespace(shape=req.name, index=i) for i in range(4)]

        with (
            mock.patch.object(sweep, "_requests", return_value=shapes),
            mock.patch.object(sweep, "_iter_results", side_effect=results),
        ):
            low = list(sweep.iter_shard(_args(offset=0, limit=2)))
            high = list(sweep.iter_shard(_args(offset=2, limit=2)))
        low_keys = {(row[1].name, row[0]) for row in low}
        high_keys = {(row[1].name, row[0]) for row in high}
        self.assertEqual(low_keys, {("a", 0), ("a", 1), ("b", 0), ("b", 1)})
        self.assertEqual(high_keys, {("a", 2), ("a", 3), ("b", 2), ("b", 3)})
        self.assertTrue(low_keys.isdisjoint(high_keys))

    def test_dtype_choices_reject_fp32(self):
        for module in (sweep, dense_prefill_table_sweep, decode_table_sweep):
            with self.subTest(module=module.__name__):
                with (
                    mock.patch.object(
                        sys, "argv", [module.__name__, "--dtype", "fp32"]
                    ),
                    self.assertRaises(SystemExit) as raised,
                ):
                    module.main()
                self.assertEqual(raised.exception.code, 2)

    def test_table_sweeps_rewrite_json_after_each_row(self):
        import json
        import tempfile
        from pathlib import Path

        for module in (dense_prefill_table_sweep, decode_table_sweep):
            with self.subTest(module=module.__name__):
                with tempfile.TemporaryDirectory() as tmp:
                    path = str(Path(tmp) / "rows.json")
                    args = SimpleNamespace(output_json=path)
                    rows: list[dict] = []
                    module._store_row(rows, {"i": 1}, args)
                    self.assertEqual(json.loads(Path(path).read_text()), [{"i": 1}])
                    module._store_row(rows, {"i": 2}, args)
                    self.assertEqual(
                        json.loads(Path(path).read_text()), [{"i": 1}, {"i": 2}]
                    )

    def test_table_sweeps_fail_only_for_admitted_execution_failures(self):
        for module in (dense_prefill_table_sweep, decode_table_sweep):
            with self.subTest(module=module.__name__):
                self.assertEqual(
                    module._rows_exit_code(
                        [{"status": "ok"}, {"status": "unsupported"}]
                    ),
                    0,
                )
                for status in ("error", "invalid", "mismatch", "crash", "timeout"):
                    self.assertEqual(
                        module._rows_exit_code([{"status": status}]),
                        1,
                        status,
                    )


if __name__ == "__main__":
    unittest.main()
