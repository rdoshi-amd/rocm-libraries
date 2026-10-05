# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side tests for stream-K implicit-GEMM backward-weight convolution.

No GPU needed. Covers the spec surface (defaults, kernel-name tags), the
validator gates and their agreement between ``validate()``,
``is_valid_wgrad_spec`` and the C++ port, admits-implies-builds over the
mode x reduction x arch x groups matrix, the host-side launch plan (grid,
workspace layout, determinism) against the C port, and byte-identical lowering
between the two engines through the ``rocke_engine`` binding.

GPU numerics live in ``test_conv_wgrad_streamk_gpu.py``.
"""

from __future__ import annotations

import dataclasses
import unittest
from dataclasses import replace

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    build_implicit_gemm_conv_wgrad,
    is_valid_wgrad_spec,
    wgrad_streamk_grid,
    wgrad_streamk_partition,
    wgrad_streamk_signature,
    wgrad_streamk_workspace_layout,
    wgrad_streamk_workspace_nbytes,
)

_DENSE = ConvProblem(N=2, Hi=12, Wi=12, C=32, K=96, Y=3, X=3, pH=1, pW=1)
# kpg = 40 is not a multiple of tile_m, so the last M tile of every group is
# partial: the case CK's own GemmM * GemmBatch fold leaves untested.
_GROUPED = ConvProblem(N=2, Hi=10, Wi=10, C=48, K=120, Y=3, X=3, pH=1, pW=1, groups=3)
# Depthwise: the only shape group merging admits, so its stream-K gate is reached.
_DW = ConvProblem(N=2, Hi=8, Wi=8, C=16, K=16, Y=3, X=3, pH=1, pW=1, groups=16)


def _spec(problem=_DENSE, *, dtype_d="fp16", edge=32, **kw) -> WgradConvSpec:
    base = dict(
        problem=problem,
        data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d=dtype_d),
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=edge,
        warp_tile_n=edge,
        warp_tile_k=16,
        streamk="dp_sk",
        streamk_ctas=6,
    )
    base.update(kw)
    return WgradConvSpec(**base)


def _engine():
    """The ``rocke_engine`` binding, or skip the calling test."""
    from rocke.core.backend import BackendError, _import_engine

    try:
        return _import_engine()
    except BackendError as e:
        raise unittest.SkipTest(f"C++ engine not importable: {e}") from None


def _spec_dict(spec: WgradConvSpec) -> dict:
    """Flatten a WgradConvSpec into the dict ``conv_wgrad_build_spec`` reads."""
    d = {
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec)
        if f.name not in ("problem", "data", "acc_epilogue", "lds_layout")
    }
    d = {k: v for k, v in d.items() if v is not None}
    d["problem"] = {
        k: v for k, v in dataclasses.asdict(spec.problem).items() if v is not None
    }
    d["dtype_a"] = spec.data.dtype_a
    d["dtype_b"] = spec.data.dtype_b
    d["dtype_d"] = spec.data.dtype_d
    return d


class TestStreamKSpecSurface(unittest.TestCase):
    def test_default_is_off_and_name_unchanged(self):
        plain = replace(_spec(), streamk="off", streamk_ctas=-1)
        self.assertNotIn("_sk", plain.kernel_name())
        self.assertEqual(
            plain.kernel_name(),
            WgradConvSpec(
                **{
                    f.name: getattr(plain, f.name)
                    for f in dataclasses.fields(plain)
                    if not f.name.startswith("streamk")
                }
            ).kernel_name(),
        )

    def test_name_tags_every_knob_that_changes_the_body(self):
        names = {
            _spec().kernel_name(),
            _spec(streamk_ctas=8).kernel_name(),
            _spec(streamk="persistent").kernel_name(),
            _spec(streamk_reduction="tree").kernel_name(),
            _spec(streamk_reduction="workspace").kernel_name(),
            _spec(dtype_d="fp32", streamk_reduction="atomic").kernel_name(),
            _spec(streamk_ctas=-1).kernel_name(),
        }
        self.assertEqual(len(names), 7, names)
        self.assertTrue(_spec().kernel_name().endswith("_sk6"))
        self.assertIn("_skauto", _spec(streamk_ctas=-1).kernel_name())

    def test_builder_resolves_auto_pool_into_the_name(self):
        k = build_implicit_gemm_conv_wgrad(_spec(streamk_ctas=-1), arch="gfx950")
        self.assertTrue(k.name.endswith("_sk256"), k.name)


class TestStreamKValidator(unittest.TestCase):
    """Every gate rejects in ``validate()`` and ``is_valid_wgrad_spec`` alike."""

    _REJECTS = [
        (dict(streamk="sideways"), "streamk must be"),
        (dict(streamk_reduction="ring"), "streamk_reduction must be"),
        (dict(streamk_ctas=0), "streamk_ctas must be"),
        (dict(streamk_ctas=-3), "streamk_ctas must be"),
        (dict(split_k=4), "both partition K_wg"),
        (dict(split_k=-1), "both partition K_wg"),
        (dict(two_stage=True, split_k=2), "both partition K_wg"),
        (dict(unroll_k=True), "runtime K trip count"),
        (dict(epilogue="cshuffle"), "epilogue='default'"),
        (dict(chiplet_swizzle=True), "chiplet_swizzle"),
        (dict(streamk_reduction="atomic"), "needs dtype_d='fp32'"),
    ]

    def test_rejects(self):
        for kw, needle in self._REJECTS:
            spec = _spec(**kw)
            with self.subTest(kw=kw):
                ok, why = is_valid_wgrad_spec(spec, "gfx950")
                self.assertFalse(ok)
                self.assertIn(needle, why)
                with self.assertRaises(ValueError) as cm:
                    spec.validate()
                self.assertEqual(str(cm.exception), why)

    def test_group_merge_and_acc_epilogue_rejected(self):
        from kernels.common._conv_implicit_gemm_common import ConvAccumulatorEpilogue

        for spec, needle in (
            (_spec(_DW, group_merge=2), "group_merge=1"),
            (_spec(acc_epilogue=ConvAccumulatorEpilogue(relu=True)), "acc_epilogue"),
        ):
            with self.subTest(needle=needle):
                ok, why = is_valid_wgrad_spec(spec, "gfx950")
                self.assertFalse(ok)
                self.assertIn(needle, why)

    def test_arch_gate(self):
        ok, why = is_valid_wgrad_spec(_spec(), "gfx90a")
        self.assertFalse(ok)
        self.assertIn("stream-K wgrad supports only gfx942, gfx950", why)
        # Wave32 targets already fail the wave-size check; either way, rejected.
        for arch in ("gfx1151", "gfx1201", "gfx1250"):
            with self.subTest(arch=arch):
                self.assertFalse(is_valid_wgrad_spec(_spec(), arch)[0])

    def test_cpp_rejects_with_the_same_reason(self):
        eng = _engine()
        for kw, _needle in self._REJECTS:
            spec = _spec(**kw)
            _ok, why = is_valid_wgrad_spec(spec, "gfx950")
            with self.subTest(kw=kw), self.assertRaises(RuntimeError) as cm:
                eng.conv_wgrad_lower_llvm(_spec_dict(spec), "gfx950")
            self.assertIn(why, str(cm.exception))


class TestStreamKAdmitsImpliesBuilds(unittest.TestCase):
    def test_matrix(self):
        built = 0
        for arch, edge in (("gfx950", 32), ("gfx942", 16)):
            for problem in (_DENSE, _GROUPED):
                for mode in ("dp_sk", "persistent"):
                    for reduction, dtype_d in (
                        ("linear", "fp16"),
                        ("tree", "bf16"),
                        ("atomic", "fp32"),
                        ("workspace", "fp16"),
                    ):
                        for ctas in (1, 4, 7, 64):
                            spec = _spec(
                                problem,
                                dtype_d=dtype_d,
                                edge=edge,
                                streamk=mode,
                                streamk_reduction=reduction,
                                streamk_ctas=ctas,
                            )
                            ok, why = is_valid_wgrad_spec(spec, arch)
                            with self.subTest(
                                arch=arch,
                                g=problem.groups,
                                mode=mode,
                                r=reduction,
                                w=ctas,
                            ):
                                self.assertTrue(ok, why)
                                build_implicit_gemm_conv_wgrad(spec, arch=arch)
                                built += 1
        self.assertEqual(built, 2 * 2 * 2 * 4 * 4)


class TestStreamKHostPlan(unittest.TestCase):
    def test_grid_matches_partition(self):
        for mode in ("dp_sk", "persistent"):
            spec = _spec(streamk=mode, streamk_ctas=4)
            part = wgrad_streamk_partition(spec)
            with self.subTest(mode=mode):
                want = 4 if mode == "persistent" else part.dp_tiles + part.sk_ctas
                self.assertEqual(wgrad_streamk_grid(spec), (want, 1, 1))

    def test_group_fold_tile_count(self):
        # m_tiles is per group: ceil(kpg / tile_m) * groups, never
        # ceil(kpg * groups / tile_m).
        part = wgrad_streamk_partition(_spec(_GROUPED))
        self.assertEqual(part.m_tiles, 3)  # ceil(40 / 64) * 3
        self.assertEqual(part.n_tiles, 3)  # ceil(3*3*16 / 64)

    def test_workspace_layout(self):
        spec = _spec(streamk_ctas=7)
        self.assertEqual(wgrad_streamk_partition(spec).sk_ctas, 7)
        flags, partials = wgrad_streamk_workspace_layout(spec)
        self.assertEqual(flags, 256)  # 7 CTAs * 4 waves * 4 B, 256-aligned
        self.assertEqual(partials, 7 * 64 * 64 * 4)
        self.assertEqual(wgrad_streamk_workspace_nbytes(spec), flags + partials)
        # 10 tiles over a pool of 5 split evenly: all data-parallel, no slots.
        self.assertEqual(wgrad_streamk_workspace_layout(_spec(streamk_ctas=5)), (0, 0))
        atomic = _spec(dtype_d="fp32", streamk_reduction="atomic")
        self.assertEqual(wgrad_streamk_workspace_nbytes(atomic), 0)
        ws = _spec(streamk_reduction="workspace")
        self.assertEqual(wgrad_streamk_workspace_nbytes(ws), 8 * 96 * 288 * 4)

    def test_signature_tracks_reduction(self):
        names = lambda s: [p["name"] for p in wgrad_streamk_signature(s)]  # noqa: E731
        self.assertEqual(names(_spec())[-2:], ["sk_flags", "sk_partials"])
        self.assertEqual(
            names(_spec(streamk_reduction="workspace"))[-2:], ["ws_ptr", "ws_bytes"]
        )
        self.assertEqual(
            len(names(_spec(dtype_d="fp32", streamk_reduction="atomic"))), 6
        )

    def test_cpp_host_plan_agrees(self):
        eng = _engine()
        for problem in (_DENSE, _GROUPED):
            for mode in ("dp_sk", "persistent"):
                for reduction, dtype_d in (
                    ("linear", "fp16"),
                    ("tree", "fp16"),
                    ("atomic", "fp32"),
                    ("workspace", "fp16"),
                ):
                    spec = _spec(
                        problem,
                        dtype_d=dtype_d,
                        streamk=mode,
                        streamk_reduction=reduction,
                        streamk_ctas=7,
                    )
                    got = eng.conv_wgrad_streamk_host(_spec_dict(spec))
                    flags, partials = wgrad_streamk_workspace_layout(spec)
                    with self.subTest(g=problem.groups, mode=mode, r=reduction):
                        self.assertEqual(tuple(got["grid"]), wgrad_streamk_grid(spec))
                        self.assertEqual(got["flags_bytes"], flags)
                        self.assertEqual(got["partials_bytes"], partials)
                        self.assertEqual(
                            got["workspace_bytes"], wgrad_streamk_workspace_nbytes(spec)
                        )
                        self.assertEqual(
                            got["is_deterministic"], reduction in ("linear", "tree")
                        )


class TestStreamKDualEngine(unittest.TestCase):
    """The C++ builder lowers every stream-K variant to the Python engine's bytes."""

    def test_lowered_ir_is_byte_identical(self):
        from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

        eng = _engine()
        cases = [
            ("gfx950", _DENSE, "dp_sk", "linear", "fp16", 32),
            ("gfx950", _DENSE, "dp_sk", "tree", "fp16", 32),
            ("gfx950", _GROUPED, "persistent", "linear", "bf16", 32),
            ("gfx950", _GROUPED, "dp_sk", "atomic", "fp32", 32),
            ("gfx942", _GROUPED, "persistent", "tree", "fp16", 16),
            ("gfx942", _DENSE, "persistent", "workspace", "fp16", 16),
        ]
        for arch, problem, mode, reduction, dtype_d, edge in cases:
            spec = _spec(
                problem,
                dtype_d=dtype_d,
                edge=edge,
                streamk=mode,
                streamk_reduction=reduction,
            )
            with self.subTest(arch=arch, g=problem.groups, mode=mode, r=reduction):
                py_ll = _lower_kernel_to_llvm_python(
                    build_implicit_gemm_conv_wgrad(spec, arch=arch), arch=arch
                )
                self.assertEqual(
                    eng.conv_wgrad_lower_llvm(_spec_dict(spec), arch), py_ll
                )

    def test_cpp_rejects_auto_pool(self):
        eng = _engine()
        with self.assertRaises(RuntimeError) as cm:
            eng.conv_wgrad_lower_llvm(_spec_dict(_spec(streamk_ctas=-1)), "gfx950")
        self.assertIn("streamk_ctas=-1 (auto)", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
