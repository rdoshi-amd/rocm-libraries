# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Dual-engine parity of the public ``lower_conv_direct_grouped`` entry point.

``rocke.core.backend.lower_conv_direct_grouped`` flattens a
``DirectConv16cSpec`` / ``DirectConv4cSpec`` / ``DirectConvSpec`` into a dict
(``conv_direct_grouped_spec_to_dict``) for the C++ binding. Every spec field
that changes the emitted kernel must survive that flattening -- the problem
``dtype`` and the 4c fused-dgrad knobs (``dgrad_fused_weights``,
``dgrad_weights_lds``) and the generic kernel's row-stream knobs included --
or the default (cpp) backend silently returns a different kernel than the
Python engine. The byte-identity gate
covers the ``_emit.c`` parity pair, not this binding path, so this test runs
``backend="both"`` (python vs cpp IR + ``.ll`` equality) over those specs.

CPU only (no kernel launch). Skips when the ``rocke_engine`` extension is not
importable.

Run:
  PYTHONPATH=<engine build>/cpp/bindings:rocke/platform/python:rocke/library \\
      <python> -m pytest rocke/library/tests/test_conv_direct_grouped_backend_parity.py -v
"""

from __future__ import annotations

import importlib.util
import unittest

from rocke.core.backend import (
    conv_direct_grouped_spec_to_dict,
    lower_conv_direct_grouped,
)

from kernels.common.conv_direct_grouped import (
    DirectConv4cSpec,
    DirectConv16cSpec,
    DirectConvProblem,
    DirectConvSpec,
    is_valid_spec,
    is_valid_spec_4c,
    make_dgrad_4c_spec,
)

_HAS_ENGINE = importlib.util.find_spec("rocke_engine") is not None
_SKIP = "" if _HAS_ENGINE else "rocke_engine extension not importable"


def _p4(dtype, N=2, H=13, W=17, groups=32, K=3):
    return DirectConvProblem(
        N=N, H=H, W=W, groups=groups, cpg=4, kpg=4, KH=K, KW=K,
        PAD=(K - 1) // 2, dtype=dtype,
    )  # fmt: skip


def _4c_cases():
    """(label, spec, arch) for the 4c forms this path must carry."""
    out = []
    for dtype in ("fp16", "bf16"):
        out.append(
            (f"4c_fprop_{dtype}", DirectConv4cSpec(problem=_p4(dtype)), "gfx950")
        )
        out.append(
            (
                f"4c_fprop_{dtype}_1x1",
                DirectConv4cSpec(problem=_p4(dtype, H=7, W=15, K=1)),
                "gfx942",
            )
        )
        for fused, lds in ((False, False), (True, False), (True, True)):
            spec = make_dgrad_4c_spec(
                _p4(dtype),
                dgrad_fused_weights=fused,
                dgrad_weights_lds=lds,
            )
            out.append(
                (f"4c_dgrad_{dtype}_fw{int(fused)}_lds{int(lds)}", spec, "gfx950")
            )
        # Fused without LDS staging is also legal on gfx942.
        spec = make_dgrad_4c_spec(_p4(dtype, N=1, H=7, W=13), dgrad_fused_weights=True)
        out.append((f"4c_dgrad_{dtype}_fw1_gfx942", spec, "gfx942"))
        # Row-staged form (stage_rows, waves_q = block_q / 4), 3x3 and 1x1.
        for bq, bg, K in ((4, 16, 3), (8, 32, 3), (4, 32, 1)):
            spec = make_dgrad_4c_spec(
                _p4(dtype, K=K),
                block_q=bq,
                block_groups=bg,
                dgrad_fused_weights=True,
                dgrad_weights_lds=True,
                stage_rows=True,
            )
            out.append((f"4c_dgrad_{dtype}_sr_bq{bq}_bg{bg}_k{K}", spec, "gfx950"))
    return out


def _16c_cases():
    out = []
    for dtype in ("fp16", "bf16"):
        for fold in (False, True):
            p = DirectConvProblem(
                N=1, H=15, W=17, groups=8, cpg=16, kpg=16, dtype=dtype
            )
            out.append(
                (
                    f"16c_{dtype}_k32{int(fold)}",
                    DirectConv16cSpec(problem=p, fold_k32=fold),
                    "gfx950",
                )
            )
    return out


def _pg(dtype, N=2, H=9, W=17, groups=8, cpg=16, kpg=16, K=3):
    return DirectConvProblem(
        N=N, H=H, W=W, groups=groups, cpg=cpg, kpg=kpg, KH=K, KW=K,
        PAD=(K - 1) // 2, dtype=dtype,
    )  # fmt: skip


#: Row-stream knob stacks of the generic kernel (the dispatch's picks).
_STREAM = {"prefetch_rows": 2, "lds_only_sync": True, "xcd_tiles": True}


def _generic_cases():
    """(label, spec, arch) over the generic kernel's paths and knobs."""
    S = DirectConvSpec
    out = []
    for dt in ("fp16", "bf16"):
        out += [
            (f"fw_{dt}", S(problem=_pg(dt), block_groups=2, dgrad_fused_weights=True), "gfx950"),
            (f"prepass_5x5_{dt}", S(problem=_pg(dt, groups=4, cpg=24, kpg=12, K=5), block_groups=1, block_h=4), "gfx950"),
            (f"rk_persistent_{dt}", S(problem=_pg(dt, groups=4), block_groups=1, block_h=4, persistent_grid=True, runtime_k_loop=True), "gfx950"),
            (f"wq2_wk2_{dt}", S(problem=_pg(dt, groups=4, cpg=32), block_q=32, block_groups=1, waves_q=2, waves_k=2), "gfx950"),
            (f"preload_sb_{dt}", S(problem=_pg(dt, groups=4), block_groups=2, preload_weights=True, double_buffer=False), "gfx950"),
            (
                f"stream_fwl_c16_{dt}",
                S(problem=_pg(dt, N=9, H=10, W=13, groups=32), block_groups=2,
                  dgrad_fused_weights=True, dgrad_weights_lds=True, waves_per_eu=4,
                  lds_pad=8, stage_out=True, **_STREAM),
                "gfx950",
            ),
            (
                f"stream_bh_{dt}",
                S(problem=_pg(dt, N=8, H=11, W=19, groups=32), block_q=32,
                  block_groups=2, block_h=8, dgrad_fused_weights=True,
                  dgrad_weights_lds=True, lds_pad=8, stage_out=True, **_STREAM),
                "gfx950",
            ),
            (
                f"stream_wm2_k32_{dt}",
                S(problem=_pg(dt, N=8, H=7, W=14, groups=16, cpg=32, kpg=32),
                  block_groups=1, fold_k32=True, dgrad_fused_weights=True,
                  dgrad_weights_lds=True, waves_m=2, lds_pad=8, stage_out=True,
                  **_STREAM),
                "gfx950",
            ),
            (
                f"stream_fw_wm2_5x5_{dt}",
                S(problem=_pg(dt, N=2, H=9, W=14, groups=8, cpg=8, kpg=32, K=5),
                  block_groups=1, block_h=4, dgrad_fused_weights=True,
                  prefetch_rows=3, lds_only_sync=True, waves_m=2, stage_out=True),
                "gfx950",
            ),
        ]  # fmt: skip
    out.append(
        (
            "stream_gfx942_fp16",
            S(
                problem=_pg("fp16"),
                block_groups=2,
                block_h=4,
                dgrad_fused_weights=True,
                prefetch_rows=2,
                lds_only_sync=True,
                lds_pad=8,
            ),
            "gfx942",
        )
    )
    return out


class TestSpecToDict(unittest.TestCase):
    """CPU-only: the flattened dict carries every kernel-shaping field."""

    def test_4c_fields_forwarded(self):
        spec = make_dgrad_4c_spec(
            _p4("bf16"), dgrad_fused_weights=True, dgrad_weights_lds=True
        )
        d = conv_direct_grouped_spec_to_dict(spec, "4c")
        self.assertEqual(d["problem"]["dtype"], "bf16")
        self.assertIs(d["dgrad_fused_weights"], True)
        self.assertIs(d["dgrad_weights_lds"], True)
        self.assertNotIn("fold_k32", d)

    def test_4c_every_kernel_field_forwarded(self):
        # Every DirectConv4cSpec field except the problem (nested) reaches the
        # binding: a new knob that is not forwarded would silently build the
        # old kernel on the cpp path.
        import dataclasses

        spec = make_dgrad_4c_spec(
            _p4("bf16"),
            block_q=8,
            dgrad_fused_weights=True,
            dgrad_weights_lds=True,
            stage_rows=True,
        )
        d = conv_direct_grouped_spec_to_dict(spec, "4c")
        for f in dataclasses.fields(spec):
            if f.name == "problem":
                continue
            with self.subTest(field=f.name):
                self.assertIn(f.name, d)
                self.assertEqual(d[f.name], getattr(spec, f.name))
        self.assertIs(d["stage_rows"], True)
        self.assertEqual(d["waves_q"], 2)

    def test_generic_every_kernel_field_forwarded(self):
        # Every DirectConvSpec field except the problem (nested) reaches the
        # binding, with non-default values for the row-stream knobs.
        import dataclasses

        for label, spec, _arch in _generic_cases():
            d = conv_direct_grouped_spec_to_dict(spec, "generic")
            for f in dataclasses.fields(spec):
                if f.name == "problem":
                    continue
                with self.subTest(case=label, field=f.name):
                    self.assertIn(f.name, d)
                    self.assertEqual(d[f.name], getattr(spec, f.name))

    def test_16c_fields_forwarded(self):
        p = DirectConvProblem(N=1, H=8, W=8, groups=8, cpg=16, kpg=16, dtype="bf16")
        d = conv_direct_grouped_spec_to_dict(
            DirectConv16cSpec(problem=p, fold_k32=False), "16c"
        )
        self.assertEqual(d["problem"]["dtype"], "bf16")
        self.assertIs(d["fold_k32"], False)
        self.assertNotIn("dgrad_fused_weights", d)


@unittest.skipIf(bool(_SKIP), _SKIP)
class TestBackendBoth(unittest.TestCase):
    """python vs cpp through the public backend entry (IR + .ll equality)."""

    def _check(self, cases, kind):
        for label, spec, arch in cases:
            with self.subTest(case=label, arch=arch):
                ok, why = (
                    is_valid_spec_4c(spec, arch=arch) if kind == "4c" else (True, "")
                )
                self.assertTrue(ok, why)
                r = lower_conv_direct_grouped(
                    spec, kind=kind, arch=arch, backend="both", want_ir=True
                )
                # Python-side name must appear in the cpp-checked .ll.
                self.assertIn(spec.kernel_name(), r.llvm_text)
                c = lower_conv_direct_grouped(spec, kind=kind, arch=arch, backend="cpp")
                self.assertEqual(c.llvm_text, r.llvm_text)

    def test_4c(self):
        self._check(_4c_cases(), "4c")

    def test_16c(self):
        self._check(_16c_cases(), "16c")

    def test_generic(self):
        for label, spec, arch in _generic_cases():
            with self.subTest(case=label):
                ok, why = is_valid_spec(spec, arch=arch)
                self.assertTrue(ok, why)
        self._check(_generic_cases(), "generic")

    def test_generic_rejects_match(self):
        # Every validator / is_valid_spec reject reason is identical in both
        # engines (the C++ engine prefixes its lowering context).
        import dataclasses

        R = dataclasses.replace
        base = DirectConvSpec(
            problem=_pg("bf16", N=9, H=14, W=13, groups=32),
            block_groups=2,
            dgrad_fused_weights=True,
            dgrad_weights_lds=True,
        )
        bad = [
            R(base, problem=R(base.problem, dtype="fp32")),
            R(base, problem=R(base.problem, PAD=0)),
            R(base, problem=R(base.problem, kpg=6)),
            R(base, block_groups=0),
            R(base, waves_m=0),
            R(base, block_groups=3),
            R(base, block_q=24),
            R(base, block_h=-4),
            R(base, fold_k32=True),
            R(base, block_q=16, waves_q=2),
            R(base, runtime_k_loop=True),
            R(base, dgrad_fused_weights=False),
            R(base, problem=R(base.problem, cpg=64, kpg=64), block_groups=1),
            R(base, prefetch_rows=4),
            R(base, stage_out=True, problem=R(base.problem, kpg=8)),
            R(base, lds_pad=4),
            R(base, waves_m=2),
            R(base, lds_only_sync=True, double_buffer=False),
            R(base, xcd_tiles=True, problem=R(base.problem, N=4)),
            R(base, problem=R(base.problem, stride=2)),
            R(
                base,
                dgrad_fused_weights=False,
                dgrad_weights_lds=False,
                persistent_grid=True,
            ),
        ]
        for spec in bad:
            msgs = []
            for backend in ("python", "cpp"):
                with self.assertRaises(Exception) as cm:
                    lower_conv_direct_grouped(
                        spec, kind="generic", arch="gfx950", backend=backend
                    )
                msgs.append(str(cm.exception))
            with self.subTest(reason=msgs[0]):
                self.assertTrue(msgs[1].endswith(msgs[0]), msgs)
        # Target-dependent reject: no transpose LDS reads on gfx942.
        msgs = []
        for backend in ("python", "cpp"):
            with self.assertRaises(Exception) as cm:
                lower_conv_direct_grouped(
                    base, kind="generic", arch="gfx942", backend=backend
                )
            msgs.append(str(cm.exception))
        self.assertIn("ds_read_b64_tr_b16", msgs[0])
        self.assertTrue(msgs[1].endswith(msgs[0]), msgs)


@unittest.skipIf(bool(_SKIP), _SKIP)
class Test4cTapBound(unittest.TestCase):
    """KH*KW > 16 is rejected with the same reason by both engines (the C++
    builder's per-tap arrays are bounded)."""

    def test_stage_rows_rejects_match(self):
        # The stage_rows validator reasons are identical in both engines.
        import dataclasses

        p = _p4("bf16")
        base = make_dgrad_4c_spec(
            p, dgrad_fused_weights=True, dgrad_weights_lds=True, stage_rows=True
        )
        bad = [
            dataclasses.replace(base, block_q=8),
            dataclasses.replace(base, waves_q=0),
            dataclasses.replace(base, dgrad_weights_lds=False),
            dataclasses.replace(base, stage_rows=False, block_q=8, waves_q=2),
            dataclasses.replace(base, problem=dataclasses.replace(base.problem, PAD=0)),
            dataclasses.replace(base, block_q=68, waves_q=17),
        ]
        for spec in bad:
            msgs = []
            for backend in ("python", "cpp"):
                with self.assertRaises(Exception) as cm:
                    lower_conv_direct_grouped(
                        spec, kind="4c", arch="gfx950", backend=backend
                    )
                msgs.append(str(cm.exception))
            with self.subTest(reason=msgs[0]):
                self.assertTrue(msgs[0].startswith("DirectConv4cSpec "))
                self.assertTrue(msgs[1].endswith(msgs[0]), msgs)

    def test_5x5_rejected_both_engines(self):
        spec = DirectConv4cSpec(problem=_p4("fp16", K=5))
        ok, why = is_valid_spec_4c(spec, arch="gfx950")
        self.assertFalse(ok)
        self.assertIn("KH*KW <= 16", why)
        for backend in ("python", "cpp"):
            with self.subTest(backend=backend):
                with self.assertRaises(Exception) as cm:
                    lower_conv_direct_grouped(
                        spec, kind="4c", arch="gfx950", backend=backend
                    )
                self.assertIn("KH*KW <= 16", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
