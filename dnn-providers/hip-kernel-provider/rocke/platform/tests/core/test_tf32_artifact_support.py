# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host checks for TF32 test wiring; no GPU numerical evidence."""

import runpy
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("native", [False, True])
def test_gpu_test_selects_available_engines(tmp_path, monkeypatch, native):
    module = runpy.run_path(str(_TESTS / "numeric" / "test_gfx942_tf32_numeric.py"))
    test = module["test_gfx942_tf32_numeric"]
    monkeypatch.setitem(test.__globals__, "get_device_arch", lambda: "gfx942")
    monkeypatch.setattr(
        module["importlib"].util, "find_spec", lambda _: object() if native else None
    )
    expected = "both" if native else "python"

    def run(output_dir, backend):
        assert output_dir == tmp_path
        assert backend == expected
        engines = ("python", "cpp") if native else ("python",)
        return {
            "status": "pass",
            "results": [{"engine": engine} for engine in engines for _ in range(12)],
        }

    monkeypatch.setitem(test.__globals__, "run", run)
    test(tmp_path)

    def broken_run(*args, **kwargs):
        raise ImportError("native binding is broken")

    monkeypatch.setitem(test.__globals__, "run", broken_run)
    with pytest.raises(ImportError, match="native binding is broken"):
        test(tmp_path)


def test_gpu_test_requires_requested_device(tmp_path, monkeypatch):
    module = runpy.run_path(str(_TESTS / "numeric" / "test_gfx942_tf32_numeric.py"))
    test = module["test_gfx942_tf32_numeric"]
    monkeypatch.setitem(test.__globals__, "get_device_arch", lambda: "gfx950")
    monkeypatch.setenv("ROCKE_REQUIRE_GFX942", "1")
    with pytest.raises(pytest.fail.Exception, match="Required gfx942 device absent"):
        test(tmp_path)
