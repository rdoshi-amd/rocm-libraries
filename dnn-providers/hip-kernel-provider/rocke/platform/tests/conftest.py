# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Pytest root config for the rocKE engine test tree. Puts the Python engine
# package root (rocke/platform/python) on sys.path so `import rocke` resolves
# without an external PYTHONPATH. Paths are derived from this file's location
# (relative), so the tree stays copy-able verbatim into another repo.
#
# source tree: this dir -> rocke/platform/tests, so parent -> rocke/platform

import sys
import json
import os
import subprocess
from pathlib import Path

import pytest


def _optimization_barrier_executable():
    """Resolve supplied or installed native coverage without building source."""
    executable = os.environ.get("ROCKE_OPTIMIZATION_BARRIER_TEST")
    if not executable and Path("CTestTestfile.cmake").is_file():
        listing = subprocess.run(
            ["ctest", "--show-only=json-v1", "-R", "^rocke_optimization_barrier$"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        tests = json.loads(listing.stdout)["tests"]
        if tests:
            if len(tests) != 1 or not tests[0].get("command"):
                raise ValueError("CTest did not resolve the native barrier test")
            executable = tests[0]["command"][0]
    if not executable:
        pytest.skip(
            "native barrier test unavailable; build or supply ROCKE_OPTIMIZATION_BARRIER_TEST"
        )
    path = Path(executable).resolve()
    if not path.is_file():
        raise ValueError(f"native barrier test executable does not exist: {path}")
    return path


@pytest.fixture(scope="session")
def native_barrier_hip():
    """Lower test kernels with the native engine, preserving execution failures."""
    executable = _optimization_barrier_executable()

    def lower(kernel, arch):
        from rocke.core.ir_serialize import serialize

        result = subprocess.run(
            [str(executable), "--hip", arch],
            input=serialize(kernel),
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if not result.stdout:
            raise ValueError("native barrier HIP lowering produced no source")
        return result.stdout

    return lower


_HERE = Path(__file__).resolve().parent
_ROCKE = _HERE.parent  # tests -> rocke/platform
_PYROOT = _ROCKE / "python"
if str(_PYROOT) not in sys.path:
    sys.path.insert(0, str(_PYROOT))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Report a ``ROCKE_BACKEND=both`` coverage gap as a skip, never a pass.

    Under ``both`` the dispatcher refuses to substitute the Python result for a
    kernel the C++ engine could not lower, so an arch with no C++ ISA backend
    (see ``backend.CPP_UNPORTED_ARCHES``) raises ``BackendCoverageGap``. Turning
    it into a skip here keeps the gap *counted and named* in the run summary.
    Letting it pass -- which is what returning the Python IR would do -- would
    make the differential lane green on kernels it never compared.

    Only ``BackendCoverageGap`` is remapped. A ``BackendMismatch`` (the engines
    disagreed) and any other engine error stay failures.

    This rewrites the report rather than the raised exception because most of
    the suite is ``unittest.TestCase``: pytest's unittest integration captures
    the error inside ``runtest()`` and only republishes it onto ``call.excinfo``
    from its own (tryfirst, non-wrapper) makereport hook, so a
    ``pytest_runtest_call`` wrapper sees no exception at all and silently does
    nothing. Wrapping makereport is the one place that sees both styles.
    """
    outcome = yield
    if call.excinfo is None:
        return
    from rocke.core.backend import BackendCoverageGap

    if not isinstance(call.excinfo.value, BackendCoverageGap):
        return
    report = outcome.get_result()
    report.outcome = "skipped"
    report.longrepr = (str(item.path), item.location[1] + 1, str(call.excinfo.value))


# The IR parity harness' attention families are the one sanctioned platform ->
# library reach, so `kernels`/`builders` must resolve here as well. Probed
# against both layouts this tree runs in: staged under tests/library/ in an
# install (the destination TheRock's test-artifact globs capture), or the sibling
# library tree in a checkout.
for _lib_root in (_HERE / "library", _HERE.parents[1] / "library"):
    if (_lib_root / "kernels").is_dir():
        if str(_lib_root) not in sys.path:
            sys.path.insert(0, str(_lib_root))
        break
