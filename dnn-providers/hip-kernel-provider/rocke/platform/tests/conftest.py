# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# Pytest root config for the rocKE engine test tree. Puts the Python engine
# package root (rocke/platform/python) on sys.path so `import rocke` resolves
# without an external PYTHONPATH. Paths are derived from this file's location
# (relative), so the tree stays copy-able verbatim into another repo.
#
# source tree: this dir -> rocke/platform/tests, so parent -> rocke/platform

import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROCKE = _HERE.parent  # tests -> rocke/platform
_PYROOT = _ROCKE / "python"
if str(_PYROOT) not in sys.path:
    sys.path.insert(0, str(_PYROOT))


@pytest.fixture(scope="session")
def native_recipe_replay_cli():
    """Use a prebuilt native VM; installed tests must never build from source."""
    suffix = ".exe" if sys.platform == "win32" else ""
    name = f"rocke_portable_ir_replay_cli{suffix}"
    explicit = os.environ.get("ROCKE_REPLAY_CLI")
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            pytest.fail(f"ROCKE_REPLAY_CLI does not name a file: {path}")
        return path
    installed = _HERE / "portable_ir" / name
    if installed.is_file():
        return installed
    if (_ROCKE / "rocke_installed_smoke.py").is_file():
        pytest.fail(f"Installed native recipe replay CLI is missing: {installed}")
    built = _ROCKE / "build" / "tests" / name
    if built.is_file():
        return built
    pytest.skip(
        "Build rocke_portable_ir_replay_cli and set ROCKE_REPLAY_CLI to its path"
    )


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
