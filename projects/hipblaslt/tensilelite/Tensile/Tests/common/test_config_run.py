################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Run-only test phase for YAML kernel configs.

This module is activated by the ``--use-cache`` pytest flag. Two mechanisms
keep it from running in other modes: the ``pytest_ignore_collect`` hook in
``conftest.py`` excludes it at collection time, and the test function itself
calls ``pytest.skip`` if the flag is absent (fallback for pytest versions or
invocation styles where the hook is not called).

Role in the split-CI workflow::

    build (test_config_build.py)  -->  artifact (.tar.gz)  -->  run (this file)

This phase extracts a pre-built artifact from ``--artifact-dir`` and benchmarks
the cached kernels on the target GPU. It requires a GPU and an artifact
previously produced by ``test_config_build.py``. The artifact is extracted into
``tmpdir`` rather than back into ``--artifact-dir``, so the shared artifact
directory is never modified by the run phase.

For local single-machine testing (build + run in one pytest session) see
``test_config.py``, which calls ``_build`` and ``_run`` via isolated
subprocesses to exercise the same artifact round-trip.
"""

import argparse
import os
import signal
import subprocess
import sys

import py
import pytest
import yaml

from Tensile import Tensile
from Tensile.Common import ClientExecutionLock

from artifact_helpers import artifact_name_for_config, extract_artifact


def _run_options(config: str) -> tuple[int, int | None]:
    with open(config) as source:
        options = yaml.safe_load(source).get("TestParameters") or {}
    repetitions = options.get("run_repetitions", 1)
    timeout = options.get("run_timeout_seconds")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("TestParameters.run_repetitions must be a positive integer")
    if timeout is not None and (type(timeout) is not int or timeout < 1):
        raise ValueError("TestParameters.run_timeout_seconds must be a positive integer")
    return repetitions, timeout


def _run_repeated(command: list[str], repetitions: int, timeout: int | None, env: dict) -> None:
    """Run every repetition in a fresh process; stop at the first failure.

    Each process owns a new session on POSIX. A timeout or Python exception
    kills its entire process group, including a blocked GPU client.
    """
    for iteration in range(1, repetitions + 1):
        print(f"Cached GPU run {iteration}/{repetitions}", flush=True)
        with subprocess.Popen(command, env=env, start_new_session=(os.name == "posix")) as process:
            try:
                returncode = process.wait(timeout=timeout)
            except BaseException:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   check=False, stdout=subprocess.DEVNULL)
                    process.kill()
                process.wait()
                raise
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)


def _run(config: str, output_dir: str, artifact_dir: str, tensile_args: list[str]) -> None:
    """Extract a pre-built artifact and run benchmarks against it.

    Callable from both the pytest wrapper below and from test_config.py via
    subprocess (where it runs in a clean process to avoid global-state bleed).
    """
    repetitions, timeout = _run_options(config)
    artifact_name = artifact_name_for_config(config)
    tarball = os.path.join(artifact_dir, artifact_name + ".tar.gz")
    assert os.path.isfile(tarball), f"Artifact tarball not found: {tarball}"
    extract_artifact(tarball, output_dir)
    if repetitions == 1 and timeout is None:
        Tensile.Tensile([config, output_dir, "--use-cache", *tensile_args])
        return

    # Wait for the shared GPU lock before starting the per-run deadline. Hold
    # it in this supervisor and remove the child's copy to avoid reacquiring
    # the same lock in another process. This also applies in --use-cache CI.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--client-lock")
    options, child_args = parser.parse_known_args(tensile_args)
    command = [sys.executable, "-c",
               "import sys; from Tensile import Tensile; Tensile.Tensile(sys.argv[1:])",
               config, output_dir, "--use-cache", *child_args]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "PYTHONUNBUFFERED": "1"}
    with ClientExecutionLock(options.client_lock):
        _run_repeated(command, repetitions, timeout, env)


def test_config_run(tensile_args: list[str], config: str, tmpdir: py.path.local, pytestconfig: pytest.Config) -> None:
    """Pytest wrapper: extract a pre-built artifact and benchmark on the GPU.

    Activated only when ``--use-cache`` is passed; collection is skipped in all
    other modes by ``conftest.pytest_ignore_collect``. Requires a GPU.

    ``--artifact-dir`` must point to the directory where ``test_config_build``
    wrote the artifact. The artifact is extracted into ``tmpdir`` so the shared
    artifact directory is not modified by this phase.
    """
    if not pytestconfig.getoption("--use-cache"):
        pytest.skip("requires --use-cache")
    artifact_name = artifact_name_for_config(config)
    output_dir = os.path.join(tmpdir.strpath, artifact_name)
    _run(config, output_dir, pytestconfig.getoption("--artifact-dir"), tensile_args)
