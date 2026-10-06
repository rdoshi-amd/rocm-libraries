# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Exercise repeated common-test processes, failure propagation and cleanup."""
import importlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

import pytest
import yaml

pytestmark = pytest.mark.unit
_COMMON = Path(__file__).resolve().parents[1] / "common"


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(_COMMON))
    return importlib.import_module("test_config_run")


def test_repetitions_use_fresh_processes(runner, tmp_path):
    pids = tmp_path / "pids"
    script = "import os,sys; open(sys.argv[1], 'a').write(str(os.getpid())+'\\n')"
    runner._run_repeated([sys.executable, "-c", script, str(pids)], 3, 10, os.environ.copy())
    actual = pids.read_text().splitlines()
    assert len(actual) == len(set(actual)) == 3
    assert str(os.getpid()) not in actual


def test_failure_stops_remaining_repetitions(runner, tmp_path):
    attempts = tmp_path / "attempts"
    script = ("import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
              "n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); "
              "sys.exit(7 if n==2 else 0)")
    with pytest.raises(subprocess.CalledProcessError) as error:
        runner._run_repeated([sys.executable, "-c", script, str(attempts)],
                             20, 10, os.environ.copy())
    assert error.value.returncode == 7
    assert attempts.read_text() == "2"


@pytest.mark.skipif(os.name != "posix", reason="checks POSIX process-group cleanup")
def test_timeout_kills_client_descendant(runner, tmp_path):
    pidfile = tmp_path / "child_pid"
    script = ("import subprocess,sys,time; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(600)']); "
              "open(sys.argv[1],'w').write(str(p.pid)); time.sleep(600)")
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            runner._run_repeated([sys.executable, "-c", script, str(pidfile)],
                                 20, 3, os.environ.copy())
        child_pid = int(pidfile.read_text())
        # A killed orphan can briefly remain a zombie until init reaps it.
        stat = Path(f"/proc/{child_pid}/stat")
        deadline = time.monotonic() + 2
        while True:
            try:
                state = stat.read_text().rsplit(")", 1)[1].split()[0]
            except FileNotFoundError:
                break
            if state == "Z":
                break
            assert time.monotonic() < deadline, "client survived the timeout"
            time.sleep(0.01)
    finally:
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("options", [
    {"run_repetitions": 0}, {"run_repetitions": True}, {"run_timeout_seconds": -1},
])
def test_invalid_run_options_fail(runner, tmp_path, options):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"TestParameters": options}))
    with pytest.raises(ValueError, match="positive integer"):
        runner._run_options(str(config))


def test_cached_runs_hold_gpu_lock_outside_deadline(runner, monkeypatch, tmp_path):
    config = _COMMON / "gemm/gfx1250/cluster_entry_handoff.yaml"
    name = runner.artifact_name_for_config(str(config))
    (tmp_path / (name + ".tar.gz")).touch()
    monkeypatch.setattr(runner, "extract_artifact", lambda *args: None)
    held = []

    @contextmanager
    def lock(path):
        held.append(path)
        yield
        held.pop()

    def run(command, repetitions, timeout, env):
        assert held == ["test-gpu.lock"]
        assert "--client-lock" not in command
        assert "test-gpu.lock" not in command
        assert "--use-cache" in command
        assert repetitions == 20 and timeout == 120
        assert env["PYTHONUNBUFFERED"] == "1"

    monkeypatch.setattr(runner, "ClientExecutionLock", lock)
    monkeypatch.setattr(runner, "_run_repeated", run)
    runner._run(str(config), str(tmp_path / "out"), str(tmp_path),
                ["--client-lock", "test-gpu.lock", "--prebuilt-client", "test-client"])
    assert not held


@pytest.mark.parametrize("arch, skipped", [
    ("gfx942", True), ("gfx1250", False), ("gfx1250-strict", False),
])
def test_cluster_regression_arch_selection(monkeypatch, arch, skipped):
    monkeypatch.syspath_prepend(str(_COMMON))
    from config_helpers import configMarks
    marks = configMarks(str(_COMMON / "gemm/gfx1250/cluster_entry_handoff.yaml"),
                        str(_COMMON.parent), [arch])
    assert any(mark.name == "skip" for mark in marks) == skipped
    assert any(mark.name == "common" for mark in marks)
    assert any(mark.name == "gfx1250-strict" for mark in marks)


@pytest.mark.parametrize("document", [{}, {"TestParameters": None}])
def test_default_run_options(runner, tmp_path, document):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(document))
    assert runner._run_options(str(config)) == (1, None)
