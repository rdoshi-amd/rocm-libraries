#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fixed-deadline GPU trials; diagnostics run only after a trial has failed."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from host_state import device, read

SCRIPT_DIR = Path(__file__).resolve().parent


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def permit_parent_debugger():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(0x59616D61, os.getppid(), 0, 0, 0) != 0:
        os.write(2, f"PR_SET_PTRACER failed: {ctypes.get_errno()}\n".encode())


def stop(process):
    for sig, wait in ((signal.SIGTERM, 2), (signal.SIGKILL, 3)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=wait)
        except subprocess.TimeoutExpired:
            pass
    return process.poll() is not None


def host_snapshot(reports, name):
    try:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "host_state.py"),
                str(reports / f"host-{name}.json"),
            ],
            timeout=20,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        save(reports / f"host-{name}-error.json", {"error": str(error)})


def capture(reports, name, process):
    root = reports / f"{name}-timeout"
    root.mkdir(exist_ok=True)
    proc = Path(f"/proc/{process.pid}")
    for filename in ("status", "wchan", "syscall", "maps", "smaps_rollup"):
        (root / filename).write_text(read(proc / filename))
    save(root / "fds.json", [device(p) for p in (proc / "fd").glob("*")])
    save(
        root / "threads.json",
        {
            p.name: {
                key: read(p / key) for key in ("comm", "wchan", "syscall", "stack")
            }
            for p in (proc / "task").glob("*")
        },
    )
    debugger_file = reports / "debugger-path.txt"
    debugger = (
        debugger_file.read_text().strip()
        if debugger_file.exists()
        else shutil.which("gdb")
    )
    if not debugger or process.poll() is not None:
        return
    env = os.environ.copy()
    tool_root = Path("diagnostic-tools/root").resolve()
    env["LD_LIBRARY_PATH"] = (
        f"{tool_root}/usr/lib/x86_64-linux-gnu:{tool_root}/lib/x86_64-linux-gnu"
    )
    env["DEBUGINFOD_URLS"] = ""
    env.pop("LD_PRELOAD", None)
    if (tool_root / "usr/lib/python3.12").is_dir():
        env["PYTHONHOME"] = str(tool_root / "usr")
    cmd = [debugger, "-nx", "-nh", "-batch"]
    if (tool_root / "usr/share/gdb").is_dir():
        cmd += [f"--data-directory={tool_root}/usr/share/gdb"]
    cmd += ["-iex", "set auto-load off"]
    for instruction in (
        "set pagination off",
        "set debuginfod enabled off",
        f"attach {process.pid}",
        "info threads",
        "thread apply all bt 40",
        "info sharedlibrary",
        "detach",
    ):
        cmd += ["-ex", instruction]
    with (root / "backtrace.txt").open("w") as output:
        try:
            run = subprocess.run(
                cmd, env=env, stdout=output, stderr=subprocess.STDOUT, timeout=40
            )
            output.write(f"\ngdb_returncode={run.returncode}\n")
        except (OSError, subprocess.TimeoutExpired) as error:
            output.write(f"Debugger error: {error}\n")
        finally:
            if process.poll() is None:
                try:
                    process.send_signal(signal.SIGCONT)
                except ProcessLookupError:
                    pass


def classify(text, returncode, timed_out, expected_tests):
    started = re.findall(r"\[==========\] Running (\d+) tests?", text)
    passed = re.findall(r"\[  PASSED  \] (\d+) tests?", text)
    sanitizer = bool(
        re.search(r"(?:ERROR|SUMMARY): (?:AddressSanitizer|LeakSanitizer)", text)
    )
    if timed_out:
        outcome = "timeout_during_tests" if started else "timeout_before_tests"
    elif sanitizer:
        outcome = "sanitizer_failure"
    elif returncode:
        outcome = "process_failure"
    elif expected_tests is not None and (
        not started
        or int(started[-1]) != expected_tests
        or not passed
        or int(passed[-1]) != expected_tests
    ):
        outcome = "incomplete_suite"
    elif expected_tests is None and "PROBE PASSED" not in text:
        outcome = "incomplete_probe"
    else:
        outcome = "pass"
    return {
        "outcome": outcome,
        "tests_started": int(started[-1]) if started else None,
        "tests_passed": int(passed[-1]) if passed else None,
        "version_banner": "hipBLASLt version:" in text,
        "sanitizer_diagnostic": sanitizer,
        "last_output": text.splitlines()[-20:],
    }


def run_case(reports, name, command, timeout_seconds, expected_tests=None):
    reports.mkdir(exist_ok=True)
    print(
        json.dumps({"case": name, "command": command, "timeout": timeout_seconds}),
        flush=True,
    )
    host_snapshot(reports, f"before-{name}")
    started = time.monotonic()
    started_utc = time.time()
    timed_out = False
    log = reports / f"{name}.log"
    with log.open("wb") as output, log.open(errors="replace") as tail:
        process = subprocess.Popen(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=permit_parent_debugger,
        )
        try:
            while process.poll() is None:
                chunk = tail.read()
                if chunk:
                    print(chunk, end="", flush=True)
                if time.monotonic() - started >= timeout_seconds:
                    timed_out = True
                    # The trial has already failed its deadline. Only now attach.
                    try:
                        capture(reports, name, process)
                    except Exception as error:
                        save(
                            reports / f"{name}-capture-error.json",
                            {"error": str(error)},
                        )
                    break
                time.sleep(0.2)
        finally:
            reaped = stop(process)
        print(tail.read(), end="", flush=True)
    text = log.read_text(errors="replace")
    result = {
        "case": name,
        "command": command,
        "started_utc": started_utc,
        "elapsed_seconds_including_diagnostics": time.monotonic() - started,
        "deadline_seconds": timeout_seconds,
        "returncode": process.returncode,
        "timed_out": timed_out,
        "reaped": reaped,
        **classify(text, process.returncode, timed_out, expected_tests),
    }
    save(reports / f"{name}-result.json", result)
    host_snapshot(reports, f"after-{name}")
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "probes"), required=True)
    parser.add_argument("--reports", type=Path, default=Path("diagnostics"))
    parser.add_argument("--artifact", type=Path, default=Path("build"))
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(128 + sig))
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(128 + sig))
    args.reports.mkdir(exist_ok=True)
    save(
        args.reports / f"{args.phase}-identity.json",
        {
            "phase": args.phase,
            "pair": os.getenv("DIAGNOSTIC_PAIR"),
            "pool": os.getenv("DIAGNOSTIC_POOL"),
            "runner": os.getenv("DIAGNOSTIC_RUNNER"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT"),
            "revision": os.getenv("GITHUB_SHA"),
        },
    )
    if args.phase == "smoke":
        result = run_case(
            args.reports,
            "smoke",
            [
                str(args.artifact.resolve() / "bin/hipblaslt-test"),
                "--gtest_filter=*smoke*",
            ],
            1200,
            expected_tests=948,
        )
        return 0 if result["outcome"] == "pass" else 1
    smoke = args.reports / "smoke-result.json"
    if not smoke.exists() or not json.loads(smoke.read_text())["reaped"]:
        raise RuntimeError("No reaped baseline process; do not add work to this GPU")
    results = []
    for kind in ("hip", "blas"):
        result = run_case(
            args.reports, kind, [str((args.reports / f"startup-{kind}").resolve())], 180
        )
        results.append(result)
        if not result["reaped"]:
            break
    return (
        0 if all(r["outcome"] == "pass" for r in results) and len(results) == 2 else 1
    )


if __name__ == "__main__":
    sys.exit(main())
