#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Read host/device evidence without initializing HIP or changing host state."""

import collections
import glob
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

ENV_KEYS = (
    "DIAGNOSTIC_POOL",
    "DIAGNOSTIC_PAIR",
    "DIAGNOSTIC_RUNNER",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_XNACK",
    "OMP_NUM_THREADS",
    "ASAN_OPTIONS",
    "LSAN_OPTIONS",
    "GTEST_SHARD_INDEX",
    "GTEST_TOTAL_SHARDS",
    "NODE_NAME",
    "K8S_NODE_NAME",
)


def read(path):
    try:
        return Path(path).read_text(errors="replace")[:131072]
    except OSError as error:
        return f"unavailable: {error}"


def link(path):
    try:
        return os.readlink(path)
    except OSError as error:
        return f"unavailable: {error}"


def device(path):
    result = {"path": str(path), "target": os.path.realpath(path)}
    try:
        info = os.stat(path)
        if stat.S_ISCHR(info.st_mode):
            major, minor = os.major(info.st_rdev), os.minor(info.st_rdev)
            sysfs = Path(f"/sys/dev/char/{major}:{minor}")
            result.update(
                major=major,
                minor=minor,
                sysfs=str(sysfs.resolve()),
                device_sysfs=str((sysfs / "device").resolve()),
                uevent=read(sysfs / "device/uevent"),
                unique_id=read(sysfs / "device/unique_id"),
            )
    except OSError as error:
        result["error"] = str(error)
    return result


def command(args):
    try:
        run = subprocess.run(args, capture_output=True, text=True, timeout=4)
        return {
            "returncode": run.returncode,
            "output": (run.stdout + run.stderr)[-65536:],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def collect():
    patterns = (
        "/sys/class/kfd/kfd/topology/nodes/*/properties",
        "/sys/class/kfd/kfd/topology/nodes/*/gpu_id",
        "/sys/class/drm/card*/device/gpu_busy_percent",
        "/sys/class/drm/card*/device/mem_info_vram_used",
        "/sys/class/drm/card*/device/uevent",
        "/sys/class/drm/card*/device/unique_id",
        "/sys/class/drm/card*/device/ras/*_err_count",
        "/sys/module/amdgpu/parameters/*",
    )
    paths = [
        "/proc/loadavg",
        "/proc/meminfo",
        "/proc/pressure/cpu",
        "/proc/pressure/io",
        "/proc/self/cgroup",
        "/sys/module/amdgpu/version",
        "/sys/module/amdgpu/srcversion",
    ]
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))
    result = {
        "timestamp": time.time(),
        "uname": list(os.uname()),
        "environment": {key: os.getenv(key) for key in ENV_KEYS},
        "pid_namespace": link("/proc/self/ns/pid"),
        "pid1_namespace": link("/proc/1/ns/pid"),
        "files": {path: read(path) for path in paths},
        "devices": [device(p) for p in sorted(glob.glob("/dev/dri/*"))]
        + [device("/dev/kfd")],
    }
    # Inspect only names/states/stacks; never collect other processes' argv/env.
    counts = collections.Counter()
    blocked = []
    scanned = 0
    for task in glob.iglob("/proc/[0-9]*/task/[0-9]*"):
        if scanned >= 20000:
            break
        scanned += 1
        status = read(f"{task}/status")
        state = next(
            (x for x in status.splitlines() if x.startswith("State:")), "unknown"
        )
        counts[state] += 1
        if "D (" in state and len(blocked) < 128:
            blocked.append(
                {
                    "task": task,
                    "comm": read(f"{task}/comm"),
                    "state": state,
                    "wchan": read(f"{task}/wchan"),
                    "stack": read(f"{task}/stack"),
                }
            )
    result["visible_task_states"] = dict(counts)
    result["visible_tasks_scanned"] = scanned
    result["blocked_visible_tasks"] = blocked
    result["kernel_log"] = command(["dmesg", "--kernel", "--ctime"])
    result["module_info"] = command(["modinfo", "amdgpu"])
    return result


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(collect(), indent=2) + "\n")
