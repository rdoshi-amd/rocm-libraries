# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check GPU targets and the launched test environment without GPU hardware."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_CONFIG = (
    SCRIPT_DIR.parent.parent
    / "projects/hipblaslt/cmake/tensilelite_supported_architectures.cmake"
)


class SanitizerConfigurationTest(unittest.TestCase):
    def test_gpu_targets(self):
        # Host ASAN must not require GPU page-fault retry support. Full ASAN and
        # the standalone device-ASAN option must retain their xnack+ targets.
        for mode, device_asan, requires_xnack in (
            ("HOST_ASAN", "OFF", False),
            ("ASAN", "OFF", True),
            ("OFF", "OFF", False),
            ("HOST_ASAN", "ON", True),
        ):
            with self.subTest(mode=mode, device_asan=device_asan):
                with tempfile.TemporaryDirectory() as temp:
                    script = Path(temp) / "targets.cmake"
                    suffix = ":xnack+" if requires_xnack else ""
                    script.write_text(
                        "cmake_minimum_required(VERSION 3.25)\n"
                        f'set(THEROCK_SANITIZER "{mode}")\n'
                        f"set(HIPBLASLT_ENABLE_ASAN {device_asan})\n"
                        f'include("{TARGET_CONFIG.as_posix()}")\n'
                        "foreach(arch gfx908 gfx90a gfx942 gfx950)\n"
                        "  tensilelite_offload_target(actual ${arch})\n"
                        f'  set(expected "${{arch}}{suffix}")\n'
                        "  if(NOT actual STREQUAL expected)\n"
                        '    message(FATAL_ERROR "${actual} != ${expected}")\n'
                        "  endif()\n"
                        "  if(NOT expected IN_LIST BASE_ARCHITECTURES)\n"
                        '    message(FATAL_ERROR "Missing default ${expected}")\n'
                        "  endif()\n"
                        "endforeach()\n"
                        "tensilelite_offload_target(explicit gfx90a:xnack-)\n"
                        'if(NOT explicit STREQUAL "gfx90a:xnack-")\n'
                        '  message(FATAL_ERROR "Explicit target was changed")\n'
                        "endif()\n",
                        encoding="utf-8",
                    )
                    subprocess.run(["cmake", "-P", str(script)], check=True)

    def test_launched_environment(self):
        for group, xnack, expected_xnack, expected_omp in (
            ("gfx90a-host-asan", "0", "0", "1"),
            ("gfx90a-host-asan", None, None, "1"),
            ("gfx90a-host-asan-debug", "0", "0", "1"),
            ("gfx90a-asan", "0", "1", "1"),
            ("gfx90a-asan", None, "1", "1"),
            ("gfx90a", "0", "0", "2"),
        ):
            with self.subTest(group=group, xnack=xnack):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    binary_dir = root / "bin"
                    binary_dir.mkdir()
                    binary = binary_dir / "hipblaslt-test"
                    binary.write_text(
                        "#!/usr/bin/env python3\n"
                        "import json, os, sys\n"
                        "print(json.dumps({\n"
                        "  'xnack': os.getenv('HSA_XNACK'),\n"
                        "  'omp': os.getenv('OMP_NUM_THREADS'),\n"
                        "  'args': sys.argv[1:],\n"
                        "}))\n",
                        encoding="utf-8",
                    )
                    binary.chmod(0o755)
                    launcher = root / "test_hipblaslt.py"
                    shutil.copyfile(SCRIPT_DIR / "test_hipblaslt.py", launcher)
                    env = os.environ.copy()
                    env.update(
                        ARTIFACT_GROUP=group,
                        AMDGPU_FAMILIES="gfx90a",
                        RUNNER_OS="Linux",
                        THEROCK_BIN_DIR=str(binary_dir),
                        THEROCK_DIR=str(root),
                        TEST_TYPE="quick",
                        SHARD_INDEX="1",
                        TOTAL_SHARDS="1",
                        OMP_NUM_THREADS="2",
                    )
                    if xnack is None:
                        env.pop("HSA_XNACK", None)
                    else:
                        env["HSA_XNACK"] = xnack
                    result = subprocess.run(
                        [sys.executable, str(launcher)],
                        env=env,
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    actual = json.loads(result.stdout)
                    self.assertEqual(actual["xnack"], expected_xnack)
                    self.assertEqual(actual["omp"], expected_omp)
                    self.assertEqual(actual["args"], ["--gtest_filter=*smoke*"])


if __name__ == "__main__":
    unittest.main()
