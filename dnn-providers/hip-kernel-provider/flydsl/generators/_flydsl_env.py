# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pinned build environment for the FlyDSL AOT generators.

Every generator in this directory compiles against *one* declared toolchain: a
pinned FlyDSL wheel, the vendored kernel sources under ``kernels_src/``, and a
recorded ROCm version. All three go into the per-arch ``SOURCE.md``, because a
checked-in code object regenerated against a different toolchain is a change
that reviews as a no-op diff. The FlyDSL release notes and this tree's Flash2
precedent both record ~1.4x swings from the ROCm version alone on identical
source.

The environment is deliberately process-wide and single-arch. FlyDSL reads
``ARCH`` from ``os.environ`` at each compile (``flydsl/utils/env.py``:
``CompileEnvManager.arch``), so mutating it mid-run would work -- and would make
the arch a kernel object was *built* for independent of the arch its filename
claims. That is the exact mislabeling hazard the Flash2 CMake block warns about,
so one arch per invocation is enforced here and the generator additionally
checks each object's own ``amdhsa.target`` after the fact.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- Pins -------------------------------------------------------------------

# The FlyDSL *compiler* (pip wheel). `kernels/` is not in the wheel; the kernel
# sources this provider compiles are vendored under kernels_src/ with their own
# per-file provenance headers, so this pin covers the compiler only.
FLYDSL_VERSION = "0.3.4"

# The FlyDSL checkout the vendored kernel sources were taken from. Reproduced in
# kernels/gfx1151/SOURCE.md; not importable here. tools/diff_upstream.py reads
# both of these -- it is the consumer that turns them from a claim into a check.
FLYDSL_KERNELS_COMMIT = "89ad52fbbb9e252396829a8a44b12e18b7acd970"
FLYDSL_KERNELS_DESCRIBE = "v0.3.4.1-19-g89ad52f"

# --- Paths ------------------------------------------------------------------

GENERATORS_DIR = Path(__file__).resolve().parent
PROVIDER_DIR = GENERATORS_DIR.parent
KERNELS_SRC_DIR = PROVIDER_DIR / "kernels_src"
KERNELS_OUT_DIR = PROVIDER_DIR / "kernels"


class FlydslEnvError(RuntimeError):
    """The pinned build environment is not the one that is installed."""


def rocm_version() -> str:
    """The ROCm version this run compiles against, for the provenance record.

    Read from ``$ROCM_PATH/.info/version``, which the SDK writes. Absent that we
    return a marker rather than guessing: an unrecorded toolchain in SOURCE.md is
    better than a wrong one, since the wrong one reads as verified.
    """
    explicit = os.environ.get("ROCM_VERSION")
    if explicit:
        return explicit.strip()
    rocm_path = os.environ.get("ROCM_PATH")
    if rocm_path:
        marker = Path(rocm_path) / ".info" / "version"
        if marker.is_file():
            return marker.read_text(encoding="utf-8").strip()
    return "unknown"


def assert_flydsl_version() -> str:
    """Import FlyDSL and confirm it is the pinned release."""
    try:
        import flydsl
    except ImportError as exc:  # pragma: no cover - environment problem
        raise FlydslEnvError(
            "flydsl is not importable. These generators need a Python environment "
            "with the pinned flydsl wheel and torch installed; see REGEN.md."
        ) from exc

    installed = getattr(flydsl, "__version__", None)
    if installed != FLYDSL_VERSION:
        raise FlydslEnvError(
            f"flydsl {installed!r} is installed but this provider pins "
            f"{FLYDSL_VERSION!r}. Regenerating against a different compiler is a "
            "change to the checked-in objects, not a refresh: install the pinned "
            "version, or update FLYDSL_VERSION here and regenerate every arch."
        )
    return installed


def prepare(arch: str) -> None:
    """Put the vendored kernel sources on the path and pin the compile env.

    Must run before any ``kernels.*`` import. ``COMPILE_ONLY=1`` is what lets a
    gfx942/gfx950 object be produced on an RDNA laptop: the launcher traces and
    compiles but never dispatches, so no matching device has to be present.
    """
    if not KERNELS_SRC_DIR.is_dir():
        raise FlydslEnvError(f"vendored kernel sources missing: {KERNELS_SRC_DIR}")

    if "kernels" in sys.modules:
        raise FlydslEnvError(
            "the 'kernels' package was imported before prepare(); the vendored "
            "sources under kernels_src/ may have been shadowed by an upstream "
            "FlyDSL checkout on PYTHONPATH"
        )

    src = str(KERNELS_SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)

    os.environ["ARCH"] = arch
    os.environ["COMPILE_ONLY"] = "1"
    os.environ["FLYDSL_DUMP_IR"] = "1"


def set_dump_dir(dump_dir: Path) -> None:
    """Point the IR dump at `dump_dir` for the next compile.

    One directory per instance. ``hsaco_from_dump`` refuses a directory holding
    more than one compiled kernel rather than picking one, so reusing a dump dir
    across instances fails loudly instead of silently packaging the first object
    under every instance's name.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_DUMP_DIR"] = str(dump_dir)


def provenance() -> dict:
    """The toolchain record embedded in manifests and SOURCE.md."""
    return {
        "flydsl_version": FLYDSL_VERSION,
        "flydsl_kernels_commit": FLYDSL_KERNELS_COMMIT,
        "flydsl_kernels_describe": FLYDSL_KERNELS_DESCRIBE,
        "rocm_version": rocm_version(),
    }
