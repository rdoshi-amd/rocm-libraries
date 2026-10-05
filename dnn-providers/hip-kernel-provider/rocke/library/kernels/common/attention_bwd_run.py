# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host launch glue of the layout-general attention backward.

Runs an :class:`~kernels.common.attention_bwd_plan.AttnBwdPlan` on one
stream: ``prep -> main -> convert(dQ) [-> convert(dK), convert(dV)]``, in the
plan's launch order, with every kernel argument packed in the
``rocke.attn_bwd.v3`` order.

The caller owns every buffer. The workspace is supplied by the caller and
sized by ``plan.workspace_bytes`` (256-byte aligned base); the run function
allocates no device memory and issues no memset: the prep kernel initialises
every workspace byte a later kernel reads.

* :func:`attn_bwd_kernels` compiles (and loads) the plan's kernels, cached by
  kernel name, arch, LLVM flavor and comgr version.
* :func:`pack_attn_bwd_args` packs one launch's kernel arguments (host only).
* :func:`run_attn_bwd` checks the workspace and issues the launches.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Mapping, MutableMapping
from typing import Any

from kernels.common.attention_bwd import attn_bwd_params, build_attn_bwd_main
from kernels.common.attention_bwd_aux import (
    build_attn_bwd_convert,
    build_attn_bwd_prep,
)

__all__ = [
    "WORKSPACE_ALIGN",
    "attn_bwd_kernels",
    "pack_attn_bwd_args",
    "run_attn_bwd",
]

WORKSPACE_ALIGN = 256

_BUILDERS = {
    "main": build_attn_bwd_main,
    "prep": build_attn_bwd_prep,
    "convert": build_attn_bwd_convert,
}

_PROCESS_CACHE: dict = {}


def _flavor() -> str:
    flavor = os.environ.get("ROCKE_LLVM_FLAVOR")
    if flavor:
        return flavor
    from rocke.core.lower_llvm import _detect_llvm_flavor

    return _detect_llvm_flavor()


def _comgr_version() -> str:
    try:
        from rocke.runtime.comgr import resolved_lib_rocm_version

        v = resolved_lib_rocm_version()
    except Exception:  # noqa: BLE001 - recorded as unknown in the cache key
        v = None
    return "unknown" if v is None else f"{v[0]}.{v[1]}"


def attn_bwd_kernels(plan, *, cache: MutableMapping | None = None) -> dict:
    """``{kernel name: (artifact, module, function)}`` for every plan launch.

    Each kernel is built from the plan's spec for its stage, compiled through
    comgr on the Python lowering backend and loaded once; ``cache`` (default:
    a process-wide dict) is keyed by ``(name, arch, flavor, comgr)``.
    """
    from rocke.helpers.compile import compile_kernel
    from rocke.runtime.hip_module import Runtime

    cache = _PROCESS_CACHE if cache is None else cache
    flavor, comgr = _flavor(), _comgr_version()
    out = {}
    for step in plan.launches:
        key = (step.kernel, plan.arch, flavor, comgr)
        if key not in cache:
            spec = plan.specs[step.stage]
            kernel = _BUILDERS[step.stage](spec, arch=plan.arch)
            art = compile_kernel(
                kernel, arch=plan.arch, capture_ir_text=False, backend="python"
            )
            if art.kernel_name != step.kernel:
                raise AssertionError(
                    f"built kernel {art.kernel_name} differs from the plan's {step.kernel}"
                )
            mod = Runtime().load_module(art.hsaco)
            cache[key] = (art, mod, mod.get_function(art.kernel_name))
        out[step.kernel] = cache[key]
    return out


def pack_attn_bwd_args(
    step,
    *,
    tensors: Mapping[str, int],
    workspace: int,
    workspace_layout: Mapping[str, tuple[int, int]],
) -> bytes:
    """Kernel-argument bytes of one :class:`LaunchStep` (``rocke.attn_bwd.v3``).

    Pointer roles resolve to ``workspace + offset`` for workspace sub-buffers,
    to ``tensors[role]`` for tensor roles, and to 0 for unused (``None``)
    pointers. A tensor role missing from ``tensors`` raises ``ValueError``.
    """
    out = []
    for name, fmt in attn_bwd_params(step.stage):
        if fmt == "Q":
            role = step.pointers[name]
            if role is None:
                ptr = 0
            elif role in workspace_layout:
                ptr = int(workspace) + int(workspace_layout[role][0])
            elif role in tensors and tensors[role] is not None:
                ptr = int(tensors[role])
            else:
                raise ValueError(f"{step.stage}: no device pointer for role {role!r}")
            out.append(struct.pack("<Q", ptr))
        elif fmt == "f":
            out.append(struct.pack("<f", float(step.scalars[name])))
        else:
            # "q" (i64) or "i" (i32); a value outside the kernarg type is a
            # predicate bug, reported by name rather than as a struct error.
            v = int(step.scalars[name])
            bits = 64 if fmt == "q" else 32
            if not -(1 << (bits - 1)) <= v < (1 << (bits - 1)):
                raise ValueError(
                    f"{step.stage}: kernarg {name}={v} does not fit i{bits}"
                )
            out.append(struct.pack("<" + fmt, v))
    return b"".join(out)


def run_attn_bwd(
    plan,
    *,
    tensors: Mapping[str, int],
    workspace: int,
    workspace_size: int,
    stream: int = 0,
    cache: MutableMapping | None = None,
) -> list[Any]:
    """Launch ``prep -> main -> convert(s)`` of ``plan`` on ``stream``.

    ``tensors`` maps the plan's pointer roles (``q k v o do lse dq dk dv``,
    plus ``seq_len_q``/``seq_len_kv``/``offsets_q``/``offsets_kv`` when the
    plan uses them) to device addresses. ``workspace`` is the caller's
    workspace base (``WORKSPACE_ALIGN``-byte aligned) of ``workspace_size``
    bytes, at least ``plan.workspace_bytes``. Nothing is allocated and no
    memset is issued. Returns the launched kernel names in order.
    """
    from rocke.runtime.hip_module import Runtime

    if workspace_size < plan.workspace_bytes:
        raise ValueError(
            f"workspace of {workspace_size} bytes is smaller than the plan's "
            f"{plan.workspace_bytes}"
        )
    if plan.workspace_bytes and int(workspace) % WORKSPACE_ALIGN:
        raise ValueError(f"workspace base is not {WORKSPACE_ALIGN}-byte aligned")
    kernels = attn_bwd_kernels(plan, cache=cache)
    packed = [
        pack_attn_bwd_args(
            step,
            tensors=tensors,
            workspace=workspace,
            workspace_layout=plan.workspace_layout,
        )
        for step in plan.launches
    ]
    rt = Runtime()
    names = []
    for step, args in zip(plan.launches, packed):
        _art, _mod, fn = kernels[step.kernel]
        rt.launch(fn, tuple(step.grid), (int(step.block), 1, 1), args, stream=stream)
        names.append(step.kernel)
    return names
