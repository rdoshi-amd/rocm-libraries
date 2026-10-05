# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SDPA binding for the unified attention candidates (ST2 / U-03).

Narrows the SDPA view of the existing unified dispatch.  The existing
dispatch/attention/ candidates are not edited; this module applies
SDPA-specific predicates on top of their Capability checks.

Rule added in U-03:
  fp16 d=256 -> declined (scalar fallback; tolerance failures + gfx942 miscompile).
  Tracked as ATT-34.  bf16 d=256 is served by the existing fast-path candidates.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .common import SdpaRequest

# ── SDPA-level decline predicates ─────────────────────────────────────────────

_FP16_D256_REASON = (
    "fp16 d=256 falls back to the scalar kernel which fails tolerance on some "
    "shapes and triggers a known hipcc miscompile on gfx942 (ATT-34); "
    "use bf16 or d in {64, 128}"
)


def _sdpa_unified_declines(req: "SdpaRequest") -> "tuple[bool, str] | None":
    """Return ``(False, reason)`` if the SDPA layer declines this request,
    or ``None`` if no SDPA-specific decline applies.

    Rules added here are in addition to, not instead of, the per-candidate
    Capability and support-closure checks in dispatch/attention/.
    """
    # dtype is already normalized to "fp16" by SdpaRequest.__post_init__.
    if req.dtype == "fp16" and req.head_dim == 256:
        return False, _FP16_D256_REASON
    return None


def sdpa_unified_admits(req: "SdpaRequest") -> "tuple[bool, str]":
    """Top-level SDPA admission check for the unified family.

    Called by the SDPA dispatcher (ST9) and directly by tests.
    Returns ``(True, "ok")`` only when:
      - no SDPA-level decline applies, AND
      - at least one existing unified candidate admits the underlying request.

    Does not select a candidate or produce a spec; that is ST9's job.
    """
    decline = _sdpa_unified_declines(req)
    if decline is not None:
        return decline

    # Delegate to the existing unified candidates' Capability checks.
    # We do NOT forward to the full support closure (which needs a real problem
    # and arch probe); Capability is the declarative tier that covers dtype/hdim/arch.
    from dispatch.attention import ATTENTION_ROUTE_REGISTRY
    from dispatch.attention.common import AttentionRequest, _request_errors

    # Build a minimal AttentionRequest for Capability check only.
    # Shape fields not yet on SdpaRequest are filled with safe defaults.
    attn_req = AttentionRequest(
        batch=1,
        nhead_q=1,
        nhead_k=1,
        seqlen_q=1,
        seqlen_k=1,
        hdim_q=req.head_dim,
        hdim_v=req.head_dim,
        arch=req.arch,
        dtype=req.dtype,
    )
    errors = _request_errors(attn_req)
    if errors:
        return False, "; ".join(errors)

    # Check whether any unified candidate's Capability covers this request.
    for candidate in ATTENTION_ROUTE_REGISTRY.candidates():
        if "unified" not in candidate.name:
            continue
        if candidate.capability is None:
            continue
        ok, _ = candidate.capability.check(attn_req)
        if ok:
            return True, "ok"

    return False, (
        f"no unified candidate covers arch={req.arch!r} "
        f"dtype={req.dtype!r} d={req.head_dim}"
    )
