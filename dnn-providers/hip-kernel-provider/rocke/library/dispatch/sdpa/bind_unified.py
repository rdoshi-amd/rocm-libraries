# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SDPA binding for the unified attention candidates (ST2 / U-03).

Narrows the SDPA view of the existing unified dispatch.  The existing
dispatch/attention/ candidates are not edited; this module applies
SDPA-specific predicates on top of the registry's own verdict.

Rules added in U-03:
  fp16 d=256  -> declined (scalar fallback; fails tolerance).  ATT-34.
  gfx1151 bf16 -> declined (WMMA is fp16-only until RD-01 / ST8).

bf16 d=256 is served on gfx950 by the dedicated ``attention_gfx950_d256``
(prefill) and ``attention_d256_decode`` candidates, and on gfx942 by the
``_d256_gfx942_fast`` cohort inside the generic unified kernel -- gfx942 has
no separate d256 candidate, so a generic ``attention_unified_2d`` verdict is
the correct answer there, not a fallback.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from dispatch.attention import (
    ATTENTION_ROUTE_REGISTRY,
    UNIFIED_HEAD_SIZES,
    AttentionRequest,
)

if TYPE_CHECKING:
    from .common import SdpaRequest

# Candidate support in this registry is shape-dependent: the specialized
# candidates are cohort-gated, so a degenerate request reaches only the generic
# fallback and hides every fast path. SdpaRequest carries no shape fields (ST9
# owns that schema), so probe one representative prefill shape and one
# representative decode shape instead, and admit if either is served.
#
# The head counts are load-bearing, not filler: at nhead_q=1 a 2048x2048
# request routes to the 3D decode path, which drops attention_gfx950_d256 from
# the verdict. 32/8 is a realistic GQA ratio and routes prefill to 2D.
_PROBE_NHEAD_Q = 32
_PROBE_NHEAD_K = 8
_PROBE_SHAPES = (
    (2048, 2048),  # prefill
    (1, 4096),     # decode
)

# ── SDPA-level decline predicates ─────────────────────────────────────────────

_FP16_D256_REASON = (
    "fp16 d=256 falls back to the scalar kernel, which fails tolerance on some "
    "shapes and is slow enough at long sequence lengths to stall graph capture "
    "(ATT-34); use bf16 at d=256, or fp16 at d in {64, 128}"
)

_GFX1151_BF16_REASON = (
    "gfx1151 serves attention through a WMMA kernel that is fp16-only today; "
    "bf16 support lands with RD-01 (ST8). The attention registry cannot express "
    "this gap, so it is declined here explicitly"
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
    if req.dtype == "bf16" and req.arch == "gfx1151":
        return False, _GFX1151_BF16_REASON
    return None


def _probe_request(
    req: "SdpaRequest", seqlen_q: int, seqlen_k: int
) -> AttentionRequest:
    return AttentionRequest(
        batch=1,
        nhead_q=_PROBE_NHEAD_Q,
        nhead_k=_PROBE_NHEAD_K,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        hdim_q=req.head_dim,
        hdim_v=req.head_dim,
        arch=req.arch,
        dtype=req.dtype,
    )


def sdpa_unified_admits(req: "SdpaRequest") -> "tuple[bool, str]":
    """Top-level SDPA admission check for the unified family.

    Called by the SDPA dispatcher (ST9) and directly by tests.
    Returns ``(True, "ok")`` only when:
      - no SDPA-level decline applies, AND
      - at least one unified candidate admits a representative shape.

    Does not select a candidate or produce a spec; that is ST9's job.
    """
    decline = _sdpa_unified_declines(req)
    if decline is not None:
        return decline

    if req.head_dim not in UNIFIED_HEAD_SIZES:
        return False, (
            f"head_dim {req.head_dim} is not a unified head size; "
            f"supported: {sorted(UNIFIED_HEAD_SIZES)}"
        )

    # ``supported()`` applies each candidate's full admits() closure and hides
    # opt-in candidates from auto selection, so it is the registry's own verdict
    # rather than a re-implementation of it.
    for seqlen_q, seqlen_k in _PROBE_SHAPES:
        if ATTENTION_ROUTE_REGISTRY.supported(_probe_request(req, seqlen_q, seqlen_k)):
            return True, "ok"

    return False, (
        f"no unified candidate covers arch={req.arch!r} "
        f"dtype={req.dtype!r} d={req.head_dim}"
    )


def sdpa_unified_candidates(req: "SdpaRequest") -> "tuple[str, ...]":
    """Names of the unified candidates that serve ``req``, for tests and triage.

    Deduped across the probe shapes, in registry order. Empty when the request
    is declined or uncovered.
    """
    if _sdpa_unified_declines(req) is not None:
        return ()
    names: list[str] = []
    for seqlen_q, seqlen_k in _PROBE_SHAPES:
        for candidate in ATTENTION_ROUTE_REGISTRY.supported(
            _probe_request(req, seqlen_q, seqlen_k)
        ):
            if candidate.name not in names:
                names.append(candidate.name)
    return tuple(names)
