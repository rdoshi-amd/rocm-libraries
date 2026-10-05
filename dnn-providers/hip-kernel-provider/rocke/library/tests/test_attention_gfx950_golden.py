# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Golden LLVM-IR byte-stability test for the gfx950 forward attention kernels.

Hashes the Python-lowered LLVM IR (SHA256) of representative gfx950 dense, tiled
2D, tiled 3D (segment + reduce) and scalar unified-attention specs and compares
against a checked-in per-flavor fixture. It is the mechanical drift gate for
default-off knobs (default ``mask_type``, ragged-causal key-pad mask): any
spec that does not opt in must keep lowering byte-identically.

``library/kernels/`` has no C++ engine mirror, so the repo-wide byte-identity
gate does not apply; the Python lowering is ground truth. Pure text lowering,
CPU lane (no GPU / comgr).

Re-bless ONLY from a tree whose kernels are known-correct, never to turn a red
drift green:

    cd rocke/library
    PYTHONPATH=../platform/python:. python tests/test_attention_gfx950_golden.py --write
"""

import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

_GOLDEN = Path(__file__).resolve().parent / "golden" / "attention_gfx950_ir_sha256.json"
_FLAVORS = ("llvm20", "llvm22")
_ARCH = "gfx950"

# See test_attention_dense_gfx942_golden.py: keep library/ ahead of tests/ so
# tests/dispatch does not shadow the real dispatch package when run directly.
_LIB_ROOT = str(Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] != _LIB_ROOT:
    sys.path.insert(0, _LIB_ROOT)


@contextmanager
def _pinned_gfx950():
    """Spec selectors consult the running device arch; pin it so the fixture is
    identical on any box."""
    with mock.patch(
        "kernels.common.attention_unified._resolve_attention_arch",
        return_value=_ARCH,
    ):
        yield


def _problem(**kw):
    from kernels.common.attention_unified import UnifiedAttentionProblem

    base = dict(
        total_q=2048,
        num_seqs=1,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        dtype="bf16",
        num_cus=120,
    )
    base.update(kw)
    if "total_q" not in kw:
        base["total_q"] = base["num_seqs"] * base["max_seqlen_q"]
    return UnifiedAttentionProblem(**base)


def _cases():
    """cid -> zero-arg builder returning a KernelDef."""
    from builders.common import attention_spec_builder as bld
    from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES
    from kernels.common.attention_unified import (
        UnifiedAttention2DSpec,
        UnifiedAttention3DSpec,
        _tiled_2d_impl,
        _tiled_3d_impl,
        build_unified_attention_2d,
        build_unified_attention_3d,
    )
    from kernels.gfx950.attention_dense import (
        Gfx950AttentionDenseSpec,
        build_attention_dense,
    )

    tile = DENSE_TILE_GEOMETRIES["default"]
    dense_base = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=16,
        num_kv_heads=4,
        head_size=128,
        causal=True,
        dtype="bf16",
        block_m=int(tile["block_m"]),
        block_n=int(tile["block_n"]),
    )

    def dense(**over):
        d = dict(dense_base)
        d.update(over)
        return lambda: build_attention_dense(Gfx950AttentionDenseSpec(**d), arch=_ARCH)

    def tiled2d(problem_kw, **spec_over):
        def build():
            from dataclasses import replace

            with _pinned_gfx950():
                _, build2d, _ = _tiled_2d_impl(_ARCH)
                spec = bld._spec_gfx950_generic(_problem(**problem_kw))
                if spec_over:
                    spec = replace(spec, **spec_over)
                return build2d(spec, arch=_ARCH)

        return build

    def tiled3d(problem_kw, reduce=False):
        def build():
            with _pinned_gfx950():
                _, red_t, build3d, build_red, _ = _tiled_3d_impl(_ARCH)
                problem = _problem(**problem_kw)
                spec = bld._spec_generic_3d(problem)
                if not reduce:
                    return build3d(spec, arch=_ARCH)
                return build_red(
                    red_t(
                        head_size=problem.head_size,
                        num_query_heads=problem.num_query_heads,
                        num_kv_heads=problem.num_kv_heads,
                        dtype=problem.dtype,
                        num_segments=spec.num_segments,
                    ),
                    arch=_ARCH,
                )

        return build

    def scalar(problem_kw, three_d=False):
        def build():
            problem = _problem(**problem_kw)
            if three_d:
                return build_unified_attention_3d(
                    UnifiedAttention3DSpec(problem, num_segments=8), arch=_ARCH
                )
            return build_unified_attention_2d(
                UnifiedAttention2DSpec(problem), arch=_ARCH
            )

        return build

    decode = dict(max_seqlen_q=1, max_seqlen_k=4096, num_seqs=4, total_q=4)
    cases = {
        # --- dense ---
        "dense/default_d128_bf16_causal": dense(),
        "dense/default_d128_fp16_full": dense(dtype="fp16", causal=False),
        "dense/default_d64_bf16_causal": dense(head_size=64),
        "dense/persist_d128_bf16_causal": dense(persistent=True, num_persistent=256),
        "dense/swa_d128_bf16_w128": dense(sliding_window=128),
        "dense/sinks_d128_bf16_causal": dense(use_sinks=True),
        "dense/ragged_self_d128_bf16_causal": dense(
            seqlen_q=197, seqlen_kv=197, ragged=True
        ),
        "dense/ragged_self_d128_bf16_full": dense(
            seqlen_q=197, seqlen_kv=197, ragged=True, causal=False
        ),
        "dense/ragged_self_persist_d128_bf16_causal": dense(
            seqlen_q=197, seqlen_kv=197, ragged=True, persistent=True
        ),
        "dense/bottom_right_d128_bf16_512x1024": dense(
            seqlen_kv=1024, causal_bottom_right=True
        ),
        "dense/bottom_right_ragged_d128_bf16_197x400": dense(
            seqlen_q=197, seqlen_kv=400, ragged=True, causal_bottom_right=True
        ),
        "dense/top_left_aligned_d128_bf16_512x1024": dense(seqlen_kv=1024),
        # --- tiled 2D (gfx950 generic spec builder) ---
        "tiled2d/prefill_d128_fp16_mha": tiled2d(
            dict(num_query_heads=16, num_kv_heads=16, dtype="fp16")
        ),
        "tiled2d/prefill_d128_bf16_gqa64x8": tiled2d(
            dict(num_query_heads=64, num_kv_heads=8)
        ),
        "tiled2d/prefill_d256_bf16_gqa64x8": tiled2d(
            dict(
                num_query_heads=64,
                num_kv_heads=8,
                head_size=256,
                max_seqlen_q=1024,
                max_seqlen_k=1024,
                total_q=1024,
            )
        ),
        "tiled2d/prefill_d128_bf16_sw128": tiled2d(dict(sliding_window=128)),
        "tiled2d/prefill_d128_bf16_sinks": tiled2d(dict(use_sinks=True)),
        "tiled2d/prefill_d128_bf16_softcap": tiled2d(dict(softcap=30.0)),
        "tiled2d/prefill_d128_bf16_alibi": tiled2d(dict(use_alibi=True)),
        "tiled2d/prefill_d128_bf16_qqbias": tiled2d(dict(use_qq_bias=True)),
        "tiled2d/prefill_d128_bf16_cross_length": tiled2d(
            dict(max_seqlen_q=1024, max_seqlen_k=3072, total_q=1024)
        ),
        "tiled2d/prefill_d128_bf16_multiseq": tiled2d(
            dict(num_seqs=4, max_seqlen_q=512, max_seqlen_k=512, total_q=2048)
        ),
        "tiled2d/prefill_d128_bf16_mphsplit_sw0": tiled2d(
            dict(num_query_heads=64, num_kv_heads=8),
            use_mask_phase_split=True,
        ),
        # --- tiled 3D ---
        "tiled3d/decode_d128_bf16": tiled3d(decode),
        "tiled3d/decode_d128_bf16_sw128": tiled3d(dict(decode, sliding_window=128)),
        "tiled3d/decode_d128_bf16_sinks": tiled3d(dict(decode, use_sinks=True)),
        "tiled3d/decode_d256_bf16": tiled3d(dict(decode, head_size=256)),
        "tiled3d/reduce_d128_bf16": tiled3d(decode, reduce=True),
        # --- scalar unified ---
        "scalar2d/prefill_d128_bf16": scalar(
            dict(total_q=64, max_seqlen_q=64, max_seqlen_k=64)
        ),
        "scalar2d/prefill_d128_bf16_sw16": scalar(
            dict(total_q=64, max_seqlen_q=64, max_seqlen_k=64, sliding_window=16)
        ),
        "scalar3d/decode_d128_bf16": scalar(decode, three_d=True),
    }
    return cases


def _current_flavor():
    from rocke.core.lower_llvm import _resolve_llvm_flavor

    return _resolve_llvm_flavor()


def _sha_for(build, flavor):
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    llvm = _lower_kernel_to_llvm_python(build(), arch=_ARCH, llvm_flavor=flavor)
    data = llvm.encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data)


def _build_doc():
    doc = {"schema": "attention_gfx950.ir_golden_sha256/v1", "flavors": {}}
    for flavor in _FLAVORS:
        cases = {}
        for cid, build in _cases().items():
            try:
                sha, nbytes = _sha_for(build, flavor)
                cases[cid] = {"sha256": sha, "bytes": nbytes}
            except Exception as e:  # pragma: no cover - diagnostic
                cases[cid] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
        doc["flavors"][flavor] = {"cases": cases}
    return doc


def test_attention_gfx950_ir_matches_golden():
    import pytest

    if not _GOLDEN.exists():
        pytest.skip("gfx950 attention golden fixture missing; generate with --write")
    golden = json.loads(_GOLDEN.read_text())
    flavor = _current_flavor()
    gflav = golden.get("flavors", {}).get(flavor)
    if not gflav:
        pytest.skip(f"no gfx950 attention golden recorded for llvm flavor {flavor!r}")
    drift = []
    for cid, build in _cases().items():
        want = gflav["cases"].get(cid, {}).get("sha256")
        if want is None:
            continue
        got, _ = _sha_for(build, flavor)
        if got != want:
            drift.append(f"{cid}: {want} -> {got}")
    assert not drift, "gfx950 attention IR drift vs golden:\n  " + "\n  ".join(drift)


if __name__ == "__main__":
    if "--write" in sys.argv:
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(json.dumps(_build_doc(), indent=2, sort_keys=True) + "\n")
        print(f"wrote {_GOLDEN}")
    else:
        test_attention_gfx950_ir_matches_golden()
