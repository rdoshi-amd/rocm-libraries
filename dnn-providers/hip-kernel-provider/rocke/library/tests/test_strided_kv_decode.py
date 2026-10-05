# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU contracts for non-paged KV metadata and the tiled decode ABI."""

from importlib import import_module
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from kernels.common.attention_kv_cache import KvTensorLayout, StridedKvCacheLayout
from kernels.common.attention_unified import _3d_signature
from rocke.core.verify import verify


@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_attention_strided_kv_golden(flavor):
    tests = Path(__file__).resolve().parent
    golden = json.loads(
        (tests / "golden" / "attention_strided_kv_ir_sha256.json").read_text()
    )
    from rocke.assets import platform_root

    env = dict(os.environ, ROCKE_LLVM_FLAVOR=flavor, ROCKE_BACKEND="python")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(platform_root() / "tests" / "instances" / "parity"),
            str(platform_root() / "python"),
            str(tests.parent),
        ]
    )
    assert set(golden[flavor]) == {str(i) for i in range(12)}
    for idx, expected in golden[flavor].items():
        data = subprocess.check_output(
            [
                sys.executable,
                str(tests / "parity" / "attention_strided_kv_emit.py"),
                idx,
            ],
            env=env,
        )
        assert hashlib.sha256(data).hexdigest() == expected, (flavor, idx)


@pytest.mark.parametrize("shape", [(3, 2, 65, 64), (2, 4, 17, 128), (1, 1, 1, 256)])
def test_contiguous_layout_arguments(shape):
    b, h, c, d = shape
    bshd = KvTensorLayout(shape, (c * h * d, d, h * d, 1))
    bhsd = KvTensorLayout(shape, (h * c * d, c * d, d, 1))
    args = StridedKvCacheLayout(bshd, bhsd).arguments()
    assert args["k_stride_token_bytes"] == h * d * 2
    assert args["v_stride_head_bytes"] == c * d * 2
    assert args["k_span_bytes"] == ((c - 1) * h * d + d) * 2
    assert args["v_span_bytes"] == c * d * 2
    assert bshd.storage_span_bytes == bhsd.storage_span_bytes == b * h * c * d * 2


def test_outer_padding_and_large_batch_offsets():
    layout = KvTensorLayout((2, 3, 65, 64), (1 << 32, 65 * 80, 80, 1))
    assert layout.arguments("k")["k_stride_batch_bytes"] == 1 << 33
    assert layout.head_span_bytes == (64 * 80 + 64) * 2


@pytest.mark.parametrize(
    "shape,strides,reason",
    [
        ((2, 2, 65, 64), (2 * 65 * 128, 65 * 128, 128, 2), "unit D"),
        ((2, 2, 65, 64), (2 * 65 * 64, 65 * 64, 63, 1), "alignment"),
        ((2, 2, 65, 64), (2 * 65 * 64, 64, 64, 1), "overlap"),
        ((2, 2, 65, 64), (-2 * 65 * 64, 65 * 64, 64, 1), "positive"),
        ((2, 2, 65, 64), (0, 65 * 64, 64, 1), "positive"),
        ((1, 1, 1 << 25, 64), (1 << 32, 1 << 31, 64, 1), "buffer-offset"),
        ((2, 2, 65, 64), (1 << 63, 65 * 64, 64, 1), "64-bit"),
    ],
)
def test_unsupported_layouts_decline(shape, strides, reason):
    with pytest.raises(ValueError, match=reason):
        KvTensorLayout(shape, strides)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_strided_segment_signature_and_identity(arch, dtype):
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    fields = dict(
        head_size=64,
        block_size=16,
        num_query_heads=4,
        num_kv_heads=2,
        dtype=dtype,
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=4,
    )
    paged = module.UnifiedAttention3DTiledSpec(**fields)
    strided = module.UnifiedAttention3DTiledSpec(**fields, kv_layout="strided")
    assert strided.kernel_name() == paged.kernel_name() + "_stridedkv"
    kernel = module.build_unified_attention_3d_tiled(strided, arch=arch)
    assert not verify(kernel)
    signature = _3d_signature(dtype, strided_kv=True)
    assert [p.name for p in kernel.params] == [p["name"] for p in signature]
    assert [p.type.name.replace(" ", "") for p in kernel.params] == [
        p["type"].replace(" ", "") for p in signature
    ]
    for kw in (
        dict(kv_storage_dtype="fp8e4m3"),
        dict(use_wide_kv_load=True),
        dict(use_i64_kv_addr=True),
    ):
        with pytest.raises(ValueError, match="strided KV requires"):
            module.UnifiedAttention3DTiledSpec(**fields, kv_layout="strided", **kw)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_strided_dispatch_is_exclusive(arch):
    from dataclasses import replace
    from dispatch.attention import (
        AttentionRequest,
        dispatch_attention,
        attention_candidates,
        attention_execution_candidates,
    )

    req = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=65,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype="bf16",
        kv_layout="strided",
        target_ctas=8,
    )
    result = dispatch_attention(req)
    assert result.candidate.name == "attention_strided_decode"
    assert result.spec.kernel_spec.kv_layout == "strided"
    for candidates in (attention_candidates(), attention_execution_candidates()):
        assert [c.name for c in candidates if c.admits(req)[0]] == [
            "attention_strided_decode"
        ]
    for changes in (
        dict(seqlen_q=2),
        dict(mask_type=1),
        dict(use_fp8=True),
        dict(use_sinks=True),
        dict(arch="gfx1250"),
    ):
        assert not result.candidate.admits(replace(req, **changes))[0]
    assert not result.candidate.admits(replace(req, kv_layout="paged"))[0]


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_native_strided_builder_matches_python(arch, flavor, monkeypatch):
    from dataclasses import asdict

    engine = pytest.importorskip("rocke_engine")
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    module = import_module(f"kernels.{arch}.attention_tiled_3d")
    spec = module.UnifiedAttention3DTiledSpec(
        head_size=64,
        block_size=16,
        num_query_heads=4,
        num_kv_heads=2,
        dtype="fp16",
        use_sinks=False,
        sliding_window=0,
        has_softcap=False,
        num_segments=4,
        num_seqs=3,
        kv_layout="strided",
    )
    kernel = module.build_unified_attention_3d_tiled(spec, arch=arch)
    expected = _lower_kernel_to_llvm_python(kernel, arch=arch)
    if arch == "gfx950":
        reduce = module.UnifiedAttentionReduceTiledSpec(
            head_size=64,
            num_query_heads=4,
            num_kv_heads=2,
            dtype="fp16",
            num_segments=4,
        )
        expected += _lower_kernel_to_llvm_python(
            module.build_unified_attention_reduce_tiled(reduce, arch=arch), arch=arch
        )
    actual = getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")(
        asdict(spec), arch
    )
    assert actual == expected
