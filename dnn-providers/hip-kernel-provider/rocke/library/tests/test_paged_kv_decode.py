# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Admission, identity and native parity for the paged tile loader."""

import hashlib
import json
from dataclasses import asdict, replace
from importlib import import_module
from pathlib import Path

import pytest
from kernels.common.attention_paged_decode import BUFFER_LIMIT, validate_paged_decode


@pytest.mark.parametrize("idx", range(48))
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_paged_native_parity_and_golden(idx, flavor, monkeypatch):
    engine = pytest.importorskip("rocke_engine")
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    from tests.parity.attention_paged_kv_emit import build, spec_for_index

    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    spec, arch = spec_for_index(idx)
    expected = _lower_kernel_to_llvm_python(build(spec, arch), arch=arch)
    if arch == "gfx942" and spec.head_size == 256:
        import re

        sizes = re.findall(r"global \[(\d+) x i8\]", expected)
        assert sizes and max(map(int, sizes)) <= 65536
    golden = json.loads(
        (
            Path(__file__).parent / "golden" / "attention_paged_kv_ir_sha256.json"
        ).read_text()
    )
    assert hashlib.sha256(expected.encode()).hexdigest() == golden[flavor][str(idx)]
    if arch == "gfx950":
        m = import_module(f"kernels.{arch}.attention_tiled_3d")
        reduce = m.UnifiedAttentionReduceTiledSpec(
            head_size=spec.head_size,
            num_query_heads=8,
            num_kv_heads=2,
            dtype=spec.dtype,
            num_segments=8,
            waves_per_eu=3,
        )
        expected += _lower_kernel_to_llvm_python(
            m.build_unified_attention_reduce_tiled(reduce, arch=arch), arch=arch
        )
    assert (
        getattr(engine, f"{arch}_attention_tiled_3d_lower_llvm")(asdict(spec), arch)
        == expected
    )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_paged_selection_and_identity(arch):
    from dispatch.attention import AttentionRequest, dispatch_attention

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
        kv_block_size=1,
        target_ctas=8,
    )
    result = dispatch_attention(req)
    assert result.candidate.name == "attention_paged_decode_t32"
    assert result.spec.kernel_spec.tile_size == 32
    for page in (1, 16, 32, 64):
        selected = dispatch_attention(
            replace(req, kv_block_size=page, algorithm="paged_decode_t32")
        )
        spec = selected.spec.kernel_spec
        assert spec.block_size == page and spec.tile_size == 32
        assert selected.spec.with_num_kv_blocks(100).kernel_spec == spec
    spec = result.spec.kernel_spec
    other = replace(spec, block_size=64, tile_size_override=64)
    assert other.kernel_name() != replace(other, tile_size_override=32).kernel_name()
    assert asdict(spec)["tile_size_override"] == 32
    for kwargs in (
        {"tile_size_override": 16},
        {"use_i64_kv_addr": True},
        {"kv_storage_dtype": "fp8e4m3"},
        {"use_wide_kv_load": True},
    ):
        with pytest.raises(ValueError, match="paged gather"):
            replace(spec, **kwargs)


class Tensor:
    def __init__(self, shape, dtype="torch.float16"):
        self.shape, self.dtype = shape, dtype
        self.device, self.is_cuda = "cuda:0", True

    def numel(self):
        import math

        return math.prod(self.shape)

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return 4096


def inputs():
    from kernels.common.attention_unified import UnifiedAttentionProblem

    p = UnifiedAttentionProblem(
        total_q=3,
        num_seqs=3,
        num_query_heads=8,
        num_kv_heads=2,
        head_size=64,
        block_size=1,
        max_seqlen_q=1,
        max_seqlen_k=33,
        dtype="fp16",
        clamp_arch="gfx950",
    )
    ts = [
        Tensor((3, 8, 64)),
        Tensor((99, 1, 2, 64)),
        Tensor((99, 1, 2, 64)),
        Tensor((3, 8, 64)),
        Tensor((4,), "torch.int32"),
        Tensor((3,), "torch.int32"),
        Tensor((3, 33), "torch.int32"),
    ]
    return p, ts


def test_paged_runtime_bounds():
    p, ts = inputs()
    validate_paged_decode(p, ts)
    for index in (1, 2):
        changed = list(ts)
        changed[index] = Tensor((BUFFER_LIMIT // 256 + 1, 1, 2, 64))
        changed[3 - index] = Tensor(changed[index].shape)
        with pytest.raises(ValueError, match="buffer-offset"):
            validate_paged_decode(p, changed)
    ts[-1] = Tensor((3, 32), "torch.int32")
    with pytest.raises(ValueError, match="block table"):
        validate_paged_decode(p, ts)
    p, ts = inputs()
    with pytest.raises(ValueError, match="token-offset"):
        validate_paged_decode(replace(p, max_seqlen_k=2**31), ts)


class AddressInterpreter:
    """Evaluate the emitted scalar address operations against a finite table."""

    const_i32 = staticmethod(int)
    add = staticmethod(lambda x, y: x + y)
    mul = staticmethod(lambda x, y: x * y)
    div = staticmethod(lambda x, y: x // y)
    mod = staticmethod(lambda x, y: x % y)
    cmp_lt = staticmethod(lambda x, y: x < y)
    select = staticmethod(lambda pred, x, y: x if pred else y)

    @staticmethod
    def global_load(table, index, dtype, align):
        assert 0 <= index < len(table), "speculative page-table read escaped allocation"
        return table[index]


@pytest.mark.parametrize("page", [1, 16, 32, 64])
@pytest.mark.parametrize("hd", [64, 128, 256])
def test_logical_tile_address_mapping(page, hd):
    from kernels.common._attention_paged_kv import paged_kv_offset

    length, heads, tile_size = 65, 2, 32
    n = (length + page - 1) // page
    # Independent physical pages for two batches, including gaps in the pool.
    table = [3 * i + 7 for i in reversed(range(2 * n))]
    for seq in range(2):
        for head in range(heads):
            for token in range(160):
                for dim in (0, hd - 8):
                    tile, row = divmod(token, tile_size)
                    got = paged_kv_offset(
                        AddressInterpreter(),
                        table,
                        seq * n,
                        head,
                        tile,
                        row * hd + dim,
                        length,
                        hd,
                        page,
                        tile_size,
                        heads,
                    )
                    if token >= length:
                        assert got == 0x7FFFFFFF
                    else:
                        physical = table[seq * n + token // page]
                        expected = 2 * (
                            ((physical * page + token % page) * heads + head) * hd + dim
                        )
                        assert got == expected


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_runtime_rejects_invalid_gather_before_launch(arch, monkeypatch):
    from dispatch.attention import AttentionRequest, dispatch_attention
    from kernels.common import attention_unified as au

    p, ts = inputs()
    p = replace(p, clamp_arch=arch)
    req = AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=33,
        hdim_q=64,
        hdim_v=64,
        arch=arch,
        dtype="fp16",
        kv_block_size=1,
        target_ctas=8,
    )
    spec = dispatch_attention(req).spec
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: arch)

    def forbidden(**kwargs):
        pytest.fail("invalid gather reached launch")

    monkeypatch.setattr(au, "_run_3d_tiled", forbidden)
    kwargs = {
        "problem": p,
        "q": ts[0],
        "k": ts[1],
        "v": ts[2],
        "out": ts[3],
        "cu_seqlens_q": ts[4],
        "seqused_k": ts[5],
        "block_table": ts[6],
        "softmax_scale": 1 / 8,
        "softcap": 0,
        "backend": "3d",
        "tuning_spec": spec,
    }
    bad = replace(
        spec,
        allow_unsupported=True,
        kernel_spec=replace(spec.kernel_spec, head_size=128),
    )
    with pytest.raises(ValueError, match="head_size"):
        au.run_unified_attention_torch(**dict(kwargs, tuning_spec=bad))
    with pytest.raises(ValueError, match="architecture"):
        au.run_unified_attention_torch(
            **dict(kwargs, problem=replace(p, clamp_arch="gfx1250"))
        )
    from types import SimpleNamespace

    incomplete = asdict(spec.kernel_spec)
    del incomplete["kv_storage_dtype"]
    with pytest.raises(ValueError, match="kv_storage_dtype"):
        au.run_unified_attention_torch(
            **dict(
                kwargs,
                tuning_spec=replace(spec, kernel_spec=SimpleNamespace(**incomplete)),
            )
        )


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_gfx942_page_one_default_uses_async_gather(dim, monkeypatch):
    from kernels.common import attention_unified as au

    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx942")
    p, _ = inputs()
    p = replace(p, head_size=dim, clamp_arch="gfx942")
    spec = au._tiled_3d_spec_from_problem(p)
    assert spec.tile_size == 32 and spec.uses_paged_gather
    assert not spec.use_wide_kv_load


def test_gather_workspace_bound():
    p, ts = inputs()
    with pytest.raises(ValueError, match="workspace"):
        validate_paged_decode(replace(p, num_seqs=65536, total_q=65536), ts)
