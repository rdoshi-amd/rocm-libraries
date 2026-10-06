# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Focused CPU/spec/IR coverage for gfx942 dense bottom-right causal attention.

The gfx950 twin is ``test_attention_dense_bottom_right.py``. The identity contract
differs on purpose: gfx950 bakes the diagonal offset, so every bottom-right shape is
its own binary. gfx942 reads it from the runtime shape params on the default grid, so
one binary serves every bottom-right shape there; only the persistent grid bakes it.
"""

from __future__ import annotations

import pytest

from kernels.common.attention_dense_spec import (
    AttentionDenseSpec,
    attention_dense_cache_key,
)
from kernels.gfx942.attention_dense import (
    Gfx942AttentionDenseSpec,
    build_attention_dense,
    supports_attention_dense,
)

_ARCH = "gfx942"


def _spec(**over) -> Gfx942AttentionDenseSpec:
    kw = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=1024,
        num_query_heads=4,
        num_kv_heads=1,
        head_size=128,
        causal=True,
        dtype="bf16",
        block_n=64,
    )
    kw.update(over)
    return Gfx942AttentionDenseSpec(**kw)


def _lowered_body(spec: Gfx942AttentionDenseSpec) -> str:
    """Lower without a toolchain and remove the identity-only symbol difference."""
    from rocke.helpers.compile import _lower_llvm_via_backend

    kernel = build_attention_dense(spec, arch=_ARCH)
    llvm = _lower_llvm_via_backend(kernel, arch=_ARCH, backend="python", spec=None)
    return llvm.replace(kernel.name, "KERNEL")


@pytest.mark.parametrize(
    "over,match",
    [
        ({"causal": False}, "requires causal=True"),
        ({"seqlen_q": 1024, "seqlen_kv": 512}, "seqlen_q <= seqlen_kv"),
        ({"sliding_window": 128}, "sliding_window"),
    ],
)
def test_bottom_right_invalid_combinations_raise(over, match):
    with pytest.raises(ValueError, match=match):
        _spec(causal_bottom_right=True, **over)


@pytest.mark.parametrize("persistent", [False, True], ids=["default", "persistent"])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("head_size", [64, 128])
def test_bottom_right_is_supported_and_builds(head_size, dtype, persistent):
    spec = _spec(
        head_size=head_size,
        dtype=dtype,
        persistent=persistent,
        causal_bottom_right=True,
    )
    ok, why = supports_attention_dense(spec, arch=_ARCH)
    assert ok, why
    assert build_attention_dense(spec, arch=_ARCH).name == spec.kernel_name()


def test_shared_spec_with_bottom_right_and_window_is_a_structured_rejection():
    """The promotion to the gfx942 spec would raise; supports must return a reason."""
    spec = AttentionDenseSpec(
        batch=1,
        seqlen_q=512,
        seqlen_kv=1024,
        num_query_heads=4,
        num_kv_heads=1,
        head_size=128,
        causal=True,
        causal_bottom_right=True,
        sliding_window=128,
    )
    ok, why = supports_attention_dense(spec, arch=_ARCH)
    assert not ok
    assert "sliding_window" in why


@pytest.mark.parametrize("persistent", [False, True], ids=["default", "persistent"])
def test_moving_bottom_right_changes_the_lowered_body(persistent):
    top_left = _spec(persistent=persistent)
    bottom_right = _spec(persistent=persistent, causal_bottom_right=True)
    assert _lowered_body(top_left) != _lowered_body(bottom_right)


@pytest.mark.parametrize("persistent", [False, True], ids=["default", "persistent"])
def test_bottom_right_has_distinct_symbol_and_cache_identity(persistent):
    top_left = _spec(persistent=persistent)
    bottom_right = _spec(persistent=persistent, causal_bottom_right=True)

    assert "br" not in top_left.kernel_name().split("_")
    assert "br" in bottom_right.kernel_name().split("_")
    assert attention_dense_cache_key(top_left, arch=_ARCH) != attention_dense_cache_key(
        bottom_right, arch=_ARCH
    )


def test_default_grid_bottom_right_shapes_share_one_compiled_identity():
    """The offset is derived from the seqlen params, so the body, the symbol, and
    the cache key are the same for every bottom-right shape on the default grid."""
    shapes = ((512, 1024), (256, 1024), (512, 512), (1024, 4096))
    specs = [
        _spec(seqlen_q=sq, seqlen_kv=skv, causal_bottom_right=True)
        for sq, skv in shapes
    ]
    assert all(s.runtime_shape for s in specs)
    assert len({s.kernel_name() for s in specs}) == 1
    assert len({attention_dense_cache_key(s, arch=_ARCH) for s in specs}) == 1
    assert len({_lowered_body(s) for s in specs}) == 1


def test_persistent_bottom_right_offsets_do_not_share_compiled_identity():
    first = _spec(
        seqlen_q=256, seqlen_kv=512, persistent=True, causal_bottom_right=True
    )
    second = _spec(
        seqlen_q=512, seqlen_kv=1024, persistent=True, causal_bottom_right=True
    )

    # The persistent body bakes the diagonal, so reusing either binary is incorrect.
    assert not first.runtime_shape and not second.runtime_shape
    assert _lowered_body(first) != _lowered_body(second)
    assert attention_dense_cache_key(first, arch=_ARCH) != attention_dense_cache_key(
        second, arch=_ARCH
    )
    assert first.kernel_name() != second.kernel_name()


def test_persistent_equal_length_bottom_right_emits_no_add_zero():
    """A baked zero offset is skipped, so the body is exactly the top-left one."""
    top_left = _spec(seqlen_q=512, seqlen_kv=512, persistent=True)
    bottom_right = _spec(
        seqlen_q=512, seqlen_kv=512, persistent=True, causal_bottom_right=True
    )
    assert "br" in bottom_right.kernel_name().split("_")
    assert _lowered_body(top_left) == _lowered_body(bottom_right)


def test_equal_length_dispatch_normalizes_to_top_left_body():
    from dispatch.attention import AttentionRequest
    from dispatch.attention.gfx942 import dense_spec_for_request

    def dispatched(mask_type):
        return dense_spec_for_request(
            AttentionRequest(
                batch=1,
                nhead_q=4,
                nhead_k=1,
                seqlen_q=512,
                seqlen_k=512,
                hdim_q=128,
                hdim_v=128,
                arch=_ARCH,
                dtype="bf16",
                mask_type=mask_type,
                algorithm="attention_dense",
            )
        )

    top_left = dispatched(1)
    bottom_right = dispatched(2)
    assert bottom_right == top_left
    assert not bottom_right.causal_bottom_right
