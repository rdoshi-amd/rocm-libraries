# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host checks for split-KV scratch ownership, independent of GPU scheduling."""

import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
from kernels.common import attention_unified as au


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(stream=17, capturing=False, allocations=[])

    @contextmanager
    def stream_context(stream):
        previous = state.stream
        state.stream = stream.cuda_stream
        try:
            yield
        finally:
            state.stream = previous

    def empty(shape, *, dtype, device):
        tensor = SimpleNamespace(shape=shape, device=device, stream=state.stream)
        state.allocations.append(tensor)
        return tensor

    torch = SimpleNamespace(
        float32="float32",
        empty=empty,
        cuda=SimpleNamespace(
            device=lambda device: nullcontext(),
            stream=stream_context,
            ExternalStream=lambda stream, device: SimpleNamespace(cuda_stream=stream),
            current_stream=lambda device=None: SimpleNamespace(
                cuda_stream=state.stream
            ),
            is_current_stream_capturing=lambda: state.capturing,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx942")
    monkeypatch.setattr(au, "_enable_3d_graph_replay", lambda problem: False)
    # Emulate the old prepared workspace cache as well: the regression must
    # catch callers consulting it, rather than merely checking a new helper.
    shared = tuple(empty((1,), dtype="float32", device="cuda:0") for _ in range(3))
    prepared = SimpleNamespace(workspace=lambda *args: shared)
    monkeypatch.setattr(au, "_get_3d_pipeline", lambda *args, **kwargs: prepared)
    return state


def _arguments():
    problem = au.UnifiedAttentionProblem(
        total_q=1,
        num_seqs=1,
        num_query_heads=4,
        num_kv_heads=1,
        head_size=64,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=65,
        dtype="fp16",
        clamp_arch="gfx942",
        target_ctas=8,
    )
    return {
        "problem": problem,
        "q": SimpleNamespace(device="cuda:0"),
        "k": object(),
        "v": object(),
        "out": object(),
        "cu_seqlens_q": object(),
        "seqused_k": object(),
        "softmax_scale": 0.125,
        "block_table": object(),
        "softcap": 0.0,
        "sinks": None,
        "bt_stride": 5,
        "warmup": 0,
        "attempts": 1,
    }


@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("stream", [0, 29])
def test_workspace_uses_effective_stream(runtime, monkeypatch, stream, capturing):
    runtime.capturing = capturing
    calls = []
    monkeypatch.setattr(
        au, "_launch_3d_pipeline", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    au._run_3d_tiled(**_arguments(), stream=stream)
    args, kwargs = calls[0]
    expected_stream = stream or 17
    assert args[3] == expected_stream
    assert kwargs["capturing"] is capturing
    for name in ("segm_output_ptr", "segm_max_ptr", "segm_expsum_ptr"):
        assert args[1][name].stream == expected_stream
        assert args[1][name] is args[2][name]
    assert runtime.stream == 17


@pytest.mark.parametrize("second_stream", [17, 29])
def test_interleaved_invocations_do_not_alias(runtime, monkeypatch, second_stream):
    arguments = _arguments()
    first_output = arguments["out"]
    second_output = object()
    calls = []

    def launch(prepared, segment, reduce, stream, *, capturing):
        calls.append((segment, reduce))
        # Suspend A between segment and reduce while B uses the same shape
        # and input objects. Shared scratch or mutable bindings corrupt A.
        segment["segm_output_ptr"].partial = reduce["output_ptr"]
        if len(calls) == 1:
            au._run_3d_tiled(**dict(arguments, out=second_output), stream=second_stream)
        assert reduce["output_ptr"] is segment["segm_output_ptr"].partial

    monkeypatch.setattr(au, "_launch_3d_pipeline", launch)
    au._run_3d_tiled(**arguments, stream=17)
    assert calls[0][1]["output_ptr"] is first_output
    assert calls[1][1]["output_ptr"] is second_output
    for name in ("segm_output_ptr", "segm_max_ptr", "segm_expsum_ptr"):
        assert calls[0][0][name] is not calls[1][0][name]


@pytest.mark.parametrize("mode", ["fenced", "unfenced", "capture"])
def test_pipeline_drains_only_after_completion(monkeypatch, mode):
    from rocke.runtime.launcher import LaunchConfig, _resolved_fence, no_fence

    operations = []

    def pipeline(values, configs, *, stream):
        operations.append(("launch", stream, _resolved_fence(configs[-1].fence)))
        return "completed"

    prepared = SimpleNamespace(
        pipeline=pipeline,
        seg_config=LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)),
        red_config=LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)),
    )
    monkeypatch.setattr(
        au,
        "release_retained_for_stream",
        lambda stream: operations.append(("release", stream)),
    )
    with no_fence() if mode == "unfenced" else nullcontext():
        result = au._launch_3d_pipeline(
            prepared, {}, {}, 29, capturing=mode == "capture"
        )
    assert result == "completed"
    assert operations == (
        [("launch", 29, True), ("release", 29)]
        if mode == "fenced"
        else [("launch", 29, False)]
    )
