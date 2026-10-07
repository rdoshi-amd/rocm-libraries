# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Value barriers preserve types and survive serialization through both engines."""

import pytest

from rocke.core.ir import (
    BF16,
    BF8E5M2,
    F16,
    F32,
    FP8E4M3,
    I1,
    I8,
    I16,
    I32,
    I64,
    IRBuilder,
    PtrType,
    Type,
    VectorType,
)
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm


_MALFORMED_ASM_ATTRIBUTES = [
    (attribute, value)
    for attribute in ("sideeffect", "convergent")
    for value in ("true", 0, 1, 0.0, 1.0)
] + [("clobber", value) for value in (0, 1, 0.0, 1.0, False)]


@pytest.mark.parametrize("attribute,value", _MALFORMED_ASM_ATTRIBUTES)
def test_malformed_asm_attribute_types(attribute, value):
    b = IRBuilder("barrier_attributes")
    result = b.optimization_barrier(b.param("value", F32))
    result.op.attrs[attribute] = value
    b.ret()
    for kernel in (b.kernel, parse(serialize(b.kernel))):
        with pytest.raises(ValueError, match=f"inline asm attribute '{attribute}'"):
            lower_kernel_to_hip(kernel, arch="gfx950")


@pytest.mark.parametrize("attribute,value", _MALFORMED_ASM_ATTRIBUTES)
def test_native_malformed_asm_attribute_types(native_barrier_hip, attribute, value):
    import subprocess

    b = IRBuilder("barrier_attributes")
    result = b.optimization_barrier(b.param("value", F32))
    result.op.attrs[attribute] = value
    b.ret()
    for kernel in (b.kernel, parse(serialize(b.kernel))):
        with pytest.raises(subprocess.CalledProcessError) as failure:
            native_barrier_hip(kernel, "gfx950")
        assert "native HIP lowering failed: status=1" in failure.value.stderr
        assert not failure.value.stdout


@pytest.mark.parametrize("attributes", [{}, {"convergent": False, "clobber": ""}])
def test_optional_asm_defaults(native_barrier_hip, attributes):
    b = IRBuilder("barrier_defaults")
    result = b.optimization_barrier(b.param("value", F32))
    result.op.attrs.pop("convergent", None)
    result.op.attrs.pop("clobber", None)
    result.op.attrs.update(attributes)
    b.ret()
    for kernel in (b.kernel, parse(serialize(b.kernel))):
        hip = lower_kernel_to_hip(kernel, arch="gfx950")
        assert native_barrier_hip(kernel, "gfx950") == hip
        assert 'asm ("" : "=v"' in hip
        assert "asm volatile" not in hip
        assert '"memory"' not in hip


def test_python_general_asm_defaults(native_barrier_hip):
    import subprocess

    b = IRBuilder("asm_defaults")
    result = b.optimization_barrier(b.param("value", F32))
    result.op.attrs.pop("sideeffect")
    b.ret()
    for kernel in (b.kernel, parse(serialize(b.kernel))):
        hip = lower_kernel_to_hip(kernel, arch="gfx950")
        assert 'asm volatile ("" : "=v"' in hip
        assert ': "memory"' in hip
        with pytest.raises(subprocess.CalledProcessError) as failure:
            native_barrier_hip(kernel, "gfx950")
        assert "native HIP lowering failed: status=5" in failure.value.stderr


@pytest.mark.parametrize(
    "dtype", [I1, I8, I16, I32, I64, BF16, F16, F32, FP8E4M3, BF8E5M2]
)
def test_scalar_barrier_roundtrip(dtype):
    b = IRBuilder("barrier")
    ptr = b.param("p", PtrType(dtype, "global"))
    tid = b.thread_id_x()
    value = b.global_load(ptr, tid, dtype)
    result = b.optimization_barrier(value)
    assert result.type == value.type
    b.global_store(ptr, tid, result)
    b.ret()
    copy = parse(serialize(b.kernel))
    for arch in ("gfx950", "gfx1250"):
        llvm = lower_kernel_to_llvm(b.kernel, arch=arch, llvm_flavor="llvm23")
        assert llvm == lower_kernel_to_llvm(copy, arch=arch, llvm_flavor="llvm23")
        assert 'asm "", "=v,0"' in llvm
        assert "asm sideeffect" not in llvm
        hip = lower_kernel_to_hip(copy, arch=arch)
        assert 'asm ("" : "=v"' in hip
        assert '"memory"' not in hip


@pytest.mark.parametrize(
    "dtype", [PtrType(F32, "global"), VectorType(F32, 2), Type("unknown")]
)
def test_rejects_non_numeric_scalar(dtype):
    b = IRBuilder("invalid_barrier")
    value = b.param("value", dtype)
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match="directly lowerable scalar"):
        b.optimization_barrier(value)
    assert serialize(b.kernel) == before


@pytest.mark.parametrize("dtype", ["fp4", "fp6", "bf6", "e8m0", "e5m3", "tf32"])
def test_logical_types_use_storage_barriers(dtype):
    from rocke.core.ir import dtype_to_ir_type
    from rocke.helpers.mma_io import storage_ir_type

    logical = dtype_to_ir_type(dtype)
    b = IRBuilder("logical_barrier")
    value = b.param("value", logical)
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match="directly lowerable scalar"):
        b.optimization_barrier(value)
    assert serialize(b.kernel) == before
    storage = storage_ir_type(dtype)
    assert storage == (I32 if dtype == "tf32" else I8)
    assert b.optimization_barrier(b.param("bits", storage)).type == storage


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
@pytest.mark.parametrize(
    "dtype", [I1, I8, I16, I32, I64, BF16, F16, F32, FP8E4M3, BF8E5M2]
)
def test_native_hip_matches_python(native_barrier_hip, arch, dtype):
    b = IRBuilder("barrier_native")
    ptr = b.param("p", PtrType(dtype, "global"))
    tid = b.thread_id_x()
    value = b.global_load(ptr, tid, dtype)
    b.global_store(ptr, tid, b.optimization_barrier(value))
    b.ret()
    original = serialize(b.kernel)
    expected = lower_kernel_to_hip(parse(original), arch=arch)
    assert native_barrier_hip(b.kernel, arch) == expected
    assert native_barrier_hip(parse(original), arch) == expected
    assert 'asm ("" : "=v"' in expected
    assert "asm volatile" not in expected
    assert '"memory"' not in expected


@pytest.mark.parametrize("arch", ["gfx950", "gfx1250"])
def test_native_predicate_producer(native_barrier_hip, arch):
    b = IRBuilder("barrier_predicate")
    ptr = b.param("p", PtrType(I32, "global"))
    tid = b.thread_id_x()
    value = b.global_load(ptr, tid, I32)
    predicate = b.cmp_lt(value, b.const_i32(0))
    result = b.optimization_barrier(predicate)
    b.global_store(ptr, tid, b.zext(result, I32))
    b.ret()
    assert native_barrier_hip(b.kernel, arch) == lower_kernel_to_hip(
        b.kernel, arch=arch
    )


@pytest.mark.parametrize("failure", ["missing", "metadata", "command"])
def test_native_fixture_errors_are_not_skips(monkeypatch, tmp_path, failure):
    import conftest
    import json
    import subprocess

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROCKE_OPTIMIZATION_BARRIER_TEST", raising=False)
    if failure == "missing":
        monkeypatch.setenv("ROCKE_OPTIMIZATION_BARRIER_TEST", "missing")
        with pytest.raises(ValueError, match="does not exist"):
            conftest._optimization_barrier_executable()
    else:
        (tmp_path / "CTestTestfile.cmake").touch()

        def run(command, **kwargs):
            if failure == "metadata":
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps({"tests": [{"command": []}]})
            )

        monkeypatch.setattr(conftest.subprocess, "run", run)
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            conftest._optimization_barrier_executable()


def test_native_fixture_discovers_installed_executable(monkeypatch, tmp_path):
    import conftest
    import json
    import subprocess

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ROCKE_OPTIMIZATION_BARRIER_TEST", raising=False)
    (tmp_path / "CTestTestfile.cmake").touch()
    executable = tmp_path / "provider_rocke_optimization_barrier_test.exe"
    executable.touch()
    listing = {"tests": [{"command": [str(executable)]}]}
    monkeypatch.setattr(
        conftest.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(listing)
        ),
    )
    assert conftest._optimization_barrier_executable() == executable


def test_native_fixture_preserves_lowering_failure(monkeypatch, tmp_path):
    import conftest
    import subprocess

    executable = tmp_path / "broken_native_test"
    executable.touch()
    monkeypatch.setenv("ROCKE_OPTIMIZATION_BARRIER_TEST", str(executable))

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="lowering failed")

    monkeypatch.setattr(conftest.subprocess, "run", fail)
    lower = conftest.native_barrier_hip.__wrapped__()
    b = IRBuilder("broken_native")
    b.ret()
    with pytest.raises(subprocess.CalledProcessError):
        lower(b.kernel, "gfx950")
