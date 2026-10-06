# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Actual SSA accumulator checks shared by LLVM and HIP WMMA emitters."""

from __future__ import annotations

from .ir import Op, Type, VectorType


def validate_unscaled_wmma(op: Op, elem: Type, count: int = 8) -> None:
    """Reject values that cannot use the selected instruction's fixed ABI.

    Generic MMA metadata allows independent src2/dst contracts. Only the
    concrete emitter can impose the physical accumulator/result signature.
    Mirrors ``_validate_unscaled_wmma`` in the native LLVM lowerer.
    """
    if len(op.operands) != 3 or len(op.results) != 1:
        raise ValueError("unscaled WMMA expects 3 operands and 1 result")
    expected = VectorType(elem, count)
    if op.operands[2].type != expected or op.result.type != expected:
        raise ValueError(f"unscaled WMMA requires src2 and dst to be {expected.name}")
