# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Codegen tests for the dynamic StreamK fixup by last arrival (SK5 Hybrid).

Every part of a split tile writes its partial and counts itself in on a
per-tile counter with ``s_atomic_inc`` bounded by ``SKSplit-1``; the part that
reads ``SKSplit-1`` (and so wraps the counter back to 0) fixes the tile up.
These tests pin the shape of that protocol on a fake writer: one atomic, its
bound, the LDS ticket broadcast that restores what it borrowed, and that no
per-part ready flag is stored or polled.
"""

import inspect
import itertools
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

# Prime the component registry before StreamK imports (avoids circular import).
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401

from rocisa.code import Label, Module
from rocisa.container import ContinuousRegister
from rocisa.instruction import (
    BufferLoadB32,
    BufferStoreB32,
    DSLoadB32,
    DSStoreB32,
    SAtomicInc,
    SBarrier,
    SSubU32,
)

from Tensile.Components.StreamK import StreamK, StreamKHybrid

pytestmark = pytest.mark.unit


class _Pool:
    def __init__(self, start=100):
        self._next = start

    def checkOut(self, n, *args, **kwargs):
        reg = self._next
        self._next += n
        return reg

    def checkOutAligned(self, n, align, *args, **kwargs):
        self._next += (-self._next) % align
        return self.checkOut(n)

    def checkIn(self, *args, **kwargs):
        return None


_KERNEL = {
    "TileProcessingStrategy": "StreamK", "WorkAssignment": "Hybrid",
    "DebugStreamK": 0, "WavefrontSize": 64, "ISA": (9, 5, 0),
    "ProblemType": {"NumIndicesC": 3, "NumIndicesFree": 2},
}


def _writer(hasSAtomic=True, pap=False):
    sgprs = _Pool()

    @contextmanager
    def alloc_tmp(size, alignment=1, tag=""):
        yield ContinuousRegister(sgprs.checkOut(size), size)

    w = SimpleNamespace(
        sgprPool=sgprs,
        vgprPool=_Pool(),
        labels=SimpleNamespace(getNameInc=lambda n, c=itertools.count(): "%s_%d" % (n, next(c))),
        allocTmpSgpr=alloc_tmp,
        isPrefetchAcrossPersistentEnabled=lambda k: pap,
        longBranchScc0=lambda label, posNeg=0, comment="": Module("longBranchScc0 " + label.getLabelName()),
        states=SimpleNamespace(
            kernel=_KERNEL,
            asmCaps={"HasSAtomic": hasSAtomic},
            archCaps={"NumXCD": 8, "CacheLineBytes": 128, "HasXCDSplitL2": True,
                      "HasInvWbDevFences": False, "RequiresXCntForVolatileVMEM": False,
                      "EnableXnackReplay": False},
            version=(9, 5, 0),
        ),
    )
    w.states.skArrivalFixupLabel = Label("SK_ArrivalFixup_test", "")
    return w


def _arrival():
    w = _writer()
    module = StreamKHybrid().emitArrival(w, _KERNEL, Label("SK_ArrivalDone_test", ""))
    return list(module.flatitems())


def test_one_atomic_bounded_by_split_minus_one():
    items = _arrival()
    atomics = [i for i in items if isinstance(i, SAtomicInc)]
    assert len(atomics) == 1, "exactly one arrival atomic per part"
    pos = items.index(atomics[0])
    bound = [i for i in items[:pos] if isinstance(i, SSubU32)
             and "sgprSKSplit" in str(i.getParams()[1])]
    assert bound, "the atomic's wrap bound must be SKSplit-1 so the last arrival resets it"
    assert "glc" in str(atomics[0]), "the ticket must come back from the atomic"


def test_no_ready_flag_traffic():
    items = _arrival()
    assert not any(isinstance(i, (BufferStoreB32, BufferLoadB32)) for i in items), (
        "the arrival protocol stores and polls no per-part ready flag")


def test_lds_mailbox_is_saved_and_restored():
    items = _arrival()
    loads = [i for i in items if isinstance(i, DSLoadB32)]
    stores = [i for i in items if isinstance(i, DSStoreB32)]
    # save + read ticket; publish ticket + restore
    assert len(loads) == 2 and len(stores) == 2
    assert items.index(loads[0]) < items.index(stores[0]), "save LDS before publishing"
    assert items.index(stores[1]) > items.index(loads[1]), "restore after every wave read it"
    # LDS idle -> ticket published -> ticket read -> mailbox restored
    assert sum(isinstance(i, SBarrier) for i in items) == 4


@pytest.mark.parametrize("hasSAtomic,pap,debug,expected", [
    (True, False, 0, True),
    (False, False, 0, False),
    (True, True, 0, False),
    (True, False, 1, False),
])
def test_uses_arrival_fixup_gating(hasSAtomic, pap, debug, expected):
    kernel = dict(_KERNEL, DebugStreamK=debug)
    assert StreamKHybrid().usesArrivalFixup(_writer(hasSAtomic, pap), kernel) is expected


def test_base_strategy_keeps_flags():
    assert StreamK.usesArrivalFixup(StreamKHybrid(), _writer(), _KERNEL) is False


def test_dynamic_store_has_no_flag_spin_when_arriving():
    # The arrival store body must not poll a ready flag (readFlag) in the loop.
    src = inspect.getsource(StreamKHybrid.storeBranches)
    body = src.split("def emitArrivalDynamicStore", 1)[1].split("def emitDynamicStore", 1)[0]
    assert "readFlag" not in body and "emitFlagStore" not in body
    assert "skArrivalFixupLabel" in body
