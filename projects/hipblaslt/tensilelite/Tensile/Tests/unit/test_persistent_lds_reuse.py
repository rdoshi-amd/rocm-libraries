# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Persistent tile re-entry must wait for every wave's epilogue LDS reads."""

from unittest.mock import MagicMock

import pytest

# Prime the component registry before importing a component.
from Tensile.KernelWriterAssembly import KernelWriterAssembly  # noqa: F401
from Tensile.Components.PersistentLoop import PersistentLoopOn
from rocisa.code import Label
from rocisa.container import ContinuousRegister
from rocisa.instruction import SCBranchSCC0, SBarrier, SWaitCnt


@pytest.mark.parametrize("streamk", [3, 4, 5])
@pytest.mark.parametrize("loop_forever", [False, True])
def test_every_persistent_reentry_path_retires_and_joins_lds_reads(streamk, loop_forever):
    writer = MagicMock()
    writer.allocTmpSgpr.return_value.__enter__.return_value = ContinuousRegister(74, 3)
    writer.longBranchScc0.side_effect = lambda label, **kwargs: SCBranchSCC0(
        labelName=label.getLabelName()
    )
    writer.rapPersistentLoopEntryLabel.return_value = "PersistentLoopStart"
    module = PersistentLoopOn().closePersistentLoop(
        writer, {"StreamK": streamk, "DebugPersistentKernelLoopForever": loop_forever}
    )
    items = module.flatitems()
    # The close-loop label is itself a branch target: the join must follow it,
    # before any mode dispatch, exit test, or branch to the next tile.
    assert isinstance(items[0], Label)
    assert isinstance(items[1], SWaitCnt)
    assert items[1].dscnt == 0
    assert isinstance(items[2], SBarrier)
    assert sum(isinstance(item, SBarrier) for item in items) == 1
