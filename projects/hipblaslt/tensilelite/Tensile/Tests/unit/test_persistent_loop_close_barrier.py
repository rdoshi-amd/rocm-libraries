################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""Every persistent-loop back-branch must be preceded by an s_barrier.

The next work item's prologue writes LDS (local writes or DirectToLds loads)
while slower waves of the workgroup may still be reading LDS in the previous
work item's epilogue. Only s_barrier orders LDS accesses across waves.
"""

import re
from contextlib import contextmanager

import pytest
from rocisa.code import Label
from rocisa.container import ContinuousRegister
from rocisa.instruction import SCBranchSCC0

from Tensile.Components.PersistentLoop import PersistentLoopOn

pytestmark = pytest.mark.unit

_ENTRY_LABEL = "PersistentLoopEntry"
_TARGETS = ("PersistentLoopStart", _ENTRY_LABEL)
_LABEL_RE = re.compile(r"^\s*[A-Za-z_][\w.]*:")


class _MockWriter:
    @contextmanager
    def allocTmpSgpr(self, num, alignment=None, tag=None):
        yield ContinuousRegister(idx=100, size=num)

    def longBranchScc0(self, label: Label, posNeg: int = 0, comment=""):
        return SCBranchSCC0(labelName=label.getLabelName(), comment=comment)

    def rapPersistentLoopEntryLabel(self, kernel):
        return _ENTRY_LABEL


def _close_loop_lines(streamK):
    module = PersistentLoopOn().closePersistentLoop(_MockWriter(), {"StreamK": streamK})
    return [line for line in str(module).splitlines() if line.strip()]


def _back_branch_indices(lines):
    """Index of the first line of each back-branch sequence."""
    indices = []
    for i, line in enumerate(lines):
        if _LABEL_RE.match(line):
            continue
        code = line.split("//")[0]
        if any(t in code for t in _TARGETS):
            prev = lines[i - 1].split("//")[0] if i else ""
            if not (indices and any(t in prev for t in _TARGETS)):
                indices.append(i)
    return indices


def _barrier_since_last_label(lines, idx):
    for line in reversed(lines[:idx]):
        if _LABEL_RE.match(line):
            return False
        if line.split("//")[0].strip().startswith("s_barrier"):
            return True
    return False


@pytest.mark.parametrize("streamK", [1, 2, 3, 4, 5])
def test_barrier_precedes_persistent_back_branch(streamK):
    lines = _close_loop_lines(streamK)
    branches = _back_branch_indices(lines)
    expected = 2 if streamK == 5 else 1
    assert len(branches) == expected, "\n".join(lines)
    for idx in branches:
        assert _barrier_since_last_label(lines, idx), (
            "missing s_barrier before persistent back-branch:\n" + "\n".join(lines))
