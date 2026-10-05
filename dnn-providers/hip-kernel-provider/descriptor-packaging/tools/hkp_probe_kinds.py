"""The producer kinds a packaging probe supports: one entry per kernel_source.kind.

Shared by hkp_probe_derive_root.py (compile-group selection) and
hkp_probe_assert.py (provenance rules). Only kinds the packer packs to kpack output
belong here. A kind with no entry is rejected by derive at configure time and fails
`no-rules-for-kind` in the assertion, so a new producer cannot pass a probe without
being described here. A new kind that packs to kpack is one entry in KINDS.

Pass-through kinds (embedded_source: the packer ships it as authored, not as kpack) and
kinds the packer does not pack to kpack cannot be probed by adding an entry: the
assertion expects a kpack archive entry per UKD. A UKD of such a kind in a KDP that
ships for a probed arch makes configure fail; it must live in a KDP that does not ship
for a probed arch, or the probe needs a design decision.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Kind:
    # kernel_source fields that define a compile group within a KDP: UKDs of one kind
    # agreeing on all of them share a compile path, so one stands for the group. Empty
    # means the kind compiles nothing and one UKD per KDP stands for it.
    group_by: tuple[str, ...]
    # Names of provenance checks (hkp_probe_assert._PROVENANCE_CHECKS) a shipped UKD of
    # this kind must satisfy, beyond provenance.origin_kind == the kind itself.
    provenance: tuple[str, ...]


KINDS = {
    "rocke": Kind(group_by=("builder",), provenance=("wheel", "comgr")),
    "hip": Kind(group_by=("source", "build"), provenance=()),
    "hsaco": Kind(group_by=(), provenance=()),
}
