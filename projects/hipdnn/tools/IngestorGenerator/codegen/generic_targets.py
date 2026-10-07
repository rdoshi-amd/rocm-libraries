# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LLVM generic GPU targets as data, and the tier algebra over them, for ``codegen/``.

An in-repo mirror of ``hkp_pack.generic_targets`` and of the C++ loader's
``archTier``: the generator may not import the kernel provider's packaging tree
(descriptor generation must not need the kernel toolchain), so each implementation
tests itself with literal cases (``tests/test_generic_targets.py``).
Packer-side verifier tools import ``hkp_pack.generic_targets`` instead; this module is
used only inside ``codegen/``.

Membership is data, never name shape: a generic-shaped name the table does not list
expands to the empty set and matches no device. An ``arch`` list ranks per device:
EXPLICIT (the device's own base id) beats GENERIC (a table generic containing the
device) beats UNRESTRICTED (an empty list).
"""

import json
from functools import cache
from pathlib import Path

TIER_EXPLICIT = 0
TIER_GENERIC = 1
TIER_UNRESTRICTED = 2

# .../projects/hipdnn/tools/IngestorGenerator/codegen/<this file>
DEFAULT_TABLE_PATH = (
    Path(__file__).resolve().parents[3]
    / "plugin_sdk"
    / "data"
    / "gpu_generic_targets.json"
)

_GENERIC_SUFFIX = "-generic"


class GenericTargets:
    """The generic target table: generic name -> member processor names, document order."""

    def __init__(self, path: Path, generics: dict):
        self._path = Path(path)
        self._generics = generics

    @classmethod
    def load(cls, path) -> "GenericTargets":
        """Read the table at @path; ``ValueError`` names the path on any defect."""
        path = Path(path)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(
                f"{path}: cannot read generic target table: {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"{path}: generic target table is not valid JSON: {exc}"
            ) from exc
        raw = doc.get("generics") if isinstance(doc, dict) else None
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: 'generics' must be an object")
        generics = {}
        for name, members in raw.items():
            if (
                not isinstance(members, list)
                or not members
                or not all(isinstance(m, str) and m for m in members)
            ):
                raise ValueError(
                    f"{path}: generics.{name} must be a non-empty array of names"
                )
            generics[name] = tuple(members)
        return cls(path, generics)

    @property
    def path(self) -> Path:
        return self._path

    def has(self, name: str) -> bool:
        return name in self._generics

    def members(self, name: str) -> tuple:
        """Members of generic @name in document order; KeyError when it is not listed."""
        return self._generics[name]

    def names(self) -> tuple:
        return tuple(self._generics)


@cache
def default_table() -> GenericTargets:
    """The table at ``DEFAULT_TABLE_PATH``, read once per process."""
    return GenericTargets.load(DEFAULT_TABLE_PATH)


def is_generic_shaped(name: str) -> bool:
    """Shape only (``...-generic``); use ``GenericTargets.has`` for membership."""
    return name.endswith(_GENERIC_SUFFIX)


def _strip_features(arch: str) -> str:
    return arch.split(":", 1)[0]


def entry_tier(entry: str, device: str, table: GenericTargets):
    """Tier of one list entry for @device (features allowed), or None when it admits it.

    A generic-shaped entry is never EXPLICIT, so an unknown generic matches no device.
    """
    if is_generic_shaped(entry):
        if table.has(entry) and _strip_features(device) in table.members(entry):
            return TIER_GENERIC
        return None
    if device == entry or device.startswith(entry + ":"):
        return TIER_EXPLICIT
    return None


def list_tier(entries, device: str, table: GenericTargets):
    """Best tier of @entries for @device; empty is UNRESTRICTED; None = no match."""
    if not entries:
        return TIER_UNRESTRICTED
    tiers = [
        t for t in (entry_tier(e, device, table) for e in entries) if t is not None
    ]
    return min(tiers) if tiers else None


def expand(entries, table: GenericTargets):
    """The set of device ids @entries admits; None (unrestricted) for an empty list."""
    if not entries:
        return None
    devices: set = set()
    for entry in entries:
        if is_generic_shaped(entry):
            if table.has(entry):
                devices.update(table.members(entry))
        else:
            devices.add(entry)
    return frozenset(devices)


def covers(outer, inner, table: GenericTargets) -> bool:
    """Is every device @inner admits also admitted by @outer?"""
    expanded_outer, expanded_inner = expand(outer, table), expand(inner, table)
    if expanded_outer is None or expanded_inner is None:
        return True
    return expanded_inner <= expanded_outer


def compete(a, b, table: GenericTargets) -> bool:
    """Do the lists tie on some candidate device (both match it at the same tier)?

    Candidates are every explicit id in either list and every member of every table
    generic in either list. Two empty lists compete.
    """
    if not a and not b:
        return True
    candidates: set = set()
    for entry in (*a, *b):
        if is_generic_shaped(entry):
            if table.has(entry):
                candidates.update(table.members(entry))
        else:
            candidates.add(entry)
    for device in candidates:
        tier = list_tier(a, device, table)
        if tier is not None and tier == list_tier(b, device, table):
            return True
    return False
