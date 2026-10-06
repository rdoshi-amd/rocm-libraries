"""LLVM generic GPU targets as data: the shared table and the tier algebra over it.

One JSON document (projects/hipdnn/plugin_sdk/data/gpu_generic_targets.json) maps each
generic target (`gfx11-generic`) to the concrete processors it supports. The C++ loader
reads the same document through a generated header; this module is the Python side, and
the two are held equal by the golden vectors in plugin_sdk/tests/data/arch_tier_vectors.json.

Membership is data, never name shape: a generic-shaped name that the table does not list
(an "unknown generic") expands to the empty set and matches no device. Provenance of the
table is validated once, by the CMake validator, not here.

An `arch` list ranks per device: EXPLICIT (the device's own base id) beats GENERIC (a table
generic containing the device) beats UNRESTRICTED (an empty list, any device). Lists are
sequences of strings.
"""

import json
from pathlib import Path

from .errors import HkpPackError

TIER_EXPLICIT = 0
TIER_GENERIC = 1
TIER_UNRESTRICTED = 2

# .../dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/<this file>
DEFAULT_TABLE_PATH = (
    Path(__file__).resolve().parents[5]
    / "projects"
    / "hipdnn"
    / "plugin_sdk"
    / "data"
    / "gpu_generic_targets.json"
)

_GENERIC_SUFFIX = "-generic"


class GenericTargets:
    """The generic target table: generic name -> member processor names, document order."""

    def __init__(self, path, generics):
        self._path = Path(path)
        self._generics = generics

    @classmethod
    def load(cls, path):
        path = Path(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise HkpPackError(f"{path}: cannot read generic target table: {exc}")
        try:
            doc = json.loads(text)
        except ValueError as exc:
            raise HkpPackError(f"{path}: generic target table is not valid JSON: {exc}")
        if not isinstance(doc, dict):
            raise HkpPackError(f"{path}: generic target table must be a JSON object")
        version = doc.get("schemaVersion")
        if type(version) is not int or version != 1:
            raise HkpPackError(f"{path}: 'schemaVersion' must be 1, got {version!r}")
        raw = doc.get("generics")
        if not isinstance(raw, dict):
            raise HkpPackError(f"{path}: 'generics' must be an object")
        generics = {}
        for name, members in raw.items():
            if not is_generic_shaped(name):
                raise HkpPackError(
                    f"{path}: generics.{name}: a generic target name must end in "
                    f"'-generic'; a concrete processor name here would turn its own "
                    f"pass into a generic one"
                )
            if not isinstance(members, list) or not members:
                raise HkpPackError(
                    f"{path}: generics.{name} must be a non-empty array of names"
                )
            for member in members:
                if not isinstance(member, str) or not member:
                    raise HkpPackError(
                        f"{path}: generics.{name} holds a member that is not a "
                        f"non-empty string: {member!r}"
                    )
            if len(set(members)) != len(members):
                raise HkpPackError(
                    f"{path}: generics.{name} lists a member more than once"
                )
            generics[name] = tuple(members)
        return cls(path, generics)

    @property
    def path(self):
        return self._path

    def has(self, name):
        return name in self._generics

    def members(self, name):
        """Members of generic @name in document order; KeyError when it is not listed."""
        return self._generics[name]

    def names(self):
        return tuple(self._generics)


def is_generic_shaped(name):
    """Shape only (`...-generic`); use GenericTargets.has for membership."""
    return name.endswith(_GENERIC_SUFFIX)


def _strip_features(arch):
    return arch.split(":", 1)[0]


def _prefix_match(device, entry):
    """The packer-side twin of archMatches PREFIX: @entry is @device's base id."""
    return device == entry or device.startswith(entry + ":")


def entry_tier(entry, device, table):
    """Tier of one list entry for @device (features allowed), or None when it admits it.

    A generic-shaped entry is never EXPLICIT, so an unknown generic matches no device.
    """
    if is_generic_shaped(entry):
        if table.has(entry) and _strip_features(device) in table.members(entry):
            return TIER_GENERIC
        return None
    if _prefix_match(device, entry):
        return TIER_EXPLICIT
    return None


def list_tier(entries, device, table):
    """Best tier of @entries for @device; an empty list is UNRESTRICTED; None = no match."""
    if not entries:
        return TIER_UNRESTRICTED
    tiers = [
        t for t in (entry_tier(e, device, table) for e in entries) if t is not None
    ]
    return min(tiers) if tiers else None


def expand(entries, table):
    """The set of device ids @entries admits; None (unrestricted) for an empty list.

    A generic stands for its members; an unknown generic contributes nothing.
    """
    if not entries:
        return None
    devices = set()
    for entry in entries:
        if is_generic_shaped(entry):
            if table.has(entry):
                devices.update(table.members(entry))
        else:
            devices.add(entry)
    return frozenset(devices)


def overlaps(a, b, table):
    """Can one device satisfy both lists? Empty overlaps everything."""
    expanded_a = expand(a, table)
    expanded_b = expand(b, table)
    if expanded_a is None or expanded_b is None:
        return True
    return not expanded_a.isdisjoint(expanded_b)


def covers(outer, inner, table):
    """Is every device @inner admits also admitted by @outer?

    Empty @outer covers anything and empty @inner is covered by anything; an unknown
    generic in @inner expands to nothing and is covered vacuously.
    """
    expanded_outer = expand(outer, table)
    expanded_inner = expand(inner, table)
    if expanded_outer is None or expanded_inner is None:
        return True
    return expanded_inner <= expanded_outer


def compete(a, b, table):
    """Do the lists tie on some candidate device (both match it at the same tier)?

    Candidates are every explicit id in either list and every member of every table
    generic in either list. Two empty lists compete.
    """
    if not a and not b:
        return True
    candidates = set()
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


def admits_target(entries, target, table):
    """Does @entries admit @target: empty, listing it literally, or a generic containing it."""
    return list_tier(entries, target, table) is not None
