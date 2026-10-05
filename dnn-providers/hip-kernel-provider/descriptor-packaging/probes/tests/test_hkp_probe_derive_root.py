"""Tests for hkp_probe_derive_root.py (pure stdlib; no compile, no kpack)."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import MAIN_FIXTURE, PROBE_DERIVE


def _run(src: Path, arch: str, out: Path):
    return subprocess.run(
        [
            sys.executable,
            str(PROBE_DERIVE),
            "--from",
            str(src),
            "--arch",
            arch,
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _derive(src: Path, arch: str, out: Path):
    r = _run(src, arch, out)
    assert r.returncode == 0, r.stderr
    return json.loads((out.parent / "expect.json").read_text())


def _rocke(name, builder="b1", arch=None, **extra):
    ukd = {
        "id": f"id-{name}",
        "name": name,
        "kernel_source": {
            "kind": "rocke",
            "source": "s.py",
            "builder": builder,
            "spec": {"n": len(name)},
        },
    }
    if arch is not None:
        ukd["arch"] = arch
    return ukd


def _hip(name, source="a.cpp", build=None, arch=None):
    ukd = {
        "id": f"id-{name}",
        "name": name,
        "kernel_source": {
            "kind": "hip",
            "source": source,
            "entry": name,
            "build": build if build is not None else {},
        },
    }
    if arch is not None:
        ukd["arch"] = arch
    return ukd


def _other(kind, name):
    return {"id": f"id-{name}", "name": name, "kernel_source": {"kind": kind}}


def _write_root(root: Path, ukds, kdp_arch=("gfx950",), kdp_file="x.kdp.json"):
    root.mkdir(parents=True, exist_ok=True)
    kdp = {"version": "1.0", "id": "kdp-id", "kernelDescriptors": list(ukds)}
    if kdp_arch is not None:
        kdp["arch"] = list(kdp_arch)
    (root / kdp_file).parent.mkdir(parents=True, exist_ok=True)
    (root / kdp_file).write_text(json.dumps(kdp, indent=2) + "\n")
    (root / "x.kmd.json").write_text('{"id": "kmd"}\n')


def _kept(out: Path, kdp_file="x.kdp.json"):
    kdp = json.loads((out / kdp_file).read_text())
    return [d["name"] for d in kdp["kernelDescriptors"]]


# --- compile-group selection ------------------------------------------------
def test_one_ukd_per_compile_group(tmp_path):
    ukds = [
        _rocke("r_b", "b1"),
        _rocke("r_a", "b1"),
        _rocke("r_c", "b2"),
        _hip("h_b", "a.cpp", {"defines": {"X": 1}}),
        _hip("h_a", "a.cpp", {"defines": {"X": 1}}),
        _hip("h_other_build", "a.cpp", {"defines": {"X": 2}}),
        _hip("h_other_source", "b.cpp", {"defines": {"X": 1}}),
    ]
    _write_root(tmp_path / "src", ukds)
    expect = _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    # rocke: 2 builders; hip: 3 (source, build) pairs.
    assert {(e["kind"], e["name"]) for e in expect} == {
        ("rocke", "r_a"),
        ("rocke", "r_c"),
        ("hip", "h_a"),
        ("hip", "h_other_build"),
        ("hip", "h_other_source"),
    }
    assert sorted(_kept(tmp_path / "w" / "root")) == sorted(e["name"] for e in expect)


def test_pick_is_independent_of_authored_order(tmp_path):
    ukds = [_rocke(n) for n in ("m", "c", "x", "a2", "a1")]
    for i, order in enumerate(itertools.permutations(ukds, 3)):
        src, out = tmp_path / f"src{i}", tmp_path / f"w{i}" / "root"
        _write_root(src, order)
        expect = _derive(src, "gfx950", out)
        want = min(u["name"] for u in order)
        assert [e["name"] for e in expect] == [want]
        assert _kept(out) == [want]


def test_hip_build_key_ignores_dict_key_order(tmp_path):
    ukds = [
        _hip("h1", build={"defines": {"A": 1, "B": 2}}),
        _hip("h2", build={"defines": {"B": 2, "A": 1}}),
    ]
    _write_root(tmp_path / "src", ukds)
    assert [
        e["name"] for e in _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    ] == ["h1"]


def test_pick_only_among_ukds_shipping_for_arch(tmp_path):
    # "a_*" sorts first but ships for gfx942 only; a wildcard UKD and an explicit
    # match remain candidates of the same group.
    ukds = [
        _rocke("a_942_only", arch=["gfx942"]),
        _rocke("b_wild"),
        _rocke("c_explicit", arch=["gfx950"]),
        _rocke("d_empty", arch=[]),
    ]
    _write_root(tmp_path / "src", ukds, kdp_arch=("gfx942", "gfx950"))
    expect = _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    assert [e["name"] for e in expect] == ["b_wild"]
    expect = _derive(tmp_path / "src", "gfx942", tmp_path / "w2" / "root")
    assert [e["name"] for e in expect] == ["a_942_only"]


def test_kdp_not_shipping_for_arch_is_copied_unchanged(tmp_path):
    src = tmp_path / "src"
    _write_root(src, [_rocke("a"), _rocke("b", "b2")], kdp_arch=("gfx942",))
    assert _derive(src, "gfx950", tmp_path / "w" / "root") == []
    assert (tmp_path / "w/root/x.kdp.json").read_bytes() == (
        src / "x.kdp.json"
    ).read_bytes()


def test_wildcard_kdp_ships_everywhere(tmp_path):
    _write_root(tmp_path / "src", [_rocke("a")], kdp_arch=None)
    assert [
        e["name"] for e in _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    ] == ["a"]


def test_expect_records_kdp_relative_path(tmp_path):
    _write_root(tmp_path / "src", [_rocke("a")], kdp_file="sub/dir/x.kdp.json")
    expect = _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    assert expect == [{"kdp": "sub/dir/x.kdp.json", "name": "a", "kind": "rocke"}]


# --- configure-time refusals (exit 2, nothing written) ----------------------
def _refused(tmp_path, ukds, message, kdp_arch=("gfx950",)):
    _write_root(tmp_path / "src", ukds, kdp_arch=kdp_arch)
    r = _run(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    assert r.returncode == 2, r.stderr
    assert r.stderr.startswith("hkp_probe_derive:")
    assert message in r.stderr
    assert not (tmp_path / "w").exists()


def test_standalone_ukd_reference_exits_2(tmp_path):
    _refused(
        tmp_path,
        [_rocke("a"), "ukd-standalone-id"],
        "standalone UKD references unsupported by probe derive",
    )


def test_standalone_reference_in_real_fixture_exits_2(tmp_path):
    # tests/fixtures/main: pointwise.kdp.json ships for gfx950 and references
    # ukd-pointwise-add-f32-b128 by id.
    r = _run(MAIN_FIXTURE, "gfx950", tmp_path / "w" / "root")
    assert r.returncode == 2, r.stderr
    assert "pointwise.kdp.json" in r.stderr
    assert "ukd-pointwise-add-f32-b128" in r.stderr
    assert not (tmp_path / "w").exists()


def test_standalone_reference_in_other_arch_kdp_is_ignored(tmp_path):
    _write_root(
        tmp_path / "src", [_rocke("a"), "ukd-standalone-id"], kdp_arch=("gfx942",)
    )
    assert _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root") == []


@pytest.mark.parametrize("kind", ["hsaco", "embedded_source", "kpack", "mystery"])
def test_kind_without_probe_support_exits_2(tmp_path, kind):
    _refused(
        tmp_path,
        [_rocke("a"), _other(kind, "m")],
        f"kind {kind!r} has no probe support",
    )


def test_unsupported_kind_for_other_arch_is_ignored(tmp_path):
    ukds = [_rocke("a"), dict(_other("hsaco", "m"), arch=["gfx942"])]
    _write_root(tmp_path / "src", ukds, kdp_arch=("gfx942", "gfx950"))
    assert [
        e["name"] for e in _derive(tmp_path / "src", "gfx950", tmp_path / "w" / "root")
    ] == ["a"]


def test_ukd_arch_outside_kdp_arch_exits_2(tmp_path):
    # The packer rejects this root; derive must not hide it by dropping the UKD.
    _refused(tmp_path, [_rocke("a", arch=["gfx942"])], "not a subset of the KDP arch")


def test_kept_ukds_with_equal_names_exit_2(tmp_path):
    _refused(tmp_path, [_rocke("same", "b1"), _rocke("same", "b2")], "share a name")


def test_malformed_kdp_exits_2(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "bad.kdp.json").write_text("{ not json")
    r = _run(src, "gfx950", tmp_path / "w" / "root")
    assert r.returncode == 2
    assert "bad.kdp.json is unreadable" in r.stderr
    assert "Traceback" not in r.stderr


# --- idempotence ------------------------------------------------------------
def _stamp_old(out: Path):
    for p in [*out.rglob("*"), out.parent / "expect.json"]:
        if p.is_file():
            os.utime(p, ns=(10**18, 10**18))


def _mtimes(out: Path):
    return {
        p.relative_to(out.parent).as_posix(): p.stat().st_mtime_ns
        for p in [*out.rglob("*"), out.parent / "expect.json"]
        if p.is_file()
    }


def test_second_derive_touches_nothing(tmp_path):
    src = tmp_path / "src"
    _write_root(src, [_rocke("b"), _rocke("a")])
    out = tmp_path / "w" / "root"
    first = _derive(src, "gfx950", out)
    _stamp_old(out)
    before = _mtimes(out)
    assert before and set(before) >= {
        "root/x.kdp.json",
        "root/x.kmd.json",
        "expect.json",
    }
    assert _derive(src, "gfx950", out) == first
    assert _mtimes(out) == before


def test_rederive_follows_source_changes(tmp_path):
    src = tmp_path / "src"
    _write_root(src, [_rocke("b"), _rocke("a")])
    out = tmp_path / "w" / "root"
    _derive(src, "gfx950", out)
    (out / "stale.txt").write_text("stale")
    (out / "stale_dir").mkdir()
    (out / "stale_dir" / "f.json").write_text("{}")
    (src / "x.kmd.json").write_text('{"id": "kmd2"}\n')
    _write_root(src, [_rocke("b"), _rocke("0first")])
    expect = _derive(src, "gfx950", out)
    assert [e["name"] for e in expect] == ["0first"]
    assert _kept(out) == ["0first"]
    assert (out / "x.kmd.json").read_text() == '{"id": "kmd"}\n'
    assert sorted(p.name for p in out.rglob("*")) == ["x.kdp.json", "x.kmd.json"]


# --- arch rules pinned to the packer's --------------------------------------
_ARCH_LISTS = [None, [], ["gfx942"], ["gfx950"], ["gfx942", "gfx950"], ["gfx90a"]]


def _derive_module():
    sys.path.insert(0, str(PROBE_DERIVE.parent))
    try:
        import hkp_probe_derive_root
    finally:
        sys.path.pop(0)
    return hkp_probe_derive_root


@pytest.mark.parametrize("doc_arch", _ARCH_LISTS)
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_arch_matches_agrees_with_packer(doc_arch, arch):
    from hkp_pack import descriptors

    doc = {} if doc_arch is None else {"arch": doc_arch}
    assert _derive_module().arch_matches(doc, arch) == descriptors.arch_matches(
        doc, arch
    )


@pytest.mark.parametrize("kdp_arch", _ARCH_LISTS)
@pytest.mark.parametrize("ukd_arch", _ARCH_LISTS)
def test_arch_subset_agrees_with_packer(ukd_arch, kdp_arch):
    from hkp_pack import descriptors

    got = _derive_module().arch_subset_ok(ukd_arch or [], kdp_arch or [])
    assert got == descriptors._arch_subset_ok(ukd_arch or [], kdp_arch or [])
