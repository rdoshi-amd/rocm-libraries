"""Generic GPU targets at pack time.

A KDP whose `arch` names a table generic (`gfx11-generic`) is compiled and packed
once, under the generic's spelling, and that one tree is copied into the folder of
every selected member of the generic. The rules that keep that sound (a generic and
its member in one list, a UKD's arch under a generic KDP, the reverse direction, an
unknown name, rocKE) are validation errors; the merge into a member folder is atomic
and refuses two shards that would write one path with different bytes.

The quick tier packs `embedded_source` and authored `hsaco` UKDs, which need no
compiler. The hipcc tier at the end compiles for real and is not quick.
"""

import copy
import json
import shutil
from pathlib import Path

import pytest

from conftest import GENERIC_TARGETS, GENERIC_TARGETS_JSON
from hkp_pack import pipeline
from hkp_pack.descriptors import load_flat_input
from hkp_pack.errors import HkpPackError
from hkp_pack.generic_targets import GenericTargets
from hkp_pack.pipeline import _merge_generic_into_member, run_pipeline

GENERIC = "gfx11-generic"
MEMBER_A = "gfx1100"
MEMBER_B = "gfx1151"
UNSELECTED_MEMBER = "gfx1101"
CO_NAME = "HsacoFixture.co"

_SHARED_SUFFIXES = (".kmd.json", ".ued.json", ".udd.json", ".uhd.json", ".umd.json")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def _load_kpack(rocm_kpack_dir):
    from hkp_pack.kpack_resolver import load_kpack

    kpack, _comp = load_kpack(rocm_kpack_dir)
    return kpack


def _embedded(uid, arch=None):
    ukd = {
        "version": "0.1",
        "id": uid,
        "name": uid,
        "kernel_source": {
            "kind": "embedded_source",
            "source_file": "kernels/PointwiseAdd.cpp",
            "entry_point": "PointwiseAdd",
        },
        "metadata": {"dtype": "FLOAT", "block_size": 64},
        "priority": 0,
    }
    if arch is not None:
        ukd["arch"] = arch
    return ukd


# What a producing UKD must carry: the consumer declaration its engine's KMD checks.
_CONTRACT = {
    "specialization_contract": {
        "schema_version": 1,
        "consumers": [
            {
                "engine_id": "ued-solo",
                "kmd_id": "kmd-solo",
                "metadata_fields": [],
                "matcher_only_fields": ["block_size", "dtype"],
                "bindings": {},
                "vocabulary": {},
            }
        ],
    }
}


def _kdp(kid, arch, ukds):
    return {
        "version": "0.1",
        "id": kid,
        "name": kid,
        "arch": arch,
        "matchers": ["umd-solo"],
        "engine": "ued-solo",
        "dispatch": "udd-solo",
        "kernelDescriptors": ukds,
    }


def _root(tmp_path, empty_arch_fixture, kdps, standalone=()):
    """A source root: the fixture's shared descriptors plus the given KDPs.

    `kdps` maps a file stem to a KDP document, `standalone` maps a file stem to a
    standalone UKD document.
    """
    root = tmp_path / "src"
    root.mkdir()
    for path in empty_arch_fixture.iterdir():
        if path.name.endswith(_SHARED_SUFFIXES):
            shutil.copyfile(path, root / path.name)
    for stem, doc in dict(kdps).items():
        _write(root / f"{stem}.kdp.json", doc)
    for stem, doc in dict(standalone).items():
        _write(root / f"{stem}.ukd.json", doc)
    return root


def _pack(root, tmp_path, arches, rocm_kpack_dir, table=GENERIC_TARGETS_JSON):
    return run_pipeline(
        source_root=root,
        arches=list(arches),
        out_root=tmp_path / "out",
        hipcc="hipcc-not-invoked",
        generic_targets_json=table,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=tmp_path / "inter",
        source_label="generic-test",
    )


def _files(folder):
    """Relative path -> bytes of every file under `folder`."""
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _load(root, table=GENERIC_TARGETS):
    return load_flat_input(root, table, log=lambda *_: None)


# --- layout ------------------------------------------------------------------


@pytest.mark.quick
def test_generic_entries_land_in_every_selected_member_folder_only(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {
            "c": _kdp("kdp-c", ["gfx942"], [_embedded("ukd-c")]),
            "explicit": _kdp("kdp-explicit", [MEMBER_B], [_embedded("ukd-e")]),
            "g": _kdp("kdp-g", [GENERIC], [_embedded("ukd-g")]),
            "wild": _kdp("kdp-wild", [], [_embedded("ukd-wild")]),
        },
    )
    results = _pack(root, tmp_path, ["gfx942", MEMBER_A, MEMBER_B], rocm_kpack_dir)

    out = tmp_path / "out"
    # One folder per selected arch: none for the generic or an unselected member, and no
    # staging directory left beside them.
    assert sorted(p.name for p in out.iterdir()) == [MEMBER_A, MEMBER_B, "gfx942"]
    assert not (out / UNSELECTED_MEMBER).exists()
    assert not (out / GENERIC).exists()
    assert {a: r.out_dir for a, r in results.items()} == {
        arch: out / arch for arch in ("gfx942", MEMBER_A, MEMBER_B)
    }

    # The generic KDP ships verbatim into each member; a nonmember never gets it.
    for member in (MEMBER_A, MEMBER_B):
        kdp = _read(out / member / "g.kdp.json")
        assert kdp["arch"] == [GENERIC]
        assert [u["arch"] for u in kdp["kernelDescriptors"]] == [[GENERIC]]
        assert _read(out / member / "wild.kdp.json")["arch"] == [member]
    assert (out / MEMBER_A / "g.kdp.json").read_bytes() == (
        out / MEMBER_B / "g.kdp.json"
    ).read_bytes()
    assert not (out / "gfx942" / "g.kdp.json").exists()

    # Concrete content stays with its own arch.
    assert (out / "gfx942" / "c.kdp.json").is_file()
    assert not (out / MEMBER_A / "c.kdp.json").exists()
    assert (out / MEMBER_B / "explicit.kdp.json").is_file()
    assert not (out / MEMBER_A / "explicit.kdp.json").exists()

    # An empty-arch KDP is not part of the generic pass.
    generic_tree = tmp_path / "inter" / ".generic-out" / GENERIC
    assert (generic_tree / "g.kdp.json").is_file()
    assert not (generic_tree / "wild.kdp.json").exists()

    # Shared engine/dispatch/matcher files are identical in every folder.
    shared = [p for p in root.iterdir() if p.name.endswith(_SHARED_SUFFIXES)]
    assert shared
    for arch in ("gfx942", MEMBER_A, MEMBER_B):
        files = _files(out / arch)
        for path in shared:
            assert files[path.name] == path.read_bytes()


@pytest.mark.quick
def test_generic_entry_is_not_materialized_when_no_member_is_selected(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {
            "c": _kdp("kdp-c", ["gfx942"], [_embedded("ukd-c")]),
            "g": _kdp("kdp-g", [GENERIC], [_embedded("ukd-g")]),
        },
    )
    _pack(root, tmp_path, ["gfx942"], rocm_kpack_dir)

    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["gfx942"]
    assert not (out / "gfx942" / "g.kdp.json").exists()
    assert not (tmp_path / "inter" / ".generic-out").exists()
    assert not (tmp_path / "inter" / GENERIC).exists()


# --- merge guard and failure policy ------------------------------------------


@pytest.mark.quick
def test_merge_collision_with_differing_bytes_is_refused_and_leaves_the_member_untouched(
    tmp_path,
):
    generic_dir = tmp_path / "inter" / ".generic-out" / GENERIC
    member_dir = tmp_path / "out" / MEMBER_A
    staging = tmp_path / "out" / f".{MEMBER_A}.merge.staging"
    _write(generic_dir / "only_generic.kdp.json", {"id": "kdp-only"})
    _write(generic_dir / "dup.ukd.json", {"id": "ukd-dup", "side": "generic"})
    _write(member_dir / "dup.ukd.json", {"id": "ukd-dup", "side": "concrete"})
    _write(member_dir / "concrete_only.kdp.json", {"id": "kdp-c"})
    before = _files(member_dir)

    with pytest.raises(HkpPackError) as err:
        _merge_generic_into_member(generic_dir, member_dir, staging)

    assert str(member_dir / "dup.ukd.json") in str(err.value)
    assert _files(member_dir) == before
    assert not staging.exists()


@pytest.mark.quick
def test_a_member_whose_concrete_pass_failed_gets_no_generic_copy(
    tmp_path, empty_arch_fixture, rocm_kpack_dir, monkeypatch
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {
            "c": _kdp("kdp-c", [MEMBER_A, MEMBER_B], [_embedded("ukd-c")]),
            "g": _kdp("kdp-g", [GENERIC], [_embedded("ukd-g")]),
        },
    )
    real = pipeline.compile_intermediate

    def failing(flat, source_root, arch, *args, **kwargs):
        if arch == MEMBER_A:
            raise HkpPackError("synthetic compile failure")
        return real(flat, source_root, arch, *args, **kwargs)

    monkeypatch.setattr(pipeline, "compile_intermediate", failing)
    with pytest.raises(HkpPackError, match="synthetic compile failure"):
        _pack(root, tmp_path, [MEMBER_A, MEMBER_B], rocm_kpack_dir)

    out = tmp_path / "out"
    assert not (out / MEMBER_A).exists()
    assert (out / MEMBER_B / "g.kdp.json").is_file()
    assert (out / MEMBER_B / "c.kdp.json").is_file()
    assert not [p for p in out.iterdir() if p.name.startswith(".")]


@pytest.mark.quick
def test_a_failed_generic_pass_leaves_no_member_copy(
    tmp_path, empty_arch_fixture, rocm_kpack_dir, monkeypatch
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {
            "c": _kdp("kdp-c", [MEMBER_A, MEMBER_B], [_embedded("ukd-c")]),
            "g": _kdp("kdp-g", [GENERIC], [_embedded("ukd-g")]),
        },
    )
    real = pipeline.pack_arch

    def failing(flat, inter, out_dir, *args, **kwargs):
        if inter.arch == GENERIC:
            raise HkpPackError("synthetic generic pack failure")
        return real(flat, inter, out_dir, *args, **kwargs)

    monkeypatch.setattr(pipeline, "pack_arch", failing)
    with pytest.raises(HkpPackError, match="synthetic generic pack failure"):
        _pack(root, tmp_path, [MEMBER_A, MEMBER_B], rocm_kpack_dir)

    out = tmp_path / "out"
    for member in (MEMBER_A, MEMBER_B):
        assert (out / member / "c.kdp.json").is_file()
        assert not (out / member / "g.kdp.json").exists()
    assert not (tmp_path / "inter" / ".generic-out" / GENERIC).exists()


@pytest.mark.quick
def test_generic_copies_of_every_producer_are_equal_across_separate_builds(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir
):
    """An embedded and an hsaco generic entry, packed once for gfx1100 and once for
    gfx1151 from separate roots: the member folders hold identical files, so the
    copies per-arch shard builds produce collapse to one. Nothing of the build's
    target list may leak into the generic copy."""
    hsaco = _hsaco_ukd("ukd-hsaco", [GENERIC])
    folders = {}
    for member in (MEMBER_A, MEMBER_B):
        base = tmp_path / member
        base.mkdir()
        root = _root(
            base,
            empty_arch_fixture,
            {
                "e": _kdp("kdp-e", [GENERIC], [_embedded("ukd-e")]),
                "h": _kdp("kdp-h", [GENERIC], [copy.deepcopy(hsaco)]),
            },
        )
        shutil.copyfile(hsaco_fixture_dir / "gfx942" / CO_NAME, root / CO_NAME)
        _pack(root, base, [member], rocm_kpack_dir)
        folders[member] = _files(base / "out" / member)
    assert any(name.endswith(".kpack") for name in folders[MEMBER_A])
    assert folders[MEMBER_A] == folders[MEMBER_B]


# --- authored hsaco under a generic ------------------------------------------


@pytest.mark.quick
def test_hsaco_under_a_generic_packs_under_the_generic_key_into_every_member_folder(
    tmp_path, empty_arch_fixture, hsaco_fixture_dir, rocm_kpack_dir
):
    authored = (hsaco_fixture_dir / "gfx942" / CO_NAME).read_bytes()
    ukd = _hsaco_ukd("ukd-hsaco", [GENERIC])
    root = _root(tmp_path, empty_arch_fixture, {"g": _kdp("kdp-g", [GENERIC], [ukd])})
    shutil.copyfile(hsaco_fixture_dir / "gfx942" / CO_NAME, root / CO_NAME)
    results = _pack(root, tmp_path, [MEMBER_A, MEMBER_B], rocm_kpack_dir)

    out = tmp_path / "out"
    shipped = _read(out / MEMBER_A / "g.kdp.json")["kernelDescriptors"][0]
    ks = shipped["kernel_source"]
    assert ks["kind"] == "kpack"
    assert ks["library"] == f"kpack/hip_kernel_provider_{GENERIC}.kpack"
    assert shipped["arch"] == [GENERIC]
    for member in (MEMBER_A, MEMBER_B):
        archive_path = out / member / ks["library"]
        assert results[member].generic_kpack_paths == (archive_path,)
        archive = _load_kpack(rocm_kpack_dir).PackedKernelArchive.read(archive_path)
        assert bytes(archive.get_kernel(ks["toc_key"], GENERIC)) == authored
        assert archive.get_kernel(ks["toc_key"], member) is None
    assert _files(out / MEMBER_A) == _files(out / MEMBER_B)


# --- arch rules at load -------------------------------------------------------


def _rocke_ukd(uid, arch=None):
    ukd = _embedded(uid, arch)
    ukd["kernel_source"] = {
        "kind": "rocke",
        "source": "pkg/kernels/attention.py",
        "builder": "build_attention",
        "spec": {"tile": 64},
    }
    return ukd


def _hsaco_ukd(uid, arch):
    ukd = _embedded(uid, arch)
    ukd["kernel_source"] = {
        "kind": "hsaco",
        "file": CO_NAME,
        "symbol": "HsacoFixtureAdd",
    }
    ukd["provenance"] = copy.deepcopy(_CONTRACT)
    return ukd


_OVERLAPPING_TABLE = {
    "schemaVersion": 1,
    "generics": {
        "gfx11-generic": ["gfx1100", "gfx1101"],
        "gfx11-b-generic": ["gfx1101", "gfx1102"],
    },
}

# (kdp arch, UKD form, UKD arch, table override). UKD form: "inline" puts the UKD in the
# KDP, "standalone" is a separate .ukd.json the KDP names, "rocke" is an inline rocKE UKD.
_REJECTED = {
    "generic_beside_its_member": ([GENERIC, MEMBER_B], "inline", None, None),
    "two_generics_sharing_a_member": (
        ["gfx11-generic", "gfx11-b-generic"],
        "inline",
        None,
        _OVERLAPPING_TABLE,
    ),
    "unknown_generic": (["gfx99-generic"], "inline", None, None),
    "standalone_narrower_than_generic_kdp": ([GENERIC], "standalone", [MEMBER_B], None),
    "inline_narrower_than_generic_kdp": ([GENERIC], "inline", [MEMBER_B], None),
    "inline_unknown_generic": ([GENERIC], "inline", ["gfx99-generic"], None),
    "inline_generic_beside_member": ([GENERIC], "inline", [GENERIC, MEMBER_B], None),
    "mixed_list_ukd_missing_the_generic": (
        ["gfx942", GENERIC],
        "inline",
        ["gfx942"],
        None,
    ),
    "standalone_without_arch_under_concrete_kdp": (
        ["gfx942"],
        "standalone",
        None,
        None,
    ),
    "standalone_without_arch_under_generic_kdp": ([GENERIC], "standalone", None, None),
    "generic_ukd_under_empty_arch_kdp": ([], "inline", [GENERIC], None),
    "generic_ukd_under_kdp_not_listing_it": ([MEMBER_B], "inline", [GENERIC], None),
    "rocke_inheriting_a_generic": ([GENERIC], "rocke", None, None),
    "rocke_with_own_generic": ([GENERIC], "rocke", [GENERIC], None),
}

_ACCEPTED = {
    "standalone_listing_the_generic": ([GENERIC], "standalone", [GENERIC]),
    "standalone_without_arch_under_arch_less_kdp": ([], "standalone", None),
    "inline_inheriting_a_concrete_kdp": (["gfx942"], "inline", None),
    "inline_inheriting_a_generic_kdp": ([GENERIC], "inline", None),
    "mixed_list_ukd_listing_every_entry": (
        ["gfx942", GENERIC],
        "inline",
        ["gfx942", GENERIC],
    ),
}


def _arch_rule_root(tmp_path, empty_arch_fixture, kdp_arch, form, ukd_arch):
    if form == "standalone":
        return _root(
            tmp_path,
            empty_arch_fixture,
            {"g": _kdp("kdp-g", kdp_arch, ["ukd-s"])},
            {"s": _embedded("ukd-s", ukd_arch)},
        )
    ukd = (
        _rocke_ukd("ukd-i", ukd_arch)
        if form == "rocke"
        else _embedded("ukd-i", ukd_arch)
    )
    return _root(tmp_path, empty_arch_fixture, {"g": _kdp("kdp-g", kdp_arch, [ukd])})


@pytest.mark.quick
@pytest.mark.parametrize("case", sorted(_REJECTED))
def test_an_arch_rule_violation_is_an_error_naming_the_kdp(
    tmp_path, empty_arch_fixture, case
):
    kdp_arch, form, ukd_arch, table_doc = _REJECTED[case]
    root = _arch_rule_root(tmp_path, empty_arch_fixture, kdp_arch, form, ukd_arch)
    table = GENERIC_TARGETS
    if table_doc is not None:
        table_path = tmp_path / "table.json"
        _write(table_path, table_doc)
        table = GenericTargets.load(table_path)
    with pytest.raises(HkpPackError) as err:
        _load(root, table)
    assert "g.kdp.json" in str(err.value)


@pytest.mark.quick
@pytest.mark.parametrize("case", sorted(_ACCEPTED))
def test_a_consistent_arch_declaration_loads(tmp_path, empty_arch_fixture, case):
    kdp_arch, form, ukd_arch = _ACCEPTED[case]
    _load(_arch_rule_root(tmp_path, empty_arch_fixture, kdp_arch, form, ukd_arch))


# --- hipcc tier (not quick) ---------------------------------------------------


def _hip_generic_root(dest, empty_arch_fixture):
    """The empty_arch hip fixture with its KDP rewritten to the generic."""
    shutil.copytree(empty_arch_fixture, dest)
    kdp_path = dest / "solo.kdp.json"
    doc = _read(kdp_path)
    doc["arch"] = [GENERIC]
    _write(kdp_path, doc)
    return dest


def _hip_pack(root, out, inter, arches, hipcc, rocm_kpack_dir):
    return run_pipeline(
        source_root=root,
        arches=list(arches),
        out_root=out,
        hipcc=hipcc,
        generic_targets_json=GENERIC_TARGETS_JSON,
        rocm_kpack_dir=rocm_kpack_dir,
        inter_root=inter,
        source_label="generic-test",
    )


def test_generic_hip_archive_is_keyed_by_the_generic_spelling(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir
):
    root = _hip_generic_root(tmp_path / "src", empty_arch_fixture)
    results = _hip_pack(
        root, tmp_path / "out", tmp_path / "inter", [MEMBER_A], hipcc, rocm_kpack_dir
    )

    shipped = _read(tmp_path / "out" / MEMBER_A / "solo.kdp.json")
    ks = shipped["kernelDescriptors"][0]["kernel_source"]
    assert shipped["arch"] == [GENERIC]
    assert ks["library"] == f"kpack/hip_kernel_provider_{GENERIC}.kpack"
    archive = _load_kpack(rocm_kpack_dir).PackedKernelArchive.read(
        tmp_path / "out" / MEMBER_A / ks["library"]
    )
    assert archive.get_kernel(ks["toc_key"], GENERIC) is not None
    assert archive.get_kernel(ks["toc_key"], MEMBER_A) is None
    assert results[MEMBER_A].kpack_path is None


def test_compiled_agreement_holds_on_a_generic_copy(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir
):
    """A shipped generic copy carries the generic as its literal arch, which names no
    device: `consumer_records` must still count its own entry as a consumer, or the
    record lookup for the copy's kernel raises."""
    from hkp_pack.desk_check import compiled_agreement

    root = _hip_generic_root(tmp_path / "src", empty_arch_fixture)
    _hip_pack(
        root, tmp_path / "out", tmp_path / "inter", [MEMBER_A], hipcc, rocm_kpack_dir
    )
    failures, _unclaimed, _verified = compiled_agreement(
        tmp_path / "out" / MEMBER_A / "solo.kdp.json", GENERIC_TARGETS, rocm_kpack_dir
    )
    assert failures == []


def test_generic_pack_is_byte_identical_across_separate_per_arch_builds(
    tmp_path, empty_arch_fixture, hipcc, rocm_kpack_dir, monkeypatch
):
    """Three separate builds of one hip root -- GPU_TARGETS gfx1100 only, gfx1151
    only, and both -- each with its own out root, inter root, cwd and absolute copy
    of the source. Every file of the generic entry is identical across all of them
    and across the member folders of the two-target build: the copies a real
    per-arch shard build produces must collapse to one."""
    builds = {}
    for label, arches in (
        ("only_a", [MEMBER_A]),
        ("only_b", [MEMBER_B]),
        ("both", [MEMBER_A, MEMBER_B]),
    ):
        base = tmp_path / label
        base.mkdir()
        root = _hip_generic_root(base / "deeper" / label / "src", empty_arch_fixture)
        (base / "cwd").mkdir()
        monkeypatch.chdir(base / "cwd")
        _hip_pack(
            root, base / "out", base / "inter" / label, arches, hipcc, rocm_kpack_dir
        )
        for member in arches:
            builds[(label, member)] = _files(base / "out" / member)

    reference_key, reference = next(iter(builds.items()))
    assert any(name.endswith(".kpack") for name in reference)
    assert any(name.endswith(".kdp.json") for name in reference)
    for key, files in builds.items():
        assert files == reference, f"{key} differs from {reference_key}"
