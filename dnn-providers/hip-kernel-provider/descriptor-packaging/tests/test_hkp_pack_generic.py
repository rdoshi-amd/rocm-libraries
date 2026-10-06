"""Generic GPU targets at pack time.

A KDP whose `arch` names a table generic (`gfx11-generic`) is compiled and packed
once, under the generic's spelling, and that one tree is copied into the folder of
every selected member of the generic. The rules that keep that sound (S5 / S6 /
reverse direction / unknown name / rocKE) are validation errors with stable
substrings; the merge into a member folder is atomic and refuses two shards that
would write one path with different bytes.

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


def _generic_kdp_root(tmp_path, empty_arch_fixture):
    return _root(
        tmp_path,
        empty_arch_fixture,
        {"g": _kdp("kdp-g", [GENERIC], [_embedded("ukd-g")])},
    )


# --- layout ------------------------------------------------------------------


@pytest.mark.quick
def test_generic_entry_layout_is_one_copy_per_selected_member_folder(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    root = _generic_kdp_root(tmp_path, empty_arch_fixture)
    results = _pack(root, tmp_path, [MEMBER_A, MEMBER_B], rocm_kpack_dir)

    out = tmp_path / "out"
    # One folder per selected member: none for the generic or an unselected member,
    # and no staging directory left beside them.
    assert sorted(p.name for p in out.iterdir()) == [MEMBER_A, MEMBER_B]
    assert not (out / UNSELECTED_MEMBER).exists()
    assert not (out / GENERIC).exists()
    assert {a: r.out_dir for a, r in results.items()} == {
        MEMBER_A: out / MEMBER_A,
        MEMBER_B: out / MEMBER_B,
    }

    first, second = _files(out / MEMBER_A), _files(out / MEMBER_B)
    assert first == second
    assert "g.kdp.json" in first

    documents = {
        name: json.loads(data)
        for name, data in first.items()
        if name.endswith((".kdp.json", ".ukd.json"))
    }
    assert documents
    kdp = documents["g.kdp.json"]
    assert kdp["arch"] == [GENERIC]
    assert [u["arch"] for u in kdp["kernelDescriptors"]] == [[GENERIC]]


@pytest.mark.quick
def test_a_member_with_only_generic_content_still_gets_a_folder(
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
    results = _pack(root, tmp_path, ["gfx942", MEMBER_A], rocm_kpack_dir)

    out = tmp_path / "out"
    assert not results[MEMBER_A].skipped
    assert (out / MEMBER_A / "g.kdp.json").is_file()
    assert not (out / MEMBER_A / "c.kdp.json").exists()
    assert (out / "gfx942" / "c.kdp.json").is_file()
    assert not (out / "gfx942" / "g.kdp.json").exists()


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


@pytest.mark.quick
def test_empty_arch_kdp_is_not_emitted_into_the_generic_copy(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {
            "wild": _kdp("kdp-wild", [], [_embedded("ukd-wild")]),
            "g": _kdp("kdp-g", [GENERIC], [_embedded("ukd-g")]),
        },
    )
    _pack(root, tmp_path, [MEMBER_A], rocm_kpack_dir)

    generic_tree = tmp_path / "inter" / ".generic-out" / GENERIC
    assert (generic_tree / "g.kdp.json").is_file()
    assert not (generic_tree / "wild.kdp.json").exists()
    # The member folder holds the wildcard through the concrete pass, narrowed to it.
    member = tmp_path / "out" / MEMBER_A
    assert _read(member / "wild.kdp.json")["arch"] == [MEMBER_A]
    assert _read(member / "g.kdp.json")["arch"] == [GENERIC]


@pytest.mark.quick
def test_explicit_and_generic_packs_coexist_in_one_member_folder(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {
            "explicit": _kdp("kdp-explicit", [MEMBER_B], [_embedded("ukd-e")]),
            "generic": _kdp("kdp-generic", [GENERIC], [_embedded("ukd-g")]),
        },
    )
    _pack(root, tmp_path, [MEMBER_A, MEMBER_B], rocm_kpack_dir)

    out = tmp_path / "out"
    both = _files(out / MEMBER_B)
    assert json.loads(both["explicit.kdp.json"])["arch"] == [MEMBER_B]
    assert json.loads(both["generic.kdp.json"])["arch"] == [GENERIC]
    only_generic = _files(out / MEMBER_A)
    assert "explicit.kdp.json" not in only_generic
    assert only_generic["generic.kdp.json"] == both["generic.kdp.json"]


@pytest.mark.quick
def test_shared_engine_dispatch_and_kmd_files_merge_cleanly(
    tmp_path, empty_arch_fixture, rocm_kpack_dir
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {
            "explicit": _kdp("kdp-explicit", [MEMBER_B], [_embedded("ukd-e")]),
            "generic": _kdp("kdp-generic", [GENERIC], [_embedded("ukd-g")]),
        },
    )
    _pack(root, tmp_path, [MEMBER_B], rocm_kpack_dir)

    member = _files(tmp_path / "out" / MEMBER_B)
    shared = [p for p in root.iterdir() if p.name.endswith(_SHARED_SUFFIXES)]
    assert shared
    for path in shared:
        assert member[path.name] == path.read_bytes()
    assert sorted(member) == sorted(
        ["explicit.kdp.json", "generic.kdp.json", *(p.name for p in shared)]
    )


# --- merge guard and failure policy ------------------------------------------


@pytest.mark.quick
def test_merge_collision_with_differing_bytes_names_the_id_and_both_paths_and_leaves_no_partial_member_folder(
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

    message = str(err.value)
    assert "would be written by shard" in message
    assert "ukd-dup" in message
    assert str(member_dir / "dup.ukd.json") in message
    assert str(generic_dir / "dup.ukd.json") in message
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
    with pytest.raises(HkpPackError) as err:
        _pack(root, tmp_path, [MEMBER_A, MEMBER_B], rocm_kpack_dir)

    assert f"{GENERIC}: synthetic generic pack failure" in str(err.value)
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
    hsaco = _embedded("ukd-hsaco", [GENERIC])
    hsaco["kernel_source"] = {
        "kind": "hsaco",
        "file": CO_NAME,
        "symbol": "HsacoFixtureAdd",
    }
    hsaco["provenance"] = copy.deepcopy(_CONTRACT)
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
    ukd = _embedded("ukd-hsaco", [GENERIC])
    ukd["kernel_source"] = {
        "kind": "hsaco",
        "file": CO_NAME,
        "symbol": "HsacoFixtureAdd",
    }
    ukd["provenance"] = copy.deepcopy(_CONTRACT)
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


# --- S5 -----------------------------------------------------------------------


@pytest.mark.quick
def test_s5_list_with_a_generic_and_its_member_is_an_error(
    tmp_path, empty_arch_fixture
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {"g": _kdp("kdp-g", [GENERIC, MEMBER_B], [_embedded("ukd-g")])},
    )
    with pytest.raises(HkpPackError) as err:
        _load(root)
    message = str(err.value)
    assert "g.kdp.json" in message
    assert f"lists '{GENERIC}' and '{MEMBER_B}'" in message
    assert f"{GENERIC} contains {MEMBER_B}" in message


@pytest.mark.quick
def test_s5_two_generics_sharing_a_member_is_an_error(tmp_path, empty_arch_fixture):
    table_path = tmp_path / "overlap.json"
    _write(
        table_path,
        {
            "schemaVersion": 1,
            "generics": {
                "gfx11-generic": ["gfx1100", "gfx1101"],
                "gfx11-b-generic": ["gfx1101", "gfx1102"],
            },
        },
    )
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {"g": _kdp("kdp-g", ["gfx11-generic", "gfx11-b-generic"], [_embedded("u")])},
    )
    with pytest.raises(HkpPackError) as err:
        _load(root, GenericTargets.load(table_path))
    message = str(err.value)
    assert message.startswith("g.kdp.json:")
    assert "lists 'gfx11-generic' and 'gfx11-b-generic'" in message
    assert "share member gfx1101" in message


# --- S6 -----------------------------------------------------------------------

_S6_RULE = "must list every generic of the KDP"


def _standalone(uid, arch=None):
    ukd = _embedded(uid, arch)
    return ukd


@pytest.mark.quick
@pytest.mark.parametrize(
    "ukd_arch, accepted",
    [([GENERIC], True), ([MEMBER_B], False)],
    ids=["exact_generic", "narrower"],
)
def test_s6_standalone_ukd_under_a_generic_kdp(
    tmp_path, empty_arch_fixture, ukd_arch, accepted
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {"g": _kdp("kdp-g", [GENERIC], ["ukd-s"])},
        {"s": _standalone("ukd-s", ukd_arch)},
    )
    if accepted:
        _load(root)
        return
    with pytest.raises(HkpPackError) as err:
        _load(root)
    message = str(err.value)
    assert _S6_RULE in message
    assert "ukd-s" in message and "s.ukd.json" in message and "g.kdp.json" in message


_S7_RULE = "has an empty 'arch' (unrestricted) but the KDP lists"


@pytest.mark.quick
@pytest.mark.parametrize(
    "kdp_arch", [["gfx942"], [GENERIC]], ids=["concrete", "generic"]
)
def test_s7_standalone_ukd_without_arch_under_a_kdp_with_an_arch_is_an_error(
    tmp_path, empty_arch_fixture, kdp_arch
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {"g": _kdp("kdp-g", kdp_arch, ["ukd-s"])},
        {"s": _standalone("ukd-s")},
    )
    with pytest.raises(HkpPackError) as err:
        _load(root)
    message = str(err.value)
    assert message.startswith("g.kdp.json:")
    assert _S7_RULE in message
    assert "'ukd-s'" in message and "s.ukd.json" in message


@pytest.mark.quick
def test_s7_standalone_ukd_without_arch_under_an_arch_less_kdp_is_accepted(
    tmp_path, empty_arch_fixture
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {"g": _kdp("kdp-g", [], ["ukd-s"])},
        {"s": _standalone("ukd-s")},
    )
    _load(root)


@pytest.mark.quick
def test_s7_inline_ukd_without_arch_under_a_concrete_kdp_inherits_and_is_accepted(
    tmp_path, empty_arch_fixture
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {"g": _kdp("kdp-g", ["gfx942"], [_embedded("ukd-i")])},
    )
    _load(root)


@pytest.mark.quick
def test_s6_inline_ukd_with_a_narrower_arch_under_a_generic_kdp_is_an_error(
    tmp_path, empty_arch_fixture
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {"g": _kdp("kdp-g", [GENERIC], [_embedded("ukd-i", [MEMBER_B])])},
    )
    with pytest.raises(HkpPackError, match=_S6_RULE) as err:
        _load(root)
    assert str(err.value).startswith("g.kdp.json:")
    assert "inline UKD 'ukd-i'" in str(err.value)


@pytest.mark.quick
@pytest.mark.parametrize(
    "ukd_arch, rule",
    [
        (["gfx99-generic"], "generic target name absent"),
        ([GENERIC, MEMBER_B], f"{GENERIC} contains {MEMBER_B}"),
    ],
    ids=["unknown_generic", "generic_beside_member"],
)
def test_per_list_rules_apply_to_an_inline_ukd_arch(
    tmp_path, empty_arch_fixture, ukd_arch, rule
):
    """An inline UKD's own list is held to the per-list rules, which name the actual
    defect rather than a later KDP-level rule."""
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {"g": _kdp("kdp-g", [GENERIC], [_embedded("ukd-i", ukd_arch)])},
    )
    with pytest.raises(HkpPackError, match=rule):
        _load(root)


@pytest.mark.quick
def test_inline_ukd_without_its_own_arch_under_a_generic_kdp_is_accepted(
    tmp_path, empty_arch_fixture
):
    _load(_generic_kdp_root(tmp_path, empty_arch_fixture))


@pytest.mark.quick
def test_s6_mixed_list_ukd_listing_every_entry_is_accepted(
    tmp_path, empty_arch_fixture
):
    both = ["gfx942", GENERIC]
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {"m": _kdp("kdp-m", both, [_embedded("ukd-m", both)])},
    )
    _load(root)


@pytest.mark.quick
def test_s6_mixed_list_ukd_missing_the_generic_is_an_error(
    tmp_path, empty_arch_fixture
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {"m": _kdp("kdp-m", ["gfx942", GENERIC], [_embedded("ukd-m", ["gfx942"])])},
    )
    with pytest.raises(HkpPackError, match=_S6_RULE) as err:
        _load(root)
    assert str(err.value).startswith("m.kdp.json:")
    assert "inline UKD 'ukd-m'" in str(err.value)


# --- reverse direction, unknown names, rocKE ---------------------------------


@pytest.mark.quick
@pytest.mark.parametrize("kdp_arch", [[], [MEMBER_B]], ids=["empty", "explicit_member"])
def test_a_generic_ukd_under_a_kdp_that_does_not_list_it_is_an_error(
    tmp_path, empty_arch_fixture, kdp_arch
):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {"k": _kdp("kdp-k", kdp_arch, [_embedded("ukd-k", [GENERIC])])},
    )
    with pytest.raises(HkpPackError) as err:
        _load(root)
    message = str(err.value)
    assert f"declares generic target '{GENERIC}'" in message
    assert "ship in no shard" in message
    assert message.startswith("k.kdp.json:")
    assert "UKD 'ukd-k'" in message


@pytest.mark.quick
def test_unknown_generic_name_is_an_error_naming_the_file(tmp_path, empty_arch_fixture):
    root = _root(
        tmp_path,
        empty_arch_fixture,
        {"u": _kdp("kdp-u", ["gfx99-generic"], [_embedded("ukd-u")])},
    )
    with pytest.raises(HkpPackError) as err:
        _load(root)
    message = str(err.value)
    assert message.startswith("u.kdp.json:")
    assert "generic target name absent" in message
    assert "'gfx99-generic'" in message and str(GENERIC_TARGETS.path) in message


@pytest.mark.quick
@pytest.mark.parametrize(
    "ukd_arch", [None, [GENERIC]], ids=["no_own_arch", "own_generic"]
)
def test_rocke_under_a_generic_kdp_is_an_error(tmp_path, empty_arch_fixture, ukd_arch):
    """A rocKE UKD is refused whether it carries the generic itself or merely
    inherits it from the KDP."""
    ukd = _embedded("ukd-r", ukd_arch)
    ukd["kernel_source"] = {
        "kind": "rocke",
        "source": "pkg/kernels/attention.py",
        "builder": "build_attention",
        "spec": {"tile": 64},
    }
    root = _root(tmp_path, empty_arch_fixture, {"r": _kdp("kdp-r", [GENERIC], [ukd])})
    with pytest.raises(HkpPackError) as err:
        _load(root)
    message = str(err.value)
    assert message.startswith("r.kdp.json:")
    assert "does not support generic targets yet" in message
    assert f"'{GENERIC}'" in message


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
