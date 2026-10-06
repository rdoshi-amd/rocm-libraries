"""The `hsaco` producer: a pre-built code object packed like a compiled one.

A pre-built object takes the same path from the walk onward as a hip or rocKE
variant -- one archive per shard, a TOC entry, the SHA256 the runtime verifies,
and a signature read out of the object -- so these tests assert the shipped
result rather than the mechanism. The objects are synthesised, so nothing here
needs a toolchain or a device.
"""

import hashlib
import json

import pytest

from pack_helpers import _kdp, _load_kpack, _read, _run, _ukd, _write_json
from synthesised_objects import _arg, _kernel, _object, requires_msgpack
from hkp_pack.descriptors import load_flat_input
from hkp_pack.errors import HkpPackError

ARCH = "gfx1151"
SYMBOL = "prebuilt_kernel"
ARGS = [_arg("global_buffer", 8, 0), _arg("by_value", 4, 8)]


@pytest.fixture(autouse=True)
def _msgpack_is_available():
    """Every case packs an object whose signature is read from its note."""
    requires_msgpack()


def _author(root, *, file="objects/prebuilt.hsaco", sha256=None, ukds=1, data=None):
    """A KDP in `root/pack/` whose UKDs name one checked-in object.

    Returns the object's bytes.
    """
    data = data if data is not None else _object([_kernel(SYMBOL, ARGS)])
    pack = root / "pack"
    (pack / "objects").mkdir(parents=True, exist_ok=True)
    (pack / "objects" / "prebuilt.hsaco").write_bytes(data)

    source = {"kind": "hsaco", "file": file, "symbol": SYMBOL}
    if sha256 is not None:
        source["sha256"] = sha256
    entries = [_ukd(f"ukd-{index}", dict(source)) for index in range(ukds)]
    _write_json(pack, "prebuilt.kdp.json", _kdp("kdp-prebuilt", [ARCH], entries))
    return data


def _shipped_kdp(out):
    return _read(out / ARCH / "pack" / "prebuilt.kdp.json")


@pytest.mark.quick
def test_packs_the_object_byte_for_byte(tmp_path, hipcc, rocm_kpack_dir):
    root = tmp_path / "root"
    data = _author(root)

    _run(root, tmp_path, hipcc, rocm_kpack_dir, [ARCH])

    ukd = _shipped_kdp(tmp_path / "out")["kernelDescriptors"][0]
    source = ukd["kernel_source"]
    assert source["kind"] == "kpack"
    assert source["symbol"] == SYMBOL
    assert source["sha256"] == hashlib.sha256(data).hexdigest()

    archive = (tmp_path / "out" / ARCH / "pack" / source["library"]).resolve()
    packed = (
        _load_kpack(rocm_kpack_dir)
        .PackedKernelArchive.read(archive)
        .get_kernel(source["toc_key"], ARCH)
    )
    assert packed == data


@pytest.mark.quick
def test_reads_the_signature_out_of_the_object(tmp_path, hipcc, rocm_kpack_dir):
    root = tmp_path / "root"
    _author(root)

    _run(root, tmp_path, hipcc, rocm_kpack_dir, [ARCH])

    source = _shipped_kdp(tmp_path / "out")["kernelDescriptors"][0]["kernel_source"]
    assert source["signature"] == [
        {"kind": "global_buffer", "size": 8, "offset": 0},
        {"kind": "by_value", "size": 4, "offset": 8},
    ]


@pytest.mark.quick
def test_records_the_object_as_its_provenance(tmp_path, hipcc, rocm_kpack_dir):
    root = tmp_path / "root"
    _author(root)

    _run(root, tmp_path, hipcc, rocm_kpack_dir, [ARCH])

    provenance = _shipped_kdp(tmp_path / "out")["kernelDescriptors"][0]["provenance"]
    assert provenance["origin_kind"] == "hsaco"
    # Root-relative, so two folders that each hold a same-named object differ.
    assert provenance["source"] == "pack/objects/prebuilt.hsaco"
    # No compiler ran here, so none is recorded.
    assert "hipcc" not in json.dumps(provenance)


@pytest.mark.quick
def test_two_ukds_naming_one_object_share_one_entry(tmp_path, hipcc, rocm_kpack_dir):
    root = tmp_path / "root"
    _author(root, ukds=2)

    _run(root, tmp_path, hipcc, rocm_kpack_dir, [ARCH])

    entries = _shipped_kdp(tmp_path / "out")["kernelDescriptors"]
    assert (
        entries[0]["kernel_source"]["toc_key"] == entries[1]["kernel_source"]["toc_key"]
    )


@pytest.mark.quick
def test_a_matching_recorded_digest_packs(tmp_path, hipcc, rocm_kpack_dir):
    data = _object([_kernel(SYMBOL, ARGS)])
    root = tmp_path / "root"
    _author(root, data=data, sha256=hashlib.sha256(data).hexdigest())

    _run(root, tmp_path, hipcc, rocm_kpack_dir, [ARCH])


@pytest.mark.quick
def test_neg_an_object_that_moved_under_its_descriptor(tmp_path, hipcc, rocm_kpack_dir):
    root = tmp_path / "root"
    _author(root, sha256="0" * 64)

    with pytest.raises(HkpPackError, match="hashes to"):
        _run(root, tmp_path, hipcc, rocm_kpack_dir, [ARCH])


@pytest.mark.quick
def test_neg_a_missing_object(tmp_path, hipcc, rocm_kpack_dir):
    root = tmp_path / "root"
    _author(root, file="objects/absent.hsaco")

    with pytest.raises(HkpPackError, match="code object not found"):
        _run(root, tmp_path, hipcc, rocm_kpack_dir, [ARCH])


@pytest.mark.quick
def test_neg_an_object_outside_the_root(tmp_path, hipcc, rocm_kpack_dir):
    root = tmp_path / "root"
    _author(root, file="../../outside.hsaco")
    (tmp_path / "outside.hsaco").write_bytes(_object([_kernel(SYMBOL, ARGS)]))

    with pytest.raises(HkpPackError, match="escapes the source root"):
        _run(root, tmp_path, hipcc, rocm_kpack_dir, [ARCH])


@pytest.mark.quick
def test_neg_a_symbol_the_object_does_not_carry(tmp_path, hipcc, rocm_kpack_dir):
    root = tmp_path / "root"
    _author(root, data=_object([_kernel("another_kernel", ARGS)]))

    with pytest.raises(HkpPackError, match="not present"):
        _run(root, tmp_path, hipcc, rocm_kpack_dir, [ARCH])


@pytest.mark.quick
def test_neg_a_malformed_recorded_digest(tmp_path):
    root = tmp_path / "root"
    _author(root, sha256="not-a-digest")

    with pytest.raises(HkpPackError, match="64 lowercase hex"):
        load_flat_input(root, log=lambda *a, **k: None)
