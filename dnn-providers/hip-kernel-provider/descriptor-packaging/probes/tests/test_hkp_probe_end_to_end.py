"""derive -> real pack -> assert, for hip-only and mixed rocke+hip roots.

The roots are built from the packer's own fixtures. Real hipcc and rocke compiles:
this is the proof the probe tools work for producers other than rocke and that the
KDP paths derive records are the ones the packed output carries.
"""

import json
import shutil
import subprocess
import sys

import pytest

from conftest import (
    MAIN_FIXTURE,
    PROBE_DERIVE,
    ROCKE_FIXTURE,
    pack_root,
    run_assert,
)

# What derive keeps from the main fixture for gfx950: pointwise.kdp.json (gfx942 and
# gfx950) holds two hip UKDs in one compile group and the wildcard KDP one more in
# another; the gfx942-only KDPs ship nothing.
_HIP_EXPECT = [
    {"kdp": "pointwise.kdp.json", "name": "PointwiseAdd f32 block64", "kind": "hip"},
    {
        "kdp": "pointwise_wild.kdp.json",
        "name": "PointwiseAdd f32 block256",
        "kind": "hip",
    },
]


def _hip_source(dst):
    """The main fixture without its standalone UKD, which derive refuses."""
    shutil.copytree(MAIN_FIXTURE, dst)
    (dst / "pointwise_add_b128.ukd.json").unlink()
    kdp = json.loads((dst / "pointwise.kdp.json").read_text())
    kdp["kernelDescriptors"] = [
        u for u in kdp["kernelDescriptors"] if isinstance(u, dict)
    ]
    (dst / "pointwise.kdp.json").write_text(json.dumps(kdp, indent=2))
    return dst


def _derive_and_pack(src, work, rocm_kpack_dir, hipcc, comgr_lib):
    derived = work / "probe" / "root"
    r = subprocess.run(
        [
            sys.executable,
            str(PROBE_DERIVE),
            "--from",
            str(src),
            "--arch",
            "gfx950",
            "--out",
            str(derived),
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    out = pack_root(derived, work, rocm_kpack_dir, hipcc, comgr_lib)
    return out, work / "probe" / "expect.json"


@pytest.fixture(scope="module")
def hip_probe(tmp_path_factory, rocm_kpack_dir, hipcc, comgr_lib):
    work = tmp_path_factory.mktemp("hip_probe")
    return _derive_and_pack(
        _hip_source(work / "src"), work, rocm_kpack_dir, hipcc, comgr_lib
    )


@pytest.fixture(scope="module")
def mixed_probe(tmp_path_factory, rocm_kpack_dir, hipcc, comgr_lib):
    work = tmp_path_factory.mktemp("mixed_probe")
    src = work / "src"
    _hip_source(src / "hip")
    shutil.copytree(ROCKE_FIXTURE, src / "rocke")
    return _derive_and_pack(src, work, rocm_kpack_dir, hipcc, comgr_lib)


def test_hip_root_passes(hip_probe, rocm_kpack_dir):
    out, expect = hip_probe
    assert json.loads(expect.read_text()) == _HIP_EXPECT
    result = run_assert(out, expect, rocm_kpack_dir)
    assert result.returncode == 0, result.stderr


def test_hip_root_with_broken_hip_source_does_not_pack(
    tmp_path, rocm_kpack_dir, hipcc, comgr_lib
):
    src = _hip_source(tmp_path / "src")
    (src / "PointwiseAdd.cpp").write_text("this is not C++\n")
    derived = tmp_path / "probe" / "root"
    subprocess.run(
        [
            sys.executable,
            str(PROBE_DERIVE),
            "--from",
            str(src),
            "--arch",
            "gfx950",
            "--out",
            str(derived),
        ],
        check=True,
    )
    with pytest.raises(pytest.fail.Exception, match="hkp_pack failed"):
        pack_root(derived, tmp_path, rocm_kpack_dir, hipcc, comgr_lib)


def test_mixed_root_passes_with_per_ukd_kinds(mixed_probe, rocm_kpack_dir, comgr_lib):
    out, expect = mixed_probe
    entries = json.loads(expect.read_text())
    assert {(e["kdp"], e["kind"]) for e in entries} == {
        ("hip/pointwise.kdp.json", "hip"),
        ("hip/pointwise_wild.kdp.json", "hip"),
        ("rocke/attention.kdp.json", "rocke"),
    }
    result = run_assert(out, expect, rocm_kpack_dir, expect_comgr=comgr_lib)
    assert result.returncode == 0, result.stderr


def test_mixed_root_origin_is_checked_per_ukd(mixed_probe, rocm_kpack_dir, tmp_path):
    # Expecting every UKD to be hip (a single global kind) must fail on the rocke one.
    out, expect = mixed_probe
    entries = [dict(e, kind="hip") for e in json.loads(expect.read_text())]
    wrong = tmp_path / "expect.json"
    wrong.write_text(json.dumps(entries))
    result = run_assert(out, wrong, rocm_kpack_dir)
    assert result.returncode == 1
    assert "FAIL provenance-origin:" in result.stderr
    assert "rocke/attention.kdp.json" in result.stderr
    assert "hip/pointwise" not in result.stderr
