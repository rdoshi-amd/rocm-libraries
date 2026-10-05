# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests of the backward resource gate helper (code-object reader, loop slicer,
AGPR-copy counter, gate rules).

CPU only. The synthetic tests build a minimal ELF code object and a
disassembly snippet in memory. The compile tests cross-compile an existing
MFMA kernel for gfx942 through comgr and skip when comgr or llvm-objdump is
missing.
"""

from __future__ import annotations

import struct

import pytest

from tests import _attention_bwd_resources as res


# --------------------------------------------------------------------------- builders
def _mp(obj) -> bytes:
    """Tiny msgpack encoder (the subset the metadata note uses)."""
    if isinstance(obj, bool):
        return b"\xc3" if obj else b"\xc2"
    if isinstance(obj, int):
        if 0 <= obj <= 0x7F:
            return bytes([obj])
        if 0 <= obj <= 0xFFFF:
            return b"\xcd" + struct.pack(">H", obj)
        return b"\xce" + struct.pack(">I", obj)
    if isinstance(obj, str):
        raw = obj.encode()
        if len(raw) < 32:
            return bytes([0xA0 | len(raw)]) + raw
        return b"\xd9" + bytes([len(raw)]) + raw
    if isinstance(obj, list):
        return bytes([0x90 | len(obj)]) + b"".join(_mp(x) for x in obj)
    if isinstance(obj, dict):
        assert len(obj) < 16
        return bytes([0x80 | len(obj)]) + b"".join(
            _mp(k) + _mp(v) for k, v in obj.items()
        )
    raise TypeError(obj)


def _elf(meta: dict, kd: bytes, name: str) -> bytes:
    """ELF64 with sections: null, .note, .rodata (kernel descriptor), .symtab, .strtab, .shstrtab."""
    desc = _mp(meta)
    note_name = b"AMDGPU\x00\x00"
    note = struct.pack("<III", 7, len(desc), 32) + note_name + desc
    note += b"\x00" * ((-len(note)) % 4)
    strtab = b"\x00" + (name + ".kd").encode() + b"\x00"
    sym = struct.pack("<IBBHQQ", 0, 0, 0, 0, 0, 0) + struct.pack(
        "<IBBHQQ", 1, 0x11, 0, 2, 0x1000, 64
    )
    shstr = b"\x00.note\x00.rodata\x00.symtab\x00.strtab\x00.shstrtab\x00"
    names = {
        n: shstr.index(n.encode())
        for n in (".note", ".rodata", ".symtab", ".strtab", ".shstrtab")
    }
    body = bytearray(b"\x00" * 64)
    offs = {}
    for key, blob in (
        ("note", note),
        ("rodata", kd),
        ("symtab", sym),
        ("strtab", strtab),
        ("shstr", shstr),
    ):
        body += b"\x00" * ((-len(body)) % 8)
        offs[key] = len(body)
        body += blob
    body += b"\x00" * ((-len(body)) % 8)
    shoff = len(body)

    def sh(name_off, typ, addr, off, size, link=0, entsize=0):
        return struct.pack(
            "<IIQQQQIIQQ", name_off, typ, 0, addr, off, size, link, 0, 8, entsize
        )

    shdrs = (
        sh(0, 0, 0, 0, 0)
        + sh(names[".note"], 7, 0, offs["note"], len(note))
        + sh(names[".rodata"], 1, 0x1000, offs["rodata"], len(kd))
        + sh(names[".symtab"], 2, 0, offs["symtab"], len(sym), link=4, entsize=24)
        + sh(names[".strtab"], 3, 0, offs["strtab"], len(strtab))
        + sh(names[".shstrtab"], 3, 0, offs["shstr"], len(shstr))
    )
    body += shdrs
    hdr = b"\x7fELF" + bytes([2, 1, 1, 64]) + b"\x00" * 8
    hdr += struct.pack("<HHIQQQIHHHHHH", 3, 224, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 6, 5)
    body[:64] = hdr
    return bytes(body)


def _kd(accum_offset: int) -> bytes:
    kd = bytearray(64)
    struct.pack_into("<I", kd, 44, accum_offset // 4 - 1)
    return bytes(kd)


META = {
    "amdhsa.kernels": [
        {
            ".name": "k0",
            ".vgpr_count": 254,
            ".agpr_count": 126,
            ".sgpr_count": 40,
            ".group_segment_fixed_size": 34816,
            ".private_segment_fixed_size": 0,
            ".vgpr_spill_count": 0,
            ".sgpr_spill_count": 0,
            ".wavefront_size": 64,
            ".max_flat_workgroup_size": 256,
        }
    ]
}


# --------------------------------------------------------------------------- reader
def test_msgpack_roundtrip_of_metadata_subset():
    assert res.decode_msgpack(_mp(META)) == META


def test_reader_splits_arch_vgpr_and_agpr_from_accum_offset():
    blob = _elf(META, _kd(128), "k0")
    r = res.read_code_object_resources(blob, arch="gfx942")
    assert (r.vgpr_total, r.arch_vgpr, r.agpr) == (254, 128, 126)
    assert (r.lds_bytes, r.scratch_bytes, r.wavefront_size) == (34816, 0, 64)


def test_reader_without_agprs_reports_the_total_as_arch_vgpr():
    meta = {
        "amdhsa.kernels": [
            dict(META["amdhsa.kernels"][0], **{".agpr_count": 0, ".vgpr_count": 96})
        ]
    }
    r = res.read_code_object_resources(_elf(meta, _kd(96), "k0"), arch="gfx942")
    assert (r.arch_vgpr, r.agpr) == (96, 0)


def test_reader_rejects_non_elf_and_unknown_kernel():
    with pytest.raises(ValueError):
        res.read_code_object_resources(b"not an elf" * 10, arch="gfx942")
    with pytest.raises(ValueError):
        res.read_code_object_resources(
            _elf(META, _kd(128), "k0"), arch="gfx942", kernel="nope"
        )


def test_occupancy_from_vgprs_and_lds():
    r = res.KernelResources(
        "k", "gfx942", 254, 128, 126, 40, 34816, 0, 0, 0, 64, 256, {}
    )
    # 512 / 256 = 2 by VGPRs; LDS allows one 4-wave workgroup per CU -> 1 wave per SIMD.
    assert res.occupancy_waves_per_simd(r, lds_capacity_bytes=65536) == 1
    r2 = res.KernelResources("k", "gfx942", 128, 128, 0, 40, 0, 0, 0, 0, 64, 256, {})
    assert res.occupancy_waves_per_simd(r2, lds_capacity_bytes=65536) == 4


# --------------------------------------------------------------------------- slicer
ASM = """
0000000000001900 <kern>:
\ts_load_dwordx2 s[0:1], s[4:5], 0x0                   // 000000001900: C0060002 00000000
\tv_accvgpr_write_b32 a0, 0                             // 000000001908: D3D94000 18000080
\tv_mov_b32_e32 v1, 0                                  // 000000001910: 7E020280
\tv_mfma_f32_16x16x16_bf16 a[0:3], v[2:3], v[4:5], a[0:3] // 000000001914: D3E10000 04020902
\tv_accvgpr_read_b32 v6, a1                            // 00000000191C: D3D84006 18000101
\ts_cbranch_scc1 65532                                 // 000000001920: BF85FFFC <kern+0x14>
\tv_add_f32_e32 v7, v7, v8                             // 000000001924: 020E1107
\tv_accvgpr_mov_b32 a2, a3                             // 000000001928: D3D84006 18000101
\ts_cbranch_vccnz 65532                                // 000000001930: BF87FFFC <kern+0x24>
\ts_endpgm                                             // 000000001934: BF810000
"""


def test_parse_disassembly_branch_targets_from_label_and_immediate():
    instrs = res.parse_disassembly(ASM)
    br = [i for i in instrs if i.target is not None]
    assert [hex(b.target) for b in br] == ["0x1914", "0x1924"]
    # immediate decoding agrees with the label: addr + 4 + 4 * simm16
    assert br[0].addr + 4 + 4 * (65532 - 65536) == br[0].target


def test_loop_slicer_and_agpr_copy_counter():
    instrs = res.parse_disassembly(ASM)
    loops = res.find_loops(instrs)
    assert len(loops) == 2
    q = res.select_q_loop(instrs, loops)
    body = q.instructions(instrs)
    assert (
        body[0].mnemonic.startswith("v_mfma") and body[-1].mnemonic == "s_cbranch_scc1"
    )
    c = res.count_agpr_copies(body)
    assert (c.reads, c.writes, c.moves) == (
        1,
        0,
        0,
    )  # the write before the loop is excluded
    assert res.count_agpr_copies(instrs).total == 3


def test_select_q_loop_requires_a_matrix_loop():
    instrs = res.parse_disassembly(
        ASM.replace("v_mfma_f32_16x16x16_bf16", "v_add_f32_e32")
    )
    with pytest.raises(ValueError):
        res.select_q_loop(instrs, res.find_loops(instrs))


def test_agpr_copy_ceiling_model():
    assert (
        res.agpr_copy_ceiling(block_m=16, block_n=128, waves=4, s_in_agpr=False).total
        == 0
    )
    c = res.agpr_copy_ceiling(block_m=32, block_n=128, waves=4, s_in_agpr=True)
    assert (c.reads, c.writes) == (2 * 32 * 128 // 256, 0)


# --------------------------------------------------------------------------- gate
def test_gate_flags_copy_traffic_and_records_flavor():
    blob = _elf(META, _kd(128), "k0")
    g = res.evaluate_resource_gate(
        blob,
        arch="gfx942",
        lds_capacity_bytes=65536,
        block_m=16,
        block_n=128,
        waves=4,
        isa_text=ASM.replace("<kern>", "<k0>"),
        flavor="llvm22",
        comgr="comgr-test",
    )
    assert "agpr_copy_traffic" in g.failures and not g.passed and g.valid
    rec = g.record()
    assert rec["flavor"] == "llvm22" and rec["q_loop_agpr_copies"]["reads"] == 1


def test_gate_is_invalid_off_the_production_flavor():
    blob = _elf(META, _kd(128), "k0")
    g = res.evaluate_resource_gate(
        blob, arch="gfx942", lds_capacity_bytes=65536, flavor="llvm20"
    )
    assert not g.failures and not g.valid and not g.passed


@pytest.mark.parametrize(
    "field,value,token",
    [
        (".private_segment_fixed_size", 16, "scratch"),
        (".vgpr_spill_count", 2, "vgpr_spill"),
        (".group_segment_fixed_size", 70000, "lds"),
    ],
)
def test_gate_rule_tokens(field, value, token):
    meta = {"amdhsa.kernels": [dict(META["amdhsa.kernels"][0], **{field: value})]}
    g = res.evaluate_resource_gate(
        _elf(meta, _kd(128), "k0"),
        arch="gfx942",
        lds_capacity_bytes=65536,
        flavor="llvm22",
    )
    assert token in g.failures


def test_gate_arch_vgpr_budget_is_separate_from_agprs():
    meta = {
        "amdhsa.kernels": [
            dict(META["amdhsa.kernels"][0], **{".vgpr_count": 300, ".agpr_count": 0})
        ]
    }
    g = res.evaluate_resource_gate(
        _elf(meta, _kd(256), "k0"),
        arch="gfx942",
        lds_capacity_bytes=65536,
        flavor="llvm22",
    )
    assert "arch_vgpr" in g.failures and "agpr" not in g.failures


# --------------------------------------------------------------------------- compile
def _compile_dense_gfx942(agpr_alloc=None):
    pytest.importorskip("numpy")
    try:
        from rocke.helpers.compile import compile_kernel

        from kernels.gfx942.attention_dense import (
            Gfx942AttentionDenseSpec,
            build_attention_dense,
        )

        spec = Gfx942AttentionDenseSpec(
            batch=1,
            seqlen_q=2048,
            seqlen_kv=2048,
            num_query_heads=32,
            num_kv_heads=8,
            head_size=128,
            causal=True,
            dtype="bf16",
            block_n=64,
        )
        k = build_attention_dense(spec, arch="gfx942")
        if agpr_alloc is not None:
            k.attrs["agpr_alloc"] = agpr_alloc
        return compile_kernel(k, arch="gfx942", capture_ir_text=False).hsaco
    except (ImportError, OSError) as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"comgr unavailable: {exc}")


def test_compiled_code_object_resources_and_loop_copies():
    """The AGPR-reserving variant splits arch VGPR / AGPR and shows loop copies."""
    if res.find_objdump() is None:
        pytest.skip("llvm-objdump not found")
    plain = _compile_dense_gfx942()
    forced = _compile_dense_gfx942(agpr_alloc=(128, 128))
    r0 = res.read_code_object_resources(plain, arch="gfx942")
    r1 = res.read_code_object_resources(forced, arch="gfx942")
    assert r0.agpr == 0 and r0.arch_vgpr == r0.vgpr_total
    assert r1.agpr > 0 and r1.arch_vgpr + r1.agpr <= r1.vgpr_total
    assert r0.scratch_bytes == 0 and r0.vgpr_spill == 0
    for blob, expect_copies in ((plain, False), (forced, True)):
        text = res.disassemble(blob, arch="gfx942")
        instrs = res.parse_disassembly(text)
        body = res.select_q_loop(instrs, res.find_loops(instrs)).instructions(instrs)
        assert (res.count_agpr_copies(body).total > 0) == expect_copies
