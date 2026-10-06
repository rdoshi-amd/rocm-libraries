#!/usr/bin/env python3
"""Reshape every scale TDM load to one W-byte-row tile per wave with LDS padding, and fix the
scale step (timing and validation experiments).

usage: patch_scale_pad.py SRC_LOGDIR NAME

The scale LR addressing expects, per 128 rows (A) or columns (B), [MMA group][K group][W bytes]
with W = 32 rows * 4 B for A and 16 * 4 B for B. Wave w loads K group w >> 1 of rows
(w & 1) * 128 .. +128: 512 contiguous global bytes as 512 / W rows of W bytes, written to LDS with
W bytes of padding after every W bytes, starting at (w & 1) * 1024 + (w >> 1) * W. The scale
address step becomes Size * 8 (two K/128 rows). Full M/N tiles only (tensor dims are set to the
tile). Writes logs/<stamp>-pad-NAME/.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = "/opt/rocm-10.1.0a20260908+bkc.20260917/lib/llvm/bin"
WIDTH = {"MXSA": 128, "MXSB": 64}
SIZE = {"MXSA": "SizeI", "MXSB": "SizeJ"}


def padControl(w):
    dwords = w // 4
    interval = dwords.bit_length() - 2      # log2(dwords) - 1
    amount = dwords - 1
    return (1 << 20) | (interval << 22) | (amount << 25)


src, name = sys.argv[1], sys.argv[2]
opts = dict(a.split("=", 1) for a in sys.argv[3:])
# bpair=1: B also uses 128 B rows, pairing its 16-column MMA groups in LDS as
# [pair][K group][32 columns x 4 B]; the B scale ds_load immediates are rewritten to match.
bPair = opts.get("bpair", "0") == "1"
if bPair:
    WIDTH["MXSB"] = 128
asm = glob.glob(f"{BENCH}/{src}/**/kernel.s", recursive=True) or \
      glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm[0]).read().split("\n")

out, nLoads, nStep, nLR = [], 0, 0, 0
for idx, l in enumerate(lines):
    m = re.match(r"ds_load_b32 (v\d+), (v\d+) offset:\d+(\s+)// scaleMXSB\[group(\d+),K=(\d+)\]", l)
    if bPair and m:
        g, k = int(m.group(4)), int(m.group(5))
        off = (g >> 1) * 256 + k * 128 + (g & 1) * 64
        l = f"ds_load_b32 {m.group(1)}, {m.group(2)} offset:{off}{m.group(3)}// scaleMXSB[group{g},K={k}]: paired layout"
        nLR += 1
    m = re.match(r"s_add_u64 s\[sgprAddress(MXS[AB]):sgprAddress\1\+1\], s\[sgprAddress\1:sgprAddress\1\+1\], 2048", l)
    if m:
        tc = m.group(1)
        out += [f"s_lshl_b32 s64, s[sgpr{SIZE[tc]}], 3              // Size * 8: two K/128 rows",
                f"s_add_u32 s[sgprAddress{tc}], s[sgprAddress{tc}], s64",
                f"s_addc_u32 s[sgprAddress{tc}+1], s[sgprAddress{tc}+1], 0"]
        nStep += 1
        continue
    if l.startswith("s_mul_i32 s64, s[sgprStreamKLocalStart], 2048"):
        tc = next(re.search(r"sgprAddress(MXS[AB])\+0", x).group(1) for x in lines[idx:idx + 4]
                  if re.search(r"sgprAddress(MXS[AB])\+0", x))
        out += [f"s_mul_i32 s64, s[sgprStreamKLocalStart], s[sgpr{SIZE[tc]}] // SK K-start * Size",
                "s_lshl_b32 s64, s64, 3                             // * 8"]
        nStep += 1
        continue
    m = re.match(r"tensor_load_to_lds s\[sgprtdm(MXS[AB])Group0", l)
    if m:
        tc = m.group(1)
        w, g0, g1 = WIDTH[tc], f"sgprtdm{tc}Group0", f"sgprtdm{tc}Group1"
        rows = 512 // w
        out += [
            f"// patched: wave w loads K group w>>1, rows (w&1)*128..+128 as {rows} x {w} B, LDS pitch {2 * w}",
            "v_readfirstlane_b32 s65, v[vgprSerial]",
            "s_lshr_b32 s65, s65, 5                             // w",
            "s_and_b32 s66, s65, 1",
            "s_lshl_b32 s66, s66, 10                            // (w & 1) * 1024",
            "s_lshr_b32 s64, s65, 1",
            f"s_mul_i32 s64, s64, {w}                            // (w >> 1) * W",
            "s_add_u32 s66, s66, s64",
            "s_lshl_b32 s64, s65, 9                             // w * 512 (original per-wave LDS offset)",
            "s_sub_u32 s66, s66, s64",
            f"s_add_u32 s[{g0}+1], s[{g0}+1], s66",
            "s_and_b32 s66, s65, 1",
            "s_lshl_b32 s66, s66, 9                             // (w & 1) * 512",
            "s_lshl_b32 s64, s65, 8                             // w * 256 (original per-wave global offset)",
            "s_sub_u32 s66, s66, s64",
            "s_lshr_b32 s65, s65, 1",
            f"s_mul_i32 s65, s65, s[sgpr{SIZE[tc]}]",
            "s_lshl_b32 s65, s65, 2                             // (w >> 1) * Size * 4",
            "s_add_u32 s66, s66, s65",
            f"s_add_u32 s[{g0}+2], s[{g0}+2], s66",
            f"s_addc_u32 s[{g0}+3], s[{g0}+3], 0",
            f"s_mov_b32 s[{g1}], {hex(padControl(w))}            // 1 B elements, pad {w} B every {w} B",
            f"s_mov_b32 s[{g1}+1], {hex(w << 16)}               // dim0 = {w}",
            f"s_mov_b32 s[{g1}+2], {hex(rows << 16)}               // dim1 = {rows}",
            f"s_mov_b32 s[{g1}+3], {hex(w << 16)}               // tile0 = {w}",
            f"s_mov_b32 s[{g1}+4], {rows}                          // tile1 = {rows}",
            f"s_mov_b32 s[{g1}+5], {w}                         // row stride = {w}",
        ]
        nLoads += 1
    out.append(l)
print(f"{nLoads} scale loads reshaped, {nStep} scale steps fixed, {nLR} B scale reads moved",
      file=sys.stderr)
assert nLoads == 10 and nStep == 12

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-pad-{name}"
os.makedirs(dst)
s = f"{dst}/kernel.s"
open(s, "w").write("\n".join(out))
subprocess.run([f"{LLVM}/clang", "-x", "assembler", "--target=amdgcn-amd-amdhsa", "-mcode-object-version=4",
                "-c", "-Xclangas", "-target-feature", "-Xclangas", "+real-true16", "-mcpu=gfx1250",
                "-mno-wavefrontsize64", s, "-o", f"{dst}/kernel.o"], check=True, stderr=subprocess.DEVNULL)
subprocess.run([f"{LLVM}/clang", "--target=amdgcn-amd-amdhsa", "-Xlinker", "--build-id=sha1",
                f"{dst}/kernel.o", "-o", f"{dst}/kernel.co"], check=True)
cfg = re.sub(r"^code-object=.*$", f"code-object={dst}/kernel.co", open(ini).read(), flags=re.M)
open(f"{dst}/ClientParameters.ini", "w").write(cfg)
print(dst)
