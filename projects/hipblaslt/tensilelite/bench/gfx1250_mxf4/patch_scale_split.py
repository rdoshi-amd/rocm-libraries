#!/usr/bin/env python3
"""Load each wave's scales as two 128 B x 2 tiles instead of one 256 B x 2 tile.

usage: patch_scale_split.py SRC_LOGDIR NAME

The scale LR addressing expects [32-row group][K group][32 rows x 4 B] in LDS, which is what two
128 B-wide tiles (global +0/+128, LDS +0/+256) produce. Full M/N tiles only: the second tile
keeps the first tile's tensor dim 0. Every scale load site is split, and the main-loop
s_wait_tensorcnt goes from 4 to 6. Writes logs/<stamp>-split-NAME/.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = "/opt/rocm-10.1.0a20260908+bkc.20260917/lib/llvm/bin"

src, name = sys.argv[1], sys.argv[2]
asm = glob.glob(f"{BENCH}/{src}/**/kernel.s", recursive=True) or \
      glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm[0]).read().split("\n")

# Bytes per MMA group row piece: A groups are 32 rows (WMMA M), B groups 16 columns (WMMA N),
# 4 B of scales per row per 128 K. A wave's 64 rows split into 256 B / width pieces.
WIDTH = {"MXSA": 128, "MXSB": 64}
loadsPerIter = 2 + sum(256 // w for w in WIDTH.values())

SIZE = {"MXSA": "SizeI", "MXSB": "SizeJ"}
out, inLoop, nLoads, nTile, nWait, nStep = [], False, 0, 0, 0, 0
for idx, l in enumerate(lines):
    # One iteration is two K/128 rows of the {K/128, M, 4} layout: Size * 8 bytes, not MT * 8.
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
    if re.match(r"/\* MAINLOOP_C0(_PortB)? start \*/", l):
        inLoop = True
    elif re.match(r"/\* MAINLOOP_C0(_PortB)? end \*/", l):
        inLoop = False
    m = re.match(r"s_or_b32 s\[sgprtdm(MXS[AB])Group1\+3\], s\[sgprtdm\1Group1\+3\], 0x1000000", l)
    if m:
        w = WIDTH[m.group(1)]
        l = f"s_or_b32 s[sgprtdm{m.group(1)}Group1+3], s[sgprtdm{m.group(1)}Group1+3], {hex(w << 16)} // tile0 = {w}"
        nTile += 1
    if inLoop and l.startswith("s_wait_tensorcnt 4"):
        l = f"s_wait_tensorcnt {loadsPerIter}                                 // loads per iteration"
        nWait += 1
    out.append(l)
    m = re.match(r"tensor_load_to_lds s\[sgprtdm(MXS[AB])Group0", l)
    if m:
        g0, w = f"sgprtdm{m.group(1)}Group0", WIDTH[m.group(1)]
        for _ in range(256 // w - 1):
            out += [f"s_add_u32 s[{g0}+1], s[{g0}+1], {2 * w}           // next MMA group: LDS += 2 * width",
                    f"s_add_u32 s[{g0}+2], s[{g0}+2], {w}           // global += width",
                    f"s_addc_u32 s[{g0}+3], s[{g0}+3], 0",
                    l]
        nLoads += 1
print(f"{nLoads} loads split, {nTile} tile0 patched, {nWait} waits -> {loadsPerIter}, "
      f"{nStep} scale steps fixed", file=sys.stderr)
assert nLoads == nTile == 10 and nWait == 2

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-split-{name}"
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
