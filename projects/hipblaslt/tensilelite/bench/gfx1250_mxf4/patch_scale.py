#!/usr/bin/env python3
"""Reorder or reshape the main-loop scale TDM loads of a finished build (timing only).

usage: patch_scale.py SRC_LOGDIR NAME [order=SA,SB,A,B] [esize=1|2|4|8] [tile0=N] [tile1=N]

order   issue order of the four main-loop load blocks (default A,B,SA,SB).
esize   scale descriptor element size in bytes (default 1). tensor dim 0 and the row stride
        are converted to elements; tile0 is in elements.
tile0   elements per row (default 256 / esize), tile1 rows (default 2).
Both main-loop copies are patched. Writes logs/<stamp>-scl-NAME/ for alt_timing.py.
"""
import glob, os, re, subprocess, sys, time

BENCH = os.path.dirname(os.path.abspath(__file__))
LLVM = "/opt/rocm-10.1.0a20260908+bkc.20260917/lib/llvm/bin"
TC = {"A": "A", "B": "B", "SA": "MXSA", "SB": "MXSB"}
ESIZE_CODE = {1: 0, 2: 1, 4: 2, 8: 3}

src, name = sys.argv[1], sys.argv[2]
opts = dict(a.split("=", 1) for a in sys.argv[3:])
order = opts.get("order", "A,B,SA,SB").split(",")
esize = int(opts.get("esize", 1))
tile0 = int(opts.get("tile0", 256 // esize))
tile1 = int(opts.get("tile1", 2))
shift = esize.bit_length() - 1
asm = glob.glob(f"{BENCH}/{src}/**/kernel.s", recursive=True) or \
      glob.glob(f"{BENCH}/{src}/**/assembly/*.s", recursive=True)
ini = glob.glob(f"{BENCH}/{src}/**/ClientParameters.ini", recursive=True)[0]
lines = open(asm[0]).read().split("\n")


def reshape(block, tc):
    out = []
    for l in block:
        if re.match(rf"s_mov_b32 s\[sgprtdm{tc}Group1\], 0x0\b", l):
            l = f"s_mov_b32 s[sgprtdm{tc}Group1], {hex(ESIZE_CODE[esize] << 16)}  // patched data_size"
        elif l.startswith("s_lshl_b32 s66, s64, 0x2"):
            l = f"s_lshl_b32 s66, s64, {2 - shift}  // patched dim0 in elements"
        elif re.match(rf"s_or_b32 s\[sgprtdm{tc}Group1\+3\], s\[sgprtdm{tc}Group1\+3\], 0x1000000", l):
            l = f"s_or_b32 s[sgprtdm{tc}Group1+3], s[sgprtdm{tc}Group1+3], {hex(tile0 << 16)}  // patched tile0"
        elif re.match(rf"s_mov_b32 s\[sgprtdm{tc}Group1\+4\], 0x2", l):
            l = f"s_mov_b32 s[sgprtdm{tc}Group1+4], {tile1}  // patched tile1"
        elif re.match(rf"s_lshl_b32 s\[sgprtdm{tc}Group1\+5\], s\[sgprSize[IJ]\], 0x2", l):
            l = re.sub(r"0x2$", str(2 - shift), l.split("//")[0].rstrip()) + "  // patched stride"
        out.append(l)
    return out


out, i, nLoops = [], 0, 0
while i < len(lines):
    l = lines[i]
    out.append(l)
    i += 1
    if not re.match(r"/\* MAINLOOP_C0(_PortB)? start \*/", l):
        continue
    nLoops += 1
    while not lines[i].startswith("/* TDM addr update: A "):
        out.append(lines[i]); i += 1
    blocks = {}
    for t in ("A", "B", "SA", "SB"):
        start = i
        while not lines[i].startswith(f"tensor_load_to_lds s[sgprtdm{TC[t]}Group0"):
            i += 1
        i += 1
        blocks[t] = lines[start:i]
    for t in ("SA", "SB"):
        blocks[t] = reshape(blocks[t], TC[t])
    for t in order:
        out += blocks[t]
assert nLoops == 2, nLoops
print(f"order {order} esize {esize} tile0 {tile0} tile1 {tile1}", file=sys.stderr)

dst = f"{BENCH}/logs/{time.strftime('%Y%m%d-%H%M%S')}-scl-{name}"
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
