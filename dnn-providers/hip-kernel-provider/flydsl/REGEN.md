<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# Regenerating the checked-in FlyDSL kernels

The `.hsaco` code objects under `kernels/<arch>/`, their `manifest.json`, the
per-arch `SOURCE.md` and the `descriptors/<arch>/` JSON set are **generated and
committed**. The build copies, packs and stages them; it never compiles them.
This file is how you reproduce them, and how you check that a tree you did not
build still matches the toolchain it claims.

Regeneration is **reproducible**: the pinned toolchain in
[`generators/_flydsl_env.py`](generators/_flydsl_env.py) produces byte-identical
objects. Verifying that is the first procedure below, and it is the one to run
after touching anything under `kernels_src/`.

> An object that regenerates to *different* bytes is a **change** to what this
> provider ships, not a refresh. See [When the bytes differ](#when-the-bytes-differ).

---

## 1. Environment

Everything below runs from the provider's `flydsl/` directory.

| Component | Pinned value | Where the pin lives |
|---|---|---|
| Python | 3.12 | — |
| `flydsl` | **0.3.4** (pip wheel) | `generators/_flydsl_env.py` `FLYDSL_VERSION` |
| FlyDSL kernel sources | commit `89ad52fbbb9e…` (`v0.3.4.1-19-g89ad52f`) | `_flydsl_env.py` `FLYDSL_KERNELS_*`, and vendored under `kernels_src/` |
| ROCm | 7.13.0 | read at runtime from `$ROCM_PATH/.info/version` |
| `torch` | any ROCm build | imported by the vendored kernel sources |
| `msgpack` | 1.2.2 | reads the AMDGPU metadata note out of each object |
| `zstandard` | 0.25.0 (`>=0.20.0`) | `rocm_kpack` compression |
| `rocm_kpack` | from source on `PYTHONPATH` | not pip-installed; see below |

`torch`, `msgpack` and `zstandard` are **not** transitive dependencies of the
`flydsl` wheel — install them explicitly.

```bash
python3 -m venv ~/flydsl-regen-venv
~/flydsl-regen-venv/bin/pip install 'flydsl==0.3.4' torch msgpack 'zstandard>=0.20.0'
```

On a box behind a TLS-intercepting proxy without the corporate CA installed, add
`--trusted-host pypi.org --trusted-host files.pythonhosted.org`.

`flydsl` compiles but does not *dispatch* here, so **no matching GPU is
required** — `prepare()` sets `COMPILE_ONLY=1`, which is what lets a gfx942
object be produced on an RDNA laptop.

`rocm_kpack` is consumed from a source checkout rather than installed:

```bash
export PYTHONPATH=/path/to/rocm-systems/shared/kpack/python
```

### The three settings every command below reads

Set these once per shell. The rest of this file uses them verbatim, so a
procedure you paste runs against your toolchain rather than someone else's:

```bash
PY=${PY:-python3}                          # the interpreter holding the pinned flydsl wheel
KPACK_PY=${KPACK_PY:?path to rocm-systems/shared/kpack/python}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export REGEN_OUT=${REGEN_OUT:-/tmp/flydsl-regen}
```

`KPACK_PY` has no sensible default — `rocm_kpack` is consumed from a source
checkout, so the `:?` makes an unset value fail at expansion rather than three
steps later inside `pack.py`.

The tree as committed was produced with Python 3.12.3, `flydsl` 0.3.4,
`torch` 2.10.0+rocm7.13.0a20260513 and ROCm 7.13.0. Those versions are the
contract; where they live on disk is not.

`ROCM_PATH` matters beyond finding the compiler: `rocm_version()` reads
`$ROCM_PATH/.info/version` and writes it into `manifest.json` and `SOURCE.md`.
With neither `ROCM_PATH` nor `ROCM_VERSION` set it records the literal
`"unknown"` rather than guessing — an unrecorded toolchain is better than a
wrong one, because a wrong one reads as verified.

### Generators run as modules

`generators/gen_rmsnorm.py` opens `from . import _flydsl_env as env`, so running
it as a script dies with `ImportError: attempted relative import with no known
parent package`. Invoke it from `flydsl/` as `python -m generators.gen_rmsnorm`.
`gen_descriptors.py` and `pack.py` are plain scripts and take either form.

---

## 2. Verify the checked-in objects reproduce

This writes nothing into the tree. It is the check that backs every claim this
directory makes about its own provenance.

```bash
cd dnn-providers/hip-kernel-provider/flydsl

# Compile into a scratch directory -- must be empty or nonexistent, since the
# generator writes files rather than clearing the directory, so an object left
# by an earlier run with a different instance list would survive and be compared.
PYTHONDONTWRITEBYTECODE=1 \
  $PY -m generators.gen_rmsnorm --arch gfx1151 --out-dir "$REGEN_OUT"
```

Then compare every object against the manifest the tree ships:

```bash
$PY - <<'PY'
import hashlib, json, os, pathlib, sys

manifest = json.loads(pathlib.Path("kernels/gfx1151/rmsnorm/manifest.json").read_text())
regen = pathlib.Path(os.environ["REGEN_OUT"]) / "gfx1151/rmsnorm"

bad = 0
for instance in manifest["instances"]:
    obj = regen / instance["file"]
    digest = hashlib.sha256(obj.read_bytes()).hexdigest() if obj.is_file() else "<missing>"
    if digest != instance["sha256"]:
        print(f"DIFFERS {instance['name']}\n  manifest {instance['sha256']}\n  regen    {digest}")
        bad += 1
print(f"{len(manifest['instances']) - bad}/{len(manifest['instances'])} byte-identical")
sys.exit(1 if bad else 0)
PY
```

The manifest and the provenance record must also come out identical:

```bash
diff kernels/gfx1151/rmsnorm/manifest.json "$REGEN_OUT/gfx1151/rmsnorm/manifest.json"
diff kernels/gfx1151/SOURCE.md             "$REGEN_OUT/gfx1151/SOURCE.md"
```

Expected, and the state of the tree as committed: **12/12 byte-identical**, both
diffs empty.

`manifest.json` agreeing is the stronger of the two checks — it carries the
`toolchain` block, so an identical manifest means the *recorded* toolchain and
the *actual* one are the same. Matching objects under a mismatched manifest
would mean the record is wrong.

### Re-run this after any bulk reformat of `kernels_src/`

The repo's `black` hook rewrites the vendored sources, and that is accepted; the
reformat was verified codegen-neutral by exactly the procedure above. The
checked-in binaries and their manifest are only trustworthy while that holds, so
the verification is the thing that keeps the acceptance honest.

---

## 3. Regenerate for real

Only when the kernels actually change: a `kernels_src/` edit, a new instance row,
a new arch, a deliberate toolchain bump.

```bash
cd dnn-providers/hip-kernel-provider/flydsl

# (1) Compile. Writes kernels/<arch>/rmsnorm/*.hsaco + manifest.json,
#     and refreshes kernels/<arch>/SOURCE.md.
$PY -m generators.gen_rmsnorm --arch gfx1151

# (2) Descriptors. Every field is derived from the manifest and the objects it
#     names, so a descriptor cannot disagree with the archive it describes.
$PY gen_descriptors.py --kernel-dir kernels --descriptor-dir descriptors --arch gfx1151

# (3) Verify the set. This is the same invocation the build runs before staging.
$PY gen_descriptors.py --kernel-dir kernels --descriptor-dir descriptors --arch gfx1151 --check
```

Steps 1 and 2 are per-arch and **one arch per invocation** — FlyDSL reads `ARCH`
from the environment at each compile, so a multi-arch run could file an object
under a name it was not built for. The generator additionally re-reads each
object's own `amdhsa.target` and refuses a mismatch.

Commit the objects, the manifest, `SOURCE.md` and the descriptors **together**.
They are one unit: `kernels/<arch>/` present without `descriptors/<arch>/` is a
configure-time `FATAL_ERROR`, because a shard holding an archive no descriptor
names is unreachable at runtime — nothing selects it, so the build passes and
proves nothing.

### Packing

`pack.py` is a **build step**, not a regeneration step — CMake runs it into the
build tree and the `.kpack` is not committed. Run it by hand only to check the
archive is deterministic too:

```bash
PYTHONPATH=$KPACK_PY PYTHONDONTWRITEBYTECODE=1 \
  $PY pack.py --kernel-dir kernels --arch gfx1151 --out-dir "${PACK_OUT:-/tmp/flydsl-pack}"
```

It packs from `manifest.json`, not from a directory glob, and re-verifies each
SHA256 against the bytes it actually writes — so descriptor and archive cannot
disagree, and a stray or half-written object cannot silently enter the archive.
Current expected output: `hip_kernel_provider_flydsl_gfx1151.kpack`, 29770 bytes,
12 kernels.

---

## 4. When the bytes differ

Work through these in order; the first three are environment, the last is a real
change.

1. **`flydsl` version.** `assert_flydsl_version()` hard-fails on anything other
   than the pin, naming this file. If you hit that error, you have the wrong
   wheel — installing the pinned one is the fix, *not* editing `FLYDSL_VERSION`.
2. **ROCm version.** Not asserted, only recorded — so this one fails silently
   into the manifest diff rather than loudly. FlyDSL's own release notes and this
   tree's Flash2 precedent both record ~1.4× performance swings from the ROCm
   version alone on identical source, so a ROCm delta is a real difference in
   what ships, not noise. Check the `toolchain` block in the manifest diff first.
3. **`kernels_src/` drift.** Either a local edit, or an upstream re-vendor that
   did not update the pins. `tools/diff_upstream.py` answers this against an
   upstream checkout.
4. **A genuine kernel change.** Then the new bytes are the point: regenerate per
   §3, commit objects + manifest + `SOURCE.md` + descriptors in one change, and
   say in the message what moved. A regenerated object committed as a refresh
   reviews as a no-op diff while changing what every user runs.

Bumping a pin is a deliberate act with a cost attached: `FLYDSL_VERSION` is
per-provider, not per-arch, so changing it obliges regenerating **every** arch in
`kernels/`. Leaving one behind ships a tree whose `SOURCE.md` files disagree
about which compiler built it.

---

## 5. Checking the vendored sources against upstream

`kernels_src/` is a vendored copy of eight FlyDSL kernel modules, one of which
carries three deliberate modifications recorded in its own header
(`kernels_src/kernels/norm/rmsnorm_kernel.py`). To see what has drifted:

```bash
python3 tools/diff_upstream.py --upstream /path/to/FlyDSL --commit
```

Expected against the pinned checkout, and the state of the tree as committed:

```
upstream at the pinned commit v0.3.4.1-19-g89ad52f (89ad52fbbb9e)
=== kernels/norm/rmsnorm_kernel.py
  …
7/8 vendored files identical to upstream after normalization; 1 differ, 0 absent upstream
```

The one differing file is the point of the tool, not a failure — read its hunks
against the header's three recorded modifications. Anything that header does not
account for is drift: either a local edit nobody wrote down, or an upstream
re-vendor that did not update the pins in `generators/_flydsl_env.py`. Exit
status reports whether the comparison held together, not whether files matched:
non-zero means a vendored file had **no upstream counterpart**, i.e. this is not
the checkout the tree was taken from and no hunk can be trusted.

It **normalizes both sides with `black` before diffing**. This is a requirement,
not a nicety: the repo's hook reformats our copy, so an un-normalized diff shows
a whole-file delta on every vendored file and the real semantic drift is
invisible inside it. The tool refuses to run if `black` is unavailable rather
than producing that diff. It needs no flydsl and no GPU — any Python 3.10+ with
`black` on `PATH` (or importable as `python -m black`) will do, which is why the
command above is not `$PY`.

Pass `--commit` to assert the checkout sits at the pinned commit. Without it the
tool warns with the actual `HEAD` and continues, since comparing against a
*different* commit is sometimes exactly what you want — it shows what upstream
has done since — but then the hunks mix local modifications with upstream's own
movement and cannot be read as drift.
