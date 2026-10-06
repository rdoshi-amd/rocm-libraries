<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# FlyDSL kernels in the hip-kernel-provider

Ahead-of-time RMSNorm and attention kernels written in
[FlyDSL](https://github.com/ROCm/FlyDSL), compiled once and **checked in**,
reaching hipDNN through the kernel ingestor.

This directory holds the kernels, the generators that produce them, and the
descriptors that make them selectable. The matching C++ — the code that decides
whether a graph may run on them and how to launch them — lives one level up, one
file per op:
[`FlydslRmsNormNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslRmsNormNative.cpp)
and
[`FlydslSdpaNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslSdpaNative.cpp).

**Nothing here is compiled by the build.** The `.hsaco` objects are committed
artifacts; the provider's shared packer packs and stages them. That is the central fact about this
directory, and it is what the rest of this file is mostly about: an artifact
whose build is not reproduced on every `ninja` has to carry its provenance and a
way to check it, or it is just bytes someone once vouched for.

## Where to go

| | |
|---|---|
| **What is covered, and what is refused** | [COVERAGE.md](COVERAGE.md) |
| **Rebuilding or verifying the kernels** | [REGEN.md](REGEN.md) |

## What ships

Two ops, 52 kernel objects, built once for the LLVM generic target
**`gfx11-generic`** and shipped to every RDNA3 / RDNA3.5 part it covers:
gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1152 and gfx1153. Each
of those arches' shards carries the same bytes (see
[REGEN.md §3](REGEN.md) for why one object set is enough, and what it needs).

**RMSNorm forward** (`hipkernel:flydsl_rmsnorm`), 12 objects, bf16 and f16:

- a **specialized** tier with `N` baked in at 3072 / 3584 / 4096 / 5120 / 8192
- a **generic** tier that reads `N` from the tensor descriptor at runtime, so
  every other width is served too — more slowly, but served

Widths outside the specialized list are a **performance** boundary, not a
coverage one. The things that are genuinely out of scope — backward, training
phase, bias, other dtypes, non-packed operands — are enumerated in
[COVERAGE.md §3](COVERAGE.md), each with the reason the shipped objects cannot
compute it.

**SDPA forward** (`hipkernel:flydsl_sdpa`), 40 objects: bf16 and f16 ×
`head_dim` 64, 96, 128 and 256 × causal on and off × with and without an
additive f32 bias (`attn_mask`, broadcast over any axis), plus a generic tier
that serves every other `head_dim` that is a multiple of 8 up to 256. Batch, both sequence lengths,
head counts (MHA, GQA, MQA), every stride, the scale, the mask bounds (causal at
either corner, right-side bands, sliding windows) and the LSE output are runtime
arguments, so each object serves every shape and layout of its class. Inference
only; not yet served: padding / varlen. Status
against hipDNN's Tier 0 / Tier 1 and every declined feature with its effort are
in [COVERAGE.md §5](COVERAGE.md).

## How it reaches hipDNN

The objects and their descriptors are a **bundle of the provider's production
descriptor root**, beside rocKE's, and ship through the same packer:

```
src/engines/kernel_ingestor_engine/descriptors/
  rocKE/…                         authored rocKE bundles (compiled at pack time)
  FlyDSL/<op>/*.json              descriptors shared by every arch of the op
  FlyDSL/<op>/gfx11-generic/      the pack (KDP), one `hsaco` UKD per object,
                                  the objects, manifest.json, SOURCE.md;
                                  `arch` lists all eight members
        │
        │  shared packer (descriptor-packaging), product root
        ▼
lib/hipdnn_plugins/engines/arch_content/hip-kernel-provider/<arch>/   one per member
  FlyDSL/…                        descriptors, rewritten to kind "kpack"
  kpack/hip_kernel_provider_<arch>.kpack   every producer's objects, one archive
```

Each authored UKD is `kind: "hsaco"` and names its object by file name. The
packer resolves the object, packs it into the archive of every arch the UKD
lists, reads the argument signature out of the object and rewrites
the UKD to `kind: "kpack"` — the same path a rocKE kernel takes after it is
compiled. The shard is installed with everything else under
`arch_content/hip-kernel-provider/`, which the plugin finds beside its own
module, so an installed plugin loads FlyDSL's kernels with **no environment
variable** and nothing FlyDSL-specific at runtime.

The ingestor reads the shard's descriptors to learn what kernels exist and what
they cost, and calls into the four native symbols each pack registers —
`hipkernel.flydsl_<op>.{graph_match,kernel_match,score,dispatch}` — to match,
rank and launch. No FlyDSL, no Python and no compiler is present at runtime;
by then these are ordinary code objects in an archive.

The descriptors are **generated from the objects they describe**
(`gen_descriptors.py`, from each `manifest.json`), so a descriptor cannot drift
from its kernel. The build re-runs `gen_descriptors.py --check` before the pack
-- every object must match its manifest's SHA256 and be built for the target its
directory names -- and `tools/check_shards.py` after it, which shows every
member's shard ships exactly the checked-in objects.

## Building it

Two flags, and the difference between them matters:

- **`HIPKERNELPROVIDER_ENABLE_FLYDSL`** — the option. States *intent*. Requires
  `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`; ON without it is a configure-time
  `FATAL_ERROR` rather than a silently inert build.
- **`HIPKERNELPROVIDER_FLYDSL_ACTIVE`** — an internal cache variable. States
  *outcome*, raised by `flydsl/CMakeLists.txt` only when the product pack
  ships FlyDSL objects for a requested architecture. Everything conditional
  keys on **this one**.

The gap between them is real: with `GPU_TARGETS=gfx942` the option is ON and yet
nothing ships, because no gfx942 kernels are checked in. That configuration
reports **dormant** and continues — it is not an error to build for an
architecture this directory does not serve. Keying the engine TUs, the test TUs
and the census on the outcome rather than the intent is what makes that
configuration compile.

```bash
cmake -B build -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPKERNELPROVIDER_ENABLE_FLYDSL=ON \
      -DHIPDNN_ENABLE_SDPA=ON
cmake --build build -j
ctest --test-dir build -R hip-kernel-provider
```

`HIPDNN_ENABLE_SDPA` is hipDNN's own switch for attention, OFF by default; it
gates `graph->sdpa()` in the frontend. The SDPA pack builds and stages without
it, but no graph can then contain an SDPA node to reach it.

`HIPKERNELPROVIDER_ENABLE_FLYDSL` shapes exactly this integration's surface —
one engine TU and one test TU per op, one census suite per op, and the `FlyDSL/`
family folder, which the packer excludes with the option OFF
(`HKP_DESCRIPTOR_FAMILIES`). No other integration's flag changes
anything FlyDSL compiles, registers or asserts, and no FlyDSL suite names another
engine. That isolation is a property to preserve, not an accident: it is checked
by building with `ENABLE_HIP_MLOPS_ENGINE=OFF` and diffing the FlyDSL case list,
which must come out empty.

## File map

`<content>` is `src/engines/kernel_ingestor_engine/descriptors/FlyDSL/` in the
provider tree; everything else is relative to this directory.

| Path | |
|---|---|
| `<content>/<op>/*.json` | descriptors shared by every arch of an op: schema, engine, heuristic, dispatch, matcher |
| `<content>/<op>/<arch>/*.hsaco` | the committed code objects |
| `<content>/<op>/<arch>/*.{kdp,ukd}.json` | the arch's pack and one `hsaco` UKD per object |
| `<content>/<op>/<arch>/manifest.json` | per-object SHA256 + knobs + semantic argument names + the toolchain that built them |
| `<content>/<op>/<arch>/SOURCE.md` | generated provenance record |
| `kernels_src/kernels/**` | vendored kernel sources; each file's header names its upstream (FlyDSL or AITER), commit and path |
| `generators/_instances.py` | **the instance table** — one row per shipped object, every op |
| `generators/_flydsl_env.py` | the pins: wheel version, upstream commits, ROCm recording |
| `generators/_codeobject.py` | reads and verifies a compiled object (target, ELF generic machine), shared by every generator |
| `generators/arch_families.json`, `_arch_families.py` | the generic targets and their member arches; read by the generators and `CMakeLists.txt` |
| `generators/_generic_targets.py` | the FlyDSL 0.3.4 shim that lets MLIR lower for a generic target |
| `generators/gen_rmsnorm.py`, `gen_sdpa.py` | compile one op's rows into objects + manifest + `SOURCE.md` |
| `gen_descriptors.py` | derives descriptors from the objects; `--check` re-verifies |
| `tools/diff_upstream.py` | vendored sources vs. their upstream checkouts, black-normalized |
| `tools/check_shards.py` | after the pack: every member's shard ships exactly the checked-in objects |

Adding an instance is a **row in `_instances.py`** plus a regeneration — the
compiler, the packer and the descriptor emitter all read that table, so there is
no second place that can disagree with it.

## The provenance chain

Each link is checkable by a command, and [REGEN.md](REGEN.md) gives each command:

```
upstream FlyDSL @ 89ad52fbbb9e, AITER @ 8253efc40595
   │  tools/diff_upstream.py          — vendored copy == upstream, but for recorded modifications
kernels_src/
   │  gen_<op>.py (flydsl 0.3.4)      — byte-reproducible: currently 52/52 identical
<content>/<op>/<arch>/*.hsaco + manifest.json
   │  gen_descriptors.py --check      — objects match manifest SHA256 and target; descriptors agree
<content>/<op>/**/*.json (hsaco UKDs)
   │  shared packer, then tools/check_shards.py — every member shard ships those bytes
arch_content/hip-kernel-provider/<arch>/
   │  TestFlydsl*Packs + census       — the shard is really there, and is what the suite read
```

The last link is the one that is easy to leave out and expensive to be without:
a test suite that builds its graphs in-process stays green against an **empty**
shard. The census exists to fail in exactly that case.

Regeneration is byte-reproducible against the pinned toolchain, which is what
lets the whole chain be re-derived rather than trusted. Two consequences worth
stating plainly:

- Objects that regenerate to **different bytes** are a *change to what this
  provider ships*, not a refresh — commit them as such, with objects, manifest,
  `SOURCE.md` and descriptors moving together. A regenerated object committed as
  a refresh reviews as a no-op diff while changing what every user runs.
- The repo's `black` hook reformats `kernels_src/`. That is accepted, and the
  reformat was verified codegen-neutral by regenerating — so re-run the
  verification after any bulk reformat, because that verification is the only
  thing keeping the acceptance honest.
