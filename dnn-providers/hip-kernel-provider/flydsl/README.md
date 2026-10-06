<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier:  MIT
-->

# FlyDSL kernels in the hip-kernel-provider

Ahead-of-time RMSNorm kernels written in [FlyDSL](https://github.com/ROCm/FlyDSL),
compiled once and **checked in**, reaching hipDNN through the kernel ingestor.

This directory holds the kernels, the generators that produce them, and the
descriptors that make them selectable. The matching C++ — the code that decides
whether a graph may run on them and how to launch them — lives one level up, at
[`src/engines/kernel_ingestor_engine/packs/FlydslRmsNormNative.cpp`](../src/engines/kernel_ingestor_engine/packs/FlydslRmsNormNative.cpp).

**Nothing here is compiled by the build.** The `.hsaco` objects are committed
artifacts; CMake packs and stages them. That is the central fact about this
directory, and it is what the rest of this file is mostly about: an artifact
whose build is not reproduced on every `ninja` has to carry its provenance and a
way to check it, or it is just bytes someone once vouched for.

## Where to go

| | |
|---|---|
| **What is covered, and what is refused** | [COVERAGE.md](COVERAGE.md) |
| **Rebuilding or verifying the kernels** | [REGEN.md](REGEN.md) |

## What ships

One op, one architecture, 12 kernel objects:

- **RMSNorm forward**, bf16 and f16, on **gfx1151**
- a **specialized** tier with `N` baked in at 3072 / 3584 / 4096 / 5120 / 8192
- a **generic** tier that reads `N` from the tensor descriptor at runtime, so
  every other width is served too — more slowly, but served

Widths outside the specialized list are a **performance** boundary, not a
coverage one. The things that are genuinely out of scope — backward, training
phase, bias, other dtypes, non-packed operands — are enumerated in
[COVERAGE.md §3](COVERAGE.md), each with the reason the shipped objects cannot
compute it.

## How it reaches hipDNN

```
kernels/gfx1151/rmsnorm/*.hsaco   ──pack.py──>  …/gfx1151/kpack/*.kpack
descriptors/gfx1151/*.json        ──copy────>  …/gfx1151/*.json
                                                └─ flydsl_arch_content/hip-kernel-provider/
```

The ingestor discovers the staged shard, reads the descriptors to learn what
kernels exist and what they cost, and calls into the four native symbols the
pack registers — `hipkernel.flydsl_rmsnorm.{graph_match,kernel_match,score,dispatch}` —
to match, rank and launch. No FlyDSL, no Python and no compiler is present at
runtime; by then these are ordinary code objects in an archive.

The descriptors are **generated from the objects they describe**
(`gen_descriptors.py`), so a descriptor cannot drift from its kernel. The packer
packs from `manifest.json` rather than a directory glob and re-verifies each
SHA256 against the bytes it writes, so a stray or half-written object cannot
enter the archive.

## Building it

Two flags, and the difference between them matters:

- **`HIPKERNELPROVIDER_ENABLE_FLYDSL`** — the option. States *intent*. Requires
  `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`; ON without it is a configure-time
  `FATAL_ERROR` rather than a silently inert build.
- **`HIPKERNELPROVIDER_FLYDSL_ACTIVE`** — an internal cache variable. States
  *outcome*, raised by `flydsl/CMakeLists.txt` only once an arch intersection
  actually succeeded. Everything conditional keys on **this one**.

The gap between them is real: with `GPU_TARGETS=gfx942` the option is ON and yet
nothing stages, because no gfx942 kernels are checked in. That configuration
reports **dormant** and continues — it is not an error to build for an
architecture this directory does not serve. Keying the engine TU, the test TU
and the census on the outcome rather than the intent is what makes that
configuration compile.

```bash
cmake -B build -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON -DHIPKERNELPROVIDER_ENABLE_FLYDSL=ON
cmake --build build -j
ctest --test-dir build -R hip-kernel-provider
```

`HIPKERNELPROVIDER_ENABLE_FLYDSL` shapes exactly this integration's surface —
one engine TU, one test TU, one census pack. No other integration's flag changes
anything FlyDSL compiles, registers or asserts, and no FlyDSL suite names another
engine. That isolation is a property to preserve, not an accident: it is checked
by building with `ENABLE_HIP_MLOPS_ENGINE=OFF` and diffing the FlyDSL case list,
which must come out empty.

## File map

| Path | |
|---|---|
| `kernels/<arch>/rmsnorm/*.hsaco` | the committed code objects |
| `kernels/<arch>/rmsnorm/manifest.json` | per-object SHA256 + knobs + the toolchain that built them |
| `kernels/<arch>/SOURCE.md` | generated provenance record |
| `descriptors/<arch>/*.json` | what the ingestor reads: kernels, matchers, dispatch, heuristics |
| `kernels_src/kernels/**` | vendored FlyDSL kernel sources, pinned to one upstream commit |
| `generators/_instances.py` | **the instance table** — one row per shipped object |
| `generators/_flydsl_env.py` | the pins: wheel version, upstream commit, ROCm recording |
| `generators/gen_rmsnorm.py` | compiles the table into objects + manifest + `SOURCE.md` |
| `gen_descriptors.py` | derives descriptors from the objects; `--check` re-verifies |
| `pack.py` | build step: manifest → `.kpack`. Its output is not committed |
| `tools/diff_upstream.py` | vendored sources vs. an upstream checkout, black-normalized |

Adding an instance is a **row in `_instances.py`** plus a regeneration — the
compiler, the packer and the descriptor emitter all read that table, so there is
no second place that can disagree with it.

## The provenance chain

Each link is checkable by a command, and [REGEN.md](REGEN.md) gives each command:

```
upstream FlyDSL @ 89ad52fbbb9e
   │  tools/diff_upstream.py          — vendored copy == upstream, but for recorded modifications
kernels_src/
   │  gen_rmsnorm.py (flydsl 0.3.4)   — byte-reproducible: currently 12/12 identical
kernels/<arch>/*.hsaco + manifest.json
   │  gen_descriptors.py --check      — descriptors agree with the objects they name
descriptors/<arch>/*.json
   │  pack.py                         — archive re-verified against the manifest
staged shard
   │  TestFlydslRmsNormPack + census  — the shard is really there, and is what the suite read
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
