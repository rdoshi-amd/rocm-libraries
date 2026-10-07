# Case Study: fp8 GEMM global-load width on gfx1250

**Goal.** The fp8e4m3 A/B → bf16 C universal GEMM emitted 8-byte global loads
for A and B where the hardware permits 16. Widen them, and establish whether
that is worth shipping.

**Scope of what is measured here.** gfx1250, wave32, 16×16×64 WMMA atom,
fp8e4m3 A/B into bf16 C, layout RCR, `pipeline=mem` / `scheduler=intrawave` /
`epilogue=default`. Two problem shapes: 4096×4096×8192 (compute-bound by
arithmetic intensity) and 64×16384×4096 (the memory-bound counterpart). The
baseline arm is this same kernel with the load width pinned to its previous
value, built from the same source tree in the same binary — not a different
library and not a different commit. All timings are HIP-event-timed
(20 iterations, 5 warmup). **Only ratios are reported**; absolute figures are
kept in the internal record, per `platform/AGENTS.md` §Compliance.

**Result.** At the measured configurations the wider load is faster, and the
margin depends strongly on which tile geometry it is paired with. It is
shipped on by default.

---

## 1. The change

One argument. `helpers/spec.py::choose_load_vec` caps the vector at
`16 // elem_bytes` — the 4-dword `buffer_load` limit — and
`gemm_universal._choose_load_vec` never passed `elem_bytes`. The picker's
default of 2 therefore capped a **1-byte** operand at 8 elements, spending
half of every A/B load:

```
before:  load <8  x i8>, ptr addrspace(1) ..., align 8
after:   load <16 x i8>, ptr addrspace(1) ..., align 16
```

The fix resolves the width from `data.dtype_a` via `_ab_dtype_bytes`, which
already existed. `TraitSpec.ab_load_elem_bytes: Optional[int] = None` overrides
it; `None` resolves from the dtype, an explicit `2` reproduces the older
emission. The knob exists so both widths can be built in one binary and
compared — which, per §3, turned out to be the only trustworthy way to measure
anything on this machine.

**Blast radius is narrow.** f16/bf16 resolve to 2 either way, so every 2-byte
kernel is byte-identical. Only 1-byte operands move. The C++ picker's
hardcoded `{8,4,2,1}` candidate ladder (permanently `elem_bytes=2`) was
replaced with `16/elem_bytes`; the conv and MoE adapters pass `2` explicitly so
their emission is unchanged.

---

## 2. What we got wrong first

Three times, and each mistake is more instructive than the fix.

**(a) The shape prediction was backwards.** An arithmetic-intensity analysis
said 4096×4096×8192 sits far above the roofline ridge (AI reduces to the
harmonic mean of M and N when K is large, so a square shape is compute-bound),
therefore a memory-side lever should do nothing there and should pay off on a
skinny shape. The opposite was observed: a clear effect on the square shape,
flat on the skinny one. AI correctly predicted which side of the ridge the
shape sits on; it did not predict that load *width* would matter there anyway.

**(b) A cross-run comparison manufactured a regression that did not exist.**
The first evaluation compared a 16-byte ladder run against an 8-byte ladder run
from hours earlier and concluded the change was ~5% slower. It was not — the
environment had changed between the two runs (§5). The sign of the result was
wrong, not just the magnitude.

**(c) The first explanation for (b) was also wrong.** The drift was attributed
to a memory-clock drop from 2400 to 1900 MHz, with a roofline argument built on
top. A machine-config snapshot taken days earlier showed `pp_dpm_mclk` offering
**one** level, 1900 MHz, identical to the current state — the 2400 figure was
SCLK. There had been no clock change, and the real cause (§5) was in the
software stack, not the hardware. The lesson is narrow and worth stating: a
plausible mechanism that explains the size of an effect is not evidence that
the mechanism occurred.

---

## 3. Method: A/B must be interleaved in one process

This is the durable finding.

Measuring arm A, then arm B, then comparing, is invalid on this machine at the
scale of effect being chased. An identical kernel — same spec, same build path,
same measurement code — was re-measured hours apart and differed by more than
any effect in this study.

The experiment that settled the question builds all arms once and times them
**round-robin inside a single process**, cold round discarded, repeated with
the arm order alternating per round to rule out an ordering effect. Whatever
degrades the machine then hits every arm equally, and the ratio stays valid
even though the absolute level does not.

A second trap sits in the harness itself. The sweep's `lever` stage A/Bs arms
**on the champion geometry the search found** — and that search ran under one
arm's emission. When a lever changes which geometry is optimal, as this one
does, that answers "what does the width do to *this* geometry", not "is the
best achievable configuration better". Those differ by a large factor here
(§4). Both are true statements; only the second answers "should we ship it".

---

## 4. Results (ratios, 4096×4096×8192)

Two geometries × two widths, interleaved, two independent runs:

| geometry | 16-byte vs 8-byte |
|---|---|
| `tile 256×128×128, warp 4×2` | 1.015× – 1.032× |
| `tile 128×128×128, warp 2×2` | **1.586× – 1.589×** |

**Best achievable vs best achievable: 1.074× – 1.088×.**

Both runs agree within ~1% on every arm but one (4.5% spread). The width helps
on both geometries and helps substantially on the one the 16-byte search
selects — which is why the single-champion A/B in §3 reports ~1.59× while the
honest best-vs-best figure is ~1.08×. Quote the latter.

At 64×16384×4096 the lever measured 0.9867×, i.e. flat within a ~2–3% spread.
These are bounded claims about these measured configurations on this target;
no general superiority is implied.

---

## 5. The toolchain changed underneath the measurements

An identical spec, re-measured through the identical code path used for the
original figure, came back **~10–15% slower** hours later. Three explanations
were considered and two were wrong.

*Not thermal*: junction 45 °C, memory 34 °C. *Not clocks*: the DPM tables were
byte-identical to a snapshot taken days earlier (§2c). *Not another tenant*:
`rocm-smi --showpids` and `/sys/class/kfd/kfd/proc/` both report **zero**
compute processes attached to the device.

One thing that definitely changed is the compiler. The ROCm installation that
`$ROCM_PATH` resolves to was repointed to a newer toolchain partway through the
day, and `/proc/<pid>/maps` confirms the kernel-build process now loads its
`libamd_comgr` from that new tree. So every figure recorded before the switch
was produced by one compiler and every figure after it by another.

**But the compiler is not the whole story, and assuming it was would have been
the third wrong explanation.** The toolchains can be A/B'd properly: emit the
LLVM IR once (identical input, hash-checked), compile it under each `comgr` in
separate processes, then time the resulting blobs round-robin in a single
process under one HIP runtime. Done that way the newer toolchain is **0.5–2.3%
slower** on these two kernels — and replaying one fixed blob under each
`libamdhip64` shows **no measurable runtime effect** at all. The generated ISA
differs only in scalar branch lowering: identical instruction count, identical
VGPR and LDS, nothing touched in the vector, MMA or memory stream.

That leaves most of the gap unattributed. The same spec built with the *same*
compiler still measures materially slower than it did before the update, and
the remaining differences in the environment — notably a driver version change
— cannot be tested without reinstalling. Cross-process measurement noise here
is around 4–5%, which absorbs part of the gap but not all of it.

The honest position is therefore: the environment changed in at least two ways,
one of them measurable and small, and **a pre-update number cannot be compared
with a post-update number** regardless of which part dominates. That is the
operative lesson, and it does not depend on resolving the cause.

Consequences, which apply to any future work in this folder:

- Ratios from a single interleaved process are sound. Absolute levels are not
  comparable across sessions, and a toolchain switch makes them incomparable
  even on an otherwise idle, thermally stable, clock-stable machine.
- A stored champion from an earlier ladder is **not** a valid baseline. Re-run
  the ladder in full, or interleave.
- **The HSACO cache key does not include the toolchain.** `sweep._spec_hash`
  covers the spec and the arch, so a cache populated before a toolchain change
  will happily serve stale objects to a run after it. Separate output
  directories per run avoided this here, but a resumed run (`--stages`)
  spanning an update would silently mix two compilers' output.
- The harness records no environment alongside its results. It should capture
  the resolved ROCm path and comgr version, clocks, temperature and
  utilisation at the start and end of every run. Until it does, a code effect
  and a toolchain that moved look identical in the output — which is exactly
  how §2(b) happened.

---

## 6. Correctness

Every arm in every comparison passed the exact numeric gate **in the measured
run** — `max_abs_diff = 0.0`, not a tolerance. Inputs are integers in −5..5,
each exact in e4m3; with `|sum| ≤ 25·K` the fp32 accumulator is exact for any
K below ~670k, so summation order cannot change the result and only the final
RNE to bf16 rounds, which the reference reproduces. The harness refuses a K
that would break the argument rather than widening the gate.

Dual-engine byte-identity holds: 69 GREEN / 1 DRIFT, the DRIFT being a
documented pre-existing failure in an unrelated attention family. Parity
configs 11 (resolved width) and 12 (pinned width) are both byte-identical
between the Python and C++ engines, so the knob is covered in both states.

---

## 7. Levers tried, and one that is not available

Swept: tile geometry (`tile_m/n/k`, `warp_m/n`), `lds_k_pad`, `lds_swizzle`,
`waves_per_eu`. Of these only `lds_k_pad` and the tile geometry moved the
result materially; `waves_per_eu` spans under a percent at the champion.

**`direct_to_lds` is not a lever on this path.** Every combination setting it
is rejected with *"WMMA path does not support direct_to_lds on gfx1250"* — 80
of 120 trait candidates per finalist died on that one line before it was
removed from the config. Listing it implied a knob the kernel does not have.

**Not attempted:** an ISA/occupancy inspection of why the wider load helps. The
plausible mechanisms (VGPR pressure from a 4-register vector, a halved
`a/b_buffer_load_inst_num` changing the `sched_group_barrier` interleave) are
unverified and are deliberately not asserted here.

---

## 8. The widening changed what a good `lds_k_pad` is

`lds_k_pad` is in **elements**, not bytes — `_lds_k = block_k + lds_k_pad`, and
the LDS row stride in bytes is that times the element width. Bank conflicts are
a byte-level effect (32 banks × 4 B = a 128 B cycle), so **the same numeric pad
means something different for a 1-byte operand than for a 2-byte one**. The
values here were inherited from a bf16 sweep and were wrong for fp8 three ways.

**They sampled half the space.** At `tile_k=64`, bf16 pads of
`[0, 8, 24, 40, 56]` give bank offsets `{0, 16, 48, 80, 112}`. The same numbers
on fp8 at `tile_k=128` give `{0, 8, 24, 40, 56}` — only the lower half of the
bank range is explored.

**The winner sat at the top of the range.** Every top-five finalist took the
largest value in the list. A winner at the edge of a search range is the
signature of a truncated search, not of an optimum.

**And the widening made all the non-zero values misaligned.** This is the part
created by this change. Phase 1 made the LDS store `<16 x i8>`, so a row stride
that is not a multiple of 16 puts alternate rows off a 16-byte boundary. Every
non-zero pad in the inherited list is ≡ 8 (mod 16). The emitter claims
`align 16` on that store **regardless of the pad**:

```
lds_k_pad=56  (row 184 B, 184 % 16 = 8):   store <16 x i8> ... align 16
```

Before the widening the store was `<8 x i8>` and 184 % 8 = 0, so the claim held.
It is now untrue for odd rows. Every measured arm still passed the exact numeric
gate, so there is no observed miscompile — the hardware tolerates the access —
but an alignment attribute is an assertion to LLVM, not a formatting detail, and
a different compiler version is entitled to act on it.

### The good pads can be derived instead of swept

LDS has 32 banks of 4 bytes. Row `r` of the staged tile begins at bank
`(r·S/4) mod 32` for a row stride of `S = tile_k + pad` bytes, so the per-row
bank stride is `(pad/4) mod 32` and the number of distinct banks the rows land
on is `32 / gcd(stride, 32)`. More distinct banks is better; one bank is a
full 32-way conflict.

Swept interleaved at the champion geometry, every class the model predicts
shows up, and the two values picked *because* the model predicted them —
32 and 64 — behave as predicted:

| pad | distinct banks | 16B-aligned | measured, relative to best |
|---:|---:|:--:|---:|
| 0 | 1 | yes | 3.38× worse |
| 16 | 8 | yes | 1.01× |
| 32 | 4 | yes | 1.32× worse |
| 48 | 8 | yes | 1.02× |
| 56 | 16 | **no** | 1.24× worse |
| 64 | 2 | yes | 1.81× worse |
| 80 | 8 | yes | **1.00×** |
| 96 | 4 | yes | 1.28× worse |
| 112 | 8 | yes | 1.04× |

Three things follow.

**The pathology is the bank offset, not the number zero.** `pad = 128` gives
the same offset 0 as `pad = 0` and is just as catastrophic (worse, in fact —
it also costs LDS). So the space is periodic in 128 bytes for a 1-byte operand,
which *bounds* the search: there is nothing useful beyond 112.

**The good set is computable.** Wanting `gcd((pad/4) mod 32, 32) == 4` (eight
distinct banks) and `pad % 16 == 0` (so the `<16 x i8>` store's alignment claim
is true) yields exactly `[16, 48, 80, 112]` — which is what the configs here
sweep, plus `0` kept as the known-bad anchor that makes the others
interpretable. Not as the screening pin, though (§9); that is 48.

**Banking and alignment are in tension, and alignment wins.** Sixteen distinct
banks would need `gcd == 2`, i.e. `pad ∈ {8, 24, 40, 56, 72, 88, 104, 120}` —
*every one* of which is ≡ 8 (mod 16) and therefore misaligns the 16-byte store.
`pad = 56` is the measured proof: best-in-class banking, worst-in-class
alignment, and it lands 24% behind `pad = 80`. That is also an independent
confirmation of the alignment finding above, reached from a different
direction, and it explains why 56 won the inherited sweep — the aligned values
simply were not in it.

The practical upshot for anyone porting this list to another operand width:
**do not copy the numbers, recompute them.** `pad` is in elements, so the whole
table shifts with the element size.

**Owed:** the emitter should derive the store's alignment attribute from the
actual row stride rather than asserting the vector width, so a misaligned pad
produces a weaker-but-true claim instead of a stronger false one.

## 9. The screening pin nearly hid the answer

Covered in full in [README.md](README.md); summarised because it shaped every
number above. Stage 1 ranks tile geometry with traits held at
`screening_traits`. Pinned at `lds_k_pad = 0`, the configuration that went on to
win ranked **18th of 180**, two places above the `tile_finalists = 20` cutoff,
and **no** `tile_k = 256` geometry reached the finalists at all (best rank 43).
Re-screened at a non-zero pin, the ranking inverted:

| `tile_k` | top-20 at pin 0 | top-20 at pin 56 |
|---|---:|---:|
| 64 | 17 | 2 |
| 128 | 3 | 14 |
| 256 | 0 | 4 |

The search was fragile, not wrong — after stage 2 tuned traits, both screens
finished within noise of each other. But a cutoff two ranks lower would have
silently reported a worse configuration as the answer, with nothing in the
output indicating anything had been missed.

---

## 10. Remaining work

1. Re-measure once the machine is healthy, and confirm the §4 ratios hold at a
   stable operating point. The §4 numbers were taken with the old, half-range
   `lds_k_pad` list (§8), so the re-run should also establish whether the new
   16-aligned values move the champion.
2. Derive the LDS store's alignment attribute from the actual row stride
   instead of asserting the vector width (§8), so a misaligned pad cannot
   produce a false `align 16`.
3. Teach the harness to record machine state per run (§5).
4. Make the `lever` stage refuse to print a ratio when an arm's own champion
   differs from the baseline's, rather than printing a misleading one (§3).
5. Investigate *why* the wider load helps, before generalising it to other
   1-byte paths — notably `moe_gemm_fused`, which still passes the default
   width and would inherit the same halving.
6. The DirectToLDS load path in the C++ engine hardcodes 2 bytes per element
   (`a_half_bytes = block_m * block_k * 2`), which is a separate latent fp8
   bug in the same area. Out of scope here; not yet filed.

---

## 11. Artifacts

- Harness, configs and the staged-sweep design: this folder, see
  [README.md](README.md).
- Kernel change: `instances/common/gemm_universal.py` (`_choose_load_vec`,
  `TraitSpec.ab_load_elem_bytes`) and its C++ mirrors in
  `cpp/helpers/spec.cpp`, `cpp/instances/common/gemm_mma.cpp`.
- Parity coverage: `tests/instances/parity/gemm_emit.{py,c}` configs 11 and 12.
- Lowering assertions, including the counterfactual that the override still
  produces the narrow load:
  `tests/instances/test_gemm_universal_fp8_lowering.py`.
- Measured results are kept in the internal record, not in this repository.
