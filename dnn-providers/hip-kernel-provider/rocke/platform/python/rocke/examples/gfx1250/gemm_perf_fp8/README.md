# gemm_perf_fp8 — staged sweep + lever A/B for the gfx1250 fp8 GEMM

A replayable harness for the fp8e4m3 A/B → bf16 C universal GEMM on gfx1250.
It does two jobs that are usually conflated:

1. **Search** — find a good tile geometry and trait set for a shape.
2. **Attribution** — A/B one change against that champion, so "did this
   optimization help" is answered independently of "what is the best config".

Job 2 is the reason the harness exists. A sweep that only reports a winner
cannot tell you whether the thing you just added contributed anything.

> **Looking for what this harness found, rather than how to drive it?**
> [`gemm_perf_fp8_case_study.md`](gemm_perf_fp8_case_study.md) is the first
> optimization taken through it end to end — the global A/B load width. Read
> its §2 and §3 before running your own A/B here: they cover three wrong
> conclusions this harness produced along the way, and why an A/B on this
> machine is only valid interleaved inside a single process.

## Why it is staged

A full cross product of geometry × traits is tens of thousands of builds, and
nearly all of them re-answer a question the cheap stage already settled. The
stages narrow the candidate set while *increasing* rigor:

| Stage | Varies | Verification | Timing |
|---|---|---|---|
| `tile` | tile / warp geometry, traits pinned | top `tile_finalists` only | in-process, `attempts` samples |
| `trait` | the full trait space over verified geometries | none — screening only | in-process, `attempts` samples |
| `final` | nothing | every candidate, in rank order | **fresh process**, `final_samples` + 1 (first discarded) |
| `lever` | one registered lever per arm | every arm | **fresh process**, same as `final` |

On the shipped `configs/sq4k.json` that is 180 geometry candidates plus 800
trait candidates plus 1 lever arm — **981 builds against a full cross product
of 7,200**, a 7.3× saving, with the expensive, trustworthy measurement spent
only on the handful that survive.

The warp grid is `[2, 4, 8]` on both axes. `8×8` is not in the 180 because
`block_size 2048 > 1024 (hardware cap) on gfx1250` rejects it; `8×4` and
`4×8` sit exactly at the cap and are kept, because a 256×256 tile at `2×2`
puts 64 WMMA tiles on each warp. Warp `1` is excluded for this shape — which
also excludes `1×2`, `2×1`, `1×4` and `4×1`. That is deliberate for a square
problem and should be **reconsidered for skinny shapes**, where few M-tiles
make a `1×N` warp grid plausible rather than degenerate.

Rigor being inverse to candidate count is deliberate. Flattening it — running
everything carefully — costs ~10× more and yields no extra information,
because the screening stages exist to *rank*, not to report.

## The screening pin is load-bearing

`screening_traits` in the config decides what stage 1 holds fixed while it
ranks geometry. **A bad pin silently deletes the winner.** If a pinned value
happens to collapse performance for one region of the tile space, every tile
in that region ranks badly in stage 1, never becomes a finalist, and is never
seen again — the trait that would have rescued it is only swept in stage 2,
over candidates stage 1 already chose.

This is not hypothetical, and it is not inherited folklore — it was measured
on this path. The first full sweep of this config pinned
`screening_traits.lds_k_pad = 0`, and the configuration that went on to win
the whole search **ranked 18th of 180 in stage 1, two places above the
`tile_finalists = 20` cutoff.** With the pin at 0, stage 1 ranked geometries
in an order that systematically penalised a large `tile_k`:

| `tile_k` | best stage-1 rank | finalists in the top 20 |
|---|---:|---:|
| 64 | 1 | 17 |
| 128 | 11 | 3 |
| 256 | 43 | **0** |

Stage 2 then swept padding and a `tile_k = 128` geometry beat every
`tile_k = 64` candidate decisively. Had `tile_finalists` been 15 instead of
20, the sweep would have reported the best `tile_k = 64` config as its answer
and never discovered otherwise — not as a visible failure, but as a lower
number with no indication anything had been missed.

Re-screening the same 180 geometries with the pin at 56 inverted the ranking,
confirming the pin — not the hardware — was deciding it:

| `tile_k` | top-20 at pin 0 | top-20 at pin 56 |
|---|---:|---:|
| 64 | 17 | 2 |
| 128 | 3 | 14 |
| 256 | 0 (best rank 43) | 4 (best rank 5) |

The winning geometry changed too. What did *not* change was the end result:
after stage 2 tuned traits, both runs finished within noise of each other.
The bad pin made the search **fragile — two ranks from discarding its own
winner — without making it wrong**, because the single good geometry that
survived was enough. That is worth stating precisely, because "the pin nearly
lost it" and "the pin lost it" call for the same fix but very different
confidence in any result already collected.

Three conclusions worth carrying:

- **`lds_k_pad = 0` is a bad screening pin for this path.** Padding is the
  single most influential trait on the winning geometry, and 0 is its worst
  value by a wide margin. The shipped config now pins 56. The bf16 harness
  this is ported from had the same hazard and dealt with it by *ordering* its
  search list so the collapsing value was not first — a fix that works and
  explains nothing.
- **A finalist cutoff is a guess about a ranking you have not validated.**
  If stage 1 eliminates every large tile, suspect the pin before concluding
  large tiles are bad, and re-run `--stages tile` with a different
  `screening_traits` — 180 builds, no trait-stage cost.
- **Re-screening is how you find out, and it is cheap.** Re-running stage 1
  under a second pin and comparing the finalist composition costs a fraction
  of the ladder and converts "this result might be an artifact of the screen"
  into either a correction or a genuinely stronger result — two searches that
  disagree on geometry but agree on the outcome say more than one search.

## Correctness is exact, not a tolerance

Inputs are integers in −5..5, each exact in e4m3. With `|sum| ≤ 25·K` the
fp32 accumulator is exact for any K below ~670k, so summation order cannot
change the result; the only rounding left is the kernel's final RNE to bf16,
which the reference reproduces. The comparison is therefore `err > 0.0`.

`PreparedProblem.create` refuses a K that breaks the argument rather than
quietly widening the gate. A tolerance here would downgrade the check without
changing any output — the failure mode an exact gate exists to prevent.

Every lever arm is verified, not just the baseline: an arm that got faster by
computing the wrong thing is precisely what an A/B harness is for.

## Results go outside the repository

`--out` (or `$ROCKE_PERF_OUT`) is required and is deliberately **absent from
the committed configs**. A tracked config carries the search space; it does
not carry a path to somebody's results directory. Per `platform/AGENTS.md`
§Compliance, measured numbers do not belong in the repo, in git history, or
in a PR — keep them in the approved internal record.

The `lever` stage additionally reports a **ratio against the baseline arm**.
That ratio, not the absolute timings, is the form that may appear in a PR
description or a qualifying benchmark document.

## Running it

```sh
export PYTHONPATH=<platform>/python
export ROCKE_PERF_OUT=<somewhere outside the repo>
export ROCKE_LLVM_FLAVOR=llvm23          # local ROCm is 10.0

# everything
python3 -m rocke.examples.gfx1250.gemm_perf_fp8.fp8_gemm_sweep

# one stage at a time; later stages read earlier stages' JSON from --out
python3 -m ....fp8_gemm_sweep --stages tile
python3 -m ....fp8_gemm_sweep --stages trait,final
python3 -m ....fp8_gemm_sweep --stages lever --levers baseline,phase1_vec16

# a different shape, no new script
python3 -m ....fp8_gemm_sweep --m 64 --n 128256 --k 4096
```

Every config key has a CLI override; `--help` lists them. Because stages read
each other's JSON from `--out`, a run can be resumed after a crash, and a
single stage can be repeated with different settings against finalists that
are already chosen.

## Adding a lever

A lever is a named set of trait/data overrides in the config's `levers`
block. `baseline` is registered as a lever rather than special-cased, so every
arm goes through the same build, verification and timing path — a difference
between arms cannot be an artifact of how they were run.

```json
"levers": {
  "baseline":     { "description": "...", "trait": {}, "data": {} },
  "phase1_vec16": { "description": "...", "trait": {"<field>": <value>} }
}
```

Only `baseline` ships today. The Phase 1 lever needs a spec field that does
not exist yet — see below.

## Where Phase 1 actually lives

Confirmed from the emitted IR for a 128×128×64 fp8 tile:

```
%gv862 = load <8 x i8>, ptr addrspace(1) %gep.2, align 8
```

8-byte global loads for A and B, where fp8 permits 16. The cause is one
argument: `gemm_universal._choose_load_vec` calls
`helpers.spec.choose_load_vec(...)` **without** `elem_bytes`, so the helper
uses its default of 2 and caps the vector at `16 // 2 = 8` elements. For
1-byte operands that is 8 bytes, half the hardware's 4-dword maximum.

`choose_load_vec` already accepts `elem_bytes`, and `_ab_dtype_bytes(spec)`
already exists from Phase 0 — so the widening itself is small. Two notes
before anyone treats it as a one-liner:

- It changes emitted IR, so it needs the C++ mirror and its own parity config
  in the same change, and the lever must be a **defaulted** spec field
  (`AGENTS.md`: a new spec field defaults to the currently-shipped value) or
  every existing descriptor breaks.
- Blast radius is narrow: f16/bf16 already pass `elem_bytes=2`, which equals
  the default, so only 1-byte operands change. Parity configs 0–10 stay
  byte-identical; only the fp8 config moves.

## Shape choice, and what to expect at 4096×4096×8192

The shipped config is the Phase 0 verified shape. It is the right shape to
*build the harness on* and to establish a baseline, and probably the wrong
shape to *demonstrate Phase 1*: its arithmetic intensity is far above the
ridge, so it is compute-bound, and Phase 1 is a memory-side lever. A flat
lever result there is a valid measurement, not a failure.

For large K, arithmetic intensity reduces to the harmonic mean of M and N —
K cancels. Memory-bound fp8 GEMM therefore needs a small M with a large N,
not a large K and not a square. Adding such a shape is a config edit; the
ladder is unchanged.

## DirectToLDS is not a lever on this path

`direct_to_lds` and `dtl_prefetch` are absent from `trait_config` on purpose.
Every combination that sets them is rejected:

```
WMMA path does not support direct_to_lds on gfx1250
```

With them present, 80 of 120 trait candidates per finalist died on that one
line. They cost no builds either way, but a config that lists them implies a
knob the kernel does not have. If the WMMA path ever gains DirectToLDS, add
the two keys back and `_load_paths` will enumerate them again — it already
suppresses `dtl_prefetch` without `direct_to_lds`, which `is_valid_spec`
would reject anyway.

Similarly, `pipelines`, `schedulers` and `epilogues` are pinned to the single
supported value each (`mem` / `intrawave` / `default`) rather than listing
alternatives that this path does not accept.

## Known gaps

- **`elf_meta` is empty**, so there is no static VGPR/LDS/occupancy filter
  before stage 1 launches anything. `sweep._extract_elf_meta` shells out to
  `llvm-objdump` and returns `{}` when the parse fails, which it currently
  does even with the tool present. Wiring this up would be the cheapest way
  to shrink stage 1 further, since it costs no GPU time.
- **Crash isolation is coarser than the bf16 original.** A latched HIP error
  stops the stage and keeps the rows measured so far, rather than respawning
  a worker and continuing past the faulting candidate. Results are preserved
  either way; a faulting sweep just stops early and says so.
- **No static-probe stage yet.** The runbook's §3.1b tier is the natural
  fifth stage once `elf_meta` works.
