# Known issues: gfx1250 MXF4 subtile

## Correctness failures outside the common-test shapes (open, found 2026-09-30)

Reproduce on GPU 2 (found on GPU 3):

```bash
TENSILE_YAML=configs/regress_subtile_mxf4.yaml ./run.sh tensile
```

The config has the two kernels from `Tensile/Tests/common/gemm/gfx12/subtile_mxf4_gfx1250.yaml`
(TN, e8 scales, block 32, DepthU 256, PGR 2, StreamK 3, TDMInst 3), with beta = 1 and random C.
Sizes are (M, N, batch, K).

| Kernel | Size | Result |
| --- | --- | --- |
| MT128x64, MI 32x16, WT 2x2 | (128, 64, 1, 256), (128, 64, 1, 512), (256, 128, 1, 256), (96, 48, 1, 256), (128, 64, 1, 384) | pass |
| | (1000, 500, 2, 768) | wrong values, 7713 of 1000000 |
| | (128, 64, 3, 4096) | wrong values, 6141 of 24576 |
| | (200, 100, 1, 1152) | wrong values, 720 of 20000 |
| MT256x256, MI 32x16, WT 4x8 | (4096, 4096, 1, 8192), (256, 256, 1, 256), (512, 512, 1, 2048), (2048, 1024, 1, 384) | pass |
| | (1000, 700, 1, 1280) | wrong values, 25445 of 700000 |
| | (300, 200, 2, 1152) | wrong values, 13786 of 120000 |
| | (256, 256, 1, 8192) | illegal memory access (GPU hang, aborts the client) |

- Not caused by the 32x16 accumulator row fix-up (`emitWmma32x16AccRowFixup`). Tensile built
  from `e91596bfb48`, before that fix, has the same pass/fail result on every size. The count of
  wrong values differs slightly between the two builds.
- Every batch > 1 size fails. Some single-batch sizes with partial M/N tiles and K of 1152 or 1280
  also fail. Edge-only (96, 48, 1, 256) and tail-only (128, 64, 1, 384) sizes pass.
- The crash is one 256x256 tile with a long K, so StreamK 3 splits a single tile's K range
  across workgroups. The partial-tile fixup is the first suspect.
- Not yet checked: the same sizes with StreamK 0, and whether the non-subtile MXF4 kernel
  (`configs/ref_nonsubtile_mxf4.yaml`) passes them.

## Main-loop overhead vs the BF16 subtile kernel (open, found 2026-10-05)

Compared against BF16 MT256x256x128, PGR 2, StreamK 3, TDMInst 3 (from
`Tests/common/gemm/gfx12/subtile_bf16_gfx1250.yaml`). Per iteration of the current MXF4 loop:

- **TDM descriptors rebuilt from scratch, about 32 SALU per load, 4 loads.** The incremental
  update (`_emitGRPtrUpdate_TLU0`: address += DepthU bytes; `_tdmSwapLdsBuffer`: XOR the LDS
  address) already leaves the descriptor ready. `refreshTDMDescriptorSubtile`
  (`InstructionEmitter.emit_gr`) then re-runs the full init before every load, rewriting the
  control word, dims, tile sizes, stride and the per-wave row clamp (`v_readfirstlane` of
  Serial). It also writes the address fields a second time. This is needed today because the
  MXSA/MXSB descriptors alias the A/B descriptor SGPRs (`.set sgprtdmMXSAGroup0,
  sgprtdmAGroup0`). BF16 does the same rebuild without any aliasing.
- **The rebuild sits in a WMMA-free window.** Order: phase-1 WMMAs, 24 reads,
  `s_wait_dscnt 0`, barrier, about 128 SALU and 4 TDM loads with 2 WMMAs, barrier, 14 address
  swaps, `s_wait_alu depctr_va_vdst(0)`, 21 reads, then the next WMMA. BF16 spreads its
  descriptor SALU one or two per WMMA and keeps WMMAs running through both barriers.
- **12 A/B read-address VGPRs instead of 2, so 14 swaps per iteration.** A has one base per K
  read (8) and B one per K read (4), each with its own swap-mask VGPR, built in "LR Offset
  Calculation for Subtile Based Tiling" (`SubtileLREmit.py`). All of them get the same padding
  (pad block = row / 2 for K offsets under 128 B, +128 B for the 2048 B group), so they differ
  only by constants: A = base + {0, 32, 64, 96, 2176, 2208, 2240, 2272}, B = base + {0, 32, 64,
  96}. Folding these constants into the ds_load immediates would leave 4 swaps and free 20
  VGPRs. BF16 has the same pattern (4 bases per tensor).
- **The read gate bunches reads at the end of phase 1.** Ungated, phase 1 ends with reads
  paired with WMMAs and 7 trailing reads. With the gate it ends with 11 WMMAs and then 24 reads,
  so `s_wait_dscnt 0` before the barrier is exposed.

None of this is on the critical path today. `inject_delay.py` inserts `s_nop` cycles at one
point of both main-loop copies and reassembles (with code object v4 the unmodified round trip
is byte-identical). Extra cycles per iteration vs the unmodified kernel, 6-8 alternating
rounds on GPU 2 (a point fully on the critical path would cost about 17 us per 128 cycles):

| Delay point | +128 cycles | +512 | +1024 |
| --- | --- | --- | --- |
| After the first barrier (descriptor window) | +1.6 us | | |
| Before `s_wait_tensorcnt` | +0.3 us | -2.4 us | +30.3 us |
| After the second barrier (swaps) | +2.0 us | | |
| Middle of phase 1 | +1.8 us | -0.8 us | +31.6 us |
| Middle of phase 2 | +2.1 us | | |

Each wave has roughly 800 idle cycles per iteration (of about 2100), the same wherever the
delay goes. The loop is paced by something outside the waves' instruction stream, most likely
TDM data supply: 68 KiB per workgroup per iteration (A and B 32 KiB each, scales 2 KiB each)
is about 16.5 TB/s over 256 CUs at 1.08 us per iteration. Removing the overheads above only
pays off once that limit moves.

## LDS segment conflicts: nearly eliminated, no timing change (2026-09-30, finished 2026-10-05)

Kernel: MT256x256x256, MI 32x16x128, TDMInst 3, StreamK 3, size (4096, 4096, 1, 65536).
Counter: `TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES`, summed over the 256
CUs. The share of CU cycles is stalls / 256 / (`GRBM_GUI_ACTIVE` / 8), since `GRBM_GUI_ACTIVE`
is summed over the 8 XCCs. Measured on GPU 3 up to the exit-path split, on GPU 2 after.

| Variant | Stall cycles | Logs (`logs/`) |
| --- | --- | --- |
| Baseline | 23.23 M (16.3% of CU cycles) | `20260930-*-pmc3-base` |
| `LDSSegmentInterleave: 1` | 4.63 M | `20260930-*-pmc3-segil` |
| `LDSSegmentInterleave: 2`, per-port read order only | 1.81 M | `20260930-222002-abs2r` |
| + read gate (4,12) on the even copy | 0.57 M | `20261005-144133-abs2g4_12` |
| + exit paths (NGLL/NLL) split per port | 0.415 M | `20261005-151549-abs2exit` |
| + gate on the odd copy, spread, flip, gates (6,16) | 0.175 M | `20261005-155513-fs6_16` |
| + prologue reads split per port (current) | 0.118 M (0.08%) | `20261005-190611-final` |
| SB in its own segment (reverted) | 1.63 M | `20260930-*-abs3` |
| No per-port read reorder | 25.77 M | `20260930-*-pmc3-nostag` |
| Reorder keyed on wave bit 1 | 5.73 M | `20260930-*-pmc3-bit1` |

The 256-iteration figures come from `pmc_ksweep.sh`, which fits stalls at K = 16k/32k/64k into a
fixed part and a per-iteration part. The current kernel has no fixed part left and about 460
stall cycles per iteration summed over all CUs (under 2 per CU).

Timing does not change. 12 alternating rounds on GPU 2 (`alt_timing.py`): 278.36 us current,
278.08 us before the prologue split, 277.90 us with the per-port read order only. The paired
spread is about 2.5 us. LDS reads are off the critical path.

How mode 2 works (`configs/segil2_timing.yaml`, validated with `configs/segil2_validate.yaml`):

- Per LDS buffer, A and SA sit in segment 0 and B and SB in segment 1.
- Wave bit 0 selects the LDS read port. Even waves read A, SA, B, SB; odd waves run their own
  copy of the prologue reads, the main loop and the NGLL/NLL exit paths, reading B and SB in
  reverse order, then A and SA (`LR_ORDER_*` in `LogicalScheduler.py`).
- Flip (`LR_ORDERS_EVEN/ODD`): each copy reverses its order in the read group right after the
  barriers. There is no barrier between that group and the next iteration's reads, so both
  copies stay in the segment they were in.
- Gate (`LR_GATE_EVEN/ODD`, (phase 1, post-barrier) slack in WMMAs): each copy holds its
  second-segment reads until that many WMMAs after the slot where the other copy finishes that
  segment, then spreads them over the remaining WMMAs. Reads only move later and stay before the
  wait that consumes them.
- `seg_model.py` is a cycle model of both ports over the generated loop. It matched the
  measured loop length (2212 vs about 2180 cycles per iteration) and the ungated stall, and
  guided the gate choice.
- Correctness matches the failures above exactly, all 276 default subtile kernels are
  byte-identical to the code before this work, and the unit tests are unchanged.

What is left and why:

- `seg_model.py` predicts zero for the current kernel, so the remaining ~460 stall cycles per
  iteration (summed over the CUs, about 1.8 per CU) come from timing the model leaves out.
  Diagnostic builds of the current kernel with one class of reads removed (results wrong; logs in
  `logs/20261005-*-res-*`), stall per iteration:

  | Variant | Stall per iteration |
  | --- | --- |
  | Current | 457 |
  | Post-barrier burst reads removed | 62 |
  | Phase-1 reads removed | 252 |
  | Scale reads (SA, SB) removed | 229 |
  | A+SA removed / B+SB removed | 401 / 413 |
  | Gates (10,24) | 302 |
  | Gates (0,0) | 4793 |

  About 85% sits in the read burst right after the barriers, where both ports issue dense reads
  together. It does not depend on which tensor is read, and more slack reduces it, so the two
  ports' bursts overlap more in hardware than in the model. Either the ports leave the barrier
  skewed, or reads (the 4-byte scale reads in particular) hold the port longer than modelled.
  Telling these apart needs per-instruction timing; the ATT trace is empty on gfx1250. Gates
  (10,24) cut it by a third but push phase-1 reads toward the barrier; not worth it at 0.08%.
- With WaveGroup 2x2 the B half follows wave bit 1, so both ports read the same B and SB data.
  B conflicts can only be avoided by timing, and the waves drift slightly apart.
- Giving SB its own segment removes the overlap on paper, but both buffers' SB copies then share
  the last free segment, the next iteration's TDM writes land in a segment being read, and
  address conflicts appear. Duplicating B and SB per port would remove the shared data but adds
  about 50% L2-to-LDS traffic, the suspected main bottleneck.
