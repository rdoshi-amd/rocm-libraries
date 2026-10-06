#!/usr/bin/env python3
"""Cycle model of LDS cross-port segment conflicts in the subtile MXF4 main loop.

usage: seg_model.py KERNEL.s [TW ...]

Parses the even-wave (label_LoopBeginL) and odd-wave (label_LoopBeginL_PortB) main loop copies
of an LDSSegmentInterleave=2 kernel, or the single copy otherwise, and runs 4 waves (waves 0/2 on
port 0, waves 1/3 on port 1) through a few iterations. TW is the issue cost of one WMMA in
cycles. Each port serves one LDS request per cycle stream (b128: 2 cycles, b32: 1 cycle); when
both ports want the same segment in the same cycle one of them stalls for that cycle.
"""
import re, sys

SEG = {"A": 0, "SA": 0, "B": 1, "SB": 1}
COST = {"A": 2, "B": 2, "SA": 1, "SB": 1}


def parse(lines, start):
  ops, i = [], start + 1
  while i < len(lines):
    l = lines[i].strip()
    if l.startswith("s_cbranch_scc0") or l.startswith("s_cbranch_scc1"):
      break
    if l.startswith("ds_load"):
      t = re.search(r"Subtile([AB])\[", l)
      tc = t.group(1) if t else "S" + re.search(r"scaleMXS([AB])", l).group(1)
      ops.append(("R", tc))
    elif l.startswith("v_wmma"):
      ops.append(("W",))
    elif l.startswith("s_wait_dscnt"):
      ops.append(("D", int(l.split()[1].rstrip(","), 0)))
    elif l.startswith("s_barrier_wait"):
      ops.append(("B",))
    i += 1
  return ops


def simulate(progs, tw, iters=6, issue=1, readLat=64):
  """progs[w] is the op list for wave w; returns stall cycles per iteration (steady state)."""
  n = len(progs)
  pc = [0] * n
  it = [0] * n
  busy = [0] * n          # wave can issue at cycle >= busy[w]
  outstanding = [[] for _ in range(n)]  # completion cycle per read (None until served)
  atBarrier = [False] * n
  phase = [1] * n         # 1: before the TDM barriers, 2 and 3: after the first/second barrier
  queues = [[], []]       # per port: [wave, tc, remaining, readIdx]
  stalls, stallsAt = 0, []
  t = 0
  lastPort = 0
  while min(it) < iters and t < 10 ** 6:
    # waves issue
    for w in range(n):
      if atBarrier[w] or t < busy[w] or it[w] >= iters:
        continue
      op = progs[w][pc[w]]
      if op[0] == "W":
        busy[w] = t + tw
      elif op[0] == "R":
        rec = [None, phase[w]]
        outstanding[w].append(rec)
        queues[w & 1].append([w, op[1], COST[op[1]], rec])
        busy[w] = t + issue
      elif op[0] == "D":
        pend = [r for r in outstanding[w] if r[0] is None or r[0] > t]
        if len(pend) > op[1]:
          continue
        outstanding[w] = pend
      elif op[0] == "B":
        atBarrier[w] = True
        phase[w] = phase[w] + 1 if pc[w] + 1 < len(progs[w]) else 1
      pc[w] += 1
      if pc[w] == len(progs[w]):
        pc[w] = 0
        it[w] += 1
        phase[w] = 1
    if all(atBarrier[w] or it[w] >= iters for w in range(n)) and any(atBarrier):
      atBarrier = [False] * n
    # ports serve
    heads = [q[0] if q else None for q in queues]
    blocked = None
    if heads[0] and heads[1] and SEG[heads[0][1]] == SEG[heads[1][1]]:
      blocked = lastPort  # alternate who loses
      lastPort ^= 1
      if min(it) >= 2:
        stalls += 1
        stallsAt.append((heads[0][1], heads[0][3][1], heads[1][1], heads[1][3][1]))
    for p in (0, 1):
      h = heads[p]
      if h is None or p == blocked:
        continue
      h[2] -= 1
      if h[2] == 0:
        h[3][0] = t + readLat
        queues[p].pop(0)
    t += 1
  from collections import Counter
  return stalls / max(1, iters - 2), Counter(stallsAt), t / iters


def runs(prog):
  """Split at waits/barriers: [(start, end)] index ranges holding only W and R ops."""
  out, s = [], 0
  for i, op in enumerate(prog + [("B",)]):
    if op[0] not in ("W", "R"):
      if i > s:
        out.append((s, i))
      s = i + 1
  return out


def gate(even, odd, slack=0):
  """Delay the even copy's first-segment-switch reads until after the WMMA following the odd
  copy's last segment-1 read in the matching run (+slack WMMAs)."""
  er, orr = runs(even), runs(odd)
  assert len(er) == len(orr)
  out = list(even)
  for (es, ee), (os_, oe) in zip(er, orr):
    seg = [op for op in odd[os_:oe] if op[0] == "R"]
    if not seg or not any(SEG[op[1]] == 1 for op in seg):
      continue
    wi, lastW = 0, None
    for op in odd[os_:oe]:
      if op[0] == "W":
        wi += 1
      elif SEG[op[1]] == 1:
        lastW = wi
    gateW = lastW + 1 + slack
    body = even[es:ee]
    ws = [op for op in body if op[0] == "W"]
    reads = []
    wi = 0
    for op in body:
      if op[0] == "W":
        wi += 1
      else:
        reads.append([wi, op])
    seen0 = False
    for r in reads:
      if SEG[r[1][1]] == 0:
        seen0 = True
      elif seen0:
        r[0] = max(r[0], min(gateW, len(ws)))
    new, k = [], 0
    for w in range(len(ws) + 1):
      while k < len(reads) and reads[k][0] <= w:
        new.append(reads[k][1])
        k += 1
      if w < len(ws):
        new.append(ws[w])
    out[es:ee] = new
  return out


def main():
  path = sys.argv[1]
  args = sys.argv[2:]
  slack = None
  if args and args[0].startswith("gate"):
    slack = int(args.pop(0)[4:] or 0)
  tws = [int(x) for x in args] or [4, 8, 16, 32]
  lines = open(path).read().splitlines()
  idx = {l.strip().rstrip(":"): i for i, l in enumerate(lines) if l.strip().endswith(":")}
  even = parse(lines, idx["label_LoopBeginL"])
  odd = parse(lines, idx["label_LoopBeginL_PortB"]) if "label_LoopBeginL_PortB" in idx else even
  if slack is not None:
    even = gate(even, odd, slack)
  progs = [even, odd, even, odd]
  for tw in tws:
    s, where, cyc = simulate(progs, tw)
    print(f"TW={tw:3d}: {s:7.1f} stall cycles/iteration/CU -> {s * 256 * 256 / 1e6:6.2f} M for 256 CUs x 256 iterations, {cyc:7.1f} cycles/iteration")
    for (t0, p0, t1, p1), c in where.most_common(6):
      print(f"    port0 {t0:2s} (phase {p0}) vs port1 {t1:2s} (phase {p1}): {c / 4:6.1f}/iteration")


if __name__ == "__main__":
  main()
