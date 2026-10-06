#!/usr/bin/env bash
# usage: pmc_ksweep.sh NAME BUILD_LOGDIR [ROUNDS]
# Segment-conflict PMC for one mode-2 build at K = 16k, 32k and 64k (64, 128, 256 iterations) on
# physical GPU $GPU_ID (default 3).
set -u
cd "$(dirname "$0")"
name=$1 dir=$2 rounds=${3:-2}
ROCM_PATH=/opt/rocm-10.1.0a20260908+bkc.20260917
export HIP_VISIBLE_DEVICES=${GPU_ID:-3} LD_LIBRARY_PATH=$ROCM_PATH/lib/llvm/lib/x86_64-unknown-linux-gnu
unset ROCR_VISIBLE_DEVICES
C=../../build_tmp/tensilelite/client/tensilelite-client
ini=$(find "$dir" -name ClientParameters.ini | head -1)
for K in 16384 32768 65536; do
  f=${ini%.ini}_K$K.ini
  sed -e "s/^problem-size=4096,4096,1,65536/problem-size=4096,4096,1,$K/" \
      -e "s/^a-strides=-1,65536,-1/a-strides=-1,$K,-1/" -e "s/^b-strides=-1,65536,-1/b-strides=-1,$K,-1/" "$ini" > "$f"
  for i in $(seq "$rounds"); do
    out=logs/$(date +%Y%m%d-%H%M%S)-pmcK-$name-K$K-r$i
    mkdir -p "$out"
    $ROCM_PATH/bin/rocprofv3 --input configs/rocprof_pmc.yaml -E configs/extra_counters.yaml \
      --output-directory "$out/rocprof" --output-file pmc -- $C --config-file "$f" > "$out/run.log" 2>&1 || echo "$out failed"
  done
done
../../.venv/bin/python - "$name" <<'PY' 2>&1 | grep -v frozen
import csv, glob, collections, statistics as st, re, sys
name = sys.argv[1]
res = collections.defaultdict(list)
for f in glob.glob(f"logs/*-pmcK-{name}-K*-r*/rocprof/pass_1/pmc_counter_collection.csv"):
    K = int(re.search(r"-K(\d+)-r", f).group(1))
    d = collections.defaultdict(dict)
    for r in csv.DictReader(open(f)):
        d[r["Dispatch_Id"]][r["Counter_Name"]] = float(r["Counter_Value"])
    res[K].append(st.median(v["TX_PERF_SEL_VMW_CROSS_PORT_SEGMENT_CONFLICT_LDS_STALLED_CYCLES"] for v in d.values()))
xs = sorted(res)
ys = [st.median(res[k]) for k in xs]
its = [k // 256 for k in xs]
n = len(xs); mx = sum(its) / n; my = sum(ys) / n
slope = sum((a - mx) * (b - my) for a, b in zip(its, ys)) / sum((a - mx) ** 2 for a in its)
print(f"{name}: " + "  ".join(f"{i} it {y/1e6:.3f}M" for i, y in zip(its, ys))
      + f"  | fit: fixed {(my - slope * mx)/1e6:.3f}M, per iteration {slope:,.0f} (x256 = {slope*256/1e6:.3f}M)")
PY
