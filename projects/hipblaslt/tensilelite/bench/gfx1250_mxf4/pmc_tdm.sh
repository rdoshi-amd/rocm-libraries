#!/usr/bin/env bash
# usage: pmc_tdm.sh name=LOGDIR [name=LOGDIR ...]
# TDM/L1/L2 supply counters for each build's ClientParameters.ini on physical GPU $GPU_ID
# (default 3). Prints the per-dispatch median of every counter, one column per build.
set -u
cd "$(dirname "$0")"
ROCM_PATH=/opt/rocm-10.1.0a20260908+bkc.20260917
export HIP_VISIBLE_DEVICES=${GPU_ID:-3} LD_LIBRARY_PATH=$ROCM_PATH/lib/llvm/lib/x86_64-unknown-linux-gnu
unset ROCR_VISIBLE_DEVICES
C=../../build_tmp/tensilelite/client/tensilelite-client
PASSES=(
  "TX_VMW_REQ_READ TX_VMW_GL1_REQ_READ_64B TX_VMW_GL1_REQ_READ_128B TX_VMW_GL1_PENDING_STALL TX_VMW_GL1_VMW_BACK_PRESSURE TX_VMW_READ_SETCONFLICT_STALL TX_VMW_LFIFO_STALL TX_VMW_MEM_REQ_FIFO_STALL GRBM_GUI_ACTIVE"
  "TX_VMW_UTCL0_REQUEST TX_VMW_UTCL0_TRANSLATION_HIT TX_VMW_UTCL0_STALL_INFLIGHT_MAX TX_VMW_UTCL0_STALL_MULTI_MISS TX_VMW_VMW_VCA_REQ_STALL TX_VMW_DATA_FIFO_STALL TX_VMW_GL1_VMW_RDRET_STALL TX_VMW_VMW_LATENCY"
  "GL2C_HIT GL2C_MISS GL2C_REQ GL2C_EA_RDREQ"
  "GL2C_TAG_STALL GL2C_SRC_FIFO_FULL GL2C_IB_STALL GL2C_LATENCY_FIFO_FULL"
  "GL2C_EA_RDREQ_LEVEL GL2C_EA_RDREQ_DRAM GL2C_BUSY GL2C_CYCLE"
  "GL1C_GL2_REQ_READ_LEVEL GL1A_BUSY GL1A_CYCLE CHC_REQ_READ CHC_REQ_READ_128B CHC_REQ_READ_256B CHA_BUSY CHA_CYCLE GL2A_BUSY GL2A_CYCLE SQ_INSTS_TDM"
)
# PMC_PASSES="A B C;D E" replaces the passes above (semicolon between passes).
if [[ -n "${PMC_PASSES:-}" ]]; then IFS=';' read -r -a PASSES <<< "$PMC_PASSES"; fi
stamp=$(date +%Y%m%d-%H%M%S)
args=()
for v in "$@"; do
  name=${v%%=*} dir=${v#*=}
  ini=$(find "$dir" -name ClientParameters.ini | head -1)
  for p in "${!PASSES[@]}"; do
    out=logs/$stamp-pmctdm-$name-p$p
    mkdir -p "$out"
    $ROCM_PATH/bin/rocprofv3 -E configs/extra_counters.yaml --pmc ${PASSES[$p]} --kernel-include-regex Cijk \
      --output-format csv --output-directory "$out" --output-file pmc -- $C --config-file "$ini" > "$out/run.log" 2>&1 \
      || echo "$out failed" >&2
  done
  args+=("$name")
done
../../.venv/bin/python - "$stamp" "${args[@]}" <<'PY' 2>&1 | grep -v frozen
import csv, glob, collections, statistics as st, sys
stamp, names = sys.argv[1], sys.argv[2:]
cols = {}
for n in names:
    med = {}
    for f in glob.glob(f"logs/{stamp}-pmctdm-{n}-p*/pmc_counter_collection.csv"):
        d = collections.defaultdict(dict)
        for r in csv.DictReader(open(f)):
            d[r["Dispatch_Id"]][r["Counter_Name"]] = float(r["Counter_Value"])
        for k in next(iter(d.values())):
            med[k] = st.median(v[k] for v in d.values() if k in v)
    cols[n] = med
keys = sorted({k for m in cols.values() for k in m})
print(f"{'counter':34s}" + "".join(f"{n:>16s}" for n in names))
for k in keys:
    print(f"{k:34s}" + "".join(f"{cols[n].get(k, float('nan')):>16,.0f}" for n in names))
PY
