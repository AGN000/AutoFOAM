#!/usr/bin/env bash
# Launch full-catalog inference test in parallel — one shard per GPU.
set -u
cd /data/foamllm3/openfoam_agent

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
END_TIME="${END_TIME:-3}"
TIMEOUT="${TIMEOUT:-120}"
PY=/home/nvidia/miniconda3/envs/vllm_env/bin/python

IFS=',' read -ra GPU_ARR <<< "$GPUS"
N=${#GPU_ARR[@]}
STAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="data/logs/full_test_${STAMP}"
mkdir -p "$OUTDIR"

echo "[full-test] $N shards across GPUs $GPUS  (end_time=$END_TIME, timeout=${TIMEOUT}s)"
echo "[full-test] logs: $OUTDIR"

PIDS=()
for i in "${!GPU_ARR[@]}"; do
    G=${GPU_ARR[$i]}
    LOG="$OUTDIR/shard${i}.log"
    OUT="$OUTDIR/shard${i}.jsonl"
    CUDA_VISIBLE_DEVICES=$G $PY scripts/full_test_parallel.py \
        --shard "$i/$N" --end-time "$END_TIME" --timeout "$TIMEOUT" \
        --out "$OUT" > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "[full-test] shard $i → GPU $G  pid=${PIDS[-1]}  log=$LOG"
done

echo "[full-test] waiting on ${#PIDS[@]} shards..."
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done

# Aggregate
$PY - <<EOF
import json, glob, collections
recs = []
for f in sorted(glob.glob("$OUTDIR/shard*.jsonl")):
    for line in open(f):
        recs.append(json.loads(line))
print(f"\n[full-test] aggregated {len(recs)} cases from {len(glob.glob('$OUTDIR/shard*.jsonl'))} shards")
ok = [r for r in recs if r.get("success")]
print(f"[full-test] PASSED: {len(ok)}/{len(recs)}  ({100*len(ok)/max(1,len(recs)):.1f}%)")
if recs:
    print(f"[full-test] avg score: {sum(r.get('score',0) for r in recs)/len(recs):.3f}")
solver_ct = collections.Counter(r.get("solver","?") for r in recs)
print("[full-test] solver distribution:")
for s, n in solver_ct.most_common(): print(f"   {s:<22} {n:>4}")
fails = [r for r in recs if not r.get("success")]
if fails:
    print(f"\n[full-test] failures ({len(fails)}):")
    for r in fails[:30]:
        print(f"  {r['case_tag']:<28} solver={r.get('solver','?'):<18} "
              f"score={r.get('score',0):.2f}  {r.get('error','')[:80]}")
    if len(fails) > 30: print(f"  ... +{len(fails)-30} more")
json.dump(recs, open("$OUTDIR/aggregated.json","w"), indent=2)
print(f"\n[full-test] full results: $OUTDIR/aggregated.json")
EOF

exit $RC
