
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-1}"
MIN_SCORE="${MIN_SCORE:-}"
TS="$(date +%Y%m%d_%H%M%S)"
WORKDIR="data/checkpoints/evolution_${TS}"
LOG="data/logs/evolve_${TS}.log"
mkdir -p "$WORKDIR" data/logs

PY="conda run -n vllm_env python"
export CUDA_VISIBLE_DEVICES="$GPU"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }

log "Starting evolution cycle  (workdir: $WORKDIR, GPU: $GPU)"

log "Step 1/6: curate dataset.json → expert_train.jsonl  (with anchor mix-in)"
CURATE_ARGS=()
[[ -n "$MIN_SCORE" ]] && CURATE_ARGS+=( --min-score "$MIN_SCORE" )
ANCHOR_FILE="${ANCHOR_FILE:-data/dataset/anchor_v1_402.jsonl}"
ANCHOR_FRACTION="${EVOLUTION_ANCHOR_FRACTION:-}"
if [[ -f "$ANCHOR_FILE" ]]; then
    CURATE_ARGS+=( --anchor "$ANCHOR_FILE" )
    [[ -n "$ANCHOR_FRACTION" ]] && CURATE_ARGS+=( --anchor-fraction "$ANCHOR_FRACTION" )
fi
$PY scripts/curate_dataset.py \
    --in  data/dataset/dataset.json \
    --out "$WORKDIR/expert_train.jsonl" \
    --pairs-out "$WORKDIR/dpo_pairs.jsonl" \
    "${CURATE_ARGS[@]}" 2>&1 | tee -a "$LOG"

ROWS=$(wc -l < "$WORKDIR/expert_train.jsonl" || echo 0)
if [[ "$ROWS" -lt 5 ]]; then
    log "Only $ROWS curated rows — not enough new signal. Aborting."
    exit 0
fi
log "Curated $ROWS rows into $WORKDIR/expert_train.jsonl"


LAST_EVAL="${LAST_EVAL:-data/eval/baseline_eval_with_files.jsonl}"
ACTIVE_LEARNING="${EVOLUTION_ACTIVE_LEARNING:-1}"
if [[ "$ACTIVE_LEARNING" == "1" && -f "$LAST_EVAL" ]]; then
    log "Step 1b/6: active learning vs $LAST_EVAL"
    ACTIVE_OUT="$WORKDIR/active.jsonl"
    $PY scripts/active_learning.py \
        --eval "$LAST_EVAL" \
        --n 30 --out "$ACTIVE_OUT" 2>&1 | tee -a "$LOG"
    if [[ -s "$ACTIVE_OUT" ]]; then
        ACTIVE_N=$(wc -l < "$ACTIVE_OUT")
        cat "$ACTIVE_OUT" >> "$WORKDIR/expert_train.jsonl"
        log "Active learning added $ACTIVE_N rows → $(wc -l < "$WORKDIR/expert_train.jsonl") total"
    else
        log "Active learning produced no new rows (model already strong everywhere?)"
    fi
else
    log "Step 1b/6: skipping active learning (EVOLUTION_ACTIVE_LEARNING=$ACTIVE_LEARNING, eval exists=$([[ -f $LAST_EVAL ]] && echo 1 || echo 0))"
fi

# ── 2. Corrective fine-tune (1 epoch on top of current adapter) ─────────────
log "Step 2/6: corrective QLoRA fine-tune ($EPOCHS epoch(s))"
$PY scripts/train_qlora.py \
    --jsonl "$WORKDIR/expert_train.jsonl" \
    --output "$WORKDIR/lora" \
    --epochs "$EPOCHS" \
    --resume 2>&1 | tee -a "$LOG"

ADAPTER_DIR="$WORKDIR/lora/final_adapter"
if [[ ! -d "$ADAPTER_DIR" ]]; then
    log "ERROR: adapter not produced at $ADAPTER_DIR — aborting."
    exit 1
fi


DPO_PAIRS_FILE="$WORKDIR/dpo_pairs.jsonl"
DPO_MIN_PAIRS="${DPO_MIN_PAIRS:-50}"
DPO_PAIR_COUNT=$(wc -l < "$DPO_PAIRS_FILE" 2>/dev/null || echo 0)
if [[ "$DPO_PAIR_COUNT" -ge "$DPO_MIN_PAIRS" ]]; then
    log "Step 2b/6: DPO pass on $DPO_PAIR_COUNT retry pairs"
    $PY scripts/train_dpo.py \
        --pairs  "$DPO_PAIRS_FILE" \
        --output "$WORKDIR/dpo" \
        --epochs 1 \
        --min-pairs "$DPO_MIN_PAIRS" 2>&1 | tee -a "$LOG"
    if [[ -d "$WORKDIR/dpo/final_adapter" ]]; then
        log "DPO adapter produced — using it as the merge source instead of SFT-only"
        ADAPTER_DIR="$WORKDIR/dpo/final_adapter"
    fi
else
    log "Step 2b/6: skipping DPO ($DPO_PAIR_COUNT pairs < $DPO_MIN_PAIRS threshold)"
fi

log "Step 3/6: merge adapter → bf16"
MERGED_DIR="$WORKDIR/merged"
PUSH_ARGS=()
[[ "${EVOLVE_PUSH_HF:-0}" == "1" ]] && PUSH_ARGS+=( --push arungovindneelan/foam-cfd-unified-14b )
$PY scripts/merge_adapter.py \
    --adapter "$ADAPTER_DIR" \
    --output  "$MERGED_DIR" \
    "${PUSH_ARGS[@]}" 2>&1 | tee -a "$LOG"


log "Step 4/6: eval gate vs data/eval/ood_100_v2.json (raw-LLM, sharded, --with-files)"
EVAL_OUT="$WORKDIR/ood_eval.jsonl"
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3,4,5,6,7}"
EVAL_DIR="$WORKDIR/ood_eval_shards"
mkdir -p "$EVAL_DIR"
IFS=',' read -ra GPU_ARR <<< "$EVAL_GPUS"
N_SHARDS=${#GPU_ARR[@]}
log "  sharding $N_SHARDS ways across GPUs $EVAL_GPUS"
PIDS=()
for i in "${!GPU_ARR[@]}"; do
    G=${GPU_ARR[$i]}
    SHARD_OUT="$EVAL_DIR/shard${i}.jsonl"
    SHARD_LOG="$EVAL_DIR/shard${i}.log"
    CUDA_VISIBLE_DEVICES=$G OPENFOAM_AGENT_LLM_OVERRIDE="$MERGED_DIR" \
        VLLM_GPU_MEM_FRAC="${EVAL_VLLM_GPU_MEM_FRAC:-0.55}" \
        VLLM_MAX_NUM_SEQS="${EVAL_VLLM_MAX_NUM_SEQS:-32}" \
        $PY scripts/full_test_parallel.py \
            --shard "$i/$N_SHARDS" --raw-llm --with-files \
            --ood-file data/eval/ood_100_v2.json \
            --end-time 3 --timeout 120 \
            --out "$SHARD_OUT" > "$SHARD_LOG" 2>&1 &
    PIDS+=($!)
done
log "  waiting on $N_SHARDS shards (pids: ${PIDS[*]})"
RC=0
for pid in "${PIDS[@]}"; do wait "$pid" || RC=1; done
log "  shards finished (combined exit $RC); aggregating into $EVAL_OUT"
cat "$EVAL_DIR"/shard*.jsonl > "$EVAL_OUT"

GATE_RESULT=$($PY - "$EVAL_OUT" data/eval/regression_gate.json <<'EOF'
import json, sys
eval_jsonl, gate_path = sys.argv[1], sys.argv[2]
gate = json.load(open(gate_path))
rows = [json.loads(l) for l in open(eval_jsonl) if l.strip()]
n = len(rows)
passed = sum(1 for r in rows if r.get("success"))
matched = sum(1 for r in rows
              if r.get("expected_solver") and r.get("solver") == r["expected_solver"])
pass_rate = passed / n if n else 0.0
match_rate = matched / n if n else 0.0
ok = pass_rate >= gate["min_pass_rate"] and match_rate >= gate["min_solver_match_rate"]
print(json.dumps({"n": n, "pass_rate": pass_rate, "match_rate": match_rate,
                  "min_pass": gate["min_pass_rate"],
                  "min_match": gate["min_solver_match_rate"], "ok": ok}))
sys.exit(0 if ok else 1)
EOF
)
GATE_OK=$?
log "Gate result: $GATE_RESULT"

if [[ $GATE_OK -ne 0 ]]; then
    FAIL_DIR="data/checkpoints/failed/${TS}"
    mkdir -p "$(dirname "$FAIL_DIR")"
    mv "$WORKDIR" "$FAIL_DIR"
    log "Aggregate eval gate FAILED — adapter quarantined at $FAIL_DIR. Keeping current model."
    exit 2
fi


BASELINE_EVAL="${BASELINE_EVAL:-data/eval/baseline_eval_with_files.jsonl}"
if [[ -f "$BASELINE_EVAL" ]]; then
    log "Step 4b/6: per-prompt regression diff vs $BASELINE_EVAL"
    DIFF_REPORT="$WORKDIR/regression_diff.json"
    if $PY scripts/regression_diff.py \
            --baseline "$BASELINE_EVAL" \
            --candidate "$EVAL_OUT" \
            --out "$DIFF_REPORT" 2>&1 | tee -a "$LOG"; then
        log "Per-prompt diff: no flagged regressions on previously-passing prompts"
    else
        FAIL_DIR="data/checkpoints/failed/${TS}"
        mkdir -p "$(dirname "$FAIL_DIR")"
        mv "$WORKDIR" "$FAIL_DIR"
        log "Per-prompt regression detected — adapter quarantined at $FAIL_DIR."
        exit 3
    fi
else
    log "Step 4b/6: skipping per-prompt diff (no $BASELINE_EVAL pinned yet)"
fi

# ── 5. Swap (or dry-run) ────────────────────────────────────────────────────
if [[ "${EVOLVE_DRY_RUN:-0}" == "1" ]]; then
    log "Step 5/6: DRY RUN — would swap LLM_MODEL → $MERGED_DIR but not touching config.py"
    log "Cycle complete (dry-run). Inspect: $WORKDIR"
    exit 0
fi

CONFIG="openfoam_agent/config.py"
log "Step 5/6: swapping LLM_MODEL in $CONFIG → $MERGED_DIR"
$PY - "$CONFIG" "$MERGED_DIR" <<'EOF'
import re, sys
config_path, new_path = sys.argv[1], sys.argv[2]
text = open(config_path).read()
new = re.sub(r'^LLM_MODEL\s*=.*$',
             f'LLM_MODEL = "{new_path}"', text, count=1, flags=re.M)
open(config_path, "w").write(new)
print(f"[swap] LLM_MODEL → {new_path}")
EOF

log "Cycle complete. Restart any running ask.sh / repl.py to pick up new model."
