#!/usr/bin/env bash
# Full pipeline: wait for data generation → QLoRA training → merge adapter
# Run inside tmux: tmux new -s foamllm -d && tmux send-keys -t foamllm 'bash scripts/run_pipeline.sh' Enter
set -euo pipefail

REPO=/data/foamllm3/openfoam_agent
LOG=$REPO/data/pipeline.log
GEN_PID=${GEN_PID:-0}

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

cd "$REPO"

# ── Step 1: Wait for generation if still running ─────────────────────────────
if [ "$GEN_PID" -gt 0 ] && kill -0 "$GEN_PID" 2>/dev/null; then
    log "Waiting for data generation (PID $GEN_PID) to finish..."
    while kill -0 "$GEN_PID" 2>/dev/null; do
        N=$(wc -l < data/dataset/expert_train.jsonl 2>/dev/null || echo 0)
        LATEST=$(ls -t data/cases/ 2>/dev/null | head -1 || echo "none")
        log "  examples=$N  latest=$LATEST"
        sleep 30
    done
fi

N=$(wc -l < data/dataset/expert_train.jsonl 2>/dev/null || echo 0)
log "Generation complete — $N examples in expert_train.jsonl"

# ── Step 2: QLoRA training ────────────────────────────────────────────────────
log "Starting QLoRA training..."
conda run -n vllm_env python scripts/train_qlora.py \
    --epochs 3 \
    --lora-r 64 \
    --lora-alpha 128 \
    --min-score 0.5 \
    2>&1 | tee -a "$LOG"

log "Training complete."

# ── Step 3: Merge adapter ─────────────────────────────────────────────────────
log "Merging LoRA adapter into base model..."
conda run -n vllm_env python scripts/merge_adapter.py \
    2>&1 | tee -a "$LOG"

log "Merge complete."
log "Pipeline finished. Run test_inference.py to evaluate."
log "  conda run -n vllm_env python scripts/test_inference.py"
