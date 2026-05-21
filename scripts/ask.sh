#!/usr/bin/env bash
# Usage:
#   bash scripts/ask.sh "2D lid-driven cavity Re=1000, 2m square, water"
# Optional env vars:
#   GPU=4          # force a specific GPU (default: auto-pick a free one)
#   TIMEOUT=300    # solver wall-clock timeout (s)
#   RETRIES=2      # max retry attempts
#   QUIET=1        # filter vLLM noise (default: 0 — show everything for debugging)

if [ -z "$1" ]; then
    echo "usage: $0 \"<your CFD prompt here>\""
    echo "example: $0 \"flow over NACA0012, chord 0.5m, Re=1e6, AoA 5 deg\""
    exit 1
fi

PROMPT="$1"
PROJ=/data/foamllm3/openfoam_agent
PY=/home/nvidia/miniconda3/envs/vllm_env/bin/python

# Pick a free GPU if not given (≥30 GB free required for the 14B model)
if [ -z "${GPU:-}" ]; then
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader \
          | awk -F', ' '{ gsub(" MiB","",$2); if ($2+0 > 30000) print $1 }' \
          | head -1)
    if [ -z "$GPU" ]; then
        echo "no free GPU with ≥30 GB available. Override with GPU=N $0 ..." >&2
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader >&2
        exit 1
    fi
fi

TIMEOUT=${TIMEOUT:-300}
RETRIES=${RETRIES:-2}
QUIET=${QUIET:-0}
# vLLM memory tuning — keep modest so it coexists with other GPU jobs.
# Override with VLLM_GPU_MEM_FRAC=0.7 or higher if you have a fully free GPU.
export VLLM_GPU_MEM_FRAC=${VLLM_GPU_MEM_FRAC:-0.55}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-32}

LOG=/tmp/ask_$(date +%Y%m%d_%H%M%S).log

echo "=========================================================="
echo "  GPU         : $GPU"
echo "  GPU mem frac: $VLLM_GPU_MEM_FRAC"
echo "  Max seqs    : $VLLM_MAX_NUM_SEQS"
echo "  Timeout     : ${TIMEOUT}s"
echo "  Retries     : $RETRIES"
echo "  Log file    : $LOG"
echo "  Prompt      : $PROMPT"
echo "=========================================================="
echo "  vLLM is loading the 14B model (~30-60s the first time)..."
echo "=========================================================="

# Source OpenFOAM env
source /home/nvidia/miniconda3/envs/openfoam2412/etc/bashrc 2>/dev/null

cd "$PROJ"
# Run with line-buffered Python output so we see progress live.
# Tee to log so we keep a copy even if grep filters interactive output.
if [ "$QUIET" = "1" ]; then
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_agent.py run "$PROMPT" \
        --timeout "$TIMEOUT" --retries "$RETRIES" 2>&1 \
      | tee "$LOG" \
      | grep --line-buffered -vE "Capturing CUDA|Processed prompts|Rendering conversations|leaked function|^ - |Could not determine OPENFOAM"
else
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/run_agent.py run "$PROMPT" \
        --timeout "$TIMEOUT" --retries "$RETRIES" 2>&1 \
      | tee "$LOG"
fi
RC=${PIPESTATUS[0]}

echo
echo "=========================================================="
if [ "$RC" -ne 0 ]; then
    echo "  ✗ Agent exited with code $RC"
    echo "  Full log: $LOG"
else
    echo "  ✓ Done (exit $RC)"
fi
echo "  Latest case directory:"
ls -td "$PROJ/data/cases"/*/ 2>/dev/null | head -1
echo "=========================================================="
exit $RC
