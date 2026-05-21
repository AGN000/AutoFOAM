#!/usr/bin/env bash
# Interactive REPL — loads the model once, accepts prompts in a loop.
#
# Usage:
#   bash scripts/repl.sh           # auto-pick a GPU with ≥30 GB free
#   GPU=4 bash scripts/repl.sh     # force a specific GPU
#
# Once inside the REPL:
#   prompt> 2D lid-driven cavity Re=1000, 2m square, water
#   prompt> flow over NACA0012, Re=1e6, AoA 5 deg
#   prompt> quit            (or Ctrl-D)

PROJ=/data/foamllm3/openfoam_agent
PY=/home/nvidia/miniconda3/envs/vllm_env/bin/python

# Pick a GPU with enough free memory if not given
if [ -z "${GPU:-}" ]; then
    GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader \
          | awk -F', ' '{ gsub(" MiB","",$2); if ($2+0 > 30000) print $1 }' \
          | head -1)
    if [ -z "$GPU" ]; then
        echo "no free GPU with ≥30 GB available. Override with GPU=N $0" >&2
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader >&2
        exit 1
    fi
fi

# Sensible defaults that coexist with other GPU jobs.
export VLLM_GPU_MEM_FRAC=${VLLM_GPU_MEM_FRAC:-0.55}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-32}

echo "=========================================================="
echo "  GPU         : $GPU"
echo "  GPU mem frac: $VLLM_GPU_MEM_FRAC"
echo "  Max seqs    : $VLLM_MAX_NUM_SEQS"
echo "  Project     : $PROJ"
echo "=========================================================="

source /home/nvidia/miniconda3/envs/openfoam2412/etc/bashrc 2>/dev/null
cd "$PROJ"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU $PY -u scripts/repl.py
