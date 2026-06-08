
PROJ=/data/foamllm3/openfoam_agent
PY=/home/nvidia/miniconda3/envs/vllm_env/bin/python


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
