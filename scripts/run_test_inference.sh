#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
cd /data/foamllm3/openfoam_agent
exec /home/nvidia/miniconda3/envs/vllm_env/bin/python scripts/test_inference.py "$@"
