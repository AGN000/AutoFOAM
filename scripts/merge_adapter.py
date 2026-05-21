#!/usr/bin/env python3
"""
Merge QLoRA adapter into the base model and save for vLLM inference.

After fine-tuning with train_qlora.py, run this to produce a merged model
that vLLM can load directly (no adapter overhead at inference time).

Usage:
    conda run -n vllm_env python scripts/merge_adapter.py

Options:
    --adapter DIR   LoRA adapter directory (default: data/checkpoints/.../final_adapter)
    --output DIR    Merged model output (default: data/checkpoints/qwen_coder_14b_merged)
    --push HF_REPO  Optional: push merged model to HuggingFace Hub
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", type=Path,
                   default=ROOT / "data/checkpoints/qwen_coder_14b_lora/final_adapter")
    p.add_argument("--output", type=Path,
                   default=ROOT / "data/checkpoints/qwen_coder_14b_merged")
    p.add_argument("--push", default="", help="HuggingFace repo id to push to")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.adapter.exists():
        print(f"[merge] ERROR: adapter not found at {args.adapter}")
        print("[merge] Run train_qlora.py first.")
        sys.exit(1)

    from openfoam_agent.config import LLM_MODEL
    from unsloth import FastLanguageModel

    print(f"[merge] Loading base model  : {LLM_MODEL}")
    print(f"[merge] Adapter             : {args.adapter}")
    print(f"[merge] Output              : {args.output}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter),
        max_seq_length=8192,
        load_in_4bit=True,
        dtype=None,
    )

    print("[merge] Merging adapter into base weights (16-bit)...")
    model = model.merge_and_unload()

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(args.output),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"[merge] Merged model saved → {args.output}")

    if args.push:
        print(f"[merge] Pushing to HuggingFace: {args.push}")
        model.push_to_hub_merged(args.push, tokenizer, save_method="merged_16bit")
        print(f"[merge] Pushed to {args.push}")

    print("\n[merge] To use with vLLM, update config.py:")
    print(f"    LLM_MODEL = '{args.output}'")
    print("[merge] Done.\n")


if __name__ == "__main__":
    main()
