#!/usr/bin/env python3
"""
QLoRA fine-tuning of Qwen2.5-Coder-14B on OpenFOAM expert training data.

Reads from data/dataset/expert_train.jsonl (Qwen chat format, reward-weighted).
Saves LoRA adapter to data/checkpoints/qwen_coder_14b_lora/final_adapter/.

Usage (single GPU, unsloth):
    conda run -n vllm_env python scripts/train_qlora.py

Options:
    --jsonl PATH     Path to expert_train.jsonl (default: data/dataset/expert_train.jsonl)
    --output DIR     Output directory for adapter (default: data/checkpoints/...)
    --epochs N       Number of training epochs (default: 3)
    --batch N        Per-device batch size (default: 1)
    --grad-accum N   Gradient accumulation steps (default: 8)
    --lora-r N       LoRA rank (default: 64)
    --lora-alpha N   LoRA alpha (default: 128)
    --lr LR          Learning rate (default: 2e-4)
    --min-score F    Minimum score to include example (default: 0.5)
    --max-seq N      Maximum sequence length (default: 8192)
    --resume         Resume from latest checkpoint in output dir
    --no-weights     Disable reward weighting (uniform loss)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", type=Path,
                   default=ROOT / "data/dataset/expert_train.jsonl")
    p.add_argument("--output", type=Path,
                   default=ROOT / "data/checkpoints/qwen_coder_14b_lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-weights", action="store_true")
    return p.parse_args()


def load_jsonl(path: Path, min_score: float) -> tuple[list[str], list[float]]:
    texts, scores = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            score = float(obj.get("score", 1.0))
            if score >= min_score:
                texts.append(obj["text"])
                scores.append(score)
    return texts, scores


def main():
    args = parse_args()

    # ── Load dataset ────────────────────────────────────────────────────────
    if not args.jsonl.exists():
        print(f"[train] ERROR: {args.jsonl} not found — run generate_training_data.py first")
        sys.exit(1)

    texts, scores = load_jsonl(args.jsonl, args.min_score)
    if not texts:
        print(f"[train] No examples with score >= {args.min_score}")
        sys.exit(1)

    # Reward weights: score² normalised to mean=1
    if args.no_weights:
        weights = [1.0] * len(texts)
    else:
        raw = [s ** 2 for s in scores]
        mean_w = sum(raw) / len(raw)
        weights = [w / mean_w for w in raw]

    print(f"\n[train] Dataset         : {args.jsonl}")
    print(f"[train] Examples        : {len(texts)} (score >= {args.min_score})")
    print(f"[train] Score range     : {min(scores):.2f} – {max(scores):.2f}  "
          f"mean={sum(scores)/len(scores):.2f}")
    print(f"[train] Reward weights  : {'uniform' if args.no_weights else 'score²'}")
    print(f"[train] LoRA rank       : {args.lora_r}  alpha={args.lora_alpha}")
    print(f"[train] Epochs          : {args.epochs}  "
          f"batch={args.batch}  grad_accum={args.grad_accum}")
    print(f"[train] Effective batch : {args.batch * args.grad_accum}")
    print(f"[train] Output          : {args.output}\n")

    # ── Load model with Unsloth ──────────────────────────────────────────────
    from openfoam_agent.config import LLM_MODEL
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LLM_MODEL,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ── Build HF dataset ─────────────────────────────────────────────────────
    dataset = Dataset.from_dict({"text": texts, "weight": weights})

    # ── Training config ──────────────────────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)

    resume_from = None
    if args.resume:
        ckpts = sorted(args.output.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        if ckpts:
            resume_from = str(ckpts[-1])
            print(f"[train] Resuming from {resume_from}")

    training_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_8bit",
        bf16=True,
        max_seq_length=args.max_seq,
        dataset_text_field="text",
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )

    # ── Reward-weighted trainer ──────────────────────────────────────────────
    from openfoam_agent.training import make_reward_weighted_trainer
    WeightedTrainer = make_reward_weighted_trainer(SFTTrainer)

    trainer = WeightedTrainer(
        reward_weights=weights,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    print("[train] Starting QLoRA fine-tuning...")
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save adapter ─────────────────────────────────────────────────────────
    adapter_dir = args.output / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\n[train] Adapter saved → {adapter_dir}")

    # ── Print training summary ───────────────────────────────────────────────
    log = trainer.state.log_history
    if log:
        train_losses = [e["loss"] for e in log if "loss" in e]
        if train_losses:
            print(f"[train] Final loss      : {train_losses[-1]:.4f}")
            print(f"[train] Initial loss    : {train_losses[0]:.4f}")
            print(f"[train] Improvement     : {train_losses[0] - train_losses[-1]:.4f}")
    print("[train] Done.\n")


if __name__ == "__main__":
    main()
