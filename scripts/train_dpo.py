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
    p.add_argument("--pairs", type=Path,
                   default=ROOT / "data/dataset/dpo_pairs.jsonl")
    p.add_argument("--output", type=Path,
                   default=ROOT / "data/checkpoints/qwen_coder_14b_dpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-r", type=int, default=32,
                   help="DPO LoRA rank (smaller than SFT — preference is a finer-grained edit)")
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-6,
                   help="DPO is sensitive — keep this an order below SFT")
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO temperature; higher = stronger pull toward chosen")
    p.add_argument("--max-seq", type=int, default=8192)
    p.add_argument("--min-pairs", type=int, default=50,
                   help="Refuse to train if pairs.jsonl has fewer rows")
    return p.parse_args()


def load_pairs(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
         
            rows.append({
                "prompt":   obj["prompt"],
                "chosen":   obj["chosen"],
                "rejected": obj["rejected"],
            })
    return rows


def main():
    args = parse_args()

    if not args.pairs.exists():
        print(f"[dpo] {args.pairs} does not exist — run curate_dataset.py --pairs-out first.")
        sys.exit(1)

    pairs = load_pairs(args.pairs)
    if len(pairs) < args.min_pairs:
        print(f"[dpo] Only {len(pairs)} pairs (need ≥ {args.min_pairs}) — refusing to train.")
        print("[dpo] Keep capturing retry pairs via Layer-1 self-correction first.")
        sys.exit(0)

    print(f"\n[dpo] Pairs            : {len(pairs)}  (from {args.pairs})")
    print(f"[dpo] LoRA rank        : {args.lora_r}  alpha={args.lora_alpha}")
    print(f"[dpo] beta             : {args.beta}")
    print(f"[dpo] LR               : {args.lr}")
    print(f"[dpo] Epochs           : {args.epochs}  "
          f"batch={args.batch}  grad_accum={args.grad_accum}")
    print(f"[dpo] Output           : {args.output}\n")

    from openfoam_agent.config import LLM_MODEL
    from unsloth import FastLanguageModel
    from trl import DPOTrainer, DPOConfig
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
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = Dataset.from_list(pairs)
    args.output.mkdir(parents=True, exist_ok=True)

    dpo_args = DPOConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit",
        bf16=True,
        max_length=args.max_seq,
        max_prompt_length=args.max_seq // 2,
        beta=args.beta,
        logging_steps=2,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )


    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("[dpo] Starting DPO training...")
    trainer.train()

    adapter_dir = args.output / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\n[dpo] Adapter saved → {adapter_dir}")
    print("[dpo] Next: scripts/merge_adapter.py --adapter "
          f"{adapter_dir} --output data/checkpoints/qwen_coder_14b_dpo_merged")


if __name__ == "__main__":
    main()
