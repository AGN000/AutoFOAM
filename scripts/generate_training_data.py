#!/usr/bin/env python3
"""
Generate augmented training data for OpenFOAM expert fine-tuning.

Runs the simulation pipeline with ground-truth CFDParams (no LLM needed)
for every entry in the prompt catalog, then saves:
  - data/dataset/dataset.json        (TrainingExample records, appended)
  - data/dataset/expert_train.jsonl  (Qwen chat format, ready for SFTTrainer)

Usage:
    conda run -n vllm_env python scripts/generate_training_data.py [--dry-run] [--tag TAG]

Options:
    --dry-run   Print catalog stats without running simulations.
    --tag TAG   Only run cases whose case_tag starts with TAG (e.g. "cavity").
    --skip N    Skip first N entries (resume after crash).
    --timeout S Simulation timeout in seconds (default 300).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openfoam_agent.prompt_catalog import PROMPT_CATALOG, PromptCase
from openfoam_agent.training import format_example
from openfoam_agent.schemas import TrainingExample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--tag", default="", help="Filter by case_tag prefix")
    p.add_argument("--skip", type=int, default=0, help="Skip first N entries")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--min-score", type=float, default=0.5)
    return p.parse_args()


def print_catalog_summary():
    from collections import Counter
    tags = [c.case_tag.split("_")[0] for c in PROMPT_CATALOG]
    counts = Counter(tags)
    print(f"\n{'='*60}")
    print(f"  Prompt catalog: {len(PROMPT_CATALOG)} entries")
    print(f"{'='*60}")
    for geom, n in sorted(counts.items()):
        print(f"  {geom:<20} {n:>3} prompts")
    print(f"{'='*60}\n")


def run_case(agent, entry: PromptCase, timeout: int) -> tuple[TrainingExample | None, float]:
    """Run one catalog entry. Returns (TrainingExample or None, score)."""
    print(f"\n[gen] ▶ {entry.case_tag}")
    print(f"      prompt: {entry.prompt[:70]}")

    t0 = time.time()
    result = agent.run_with_params(
        prompt=entry.prompt,
        params=entry.params,
        max_retries=2,
        use_gmsh=True,
        case_name=entry.case_tag,
        sim_timeout=timeout,
    )
    elapsed = time.time() - t0

    status = "✓" if result.success else "✗"
    print(f"      {status} score={result.score:.2f}  solver={result.solver}  "
          f"elapsed={elapsed:.0f}s  {result.feedback}")

    if result.success and result.params and result.case_dir:
        example = TrainingExample(
            prompt=entry.prompt,
            refined_prompt=entry.prompt,
            params=result.params,
            case_dir=result.case_dir,
            solver=result.solver,
            score=result.score,
            feedback=result.feedback + (f" | {entry.expert_notes}" if entry.expert_notes else ""),
            converged=result.success,
            runtime=result.runtime,
            timestamp=time.time(),
            case_files_text=agent._read_case_files(Path(result.case_dir)),
        )
        return example, result.score
    return None, result.score


def main():
    args = parse_args()

    print_catalog_summary()

    catalog = PROMPT_CATALOG
    if args.tag:
        catalog = [c for c in catalog if c.case_tag.startswith(args.tag)]
        print(f"[gen] Filtered to {len(catalog)} entries matching tag prefix '{args.tag}'\n")

    if args.skip:
        catalog = catalog[args.skip:]
        print(f"[gen] Skipping first {args.skip} → {len(catalog)} remaining\n")

    if args.dry_run:
        print("[gen] --dry-run: exiting without running simulations.")
        return

    # Import agent (heavy; don't import during --dry-run)
    from openfoam_agent.agent import OpenFOAMAgent
    agent = OpenFOAMAgent(use_llm=False)

    dataset_dir = ROOT / "data" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_json = dataset_dir / "dataset.json"
    expert_jsonl = dataset_dir / "expert_train.jsonl"

    # Load existing dataset
    existing: list[dict] = []
    if dataset_json.exists():
        try:
            existing = json.loads(dataset_json.read_text())
        except Exception:
            pass
    print(f"[gen] Starting with {len(existing)} existing examples in dataset.json\n")

    collected = 0
    failed = 0
    scores = []

    for i, entry in enumerate(catalog):
        try:
            example, score = run_case(agent, entry, args.timeout)
            scores.append(score)

            if example and score >= args.min_score:
                # Append to dataset.json
                existing.append(json.loads(example.model_dump_json()))
                dataset_json.write_text(json.dumps(existing, indent=2))

                # Append to expert_train.jsonl (Qwen chat format)
                formatted = format_example(example)
                with expert_jsonl.open("a") as f:
                    f.write(json.dumps({"text": formatted, "score": score}) + "\n")

                collected += 1
            else:
                failed += 1

        except KeyboardInterrupt:
            print(f"\n[gen] Interrupted at entry {i} ({entry.case_tag})")
            break
        except Exception as e:
            print(f"[gen] ERROR on {entry.case_tag}: {e}")
            failed += 1

    # Summary
    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"\n{'='*60}")
    print(f"  Generation complete")
    print(f"  Collected : {collected} examples (score >= {args.min_score})")
    print(f"  Failed    : {failed}")
    print(f"  Avg score : {avg_score:.3f}")
    print(f"  dataset   : {dataset_json}")
    print(f"  expert    : {expert_jsonl}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
