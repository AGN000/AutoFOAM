#!/usr/bin/env python3
"""Curate the rolling capture file (data/dataset/dataset.json) into a
training-ready Qwen-chat JSONL (data/dataset/expert_train.jsonl).

Steps:
  1. Load every TrainingExample from dataset.json (score >= 0.5 captures).
  2. De-duplicate by (prompt, solver, params hash) — keep the highest score.
  3. Drop rows below --min-score (default: config.MIN_RETRAIN_SCORE = 0.65).
  4. Optional: mine retry pairs from attempts.jsonl
     (low-score first attempt + higher-score retry on the same prompt) →
     emit them as a sidecar JSONL (--pairs-out) for future DPO training.
  5. Format each surviving row with training.format_example() (Qwen chat
     template) and write {"text": ..., "score": ...} per line.

Usage:
    python scripts/curate_dataset.py
    python scripts/curate_dataset.py --min-score 0.7 --max-examples 1000
    python scripts/curate_dataset.py --pairs-out data/dataset/dpo_pairs.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openfoam_agent.config import (
    DATASET_DIR, ATTEMPTS_LOG, MIN_RETRAIN_SCORE,
    ANCHOR_DATASET, EVOLUTION_ANCHOR_FRACTION,
)
from openfoam_agent.schemas import TrainingExample, CFDParams
from openfoam_agent.training import (
    format_example, format_dpo_prompt, format_dpo_response, _build_expert_analysis,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_file", type=Path,
                   default=DATASET_DIR / "dataset.json")
    # NOTE: default output is intentionally NOT data/dataset/expert_train.jsonl
    # — that path is the frozen v1 anchor corpus (anchor_v1_402.jsonl) and must
    # never be overwritten. Use a versioned default.
    p.add_argument("--out", type=Path,
                   default=DATASET_DIR / "curated_latest.jsonl")
    p.add_argument("--min-score", type=float, default=MIN_RETRAIN_SCORE,
                   help=f"Drop rows below this score (default: {MIN_RETRAIN_SCORE})")
    p.add_argument("--max-examples", type=int, default=10000,
                   help="Cap output at this many rows (highest-score first)")
    p.add_argument("--attempts", type=Path, default=ATTEMPTS_LOG,
                   help="Path to attempts.jsonl for retry-pair mining")
    p.add_argument("--pairs-out", type=Path, default=None,
                   help="If set, emit (chosen, rejected) DPO pairs to this JSONL")
    p.add_argument("--anchor", type=Path, default=None,
                   help=f"If set, mix in random rows from this anchor JSONL "
                        f"so it makes up --anchor-fraction of the output "
                        f"(default fraction: {EVOLUTION_ANCHOR_FRACTION}). "
                        f"Production path: {ANCHOR_DATASET}")
    p.add_argument("--anchor-fraction", type=float,
                   default=EVOLUTION_ANCHOR_FRACTION,
                   help="Target fraction of output rows that come from --anchor")
    return p.parse_args()


def _params_hash(params_dict: dict) -> str:
    """Stable hash of the params dict for de-duplication.

    We hash on the physics-relevant subset (geometry, solver-defining flags,
    Re) so that two semantically-identical runs collapse even if they differ
    on cosmetic fields like extraction_notes or end_time.
    """
    keys = ("geometry_type", "is_3d", "is_transient", "is_compressible",
            "has_heat_transfer", "is_multiphase", "flow_regime",
            "turbulence_model", "reynolds_number")
    sub = {k: params_dict.get(k) for k in keys}
    blob = json.dumps(sub, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def dedupe(examples: list[TrainingExample]) -> list[TrainingExample]:
    """Keep the highest-score entry for each (prompt, solver, params_hash)."""
    best: dict[tuple, TrainingExample] = {}
    for ex in examples:
        key = (ex.prompt.strip().lower(),
               ex.solver,
               _params_hash(json.loads(ex.params.model_dump_json())))
        if key not in best or ex.score > best[key].score:
            best[key] = ex
    return list(best.values())


def _row_to_training_example(row: dict) -> TrainingExample:
    """Build a TrainingExample stub from an attempts.jsonl row so we can
    reuse _build_expert_analysis() for the DPO response side."""
    return TrainingExample(
        prompt=row["prompt"],
        refined_prompt=row.get("refined_prompt", row["prompt"]),
        params=CFDParams.model_validate(row["params"]),
        case_dir=row.get("case_dir", ""),
        solver=row["solver"],
        score=row["score"],
        feedback=row.get("feedback", ""),
        converged=row.get("converged", False),
        runtime=row.get("runtime", 0.0),
        timestamp=row.get("timestamp", 0.0),
        case_files_text=row.get("case_files_text", ""),
    )


def mine_retry_pairs(attempts_path: Path) -> list[dict]:
    """Find (low-score attempt, higher-score retry) on the same prompt.

    Emits TRL-DPO-ready rows:
      {"prompt": <Qwen-chat prefix up to assistant tag>,
       "chosen":   <full assistant response of high-score attempt>,
       "rejected": <full assistant response of low-score attempt>,
       "chosen_score": ..., "rejected_score": ...,
       "solver_chosen": ..., "solver_rejected": ...}

    Only emits pairs where the score gap is meaningful (>= 0.2) and the
    chosen attempt crossed the success threshold (score >= 0.5).
    """
    if not attempts_path.exists():
        return []
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    with open(attempts_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            by_prompt[row["prompt"].strip().lower()].append(row)

    pairs: list[dict] = []
    for prompt_key, rows in by_prompt.items():
        if len(rows) < 2:
            continue
        rows_sorted = sorted(rows, key=lambda r: r["score"])
        worst, best = rows_sorted[0], rows_sorted[-1]
        if best["score"] - worst["score"] < 0.2:
            continue
        if best["score"] < 0.5:
            continue
        try:
            best_ex = _row_to_training_example(best)
            worst_ex = _row_to_training_example(worst)
        except Exception:
            continue   # malformed params — skip
        chosen_response = format_dpo_response(
            _build_expert_analysis(best_ex),
            best_ex.case_files_text or "(no case files captured)",
        )
        rejected_response = format_dpo_response(
            _build_expert_analysis(worst_ex),
            worst_ex.case_files_text or "(no case files captured)",
        )
        pairs.append({
            "prompt":   format_dpo_prompt(best_ex.refined_prompt or best_ex.prompt),
            "chosen":   chosen_response,
            "rejected": rejected_response,
            "chosen_score": best["score"],
            "rejected_score": worst["score"],
            "solver_chosen": best["solver"],
            "solver_rejected": worst["solver"],
        })
    return pairs


def main():
    args = parse_args()

    if not args.in_file.exists():
        print(f"[curate] {args.in_file} does not exist — nothing to curate.")
        sys.exit(1)

    raw = json.loads(args.in_file.read_text())
    examples = [TrainingExample.model_validate(e) for e in raw]
    print(f"[curate] Loaded {len(examples)} rows from {args.in_file}")

    deduped = dedupe(examples)
    print(f"[curate] After dedupe: {len(deduped)} rows")

    surviving = [e for e in deduped if e.score >= args.min_score]
    print(f"[curate] After --min-score {args.min_score}: {len(surviving)} rows")

    surviving.sort(key=lambda e: e.score, reverse=True)
    surviving = surviving[:args.max_examples]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for ex in surviving:
            text = format_example(ex)
            f.write(json.dumps({"text": text, "score": ex.score}) + "\n")
    n_new = len(surviving)
    print(f"[curate] Wrote {n_new} training rows → {args.out}")

    if surviving:
        scores = [e.score for e in surviving]
        print(f"[curate] Score range: {min(scores):.2f} – {max(scores):.2f}  "
              f"mean={sum(scores)/len(scores):.2f}")

    # ── Anchor mix-in (Layer 5) ──────────────────────────────────────────────
    # Append random rows from the frozen v1 anchor so the model can't drift
    # away from its original training corpus over many evolve cycles.
    # Target ratio: anchor_fraction = n_anchor / (n_anchor + n_new)
    # → n_anchor = n_new * f / (1 - f).
    if args.anchor is not None and args.anchor_fraction > 0:
        if not args.anchor.exists():
            print(f"[curate] WARNING: anchor file not found at {args.anchor} — skipping mix-in.")
        elif n_new == 0:
            print(f"[curate] No new rows — skipping anchor mix-in (would be 100% anchor).")
        else:
            import random
            anchor_rows = [l for l in args.anchor.read_text().splitlines() if l.strip()]
            f = max(0.0, min(0.99, args.anchor_fraction))
            n_anchor_target = int(round(n_new * f / max(1e-6, 1 - f)))
            n_anchor = min(n_anchor_target, len(anchor_rows))
            if n_anchor == 0:
                print(f"[curate] Anchor target rounded to 0 — skipping.")
            else:
                random.seed(42)   # reproducible across cycles
                sampled = random.sample(anchor_rows, n_anchor)
                with open(args.out, "a") as out_f:
                    for line in sampled:
                        out_f.write(line.rstrip() + "\n")
                total = n_new + n_anchor
                actual_frac = n_anchor / total
                print(f"[curate] Anchor mix-in: {n_anchor}/{len(anchor_rows)} rows "
                      f"from {args.anchor.name} "
                      f"(target {f:.0%}, actual {actual_frac:.0%} of {total} total)")

    # ── Retry-pair mining for future DPO ─────────────────────────────────────
    if args.pairs_out is not None:
        pairs = mine_retry_pairs(args.attempts)
        args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.pairs_out, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        print(f"[curate] Mined {len(pairs)} retry pairs → {args.pairs_out}")
        if len(pairs) >= 200:
            print(f"[curate] You now have ≥200 pairs — DPO follow-up is unblocked.")


if __name__ == "__main__":
    main()
