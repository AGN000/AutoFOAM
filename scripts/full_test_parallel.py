#!/usr/bin/env python3
"""
Run the full PROMPT_CATALOG through the fine-tuned model end-to-end,
sharded across GPUs in parallel. Each catalog prompt is sent to the LLM
(param extraction → solver pick → mesh → case write → solver run for
end_time=3 iterations). Results aggregated to one JSONL.

Usage:
    bash scripts/run_full_test_parallel.sh [--gpus 0,1,2,3,4,5,6,7] [--end-time 3] [--timeout 120]

Or run a single shard directly (used internally by the launcher):
    python scripts/full_test_parallel.py --shard I/N --gpu G --end-time 3 --timeout 120 --out PATH
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", required=True, help="I/N — this shard index/total")
    p.add_argument("--tags-file", default="", help="If set, restrict catalog to case_tags listed in this file")
    p.add_argument("--ood-file", default="", help="JSON file of [{tag, prompt, expected_solver}] OOD cases")
    p.add_argument("--raw-llm", action="store_true",
                   help="Disable agent-side keyword guards (honest model evaluation)")
    p.add_argument("--end-time", type=float, default=3.0)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--out", required=True, help="JSONL output path for this shard")
    p.add_argument("--with-files", action="store_true",
                   help="Include case_files_text + final_residuals per row "
                        "(needed by regression_diff.py for per-prompt A/B)")
    return p.parse_args()


def main():
    args = parse_args()
    i_str, n_str = args.shard.split("/")
    shard_i, shard_n = int(i_str), int(n_str)

    # Disable noisy progress bars
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    if args.raw_llm:
        os.environ["OPENFOAM_AGENT_RAW_LLM"] = "1"

    from openfoam_agent.prompt_catalog import PROMPT_CATALOG
    from openfoam_agent.agent import OpenFOAMAgent

    if args.ood_file:
        ood_recs = json.loads(Path(args.ood_file).read_text())
        from types import SimpleNamespace
        base = [SimpleNamespace(case_tag=r["tag"], prompt=r["prompt"],
                                  expected_solver=r.get("expected_solver"))
                 for r in ood_recs]
    else:
        base = PROMPT_CATALOG
        if args.tags_file:
            wanted = set(Path(args.tags_file).read_text().split())
            base = [c for c in base if c.case_tag in wanted]
    catalog = [c for k, c in enumerate(base) if k % shard_n == shard_i]
    print(f"[shard {shard_i}/{shard_n}] {len(catalog)} cases on GPU {os.environ.get('CUDA_VISIBLE_DEVICES','?')}",
          flush=True)

    agent = OpenFOAMAgent(use_llm=True)
    # Don't append to shared dataset.json from parallel workers
    agent._save_to_dataset = lambda *a, **kw: None  # type: ignore

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for k, entry in enumerate(catalog):
            t0 = time.time()
            rec: dict = {
                "shard": shard_i,
                "case_tag": entry.case_tag,
                "prompt": entry.prompt,
                "expected_solver": getattr(entry, "expected_solver", None),
            }
            try:
                r = agent.run(
                    prompt=entry.prompt,
                    max_retries=1,
                    use_gmsh=True,
                    case_name=f"fulltest_{entry.case_tag}_{shard_i}_{k}",
                    sim_timeout=args.timeout,
                    end_time_override=args.end_time,
                )
                rec.update(
                    success=r.success,
                    score=r.score,
                    solver=r.solver,
                    feedback=(r.feedback or "")[:120],
                    error=(r.error or "")[:120],
                    elapsed=time.time() - t0,
                )
                if args.with_files:
                    # Snapshot case files + residuals for downstream
                    # per-prompt diff (regression_diff.py).
                    rec["final_residuals"] = r.residuals
                    rec["case_files_text"] = (
                        agent._read_case_files(__import__("pathlib").Path(r.case_dir))
                        if r.case_dir else ""
                    )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[shard {shard_i}] TRACEBACK for {entry.case_tag}:\n{tb}", flush=True)
                rec.update(
                    success=False, score=0.0, solver="error",
                    feedback="", error=f"EXC: {e}"[:160],
                    traceback=tb[-400:],
                    elapsed=time.time() - t0,
                )
            f.write(json.dumps(rec) + "\n")
            f.flush()
            icon = "✓" if rec.get("success") else "✗"
            print(f"[shard {shard_i}] {k+1:02d}/{len(catalog)} {icon} "
                  f"{entry.case_tag:<28} solver={rec['solver']:<18} "
                  f"score={rec['score']:.2f} t={rec['elapsed']:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
