#!/usr/bin/env python3
"""
Validate the full RAG → mesh → case-write → solver pipeline for all
catalog prompt cases.

Each case runs OpenFOAM for only 2 iterations (end_time=2) — enough to confirm
the solver starts cleanly with no FOAM FATAL ERROR.  Full convergence is NOT
the goal here; a score of any value with no crash is a PASS.

Usage:
    conda run -n vllm_env python scripts/validate_rag_pipeline.py [--tag TAG]

Options:
    --tag TAG   Only run cases whose case_tag starts with TAG (e.g. "pipe").
    --timeout S Simulation timeout in seconds per case (default 60).
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openfoam_agent.prompt_catalog import PROMPT_CATALOG, PromptCase


@dataclass
class ValidationResult:
    case_tag: str
    prompt: str
    solver: str
    rag_hits: list[str]
    started: bool        # OpenFOAM launched without FATAL ERROR
    score: float
    error: str
    elapsed: float


_PASS = "✓"
_FAIL = "✗"
_WARN = "~"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="", help="Filter by case_tag prefix")
    p.add_argument("--timeout", type=int, default=60)
    return p.parse_args()


def validate_case(
    agent,
    entry: PromptCase,
    timeout: int,
) -> ValidationResult:
    t0 = time.time()
    result = agent.run_with_params(
        prompt=entry.prompt,
        params=entry.params,
        max_retries=1,
        use_gmsh=True,
        case_name=f"val_{entry.case_tag}",
        sim_timeout=timeout,
        end_time_override=2.0,   # only 2 time-steps — smoke test
    )
    elapsed = time.time() - t0

    # "started" = no FOAM FATAL ERROR (even partial run is fine for validation)
    fatal = "FOAM FATAL" in result.error.upper() if result.error else False
    started = not fatal

    return ValidationResult(
        case_tag=entry.case_tag,
        prompt=entry.prompt,
        solver=result.solver,
        rag_hits=result.rag_examples_used,
        started=started,
        score=result.score,
        error=result.error[:80] if result.error else "",
        elapsed=elapsed,
    )


def print_summary(results: list[ValidationResult]):
    passed = [r for r in results if r.started]
    failed = [r for r in results if not r.started]

    # Group by geometry type
    groups: dict[str, list[ValidationResult]] = {}
    for r in results:
        geom = r.case_tag.split("_")[0]
        groups.setdefault(geom, []).append(r)

    print(f"\n{'='*72}")
    print(f"  RAG + Pipeline Validation  —  {len(passed)}/{len(results)} PASSED")
    print(f"{'='*72}")

    header = f"  {'TAG':<28} {'SOLVER':<12} {'RAG HITS':<24} {'OK':<4} {'SCORE':<6} {'t(s)'}"
    print(header)
    print(f"  {'-'*68}")

    for geom, group in sorted(groups.items()):
        print(f"  [{geom.upper()}]")
        for r in group:
            icon = _PASS if r.started else _FAIL
            rag_str = ", ".join(r.rag_hits[:2]) if r.rag_hits else "—"
            if len(rag_str) > 22:
                rag_str = rag_str[:22] + "…"
            print(
                f"  {r.case_tag:<28} {r.solver:<12} {rag_str:<24} "
                f"{icon:<4} {r.score:<6.2f} {r.elapsed:.0f}s"
            )
            if not r.started and r.error:
                print(f"    ↳ ERROR: {r.error}")
        print()

    print(f"  PASSED : {len(passed)}")
    print(f"  FAILED : {len(failed)}")
    if failed:
        print(f"\n  Failed cases:")
        for r in failed:
            print(f"    {r.case_tag}: {r.error}")
    print(f"{'='*72}\n")


def main():
    args = parse_args()

    catalog = PROMPT_CATALOG
    if args.tag:
        catalog = [c for c in catalog if c.case_tag.startswith(args.tag)]
        print(f"[val] Filtered to {len(catalog)} cases matching '{args.tag}'")

    print(f"[val] Validating {len(catalog)} cases  (end_time=2, timeout={args.timeout}s)\n")

    from openfoam_agent.agent import OpenFOAMAgent
    agent = OpenFOAMAgent(use_llm=False)

    results: list[ValidationResult] = []
    for i, entry in enumerate(catalog):
        print(f"[val] {i+1:02d}/{len(catalog)}  {entry.case_tag:<30}", end=" ", flush=True)
        try:
            r = validate_case(agent, entry, args.timeout)
            icon = _PASS if r.started else _FAIL
            rag_str = ",".join(r.rag_hits[:2]) if r.rag_hits else "no-rag"
            print(f"{icon}  score={r.score:.2f}  rag=[{rag_str}]  {r.elapsed:.0f}s")
        except Exception as e:
            r = ValidationResult(
                case_tag=entry.case_tag,
                prompt=entry.prompt,
                solver="unknown",
                rag_hits=[],
                started=False,
                score=0.0,
                error=str(e)[:80],
                elapsed=time.time(),
            )
            print(f"{_FAIL}  EXCEPTION: {e}")
        results.append(r)

    print_summary(results)

    # Exit non-zero if any case failed startup
    failed = [r for r in results if not r.started]
    sys.exit(len(failed))


if __name__ == "__main__":
    main()
