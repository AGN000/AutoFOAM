#!/usr/bin/env python3
"""
Test the fine-tuned OpenFOAM agent with sample prompts and score results.

Runs a set of held-out test prompts through the full pipeline (LLM param
extraction → mesh → case write → simulation) and reports scores.

Usage:
    conda run -n vllm_env python scripts/test_inference.py [--model PATH]

Options:
    --model PATH    Model to test (default: config.LLM_MODEL, the fine-tuned merged model)
    --tag TAG       Only run prompts whose tag starts with TAG
    --timeout S     Simulation timeout per case (default: 120)
    --n N           Number of test prompts to run (default: all)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Held-out test prompts (not in training catalog) ─────────────────────────
TEST_PROMPTS = [
    # Geometry / solver inference tests (agent must choose correct solver)
    ("cavity_holdout",
     "Run a 2D square cavity simulation with a moving lid at Re=1000, 1m domain, air"),
    ("pipe_holdout",
     "Simulate turbulent flow in a 3D circular pipe: diameter 4cm, length 40cm, Re=30000"),
    ("cylinder_holdout",
     "CFD simulation of 2D flow past a circular cylinder at Reynolds number 150, D=0.1m"),
    ("bfs_holdout",
     "Backward-facing step simulation: step height 8cm, Re=600, 2D laminar, air"),
    ("channel_holdout",
     "Turbulent 2D channel flow, Re=8000, height=0.08m, length=4m, kOmegaSST"),

    # Solver selection tests (agent must infer solver from physics description)
    ("transient_infer",
     "Unsteady vortex shedding behind a 2D cylinder, Re=200, diameter=10cm — capture time-varying wake"),
    ("compressible_infer",
     "High-speed air flow at Mach 0.5 through a 2D channel, L=2m H=0.1m, steady"),
    ("buoyancy_infer",
     "Natural convection in a heated square room 3m×3m, hot wall on left side, air"),
    ("multiphase_infer",
     "Water column collapse in a 2D box 4m wide 2m tall, dam break simulation"),

    # Unit / dimension tests
    ("units_cm",
     "Pipe flow: diameter=5cm, length=50cm, inlet velocity=10m/s, turbulent air"),
    ("units_mm",
     "Laminar flow in a channel: height=50mm, length=500mm, Re=1500, water"),

    # Expert phrasing tests
    ("expert_naca",
     "RANS kOmegaSST simulation of NACA0012, chord=0.5m, AoA=4°, Re=800000, air"),
    ("expert_wedge",
     "Axisymmetric pipe flow using wedge geometry, D=20mm, L=300mm, Re=800, Hagen-Poiseuille"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="", help="Override model path")
    p.add_argument("--tag", default="", help="Filter by tag prefix")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--n", type=int, default=0, help="Max prompts to run (0=all)")
    return p.parse_args()


def main():
    args = parse_args()

    prompts = TEST_PROMPTS
    if args.tag:
        prompts = [(t, p) for t, p in prompts if t.startswith(args.tag)]
    if args.n:
        prompts = prompts[:args.n]

    print(f"\n[test] Running {len(prompts)} inference tests")
    print(f"[test] Timeout per case: {args.timeout}s\n")

    # Optionally override model
    if args.model:
        import openfoam_agent.config as cfg
        cfg.LLM_MODEL = args.model
        print(f"[test] Using model: {args.model}\n")

    from openfoam_agent.agent import OpenFOAMAgent
    agent = OpenFOAMAgent(use_llm=True)  # LLM ON for inference test

    results = []
    for tag, prompt in prompts:
        print(f"[test] {tag}: {prompt[:65]}...")
        t0 = time.time()
        try:
            result = agent.run(
                prompt=prompt,
                max_retries=1,
                use_gmsh=True,
                case_name=f"test_{tag}",
                sim_timeout=args.timeout,
            )
            elapsed = time.time() - t0
            icon = "✓" if result.success else "✗"
            print(f"       {icon} score={result.score:.2f}  solver={result.solver}  "
                  f"t={elapsed:.0f}s  {result.feedback[:50]}")
            results.append((tag, result.score, result.solver, result.success, elapsed))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"       ✗ EXCEPTION: {e}")
            results.append((tag, 0.0, "error", False, elapsed))

    # Summary table
    passed = [r for r in results if r[3]]
    print(f"\n{'='*68}")
    print(f"  Inference Test Results  —  {len(passed)}/{len(results)} PASSED")
    print(f"{'='*68}")
    print(f"  {'TAG':<25} {'SOLVER':<18} {'SCORE':<7} {'OK'}")
    print(f"  {'-'*60}")
    for tag, score, solver, ok, elapsed in results:
        icon = "✓" if ok else "✗"
        print(f"  {tag:<25} {solver:<18} {score:<7.2f} {icon}  ({elapsed:.0f}s)")
    if results:
        avg = sum(r[1] for r in results) / len(results)
        print(f"\n  Average score : {avg:.3f}")
    print(f"{'='*68}\n")

    sys.exit(0 if len(passed) >= len(results) * 0.7 else 1)


if __name__ == "__main__":
    main()
