
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openfoam_agent.config import (
    DATASET_DIR, MIN_RETRAIN_SCORE, ACTIVE_LEARNING_THRESHOLD, get_llm,
)
from openfoam_agent.solver_selector import select_solver


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", type=Path, required=True,
                   help="Eval JSONL to compute per-solver match rate from")
    p.add_argument("--n", type=int, default=30,
                   help="Number of new prompts to generate")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL of {text, score} training rows")
    p.add_argument("--threshold", type=float, default=ACTIVE_LEARNING_THRESHOLD,
                   help=f"Skip if every family is above this match-rate "
                        f"(default: {ACTIVE_LEARNING_THRESHOLD})")
    p.add_argument("--min-score", type=float, default=MIN_RETRAIN_SCORE,
                   help="Only emit rows scoring at or above this threshold")
    p.add_argument("--family", default=None,
                   help="Override auto-detected family (for testing)")
    p.add_argument("--end-time", type=float, default=10.0,
                   help="Solver iterations / time-steps for active runs")
    p.add_argument("--timeout", type=int, default=180,
                   help="Per-case wall-clock timeout (seconds)")
    p.add_argument("--adversarial", action="store_true",
                   help="Generate prompts deliberately tricky for first-try success "
                        "(unusual fluid combos, ambiguous BC hints, edge-case Re). "
                        "Used to populate attempts.jsonl with retry pairs for DPO.")
    return p.parse_args()


def detect_weakest_family(eval_path: Path, threshold: float) -> str | None:

    rows = [json.loads(l) for l in eval_path.read_text().splitlines() if l.strip()]
    by_solver: dict[str, list[bool]] = collections.defaultdict(list)
    for r in rows:
        exp = r.get("expected_solver")
        got = r.get("solver")
        if not exp:
            continue
        by_solver[exp].append(got == exp)
    if not by_solver:
        return None
    rates = [(exp, sum(hits) / len(hits), len(hits))
             for exp, hits in by_solver.items()]
    rates.sort(key=lambda t: (t[1], t[2]))
    print(f"[active] Per-family match rates ({eval_path.name}):")
    for exp, rate, n in sorted(rates, key=lambda t: t[0]):
        marker = " ← WEAKEST" if (exp, rate, n) == rates[0] else ""
        print(f"          {exp:<22} {rate:.1%}  (n={n}){marker}")
    weakest_solver, weakest_rate, _ = rates[0]
    if weakest_rate >= threshold:
        print(f"[active] All families ≥ {threshold:.0%} — no targeting needed.")
        return None
    return weakest_solver


def collect_seed_prompts(target_family: str, max_seeds: int = 8) -> list[str]:
    from openfoam_agent.prompt_catalog import PROMPT_CATALOG
    seeds = []
    for case in PROMPT_CATALOG:
        try:
            if select_solver(case.params) == target_family:
                seeds.append(case.prompt)
        except Exception:
            continue
    if len(seeds) > max_seeds:
        # Take a spread across the catalog rather than the first N
        step = max(1, len(seeds) // max_seeds)
        seeds = seeds[::step][:max_seeds]
    print(f"[active] Found {len(seeds)} seed prompts for {target_family}")
    return seeds


GEN_SYSTEM = """You are an expert CFD engineer authoring user-style natural-language \
prompts for an automated OpenFOAM agent. The agent must select the correct \
solver and produce a runnable case from the prompt.

You will be given a list of EXAMPLE prompts and a TARGET SOLVER. Your job is \
to write a fresh prompt that:
- exercises the same physics regime (so the correct OpenFOAM solver remains \
  the target),
- VARIES at least two of: geometry dimensions, Reynolds number / Mach / \
  characteristic velocity, fluid (water vs air), 2D vs 3D, BC details,
- is phrased differently from any example (paraphrase, don't copy),
- is a complete single-sentence English request — no markdown, no JSON, no \
  numbered list — just a CFD problem statement a researcher might type.

Output ONLY the new prompt, nothing else. Do not name the solver in the prompt.
"""

ADVERSARIAL_SYSTEM = """You are an expert CFD engineer authoring DELIBERATELY TRICKY \
user-style prompts for an automated OpenFOAM agent. The goal is prompts that a \
careless first-pass setup would get WRONG, so we can train the agent to handle them.

Given EXAMPLE prompts and a TARGET SOLVER, write a fresh prompt that:
- still requires the TARGET SOLVER (so the regime matches),
- uses AT LEAST ONE of these difficulty hooks:
    * borderline Reynolds number that sits exactly on a regime boundary \
      (e.g. Re=2300 for laminar/turbulent, Re=300000 for boundary-layer transition),
    * unusual fluid choice (engine oil, glycerine, mercury, R134a, kerosene) so \
      the agent's default fluid table doesn't apply,
    * ambiguous geometry hint that needs careful interpretation \
      (e.g. "thin slot 200µm wide", "axisymmetric inlet", "L-shaped manifold"),
    * unusual aspect ratio (very long pipe L/D=200, very short channel L/H=2),
    * temperature-dependent property request, free-stream turbulence intensity \
      hint, or initial condition hint not in the standard catalog,
- remains a complete, plausible single-sentence CFD request a real researcher \
  would type — NOT nonsense, NOT contradictory.

Output ONLY the new prompt. Do not name the solver. Do not add commentary.
"""


def generate_variants(llm, target_family: str, seeds: list[str], n: int,
                       adversarial: bool = False) -> list[str]:

    if not seeds:
        return []
    from vllm import SamplingParams

    temp = 1.0 if adversarial else 0.9
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=220, n=1)
    system = ADVERSARIAL_SYSTEM if adversarial else GEN_SYSTEM
    label = "adversarial" if adversarial else "variants"
    out: list[str] = []
    seen: set[str] = set()
    print(f"[active] Generating {n} {label} prompts for {target_family}…")
    while len(out) < n:
              i = len(out)
        sample = seeds[i % len(seeds): i % len(seeds) + 3] or seeds[:3]
        user = (f"TARGET SOLVER: {target_family}\n\n"
                f"EXAMPLE PROMPTS (write a new one in the same regime, do not copy):\n"
                + "\n".join(f"- {s}" for s in sample))
        chats = [{"role": "system", "content": system},
                 {"role": "user", "content": user}]
        try:
            resp = llm.chat(chats, sampling_params=sp)
        except Exception as e:
            print(f"[active] LLM error on variant {len(out)+1}: {e}")
            break
        text = resp[0].outputs[0].text.strip().strip('"').strip()
        # Single-line, drop common bullet prefixes the model adds anyway
        text = text.split("\n")[0].lstrip("- ").lstrip("* ").strip()
        if not text or len(text) < 20:
            continue
        key = text.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        print(f"[active]   variant {len(out):2d}: {text[:90]}")
    return out


def run_agent_on_prompts(prompts: list[str], end_time: float, timeout: int):

    from openfoam_agent.agent import OpenFOAMAgent
    agent = OpenFOAMAgent(use_llm=True)

    agent._save_to_dataset = lambda *a, **kw: None   # type: ignore
    for p in prompts:
        case_name = f"active_{uuid.uuid4().hex[:8]}"
        try:
            res = agent.run(prompt=p, max_retries=2,
                            case_name=case_name,
                            sim_timeout=timeout,
                            end_time_override=end_time)
            yield p, res
        except Exception as e:
            print(f"[active] agent error on {p[:60]!r}: {e}")
            continue


def main():
    args = parse_args()

    if not args.eval.exists():
        print(f"[active] ERROR: eval file not found at {args.eval}")
        sys.exit(1)

    target_family = args.family or detect_weakest_family(args.eval, args.threshold)
    if target_family is None:

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("")
        print(f"[active] No target family — wrote empty {args.out}")
        return

    seeds = collect_seed_prompts(target_family)
    if not seeds:
        print(f"[active] No catalog prompts route to {target_family} — aborting.")
        sys.exit(0)

    llm = get_llm()
    variants = generate_variants(llm, target_family, seeds, args.n,
                                  adversarial=args.adversarial)
    if not variants:
        print(f"[active] LLM produced no variants — aborting.")
        sys.exit(0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_emitted = 0
    n_off_target = 0
    t0 = time.time()
    from openfoam_agent.training import format_example
    from openfoam_agent.schemas import TrainingExample

    with open(args.out, "w") as fout:
        for k, (prompt, res) in enumerate(run_agent_on_prompts(variants,
                                                                args.end_time,
                                                                args.timeout)):
            elapsed = time.time() - t0
            mark = "✓" if res.score >= args.min_score else "✗"
            extra = ""
            if res.solver != target_family:
                n_off_target += 1
                extra = f"  off-target (got {res.solver})"
            print(f"[active] {k+1:2d}/{len(variants)} {mark} score={res.score:.2f} "
                  f"solver={res.solver} t={elapsed:.0f}s{extra}")
            if res.score < args.min_score or not res.case_dir:
                continue

            ex = TrainingExample(
                prompt=prompt,
                refined_prompt=res.refined_prompt or prompt,
                params=res.params,
                case_dir=res.case_dir,
                solver=res.solver,
                score=res.score,
                feedback=res.feedback,
                converged=bool(res.success),
                runtime=res.runtime,
                timestamp=time.time(),
                case_files_text=_read_files(res.case_dir),
            )
            fout.write(json.dumps({"text": format_example(ex),
                                    "score": res.score}) + "\n")
            n_emitted += 1

    print(f"\n[active] Emitted {n_emitted} high-score rows → {args.out}")
    print(f"[active]   {len(variants)} prompts attempted, "
          f"{n_off_target} off-target solver picks")


def _read_files(case_dir: str) -> str:

    from pathlib import Path
    parts = []
    p = Path(case_dir)
    for sub in ("system", "constant", "0"):
        d = p / sub
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file():
                try:
                    parts.append(f"### {f.relative_to(p)}\n```\n"
                                 f"{f.read_text(errors='ignore')}\n```")
                except Exception:
                    pass
    return "\n\n".join(parts)


if __name__ == "__main__":
    main()
