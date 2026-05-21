#!/usr/bin/env python3
"""Interactive REPL for the OpenFOAM agent.

Loads vLLM once, then accepts prompts in a loop until EOF / 'quit'.

Usage:
    bash scripts/repl.sh                    # auto-pick GPU
    GPU=4 bash scripts/repl.sh              # specific GPU

Inside the REPL:
    prompt> 2D lid-driven cavity Re=1000, 2m square, water
    ...
    prompt> flow over NACA0012, chord 0.5m, AoA 5 deg, Re=1e6
    ...
    prompt> quit         (or Ctrl-D)
"""
from __future__ import annotations

import sys
import time
import os
from pathlib import Path

# Enable arrow-key editing, ↑/↓ history recall, and persistent history.
# Without this, Python's input() is line-mode-only and arrow keys produce
# escape sequences instead of moving the cursor.
try:
    import readline
    _hist = Path.home() / ".openfoam_agent_repl_history"
    try:
        readline.read_history_file(str(_hist))
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)
    import atexit
    atexit.register(readline.write_history_file, str(_hist))
except ImportError:
    pass  # Windows fallback (won't have arrow keys but won't crash)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _print_banner():
    print()
    print("=" * 70)
    print("  OpenFOAM Agent — Interactive REPL")
    print("=" * 70)
    print("  Type your CFD request and press Enter.")
    print("  Special commands:")
    print("    quit / exit / Ctrl-D   leave the REPL")
    print("    last                   re-print the last result")
    print("    cases                  list recent case directories")
    print("    timeout=N              set solver timeout in seconds (default 300)")
    print("    retries=N              set retry count (default 1)")
    print("=" * 70)
    print()


def main():
    print("[repl] loading vLLM 14B model on configured GPU... (~60s first time)")
    t0 = time.time()
    from openfoam_agent.agent import OpenFOAMAgent
    from openfoam_agent.config import CASES_DIR
    agent = OpenFOAMAgent(use_llm=True)
    agent._init_components()  # actually load the model now, not on first prompt
    print(f"[repl] model ready in {time.time()-t0:.1f}s")

    _print_banner()

    timeout = 300
    retries = 1
    last_result = None
    n = 0

    while True:
        try:
            line = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[repl] bye")
            return

        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            print("[repl] bye")
            return
        if line == "last":
            if last_result is None:
                print("(no previous result yet)")
            else:
                r = last_result
                print(f"  score   : {r.score:.2f}")
                print(f"  solver  : {r.solver}")
                print(f"  case    : {r.case_dir}")
                print(f"  feedback: {r.feedback}")
            continue
        if line == "cases":
            cases = sorted(Path(CASES_DIR).glob("*_attempt*"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
            for c in cases[:10]:
                print(f"  {c}")
            continue
        if line.startswith("timeout="):
            try:
                timeout = int(line.split("=", 1)[1])
                print(f"[repl] timeout set to {timeout}s")
            except ValueError:
                print("usage: timeout=300")
            continue
        if line.startswith("retries="):
            try:
                retries = int(line.split("=", 1)[1])
                print(f"[repl] retries set to {retries}")
            except ValueError:
                print("usage: retries=1")
            continue

        n += 1
        case_name = f"repl_{n:03d}"
        print(f"[repl] running... (timeout={timeout}s, retries={retries})")
        t0 = time.time()
        try:
            result = agent.run(
                prompt=line,
                max_retries=retries,
                use_gmsh=True,
                case_name=case_name,
                sim_timeout=timeout,
            )
            last_result = result
            elapsed = time.time() - t0
            icon = "✓" if result.success else "✗"
            print()
            print(f"  {icon} score   : {result.score:.2f}")
            print(f"    solver  : {result.solver}")
            print(f"    case    : {result.case_dir}")
            print(f"    feedback: {result.feedback[:120]}")
            print(f"    elapsed : {elapsed:.0f}s")
            print()
        except Exception as e:
            print(f"[repl] EXCEPTION: {e}")


if __name__ == "__main__":
    main()
