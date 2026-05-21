#!/usr/bin/env python3
"""CLI entry point for the OpenFOAM AI agent."""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openfoam_agent.agent import OpenFOAMAgent, SEED_PROMPTS


def main():
    parser = argparse.ArgumentParser(description="OpenFOAM AI Agent")
    subparsers = parser.add_subparsers(dest="command")

    # run command
    run_p = subparsers.add_parser("run", help="Run a simulation from a prompt")
    run_p.add_argument("prompt", nargs="?", help="Simulation prompt")
    run_p.add_argument("--file", type=Path, help="File with one prompt per line")
    run_p.add_argument("--no-gmsh", action="store_true", help="Use blockMesh instead of gmsh")
    run_p.add_argument("--retries", type=int, default=3)
    run_p.add_argument("--timeout", type=int, default=300)
    run_p.add_argument("--no-llm", action="store_true", help="Skip LLM (use fallback params)")

    # train command
    train_p = subparsers.add_parser("train", help="Fine-tune the model on collected data")
    train_p.add_argument("--min-score", type=float, default=0.6)
    train_p.add_argument("--max-examples", type=int, default=500)
    train_p.add_argument("--epochs", type=int, default=2)

    # collect command
    collect_p = subparsers.add_parser("collect", help="Collect training data using seed prompts")
    collect_p.add_argument("--n", type=int, default=len(SEED_PROMPTS), help="Number of seed prompts")

    # ui command
    ui_p = subparsers.add_parser("ui", help="Launch Gradio interface")
    ui_p.add_argument("--port", type=int, default=7860)
    ui_p.add_argument("--share", action="store_true")

    # index command
    idx_p = subparsers.add_parser("index", help="Index tutorials into ChromaDB")

    args = parser.parse_args()

    if args.command == "run":
        prompts = []
        if args.prompt:
            prompts = [args.prompt]
        elif args.file and args.file.exists():
            prompts = [l.strip() for l in args.file.read_text().splitlines() if l.strip()]
        else:
            print("Error: provide a prompt or --file")
            sys.exit(1)

        agent = OpenFOAMAgent(use_llm=not args.no_llm)
        for prompt in prompts:
            print(f"\nRunning: {prompt}")
            result = agent.run(
                prompt,
                use_gmsh=not args.no_gmsh,
                max_retries=args.retries,
                sim_timeout=args.timeout,
            )
            print(f"Score: {result.score:.2f} | Solver: {result.solver} | {result.feedback}")
            if result.case_dir:
                print(f"Case: {result.case_dir}")

    elif args.command == "train":
        from openfoam_agent.training import train_qlora
        train_qlora(min_score=args.min_score, max_examples=args.max_examples, num_epochs=args.epochs)

    elif args.command == "collect":
        from openfoam_agent.training import collect_training_episodes
        agent = OpenFOAMAgent(use_llm=True)
        examples = collect_training_episodes(agent, SEED_PROMPTS[:args.n])
        print(f"\nCollected {len(examples)} training examples")

    elif args.command == "ui":
        from openfoam_agent.ui import launch_ui
        launch_ui(port=args.port, share=args.share)

    elif args.command == "index":
        from scripts.index_tutorials import main as index_main
        index_main()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
