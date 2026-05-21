from __future__ import annotations

import json
from pathlib import Path

from .config import TUTORIALS_DIR, CHROMA_DIR
from .agent import SEED_PROMPTS


def launch_ui(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    import gradio as gr
    from .agent import OpenFOAMAgent

    agent = OpenFOAMAgent(use_llm=True)

    def run_simulation(prompt: str, use_gmsh: bool, max_retries: int, sim_timeout: int):
        if not prompt.strip():
            return "Please enter a prompt.", "", "{}", 0.0, ""
        try:
            result = agent.run(
                prompt=prompt,
                use_gmsh=use_gmsh,
                max_retries=int(max_retries),
                sim_timeout=int(sim_timeout),
            )
            params_json = result.params.model_dump_json(indent=2) if result.params else "{}"
            rag_str = ", ".join(result.rag_examples_used) if result.rag_examples_used else "none"
            log_preview = ""
            if result.case_dir:
                log_file = Path(result.case_dir) / "agent.log"
                if log_file.exists():
                    log_preview = log_file.read_text()[-5000:]
            return (
                result.refined_prompt,
                f"Solver: {result.solver} | Attempt: {result.attempt + 1} | Similar cases: {rag_str}",
                params_json,
                result.score,
                log_preview,
            )
        except Exception as e:
            return str(e), "", "{}", 0.0, ""

    def collect_data(n_prompts: int, min_score: float):
        from .agent import SEED_PROMPTS
        from .training import collect_training_episodes
        prompts = SEED_PROMPTS[:int(n_prompts)]
        examples = collect_training_episodes(agent, prompts, min_score=min_score)
        return f"Collected {len(examples)} training examples (score >= {min_score})"

    def start_training(min_score: float, max_examples: int, epochs: int):
        from .training import train_qlora
        try:
            train_qlora(min_score=min_score, max_examples=int(max_examples), num_epochs=int(epochs))
            return "Training complete. Adapter saved to data/checkpoints/qwen_coder_14b_lora/final_adapter"
        except Exception as e:
            return f"Training failed: {e}"

    def browse_tutorial(case_name: str):
        case_dir = TUTORIALS_DIR / case_name
        if not case_dir.exists():
            return "Case not found."
        parts = []
        for f_name in ("README.md", "README.txt", "README"):
            fp = case_dir / f_name
            if fp.exists():
                parts.append(fp.read_text(errors="ignore")[:2000])
                break
        ctrl = case_dir / "system" / "controlDict"
        if ctrl.exists():
            parts.append(f"\n--- system/controlDict ---\n{ctrl.read_text(errors='ignore')[:1000]}")
        return "\n".join(parts) if parts else "No README found."

    tutorial_names = sorted([d.name for d in TUTORIALS_DIR.iterdir() if d.is_dir()]) if TUTORIALS_DIR.exists() else []

    with gr.Blocks(title="OpenFOAM AI Agent") as demo:
        gr.Markdown("# OpenFOAM AI Agent\n_Prompt → Mesh → Simulation → Results_")

        with gr.Tab("Run Simulation"):
            with gr.Row():
                prompt_in = gr.Textbox(
                    label="Describe your simulation",
                    lines=3,
                    placeholder="e.g. turbulent flow in a 2D channel at Re=50000",
                )
            with gr.Row():
                use_gmsh = gr.Checkbox(label="Use gmsh mesh", value=True)
                max_retries = gr.Slider(1, 5, value=3, step=1, label="Max retries")
                sim_timeout = gr.Slider(60, 600, value=300, step=30, label="Timeout (s)")
            run_btn = gr.Button("Generate & Run", variant="primary")

            refined_out = gr.Textbox(label="Refined Prompt", lines=3)
            info_out = gr.Textbox(label="Solver / Info")
            with gr.Row():
                params_out = gr.Code(label="Extracted Parameters (JSON)", language="json")
                score_out = gr.Number(label="Score (0–1)")
            log_out = gr.Textbox(label="Simulation Log (last 5000 chars)", lines=15, max_lines=25)

            run_btn.click(
                run_simulation,
                inputs=[prompt_in, use_gmsh, max_retries, sim_timeout],
                outputs=[refined_out, info_out, params_out, score_out, log_out],
            )

        with gr.Tab("Train Model"):
            gr.Markdown("### Collect training data and fine-tune the model")
            with gr.Row():
                n_prompts = gr.Slider(1, len(SEED_PROMPTS) or 10, value=5, step=1, label="Seed prompts to use")
                min_score_collect = gr.Slider(0.3, 0.9, value=0.5, step=0.05, label="Min score to collect")
            collect_btn = gr.Button("Collect Training Data")
            collect_out = gr.Textbox(label="Collection result")
            collect_btn.click(collect_data, inputs=[n_prompts, min_score_collect], outputs=[collect_out])

            gr.Markdown("---")
            with gr.Row():
                min_score_train = gr.Slider(0.3, 0.9, value=0.6, step=0.05, label="Min score for training")
                max_examples_in = gr.Slider(10, 500, value=100, step=10, label="Max examples")
                epochs_in = gr.Slider(1, 5, value=2, step=1, label="Epochs")
            train_btn = gr.Button("Start QLoRA Training", variant="primary")
            train_out = gr.Textbox(label="Training result")
            train_btn.click(start_training, inputs=[min_score_train, max_examples_in, epochs_in], outputs=[train_out])

        with gr.Tab("Tutorial Browser"):
            gr.Markdown("### Browse downloaded OpenFOAM tutorial cases")
            case_dropdown = gr.Dropdown(choices=tutorial_names, label="Select tutorial case")
            browse_btn = gr.Button("Load")
            case_out = gr.Textbox(label="Case info", lines=20, max_lines=40)
            browse_btn.click(browse_tutorial, inputs=[case_dropdown], outputs=[case_out])

    demo.queue()
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    launch_ui()
