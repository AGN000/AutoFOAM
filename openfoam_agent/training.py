from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .config import LLM_MODEL, DATASET_DIR, CHECKPOINTS_DIR, MAX_SEQ_LEN
from .schemas import TrainingExample, CFDParams
from .agent import SEED_PROMPTS



EXPERT_SYSTEM_PROMPT = """\
You are an expert OpenFOAM CFD engineer with deep knowledge of computational \
fluid dynamics, turbulence modelling, mesh generation, and numerical methods. \
When given a simulation request you:

1. Analyse the flow physics: geometry, Reynolds number, flow regime, \
   compressibility, heat transfer, multiphase, and turbulence model.
2. Choose the correct OpenFOAM solver based on problem physics:
   - simpleFoam      : steady-state incompressible
   - icoFoam         : transient laminar incompressible (Re < 2300)
   - pimpleFoam      : transient turbulent incompressible
   - rhoSimpleFoam   : steady compressible (density varies, e.g. Ma > 0.3)
   - rhoPimpleFoam   : transient compressible
   - buoyantSimpleFoam  : steady with buoyancy / heat transfer
   - buoyantPimpleFoam  : transient with buoyancy / heat transfer
   - interFoam       : VOF two-phase / free-surface (water + air)
   If the user specifies a solver, use it; otherwise infer from the physics.
3. Select turbulence model: laminar (Re<2300), kOmegaSST (most turbulent cases), \
   kEpsilon (high-Re free-shear flows).
4. Specify mesh strategy and boundary conditions for each patch.
5. Generate complete, production-quality OpenFOAM case files for 0/, constant/, \
   and system/ directories.

Your response format:
## CFD Analysis
Brief engineering rationale (solver choice, turbulence model, mesh notes, \
expected physics).

## OpenFOAM Case Files
Complete file contents for all required OpenFOAM dictionaries."""


def _build_expert_analysis(example: TrainingExample) -> str:
    p: Optional[CFDParams] = example.params
    if p is None:
        return f"Solver: {example.solver}. Score: {example.score:.2f}."

    re = p.reynolds_number
    regime = p.flow_regime.value if p.flow_regime else "unknown"
    turb = p.turbulence_model.value if p.turbulence_model else "laminar"
    dim = "3D" if p.is_3d else "2D"
    geom = p.geometry_type.value if p.geometry_type else "box"
    re_str = f"{re:.4g}" if re is not None else "n/a"

    lines = [
        f"**Geometry:** {geom} ({dim}), "
        f"L={p.length:.3g}m × W={p.width:.3g}m"
        + (f" × H={p.height:.3g}m" if p.is_3d else ""),
        f"**Fluid:** {'air' if p.kinematic_viscosity > 5e-6 else 'water'} "
        f"(ν={p.kinematic_viscosity:.2e} m²/s, ρ={p.density:.3g} kg/m³)",
        f"**Re = {re_str}** → {regime} regime",
        f"**Solver:** {example.solver} "
        f"({'transient' if p.is_transient else 'steady'}, "
        f"{'compressible' if p.is_compressible else 'multiphase VOF' if p.is_multiphase else 'incompressible'})",
        f"**Turbulence:** {turb}" + (
            " — kOmegaSST uses omegaWallFunction / kqRWallFunction on walls"
            if turb == "kOmegaSST" else
            " — no turbulence model (laminar)"
        ),
        f"**Simulation score:** {example.score:.2f} (converged={example.converged})",
    ]
    if example.feedback:
        lines.append(f"**Notes:** {example.feedback}")
    return "\n".join(lines)




_QWEN_CHAT_TEMPLATE = """\
<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{user}
<|im_end|>
<|im_start|>assistant
## CFD Analysis

{analysis}

## OpenFOAM Case Files

{case_files}
<|im_end|>"""


_ALPACA_TEMPLATE = """\
### Instruction:
{instruction}

### Context (similar cases):
{context}

### Response:
{response}"""


def format_example(example: TrainingExample, rag_context: str = "") -> str:

    analysis = _build_expert_analysis(example)
    case_files = example.case_files_text or "(no case files captured)"
    return _QWEN_CHAT_TEMPLATE.format(
        system=EXPERT_SYSTEM_PROMPT,
        user=example.refined_prompt or example.prompt,
        analysis=analysis,
        case_files=case_files,
    )




def format_dpo_prompt(user_prompt: str) -> str:
    """The Qwen-chat prefix up to (but not including) the assistant content."""
    return (f"<|im_start|>system\n{EXPERT_SYSTEM_PROMPT}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n")


def format_dpo_response(analysis: str, case_files: str) -> str:
    """The assistant-side of the chat (everything after the assistant tag)."""
    return f"## CFD Analysis\n\n{analysis}\n\n## OpenFOAM Case Files\n\n{case_files}<|im_end|>"


def format_example_alpaca(example: TrainingExample, rag_context: str = "") -> str:
    return _ALPACA_TEMPLATE.format(
        instruction=example.refined_prompt or example.prompt,
        context=rag_context or "No similar cases found.",
        response=example.case_files_text,
    )


def load_dataset(
    dataset_path: Optional[Path] = None,
    min_score: float = 0.5,
    max_examples: int = 500,
) -> list[TrainingExample]:
    dataset_path = dataset_path or DATASET_DIR / "dataset.json"
    if not dataset_path.exists():
        return []
    raw = json.loads(dataset_path.read_text())
    examples = [TrainingExample.model_validate(e) for e in raw]
    examples = [e for e in examples if e.score >= min_score]
    examples.sort(key=lambda e: e.score, reverse=True)
    return examples[:max_examples]


def compute_reward_weights(examples: list[TrainingExample]) -> list[float]:
    weights = [e.score ** 2 for e in examples]
    mean_w = sum(weights) / len(weights) if weights else 1.0
    return [w / mean_w for w in weights]


def collect_training_episodes(
    agent,
    prompts: Optional[list[str]] = None,
    min_score: float = 0.5,
) -> list[TrainingExample]:
    prompts = prompts or SEED_PROMPTS
    collected = []
    for i, prompt in enumerate(prompts):
        print(f"\n[training] Episode {i + 1}/{len(prompts)}: {prompt[:60]}")
        result = agent.run(prompt, max_retries=2)
        print(f"[training] Score: {result.score:.2f}")
        if result.score >= min_score:
            collected.append(TrainingExample(
                prompt=prompt,
                refined_prompt=result.refined_prompt,
                params=result.params,
                case_dir=result.case_dir,
                solver=result.solver,
                score=result.score,
                feedback=result.feedback,
                converged=result.success,
                runtime=result.runtime,
                timestamp=__import__("time").time(),
                case_files_text=agent._read_case_files(Path(result.case_dir)) if result.case_dir else "",
            ))
    return collected


def make_reward_weighted_trainer(base_trainer_cls):


    class RewardWeightedSFTTrainer(base_trainer_cls):
        def __init__(self, reward_weights: list[float], *args, **kwargs):
            super().__init__(*args, **kwargs)
            import torch
            self._reward_weights = torch.tensor(reward_weights, dtype=torch.float32)
            self._step = 0

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            result = super().compute_loss(model, inputs,
                                          return_outputs=return_outputs, **kwargs)
            loss_val = result[0] if return_outputs else result
            if len(self._reward_weights) > 0:
                idx = self._step % len(self._reward_weights)
                w = self._reward_weights[idx].to(loss_val.device)
                loss_val = loss_val * w
                self._step += 1
            return (loss_val, result[1]) if return_outputs else loss_val

    return RewardWeightedSFTTrainer


def train_qlora(
    base_model: str = LLM_MODEL,
    dataset_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    min_score: float = 0.6,
    max_examples: int = 500,
    num_epochs: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    output_dir = output_dir or CHECKPOINTS_DIR / "qwen_coder_14b_lora"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load examples
    examples = load_dataset(dataset_path, min_score=min_score, max_examples=max_examples)
    if not examples:
        print("[training] No training examples found. Run collect_training_episodes first.")
        return

    print(f"[training] Training on {len(examples)} examples (score >= {min_score})")
    reward_weights = compute_reward_weights(examples)

    # Format dataset
    texts = [format_example(e) for e in examples]
    dataset = Dataset.from_dict({"text": texts, "reward": reward_weights})

    # Load model with QLoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_8bit",
        bf16=True,
        max_seq_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        logging_steps=10,
        save_steps=100,
        report_to="none",
        dataloader_num_workers=4,
    )

    WeightedTrainer = make_reward_weighted_trainer(SFTTrainer)
    trainer = WeightedTrainer(
        reward_weights=reward_weights,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("[training] Starting QLoRA fine-tuning...")
    trainer.train()
    adapter_dir = output_dir / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[training] Adapter saved to {adapter_dir}")
