from __future__ import annotations

from pathlib import Path
from typing import Optional

OPENFOAM_BASHRC = "/home/nvidia/miniconda3/envs/openfoam2412/etc/bashrc"
TUTORIALS_DIR = Path("/data/foamllm2/github/OpenFOAM_Tutorials_")
PROJECT_ROOT = Path("/data/foamllm3/openfoam_agent")

CASES_DIR = PROJECT_ROOT / "data/cases"
LOGS_DIR = PROJECT_ROOT / "data/logs"
DATASET_DIR = PROJECT_ROOT / "data/dataset"
CHECKPOINTS_DIR = PROJECT_ROOT / "data/checkpoints"
CHROMA_DIR = PROJECT_ROOT / "data/chroma_db"

LLM_MODEL = "/data/foamllm3/openfoam_agent/data/checkpoints/qwen_coder_14b_merged"
# Layer-3 evolve.sh sets this to point the current subprocess at a candidate
# adapter for the regression-gate eval, without touching the production
# config.py value.
import os as _os_override
LLM_MODEL = _os_override.environ.get("OPENFOAM_AGENT_LLM_OVERRIDE", LLM_MODEL)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_SEQ_LEN = 8192
import os as _os
VLLM_GPU_MEMORY_FRACTION = float(_os.environ.get("VLLM_GPU_MEM_FRAC", "0.85"))
VLLM_MAX_NUM_SEQS = int(_os.environ.get("VLLM_MAX_NUM_SEQS", "256"))

# ── Self-evolution knobs (Layer 1 / 2 / 3) ──────────────────────────────────
# Score below this triggers a Layer-1 self-correction retry. Set ABOVE the
# 0.5 capture threshold so that any borderline run (didn't crash, but isn't
# clean either) still gets a retry attempt with enriched failure context.
# This is also what feeds Layer-4 DPO: the (low first attempt, higher retry)
# pair on the same prompt is exactly the preference signal DPO needs.
# Was 0.4 originally — moved to 0.7 once we learned that anything 0.5–0.7
# almost never triggered retry in practice, starving DPO of pairs.
RETRY_SCORE_THRESHOLD = float(_os.environ.get("RETRY_SCORE_THRESHOLD", "0.7"))
# Minimum score for a row to enter the curated training corpus (Layer 2).
# Higher than the 0.5 capture threshold so we only train on *good* runs.
MIN_RETRAIN_SCORE = float(_os.environ.get("MIN_RETRAIN_SCORE", "0.65"))
# Fire evolve.sh once dataset.json has accumulated this many new high-score
# rows since the last successful evolution cycle (Layer 3).
EVOLUTION_BATCH_SIZE = int(_os.environ.get("EVOLUTION_BATCH_SIZE", "25"))
# Pinned baseline for the regression gate. evolve.sh refuses to swap if a
# fresh adapter regresses below these numbers on data/eval/ood_100_v2.json.
EVAL_GATE_FILE = PROJECT_ROOT / "data/eval/regression_gate.json"
# Where every retry attempt (success or fail) is appended for retry-pair
# mining. Distinct from dataset.json (curated, score≥0.5 only).
ATTEMPTS_LOG = DATASET_DIR / "attempts.jsonl"

# ── Anti-collapse defenses (Layers 5/6/7) ───────────────────────────────────
# Frozen v1 corpus (the original 402 rows the 14 B model was first trained on).
# Mixed back into every evolve cycle to prevent forgetting. NEVER overwrite
# this file — curate_dataset.py writes to its own versioned output by default.
ANCHOR_DATASET = DATASET_DIR / "anchor_v1_402.jsonl"
# Fraction of the training mix that should come from the anchor on each cycle.
# 0.0 disables; 0.3 means ~30 % anchor + 70 % new high-score rows.
EVOLUTION_ANCHOR_FRACTION = float(_os.environ.get("EVOLUTION_ANCHOR_FRACTION", "0.3"))
# When 1, evolve.sh runs active_learning.py to author + score new prompts in
# the weakest solver family from the previous eval. Skipped automatically if
# the previous eval had no family below the active-learning threshold.
EVOLUTION_ACTIVE_LEARNING = int(_os.environ.get("EVOLUTION_ACTIVE_LEARNING", "1"))
# Match-rate below this triggers active-learning targeting for that family.
ACTIVE_LEARNING_THRESHOLD = float(_os.environ.get("ACTIVE_LEARNING_THRESHOLD", "0.95"))

_llm_instance = None


def get_llm():
    global _llm_instance
    if _llm_instance is None:
        from vllm import LLM
        _llm_instance = LLM(
            model=LLM_MODEL,
            max_model_len=MAX_SEQ_LEN,
            gpu_memory_utilization=VLLM_GPU_MEMORY_FRACTION,
            max_num_seqs=VLLM_MAX_NUM_SEQS,
            dtype="bfloat16",
        )
    return _llm_instance


def get_unsloth_model(load_for_training: bool = False):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LLM_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    if load_for_training:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    return model, tokenizer
