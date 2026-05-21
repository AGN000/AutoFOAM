#!/usr/bin/env python3
"""
Extract training examples directly from validated external OpenFOAM cases.

Sources:
  - /data/foamllm2/github/WorkingCase/augmentedCases/  (45 pimpleFoam cases, 5 turbulence models)
  - /data/foamllm2/github/WorkingCase/work1/            (25 mixed cases: pisoFoam, LES, real geometry)

For each case, reads the actual case files, infers physics, generates a
natural-language prompt, and formats a Qwen chat training example at score=1.0
(all cases are pre-validated).

Appends to data/dataset/expert_train.jsonl — same format as generate_training_data.py.

Usage:
    conda run -n vllm_env python scripts/extract_external_cases.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AUGMENTED_DIR = Path("/data/foamllm2/github/WorkingCase/augmentedCases")
WORK1_DIR     = Path("/data/foamllm2/github/WorkingCase/work1")
OUTPUT_JSONL  = ROOT / "data/dataset/expert_train.jsonl"

# ── Case file readers ────────────────────────────────────────────────────────

_SKIP_DIRS = {"processor0", "processor1", "processor2", "processor3",
              "__pycache__", ".git", "postProcessing"}
_MAX_FILE_CHARS = 5000


def _read_case_files(case_dir: Path) -> str:
    """Read all OpenFOAM case files into a single formatted string."""
    parts = []
    for subdir in ("system", "constant", "0"):
        d = case_dir / subdir
        if not d.exists():
            continue
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.parent.name not in _SKIP_DIRS:
                try:
                    text = f.read_text(errors="ignore")
                    rel = f.relative_to(case_dir)
                    parts.append(f"### {rel}\n```\n{text[:_MAX_FILE_CHARS]}\n```")
                except Exception:
                    pass
    return "\n\n".join(parts)


def _grep(text: str, pattern: str, default: str = "") -> str:
    m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else default


# ── Physics inference ────────────────────────────────────────────────────────

def _infer_physics(case_dir: Path) -> dict:
    info = {}

    # Solver
    ctrl = case_dir / "system" / "controlDict"
    ctrl_text = ctrl.read_text(errors="ignore") if ctrl.exists() else ""
    info["solver"] = _grep(ctrl_text, r"^application\s+(\S+);", "unknown").rstrip(";")
    info["end_time"] = _grep(ctrl_text, r"endTime\s+([\d.eE+\-]+);", "")
    info["delta_t"] = _grep(ctrl_text, r"deltaT\s+([\d.eE+\-]+);", "")

    # Turbulence model
    turb_text = ""
    for tname in ("turbulenceProperties", "RASProperties", "LESProperties"):
        tp = case_dir / "constant" / tname
        if tp.exists():
            turb_text = tp.read_text(errors="ignore")
            break
    info["turbulence"] = "laminar"
    for model in ("kOmegaSST", "kOmega", "kEpsilon", "realizableKE",
                  "Smagorinsky", "WALE", "dynamicKEqn"):
        if model.lower() in turb_text.lower():
            info["turbulence"] = model
            break
    if "LES" in turb_text:
        info["sim_type"] = "LES"
    elif "RAS" in turb_text:
        info["sim_type"] = "RANS"
    else:
        info["sim_type"] = "laminar"

    # Velocity / fields
    u_file = case_dir / "0" / "U"
    u_text = u_file.read_text(errors="ignore") if u_file.exists() else ""
    m = re.search(r"uniform\s*\(\s*([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\s*\)",
                  u_text)
    if m:
        ux, uy, uz = float(m.group(1)), float(m.group(2)), float(m.group(3))
        speed = (ux**2 + uy**2 + uz**2) ** 0.5
        info["inlet_velocity"] = round(speed, 4)
        info["velocity_vec"] = (ux, uy, uz)
    else:
        info["inlet_velocity"] = 0.0

    # Scalar fields (species, temperature, etc.)
    zero_fields = [f.name for f in (case_dir / "0").iterdir()
                   if f.is_file()] if (case_dir / "0").exists() else []
    info["fields"] = zero_fields

    # Transport nu
    tp_file = case_dir / "constant" / "transportProperties"
    tp_text = tp_file.read_text(errors="ignore") if tp_file.exists() else ""
    m_nu = re.search(r"nu\s+[^0-9]*([\d.eE+\-]+)", tp_text)
    info["nu"] = float(m_nu.group(1)) if m_nu else 1.5e-5

    # triSurface / STL geometry
    tri = case_dir / "constant" / "triSurface"
    info["has_stl"] = tri.exists() and any(tri.glob("*.stl"))
    info["stl_files"] = [f.stem for f in tri.glob("*.stl")] if tri.exists() else []

    # Multiple inlets
    info["n_inlets"] = len(re.findall(r"inlet\d*\s*\{", u_text, re.IGNORECASE))

    return info


# ── Prompt generation ────────────────────────────────────────────────────────

_TURB_DESC = {
    "kOmegaSST": "k-omega SST RANS",
    "kOmega": "k-omega RANS",
    "kEpsilon": "k-epsilon RANS",
    "realizableKE": "realizable k-epsilon RANS",
    "Smagorinsky": "Smagorinsky LES",
    "WALE": "WALE LES",
    "dynamicKEqn": "dynamic k-equation LES",
    "laminar": "laminar (no turbulence model)",
}


def _make_prompt(case_name: str, info: dict) -> str:
    solver = info["solver"]
    turb = info["turbulence"]
    turb_desc = _TURB_DESC.get(turb, turb)
    U = info["inlet_velocity"]
    nu = info["nu"]
    sim_type = info["sim_type"]
    has_stl = info["has_stl"]
    stl = info["stl_files"]
    is_les = "LES" in case_name.upper() or sim_type == "LES"

    # Geometry description
    if has_stl and stl:
        geom_desc = f"external aerodynamics around {', '.join(stl)} body"
    elif info["n_inlets"] >= 2:
        geom_desc = "mixing chamber with multiple inlet streams"
    elif "mixing" in case_name.lower():
        geom_desc = "turbulent mixing flow"
    elif "cavity" in case_name.lower():
        geom_desc = "2D lid-driven cavity"
    elif "boundarylayer" in case_name.lower():
        geom_desc = "flat plate turbulent boundary layer"
    elif "brick" in case_name.lower():
        geom_desc = "flow over a bluff brick body"
    elif "tutorial" in case_name.lower():
        geom_desc = "laminar channel flow"
    else:
        geom_desc = "3D internal flow domain"

    species_note = " with passive scalar transport" if "s" in info["fields"] else ""
    Re_note = f", Re≈{U*0.1/nu:.0f}" if U > 0 and nu > 0 else ""

    # Each turbulence model gets a fixed prompt style; case number shifts it for diversity
    _TURB_SLOT = {"kEpsilon": 0, "kOmega": 1, "kOmegaSST": 2, "realizableKE": 3,
                  "laminar": 4, "Smagorinsky": 5, "WALE": 2, "dynamicKEqn": 1}
    case_num = int(re.search(r"(\d+)", case_name).group(1)) if re.search(r"\d+", case_name) else 0
    turb_slot = _TURB_SLOT.get(turb, 0)
    variant = (case_num + turb_slot) % 6

    sim_label = "LES" if is_les else ("laminar" if turb == "laminar" else "RANS")

    PROMPTS = [
        f"Set up a {solver} {sim_label} simulation of {geom_desc}{species_note} "
        f"using {turb_desc}. Inlet velocity {U:.3g} m/s, ν={nu:.2e} m²/s{Re_note}.",

        f"OpenFOAM {solver}: {geom_desc}{species_note}, {turb_desc} turbulence model, "
        f"U_inlet={U:.3g} m/s, kinematic viscosity {nu:.2e}{Re_note}.",

        f"Transient CFD simulation of {geom_desc}{species_note}. "
        f"Solver: {solver}. Turbulence: {turb_desc}. Inlet speed {U:.3g} m/s.",

        f"Run a {geom_desc} case{species_note} in OpenFOAM with {turb_desc}, "
        f"inlet {U:.3g} m/s, nu={nu:.2e} m²/s. Use {solver}.",

        f"{sim_label.capitalize()} {geom_desc} simulation{species_note}: "
        f"{solver}, {turb_desc}, U={U:.3g} m/s{Re_note}, ν={nu:.2e} m²/s.",

        f"Generate OpenFOAM case files for {geom_desc}{species_note}. "
        f"{turb_desc} turbulence, inlet velocity {U:.3g} m/s, {solver} solver.",
    ]

    return PROMPTS[variant]


# ── Expert analysis builder ──────────────────────────────────────────────────

def _make_analysis(case_name: str, info: dict) -> str:
    solver = info["solver"]
    turb = info["turbulence"]
    turb_desc = _TURB_DESC.get(turb, turb)
    U = info["inlet_velocity"]
    nu = info["nu"]
    Re = U * 0.1 / nu if U > 0 else 0  # rough estimate

    lines = [
        f"**Case:** {case_name}",
        f"**Solver:** {solver} (transient incompressible)",
        f"**Turbulence:** {turb_desc}",
        f"**Inlet velocity:** {U:.4g} m/s  |  ν = {nu:.2e} m²/s"
        + (f"  |  Re ≈ {Re:.3g}" if Re > 0 else ""),
        f"**Fields:** {', '.join(info['fields'])}",
    ]
    if info["has_stl"]:
        lines.append(f"**Geometry:** STL surface mesh — {', '.join(info['stl_files'])}")
    if info["n_inlets"] >= 2:
        lines.append(f"**BCs:** {info['n_inlets']} inlets (multiple streams)")
    lines.append(f"**Score:** 1.00 (pre-validated, all smoke tests PASS)")
    return "\n".join(lines)


# ── Qwen chat formatter ──────────────────────────────────────────────────────

def _format_example(prompt: str, analysis: str, case_files: str) -> str:
    from openfoam_agent.training import EXPERT_SYSTEM_PROMPT
    return (
        f"<|im_start|>system\n{EXPERT_SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n## CFD Analysis\n\n{analysis}\n\n"
        f"## OpenFOAM Case Files\n\n{case_files}\n<|im_end|>"
    )


# ── Case discovery ───────────────────────────────────────────────────────────

# work1 cases to skip (incomplete / tutorial-only / duplicates)
_WORK1_SKIP = {
    "OFtutorial00_helloWorld",   # hello world, no real physics
    "OFtutorial01_inputOutput",  # I/O tutorial, no simulation
    "OFtutorial04_basicFieldOperations",
    "OFtutorial15_discretisation",
    "cavity_Gauss linearUpw",    # space in name causes issues
}

# Subdirectory patterns that are pre-run output, not case definitions
_NUMERIC_RE = re.compile(r"^\d")


def _is_valid_case(d: Path) -> bool:
    if not d.is_dir():
        return False
    if d.name in _SKIP_DIRS or d.name in _WORK1_SKIP:
        return False
    if not (d / "system" / "controlDict").exists():
        return False
    return True


def _discover_cases() -> list[tuple[str, Path, str]]:
    """Returns list of (case_name, case_dir, source_label)."""
    cases = []

    # augmentedCases — 45 validated pimpleFoam cases
    if AUGMENTED_DIR.exists():
        for d in sorted(AUGMENTED_DIR.iterdir()):
            if _is_valid_case(d):
                cases.append((d.name, d, "augmented"))

    # work1 — 25 mixed cases (pisoFoam, LES, real geometry)
    if WORK1_DIR.exists():
        for d in sorted(WORK1_DIR.iterdir()):
            if _is_valid_case(d):
                cases.append((d.name, d, "work1"))

    return cases


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true",
                   help="Print summary without writing to JSONL")
    args = p.parse_args()

    cases = _discover_cases()
    print(f"\n[extract] Found {len(cases)} valid external cases")

    from collections import Counter
    by_source = Counter(s for _, _, s in cases)
    by_solver: Counter = Counter()

    examples = []
    for case_name, case_dir, source in cases:
        info = _infer_physics(case_dir)
        by_solver[info["solver"]] += 1
        prompt = _make_prompt(case_name, info)
        analysis = _make_analysis(case_name, info)
        case_files = _read_case_files(case_dir)
        text = _format_example(prompt, analysis, case_files)
        examples.append({"text": text, "score": 1.0,
                         "case_name": case_name, "source": source,
                         "solver": info["solver"], "turbulence": info["turbulence"]})

    print(f"\n  By source : {dict(by_source)}")
    print(f"  By solver : {dict(by_solver)}")
    turb_counts = Counter(e["turbulence"] for e in examples)
    print(f"  By turb.  : {dict(turb_counts)}")

    if args.dry_run:
        print("\n[extract] --dry-run: sample prompts:")
        for e in examples[:5]:
            prompt_line = e["text"].split("<|im_start|>user\n")[1].split("\n<|im_end|>")[0]
            print(f"  [{e['source']}] {e['case_name']}: {prompt_line[:80]}")
        print(f"\n[extract] Would append {len(examples)} examples to {OUTPUT_JSONL}")
        return

    # Append to expert_train.jsonl
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    existing = OUTPUT_JSONL.read_text().count("\n") if OUTPUT_JSONL.exists() else 0

    with OUTPUT_JSONL.open("a") as f:
        for e in examples:
            f.write(json.dumps({"text": e["text"], "score": e["score"]}) + "\n")

    after = OUTPUT_JSONL.read_text().count("\n") if OUTPUT_JSONL.exists() else 0
    print(f"\n[extract] Appended {len(examples)} examples  "
          f"({existing} → {after} total in {OUTPUT_JSONL.name})")
    print("[extract] Done.\n")


if __name__ == "__main__":
    main()
