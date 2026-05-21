#!/usr/bin/env python3
"""
Build a HuggingFace-publishable per-file dataset from data/dataset/dataset.json.

Schema follows FoamGPT (https://huggingface.co/datasets/LeoYML/FoamGPT):
  case_name, case_domain, case_category, case_solver,
  folder_name, file_name, file_content,
  user_requirement, system_prompt, user_prompt

Each TrainingExample row in dataset.json is exploded into one JSONL row per
OpenFOAM dictionary file (system/*, constant/*, 0/*) parsed out of
case_files_text. The Qwen-chat training file (expert_train.jsonl) is left
untouched — that path still drives train_qlora.py.

Usage:
    python scripts/build_hf_dataset.py
    # writes data/dataset/foam_openfoam_dataset.jsonl
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/dataset/dataset.json"
OUT = ROOT / "data/dataset/foam_openfoam_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are an expert OpenFOAM CFD engineer. Given a natural-language "
    "simulation request and a target file path inside an OpenFOAM case, "
    "emit the exact contents of that single file. Use OpenFOAM v2412 syntax. "
    "Do not include explanations or markdown fences — output only the "
    "dictionary text that would live at that path."
)

SOLVER_DOMAIN = {
    "simpleFoam":         "incompressible flow (steady, RANS)",
    "pimpleFoam":         "incompressible flow (transient, URANS/LES)",
    "icoFoam":            "incompressible laminar flow (transient)",
    "buoyantSimpleFoam":  "buoyancy-driven heat transfer (steady)",
    "interFoam":          "multiphase flow (VOF, two incompressible phases)",
    "rhoSimpleFoam":      "compressible flow (steady)",
    "rhoPimpleFoam":      "compressible flow (transient)",
    "pisoFoam":           "incompressible flow (transient, legacy PISO)",
}

# user_requirement template per OpenFOAM file (substring match)
FILE_REQUIREMENT = {
    "controlDict":        "Configure run control: application solver, time range, deltaT, write intervals, runtime modifiability.",
    "fvSchemes":          "Specify discretisation schemes (ddt, grad, div, laplacian, interpolation, snGrad) appropriate for the solver and flow regime.",
    "fvSolution":         "Configure linear solvers, tolerances, and pressure-velocity coupling (SIMPLE/PIMPLE/PISO) settings.",
    "decomposeParDict":   "Configure parallel decomposition for the case.",
    "transportProperties":"Define Newtonian transport properties (kinematic viscosity, density) for the working fluid.",
    "turbulenceProperties":"Select the turbulence simulation type (laminar / RAS / LES) and the model.",
    "thermophysicalProperties":"Define the thermophysical model: equation of state, transport, thermodynamics for compressible flow.",
    "g":                  "Define the gravitational acceleration vector for buoyant or multiphase cases.",
    "U":                  "Set initial and boundary conditions for the velocity field.",
    "p":                  "Set initial and boundary conditions for the pressure field.",
    "p_rgh":              "Set initial and boundary conditions for the modified pressure (p_rgh) used in buoyant / multiphase solvers.",
    "T":                  "Set initial and boundary conditions for the temperature field.",
    "k":                  "Set initial and boundary conditions for turbulent kinetic energy.",
    "omega":              "Set initial and boundary conditions for the specific dissipation rate omega.",
    "epsilon":            "Set initial and boundary conditions for the dissipation rate epsilon.",
    "nut":                "Set initial and boundary conditions for the turbulent viscosity nut.",
    "nuTilda":            "Set initial and boundary conditions for the Spalart-Allmaras working variable.",
    "alpha.water":        "Set initial and boundary conditions for the VOF water phase fraction.",
    "alpha.air":          "Set initial and boundary conditions for the VOF air phase fraction.",
    "blockMeshDict":      "Define the block-structured mesh: vertices, blocks, edges, boundary patches.",
    "snappyHexMeshDict":  "Configure castellation/snapping/layer-addition for the snappyHexMesh stage.",
}

FILE_RE = re.compile(r"^### (.+?)\n```\n(.*?)\n```", re.S | re.M)


GEOM_CATEGORY = {
    "lid_driven_cavity":  "lid-driven cavity",
    "box":                "generic duct / box",
    "pipe":               "pipe flow",
    "cylinder":           "flow over cylinder",
    "channel":            "channel flow",
    "bfs":                "backward-facing step",
    "airfoil":            "external aerodynamics (airfoil)",
    "wedge":              "axisymmetric wedge",
}

HEX_RE = re.compile(r"^[0-9a-f]{6,}$")


def derive_case_name(case_dir: str, tag_hint: str, solver: str, idx: int) -> str:
    raw = Path(case_dir).name
    raw = raw.replace("_attempt0", "").replace("case_", "").replace("val_", "")
    if HEX_RE.match(raw) or not raw:
        return f"{solver}_{idx:04d}"
    return raw


PROMPT_KEYWORDS = [
    ("cavity",   "lid-driven cavity"),
    ("airfoil",  "external aerodynamics (airfoil)"),
    ("naca",     "external aerodynamics (airfoil)"),
    ("dam",      "multiphase / VOF (dam break)"),
    ("slosh",    "multiphase / VOF (sloshing)"),
    ("wave",     "multiphase / VOF (wave channel)"),
    ("nozzle",   "compressible nozzle / duct"),
    ("backward", "backward-facing step"),
    ("bfs",      "backward-facing step"),
    ("cylinder", "flow over cylinder"),
    ("pipe",     "pipe flow"),
    ("channel",  "channel flow"),
    ("wedge",    "axisymmetric wedge"),
]


def category_from_tag(tag: str, params: dict | None = None, prompt: str = "") -> str:
    p_low = (prompt or "").lower()
    for kw, cat in PROMPT_KEYWORDS:
        if kw in p_low:
            return cat
    if params and params.get("is_multiphase"):
        return "multiphase / VOF"
    if params and params.get("has_heat_transfer"):
        return "buoyancy-driven heat transfer"
    if params:
        gt = params.get("geometry_type")
        if gt in GEOM_CATEGORY:
            return GEOM_CATEGORY[gt]
    head = tag.split("_")[0]
    if HEX_RE.match(head):
        return "generic duct / box"
    return {
        "ico": "lid-driven cavity / laminar",
        "cav": "lid-driven cavity",
        "cavity": "lid-driven cavity",
        "pipe": "pipe flow",
        "cyl": "flow over cylinder",
        "chan": "channel flow",
        "bfs": "backward-facing step",
        "airfoil": "external aerodynamics (airfoil)",
        "wedge": "axisymmetric wedge",
        "box": "generic duct / box",
        "buoy": "buoyancy-driven cavity",
        "comp": "compressible duct / nozzle",
        "rhoSimple": "compressible (steady)",
        "rhoPimple": "compressible (transient)",
        "pimple": "incompressible transient",
        "multiphase": "multiphase / VOF",
    }.get(head, head)


def requirement_for(file_name: str, solver: str) -> str:
    base = file_name.split("/")[-1]
    if base in FILE_REQUIREMENT:
        return FILE_REQUIREMENT[base]
    for k, v in FILE_REQUIREMENT.items():
        if k in base:
            return v
    return f"Generate the {file_name} dictionary for the {solver} case."


def split_case_files(text: str) -> list[tuple[str, str]]:
    """Return list of (relative_path, content) parsed from case_files_text."""
    return [(m.group(1).strip(), m.group(2).rstrip() + "\n")
            for m in FILE_RE.finditer(text)]


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    src = json.loads(SRC.read_text())
    print(f"[build] source records: {len(src)}")

    n_rows = 0
    solver_ct: Counter[str] = Counter()
    file_ct: Counter[str] = Counter()

    with OUT.open("w") as f:
        for idx, ex in enumerate(src):
            if not ex.get("converged"):
                continue
            if ex.get("score", 0) < 0.5:
                continue
            solver = ex.get("solver") or "simpleFoam"
            domain = SOLVER_DOMAIN.get(solver, "CFD simulation")
            params = ex.get("params") or {}
            prompt = ex.get("refined_prompt") or ex.get("prompt") or ""
            case_name = derive_case_name(ex["case_dir"], "", solver, idx)
            category = category_from_tag(case_name, params, prompt)
            files = split_case_files(ex.get("case_files_text") or "")
            for rel, content in files:
                folder, _, fname = rel.rpartition("/")
                folder = folder + "/" if folder else "./"
                row = {
                    "case_name":        case_name,
                    "case_domain":      domain,
                    "case_category":    category,
                    "case_solver":      solver,
                    "folder_name":      folder,
                    "file_name":        fname,
                    "file_content":     content,
                    "user_requirement": prompt,
                    "system_prompt":    SYSTEM_PROMPT,
                    "user_prompt": (
                        f"Simulation request: {prompt}\n"
                        f"Solver: {solver}\n"
                        f"Generate the contents of `{folder}{fname}`."
                    ),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                solver_ct[solver] += 1
                file_ct[fname] += 1

    print(f"[build] wrote {n_rows} rows -> {OUT}")
    print("\n[build] rows per solver:")
    for s, n in solver_ct.most_common():
        print(f"  {s:<22} {n:>5}")
    print("\n[build] rows per file (top 15):")
    for fn, n in file_ct.most_common(15):
        print(f"  {fn:<28} {n:>5}")


if __name__ == "__main__":
    main()
