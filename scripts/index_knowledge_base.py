#!/usr/bin/env python3
"""
Index (or re-index) the OpenFOAM tutorial cases AND the canonical knowledge base
into ChromaDB.

Run this once after cloning the repo or whenever knowledge_base.py changes:
    conda run -n vllm_env python scripts/index_knowledge_base.py

What it does:
  1. Clears the existing ChromaDB collection.
  2. Re-indexes all tutorial cases from TUTORIALS_DIR with enriched physics tags.
  3. Indexes the canonical knowledge base (cavity, pipe, cylinder, channel, BFS,
     airfoil, wedge, buoyancy) as expert reference documents.
  4. Prints a summary of what was indexed and runs a quick retrieval test.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from openfoam_agent.rag import TutorialRAG
from openfoam_agent.config import TUTORIALS_DIR, CHROMA_DIR

AUGMENTED_DIR = Path("/data/foamllm2/github/WorkingCase/augmentedCases")
WORK1_DIR     = Path("/data/foamllm2/github/WorkingCase/work1")

_WORK1_SKIP = {
    "OFtutorial00_helloWorld",
    "OFtutorial01_inputOutput",
    "OFtutorial04_basicFieldOperations",
    "OFtutorial15_discretisation",
    "cavity_Gauss linearUpw",
}


def main():
    print(f"\n[index] ChromaDB path : {CHROMA_DIR}")
    print(f"[index] Tutorials dir : {TUTORIALS_DIR}")

    rag = TutorialRAG()

    # ── Step 1: Clear existing collection ──────────────────────────────────
    try:
        rag.client.delete_collection("openfoam_tutorials")
        print("[index] Cleared existing collection.")
    except Exception:
        pass
    import chromadb
    from sentence_transformers import SentenceTransformer
    from openfoam_agent.config import EMBEDDING_MODEL
    rag.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    rag.encoder = SentenceTransformer(EMBEDDING_MODEL)
    rag.collection = rag.client.get_or_create_collection(
        name="openfoam_tutorials",
        metadata={"hnsw:space": "cosine"},
    )

    # ── Step 2: Index tutorials ─────────────────────────────────────────────
    if TUTORIALS_DIR.exists():
        n_tutorials = rag.index_tutorials(TUTORIALS_DIR)
        print(f"[index] Indexed {n_tutorials} tutorial documents from {TUTORIALS_DIR}")
    else:
        print(f"[index] WARNING: tutorials dir not found: {TUTORIALS_DIR}")
        n_tutorials = 0

    # ── Step 3: Index knowledge base ────────────────────────────────────────
    n_kb = rag.index_knowledge_base()
    print(f"[index] Indexed {n_kb} knowledge-base documents")

    # ── Step 4: Index external validated cases ──────────────────────────────
    augmented_cases = []
    if AUGMENTED_DIR.exists():
        augmented_cases = [d for d in sorted(AUGMENTED_DIR.iterdir())
                           if d.is_dir() and (d / "system" / "controlDict").exists()]
        n_aug = rag.index_external_cases(augmented_cases, source_label="augmented")
        print(f"[index] Indexed {n_aug} chunks from {len(augmented_cases)} augmentedCases")
    else:
        print(f"[index] WARNING: augmentedCases dir not found: {AUGMENTED_DIR}")

    work1_cases = []
    if WORK1_DIR.exists():
        work1_cases = [d for d in sorted(WORK1_DIR.iterdir())
                       if d.is_dir() and d.name not in _WORK1_SKIP
                       and (d / "system" / "controlDict").exists()]
        n_w1 = rag.index_external_cases(work1_cases, source_label="work1")
        print(f"[index] Indexed {n_w1} chunks from {len(work1_cases)} work1 cases")
    else:
        print(f"[index] WARNING: work1 dir not found: {WORK1_DIR}")

    total = rag.collection.count()
    print(f"[index] Total documents in ChromaDB: {total}")

    # ── Step 4: Quick retrieval test ────────────────────────────────────────
    print("\n[index] Quick retrieval test:")
    print(f"{'─'*70}")

    from openfoam_agent.schemas import CFDParams, GeometryType, FlowRegime, TurbulenceModel

    test_queries = [
        ("2D lid-driven cavity flow at Re=100",
         CFDParams(geometry_type=GeometryType.LID_DRIVEN_CAVITY, is_3d=False,
                   length=1.0, width=1.0, height=0.001, reynolds_number=100,
                   inlet_velocity=0.0015, kinematic_viscosity=1.5e-5, density=1.225,
                   flow_regime=FlowRegime.LAMINAR, turbulence_model=TurbulenceModel.LAMINAR,
                   is_transient=False, is_compressible=False, has_heat_transfer=False,
                   is_multiphase=False, end_time=2000), "simpleFoam"),

        ("turbulent pipe flow Re=50000, diameter=0.05m",
         CFDParams(geometry_type=GeometryType.PIPE, is_3d=True,
                   length=0.5, width=0.05, height=0.05, diameter=0.05, reynolds_number=50000,
                   inlet_velocity=15.0, kinematic_viscosity=1.5e-5, density=1.225,
                   flow_regime=FlowRegime.TURBULENT, turbulence_model=TurbulenceModel.K_OMEGA_SST,
                   is_transient=False, is_compressible=False, has_heat_transfer=False,
                   is_multiphase=False, end_time=1000), "simpleFoam"),

        ("2D flow around a cylinder Re=200, D=0.1m",
         CFDParams(geometry_type=GeometryType.CYLINDER, is_3d=False,
                   length=2.0, width=0.8, height=0.001, diameter=0.1, reynolds_number=200,
                   inlet_velocity=0.3, kinematic_viscosity=1.5e-5, density=1.225,
                   flow_regime=FlowRegime.LAMINAR, turbulence_model=TurbulenceModel.LAMINAR,
                   is_transient=False, is_compressible=False, has_heat_transfer=False,
                   is_multiphase=False, end_time=1000), "simpleFoam"),

        ("2D turbulent channel flow Re=10000",
         CFDParams(geometry_type=GeometryType.CHANNEL, is_3d=False,
                   length=5.0, width=0.1, height=0.001, reynolds_number=10000,
                   inlet_velocity=3.0, kinematic_viscosity=1.5e-5, density=1.225,
                   flow_regime=FlowRegime.TURBULENT, turbulence_model=TurbulenceModel.K_OMEGA_SST,
                   is_transient=False, is_compressible=False, has_heat_transfer=False,
                   is_multiphase=False, end_time=1000), "simpleFoam"),

        ("backward-facing step Re=800, step height=0.1m",
         CFDParams(geometry_type=GeometryType.BACKWARD_FACING_STEP, is_3d=False,
                   length=2.2, width=0.2, height=0.001, reynolds_number=800,
                   inlet_velocity=0.12, kinematic_viscosity=1.5e-5, density=1.225,
                   flow_regime=FlowRegime.LAMINAR, turbulence_model=TurbulenceModel.LAMINAR,
                   is_transient=False, is_compressible=False, has_heat_transfer=False,
                   is_multiphase=False, end_time=2000), "simpleFoam"),

        ("NACA0012 airfoil angle of attack 5 degrees Re=1e6",
         CFDParams(geometry_type=GeometryType.AIRFOIL, is_3d=False,
                   length=1.0, width=20.0, height=0.001, angle_of_attack=5.0,
                   reynolds_number=1e6, inlet_velocity=15.0, kinematic_viscosity=1.5e-5,
                   density=1.225, flow_regime=FlowRegime.TURBULENT,
                   turbulence_model=TurbulenceModel.K_OMEGA_SST, is_transient=False,
                   is_compressible=False, has_heat_transfer=False,
                   is_multiphase=False, end_time=2000), "simpleFoam"),
    ]

    for prompt, params, solver in test_queries:
        results = rag.retrieve(params, solver, n_results=3, prompt=prompt)
        hits = [(r["case_name"], f"{r['distance']:.3f}") for r in results]
        print(f"  Q: {prompt[:50]:<50}")
        for name, dist in hits:
            tag = "KB" if name.startswith("KnowledgeBase") else "TU"
            print(f"     [{tag}] {name:<40} dist={dist}")
        print()

    print(f"{'─'*70}")
    print("[index] Done. Knowledge base is ready for RAG retrieval.\n")


if __name__ == "__main__":
    main()
