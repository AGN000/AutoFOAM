#!/usr/bin/env python3
"""One-time script to index OpenFOAM tutorial cases into ChromaDB."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openfoam_agent.rag import TutorialRAG
from openfoam_agent.config import TUTORIALS_DIR, CHROMA_DIR


def main():
    parser = argparse.ArgumentParser(description="Index OpenFOAM tutorials into ChromaDB")
    parser.add_argument("--tutorials-dir", type=Path, default=TUTORIALS_DIR)
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Count chunks without indexing")
    args = parser.parse_args()

    if not args.tutorials_dir.exists():
        print(f"Tutorials directory not found: {args.tutorials_dir}")
        sys.exit(1)

    cases = [d for d in args.tutorials_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    print(f"Found {len(cases)} tutorial cases in {args.tutorials_dir}")

    if args.dry_run:
        print(f"Estimated chunks: ~{len(cases) * 5}")
        return

    rag = TutorialRAG(chroma_dir=args.chroma_dir)
    n_docs = rag.index_tutorials(args.tutorials_dir)
    print(f"Indexed {n_docs} document chunks into ChromaDB at {args.chroma_dir}")

    # Test query
    from openfoam_agent.schemas import CFDParams, GeometryType, FlowRegime, TurbulenceModel
    test_params = CFDParams(
        geometry_type=GeometryType.CYLINDER,
        is_3d=False, length=2.0, width=0.4, height=0.001,
        diameter=0.1, inlet_velocity=1.0, kinematic_viscosity=1.5e-4,
        reynolds_number=667, flow_regime=FlowRegime.LAMINAR,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_transient=True, is_compressible=False,
        has_heat_transfer=False, is_multiphase=False,
        end_time=5.0, extraction_notes="test", outlet_pressure=0.0, density=1.225,
    )
    results = rag.retrieve(test_params, "icoFoam", n_results=3)
    print(f"\nTest query for 2D cylinder (icoFoam, laminar):")
    for r in results:
        print(f"  - {r['case_name']} (distance={r['distance']:.3f})")


if __name__ == "__main__":
    main()
