from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .schemas import CFDParams
from .config import TUTORIALS_DIR, CHROMA_DIR, EMBEDDING_MODEL


_CHUNK_TYPES = ["case_summary", "control_dict", "fv_schemes", "boundary_conditions", "gmsh_geo"]

_MAX_CHUNK_TOKENS = 1500  # ~6000 chars


def _truncate(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    # Truncate at last closing brace before limit
    cut = text[:max_chars]
    idx = cut.rfind("}")
    return cut[: idx + 1] if idx > 0 else cut


def _read_dir_files(directory: Path, extensions: tuple = ("",)) -> str:
    parts = []
    if not directory.exists():
        return ""
    for f in sorted(directory.iterdir()):
        if f.is_file() and (not extensions[0] or f.suffix in extensions):
            try:
                content = f.read_text(errors="ignore")
                parts.append(f"=== {f.name} ===\n{content}")
            except Exception:
                pass
    return "\n\n".join(parts)


def _find_readme(case_dir: Path) -> str:
    for name in ("README.md", "README.txt", "README", "readme.md"):
        p = case_dir / name
        if p.exists():
            return p.read_text(errors="ignore")
    return f"Case: {case_dir.name}"


def _detect_solver(case_dir: Path) -> str:
    ctrl = case_dir / "system" / "controlDict"
    if not ctrl.exists():
        # Try nested system dirs
        for ctrl in case_dir.rglob("controlDict"):
            break
        else:
            return "unknown"
    text = ctrl.read_text(errors="ignore")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*"):
            continue
        if "application" in stripped:
            parts = stripped.split()
            if len(parts) >= 2:
                return parts[-1].rstrip(";")
    return "unknown"


def _detect_physics_tags(case_dir: Path) -> str:
    """Extract physics keywords from case files for richer embeddings."""
    tags = []
    # Check turbulence model from constant/turbulenceProperties or RASProperties
    for fname in ("turbulenceProperties", "RASProperties", "LESProperties"):
        fpath = case_dir / "constant" / fname
        if not fpath.exists():
            for fp in case_dir.rglob(fname):
                fpath = fp
                break
        if fpath.exists():
            text = fpath.read_text(errors="ignore")
            for model in ("kOmegaSST", "kEpsilon", "Smagorinsky", "laminar"):
                if model in text:
                    tags.append(f"turbulence model {model}")
                    break
    # Check 0/ fields for velocity magnitude clues
    u_field = case_dir / "0" / "U"
    if not u_field.exists():
        for fp in case_dir.rglob("0/U"):
            u_field = fp
            break
    if u_field.exists():
        text = u_field.read_text(errors="ignore")
        # Look for uniform velocity values
        import re as _re
        m = _re.search(r"uniform\s*\(\s*([\d.eE+-]+)", text)
        if m:
            try:
                u = float(m.group(1))
                tags.append(f"inlet velocity {u:.2g} m/s")
            except ValueError:
                pass
    # Check geometry name hints
    name = case_dir.name.lower()
    for keyword in ("cavity", "pipe", "cylinder", "channel", "step", "airfoil",
                    "wing", "duct", "sphere", "wedge", "axisymmetric",
                    "turbulent", "laminar", "transient", "steady", "2d", "3d"):
        if keyword in name:
            tags.append(keyword)
    return " ".join(tags)


class TutorialRAG:
    def __init__(self, chroma_dir: Optional[Path] = None,
                 embedding_model: str = EMBEDDING_MODEL):
        import chromadb
        from sentence_transformers import SentenceTransformer

        chroma_dir = chroma_dir or CHROMA_DIR
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.encoder = SentenceTransformer(embedding_model)
        self.collection = self.client.get_or_create_collection(
            name="openfoam_tutorials",
            metadata={"hnsw:space": "cosine"},
        )

    def index_tutorials(self, tutorials_dir: Optional[Path] = None) -> int:
        tutorials_dir = tutorials_dir or TUTORIALS_DIR
        docs, metas, ids = [], [], []
        count = 0

        for case_dir in sorted(tutorials_dir.iterdir()):
            if not case_dir.is_dir() or case_dir.name.startswith("."):
                continue

            solver = _detect_solver(case_dir)
            readme = _find_readme(case_dir)
            is_2d = "2D" in case_dir.name or "2d" in readme.lower()

            base_meta = {
                "case_name": case_dir.name,
                "solver": solver,
                "is_2d": str(is_2d),
            }

            physics_tags = _detect_physics_tags(case_dir)

            # Chunk 1: case summary — enriched with physics keywords
            summary = (
                f"Case: {case_dir.name}\nSolver: {solver}\n2D: {is_2d}\n"
                f"Physics: {physics_tags}\n\n{readme}"
            )
            docs.append(_truncate(summary))
            metas.append({**base_meta, "chunk_type": "case_summary"})
            ids.append(f"{case_dir.name}_summary")

            # Chunk 2: controlDict
            ctrl = case_dir / "system" / "controlDict"
            if ctrl.exists():
                docs.append(_truncate(ctrl.read_text(errors="ignore")))
                metas.append({**base_meta, "chunk_type": "control_dict"})
                ids.append(f"{case_dir.name}_controlDict")

            # Chunk 3: fvSchemes + fvSolution
            fv_text = _read_dir_files(case_dir / "system")
            if fv_text:
                docs.append(_truncate(fv_text))
                metas.append({**base_meta, "chunk_type": "fv_schemes"})
                ids.append(f"{case_dir.name}_fv")

            # Chunk 4: 0/ boundary conditions
            bc_text = _read_dir_files(case_dir / "0")
            if bc_text:
                docs.append(_truncate(bc_text))
                metas.append({**base_meta, "chunk_type": "boundary_conditions"})
                ids.append(f"{case_dir.name}_bc")

            # Chunk 5: .geo file if present
            geo_files = list(case_dir.rglob("*.geo"))
            if geo_files:
                geo_text = geo_files[0].read_text(errors="ignore")
                docs.append(_truncate(geo_text))
                metas.append({**base_meta, "chunk_type": "gmsh_geo"})
                ids.append(f"{case_dir.name}_geo")

            count += 1

        if docs:
            embeddings = self.encoder.encode(docs, show_progress_bar=True).tolist()
            self.collection.upsert(
                documents=docs,
                embeddings=embeddings,
                metadatas=metas,
                ids=ids,
            )

        return len(docs)

    @staticmethod
    def _build_query(params: CFDParams, solver: str, prompt: str = "") -> str:
        """Build a natural-language RAG query from params + optional user prompt."""
        if prompt:
            # Use the actual user prompt as primary signal — gives best embedding match
            re_str = f"Reynolds number {params.reynolds_number:.0f}" if params.reynolds_number else ""
            geom = params.geometry_type.value.replace("_", " ")
            regime = params.flow_regime.value
            dim = "3D" if params.is_3d else "2D"
            turb = params.turbulence_model.value
            state = "transient unsteady" if params.is_transient else "steady state"
            # Combine user prompt with physics context for richer embedding
            return (
                f"{prompt}. "
                f"{dim} {geom} flow, {regime} regime, {re_str}, "
                f"{solver} solver, {turb} turbulence model, {state}."
            )
        # Fallback if no prompt: natural-language description from params
        re_str = f"Reynolds number {params.reynolds_number:.0f}" if params.reynolds_number else ""
        geom = params.geometry_type.value.replace("_", " ")
        regime = params.flow_regime.value
        dim = "3D" if params.is_3d else "2D"
        turb = params.turbulence_model.value
        state = "transient" if params.is_transient else "steady state"
        fluid = "air" if params.kinematic_viscosity > 5e-6 else "water"
        return (
            f"{dim} {geom} flow simulation. "
            f"{regime.capitalize()} flow with {re_str}. "
            f"Solver: {solver}. Turbulence model: {turb}. "
            f"{state.capitalize()} incompressible {fluid} simulation."
        )

    def retrieve(
        self,
        params: CFDParams,
        solver: str,
        n_results: int = 3,
        prompt: str = "",
    ) -> list[dict]:
        query = self._build_query(params, solver, prompt)
        query_embedding = self.encoder.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, self.collection.count()),
            where={"chunk_type": "case_summary"},
        )

        retrieved = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                retrieved.append({
                    "case_name": results["metadatas"][0][i].get("case_name", doc_id),
                    "chunk_type": results["metadatas"][0][i].get("chunk_type", ""),
                    "content": results["documents"][0][i],
                    "distance": results["distances"][0][i] if results.get("distances") else 0.0,
                    "metadata": results["metadatas"][0][i],
                })
        return retrieved

    def retrieve_with_content(
        self,
        params: CFDParams,
        solver: str,
        n_cases: int = 3,
        prompt: str = "",
    ) -> list[dict]:
        """Like retrieve(), but for each top-K matched case_name also pulls the
        case's controlDict / fvSchemes-fvSolution / boundary-conditions chunks
        from the same chroma collection. Returns adaptable file contents the
        agent's case writer can use as few-shot context.

        Output schema per case:
          {case_name, summary, control_dict, fv_schemes, boundary_conditions,
           solver, distance}

        Empty strings if a chunk type is missing for that case (some tutorials
        have no 0/ directory etc.).
        """
        # Pull more candidates than we need so we can filter out summary-only
        # KnowledgeBase entries (which have no controlDict / fvSchemes / BC
        # chunks indexed and so can't serve as adaptable templates).
        oversample = max(n_cases * 4, 8)
        ranked = self.retrieve(params, solver, n_results=oversample, prompt=prompt)
        out: list[dict] = []
        for r in ranked:
            if len(out) >= n_cases:
                break
            case_name = r.get("case_name")
            if not case_name:
                continue
            entry = {
                "case_name": case_name,
                "summary": r.get("content", ""),
                "solver": r.get("metadata", {}).get("solver", ""),
                "distance": r.get("distance", 0.0),
                "control_dict": "",
                "fv_schemes": "",
                "boundary_conditions": "",
            }
            # Pull the other chunks for this case_name.
            sub = self.collection.get(
                where={"case_name": case_name},
                limit=20,
            )
            if sub and sub.get("metadatas"):
                for meta, doc in zip(sub["metadatas"], sub["documents"]):
                    ct = meta.get("chunk_type", "")
                    if ct == "control_dict":
                        entry["control_dict"] = doc
                    elif ct == "fv_schemes":
                        entry["fv_schemes"] = doc
                    elif ct == "boundary_conditions":
                        entry["boundary_conditions"] = doc
            # Skip if no file content was found — summary alone isn't an
            # adaptable template. Fall back to summary-only only if we
            # exhaust the candidate list without finding file-bearing cases.
            has_files = any((entry["control_dict"], entry["fv_schemes"],
                              entry["boundary_conditions"]))
            if has_files:
                out.append(entry)
        # Fallback: if absolutely no file-bearing case was found, return the
        # top-ranked summaries so the caller still has *something*.
        if not out and ranked:
            r = ranked[0]
            out.append({
                "case_name": r.get("case_name", ""),
                "summary": r.get("content", ""),
                "solver": r.get("metadata", {}).get("solver", ""),
                "distance": r.get("distance", 0.0),
                "control_dict": "", "fv_schemes": "", "boundary_conditions": "",
            })
        return out

    @staticmethod
    def format_content_context(cases: list[dict],
                                max_chars_per_chunk: int = 2000) -> str:
        """Format retrieve_with_content() output as a few-shot prompt block
        suitable for injecting into the case-writer's system message."""
        if not cases:
            return ""
        parts = ["=== Adaptable OpenFOAM Tutorial Cases (use these as templates) ==="]
        for c in cases:
            header = f"\n--- {c['case_name']}  (solver: {c.get('solver','?')}) ---"
            parts.append(header)
            for label, key in (("controlDict", "control_dict"),
                                ("fvSchemes/fvSolution", "fv_schemes"),
                                ("0/ boundary conditions", "boundary_conditions")):
                content = c.get(key, "").strip()
                if not content:
                    continue
                if len(content) > max_chars_per_chunk:
                    content = content[:max_chars_per_chunk] + "\n... [truncated]"
                parts.append(f"\n### {label}\n```\n{content}\n```")
        return "\n".join(parts)

    def index_knowledge_base(self) -> int:
        """Index canonical expert knowledge documents into ChromaDB."""
        from .knowledge_base import get_knowledge_entries
        entries = get_knowledge_entries()
        docs, metas, ids = [], [], []
        for e in entries:
            docs.append(e.content)
            metas.append({
                "case_name": e.case_name,
                "solver": "simpleFoam",  # generic default
                "is_2d": "True",
                "chunk_type": "case_summary",
                "source": "knowledge_base",
            })
            ids.append(e.doc_id)
        if docs:
            embeddings = self.encoder.encode(docs, show_progress_bar=False).tolist()
            self.collection.upsert(
                documents=docs,
                embeddings=embeddings,
                metadatas=metas,
                ids=ids,
            )
        return len(docs)

    def index_external_cases(self, case_dirs: list[Path], source_label: str = "external") -> int:
        """Index pre-validated external OpenFOAM cases into ChromaDB."""
        docs, metas, ids = [], [], []

        _SKIP = {"processor0", "processor1", "processor2", "processor3",
                 "__pycache__", ".git", "postProcessing"}

        for case_dir in sorted(case_dirs):
            if not case_dir.is_dir():
                continue
            if not (case_dir / "system" / "controlDict").exists():
                continue

            solver = _detect_solver(case_dir)
            physics_tags = _detect_physics_tags(case_dir)
            is_2d = "2d" in case_dir.name.lower()

            base_meta = {
                "case_name": f"{source_label}/{case_dir.name}",
                "solver": solver,
                "is_2d": str(is_2d),
                "source": source_label,
            }

            # Chunk 1: case summary with physics tags
            summary = (
                f"Case: {case_dir.name}\nSource: {source_label}\n"
                f"Solver: {solver}\nPhysics: {physics_tags}\n"
            )
            docs.append(_truncate(summary))
            metas.append({**base_meta, "chunk_type": "case_summary"})
            ids.append(f"{source_label}_{case_dir.name}_summary")

            # Chunk 2: controlDict
            ctrl = case_dir / "system" / "controlDict"
            if ctrl.exists():
                docs.append(_truncate(ctrl.read_text(errors="ignore")))
                metas.append({**base_meta, "chunk_type": "control_dict"})
                ids.append(f"{source_label}_{case_dir.name}_controlDict")

            # Chunk 3: fvSchemes + fvSolution
            fv_text = _read_dir_files(case_dir / "system")
            if fv_text:
                docs.append(_truncate(fv_text))
                metas.append({**base_meta, "chunk_type": "fv_schemes"})
                ids.append(f"{source_label}_{case_dir.name}_fv")

            # Chunk 4: boundary conditions
            bc_text = _read_dir_files(case_dir / "0")
            if bc_text:
                docs.append(_truncate(bc_text))
                metas.append({**base_meta, "chunk_type": "boundary_conditions"})
                ids.append(f"{source_label}_{case_dir.name}_bc")

        if docs:
            embeddings = self.encoder.encode(docs, show_progress_bar=True).tolist()
            self.collection.upsert(
                documents=docs,
                embeddings=embeddings,
                metadatas=metas,
                ids=ids,
            )
        return len(docs)

    def format_context(self, retrieved: list[dict]) -> str:
        if not retrieved:
            return ""
        parts = ["=== Similar Tutorial Cases ==="]
        for r in retrieved:
            parts.append(f"\n--- {r['case_name']} (solver: {r['metadata'].get('solver','?')}) ---")
            # Only show first 800 chars of each case summary
            parts.append(r["content"][:800])
        return "\n".join(parts)
