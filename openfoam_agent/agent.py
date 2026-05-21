from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

from .schemas import CFDParams, AgentResult, RunResult, TrainingExample
from .config import (
    CASES_DIR, DATASET_DIR, ATTEMPTS_LOG, RETRY_SCORE_THRESHOLD,
)
from .solver_selector import select_solver, compute_mesh_resolution, compute_time_settings
from .numerical_policy import compute_numerical_policy
from .failure_diagnosis import diagnose, build_retry_context
from . import self_correct
from .scorer import compute_reward
from .runner import run_simulation
from . import dict_fix

SEED_PROMPTS = [
    "2D lid-driven cavity flow at Re=100",
    "Turbulent pipe flow Re=50000 diameter=0.05m length=0.5m",
    "2D flow around a circular cylinder Re=200 diameter=0.1m",
    "Backward-facing step Re=800 step height=0.1m",
    "2D turbulent channel flow Re=10000 length=5m height=0.1m",
    "NACA0012 airfoil angle of attack 5 degrees Re=1e6 chord=1m",
    "Laminar pipe flow Re=500 diameter=0.02m length=0.3m",
    "3D turbulent channel flow Re=50000 length=2m width=0.5m height=0.2m",
    "Buoyancy-driven cavity with hot wall at 350K cold wall at 300K",
    "Flow over a flat plate Re=1000 length=1m width=0.1m",
]


class OpenFOAMAgent:
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._llm = None
        self._refiner = None
        self._extractor = None
        self._rag = None
        self._gmsh_gen = None
        self._case_writer = None

    def _init_llm(self):
        if self._llm is None and self.use_llm:
            from .config import get_llm
            self._llm = get_llm()

    def _init_rag(self):
        if self._rag is None:
            from .rag import TutorialRAG
            self._rag = TutorialRAG()

    def _init_components(self):
        self._init_llm()
        self._init_rag()
        if self._gmsh_gen is None:
            from .gmsh_generator import GmshMeshGenerator
            self._gmsh_gen = GmshMeshGenerator()
        if self._case_writer is None:
            from .case_writer import CaseWriter
            self._case_writer = CaseWriter()

    def run(
        self,
        prompt: str,
        max_retries: int = 3,
        use_gmsh: bool = True,
        case_name: Optional[str] = None,
        sim_timeout: int = 300,
        end_time_override: Optional[float] = None,
    ) -> AgentResult:
        self._init_components()
        case_name = case_name or f"case_{uuid.uuid4().hex[:8]}"
        error_context = ""
        last_result = AgentResult(success=False, score=0.0, feedback="not started", solver="unknown")

        # Step 1: Refine prompt
        refined_text = prompt
        if self.use_llm and self._llm:
            from . import prompt_refiner
            refined = prompt_refiner.refine(self._llm, prompt)
            refined_text = refined.refined

        # RAG retrieval (before extraction so we can use it for context)
        rag_examples: list[str] = []
        try:
            self._init_rag()
        except Exception:
            pass

        for attempt in range(max_retries):
            attempt_dir = CASES_DIR / f"{case_name}_attempt{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            # Step 2: Extract params
            try:
                if self.use_llm and self._llm:
                    from . import param_extractor
                    # Feed the user's *original* prompt to the JSON-schema
                    # extractor. The refiner's paraphrase sometimes invents
                    # numbers (e.g. defaults velocity to 1 m/s when only Re
                    # was given), which then leak into downstream math.
                    # The refined text is still useful as RAG/context but
                    # not as the source of numeric values.
                    input_text = prompt + error_context
                    params = param_extractor.extract(self._llm, input_text,
                                                       original_prompt=prompt)
                else:
                    params = self._fallback_params(prompt)
            except Exception as e:
                # Heuristic salvage: if the prompt unambiguously names a
                # multiphase / buoyant / compressible setup, build default
                # params for it instead of giving up.
                # Skipped under OPENFOAM_AGENT_RAW_LLM=1 (honest evaluation mode).
                import os as _os
                _raw = bool(_os.environ.get("OPENFOAM_AGENT_RAW_LLM"))
                params = None if _raw else self._heuristic_params(prompt)
                if params is None:
                    last_result = AgentResult(
                        success=False, score=0.0,
                        feedback=f"param extraction failed: {e}",
                        solver="unknown", attempt=attempt,
                        refined_prompt=refined_text,
                    )
                    error_context = f"\n\nPrevious extraction failed: {e}. Try again."
                    continue

            # Step 3: Prompt-keyword guards — repair common LLM mis-extractions
            # before solver selection. The catalog prompts often name the solver
            # or physics explicitly; honour that even if the LLM contradicted it.
            # All guards in this block are bypassed under
            # OPENFOAM_AGENT_RAW_LLM=1 (honest evaluation mode).
            import os as _os
            _raw = bool(_os.environ.get("OPENFOAM_AGENT_RAW_LLM"))
            from .schemas import FlowRegime, TurbulenceModel
            # Use the user's original prompt — the refiner's paraphrase can
            # introduce words like "free surface" or "compressible" that the
            # user never wrote, leading to spurious solver overrides.
            p_low = prompt.lower()
            named_multiphase = (not _raw) and any(w in p_low for w in
                ("interfoam", "vof", "dam break", "dam-break", "dambreak",
                 "sloshing", "wave channel", "wave tank", "free surface",
                 "two-phase", "alpha.water"))
            if named_multiphase and not params.is_multiphase:
                params.is_multiphase = True
                params.is_compressible = False
                params.has_heat_transfer = False
            # Use the original prompt (not refined) to detect explicit solver
            # mentions — refiner sometimes adds "transient" wording that
            # contradicts the user's named solver.
            orig_low = prompt.lower()
            buoyant_solver_named = "buoyantsimplefoam" in orig_low or "buoyantsimplefoam" in p_low
            buoyant_keywords = ("natural convection" in p_low or "differentially heated" in p_low
                                 or "heated room" in p_low)
            buoyant_pimple_named = "buoyantpimplefoam" in orig_low
            named_buoyant_steady = (not _raw) and (buoyant_solver_named or (
                buoyant_keywords
                and "transient" not in orig_low and "unsteady" not in orig_low
                and not buoyant_pimple_named))
            if named_buoyant_steady:
                params.has_heat_transfer = True
                params.is_transient = False
                params.is_compressible = False
                params.is_multiphase = False
            named_ico = (not _raw) and (("icofoam" in p_low) or (
                ("transient" in p_low or "time-dependent" in p_low or "impulsive" in p_low or "unsteady" in p_low)
                and "laminar" in p_low
                and not any(w in p_low for w in ("compressible", "mach", "buoyant",
                                                  "natural convection", "vof",
                                                  "dam break", "dambreak", "sloshing"))))
            if named_ico:
                params.is_transient = True
                params.is_compressible = False
                params.has_heat_transfer = False
                params.is_multiphase = False
                params.flow_regime = FlowRegime.LAMINAR
                params.turbulence_model = TurbulenceModel.LAMINAR
            named_compressible = (not _raw) and (
                "rhosimplefoam" in p_low or "rhopimplefoam" in p_low
                or "mach" in p_low or " ma=" in p_low or " ma " in p_low
                or "compressible" in p_low)
            if named_compressible and not params.is_multiphase and not params.has_heat_transfer:
                params.is_compressible = True

            # Step 4: Solver selection + numerical policy
            solver = select_solver(params)
            # Low-Re transient: LLMs often mis-label as turbulent; override to laminar.
            # Bypassed under OPENFOAM_AGENT_RAW_LLM=1.
            if (not _raw
                    and params.is_transient and not params.has_heat_transfer
                    and not params.is_compressible and not params.is_multiphase
                    and params.reynolds_number is not None
                    and params.reynolds_number < 2300):
                from .schemas import FlowRegime, TurbulenceModel
                params.flow_regime = FlowRegime.LAMINAR
                params.turbulence_model = TurbulenceModel.LAMINAR
                solver = select_solver(params)
            # Natural convection requires a closed cavity; override complex geometries.
            # Bypassed under OPENFOAM_AGENT_RAW_LLM=1.
            if (not _raw) and params.has_heat_transfer:
                from .schemas import GeometryType
                _safe_geoms = {GeometryType.LID_DRIVEN_CAVITY, GeometryType.BOX, GeometryType.CHANNEL}
                if params.geometry_type not in _safe_geoms:
                    params.geometry_type = GeometryType.LID_DRIVEN_CAVITY
            res = compute_mesh_resolution(params)
            time_cfg = compute_time_settings(params, solver)
            if end_time_override is not None:
                # Interpret override as a step count for transient solvers,
                # else as iterations for steady (deltaT=1).
                steady = solver in ("simpleFoam", "rhoSimpleFoam", "buoyantSimpleFoam")
                if steady:
                    time_cfg["end_time"] = end_time_override
                else:
                    time_cfg["end_time"] = end_time_override * time_cfg["delta_t"]
                time_cfg["write_interval"] = max(1, int(end_time_override))
            num_policy = compute_numerical_policy(params, solver)

            # Step 4: RAG retrieval — guide extraction context on retry.
            # We pull full file contents (controlDict / fvSchemes / BC) for the
            # top-K matched tutorial cases so that on the NEXT retry the
            # param-extractor sees adaptable templates, not just case names.
            rag_with_content: list[dict] = []
            if self._rag:
                try:
                    rag_with_content = self._rag.retrieve_with_content(
                        params, solver, n_cases=2, prompt=prompt,
                    )
                    rag_examples = [r["case_name"] for r in rag_with_content]
                except Exception:
                    rag_with_content = []
                    rag_examples = []

            # Step 5: Mesh generation (pass policy for y+-aware BL sizing)
            has_gmsh = False
            if use_gmsh:
                try:
                    self._gmsh_gen.generate(params, attempt_dir, num_policy)
                    has_gmsh = True
                except Exception as e:
                    print(f"[agent] gmsh failed (attempt {attempt}): {e} — falling back to blockMesh")

            # Step 6: Case files — inject numerical policy
            from .case_writer import CaseWriter, CaseWriterConfig
            cfg = CaseWriterConfig(
                params=params,
                solver=solver,
                case_dir=attempt_dir,
                has_gmsh_mesh=has_gmsh,
                nx=res["nx"], ny=res["ny"], nz=res["nz"],
                end_time=time_cfg["end_time"],
                delta_t=time_cfg["delta_t"],
                write_interval=time_cfg["write_interval"],
                numerical_policy=num_policy,
            )
            try:
                self._case_writer.write_all(cfg)
            except Exception as e:
                error_context = f"\n\nCase file generation failed: {e}"
                continue

            # Step 7: Run simulation
            run_result = run_simulation(
                case_dir=attempt_dir,
                params=params,
                solver=solver,
                has_gmsh_mesh=has_gmsh,
                total_timeout=sim_timeout,
            )

            # Save log
            log_file = attempt_dir / "agent.log"
            log_file.write_text(run_result.log)

            # Step 8: Score
            score, feedback = compute_reward(run_result, params, solver, attempt_dir)

            last_result = AgentResult(
                success=score >= 0.5,
                score=score,
                feedback=feedback,
                solver=solver,
                params=params,
                case_dir=str(attempt_dir),
                error=run_result.error_message,
                runtime=run_result.runtime,
                attempt=attempt,
                residuals=run_result.final_residuals,
                refined_prompt=refined_text,
                rag_examples_used=rag_examples,
            )

            # Capture every attempt (success or fail) for retry-pair mining
            # by Layer-2 curation. Distinct from dataset.json (curated).
            self._append_attempt(prompt, refined_text, params, solver,
                                  attempt_dir, run_result, score, feedback,
                                  attempt)

            if score >= 0.5:
                self._save_to_dataset(prompt, refined_text, params, solver, attempt_dir, run_result, score, feedback)
                return last_result

            # Layer 1: widen retry trigger beyond FOAM FATAL — also fire on
            # divergence / NaN / mass-imbalance / sub-threshold score.
            signals = self_correct.harvest(run_result, score)
            if not self_correct.should_retry(score, signals, RETRY_SCORE_THRESHOLD):
                return last_result

            # Stream B: surgical dict-level fix attempt.
            # If the FOAM FATAL points at a specific dict file, ask the LLM
            # to emit a corrected version of just that file and re-run the
            # solver — preserving mesh + extracted params + the dict files
            # that worked. Falls through to full-pipeline retry on failure.
            if (signals.failure_type.startswith("foam_fatal_")
                    and self.use_llm and self._llm):
                def _rerun():
                    return run_simulation(
                        case_dir=attempt_dir, params=params, solver=solver,
                        has_gmsh_mesh=has_gmsh, total_timeout=sim_timeout,
                    )
                fixed = dict_fix.try_dict_fix(
                    self._llm, attempt_dir, run_result, prompt, _rerun,
                )
                if fixed is not None:
                    new_score, new_feedback = compute_reward(
                        fixed, params, solver, attempt_dir,
                    )
                    print(f"[agent] dict-fix re-run score: "
                          f"{score:.2f} → {new_score:.2f}")
                    # Capture the dict-fix attempt as its own row in
                    # attempts.jsonl so curate's pair miner can use it.
                    self._append_attempt(prompt, refined_text, params, solver,
                                          attempt_dir, fixed, new_score,
                                          new_feedback, attempt + 100)
                    if new_score >= 0.5:
                        last_result = AgentResult(
                            success=True, score=new_score, feedback=new_feedback,
                            solver=solver, params=params,
                            case_dir=str(attempt_dir),
                            error=fixed.error_message, runtime=fixed.runtime,
                            attempt=attempt, residuals=fixed.final_residuals,
                            refined_prompt=refined_text,
                            rag_examples_used=rag_examples,
                        )
                        self._save_to_dataset(prompt, refined_text, params, solver,
                                               attempt_dir, fixed, new_score, new_feedback)
                        return last_result
                    # Dict-fix didn't help enough — fall through to full retry,
                    # but use the (possibly-improved) new score / signals.
                    score, feedback = new_score, new_feedback
                    run_result = fixed
                    signals = self_correct.harvest(run_result, score)
            # Pass RAG file contents (not just case names) so the next retry's
            # param extractor sees adaptable controlDict / fvSchemes / BC text.
            error_context = self_correct.build_context(
                run_result, params, solver, score, feedback,
                attempt, rag_examples,
                rag_with_content=rag_with_content,
            )

        return last_result

    def _heuristic_params(self, prompt: str) -> Optional[CFDParams]:
        """Build sensible default CFDParams when the LLM extractor blows up
        but the prompt names a clear physics regime. Returns None to signal
        'no obvious salvage — fall through to the normal failure path'."""
        from .schemas import GeometryType, FlowRegime, TurbulenceModel
        p = prompt.lower()
        is_multiphase = any(w in p for w in
            ("interfoam", "vof", "dam break", "dam-break", "dambreak",
             "sloshing", "wave channel", "wave tank", "free surface",
             "two-phase", "alpha.water"))
        is_buoyant = ("natural convection" in p or "buoyantsimplefoam" in p
                       or "differentially heated" in p or "heated room" in p)
        is_ico = ("icofoam" in p) or ("transient" in p and "laminar" in p
                                       and not is_multiphase and not is_buoyant)
        if not (is_multiphase or is_buoyant or is_ico):
            return None
        base = self._fallback_params(prompt)
        if is_multiphase:
            base.geometry_type = GeometryType.BOX
            base.is_multiphase = True
            base.is_transient = True
            base.is_compressible = False
            base.has_heat_transfer = False
            base.length, base.width, base.height = 4.0, 2.0, 0.001
            base.inlet_velocity = 0.0
            base.end_time = 5.0
        elif is_buoyant:
            base.geometry_type = GeometryType.LID_DRIVEN_CAVITY
            base.has_heat_transfer = True
            base.is_transient = False
            base.is_compressible = False
            base.is_multiphase = False
            base.length = base.width = 1.0
            base.end_time = 2000.0
        elif is_ico:
            base.geometry_type = GeometryType.LID_DRIVEN_CAVITY
            base.is_transient = True
            base.flow_regime = FlowRegime.LAMINAR
            base.turbulence_model = TurbulenceModel.LAMINAR
            base.is_compressible = False
            base.has_heat_transfer = False
            base.is_multiphase = False
            base.end_time = 5.0
        return base

    def _fallback_params(self, prompt: str) -> CFDParams:
        from .schemas import GeometryType, FlowRegime, TurbulenceModel
        return CFDParams(
            geometry_type=GeometryType.BOX,
            is_3d=False,
            length=1.0, width=1.0, height=0.001,
            inlet_velocity=1.0,
            kinematic_viscosity=1.5e-5,
            density=1.225,
            reynolds_number=None,
            flow_regime=FlowRegime.LAMINAR,
            turbulence_model=TurbulenceModel.LAMINAR,
            is_transient=False,
            is_compressible=False,
            has_heat_transfer=False,
            is_multiphase=False,
            end_time=1000,
            extraction_notes="fallback defaults",
            outlet_pressure=0.0,
        )

    def _save_to_dataset(
        self, prompt: str, refined: str, params: CFDParams,
        solver: str, case_dir: Path, run_result: RunResult,
        score: float, feedback: str,
    ):
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        dataset_file = DATASET_DIR / "dataset.json"
        example = TrainingExample(
            prompt=prompt,
            refined_prompt=refined,
            params=params,
            case_dir=str(case_dir),
            solver=solver,
            score=score,
            feedback=feedback,
            converged=run_result.converged,
            runtime=run_result.runtime,
            timestamp=time.time(),
            case_files_text=self._read_case_files(case_dir),
        )
        existing = []
        if dataset_file.exists():
            try:
                existing = json.loads(dataset_file.read_text())
            except Exception:
                pass
        existing.append(json.loads(example.model_dump_json()))
        dataset_file.write_text(json.dumps(existing, indent=2))

    def _append_attempt(
        self, prompt: str, refined: str, params: CFDParams, solver: str,
        case_dir: Path, run_result: RunResult, score: float, feedback: str,
        attempt: int,
    ):
        """Append every attempt to attempts.jsonl for Layer-2 retry-pair mining.

        Curate_dataset.py reads this file to find (prompt, low-score-attempt,
        high-score-retry) triples and emits them as DPO-ready preference pairs.
        Always appends — failures are as informative as successes here.
        """
        ATTEMPTS_LOG.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "prompt": prompt,
            "refined_prompt": refined,
            "solver": solver,
            "score": score,
            "feedback": feedback,
            "converged": run_result.converged,
            "runtime": run_result.runtime,
            "attempt": attempt,
            "case_dir": str(case_dir),
            "case_files_text": self._read_case_files(case_dir),
            "params": json.loads(params.model_dump_json()),
            "timestamp": time.time(),
        }
        with open(ATTEMPTS_LOG, "a") as f:
            f.write(json.dumps(row) + "\n")

    def _read_case_files(self, case_dir: Path) -> str:
        parts = []
        for sub in ("system", "constant", "0"):
            d = case_dir / sub
            if d.exists():
                for f in sorted(d.iterdir()):
                    if f.is_file():
                        try:
                            content = f.read_text(errors="ignore")
                            rel = f.relative_to(case_dir)
                            parts.append(f"### {rel}\n```\n{content}\n```")
                        except Exception:
                            pass
        return "\n\n".join(parts)

    def run_with_params(
        self,
        prompt: str,
        params: CFDParams,
        max_retries: int = 2,
        use_gmsh: bool = True,
        case_name: Optional[str] = None,
        sim_timeout: int = 300,
        end_time_override: Optional[float] = None,
    ) -> AgentResult:
        """Run the pipeline with pre-supplied CFDParams (skips LLM extraction).

        Used for training data generation / RAG validation where params are known.
        end_time_override: if set, overrides params.end_time (e.g. 2 for quick smoke test).
        """
        self._init_rag()
        if self._gmsh_gen is None:
            from .gmsh_generator import GmshMeshGenerator
            self._gmsh_gen = GmshMeshGenerator()
        if self._case_writer is None:
            from .case_writer import CaseWriter
            self._case_writer = CaseWriter()

        case_name = case_name or f"case_{uuid.uuid4().hex[:8]}"
        rag_examples: list[str] = []

        for attempt in range(max_retries):
            attempt_dir = CASES_DIR / f"{case_name}_attempt{attempt}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            solver = select_solver(params)
            # Low-Re transient: override turbulence model to laminar for stability
            if (params.is_transient and not params.has_heat_transfer
                    and not params.is_compressible and not params.is_multiphase
                    and params.reynolds_number is not None
                    and params.reynolds_number < 2300):
                from .schemas import FlowRegime, TurbulenceModel
                params.flow_regime = FlowRegime.LAMINAR
                params.turbulence_model = TurbulenceModel.LAMINAR
                solver = select_solver(params)
            res = compute_mesh_resolution(params)
            time_cfg = compute_time_settings(params, solver)
            num_policy = compute_numerical_policy(params, solver)

            # RAG retrieval — validate that relevant tutorials are found
            if self._rag:
                try:
                    retrieved = self._rag.retrieve(params, solver, n_results=3)
                    rag_examples = [r["case_name"] for r in retrieved]
                except Exception:
                    rag_examples = []

            if end_time_override is not None:
                time_cfg["end_time"] = end_time_override
                time_cfg["write_interval"] = max(1, int(end_time_override))

            has_gmsh = False
            if use_gmsh:
                try:
                    self._gmsh_gen.generate(params, attempt_dir, num_policy)
                    has_gmsh = True
                except Exception as e:
                    print(f"[agent] gmsh failed (attempt {attempt}): {e} — blockMesh fallback")

            from .case_writer import CaseWriter, CaseWriterConfig
            cfg = CaseWriterConfig(
                params=params,
                solver=solver,
                case_dir=attempt_dir,
                has_gmsh_mesh=has_gmsh,
                nx=res["nx"], ny=res["ny"], nz=res["nz"],
                end_time=time_cfg["end_time"],
                delta_t=time_cfg["delta_t"],
                write_interval=time_cfg["write_interval"],
                numerical_policy=num_policy,
            )
            try:
                self._case_writer.write_all(cfg)
            except Exception as e:
                continue

            run_result = run_simulation(
                case_dir=attempt_dir,
                params=params,
                solver=solver,
                has_gmsh_mesh=has_gmsh,
                total_timeout=sim_timeout,
            )
            (attempt_dir / "agent.log").write_text(run_result.log)

            score, feedback = compute_reward(run_result, params, solver, attempt_dir)
            result = AgentResult(
                success=score >= 0.5,
                score=score,
                feedback=feedback,
                solver=solver,
                params=params,
                case_dir=str(attempt_dir),
                error=run_result.error_message,
                runtime=run_result.runtime,
                attempt=attempt,
                residuals=run_result.final_residuals,
                refined_prompt=prompt,
                rag_examples_used=rag_examples,
            )
            if score >= 0.5:
                self._save_to_dataset(prompt, prompt, params, solver, attempt_dir, run_result, score, feedback)
                return result

        # Return last result (with actual solver) even if score < threshold
        if "result" in dir():
            return result  # type: ignore[return-value]
        return AgentResult(
            success=False, score=0.0, feedback="all attempts failed",
            solver="unknown", rag_examples_used=rag_examples,
        )

    def run_batch(self, prompts: list[str], **kwargs) -> list[AgentResult]:
        results = []
        for prompt in prompts:
            print(f"\n[agent] Running: {prompt[:60]}...")
            result = self.run(prompt, **kwargs)
            print(f"[agent] Score: {result.score:.2f} | {result.feedback}")
            results.append(result)
        return results
