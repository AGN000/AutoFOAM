from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .schemas import RunResult


class FailureType(str, Enum):
    FOAM_FATAL_MESH = "foam_fatal_mesh"
    FOAM_FATAL_BC = "foam_fatal_bc"
    FOAM_FATAL_SCHEME = "foam_fatal_scheme"
    FP_EXCEPTION = "fp_exception"
    DIVERGENCE = "divergence"
    TIMEOUT = "timeout"
    NO_CONVERGENCE = "no_convergence"
    UNKNOWN = "unknown"


@dataclass
class DiagnosisResult:
    failure_type: FailureType
    message: str
    hints: dict = field(default_factory=dict)


def diagnose(run_result: RunResult, score: float = 0.0) -> DiagnosisResult:
    log = run_result.log
    err = run_result.error_message
    combined = (log + " " + err).lower()

    # ── FOAM FATAL errors ──────────────────────────────────────────────────
    if "foam fatal" in combined:
        mesh_kw = ("polymesh", "non-orthogon", "skewness", "checkmesh",
                   "mesh quality", "face area", "cell volume", "determinant")
        bc_kw = ("patchfield", "boundaryfield", "patch type", "inconsistent patch",
                 "type symmetry", "type empty", "type wedge", "type wall",
                 "symmetryplane", "no convergence")
        scheme_kw = ("fvschemes", "div(", "entry not found", "laplacian",
                     "sngrad", "entry 'method'", "writefrequency", "writeprecision")

        if any(k in combined for k in mesh_kw):
            return DiagnosisResult(
                FailureType.FOAM_FATAL_MESH,
                f"Mesh quality fatal error: {err[:200]}",
                {"regen_mesh": True, "coarser_mesh": True},
            )
        if any(k in combined for k in bc_kw):
            return DiagnosisResult(
                FailureType.FOAM_FATAL_BC,
                f"Boundary condition mismatch: {err[:200]}",
                {"fix_bc": True},
            )
        if any(k in combined for k in scheme_kw):
            return DiagnosisResult(
                FailureType.FOAM_FATAL_SCHEME,
                f"Numerical scheme configuration error: {err[:200]}",
                {"fix_schemes": True},
            )
        return DiagnosisResult(
            FailureType.FOAM_FATAL_MESH,
            f"FOAM FATAL: {err[:200]}",
            {"regen_mesh": True},
        )

    # ── Floating point / NaN divergence ───────────────────────────────────
    fp_kw = ("floating point", "sigfpe", "-nan", "1.#ind", "overflow",
             "maximum number of iterations exceeded")
    if any(k in combined for k in fp_kw):
        return DiagnosisResult(
            FailureType.FP_EXCEPTION,
            "Floating point exception — numerical divergence",
            {"reduce_relaxation": True, "reduce_dt": True, "use_upwind": True},
        )

    # ── Diverging residuals ────────────────────────────────────────────────
    residuals = run_result.final_residuals
    if residuals:
        max_res = max(residuals.values())
        if max_res > 10.0:
            return DiagnosisResult(
                FailureType.DIVERGENCE,
                f"Residuals diverged: max={max_res:.2e}",
                {"reduce_relaxation": True, "reduce_dt": True},
            )

    # ── Timeout ───────────────────────────────────────────────────────────
    if "timeout" in combined or run_result.runtime > 280:
        return DiagnosisResult(
            FailureType.TIMEOUT,
            f"Simulation timeout ({run_result.runtime:.0f}s)",
            {"coarser_mesh": True},
        )

    # ── No residuals output (silent failure) ──────────────────────────────
    if not residuals and run_result.success:
        return DiagnosisResult(
            FailureType.UNKNOWN,
            "Simulation ran but produced no residuals",
            {"fix_schemes": True, "fix_bc": True},
        )

    # ── Did not converge ──────────────────────────────────────────────────
    return DiagnosisResult(
        FailureType.NO_CONVERGENCE,
        "Simulation did not converge",
        {"reduce_relaxation": True, "increase_correctors": True},
    )


def build_retry_context(
    diagnosis: DiagnosisResult,
    attempt: int,
    score: float,
    rag_examples: list[str] | None = None,
) -> str:
    lines = [
        f"\n\nAttempt {attempt + 1} failed (score={score:.2f}).",
        f"Diagnosis: {diagnosis.message}",
        "Fixes to apply:",
    ]
    h = diagnosis.hints
    if h.get("reduce_relaxation"):
        lines.append("- Reduce relaxation factors: p=0.3, U=0.5, k/omega=0.4")
    if h.get("reduce_dt"):
        lines.append("- Reduce time step to maintain CFL < 0.5")
    if h.get("use_upwind"):
        lines.append("- Use Gauss upwind for div(phi,U) scheme")
    if h.get("fix_bc"):
        lines.append("- Verify boundary condition types are consistent with polyMesh patch types")
    if h.get("regen_mesh") or h.get("coarser_mesh"):
        lines.append("- Use a coarser or simpler mesh; check geometry parameters")
    if h.get("fix_schemes"):
        lines.append("- Ensure fvSchemes has valid entries for the chosen solver")
    if h.get("increase_correctors"):
        lines.append("- Increase nCorrectors to 3 and nNonOrthogonalCorrectors to 2")
    if rag_examples:
        lines.append(f"- Reference similar cases: {', '.join(rag_examples[:2])}")
    return "\n".join(lines)
