from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .schemas import CFDParams, RunResult
from .config import OPENFOAM_BASHRC


def _run_cmd(cmd: str, cwd: Path, timeout: int) -> tuple[int, str, str]:
    full_cmd = f"bash -c 'source {OPENFOAM_BASHRC} && {cmd}'"
    try:
        proc = subprocess.run(
            full_cmd, shell=True, cwd=str(cwd),
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        def _decode(s):
            if isinstance(s, bytes):
                return s.decode("utf-8", errors="replace")
            return s or ""
        partial = _decode(e.stdout) + _decode(e.stderr)
        return -1, partial, f"TIMEOUT after {timeout}s"


def _check_convergence(log: str) -> bool:
    convergence_markers = (
        "SIMPLE solution converged",
        "PISO: converged",
        "solver converged",
    )
    if any(m in log for m in convergence_markers):
        return True
    if "FOAM FATAL ERROR" in log or "FOAM FATAL Exception" in log:
        return False
    # Completed time loop without fatal error
    if "ExecutionTime" in log and "End" in log and "FOAM FATAL" not in log:
        return True
    return False


def _parse_residuals(log: str) -> tuple[dict[str, float], dict[str, list[float]]]:
    final: dict[str, float] = {}
    history: dict[str, list[float]] = {}

    for line in log.splitlines():
        # OpenFOAM residual line format:
        # "Solving for Ux, Initial residual = 0.5, Final residual = 1e-6, No Iterations 10"
        if "Solving for" in line and "Initial residual" in line:
            parts = line.split(",")
            field_part = parts[0].split("Solving for")[-1].strip()
            field = field_part.split()[0] if field_part else ""
            init_val = final_val = None
            for part in parts:
                if "Initial residual" in part:
                    try:
                        init_val = float(part.split("=")[-1].strip())
                    except ValueError:
                        pass
                elif "Final residual" in part:
                    try:
                        final_val = float(part.split("=")[-1].strip())
                    except ValueError:
                        pass
            # Track Initial residual for convergence history (decreases 1→0)
            if init_val is not None:
                if field not in history:
                    history[field] = []
                history[field].append(init_val)
            # Final residual = last solver iteration result
            if final_val is not None:
                final[field] = final_val

    return final, history


def _parse_non_ortho(log: str) -> tuple[float, float]:
    non_ortho, skewness = 0.0, 0.0
    for line in log.splitlines():
        if "Max non-orthogonality" in line:
            try:
                non_ortho = float(line.split("=")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "Max skewness" in line:
            try:
                skewness = float(line.split("=")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return non_ortho, skewness


def _extract_runtime(log: str) -> float:
    for line in reversed(log.splitlines()):
        if "ExecutionTime" in line:
            try:
                return float(line.split("=")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return 0.0


def _get_solver_from_case(case_dir: Path) -> str:
    ctrl = case_dir / "system" / "controlDict"
    if not ctrl.exists():
        return "simpleFoam"
    text = ctrl.read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("application") and ";" in stripped:
            return stripped.split()[-1].rstrip(";")
    return "simpleFoam"


def run_simulation(
    case_dir: Path,
    params: CFDParams,
    solver: str,
    has_gmsh_mesh: bool = False,
    total_timeout: int = 300,
) -> RunResult:
    case_dir = Path(case_dir)
    full_log = ""

    # Step 1: blockMesh (only if no gmsh mesh)
    if not has_gmsh_mesh:
        rc, out, err = _run_cmd("blockMesh", case_dir, 120)
        full_log += out + err
        if rc != 0:
            return RunResult(
                success=False, converged=False, runtime=0.0,
                error_message=f"blockMesh failed:\n{err[-1000:]}",
                log=full_log,
            )

    # Step 2: checkMesh
    rc, out, err = _run_cmd("checkMesh", case_dir, 60)
    check_log = out + err
    full_log += check_log
    non_ortho, skewness = _parse_non_ortho(check_log)

    if "has invalid cells" in check_log or ("Failed" in check_log and rc != 0):
        return RunResult(
            success=False, converged=False, runtime=0.0,
            mesh_max_non_ortho=non_ortho, mesh_max_skewness=skewness,
            error_message="checkMesh: invalid mesh",
            log=full_log,
        )

    # Remove stale timestep directories from previous runs (OpenFOAM restarts
    # from the latest directory, which could be from an earlier failed run).
    for d in case_dir.iterdir():
        if d.is_dir() and d.name.lstrip("-").replace(".", "", 1).isdigit():
            if d.name not in ("0",):
                shutil.rmtree(d, ignore_errors=True)

    # Step 3: solver
    t0 = time.time()
    rc, out, err = _run_cmd(solver, case_dir, total_timeout)
    runtime = time.time() - t0
    solver_log = out + err
    full_log += solver_log

    if rc == -1:
        return RunResult(
            success=False, converged=False, runtime=runtime,
            mesh_max_non_ortho=non_ortho, mesh_max_skewness=skewness,
            error_message=f"Solver TIMEOUT after {total_timeout}s",
            log=full_log,
        )

    final_res, res_history = _parse_residuals(solver_log)
    converged = _check_convergence(solver_log)
    actual_runtime = _extract_runtime(solver_log) or runtime

    error_msg = ""
    if "FOAM FATAL ERROR" in solver_log or "FOAM FATAL Exception" in solver_log:
        for line in solver_log.splitlines():
            if "FOAM FATAL" in line:
                error_msg = line.strip()
                break

    return RunResult(
        success=(rc == 0),
        converged=converged,
        runtime=actual_runtime,
        final_residuals=final_res,
        residual_history=res_history,
        mesh_max_non_ortho=non_ortho,
        mesh_max_skewness=skewness,
        error_message=error_msg,
        log=full_log,
    )
