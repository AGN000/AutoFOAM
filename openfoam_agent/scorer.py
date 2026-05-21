from __future__ import annotations

from pathlib import Path

from .schemas import CFDParams, RunResult
from .solver_selector import select_solver


def _bc_valid(case_dir: Path) -> bool:
    zero_dir = Path(case_dir) / "0"
    if not zero_dir.exists():
        return False
    has_inlet = has_outlet = False
    for f in zero_dir.iterdir():
        if not f.is_file():
            continue
        text = f.read_text(errors="ignore")
        if "inlet" in text and "type" in text:
            has_inlet = True
        if "outlet" in text and "type" in text:
            has_outlet = True
    return has_inlet and has_outlet


def _residuals_monotone(history: dict[str, list[float]]) -> bool:
    """Last-half mean < first-half mean for majority of tracked fields."""
    if not history:
        return False
    n_good = 0
    for field, vals in history.items():
        if len(vals) >= 4:
            mid = len(vals) // 2
            first_avg = sum(vals[:mid]) / len(vals[:mid])
            last_avg = sum(vals[mid:]) / len(vals[mid:])
            if last_avg < first_avg:
                n_good += 1
    return n_good > 0


def _residuals_plateau(history: dict[str, list[float]], final_residuals: dict[str, float]) -> bool:
    """Detect stagnation: residuals stuck at a high value (>1e-3), not at convergence."""
    for field, vals in history.items():
        if len(vals) < 10:
            continue
        final = final_residuals.get(field, vals[-1])
        if final < 1e-3:
            continue  # already converged — flat tail is fine
        tail = vals[int(len(vals) * 0.8):]
        if len(tail) >= 3:
            tail_range = max(tail) / max(min(tail), 1e-20)
            if tail_range < 1.1:
                return True  # stuck at high value
    return False


def _check_mass_conservation(log: str) -> bool:
    """Scan log for continuity residual below 1e-3."""
    for line in log.splitlines():
        low = line.lower()
        if "continuity errors" in low or "continuity" in low:
            for tok in line.split():
                try:
                    v = float(tok)
                    if abs(v) < 1e-3:
                        return True
                    break
                except ValueError:
                    pass
    return False


def _residual_trend_quality(history: dict[str, list[float]]) -> float:
    """Return 0.0–1.0 score for residual convergence quality.

    Considers: rate of decrease, absence of late spikes, final magnitude.
    """
    if not history:
        return 0.0
    scores = []
    for field, vals in history.items():
        if len(vals) < 4:
            continue
        initial = max(vals[:3])
        final = vals[-1]
        if initial <= 0 or final <= 0:
            continue
        # Reduction ratio (log scale)
        ratio = initial / max(final, 1e-20)
        # Penalise late spikes (last 10% > 2x median of last 30%)
        last10 = vals[int(len(vals) * 0.9):]
        last30 = vals[int(len(vals) * 0.7):]
        median30 = sorted(last30)[len(last30) // 2] if last30 else 1e-10
        spike_penalty = 1.0 if (last10 and max(last10) < 2 * max(median30, 1e-20)) else 0.7
        import math
        field_score = min(1.0, math.log10(max(ratio, 1.0)) / 4.0) * spike_penalty
        scores.append(field_score)
    return sum(scores) / len(scores) if scores else 0.0


def compute_reward(
    run_result: RunResult, params: CFDParams, solver: str, case_dir: Path = None
) -> tuple[float, str]:
    score = 0.0
    feedback = []
    log = run_result.log

    # ── Immediate fatal failures ───────────────────────────────────────────
    if "FOAM FATAL ERROR" in log or "FOAM FATAL Exception" in log:
        return 0.0, f"FATAL ERROR: {run_result.error_message[:200]}"
    if "TIMEOUT" in run_result.error_message or run_result.runtime < 0:
        return 0.0, "TIMEOUT"

    # ── Convergence signal: +0.40 / +0.15 / +0.05 ────────────────────────
    if run_result.converged:
        score += 0.40
    elif run_result.success and run_result.final_residuals:
        score += 0.15
        feedback.append("ran but did not converge")
    elif run_result.success:
        score += 0.05
        feedback.append("ran with no residual output")

    # ── Residual magnitude: +0.20 / +0.10 ────────────────────────────────
    if run_result.final_residuals:
        max_res = max(run_result.final_residuals.values())
        if max_res < 1e-4:
            score += 0.20
        elif max_res < 1e-3:
            score += 0.10
            feedback.append(f"residuals at {max_res:.1e}")
        else:
            feedback.append(f"residuals at {max_res:.1e}")
    else:
        feedback.append("no residuals found in log")

    # ── Residual trend quality: +0.0 to +0.10 ────────────────────────────
    trend = _residual_trend_quality(run_result.residual_history)
    score += trend * 0.10
    if trend < 0.2 and run_result.residual_history:
        feedback.append("poor residual convergence trend")

    # ── Mass conservation: +0.05 ──────────────────────────────────────────
    if _check_mass_conservation(log):
        score += 0.05
    else:
        feedback.append("mass conservation not verified")

    # ── Correct solver: +0.10 ─────────────────────────────────────────────
    expected = select_solver(params)
    if solver == expected:
        score += 0.10
    else:
        feedback.append(f"solver {solver} (expected {expected})")

    # ── Valid BCs: +0.05 ──────────────────────────────────────────────────
    if case_dir and _bc_valid(case_dir):
        score += 0.05
    elif not case_dir:
        score += 0.03

    # ── Penalties ─────────────────────────────────────────────────────────
    if run_result.mesh_max_non_ortho > 70:
        score -= 0.10
        feedback.append(f"poor mesh quality (non-ortho={run_result.mesh_max_non_ortho:.1f}°)")

    if run_result.runtime > 300:
        score -= 0.10
        feedback.append(f"slow ({run_result.runtime:.0f}s)")

    if _residuals_plateau(run_result.residual_history, run_result.final_residuals):
        score -= 0.05
        feedback.append("residuals plateaued (stagnated)")

    score = max(0.0, min(1.0, score))
    feedback_str = "; ".join(feedback) if feedback else "OK"
    return score, feedback_str
