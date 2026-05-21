"""Self-correction context builder for the agent retry loop.

Extracts structured failure signals from a RunResult + score and builds
an enriched retry context that goes back into the next attempt's prompt.

Used by agent.run() in place of failure_diagnosis.build_retry_context()
when the score falls below RETRY_SCORE_THRESHOLD. The richer context
(scorer feedback string + log tail + named cause) is what lets the LLM
actually fix mistakes on attempt N+1 instead of repeating them.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schemas import RunResult, CFDParams
from .failure_diagnosis import diagnose, build_retry_context, FailureType


@dataclass
class CorrectionSignals:
    """Structured failure signals harvested from the run."""
    diverged: bool
    mass_imbalance: float          # NaN if not parseable
    has_nan: bool
    has_fp_exception: bool
    log_tail: str                  # last ~40 lines
    failure_type: str              # FailureType enum value
    diagnosis_message: str
    scorer_feedback: str


def _parse_mass_imbalance(log: str) -> float:
    """Pull the worst-case continuity error from the solver log.

    Looks for the OpenFOAM 'time step continuity errors : sum local = X,
    global = Y, cumulative = Z' line. Returns max(|local|) over the run,
    or float('nan') if no such line was found.
    """
    worst = float("nan")
    for line in log.splitlines():
        low = line.lower()
        if "continuity errors" not in low:
            continue
        # "sum local = 1.234e-05, global = ..."
        for token in ("sum local =", "local =", "global ="):
            idx = low.find(token)
            if idx < 0:
                continue
            tail = line[idx + len(token):].lstrip()
            num = ""
            for ch in tail:
                if ch in "0123456789.eE+-":
                    num += ch
                else:
                    break
            try:
                v = abs(float(num))
                if v != v:    # NaN
                    continue
                if worst != worst or v > worst:
                    worst = v
            except ValueError:
                continue
            break
    return worst


def _is_diverged(run_result: RunResult) -> bool:
    """Residuals exploded, or last value > 10 (the failure_diagnosis threshold)."""
    if not run_result.final_residuals:
        return False
    return max(run_result.final_residuals.values()) > 10.0


def harvest(run_result: RunResult, score: float) -> CorrectionSignals:
    log = run_result.log
    log_lines = log.splitlines()
    log_tail = "\n".join(log_lines[-40:]) if log_lines else ""
    combined = (log + " " + run_result.error_message).lower()
    has_nan = "-nan" in combined or "1.#ind" in combined
    has_fp = ("floating point" in combined or "sigfpe" in combined
              or "overflow" in combined)
    diag = diagnose(run_result, score)
    # Re-use the existing scorer feedback by re-running compute_reward only
    # if we don't already have it; the agent already has it as `feedback`.
    return CorrectionSignals(
        diverged=_is_diverged(run_result),
        mass_imbalance=_parse_mass_imbalance(log),
        has_nan=has_nan,
        has_fp_exception=has_fp,
        log_tail=log_tail,
        failure_type=diag.failure_type.value,
        diagnosis_message=diag.message,
        scorer_feedback="",   # filled by caller
    )


def should_retry(score: float, signals: CorrectionSignals,
                 threshold: float) -> bool:
    """Trigger a retry on any of: low score, divergence, NaN/FP, mass-imbalance."""
    if score < threshold:
        return True
    if signals.diverged or signals.has_nan or signals.has_fp_exception:
        return True
    mi = signals.mass_imbalance
    if mi == mi and mi > 1e-2:    # not NaN, and bad
        return True
    return False


def build_context(
    run_result: RunResult,
    params: CFDParams,
    solver: str,
    score: float,
    feedback: str,
    attempt: int,
    rag_examples: list[str] | None = None,
    rag_with_content: list[dict] | None = None,
) -> str:
    """Build the next-attempt prompt suffix.

    Combines:
      (1) the existing failure_diagnosis hints (relaxation, BC fixes, etc.),
      (2) the scorer's textual feedback,
      (3) structured signals (mass-imbalance number, divergence flag),
      (4) last 40 lines of solver log,
      (5) (Stream A) full controlDict / fvSchemes / boundary-conditions text
          from the top-K matched tutorial cases, so the retry param-extractor
          sees adaptable templates instead of just case names.

    The LLM sees this exact text appended after the original prompt.
    """
    diag = diagnose(run_result, score)
    base = build_retry_context(diag, attempt, score, rag_examples)
    signals = harvest(run_result, score)

    extras: list[str] = []
    extras.append("")
    extras.append(f"Previous solver: {solver}")
    if params.turbulence_model:
        extras.append(f"Previous turbulence model: {params.turbulence_model.value}")
    extras.append(f"Scorer feedback: {feedback}")
    if signals.diverged:
        extras.append("- Residuals diverged (max > 10). Use stronger relaxation "
                      "and a smaller initial time step.")
    mi = signals.mass_imbalance
    if mi == mi and mi > 1e-3:
        extras.append(f"- Continuity / mass-imbalance is high ({mi:.2e}). "
                      "Increase nNonOrthogonalCorrectors and check inlet/outlet flux balance.")
    if signals.has_nan:
        extras.append("- NaN encountered in solver output. Reduce deltaT, "
                      "use upwind for div(phi,U), and check BC values for typos.")
    if signals.has_fp_exception:
        extras.append("- Floating-point exception. Likely cause: zero divisor in "
                      "BC, or unbounded turbulence quantity. Add bounded schemes.")
    if signals.log_tail:
        extras.append("\nLast lines of solver log:")
        extras.append("```")
        extras.append(signals.log_tail)
        extras.append("```")

    # Stream A: include full file content from similar tutorial cases.
    # The param extractor's LLM call will see this and can extract better
    # params (e.g. recognise the right BC patch naming, the right turbulence
    # quantities, the right fvSchemes choices for this regime).
    if rag_with_content:
        try:
            from .rag import TutorialRAG
            ctx = TutorialRAG.format_content_context(
                rag_with_content, max_chars_per_chunk=1500,
            )
        except Exception:
            ctx = ""
        if ctx:
            extras.append("")
            extras.append(ctx)

    return base + "\n" + "\n".join(extras)
