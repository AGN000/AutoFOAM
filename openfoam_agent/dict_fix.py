"""Dict-level fix loop (Stream B).

When a solver run produces a FOAM FATAL that points at a specific dictionary
(e.g.\\ ``system/fvSchemes line 23: keyword 'div(phi,U)' undefined``), the
default agent retry path re-runs the WHOLE pipeline (param extract → mesh
→ case write → solve). That is wasteful for a one-line dict typo and it
also discards the parts of the case that worked.

This module attempts a surgical fix instead:

    parse_fatal_file_target(log)   →  the offending dict path (or None)
    read_dict(case_dir, path)      →  current contents
    propose_fix(llm, ...)          →  corrected file content from the LLM
    apply_fix(case_dir, ...)       →  write the new file in-place
    rerun_solver(case_dir, ...)    →  invoke the solver again

If the dict-fix succeeds, we get a usable run without re-meshing or
re-extracting params. If the dict-fix fails, the agent falls back to its
existing full-pipeline retry. The two paths are complementary.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Patterns OpenFOAM emits in FOAM FATAL messages that point at a file.
#   "from file /path/to/case/system/fvSchemes at line 23"
#   "in file 'system/controlDict' at line 7"
#   "/path/to/case/0/U at line 15"
_FILE_AT_LINE = re.compile(
    r"(?:from\s+file\s+|in\s+file\s+'?|^\s*)"
    r"(/?[\w./-]*?(?:system|constant|0)/[\w.-]+)"
    r"\s*'?\s+at\s+line\s+(\d+)",
    re.MULTILINE | re.IGNORECASE,
)


def parse_fatal_file_target(log: str, case_dir: Optional[Path] = None) -> Optional[Path]:
    """Return the dict file path mentioned in a FOAM FATAL message, if any.

    Resolves to an absolute path under case_dir if given. Returns None when
    the FATAL doesn't blame a specific file (mesh-quality failures, generic
    crashes, NaN propagation), so the caller knows to fall back to the
    full-pipeline retry.
    """
    m = _FILE_AT_LINE.search(log)
    if not m:
        return None
    raw = m.group(1)
    p = Path(raw)
    # Prefer paths that resolve under the supplied case_dir.
    if case_dir is not None:
        for keep in range(min(3, len(p.parts)), 0, -1):
            candidate = case_dir.joinpath(*p.parts[-keep:])
            if candidate.exists():
                return candidate
    if p.exists():
        return p
    # No file on disk to verify against (e.g. synthetic test, or the FATAL
    # came from a relative path the agent can no longer locate). Return
    # the parsed path so the caller can still log / surface what was named.
    return p


def read_dict(target: Path) -> str:
    return target.read_text(errors="ignore")


_FIX_SYSTEM = """You are an expert OpenFOAM v2412 dictionary editor.

You will be given:
  1. The current FAILING contents of a single OpenFOAM dictionary file.
  2. The relevant lines of the solver's FOAM FATAL ERROR message.
  3. The original user prompt for context (geometry / regime / fluid).

Your job: emit the COMPLETE CORRECTED contents of that one dictionary file.
Rules:
  - Output the entire file, not a diff. The file must be valid OpenFOAM v2412.
  - Preserve every entry that wasn't part of the error. Do not refactor for style.
  - Make the smallest change that fixes the error.
  - Keep the FoamFile header block intact (FoamFile { version 2.0; format ascii; ... }).
  - DO NOT wrap the output in markdown fences. DO NOT add commentary before
    or after. Output begins with the file's first character and ends with
    its last.
"""


def propose_fix(llm, target: Path, current_text: str, log_tail: str,
                user_prompt: str) -> str:
    """Ask the LLM to emit a corrected version of the offending dict file.

    Returns the new file content (string). On any error (LLM crash, empty
    output) returns "" — caller should treat that as 'fix unavailable, fall
    back to full-pipeline retry'.
    """
    rel = target.name
    user = (
        f"FAILING FILE (relative path: {rel}):\n"
        f"```\n{current_text}\n```\n\n"
        f"FOAM FATAL message (last lines):\n"
        f"```\n{log_tail}\n```\n\n"
        f"ORIGINAL USER PROMPT (context):\n{user_prompt}\n"
    )
    chats = [
        {"role": "system", "content": _FIX_SYSTEM},
        {"role": "user",   "content": user},
    ]
    try:
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=2048, n=1)
        resp = llm.chat(chats, sampling_params=sp)
    except Exception:
        return ""
    text = resp[0].outputs[0].text
    # Strip accidental fences / leading whitespace
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def apply_fix(target: Path, new_content: str) -> None:
    """Atomic-ish replace: write to a temp sibling and rename."""
    if not new_content.strip():
        return
    tmp = target.with_suffix(target.suffix + ".dictfix.tmp")
    tmp.write_text(new_content)
    tmp.replace(target)


def try_dict_fix(
    llm,
    case_dir: Path,
    run_result,
    user_prompt: str,
    runner_callable,
) -> Optional[object]:
    """End-to-end: parse → read → propose → apply → re-run → return new RunResult.

    Returns a RunResult if a dict-level fix was attempted, else None.
    The caller decides whether the new RunResult is good enough or whether
    to escalate to the full-pipeline retry.
    """
    target = parse_fatal_file_target(run_result.log + "\n" + run_result.error_message,
                                      case_dir=case_dir)
    if target is None or not target.exists():
        return None
    current = read_dict(target)
    log_tail = "\n".join(run_result.log.splitlines()[-30:])
    new = propose_fix(llm, target, current, log_tail, user_prompt)
    if not new or new == current.strip():
        return None
    apply_fix(target, new)
    return runner_callable()
