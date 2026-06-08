
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_FILE_AT_LINE = re.compile(
    r"(?:from\s+file\s+|in\s+file\s+'?|^\s*)"
    r"(/?[\w./-]*?(?:system|constant|0)/[\w.-]+)"
    r"\s*'?\s+at\s+line\s+(\d+)",
    re.MULTILINE | re.IGNORECASE,
)


def parse_fatal_file_target(log: str, case_dir: Optional[Path] = None) -> Optional[Path]:

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
