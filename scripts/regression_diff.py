from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, required=True,
                   help="Pinned baseline eval JSONL (with --with-files capture)")
    p.add_argument("--candidate", type=Path, required=True,
                   help="Candidate eval JSONL to compare against the baseline")
    p.add_argument("--score-tolerance", type=float, default=0.10,
                   help="Score drop > this is flagged (default: 0.10)")
    p.add_argument("--out", type=Path, default=None,
                   help="If set, write the per-prompt flag report to this JSON")
    p.add_argument("--strict-files", action="store_true",
                   help="Treat FILES_CHANGE as a hard regression (default: log only)")
    return p.parse_args()


def load(path: Path) -> dict[str, dict]:
    if not path.exists():
        print(f"[diff] ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(2)
    by_tag: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        tag = r.get("case_tag")
        if tag:
            by_tag[tag] = r
    return by_tag


def main():
    args = parse_args()
    base = load(args.baseline)
    cand = load(args.candidate)
    if not base:
        print(f"[diff] ERROR: no rows in baseline {args.baseline}", file=sys.stderr)
        sys.exit(2)

    only_in_baseline = set(base) - set(cand)
    only_in_candidate = set(cand) - set(base)
    common = set(base) & set(cand)

    flags_per_tag: dict[str, list[str]] = {}
    files_changed: list[str] = []

    for tag in sorted(common):
        b, c = base[tag], cand[tag]
        flags: list[str] = []
        if bool(b.get("success")) != bool(c.get("success")):
            # Only call it a regression if baseline was the passing side
            if b.get("success") and not c.get("success"):
                flags.append("SUCCESS_FLIP")
            else:
                flags.append("SUCCESS_FLIP_IMPROVE")   # not a regression
        if b.get("solver") != c.get("solver"):
            flags.append("SOLVER_CHANGE")
        score_b = float(b.get("score") or 0.0)
        score_c = float(c.get("score") or 0.0)
        if score_b - score_c > args.score_tolerance:
            flags.append(f"SCORE_DROP({score_b:.2f}→{score_c:.2f})")
        # Files-change check only fires if both sides captured them
        bf = b.get("case_files_text", "")
        cf = c.get("case_files_text", "")
        if bf and cf and bf != cf:
            files_changed.append(tag)
            if args.strict_files:
                flags.append("FILES_CHANGE")
        if flags:
            flags_per_tag[tag] = flags


    regressions = {tag: f for tag, f in flags_per_tag.items()
                   if any(fl in ("SUCCESS_FLIP", "SOLVER_CHANGE", "FILES_CHANGE")
                          or fl.startswith("SCORE_DROP")
                          for fl in f)}

    print(f"[diff] baseline  : {args.baseline.name} ({len(base)} cases)")
    print(f"[diff] candidate : {args.candidate.name} ({len(cand)} cases)")
    print(f"[diff] common    : {len(common)}")
    if only_in_baseline:
        print(f"[diff] missing in candidate ({len(only_in_baseline)}): "
              + ", ".join(sorted(only_in_baseline)[:6])
              + (" …" if len(only_in_baseline) > 6 else ""))
    if only_in_candidate:
        print(f"[diff] new in candidate ({len(only_in_candidate)})")
    print(f"[diff] case-files differ on {len(files_changed)} prompts "
          f"({'STRICT — counts as regression' if args.strict_files else 'logged only'})")

    if regressions:
        print(f"\n[diff] REGRESSIONS on {len(regressions)} prompts:")
        for tag in sorted(regressions):
            print(f"  {tag:<32} {', '.join(regressions[tag])}")
    else:
        print(f"\n[diff] No flagged regressions — candidate ≥ baseline per-prompt.")


    improvements = {tag: f for tag, f in flags_per_tag.items()
                    if "SUCCESS_FLIP_IMPROVE" in f and tag not in regressions}
    if improvements:
        print(f"\n[diff] Improvements (candidate now passes): {len(improvements)}")
        for tag in sorted(improvements)[:10]:
            print(f"  {tag}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "baseline": str(args.baseline),
            "candidate": str(args.candidate),
            "n_common": len(common),
            "regressions": regressions,
            "improvements": list(improvements),
            "files_changed": files_changed,
            "missing_in_candidate": sorted(only_in_baseline),
            "new_in_candidate": sorted(only_in_candidate),
        }, indent=2))
        print(f"\n[diff] full report → {args.out}")

    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
