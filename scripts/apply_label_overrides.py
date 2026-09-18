"""Apply the researcher's rule-based overrides to extracted KT API labels.

Re-runnable, and meant to be re-run after any re-extraction.

Rule (2026-09-18): a `none` label on a row whose student answer was never
recorded becomes `slip`. In those rows the judge finds the written work
correct, so the only thing that can have gone wrong is the submission itself
— a slip by our definition. `none` is kept where an answer *was* recorded
(e.g. 917440 submitted "-48." against the answer -48), since there the
mismatch is not simply a missing response.
"""
import _bootstrap  # noqa: F401

import argparse
from collections import Counter

from kc_judge.data import load_incorrect
from kc_judge.io import read_jsonl, write_jsonl
from kc_judge.kt_api import KT_OUT_DIR

OVERRIDE_NOTE = "override: none -> slip (work correct, student answer not recorded)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gpt-5.4-mini")
    args = ap.parse_args()

    answered = {
        r["session_id"]: bool(r["student_answer_set"].strip()) for r in load_incorrect()
    }
    path = KT_OUT_DIR / "labels" / f"{args.judge}.jsonl"
    rows = list(read_jsonl(path))
    changed = Counter()
    for r in rows:
        if r["label"] == "none" and not answered[r["session_id"]]:
            r["label"] = "slip"
            r["note"] = OVERRIDE_NOTE
            changed[r["session_id"]] += 1
    write_jsonl(path, rows)
    print(f"{path}: {len(changed)} rows overridden -> slip")
    print("  session_ids:", sorted(changed))
    print("  labels now:", dict(Counter(r["label"] for r in rows)))


if __name__ == "__main__":
    main()
