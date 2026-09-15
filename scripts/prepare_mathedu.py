"""Build the MathEdu gold-label evaluation set.

Keeps wrong answers whose teacher review lists exactly one error of a type we
can map to concept_gap / slip, attaches the MathQA problem, and writes
data/mathedu_eval.jsonl plus a balanced uid list (all slip rows + an equal
number of concept_gap rows, seed 42) in outputs/mathedu/balanced_ids.json.
"""
import _bootstrap  # noqa: F401

import json
import random
from collections import Counter

from kc_judge.io import write_jsonl
from kc_judge.mathedu import (
    BALANCED_IDS_PATH,
    EVAL_PATH,
    GOLD_MAP,
    SPLITS,
    load_mathqa,
    load_split,
    numbers_overlap,
    problem_for,
)

SEED = 42


def main() -> None:
    mathqa_train = load_mathqa("train")
    mathqa_test = load_mathqa("test")
    print(f"MathQA train {len(mathqa_train)} rows, test {len(mathqa_test)} rows")

    rows = []
    for split in SPLITS:
        for r in load_split(split):
            if r["correct_or_not"] != "wrong":
                continue
            errors = (r.get("teacher_review") or {}).get("error") or []
            if len(errors) != 1 or errors[0].get("error_type") not in GOLD_MAP:
                continue
            err = errors[0]
            prob = problem_for(r["id"], mathqa_train, mathqa_test)
            student_text = f"{r['student_process']} {r['student_answer']}"
            rows.append(
                {
                    # A problem id can appear for several students within a
                    # split, so the student id is part of the key.
                    "uid": f"{split}:{r['id']}:s{r['student_id']}",
                    "split": split,
                    "id": r["id"],
                    "student_id": r["student_id"],
                    "gold": GOLD_MAP[err["error_type"]],
                    "error_type": err["error_type"],
                    "error_equation": err.get("error_equation", ""),
                    "teacher_advice_en": err.get("teacher_advice_en", ""),
                    "problem": prob["Problem"],
                    "options": prob["options"],
                    "correct": prob["correct"],
                    "rationale": prob["Rationale"],
                    "category": prob["category"],
                    "student_answer": r["student_answer"],
                    "student_process": r["student_process"],
                    "problem_match": numbers_overlap(student_text, prob["Problem"]),
                }
            )

    uids = [r["uid"] for r in rows]
    assert len(uids) == len(set(uids)), "uid collision"
    n = write_jsonl(EVAL_PATH, rows)
    print(f"wrote {n} rows -> {EVAL_PATH}")
    print("gold x split:", dict(Counter((r["split"], r["gold"]) for r in rows)))
    print("error_type:", dict(Counter(r["error_type"] for r in rows)))
    for region, pred in (("train.json", lambda r: r["id"] < 29837), ("test.json", lambda r: r["id"] >= 29837)):
        sub = [r for r in rows if pred(r)]
        ok = sum(r["problem_match"] for r in sub)
        print(f"problem_match {region}: {ok}/{len(sub)} ({ok / max(1, len(sub)):.0%})")

    slip = [r["uid"] for r in rows if r["gold"] == "slip"]
    concept = [r["uid"] for r in rows if r["gold"] == "concept_gap"]
    sampled = random.Random(SEED).sample(concept, len(slip))
    balanced = sorted(slip + sampled)
    BALANCED_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BALANCED_IDS_PATH, "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "n_slip": len(slip), "n_concept_gap": len(sampled), "uids": balanced}, f, indent=1)
    print(f"balanced set: {len(slip)} slip + {len(sampled)} concept_gap -> {BALANCED_IDS_PATH}")


if __name__ == "__main__":
    main()
