"""Score API-judge labels against MathEdu teacher labels.

    python scripts/eval_mathedu.py               # every model with labels
    python scripts/eval_mathedu.py --model gpt-5.1

Reports, for the balanced subset and the full set: accuracy, balanced
accuracy, macro-F1 (ambiguous counts as wrong), coverage (share of decided
rows) and accuracy on decided rows, the gold x predicted confusion, and recall
per original teacher error type. Writes outputs/mathedu/eval_{model}.csv (one
row per uid) and outputs/mathedu/summary.md.
"""
import _bootstrap  # noqa: F401

import argparse

import pandas as pd

from kc_judge.io import read_jsonl
from kc_judge.mathedu import API_MODELS, OUT_DIR, load_balanced_ids, load_eval

CLASSES = ("concept_gap", "slip")


def metrics(df: pd.DataFrame) -> dict:
    gold, pred = df["gold"], df["pred"]
    decided = pred.isin(CLASSES)
    recalls = {c: float((pred[gold == c] == c).mean()) if (gold == c).any() else float("nan") for c in CLASSES}
    f1s = []
    for c in CLASSES:
        tp = int(((gold == c) & (pred == c)).sum())
        fp = int(((gold != c) & (pred == c)).sum())
        fn = int(((gold == c) & (pred != c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "n": len(df),
        "accuracy": float((gold == pred).mean()),
        "balanced_accuracy": sum(recalls.values()) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "coverage": float(decided.mean()),
        "accuracy_decided": float((gold[decided] == pred[decided]).mean()) if decided.any() else float("nan"),
        "recall_concept_gap": recalls["concept_gap"],
        "recall_slip": recalls["slip"],
    }


def evaluate(model: str, gold_rows: list[dict], balanced: set[str]) -> tuple[pd.DataFrame, list[dict]]:
    labels = {r["uid"]: r for r in read_jsonl(OUT_DIR / "labels" / f"{model}.jsonl")}
    gens = {r["uid"]: r for r in read_jsonl(OUT_DIR / "gen" / f"{model}.jsonl")}
    rows = []
    for g in gold_rows:
        l, gen = labels.get(g["uid"]), gens.get(g["uid"])
        if l is None:
            continue
        rows.append(
            {
                "uid": g["uid"], "split": g["split"], "gold": g["gold"], "error_type": g["error_type"],
                "pred": l["label"], "evidence": l.get("evidence", ""), "finish_reason": l.get("finish_reason", ""),
                "problem_match": g["problem_match"], "balanced": g["uid"] in balanced,
                "teacher_advice_en": g["teacher_advice_en"],
                "generation": gen["generation"] if gen else "",
            }
        )
    df = pd.DataFrame(rows)
    missing = len(gold_rows) - len(df)
    df.to_csv(OUT_DIR / f"eval_{model}.csv", index=False, encoding="utf-8-sig")

    lines = [f"## {model}", "", f"labelled rows: {len(df)} / {len(gold_rows)} (missing {missing})", ""]
    summary = []
    for name, sub in (("balanced", df[df["balanced"]]), ("full", df), ("full, problem_match only", df[df["problem_match"]])):
        m = metrics(sub)
        summary.append({"model": model, "subset": name, **m})
        lines.append(f"### {name} (n={m['n']})")
        lines.append("")
        lines.append("| accuracy | balanced acc | macro-F1 | coverage | acc (decided) | recall concept_gap | recall slip |")
        lines.append("|---|---|---|---|---|---|---|")
        lines.append(
            f"| {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['macro_f1']:.3f} | {m['coverage']:.3f} "
            f"| {m['accuracy_decided']:.3f} | {m['recall_concept_gap']:.3f} | {m['recall_slip']:.3f} |"
        )
        lines.append("")
        conf = pd.crosstab(sub["gold"], sub["pred"], rownames=["gold"], colnames=["pred"], dropna=False)
        lines.append("confusion (rows = gold, cols = predicted):")
        lines.append("")
        lines.append("```")
        lines.append(conf.to_string())
        lines.append("```")
        by_type = pd.DataFrame(
            {
                "n": sub.groupby("error_type").size(),
                "recall": sub.assign(hit=sub["pred"] == sub["gold"]).groupby("error_type")["hit"].mean(),
                "ambiguous": sub.assign(amb=sub["pred"] == "ambiguous").groupby("error_type")["amb"].mean(),
            }
        )
        lines.append("")
        lines.append("recall by teacher error type:")
        lines.append("")
        lines.append("```")
        lines.append(by_type.to_string())
        lines.append("```")
        lines.append("")
    return df, summary, lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=API_MODELS)
    args = ap.parse_args()

    gold_rows = load_eval()
    balanced = load_balanced_ids()
    models = [args.model] if args.model else [m for m in API_MODELS if (OUT_DIR / "labels" / f"{m}.jsonl").exists()]

    all_summary, all_lines = [], ["# MathEdu gold-label evaluation", "",
                                  f"gold rows: {len(gold_rows)} (balanced subset: {len(balanced)}, seed 42)", ""]
    for m in models:
        _, summary, lines = evaluate(m, gold_rows, balanced)
        all_summary += summary
        all_lines += lines
    table = pd.DataFrame(all_summary)
    all_lines[4:4] = ["## Overview", "", "```", table.round(3).to_string(index=False), "```", ""]
    text = "\n".join(all_lines)
    (OUT_DIR / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {OUT_DIR / 'summary.md'} and eval_{{model}}.csv")


if __name__ == "__main__":
    main()
