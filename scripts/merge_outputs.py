"""Join the 8 judgement columns onto the INCORRECT rows and write CSV + JSONL.

Columns added per (level in {type, theme}) x (model in {math7b, 7b}):
    {level}_kc_label_{model}   gpt-4o-mini extracted verdict: concept_gap | slip | ambiguous
    {level}_kc_raw_{model}     the 7B model's full generation
    {level}_kc_finish_{model}  how that generation ended: eos (complete) | length (cut at max_new_tokens)

Each API judge in outputs/kt_api adds three more, against curriculum_type_title only:
    type_kc_label_{judge}      concept_gap | slip | ambiguous | none, after any override
    type_kc_verdict_{judge}    the judge's own verdict line, before extraction or override
    type_kc_raw_{judge}        that judge's full judgement
"""
import _bootstrap  # noqa: F401

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from kc_judge import LEVELS, MODELS
from kc_judge.kt_api import KT_LABELS, KT_OUT_DIR
from kc_judge.data import load_incorrect
from kc_judge.io import read_jsonl, write_jsonl

GEN_DIR = Path("outputs/gen")
LABEL_DIR = Path("outputs/labels")
OUT_STEM = Path("outputs/kc_judgements")

NEW_COLS = [
    f"{lv}_kc_{kind}_{m}" for m in MODELS for lv in LEVELS for kind in ("label", "raw", "finish")
]
LABEL_SET = [x for x in KT_LABELS if x != "none"]


VERDICT_RE = re.compile(r"판정:\s*(concept_gap|slip|ambiguous)")


def verdict_of(generation: str) -> str:
    """The judge's own verdict line, so an overridden label never hides it."""
    found = VERDICT_RE.findall(generation)
    return found[-1] if found else ""


def api_judges() -> list[str]:
    """API judges that have both generations and labels, by file stem."""
    gens = {p.stem for p in (KT_OUT_DIR / "gen").glob("*.jsonl")}
    return sorted(gens & {p.stem for p in (KT_OUT_DIR / "labels").glob("*.jsonl")})


def short(judge: str) -> str:
    """gpt-5.4-mini -> gpt54mini, so the column name stays a plain identifier."""
    return judge.replace("-", "").replace(".", "").replace("_", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-missing", action="store_true",
                    help="write output even if some rows lack judgements")
    args = ap.parse_args()

    rows = load_incorrect()
    by_id = {r["session_id"]: dict(r) for r in rows}
    for r in by_id.values():
        for c in NEW_COLS:
            r[c] = ""

    judges = api_judges()
    for r in by_id.values():
        for j in judges:
            for kind in ("label", "verdict", "raw"):
                r[f"type_kc_{kind}_{short(j)}"] = ""

    missing = Counter()
    for j in judges:
        gens = {g["session_id"]: g["generation"] for g in read_jsonl(KT_OUT_DIR / "gen" / f"{j}.jsonl")}
        labels = {g["session_id"]: g["label"] for g in read_jsonl(KT_OUT_DIR / "labels" / f"{j}.jsonl")}
        for sid, r in by_id.items():
            r[f"type_kc_raw_{short(j)}"] = gens.get(sid, "")
            r[f"type_kc_verdict_{short(j)}"] = verdict_of(gens.get(sid, ""))
            r[f"type_kc_label_{short(j)}"] = labels.get(sid, "")
            if sid not in gens:
                missing[f"type_kc_raw_{short(j)}"] += 1
            if sid not in labels:
                missing[f"type_kc_label_{short(j)}"] += 1

    for m in MODELS:
        for lv in LEVELS:
            gens = {g["session_id"]: g for g in read_jsonl(GEN_DIR / f"{m}_{lv}.jsonl")}
            labels = {g["session_id"]: g["label"] for g in read_jsonl(LABEL_DIR / f"{m}_{lv}.jsonl")}
            for sid, r in by_id.items():
                if sid in gens:
                    r[f"{lv}_kc_raw_{m}"] = gens[sid]["generation"]
                    r[f"{lv}_kc_finish_{m}"] = gens[sid]["finish_reason"]
                else:
                    missing[f"{lv}_kc_raw_{m}"] += 1
                if sid in labels:
                    r[f"{lv}_kc_label_{m}"] = labels[sid]
                else:
                    missing[f"{lv}_kc_label_{m}"] += 1

    if missing:
        print("WARNING missing values:", dict(missing))
        if not args.allow_missing:
            raise SystemExit("re-run run_judge.py / extract_labels.py, or pass --allow-missing")

    out_rows = [by_id[r["session_id"]] for r in rows]
    write_jsonl(OUT_STEM.with_suffix(".jsonl"), out_rows)
    df = pd.DataFrame(out_rows)
    df.to_csv(OUT_STEM.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"wrote {len(df)} rows x {len(df.columns)} cols -> {OUT_STEM}.csv / .jsonl")

    print("\nlabel distribution")
    for m in MODELS:
        for lv in LEVELS:
            col = f"{lv}_kc_label_{m}"
            print(f"  {col:28s}", dict(Counter(df[col])))
    for j in judges:
        for kind in ("label", "verdict"):
            col = f"type_kc_{kind}_{short(j)}"
            print(f"  {col:28s}", dict(Counter(df[col])))
    print("\nagreement between models (rows where both labelled)")
    for lv in LEVELS:
        a, b = df[f"{lv}_kc_label_math7b"], df[f"{lv}_kc_label_7b"]
        both = (a != "") & (b != "")
        if both.any():
            print(f"  {lv:6s} n={int(both.sum())} agree={(a[both] == b[both]).mean():.3f}")
            print(pd.crosstab(a[both], b[both], rownames=["math7b"], colnames=["7b"]).to_string())

    for j in judges:
        a = df[f"type_kc_label_{short(j)}"]
        decided = a.isin(LABEL_SET)
        print()
        print(f"agreement of {j} with the 7B judges (type level, rows it labelled)")
        for m in MODELS:
            b = df[f"type_kc_label_{m}"]
            both = decided & (b != "")
            print(f"  {m:8s} n={int(both.sum())} agree={(a[both] == b[both]).mean():.3f}")
            print(pd.crosstab(a[both], b[both], rownames=[j], colnames=[m]).to_string())


if __name__ == "__main__":
    main()
