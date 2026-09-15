"""Join raw judge generations and GPT-extracted labels by session_id.

Example:
    python scripts/join_gen_labels.py --model math7b --level type
    python scripts/join_gen_labels.py --model math7b --level theme
    python scripts/join_gen_labels.py            # every (model, level) that has both files

Output:
    outputs/joined/math7b_type.csv
    outputs/joined/math7b_type.jsonl

Audit columns: `evidence_in_generation` checks that the extractor's quoted
evidence really is a substring of the judge's generation, and
`generation_truncated` flags judge outputs cut off at max_new_tokens. Rows
with truncated=True and evidence_in_generation=False are the ones most
likely to be a label hallucinated over a broken generation.
"""
import _bootstrap  # noqa: F401

import argparse
import re
from pathlib import Path

import pandas as pd

from kc_judge import LEVELS, MODELS
from kc_judge.io import read_jsonl, write_jsonl

GEN_DIR = Path("outputs/gen")
LABEL_DIR = Path("outputs/labels")
OUT_DIR = Path("outputs/joined")


def _norm(s: str) -> str:
    """Letters/digits only: LaTeX markup, spacing and case are what the
    extractor most often changes when it "quotes"."""
    return re.sub(r"[^0-9a-z가-힣]", "", s.lower())


def load_unique(path: Path, source_name: str) -> dict:
    """Load JSONL as {session_id: row} and fail on duplicate session_ids."""
    by_id = {}
    duplicates = []
    for row in read_jsonl(path):
        sid = row["session_id"]
        if sid in by_id:
            duplicates.append(sid)
        by_id[sid] = row
    if duplicates:
        dupes = sorted(set(duplicates))
        raise ValueError(
            f"{source_name}: duplicate session_id found: {dupes[:20]}"
            + (" ..." if len(dupes) > 20 else "")
        )
    return by_id


def join_one(model: str, level: str) -> pd.DataFrame:
    gen_path = GEN_DIR / f"{model}_{level}.jsonl"
    label_path = LABEL_DIR / f"{model}_{level}.jsonl"
    if not gen_path.exists():
        raise FileNotFoundError(gen_path)
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    gens = load_unique(gen_path, "generation")
    labels = load_unique(label_path, "label")

    joined = []
    for sid in sorted(set(gens) | set(labels)):
        g = gens.get(sid, {})
        l = labels.get(sid, {})
        generation = g.get("generation", "")
        evidence = l.get("evidence", "")
        joined.append(
            {
                "session_id": sid,
                # ----- judge (Qwen) output -----
                "model": g.get("model", ""),
                "level": g.get("level", level),
                "prompt_hash": g.get("prompt_hash", ""),
                "prompt_tokens": g.get("prompt_tokens", ""),
                "gen_tokens": g.get("gen_tokens", ""),
                "gen_finish_reason": g.get("finish_reason", ""),
                "generation": generation,
                # ----- extractor (gpt-4o-mini) output -----
                "extractor": l.get("extractor", ""),
                "label": l.get("label", ""),
                "evidence": evidence,
                "note": l.get("note", ""),
                # ----- audit -----
                "has_generation": sid in gens,
                "has_label": sid in labels,
                # The extractor is told to quote verbatim, so check it did;
                # the normalized check forgives dropped LaTeX / spacing / case.
                "evidence_in_generation": bool(evidence) and evidence.strip() in generation,
                "evidence_in_generation_normalized": bool(evidence) and _norm(evidence) in _norm(generation),
                "generation_truncated": g.get("finish_reason") == "length",
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = OUT_DIR / f"{model}_{level}"
    write_jsonl(out_stem.with_suffix(".jsonl"), joined)
    df = pd.DataFrame(joined)
    df.to_csv(out_stem.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    # ---------- Summary ----------
    print(f"\n[{model}/{level}]")
    print(f"Generation rows : {len(gens)}")
    print(f"Label rows      : {len(labels)}")
    print(f"Joined IDs      : {len(joined)}")
    print(f"Missing labels  : {int((~df['has_label']).sum())}")
    print(f"Missing gens    : {int((~df['has_generation']).sum())}")
    print(f"Qwen length-cut : {int(df['generation_truncated'].sum())}")
    print("Labels          :", df["label"].value_counts().to_dict())

    has_ev = df["evidence"].astype(str).str.strip() != ""
    strict_miss = df[has_ev & ~df["evidence_in_generation"]]
    bad_evidence = df[has_ev & ~df["evidence_in_generation_normalized"]]
    print(f"Evidence not found verbatim in generation: {len(strict_miss)}"
          f"  (after normalizing LaTeX/space/case: {len(bad_evidence)})")
    suspicious = df[df["generation_truncated"] & ~df["evidence_in_generation_normalized"] & (df["label"] != "ambiguous")]
    print(f"Truncated + evidence missing + non-ambiguous label (likely hallucinated): {len(suspicious)}")
    if len(bad_evidence):
        print("\nSuspicious session_ids (evidence not in generation even normalized; first 30):")
        print(
            bad_evidence[["session_id", "gen_finish_reason", "label", "evidence"]]
            .assign(evidence=lambda d: d["evidence"].str.slice(0, 80))
            .head(30)
            .to_string(index=False)
        )
    print(f"\nWrote:\n  {out_stem}.csv\n  {out_stem}.jsonl")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), help="e.g. math7b or 7b")
    ap.add_argument("--level", choices=list(LEVELS))
    args = ap.parse_args()

    models = [args.model] if args.model else list(MODELS)
    levels = [args.level] if args.level else list(LEVELS)
    for m in models:
        for lv in levels:
            if args.model is None or args.level is None:
                # sweep mode: skip combos that are not ready yet
                if not (GEN_DIR / f"{m}_{lv}.jsonl").exists() or not (LABEL_DIR / f"{m}_{lv}.jsonl").exists():
                    print(f"[{m}/{lv}] not ready, skipping")
                    continue
            join_one(m, lv)


if __name__ == "__main__":
    main()
