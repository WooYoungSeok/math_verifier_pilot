"""MathEdu (teacher-labelled student work on MathQA problems) as a gold-label
check for the concept_gap / slip judgement pipeline."""
import json
import re
from pathlib import Path

from .io import read_jsonl

DATA_DIR = Path("data")
SPLITS = ("train", "val", "test")
EVAL_PATH = DATA_DIR / "mathedu_eval.jsonl"
OUT_DIR = Path("outputs/mathedu")
BALANCED_IDS_PATH = OUT_DIR / "balanced_ids.json"

# Teacher error types we can map onto the pipeline's labels.
GOLD_MAP = {
    "Wrong mathematical operation/concept": "concept_gap",
    "Arithmetical error": "slip",
    "Careless error": "slip",
}

# mathedu `id` indexes the concatenation of the original MathQA files
# (train 29,837 / dev 4,475 / test 2,985); rows only fall in train and test.
MATHQA_TRAIN_ROWS = 29837
MATHQA_TEST_OFFSET = MATHQA_TRAIN_ROWS + 4475

API_MODELS = ("gpt-5.4-mini", "gpt-5.1")

# The judge is not given a named target concept for this dataset.
GENERIC_CONCEPT = "the mathematical concept or operation this problem requires"

_NUM = re.compile(r"\d+(?:\.\d+)?")


def load_split(split: str) -> list[dict]:
    with open(DATA_DIR / f"mathedu_{split}.json", encoding="utf-8") as f:
        return json.load(f)


def load_mathqa(name: str) -> list[dict]:
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def problem_for(mathedu_id: int, train: list[dict], test: list[dict]) -> dict:
    if mathedu_id < MATHQA_TRAIN_ROWS:
        return train[mathedu_id]
    if mathedu_id >= MATHQA_TEST_OFFSET:
        return test[mathedu_id - MATHQA_TEST_OFFSET]
    raise ValueError(f"mathedu id {mathedu_id} falls in the MathQA dev range, which is not available")


def numbers_overlap(student_text: str, problem_text: str) -> bool:
    """Cheap sanity check that a student's work belongs to the problem: they
    share at least one number of 2+ digits (excluding round constants)."""
    skip = {"10", "100", "1000"}
    s = {n for n in _NUM.findall(student_text) if len(n) >= 2 and n not in skip}
    p = {n for n in _NUM.findall(problem_text) if len(n) >= 2 and n not in skip}
    return bool(s & p)


def load_eval() -> list[dict]:
    rows = list(read_jsonl(EVAL_PATH))
    if not rows:
        raise SystemExit(f"{EVAL_PATH} missing or empty; run scripts/prepare_mathedu.py first")
    return rows


def load_balanced_ids() -> set[str]:
    with open(BALANCED_IDS_PATH, encoding="utf-8") as f:
        return set(json.load(f)["uids"])
