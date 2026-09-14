"""Loading KT-PSP-25 and rendering its JSON-encoded fields as readable text."""
import html
import json
import re
from pathlib import Path

from huggingface_hub import hf_hub_download

from .io import read_jsonl

DATASET_REPO = "jungypark/KT-PSP-25"
DATASET_FILE = "KT-PSP-25.json"
INCORRECT_PATH = Path("data/incorrect.jsonl")

_MC_ROW = re.compile(
    r"<mc-row>\s*<mc-index-item>(.*?)</mc-index-item>\s*<mc-item>(.*?)</mc-item>\s*</mc-row>",
    re.S,
)
_MC_BLOCK = re.compile(r"<mc>.*?</mc>", re.S)
_BOX = re.compile(r'<box(?:\s+label="([^"]*)")?\s*>(.*?)</box>', re.S)
_TAG = re.compile(r"</?(?:box|underline|mc|mc-body|mc-row|mc-index-item|mc-item)[^>]*>")


def load_raw() -> list[dict]:
    path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_incorrect() -> list[dict]:
    rows = list(read_jsonl(INCORRECT_PATH))
    if not rows:
        raise SystemExit(f"{INCORRECT_PATH} missing or empty; run scripts/prepare_data.py first")
    return rows


def mc_rows(text: str) -> list[tuple[str, str]]:
    """(index label, item) pairs from every <mc> block in a problem text."""
    return [(a.strip(), b.strip()) for a, b in _MC_ROW.findall(text)]


def clean_text(text: str, drop_mc: bool = False) -> str:
    """Render the light HTML markup as plain text: <box> becomes an indented
    block with its label, <mc> tables become one 'label item' line each (or
    are dropped when they duplicate the answer choices), other tags vanish."""

    def flatten(m: re.Match) -> str:
        if drop_mc:
            return "\n"
        return "\n" + "\n".join(f"{a} {b}" for a, b in mc_rows(m.group(0))) + "\n"

    def box(m: re.Match) -> str:
        label, body = (m.group(1) or "").strip(), m.group(2).strip()
        head = f"[{label}]\n" if label else ""
        return f"\n{head}{body}\n"

    text = html.unescape(text)
    text = _MC_BLOCK.sub(flatten, text)
    text = _BOX.sub(box, text)
    text = _TAG.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_problem_text(row: dict) -> str:
    # When problem_option is empty the <mc> block in the text *is* the choice
    # list; it is rendered separately by format_options, so drop it here.
    return clean_text(row["problem_text"], drop_mc=not row.get("problem_option"))


def parse_options(row: dict) -> list[tuple[str, str]]:
    """Answer choices as (label, text). Falls back to the <mc> rows embedded in
    problem_text, which is where the choices live when problem_option is empty."""
    if row.get("problem_option"):
        opts = json.loads(row["problem_option"])
        return [(o.get("label") or f"({o['index']})", o.get("text", "")) for o in opts]
    return mc_rows(row.get("problem_text", ""))


def format_options(row: dict) -> str:
    opts = parse_options(row)
    if not opts:
        return "(주관식)"
    return "\n".join(f"{label} {text}" for label, text in opts)


def _format_answer(a: dict, options: list[tuple[str, str]]) -> str:
    t = a.get("type")
    if t == "choice":
        i = int(a["index"])
        if 1 <= i <= len(options):
            label, text = options[i - 1]
            return f"{label} {text}".strip()
        return f"{i}번"
    if t == "fraction":
        return f"{a['numerator']}/{a['denominator']}"
    return str(a.get("value", ""))


def format_answer_set(raw: str, row: dict) -> str:
    """Render solution_answer_sets / student_answer_set JSON as text."""
    if not raw or not raw.strip():
        return "(기록 없음)"
    try:
        sets = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    options = parse_options(row)
    parts = []
    for s in sets:
        answers = [_format_answer(a, options) for a in s.get("answers", [])]
        parts.append(", ".join(answers) if answers else "(기록 없음)")
    return " / ".join(parts) if parts else "(기록 없음)"
