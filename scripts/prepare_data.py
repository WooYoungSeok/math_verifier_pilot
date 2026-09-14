"""Download KT-PSP-25 and write the INCORRECT rows to data/incorrect.jsonl."""
import _bootstrap  # noqa: F401

from kc_judge.data import INCORRECT_PATH, load_raw
from kc_judge.io import write_jsonl


def main() -> None:
    rows = load_raw()
    incorrect = [r for r in rows if r["result"] == "INCORRECT"]
    ids = [r["session_id"] for r in incorrect]
    assert len(ids) == len(set(ids)), "session_id is not unique among INCORRECT rows"
    n = write_jsonl(INCORRECT_PATH, incorrect)
    print(f"total rows: {len(rows)}  incorrect: {n}  -> {INCORRECT_PATH}")


if __name__ == "__main__":
    main()
