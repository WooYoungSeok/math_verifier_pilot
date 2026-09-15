"""Extract the 3-way verdict from each 7B generation with gpt-4o-mini.

    python scripts/extract_labels.py                    # all model x level combos
    python scripts/extract_labels.py --model 7b --level type
    python scripts/extract_labels.py --dataset mathedu  # API-judge generations (outputs/mathedu/gen)

Output: outputs/labels/{model}_{level}.jsonl. Resumable: session_ids already
labelled are skipped.
"""
import _bootstrap  # noqa: F401

import argparse
import asyncio
import random
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm import tqdm

from kc_judge import LABELS, LEVELS, MODELS
from kc_judge.data import load_incorrect
from kc_judge.io import JsonlAppender, done_ids, read_jsonl
from kc_judge.prompts import build_extract_messages

GEN_DIR = Path("outputs/gen")
LABEL_DIR = Path("outputs/labels")
EXTRACT_MODEL = "gpt-4o-mini"


class Verdict(BaseModel):
    label: Literal["concept_gap", "slip", "ambiguous"]
    evidence: str


async def classify(client: AsyncOpenAI, sem: asyncio.Semaphore, concept: str, generation: str) -> dict:
    if not generation.strip():
        return {"label": "ambiguous", "evidence": "", "note": "empty generation"}
    async with sem:
        for attempt in range(6):
            try:
                resp = await client.chat.completions.parse(
                    model=EXTRACT_MODEL,
                    messages=build_extract_messages(concept, generation),
                    response_format=Verdict,
                    # Greedy first; a little temperature on retries breaks the
                    # rare degenerate loop where the evidence string never ends.
                    temperature=0 if attempt == 0 else 0.5,
                    max_tokens=600,
                )
                v = resp.choices[0].message.parsed
                if v is None:  # refusal / schema failure
                    return {"label": "ambiguous", "evidence": "", "note": "no parsed output"}
                return {"label": v.label, "evidence": v.evidence}
            except Exception as e:  # rate limit, transient network
                if attempt == 5:
                    return {"label": "ambiguous", "evidence": "", "note": f"error: {e!r}"}
                await asyncio.sleep(2**attempt + random.random())


async def run_extraction(client: AsyncOpenAI, gen_path: Path, out: Path, concept_of, key: str, tag: str, args) -> None:
    """Label every generation in gen_path that is not yet in out.
    concept_of(gen_row) -> the concept name to put in the extractor prompt."""
    gens = list(read_jsonl(gen_path))
    done = done_ids(out, key=key)
    todo = [g for g in gens if g[key] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[{tag}] generations={len(gens)} done={len(done)} todo={len(todo)} -> {out}")
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)

    async def one(g):
        v = await classify(client, sem, concept_of(g), g["generation"])
        return {
            key: g[key],
            "model": g["model"],
            **({"level": g["level"]} if "level" in g else {}),
            "extractor": EXTRACT_MODEL,
            "finish_reason": g.get("finish_reason"),
            **v,
        }

    with JsonlAppender(out) as w:
        tasks = [asyncio.create_task(one(g)) for g in todo]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=tag):
            w.write(await fut)


async def main_kt(args) -> None:
    rows = load_incorrect()
    concept_by_id = {r["session_id"]: r for r in rows}
    client = AsyncOpenAI()
    models = [args.model] if args.model else list(MODELS)
    levels = [args.level] if args.level else list(LEVELS)
    for m in models:
        for lv in levels:
            gen_path = GEN_DIR / f"{m}_{lv}.jsonl"
            if not gen_path.exists():
                print(f"[{m}/{lv}] no generations yet, skipping")
                continue
            field = LEVELS[lv]
            await run_extraction(
                client, gen_path, LABEL_DIR / f"{m}_{lv}.jsonl",
                lambda g, field=field: concept_by_id[g["session_id"]][field],
                "session_id", f"{m}/{lv}", args,
            )


async def main_mathedu(args) -> None:
    from kc_judge.mathedu import API_MODELS, GENERIC_CONCEPT, OUT_DIR

    client = AsyncOpenAI()
    models = [args.model] if args.model else list(API_MODELS)
    for m in models:
        gen_path = OUT_DIR / "gen" / f"{m}.jsonl"
        if not gen_path.exists():
            print(f"[{m}] no generations yet, skipping")
            continue
        await run_extraction(
            client, gen_path, OUT_DIR / "labels" / f"{m}.jsonl",
            lambda g: GENERIC_CONCEPT, "uid", f"mathedu/{m}", args,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["kt", "mathedu"], default="kt",
                    help="kt: outputs/gen/{model}_{level}; mathedu: outputs/mathedu/gen/{api model}")
    ap.add_argument("--model", help="kt: math7b|7b; mathedu: gpt-5.4-mini|gpt-5.1")
    ap.add_argument("--level", choices=list(LEVELS))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(main_mathedu(args) if args.dataset == "mathedu" else main_kt(args))


if __name__ == "__main__":
    main()
