"""Extract the 3-way verdict from each 7B generation with gpt-4o-mini.

    python scripts/extract_labels.py                    # all model x level combos
    python scripts/extract_labels.py --model 7b --level type

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
                    temperature=0,
                )
                v = resp.choices[0].message.parsed
                if v is None:  # refusal / schema failure
                    return {"label": "ambiguous", "evidence": "", "note": "no parsed output"}
                return {"label": v.label, "evidence": v.evidence}
            except Exception as e:  # rate limit, transient network
                if attempt == 5:
                    return {"label": "ambiguous", "evidence": "", "note": f"error: {e!r}"}
                await asyncio.sleep(2**attempt + random.random())


async def run_combo(client: AsyncOpenAI, model_key: str, level: str, concept_by_id: dict, args) -> None:
    gen_path = GEN_DIR / f"{model_key}_{level}.jsonl"
    out = LABEL_DIR / f"{model_key}_{level}.jsonl"
    gens = list(read_jsonl(gen_path))
    done = done_ids(out)
    todo = [g for g in gens if g["session_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[{model_key}/{level}] generations={len(gens)} done={len(done)} todo={len(todo)} -> {out}")
    if not todo:
        return

    sem = asyncio.Semaphore(args.concurrency)
    field = LEVELS[level]

    async def one(g):
        concept = concept_by_id[g["session_id"]][field]
        v = await classify(client, sem, concept, g["generation"])
        return {
            "session_id": g["session_id"],
            "model": g["model"],
            "level": level,
            "extractor": EXTRACT_MODEL,
            "finish_reason": g.get("finish_reason"),
            **v,
        }

    with JsonlAppender(out) as w:
        tasks = [asyncio.create_task(one(g)) for g in todo]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"{model_key}/{level}"):
            w.write(await fut)


async def main_async(args) -> None:
    rows = load_incorrect()
    concept_by_id = {r["session_id"]: r for r in rows}
    client = AsyncOpenAI()
    models = [args.model] if args.model else list(MODELS)
    levels = [args.level] if args.level else list(LEVELS)
    for m in models:
        for lv in levels:
            if not (GEN_DIR / f"{m}_{lv}.jsonl").exists():
                print(f"[{m}/{lv}] no generations yet, skipping")
                continue
            await run_combo(client, m, lv, concept_by_id, args)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--level", choices=list(LEVELS))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
