"""Judge MathEdu rows with an OpenAI API model (concept gap vs slip).

    python scripts/run_api_judge.py --model gpt-5.4-mini --limit 5
    python scripts/run_api_judge.py --model gpt-5.1

Output: outputs/mathedu/gen/{model}.jsonl, same shape as run_judge.py rows
(uid instead of session_id). Resumable: uids already present are skipped.
"""
import _bootstrap  # noqa: F401

import argparse
import asyncio
import random
import time

from openai import AsyncOpenAI, BadRequestError
from tqdm import tqdm

from kc_judge.io import JsonlAppender, done_ids
from kc_judge.mathedu import API_MODELS, OUT_DIR, load_balanced_ids, load_eval
from kc_judge.prompts import build_mathedu_messages

GEN_DIR = OUT_DIR / "gen"


async def judge(client: AsyncOpenAI, sem: asyncio.Semaphore, row: dict, args) -> dict:
    messages = build_mathedu_messages(row)
    params = {"max_completion_tokens": args.max_completion_tokens}
    if args.temperature is not None:
        params["temperature"] = args.temperature
    async with sem:
        for attempt in range(6):
            try:
                resp = await client.chat.completions.create(model=args.model, messages=messages, **params)
                break
            except BadRequestError as e:
                # Some reasoning models refuse an explicit temperature; drop it
                # once and say so rather than silently changing the setting.
                if "temperature" in str(e) and "temperature" in params:
                    print(f"\n{args.model} rejected temperature={params['temperature']}: {e}. Retrying without it.")
                    params.pop("temperature")
                    continue
                raise
            except Exception as e:  # rate limit, transient network
                if attempt == 5:
                    return {"uid": row["uid"], "model": args.model, "generation": "", "finish_reason": f"error: {e!r}",
                            "gen_tokens": 0, "reasoning_tokens": 0}
                await asyncio.sleep(2**attempt + random.random())
    choice = resp.choices[0]
    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "uid": row["uid"],
        "model": args.model,
        "temperature": params.get("temperature"),
        "prompt_tokens": usage.prompt_tokens,
        "gen_tokens": usage.completion_tokens,
        "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0,
        "finish_reason": "eos" if choice.finish_reason == "stop" else choice.finish_reason,
        "generation": (choice.message.content or "").strip(),
    }


async def main_async(args) -> None:
    rows = load_eval()
    if args.balanced_only:
        keep = load_balanced_ids()
        rows = [r for r in rows if r["uid"] in keep]
    out = GEN_DIR / f"{args.model}.jsonl"
    done = done_ids(out, key="uid")
    todo = [r for r in rows if r["uid"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[{args.model}] rows={len(rows)} done={len(done)} todo={len(todo)} -> {out}")
    if not todo:
        return

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.time()
    n_tokens = 0
    with JsonlAppender(out) as w:
        tasks = [asyncio.create_task(judge(client, sem, r, args)) for r in todo]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=args.model):
            row = await fut
            n_tokens += row["gen_tokens"]
            w.write(row)
    dt = time.time() - t0
    print(f"  {len(todo)} gens, {n_tokens} completion tokens in {dt / 60:.1f} min")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=API_MODELS, required=True)
    ap.add_argument("--temperature", type=float, default=1.0, help="pass a negative value to omit")
    ap.add_argument("--max-completion-tokens", type=int, default=3000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--balanced-only", action="store_true")
    args = ap.parse_args()
    if args.temperature is not None and args.temperature < 0:
        args.temperature = None
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
