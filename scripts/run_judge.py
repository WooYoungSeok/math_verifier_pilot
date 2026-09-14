"""Generate KC-gap judgements for INCORRECT rows with a local Qwen 7B model.

    python scripts/run_judge.py --model math7b            # both levels
    python scripts/run_judge.py --model 7b --limit 20     # smoke test

Output: outputs/gen/{model}_{level}.jsonl, one row per session_id. Rows already
present are skipped, so an interrupted run resumes where it stopped.
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kc_judge import LEVELS, MODELS
from kc_judge.data import load_incorrect
from kc_judge.io import JsonlAppender, done_ids
from kc_judge.prompts import build_messages

GEN_DIR = Path("outputs/gen")


def gen_path(model_key: str, level: str) -> Path:
    return GEN_DIR / f"{model_key}_{level}.jsonl"


def load_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.eval()
    return tok, model


def run_level(tok, model, model_key: str, level: str, rows: list[dict], args) -> None:
    out = gen_path(model_key, level)
    done = done_ids(out)
    todo = [r for r in rows if r["session_id"] not in done]
    if args.limit:
        todo = todo[: max(0, args.limit - len(done))]
    print(f"[{model_key}/{level}] done={len(done)} todo={len(todo)} -> {out}")
    if not todo:
        return

    prompts = []
    for r in todo:
        text = tok.apply_chat_template(
            build_messages(r, level), tokenize=False, add_generation_prompt=True
        )
        prompts.append((r["session_id"], text, len(tok(text).input_ids)))
    # Length-sorted batches keep padding to a minimum.
    prompts.sort(key=lambda p: p[2])
    max_prompt = max(p[2] for p in prompts)
    print(f"  prompt tokens: max={max_prompt}, ctx budget={args.max_ctx}")

    eos_ids = [tok.eos_token_id]
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tok.eos_token_id:
        eos_ids.append(im_end)

    t0 = time.time()
    n_tokens = 0
    with JsonlAppender(out) as w:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"{model_key}/{level}"):
            batch = prompts[i : i + args.batch_size]
            enc = tok(
                [p[1] for p in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_ctx - args.max_new_tokens,
            ).to(model.device)
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    eos_token_id=eos_ids,
                    pad_token_id=tok.pad_token_id,
                )
            new = gen[:, enc.input_ids.shape[1] :]
            for (sid, text, n_in), ids in zip(batch, new):
                ids = ids.tolist()
                stop = next((k for k, t in enumerate(ids) if t in eos_ids), None)
                body = ids if stop is None else ids[:stop]
                n_tokens += len(body)
                w.write(
                    {
                        "session_id": sid,
                        "model": MODELS[model_key],
                        "level": level,
                        "prompt_hash": hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
                        "prompt_tokens": n_in,
                        "gen_tokens": len(body),
                        "finish_reason": "eos" if stop is not None else "length",
                        "generation": tok.decode(body, skip_special_tokens=True).strip(),
                    }
                )
    dt = time.time() - t0
    print(f"  {len(prompts)} gens, {n_tokens} tokens in {dt/60:.1f} min ({n_tokens/dt:.0f} tok/s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--levels", nargs="+", choices=list(LEVELS), default=list(LEVELS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-ctx", type=int, default=4096, help="Qwen2.5-Math context limit")
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows per level (debug)")
    args = ap.parse_args()

    rows = load_incorrect()
    tok, model = load_model(MODELS[args.model])
    for level in args.levels:
        run_level(tok, model, args.model, level, rows, args)


if __name__ == "__main__":
    main()
