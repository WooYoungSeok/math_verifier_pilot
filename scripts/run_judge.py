"""Generate KC-gap judgements for INCORRECT rows with a local Qwen 7B model.

    python scripts/run_judge.py --model math7b            # both levels
    python scripts/run_judge.py --model 7b --limit 20     # smoke test

Output: outputs/gen/{model}_{level}.jsonl, one row per session_id. Rows already
present are skipped, so an interrupted run resumes where it stopped.
"""
import _bootstrap  # noqa: F401

import argparse
import hashlib
import os
import time
from pathlib import Path

# Growing KV caches fragment the caching allocator; on Windows a full GPU then
# silently spills to system RAM and generation slows 20x instead of OOM-ing.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

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


def chunked_prefill(model, input_ids, attention_mask, chunk: int, max_new_tokens: int) -> StaticCache:
    """Fill a static KV cache for all but the last prompt token, `chunk` rows
    at a time. Prefill activations (~195 KB per token for a 7B model) are what
    cap the batch size, and they only need to exist for the rows being
    prefilled; the cache itself (~57 KB per token) is what the whole batch
    decodes from. A static cache keeps decode memory flat (no per-step
    re-allocation), and generate() picks it up as past_key_values and only
    processes the final prompt token."""
    B, T = input_ids.shape
    cfg = model.config
    cache = StaticCache(config=cfg, max_cache_len=T - 1 + max_new_tokens)
    cache.early_initialization(
        B, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads, model.dtype, model.device
    )
    ids, mask = input_ids[:, :-1], attention_mask[:, :-1]
    pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
    for j in range(0, B, chunk):
        part = DynamicCache()
        model(
            input_ids=ids[j : j + chunk],
            attention_mask=mask[j : j + chunk],
            position_ids=pos[j : j + chunk],
            past_key_values=part,
            use_cache=True,
            logits_to_keep=1,
        )
        for layer, static_layer in zip(part.layers, cache.layers):
            static_layer.keys[j : j + chunk, :, : T - 1].copy_(layer.keys)
            static_layer.values[j : j + chunk, :, : T - 1].copy_(layer.values)
        del part
    return cache


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

    # Batch size is capped by the KV-cache budget (~57 KB per token for a 7B
    # model in bf16; 72k tokens ~ 4 GB), so long prompts get smaller batches
    # and peak memory stays under the card's 24 GB.
    batches, i = [], 0
    while i < len(prompts):
        n = args.batch_size
        while n > 1 and n * (prompts[min(i + n, len(prompts)) - 1][2] + args.max_new_tokens) > args.batch_tokens:
            n -= 1
        batches.append(prompts[i : i + n])
        i += n

    t0 = time.time()
    n_tokens = 0
    with JsonlAppender(out) as w:
        pbar = tqdm(batches, desc=f"{model_key}/{level}")
        for batch in pbar:
            tb = time.time()
            enc = tok(
                [p[1] for p in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_ctx - args.max_new_tokens,
            ).to(model.device)
            with torch.inference_mode():
                cache = chunked_prefill(
                    model, enc.input_ids, enc.attention_mask, args.prefill_chunk, args.max_new_tokens
                )
                gen = model.generate(
                    **enc,
                    past_key_values=cache,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    eos_token_id=eos_ids,
                    pad_token_id=tok.pad_token_id,
                    disable_compile=True,
                )
                del cache
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
            del gen, enc, new
            torch.cuda.empty_cache()
            pbar.set_postfix(
                bs=len(batch),
                plen=batch[-1][2],
                s=f"{time.time() - tb:.0f}",
                peakGB=f"{torch.cuda.max_memory_reserved() / 2**30:.1f}",
            )
    dt = time.time() - t0
    print(f"  {len(prompts)} gens, {n_tokens} tokens in {dt/60:.1f} min ({n_tokens/dt:.0f} tok/s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--levels", nargs="+", choices=list(LEVELS), default=list(LEVELS))
    ap.add_argument("--batch-size", type=int, default=48, help="max rows per batch")
    ap.add_argument("--batch-tokens", type=int, default=72000,
                    help="max rows * (prompt + max_new_tokens) per batch (KV budget, ~4 GB)")
    ap.add_argument("--prefill-chunk", type=int, default=4, help="rows per prefill forward")
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
