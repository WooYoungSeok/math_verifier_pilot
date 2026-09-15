# KT-PSP-25 Knowledge-Component gap judgement

For every `INCORRECT` row of [jungypark/KT-PSP-25](https://huggingface.co/datasets/jungypark/KT-PSP-25),
two local judges (`Qwen/Qwen2.5-Math-7B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`) read the
student's `problem_solving_process` and decide whether the error came from not knowing
(1) the `curriculum_type_title` concept and (2) the `curriculum_theme_title` concept.
`gpt-4o-mini` then extracts each free-form judgement into one of
`concept_gap` / `slip` / `ambiguous`.

## Run

```
pip install -r requirements.txt          # .env needs OPENAI_API_KEY
python scripts/prepare_data.py           # -> data/incorrect.jsonl (5,773 rows)
python scripts/run_judge.py --model math7b   # -> outputs/gen/math7b_{type,theme}.jsonl
python scripts/run_judge.py --model 7b       # -> outputs/gen/7b_{type,theme}.jsonl
python scripts/extract_labels.py             # -> outputs/labels/*.jsonl  (gpt-4o-mini)
python scripts/join_gen_labels.py            # -> outputs/joined/*.{csv,jsonl}  (audit: generation vs label)
python scripts/merge_outputs.py              # -> outputs/kc_judgements.{csv,jsonl}
```

`run_all.cmd` runs both `run_judge.py` calls detached from the terminal (they take hours).

Every step is resumable (rows already present in the output file are skipped).
`run_judge.py` uses HF transformers with greedy decoding, bf16, length-sorted batches
(~24 GB GPU, ~4 h per model on an RTX 4090).

## Output columns (added to the 20 original columns)

| column | meaning |
|---|---|
| `type_kc_label_math7b` | Math-7B verdict on `curriculum_type_title`, extracted by gpt-4o-mini |
| `type_kc_raw_math7b` | Math-7B full generation for that verdict |
| `theme_kc_label_math7b` | Math-7B verdict on `curriculum_theme_title` |
| `theme_kc_raw_math7b` | Math-7B full generation |
| `type_kc_finish_math7b` | `eos` = generation finished, `length` = cut off at 1024 tokens (verdict may be missing) |
| `type_kc_label_7b`, `type_kc_raw_7b`, `type_kc_finish_7b`, `theme_kc_*_7b` | same for Qwen2.5-7B-Instruct |

Labels: `concept_gap` = clearly failed because the concept is missing/misunderstood,
`slip` = concept understood, failed by a computational/transcription/selection slip,
`ambiguous` = judge said "판단 불가", gave no conclusion, or could not be classified either way.

Prompts are in `src/kc_judge/prompts.py`; the judge is asked to end with a
`최종 판정: 개념 부족 | 단순 실수 | 판단 불가` line, but the extractor reads the whole generation.

## Audit files (`outputs/joined/{model}_{level}.csv`)

One row per session_id with the judge generation and the extracted label side by side, plus
`evidence` (the sentence gpt-4o-mini quoted as its basis), `evidence_in_generation` (strict
substring check), `evidence_in_generation_normalized` (ignoring LaTeX/space/case) and
`generation_truncated`. Rows with `generation_truncated=True` and no matching evidence are the
ones where the label was most likely inferred from an unfinished judgement.
