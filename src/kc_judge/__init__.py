"""KT-PSP-25 incorrect-answer -> knowledge-component gap judgement pipeline."""

MODELS = {
    "math7b": "Qwen/Qwen2.5-Math-7B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
}

LEVELS = {
    "type": "curriculum_type_title",
    "theme": "curriculum_theme_title",
}

LABELS = ("concept_gap", "slip", "ambiguous")
