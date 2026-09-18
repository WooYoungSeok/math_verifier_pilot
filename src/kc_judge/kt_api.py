"""KT-PSP-25 judged by an API model against curriculum_type_title.

Unlike the 7B run this asks one question per row (the type-level concept) and
allows the extractor a fourth label, `none`, for judgements whose verdict
cannot be read off with confidence.
"""
from pathlib import Path

KT_OUT_DIR = Path("outputs/kt_api")
KT_LABELS = ("concept_gap", "slip", "ambiguous", "none")
