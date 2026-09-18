"""Korean prompts for the 7B judges and for the gpt-4o-mini label extractor."""
from . import LEVELS
from .data import clean_problem_text, clean_text, format_answer_set, format_options

SYSTEM_PROMPT = (
    "당신은 고등학교 수학 교육 전문가입니다. 학생의 오답 풀이를 분석하여, "
    "학생이 특정 수학 개념을 몰라서 틀린 것인지 아니면 개념은 알지만 부주의로 인한 단순 실수로 틀린 것인지 판정합니다. "
    "여기서 '실수'는 부주의한 오류(mistake)를 뜻하며 실수(real number)가 아닙니다. 반드시 한국어로 답하세요."
)

LEVEL_DESC = {
    "type": "유형(type) 개념",
    "theme": "주제(theme) 개념",
}

USER_TEMPLATE = """다음은 한 학생이 틀린 수학 문제와 그 학생의 풀이 과정입니다.

## 판정 대상 개념
- 대상 개념: **{concept}** ({level_desc})
- 교육과정 위치: {chapter} > {section} > {unit} > {theme} > {type}

## 문제
{problem}

## 보기
{options}

## 정답
{answer}

## 정답 풀이 (모범 풀이)
{explanation}

## 학생이 제출한 답
{student_answer}

## 학생 풀이 과정
{student_process}

## 지시
1. 학생 풀이를 정답 풀이와 단계별로 비교하여, 오류가 처음 발생한 지점을 구체적으로 찾으세요.
2. 그 오류가 대상 개념 "{concept}"에 대한 이해 부족(개념을 모르거나 잘못 알고 있음)에서 비롯된 것인지, \
아니면 대상 개념은 제대로 이해하고 적용했으나 계산 오류·전사 오류·조건 누락·답 선택 오류 등 부주의로 인한 단순 실수에서 비롯된 것인지 판단하세요.
3. 학생 풀이가 비어 있거나 너무 짧아 근거를 찾을 수 없으면, 또는 두 가지 중 어느 쪽인지 분명히 가르기 어려우면 "판단 불가"로 판정하세요.
4. 분석을 간결하게 서술한 뒤, 마지막 줄에 반드시 아래 세 가지 중 하나만 정확히 적으세요.

최종 판정: 개념 부족
최종 판정: 단순 실수
최종 판정: 판단 불가"""


def build_messages(row: dict, level: str) -> list[dict]:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {list(LEVELS)}")
    concept = row[LEVELS[level]]
    process = row.get("problem_solving_process", "").strip() or "(풀이 없음)"
    user = USER_TEMPLATE.format(
        concept=concept,
        level_desc=LEVEL_DESC[level],
        chapter=row["curriculum_chapter_title"],
        section=row["curriculum_section_title"],
        unit=row["curriculum_unit_title"],
        theme=row["curriculum_theme_title"],
        type=row["curriculum_type_title"],
        problem=clean_problem_text(row),
        options=format_options(row),
        answer=format_answer_set(row["solution_answer_sets"], row),
        explanation=clean_text(row["solution_explanation"]),
        student_answer=format_answer_set(row["student_answer_set"], row),
        student_process=process,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


EXTRACT_SYSTEM = (
    "당신은 텍스트 분류기입니다. 수학 교육 전문가(AI)가 학생의 오답 풀이를 분석하여 쓴 판정문을 읽고, "
    "그 판정문이 어떤 결론을 내렸는지만 추출합니다. 판정문의 내용이 옳은지는 평가하지 않습니다."
)

EXTRACT_TEMPLATE = """대상 개념: "{concept}"

아래 판정문은 학생이 위 개념을 몰라서 틀렸는지 분석한 글입니다. 이 판정문이 내린 결론을 다음 세 가지 중 하나로 분류하세요.

- concept_gap: 학생이 대상 개념 "{concept}"을(를) 모르거나 잘못 이해해서 틀렸다고 명확히 결론냄
- slip: 대상 개념은 이해하고 있으나 계산 오류, 전사 오류, 조건 누락, 답 선택 오류 등 부주의로 인한 단순 실수로 틀렸다고 명확히 결론냄
- ambiguous: 판정문이 "판단 불가"로 결론내렸거나, 결론 없이 끝났거나, 두 가지가 섞여 있어 어느 한쪽으로 분명히 분류할 수 없거나, 판정 대신 문제 풀이만 하고 있는 경우

판정문에 "최종 판정:" 줄이 있으면 그것을 우선하되, 본문과 명백히 모순되면 ambiguous로 분류하세요.
판정문은 영어나 중국어로 쓰여 있을 수 있습니다. 판정문을 쓴 AI가 한국어 "실수"를 "real number"/"实数"로 오역하는 경우가 있으므로, 최종 판정 자리에 "real number", "实数", "simple real error", "simple mistake", "calculation error", "单纯失误" 등이 있으면 slip으로, "concept deficiency", "lack of understanding of the concept", "概念不足", "concept结索" 등이 있으면 concept_gap으로 읽으세요.
evidence에는 분류 근거가 된 판정문의 구절을 한 문장만(200자 이내) 그대로 인용하세요.

## 판정문
{generation}"""


def build_extract_messages(concept: str, generation: str) -> list[dict]:
    return [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": EXTRACT_TEMPLATE.format(concept=concept, generation=generation)},
    ]


# ---------------------------------------------------------------------------
# MathEdu (English, no named target concept) — same structure as the Korean
# judge prompt so the API results are comparable with the 7B pipeline.

MATHEDU_SYSTEM_PROMPT = (
    "You are an expert mathematics teacher. You analyse a student's incorrect solution and decide "
    "whether the student failed because of a wrong or missing mathematical concept or operation, "
    "or because of a careless slip made while executing a correct method. "
    "Here 'slip' means a careless mistake, not a real number."
)

MATHEDU_USER_TEMPLATE = """Below is a math word problem that a student answered incorrectly, together with the student's work.

## Problem
{problem}

## Options
{options}

## Correct option
{correct}

## Reference solution
{rationale}

## Student's submitted answer
{student_answer}

## Student's work
{student_process}

## Instructions
1. Compare the student's work with the reference solution step by step and identify the first point where the error occurs.
2. Decide whether that error comes from a wrong or missing mathematical concept or operation \
(the student set up the wrong relationship, applied the wrong operation, misunderstood what the problem asks, \
or used an incorrect formula or method), or whether the method was correct and the student only made a careless slip \
(an arithmetic mistake, a copying or transcription error, a dropped unit, or writing down a different number than the one computed).
3. If the student's work is empty or too short to tell, or if the two causes cannot be clearly separated, answer "undecidable".
4. Write a concise analysis, then on the last line write exactly one of the following:

Final verdict: concept gap
Final verdict: slip
Final verdict: undecidable"""


def build_mathedu_messages(row: dict) -> list[dict]:
    user = MATHEDU_USER_TEMPLATE.format(
        problem=row["problem"].strip(),
        options=row["options"].strip() or "(open answer)",
        correct=row["correct"].strip(),
        rationale=row["rationale"].strip().strip('"'),
        student_answer=row["student_answer"].strip() or "(not recorded)",
        student_process=row["student_process"].strip() or "(no work shown)",
    )
    return [
        {"role": "system", "content": MATHEDU_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# KT-PSP-25, API judge (gpt-5.4-mini). Target concept is curriculum_type_title
# only; the whole curriculum path is given so the judge can see where that
# concept sits. The three labels follow the researcher's definitions: a slip
# covers anything mechanical (including misreading a number or condition),
# while anything that cannot be pinned down from the work shown — or a wrong
# unit / everyday assumption, where a slip and a real gap look identical —
# goes to ambiguous rather than being forced into slip.

KT_API_SYSTEM_PROMPT = (
    "당신은 고등학교 수학 교육 전문가입니다. 학생이 틀린 문제의 풀이 과정을 분석하여, "
    "그 문제가 속한 단원의 개념을 몰라서 틀린 것인지 판정합니다. "
    "여기서 '실수'는 부주의한 오류(mistake)를 뜻하며 실수(real number)가 아닙니다. "
    "반드시 한국어로 답하세요."
)

KT_API_USER_TEMPLATE = """다음은 한 학생이 틀린 수학 문제와 그 학생의 풀이 과정입니다.

## 판정 대상 개념
- 대상 개념: **{concept}**
- 교육과정 위치: {chapter} > {section} > {unit} > {theme} > {type}

## 문제
{problem}

## 보기
{options}

## 정답
{answer}

## 정답 풀이 (모범 풀이)
{explanation}

## 학생이 제출한 답
{student_answer}

## 학생 풀이 과정
{student_process}

## 판정 기준
학생 풀이를 정답 풀이와 비교하여 아래 세 가지 중 하나로 판정하세요.

- **concept_gap**: 대상 개념 "{concept}"을(를) 몰라서 틀린 경우. 대상 개념에는 정답 풀이에서 \
그 개념을 적용하는 데 직접 필요한 절차까지 포함합니다. 학생이 그 개념을 어떻게 잘못 알고 있는지, \
무엇을 모르고 있는지 오류 내용을 구체적으로 작성하세요.

- **slip**: 대상 개념은 올바르게 이해하고 적용했으나 사소한 실수로 틀린 경우. 다음이 여기에 해당합니다.
  - 계산 오류, 부호 오류
  - 식을 전개·정리하는 과정에서 숫자나 항을 빠뜨리거나 잘못 옮겨 적은 오류
  - 문제의 숫자나 조건을 잘못 읽은 것이 오류의 전부인 경우
  - 풀이는 맞았으나 답을 잘못 고르거나 잘못 입력한 경우
  어떤 실수인지 사유를 명시하세요.

- **ambiguous**: 다음 중 하나에 해당하면 단순 실수로 분류하지 말고 ambiguous로 판정하고, \
어느 경우이며 왜 그렇게 보았는지 사유를 명시하세요.
  - 단위(예: 1kg을 100g으로 계산), 사회적 통념(예: 일주일을 6일로 계산) 등 해당 단원에서 다루는 \
수학 개념이 아닌 것을 잘못 적용해서 틀린 경우 — 실수인지 그 개념을 정말 모르는 것인지 가릴 수 없습니다.
  - 학생 풀이가 없거나 너무 짧아 오류의 원인을 파악할 수 없는 경우
  - 여러 상황이 가능하여 어떤 오류인지 풀이만으로는 하나로 정할 수 없는 경우

## 출력 형식
판정 근거를 간결하게 서술한 뒤, 마지막 줄에 아래 세 가지 중 하나만 정확히 적으세요.

판정: concept_gap
판정: slip
판정: ambiguous"""


def build_kt_api_messages(row: dict) -> list[dict]:
    """KT-PSP-25 judgement prompt for the API judge (curriculum_type_title)."""
    process = row.get("problem_solving_process", "").strip() or "(풀이 없음)"
    user = KT_API_USER_TEMPLATE.format(
        concept=row["curriculum_type_title"],
        chapter=row["curriculum_chapter_title"],
        section=row["curriculum_section_title"],
        unit=row["curriculum_unit_title"],
        theme=row["curriculum_theme_title"],
        type=row["curriculum_type_title"],
        problem=clean_problem_text(row),
        options=format_options(row),
        answer=format_answer_set(row["solution_answer_sets"], row),
        explanation=clean_text(row["solution_explanation"]),
        student_answer=format_answer_set(row["student_answer_set"], row),
        student_process=process,
    )
    return [
        {"role": "system", "content": KT_API_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


KT_EXTRACT_SYSTEM = (
    "당신은 텍스트 분류기입니다. 수학 교육 전문가(AI)가 학생의 오답 풀이를 분석하여 쓴 판정문을 읽고, "
    "그 판정문이 내린 판정만 추출합니다. 판정문의 내용이 옳은지는 평가하지 않습니다."
)

KT_EXTRACT_TEMPLATE = """대상 개념: "{concept}"

아래 판정문은 학생이 위 개념을 몰라서 틀렸는지 분석한 글입니다. 판정문이 내린 판정을 추출하세요.
판정문이 그 판정을 내렸다고 볼 확실한 근거가 있을 때만 해당 라벨로 추출하고, 근거를 찾을 수 없으면 none으로 추출하세요.

- concept_gap: 대상 개념 "{concept}"(그 개념을 적용하는 데 직접 필요한 절차 포함)을 몰라서 틀렸다고 판정함
- slip: 대상 개념은 이해하고 적용했으나 계산 오류, 부호 오류, 식 전개 중 숫자·항 누락이나 전사 오류, \
문제의 숫자·조건 오독, 답 선택·입력 실수 등 사소한 실수로 틀렸다고 판정함
- ambiguous: 단위나 사회적 통념 등 해당 단원의 수학 개념이 아닌 것을 잘못 적용해서 틀렸다고 보았거나, \
풀이가 없거나 짧아 원인을 파악할 수 없다고 보았거나, 여러 상황이 가능해 하나로 정할 수 없다고 판정함
- none: 판정문에 판정이 없거나, 어느 판정인지 확실한 근거를 찾을 수 없거나, 서로 모순되는 판정이 둘 이상 제시된 경우

판정문 마지막의 "판정:" 줄이 있으면 그것을 우선하되, 본문의 서술과 명백히 모순되면 none으로 분류하세요.
evidence에는 판정의 근거가 된 판정문의 문장을 하나만(200자 이내) 그대로 인용하세요. none이면 빈 문자열로 두세요.

## 판정문
{generation}"""


def build_kt_extract_messages(concept: str, generation: str) -> list[dict]:
    return [
        {"role": "system", "content": KT_EXTRACT_SYSTEM},
        {"role": "user", "content": KT_EXTRACT_TEMPLATE.format(concept=concept, generation=generation)},
    ]
