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
evidence에는 분류 근거가 된 판정문의 구절을 한 문장만 그대로 인용하세요.

## 판정문
{generation}"""


def build_extract_messages(concept: str, generation: str) -> list[dict]:
    return [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": EXTRACT_TEMPLATE.format(concept=concept, generation=generation)},
    ]
