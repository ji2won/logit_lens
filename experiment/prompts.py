"""프롬프트 템플릿.

- 프롬프트는 정답이 시작되기 직전에서 끝난다. 정답 토큰은 따로 붙여 점수를 낸다.
- few-shot 예시는 한국어·중국어로만 쓴다. 영어가 들어가는 것은 양성 대조(translate_en)뿐이다.
- 과제
    translate     한국어 → 중국어
    translate_en  한국어 → (영어) → 중국어. 문항의 영어 다의어를 프롬프트에 넣어 영어 경유를 강제
    repeat        한국어 → 한국어 반복 (영어가 필요 없는 과제)
"""
from __future__ import annotations

# (한국어, 중국어, 영어). 데이터셋 문항·대조어와 겹치면 안 된다(check_fewshot).
FEWSHOT = [
    ("학교", "学校", "school"),
    ("하늘", "天空", "sky"),
    ("친구", "朋友", "friend"),
    ("연필", "铅笔", "pencil"),
    ("창문", "窗户", "window"),
]

# 버전별 (머리말, 한 줄 형식). 형식 함수는 (원문, 영어 또는 None, 라벨) → (정답 앞부분, 정답 뒷부분)
_TRANSLATE = {
    "v1": ("", lambda src, en: (f"한국어：{src}\n" + (f"English：{en}\n" if en else "") + "中文：", "")),
    "v2": ("다음 한국어 단어를 중국어로 번역하세요.\n",
           lambda src, en: (f"{src}→" + (f"{en}→" if en else ""), "")),
    "v3": ("请把韩语词翻译成中文。\n",
           lambda src, en: (f"韩语：{src}\n" + (f"英语：{en}\n" if en else "") + "中文：", "")),
    "v4": ("",
           lambda src, en: (f"「{src}」" + (f"(영어「{en}」)" if en else "") + "의 중국어는「", "」")),
}
_REPEAT = {
    "v1": ("", lambda src, en: (f"한국어：{src}\n한국어：", "")),
    "v2": ("다음 단어를 그대로 따라 쓰세요.\n", lambda src, en: (f"{src}→", "")),
    "v3": ("请照抄下面的韩语词。\n", lambda src, en: (f"韩语：{src}\n照抄：", "")),
    "v4": ("", lambda src, en: (f"「{src}」 다시 쓰기:「", "」")),
}


def build(task: str, version: str, word: str, en_hint: str = "") -> str:
    """정답 바로 앞에서 끝나는 프롬프트 문자열."""
    if task == "repeat":
        head, fmt = _REPEAT[version]
        shots = [fmt(ko, None) for ko, _, _ in FEWSHOT]
        lines = [pre + ko + post for (pre, post), (ko, _, _) in zip(shots, FEWSHOT)]
        return head + "\n".join(lines) + "\n" + fmt(word, None)[0]
    head, fmt = _TRANSLATE[version]
    use_en = task == "translate_en"
    lines = []
    for ko, zh, en in FEWSHOT:
        pre, post = fmt(ko, en if use_en else None)
        lines.append(pre + zh + post)
    return head + "\n".join(lines) + "\n" + fmt(word, en_hint if use_en else None)[0]


def check_fewshot(items) -> None:
    """few-shot 단어가 문항 입력어·대조어·정답과 겹치면 오류."""
    ko_shot = {k for k, _, _ in FEWSHOT}
    zh_shot = {z for _, z, _ in FEWSHOT}
    bad = []
    for it in items:
        if ({it.ko, *it.controls} & ko_shot) or ({it.zh_answer, it.zh_distractor} & zh_shot):
            bad.append(it.id)
    if bad:
        raise ValueError(f"few-shot 단어가 문항과 겹침: {bad}")
