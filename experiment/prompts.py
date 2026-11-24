"""프롬프트 템플릿.

- 프롬프트는 정답이 시작되기 직전에서 끝난다. 정답 토큰은 따로 붙여 점수를 낸다.
- few-shot 예시는 한국어·중국어로만 쓴다. 영어가 들어가는 것은 양성 대조(translate_en)뿐이다.
- 과제
    translate     한국어 → 중국어
    translate_en  한국어 → (영어) → 중국어. 문항의 영어 다의어를 프롬프트에 넣어 영어 경유를 강제
    translate_ctx 예문 속 한국어 → 중국어. B 동음이의어의 뜻을 예문으로 정해 준다.
                  예문은 틀({W} 자리 + 받침에 맞춰 고르는 조사)이라 대조어도 같은 예문에 넣는다
                  (시장에서 배 한 상자를 샀다 → 시장에서 복숭아 한 상자를 샀다). few-shot에도 예문이 붙는다.
    repeat        한국어 → 한국어 반복 (영어가 필요 없는 과제)
"""
from __future__ import annotations

import re

# (한국어, 중국어, 영어). 데이터셋 문항·대조어와 겹치면 안 된다(check_fewshot).
FEWSHOT = [
    ("학교", "学校", "school"),
    ("거울", "镜子", "mirror"),
    ("친구", "朋友", "friend"),
    ("연필", "铅笔", "pencil"),
    ("창문", "窗户", "window"),
]

# translate_ctx용 few-shot 예문 틀 (FEWSHOT과 같은 순서)
FEWSHOT_CTX = [
    "{W}에 가는 길에 비가 내렸다.",
    "{W}{을/를} 보며 머리를 빗었다.",
    "주말에 {W}{와/과} 영화를 보았다.",
    "{W}{으로/로} 이름을 썼다.",
    "더워서 {W}{을/를} 활짝 열었다.",
]

_JOSA = re.compile(r"\{W\}(?:\{([^/{}]+)/([^/{}]+)\})?")


def fill(template: str, word: str) -> str:
    """예문 틀의 {W}에 단어를 넣고, 뒤의 {받침 있을 때/없을 때} 조사를 단어의 마지막 글자에 맞춰 고른다.
    으로/로는 ㄹ 받침 뒤에서도 '로'를 쓴다."""
    if "{W}" not in template:
        raise ValueError(f"예문 틀에 {{W}}가 없음: {template}")
    code = ord(word[-1]) - 0xAC00
    jong = code % 28 if 0 <= code < 11172 else 0

    def rep(m):
        a, b = m.group(1), m.group(2)
        if a is None:
            return word
        pair = {a, b}
        if pair == {"으로", "로"}:
            return word + ("으로" if jong not in (0, 8) else "로")
        cons, vow = (a, b) if a in ("을", "이", "은", "과", "으로", "이다", "이랑") else (b, a)
        return word + (cons if jong else vow)

    return _JOSA.sub(rep, template)


# translate_ctx 버전별 (머리말, 형식). 형식 함수는 (예문, 단어) → (정답 앞부분, 정답 뒷부분)
_CTX = {
    "v1": ("", lambda sent, w: (f"문장：{sent}\n한국어：{w}\n中文：", "")),
    "v2": ("다음 문장에 쓰인 한국어 단어를 중국어로 번역하세요.\n", lambda sent, w: (f"{sent} / {w}→", "")),
    "v3": ("请把句子中的韩语词翻译成中文。\n", lambda sent, w: (f"句子：{sent}\n韩语：{w}\n中文：", "")),
    "v4": ("", lambda sent, w: (f"「{sent}」에서 「{w}」의 중국어는「", "」")),
}


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


def build(task: str, version: str, word: str, en_hint: str = "", ctx: str = "") -> str:
    """정답 바로 앞에서 끝나는 프롬프트 문자열. ctx(예문 틀)는 translate_ctx에서만 쓴다."""
    if task == "repeat":
        head, fmt = _REPEAT[version]
        shots = [fmt(ko, None) for ko, _, _ in FEWSHOT]
        lines = [pre + ko + post for (pre, post), (ko, _, _) in zip(shots, FEWSHOT)]
        return head + "\n".join(lines) + "\n" + fmt(word, None)[0]
    if task == "translate_ctx":
        if not ctx:
            raise ValueError("translate_ctx에는 예문 틀(ctx)이 필요하다")
        head, fmt = _CTX[version]
        lines = []
        for (ko, zh, _), t in zip(FEWSHOT, FEWSHOT_CTX):
            pre, post = fmt(fill(t, ko), ko)
            lines.append(pre + zh + post)
        return head + "\n".join(lines) + "\n" + fmt(fill(ctx, word), word)[0]
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
