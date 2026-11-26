"""데이터셋 A 문항 → 프롬프트 표.
 
과제
  translate : 한국어 → 중국어 few-shot (영어 없음). 템플릿 4개(F2 신뢰도용)
  repeat    : 한국어 → 한국어 반복 (예측 1: 의미 해석 없는 과제에서도 영어 우세가 보이는지)
 
조건 (cond)
  input  : 한국어 입력어 (ko_input)                → 번역 성공 여부, I(w)의 왼쪽 항
  ctrl1/2: 대조어 (ko_control_final의 두 단어)     → I(w)의 오른쪽 항
  en     : 영어 단어 자체를 입력 (F1 양성 대조)     → 영어 입력에서 간섭이 더 커야 함
 
few-shot 단어는 데이터셋 A의 입력어·대조어·정답·방해어와 겹치지 않고,
영어·한국어 모두 흔한 다의어가 아닌 것으로 골랐다(사과=謝過, 책=to book 등은 뺌).
템플릿을 바꾸려면 아래 TEMPLATES만 고치면 된다.
"""
import pandas as pd
 
# 정답을 모두 두 글자로: 한 글자 예시(猫, 水)가 있으면 모델이 春天 대신 春처럼 한 글자로 답하는 경향이 생김
SHOTS_A = [("우산", "雨伞"), ("의자", "椅子"), ("학교", "学校"), ("기차", "火车"), ("병원", "医院")]
SHOTS_B = [("우유", "牛奶"), ("할머니", "奶奶"), ("바다", "大海"), ("연필", "铅笔"), ("창문", "窗户")]
 
# 각 템플릿: (shots, 예시 한 줄 포맷, 질문 포맷, 정답 앞 구분자)
# 구분자 sep: 모델이 이어서 낼 정답 문자열 앞에 붙는 것. ':' / '='로 끝나면 ' ', 따옴표로 끝나면 ''.
TEMPLATES = {
    "T1": (SHOTS_A, "한국어: {ko}\n中文: {zh}\n\n", "한국어: {ko}\n中文:", " "),
    "T2": (SHOTS_A, "{ko} = {zh}\n", "{ko} =", " "),
    "T3": (SHOTS_A, '"{ko}" → "{zh}"\n', '"{ko}" → "', ""),
    "T4": (SHOTS_B, "한국어: {ko}\n中文: {zh}\n\n", "한국어: {ko}\n中文:", " "),
}
REPEAT = (SHOTS_A, "한국어: {ko}\n한국어: {ko}\n\n", "한국어: {ko}\n한국어:", " ")
 
 
def _fill(tpl, word):
    shots, ex, q, sep = tpl
    return "".join(ex.format(ko=k, zh=z) for k, z in shots) + q.format(ko=word), sep
 
 
# 대화 형식(--chat)에서 붙이는 지시문. 영어는 넣지 않는다.
CHAT_INSTR = {
    "translate": "아래 예시처럼 마지막 한국어 단어를 중국어(简体中文) 단어 하나로 번역하세요. 번역한 단어만 쓰세요.",
    "repeat": "아래 예시처럼 마지막 한국어 단어를 그대로 한 번 더 쓰세요. 단어만 쓰세요.",
}
 
 
def to_chat(P: pd.DataFrame, tok) -> pd.DataFrame:
    """Instruct 모델용. 지시문 + 기존 few-shot 프롬프트를 사용자 메시지로 넣고, 모델 답변 차례에서 끊는다.
    답은 답변 차례 맨 앞에 바로 오므로 sep=''. 템플릿 4개(T1–T4)의 차이는 사용자 메시지 안에 그대로 남는다."""
    P = P.copy()
    P["prompt"] = [tok.apply_chat_template([{"role": "user", "content": CHAT_INSTR[t] + "\n\n" + p}],
                                           tokenize=False, add_generation_prompt=True)
                   for t, p in zip(P.task, P.prompt)]
    P["sep"] = ""
    return P
 
 
def build(df: pd.DataFrame, templates=None, with_repeat=True) -> pd.DataFrame:
    templates = templates or list(TEMPLATES)
    rows = []
    for _, r in df.iterrows():
        c1, c2 = [c.strip() for c in str(r.ko_control_final).split(";")][:2]
        # 'en' 조건 입력어: A는 en 그대로, B(한국어 다의어)는 en_input(정답 뜻의 영어, 배→ship)
        en_in = r.en_input if "en_input" in r and isinstance(r.en_input, str) else r.en
        conds = {"input": r.ko_input, "ctrl1": c1, "ctrl2": c2, "en": en_in}
        for t in templates:
            for cond, word in conds.items():
                prompt, sep = _fill(TEMPLATES[t], word)
                rows.append(dict(item=r.id, en=r.en, task="translate", template=t, cond=cond, word=word,
                                 prompt=prompt, sep=sep, answer=r.zh_answer, distractor=r.zh_distractor))
        if with_repeat:
            prompt, sep = _fill(REPEAT, r.ko_input)
            rows.append(dict(item=r.id, en=r.en, task="repeat", template="R", cond="input", word=r.ko_input,
                             prompt=prompt, sep=sep, answer=r.ko_input, distractor=r.zh_distractor))
    out = pd.DataFrame(rows)
    out.insert(0, "pid", range(len(out)))
    return out
 
 
def build_C(df: pd.DataFrame, templates=None, with_repeat=True) -> pd.DataFrame:
    """데이터셋 C(일반 단어: 바다 / 海 / sea, 무관 단어 문화 / 文化 / culture).
    조건은 input 하나. 방해어 자리에는 무관 단어의 중국어(zh_unrelated)를 넣어 출력 쪽 확률도 남긴다.
    무관 단어는 다른 문항의 입력어이기도 해서(순열), '같은 영어 토큰을 관련 입력 / 무관 입력에서 읽은 값'을
    따로 프롬프트를 만들지 않고 비교할 수 있다(analyze_C.py)."""
    templates = templates or list(TEMPLATES)
    rows = []
    for _, r in df.iterrows():
        for t in templates:
            prompt, sep = _fill(TEMPLATES[t], r.ko)
            rows.append(dict(item=r.id, en=r.en, task="translate", template=t, cond="input", word=r.ko,
                             prompt=prompt, sep=sep, answer=r.zh, distractor=r.zh_unrelated))
        if with_repeat:
            prompt, sep = _fill(REPEAT, r.ko)
            rows.append(dict(item=r.id, en=r.en, task="repeat", template="R", cond="input", word=r.ko,
                             prompt=prompt, sep=sep, answer=r.ko, distractor=r.zh_unrelated))
    out = pd.DataFrame(rows)
    out.insert(0, "pid", range(len(out)))
    return out
 
 
if __name__ == "__main__":
    import sys
    p = build(pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "../dataset_A/A_final_v2.csv"))
    print(len(p), p.groupby(["task", "cond"]).size().to_dict())
    print(p.prompt.iloc[0] + "⟨답⟩")
 