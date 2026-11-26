"""B_final.csv(한국어 다의어, '배' 음성 대조) → 실험 A 파이프라인이 읽는 열 이름으로 바꾼 B_final_v1.csv.
 
열 대응
  en          = en_distractor   lens 점수 s_L을 잴 영어 단어. A에서 방해어 뜻과 이어진 영어(spring)를 쟀듯,
                                B에서도 방해어 뜻의 영어(배→梨이면 pear)를 잰다.
  en_input    = en_target       'en' 조건 입력어. A에서는 영어 입력이 간섭을 키워야 했지만(양성 대조),
                                B에서는 영어(ship)로 넣으면 한국어 중의성이 사라지므로 간섭이 줄어야 정상.
  zipf_en     = zipf_en_distractor
  etym_ko_input = word_etym
나머지(ko_input, zh_answer, zh_distractor, ko_control_final, zipf_ko_input, zipf_zh_distractor)는 이름이 같다.
 
few-shot 예시 단어(prompts.py SHOTS_A/B)와 겹치는 대조어는 바꾼다.
    python make_B_v1.py B_final.csv B_final_v1.csv
"""
import sys
 
import pandas as pd
 
sys.path.insert(0, "../exp_A")
from prompts import SHOTS_A, SHOTS_B  # noqa: E402
 
SWAP = {"우유": "주스"}  # few-shot(우유→牛奶)과 겹침. 같은 음료 범주로 교체
 
src = sys.argv[1] if len(sys.argv) > 1 else "B_final.csv"
dst = sys.argv[2] if len(sys.argv) > 2 else "B_final_v1.csv"
d = pd.read_csv(src)
d = d[d.keep.astype(str).str.upper().isin(["Y", "TRUE"])].copy()
 
shot_ko = {k for k, _ in SHOTS_A + SHOTS_B}
shot_zh = {z for _, z in SHOTS_A + SHOTS_B}
notes = []
for i, r in d.iterrows():
    c = [x.strip() for x in str(r.ko_control_final).split(";")]
    new = [SWAP.get(x, x) for x in c]
    if new != c:
        notes.append(f"{r.id}: 대조어 {c} → {new} (few-shot과 겹침)")
        d.at[i, "ko_control_final"] = ";".join(new)
    bad = [w for w in [r.ko_input, *new] if w in shot_ko] + [w for w in [r.zh_answer, r.zh_distractor] if w in shot_zh]
    if bad:
        notes.append(f"{r.id}: 아직 few-shot과 겹침 {bad} → 확인 필요")
 
d["en"] = d.en_distractor
d["en_input"] = d.en_target
d["zipf_en"] = d.zipf_en_distractor
d["etym_ko_input"] = d.word_etym
d["direction"] = "B"
d.to_csv(dst, index=False, encoding="utf-8-sig")
print(f"{len(d)}행 저장: {dst} (단어 {d.ko_input.nunique()}개, 영어 {d.en.nunique()}개)")
print("\n".join(notes) or "few-shot 겹침 없음")
print(d.word_etym.value_counts().to_dict())
 