"""위키백과 문장 받기 (μ 계산 + affine 학습용, D5). 한국어·중국어 각 N문장, 섞어서 한 파일로.
run_cache.py가 짝수 줄은 μ, 홀수 줄은 affine 학습에 쓰므로 두 용도의 문장이 겹치지 않는다.
한 줄 = "언어\t문장". 섞어서 저장하므로 두 용도 모두에 세 언어가 고루 들어간다.
데이터셋 A 단어(입력어·대조어·정답·방해어)가 들어간 문장은 뺀다(누설 방지).
 
    pip install datasets
    python fetch_wiki.py --n 1000 --out wiki_ko_zh.txt
"""
import argparse
import random
import re
 
import pandas as pd
from datasets import load_dataset
 
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1000, help="언어당 문장 수")
ap.add_argument("--langs", default="ko,zh,en", help="affine이 특정 언어 쪽으로 기울지 않게 영어도 같은 수만큼 넣음")
ap.add_argument("--out", default="wiki_ko_zh.txt")
ap.add_argument("--data", default="../dataset_A/A_final_v2.csv")
ap.add_argument("--snapshot", default="20231101")
a = ap.parse_args()
random.seed(0)
 
d = pd.read_csv(a.data)
ban = set(d.ko_input) | set(d.zh_answer) | set(d.zh_distractor) | set(";".join(d.ko_control_final).split(";"))
ban = {w for w in ban if isinstance(w, str) and len(w) >= 2}  # 한 글자 단어는 너무 흔해서 제외 기준에서 뺌
 
SPLIT = {"ko": r"(?<=[.!?])\s+", "zh": r"(?<=[。！？])", "en": r"(?<=[.!?])\s+"}
LEN = {"ko": (20, 120), "zh": (12, 80), "en": (40, 200)}
ban_en = set(d.en.str.lower())  # 영어 문장에는 데이터셋 A의 영어 단어가 들어간 것도 뺌
out = []
for lang in a.langs.split(","):
    ds = load_dataset("wikimedia/wikipedia", f"{a.snapshot}.{lang}", split="train", streaming=True).shuffle(seed=0, buffer_size=2000)
    got = []
    for art in ds:
        for s in re.split(SPLIT[lang], art["text"]):
            s = s.strip().replace("\n", " ")
            bad = any(w in s for w in ban) or (lang == "en" and ban_en & set(re.findall(r"[a-z]+", s.lower())))
            if LEN[lang][0] <= len(s) <= LEN[lang][1] and not bad:
                got.append(f"{lang}\t{s}")
                break  # 문서당 한 문장
        if len(got) >= a.n:
            break
    out += got
    print(lang, len(got))
random.shuffle(out)
open(a.out, "w", encoding="utf-8").write("\n".join(out) + "\n")
print("저장", a.out, len(out))
 