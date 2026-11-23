#!/usr/bin/env python3
"""한·중 위키백과에서 문장을 언어별 N개 뽑는다 (평균 성분 계산, 아핀 렌즈 학습용).

  python -m experiment.prep_corpus --n 1000
  → data/wiki_ko.jsonl, data/wiki_zh.jsonl   ({"lang":..., "text":...} 한 줄에 한 문장)

HF datasets의 wikimedia/wikipedia(20231101 덤프)를 스트리밍으로 읽어 전체를 받지 않는다.
다양성을 위해 문서당 최대 2문장, 길이 15–200자 문장만 쓴다.
"""
from __future__ import annotations

import argparse
import json
import random
import re

from .config import DATA_DIR

SPLIT = {"ko": re.compile(r"(?<=[.?!])\s+"), "zh": re.compile(r"(?<=[。！？])")}
NEED = {"ko": re.compile(r"[가-힣]"), "zh": re.compile(r"[一-鿿]")}


def sentences(lang: str, n: int, seed: int, dump: str):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", f"{dump}.{lang}", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    rng = random.Random(seed)
    out = []
    for doc in ds:
        cand = []
        for para in doc["text"].split("\n"):
            for s in SPLIT[lang].split(para.strip()):
                s = s.strip()
                if 15 <= len(s) <= 200 and NEED[lang].search(s) and not s.startswith(("|", "{", "*", "#")):
                    cand.append(s)
        rng.shuffle(cand)
        out.extend(cand[:2])
        if len(out) >= n:
            return out[:n]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default="20231101")
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for lang in ("ko", "zh"):
        ss = sentences(lang, args.n, args.seed, args.dump)
        path = DATA_DIR / f"wiki_{lang}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for s in ss:
                f.write(json.dumps({"lang": lang, "text": s}, ensure_ascii=False) + "\n")
        print(f"{lang}: {len(ss)}문장 → {path}", flush=True)


if __name__ == "__main__":
    main()
    # datasets 스트리밍의 백그라운드 스레드가 남아 프로세스가 끝나지 않는 경우가 있어 강제 종료
    import os
    os._exit(0)
