#!/usr/bin/env python3
"""
묶음 C(일반 단어) 후보 검사 → 선별 → 무관 단어 짝짓기 → C_final.csv

  1) 겹침: A·B의 입력어·대조어·영어, few-shot 단어와 겹치면 제외 (OVERLAP_A, OVERLAP_B, FEWSHOT)
  2) 빈도(wordfreq Zipf): 한국어·영어 < 3.0, 중국어 목록 밖이면 제외 (FREQ_*)
  3) CC-CEDICT: 중국어가 사전에 있고 뜻풀이에 그 영어가 있는가 (CEDICT_NOT_FOUND, CEDICT_EN_MISS)
     → 사람 판단 J로 덮어쓴다
  4) 사람 판단: 영어 다의어 EN_POLY, 한국어 동형어 KO_HOMOGRAPH
  5) (기록만) 한자어가 중국어와 같은 한자인가(sino_same), 모델별 ' en' 토큰 수(ntok_en_*)
  6) 남은 문항끼리 무관 단어를 짝짓는다. 문항마다 다른 문항 하나를 무관 단어로 배정하되(순열, 각 단어는
     무관 단어로 한 번만 쓰임) 영어 Zipf 차를 최소화한다. 금지: 같은 범주, 중국어 글자 공유,
     영어 한쪽이 다른 쪽을 포함(sun/sunflower), 영어 앞 3글자 같음(첫 토큰 겹침 방지).

사용법
  pip install "wordfreq[cjk]" pandas scipy transformers
  python make_C_draft.py
  python build_C.py --cedict ../cedict_ts.u8 [--tokenizers]
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment.prompts import FEWSHOT  # noqa: E402

TOKENIZERS = {
    "llama": "meta-llama/Llama-3.2-3B",
    "exaone": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    "qwen": "Qwen/Qwen3-1.7B",
}
MIN_ZIPF = 3.0

# ---------------------------------------------------------------- 사람 판단
# 영어 다의어(중간층에서 영어가 떠도 그 뜻인지 알 수 없음). Y 제외, ? 경계로 제외
EN_POLY = {
    "wound": ("Y", "wind의 과거분사와 표기가 같음"),
    "desert": ("?", "동사 desert(버리다)와 표기가 같음"),
    "age": ("?", "나이 / 시대(the Bronze Age)"),
}
# 한국어 동형어·다른 용법이 흔함
KO_HOMOGRAPH = {
    "도로": ("?", "부사 '도로'(되돌려)가 흔함"),
    "선물": ("?", "先物(선물 거래)이 뉴스에서 흔함"),
    "공주": ("?", "지명 공주(公州)"),
    "보석": ("?", "保釋(보석 석방)이 뉴스에서 흔함"),
}
# CEDICT 판정 덮어쓰기: zh → (OK|BAD, 메모)
J = {
    "乌龟": ("OK", "tortoise로 풀이되나 turtle 통칭으로 씀. 속어 cuckold 있음"),
    "蚯蚓": ("OK", "earthworm. worm의 대표 뜻"),
    "斧头": ("OK", "ax(미국 표기) = axe"),
    "照片": ("OK", "photograph = photo"),
    "宝石": ("OK", "gem / precious stone = jewel"),
    "作家": ("OK", "author = writer"),
    "树叶": ("OK", "tree leaves"),
    "悲伤": ("OK", "형용사 풀이지만 명사로도 씀"),
    "数学": ("OK", "mathematics = math"),
    "愤怒": ("OK", "angry / wrath. 명사로도 씀"),
    "诞生": ("OK", "to be born. 명사로도 씀"),
    "结婚": ("OK", "to marry. 명사로도 씀"),
}


def load_cedict(path: Path):
    """→ ({간체: [뜻풀이...]}, {번체: {간체...}})"""
    simp_idx: dict[str, list[str]] = {}
    trad2simp: dict[str, set[str]] = {}
    for line in path.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        trad, simp, rest = line.split(" ", 2)
        pinyin = rest[rest.index("[") + 1:rest.index("]")]
        gloss = rest[rest.index("/") + 1:].strip("/").replace("/", " / ")
        simp_idx.setdefault(simp, []).append(f"{pinyin}: {gloss}")
        trad2simp.setdefault(trad, set()).add(simp)
    return simp_idx, trad2simp


def en_in_gloss(en: str, glosses: list[str]) -> bool:
    pat = re.compile(rf"\b{re.escape(en.lower())}(s|es)?\b")
    return any(pat.search(g.lower()) for g in glosses)


def zipf_fn(lang):
    import wordfreq
    fd = wordfreq.get_frequency_dict(lang)

    def f(w):
        w = w.strip().lower() if lang == "en" else w.strip()
        return round(math.log10(fd[w] * 1e9), 2) if w in fd else 0.0
    return f


def ab_sets():
    ko, en = set(), set()
    a = pd.read_csv(HERE / "A_final.csv", keep_default_na=False)
    for r in a.itertuples():
        ko |= {r.ko_input, r.ko_reverse_input, *r.ko_control_final.split(";")}
        en.add(r.en.lower())
    b = pd.read_csv(HERE / "B_final.csv", keep_default_na=False)
    for r in b.itertuples():
        ko |= {r.ko_input, *r.ko_control_final.split(";")}
        en |= {r.en_target.lower(), r.en_distractor.lower()}
    ka, kb = set(), set()
    for r in a.itertuples():
        ka |= {r.ko_input, r.ko_reverse_input, *r.ko_control_final.split(";")}
    for r in b.itertuples():
        kb |= {r.ko_input, *r.ko_control_final.split(";")}
    ea = {r.en.lower() for r in a.itertuples()}
    eb = {w.lower() for r in b.itertuples() for w in (r.en_target, r.en_distractor)}
    return ka - {""}, kb - {""}, ea, eb


def pair_unrelated(d: pd.DataFrame, seed_penalty: float = 100.0) -> list[int]:
    from scipy.optimize import linear_sum_assignment
    n = len(d)
    en, zh, cat, z = d.en.str.lower().tolist(), d.zh.tolist(), d.category.tolist(), d.zipf_en.astype(float).tolist()
    cost = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            bad = (i == j or cat[i] == cat[j] or set(zh[i]) & set(zh[j])
                   or en[i] in en[j] or en[j] in en[i] or en[i][:3] == en[j][:3])
            cost[i, j] = 1e6 if bad else abs(z[i] - z[j])
    rows, cols = linear_sum_assignment(cost)
    assert cost[rows, cols].max() < 1e6, "무관 단어 배정 실패"
    return list(cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(HERE / "C_candidates_draft.csv"))
    ap.add_argument("--annotated", default=str(HERE / "C_candidates_annotated.csv"))
    ap.add_argument("--output", default=str(HERE / "C_final.csv"))
    ap.add_argument("--cedict", default=str(HERE.parent / "cedict_ts.u8"))
    ap.add_argument("--tokenizers", action="store_true", help="로컬 HF 캐시의 세 모델 토크나이저로 토큰 수 기록")
    args = ap.parse_args()

    d = pd.read_csv(args.input, keep_default_na=False)
    ka, kb, ea, eb = ab_sets()
    fs_ko = {k for k, _, _ in FEWSHOT}
    fs_zh = {z for _, z, _ in FEWSHOT}
    fs_en = {e for _, _, e in FEWSHOT}
    zko, zen, zzh = zipf_fn("ko"), zipf_fn("en"), zipf_fn("zh")
    simp_idx, trad2simp = load_cedict(Path(args.cedict))

    toks = {}
    if args.tokenizers:
        from transformers import AutoTokenizer
        for short, name in TOKENIZERS.items():
            try:
                toks[short] = AutoTokenizer.from_pretrained(name, local_files_only=True, trust_remote_code=True)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] {name} 토크나이저 없음: {e}", file=sys.stderr)

    out = []
    for r in d.to_dict("records"):
        fl = []
        if r["ko"] in ka or r["en"].lower() in ea:
            fl.append("OVERLAP_A")
        if r["ko"] in kb or r["en"].lower() in eb:
            fl.append("OVERLAP_B")
        if r["ko"] in fs_ko or r["zh"] in fs_zh or r["en"] in fs_en:
            fl.append("FEWSHOT")
        if " " in r["en"].strip():
            fl.append("EN_MULTIWORD")

        r["zipf_ko"], r["zipf_en"], r["zipf_zh"] = zko(r["ko"]), zen(r["en"]), zzh(r["zh"])
        if r["zipf_ko"] < MIN_ZIPF:
            fl.append("FREQ_KO_LOW")
        if r["zipf_en"] < MIN_ZIPF:
            fl.append("FREQ_EN_LOW")
        if r["zipf_zh"] == 0:
            fl.append("FREQ_ZH_ZERO")

        g = simp_idx.get(r["zh"], [])
        r["cedict_gloss"] = " || ".join(g)[:200]
        st = "OK" if g and en_in_gloss(r["en"], g) else ("EN_MISS" if g else "NOT_FOUND")
        st, note = J.get(r["zh"], (st, ""))
        r["cedict_ok"], r["cedict_note"] = st, note
        if st != "OK":
            fl.append(f"CEDICT_{st}")

        if r["en"] in EN_POLY:
            v, why = EN_POLY[r["en"]]
            r["en_poly"] = f"{v}:{why}"
            if v in ("Y", "?"):
                fl.append("EN_POLY" if v == "Y" else "EN_POLY_BORDERLINE")
        if r["ko"] in KO_HOMOGRAPH:
            v, why = KO_HOMOGRAPH[r["ko"]]
            r["ko_homograph"] = f"{v}:{why}"
            if v in ("Y", "?"):
                fl.append("KO_HOMOGRAPH" if v == "Y" else "KO_HOMOGRAPH_BORDERLINE")

        # 한자어의 한자가 중국어 정답과 같은가 (분석 공변량. 제외하지 않음)
        h = r["hanja"]
        r["sino_same"] = "Y" if h and (r["zh"] == h or r["zh"] in trad2simp.get(h, ())) else "N"
        r["ko_len"] = len(r["ko"])
        for short, tok in toks.items():
            r[f"ntok_en_{short}"] = len(tok.encode(" " + r["en"], add_special_tokens=False))
        r["auto_flags"] = ";".join(fl)
        out.append(r)

    ann = pd.DataFrame(out)
    ann["keep"] = np.where(ann.auto_flags == "", "Y", "N")
    keep = ann[ann.keep == "Y"].reset_index(drop=True)

    cols = pair_unrelated(keep)
    keep["ko_unrelated"] = keep.ko.values[cols]
    keep["en_unrelated"] = keep.en.values[cols]
    keep["zh_unrelated"] = keep.zh.values[cols]
    keep["zipf_en_unrelated"] = keep.zipf_en.values[cols]
    keep["category_unrelated"] = keep.category.values[cols]
    ann = ann.merge(keep[["id", "ko_unrelated", "en_unrelated", "zh_unrelated"]], on="id", how="left")
    ann.to_csv(args.annotated, index=False, encoding="utf-8-sig")

    keep["id"] = [f"C{i:03d}" for i in range(1, len(keep) + 1)]
    final_cols = ["id", "ko", "zh", "en", "en_unrelated", "zh_unrelated", "ko_unrelated", "etym", "hanja",
                  "category", "category_unrelated", "sino_same", "zipf_ko", "zipf_en", "zipf_zh",
                  "zipf_en_unrelated"] + [c for c in keep.columns if c.startswith("ntok_en_")]
    keep[final_cols].to_csv(args.output, index=False, encoding="utf-8-sig")

    c = Counter(f for fl in ann.auto_flags for f in fl.split(";") if f)
    gap = (keep.zipf_en - keep.zipf_en_unrelated).abs()
    print(f"[done] 후보 {len(ann)} → 최종 {len(keep)} → {args.output}")
    print(f"       제외 사유: {dict(c)}")
    print(f"       어종: {dict(Counter(keep.etym))}  범주: {dict(Counter(keep.category))}")
    print(f"       무관 단어 영어 Zipf 차: 평균 {gap.mean():.3f}, 최대 {gap.max():.2f}")
    for short in toks:
        n1 = (keep[f"ntok_en_{short}"] == 1).sum()
        print(f"       ' en' 1토큰 ({short}): {n1}/{len(keep)}")


if __name__ == "__main__":
    main()
