#!/usr/bin/env python3
"""ablate.py 결과 요약.

  python -m experiment.ablate_analysis --model llama3.2-3b
  → results/<model>/ablate/summary.md

Δ = (개입 후 log p) − (깨끗한 log p). 음수일수록 출력이 무너진 것. 프롬프트 4개 평균 후 문항 단위로 본다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import N_BOOT, RESULTS_DIR, ROOT, SEED

COND_ORDER = ["en_item", "en_other", "en_generic", "random", "ko_input", "zh_answer"]
COND_KO = {"en_item": "이 문항의 영어", "en_other": "다른 문항의 영어", "en_generic": "공통 영어 조각",
           "random": "무작위 방향", "ko_input": "입력 한국어 토큰", "zh_answer": "중국어 정답 토큰"}


def boot_ci(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_r(x, y, n=N_BOOT, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        rs.append(spearmanr(x[i], y[i])[0])
    return spearmanr(x, y)[0], np.nanpercentile(rs, 2.5), np.nanpercentile(rs, 97.5)


def fmt(m, lo, hi, p=3):
    return f"{m:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    args = ap.parse_args()
    d = RESULTS_DIR / args.model / "ablate"
    raw = pd.read_csv(d / "ablate_raw.csv")
    dirs = pd.read_csv(d / "directions.csv")

    clean = raw[raw.cond == "clean"].set_index(["id", "version"])
    x = raw[raw.cond != "clean"].copy()
    for c in ["lp_answer", "lp_answer_first", "lp_distractor", "lp_distractor_first"]:
        x["d_" + c.replace("lp_", "")] = x[c].values - clean.loc[list(zip(x.id, x.version)), c].values
    x["d_gap"] = x.d_distractor - x.d_answer           # 방해어가 정답 대비 상대적으로 얼마나 변했나
    x["top1_changed"] = x.top1_changed.astype(float)
    it = x.groupby(["id", "band", "cond"], as_index=False)[
        ["d_answer", "d_answer_first", "d_distractor", "d_gap", "top1_changed", "removed_frac", "k"]].mean()

    md = [f"# 인과 개입: 영어 방향 제거 ({args.model})\n",
          "A 번역 과제, 입력어 프롬프트. 구간(mid 25–75%, late 75–97%)의 모든 층 출력에서, 프롬프트 위치마다 "
          "방향 묶음에 투영한 성분을 지웠다(h ← h − Ph). Δ = 개입 후 − 깨끗한 log p (nat). "
          "프롬프트 4개 평균 후 문항 단위 평균, 95% bootstrap CI.\n",
          f"깨끗한 상태: log p(정답) 평균 {clean.lp_answer.mean():.3f}, 첫 토큰 {clean.lp_answer_first.mean():.3f}, "
          f"log p(방해어) {clean.lp_distractor.mean():.3f}\n"]

    rows = []
    for b in it.band.unique():
        for c in COND_ORDER:
            g = it[(it.band == b) & (it.cond == c)]
            if g.empty:
                continue
            rows.append(dict(band=b, 조건=f"{c} ({COND_KO[c]})", k=round(g.k.mean(), 1),
                             지운비율=g.removed_frac.mean(),
                             Δ정답=fmt(*boot_ci(g.d_answer)), Δ정답첫토큰=fmt(*boot_ci(g.d_answer_first)),
                             Δ방해어=fmt(*boot_ci(g.d_distractor)), top1바뀜=g.top1_changed.mean()))
    md += ["## 1. 조건별 효과\n",
           "'지운비율' = 프롬프트 마지막 위치에서 ‖Ph‖/‖h‖ (구간 평균). top1바뀜 = 최종 출력 1위 토큰이 바뀐 비율.\n",
           md_table(pd.DataFrame(rows)), ""]

    # 짝 비교
    rows = []
    w = it.pivot_table(index=["id", "band"], columns="cond", values="d_answer").reset_index()
    for b in w.band.unique():
        g = w[w.band == b]
        for a, c in [("en_item", "en_other"), ("en_item", "random"), ("en_item", "en_generic"),
                     ("zh_answer", "en_item")]:
            if a in g and c in g:
                rows.append(dict(band=b, 비교=f"{a} − {c}", ΔΔ정답=fmt(*boot_ci(g[a] - g[c]))))
    md += ["## 2. 짝 비교 (같은 문항 안에서 Δ정답의 차)\n",
           "en_item − en_other가 뚜렷이 음수여야 '이 문항의 영어 개념'이 출력에 쓰인다고 볼 수 있다. "
           "en_other도 영어 토큰이고 묶음 크기를 맞췄으므로 '영어라서'의 효과는 빠진다.\n",
           md_table(pd.DataFrame(rows)), ""]

    # 부풀림: decodability vs causal effect
    dec = dirs.groupby(["id", "band"], as_index=False)[["decod_en_item", "decod_probe"]].mean()
    m = w.merge(dec, on=["id", "band"])
    m["causal"] = -(m.en_item - m.en_other)            # 클수록 영어 개념 제거가 더 해롭다
    a_en = Path(ROOT / "A_en.csv")
    keep = set(pd.read_csv(a_en).id) if a_en.exists() else set()
    rows = []
    for b in m.band.unique():
        g = m[m.band == b]
        r = boot_r(g.decod_en_item, g.causal)
        row = dict(band=b, n=len(g), 렌즈영어확률평균=g.decod_en_item.mean(),
                   r_읽힘_vs_쓰임=fmt(*r, p=2))
        if keep:
            ge, gr = g[g.id.isin(keep)], g[~g.id.isin(keep)]
            row.update(A_en_causal=fmt(*boot_ci(ge.causal)), 나머지_causal=fmt(*boot_ci(gr.causal)))
        rows.append(row)
    md += ["## 3. 읽히는 영어 vs 쓰이는 영어 (부풀림)\n",
           "읽힘 = 깨끗한 순전파에서 en_item 토큰 확률 합의 구간 최댓값(바닐라 렌즈). "
           "쓰임(causal) = −(Δ정답[en_item] − Δ정답[en_other]). 렌즈 영어가 진짜 경유라면 읽힘이 큰 문항일수록 "
           "쓰임도 커야 한다(r > 0). A_en = 렌즈에 probe 영어가 뜨는 44문항.\n",
           md_table(pd.DataFrame(rows)), ""]

    # 간섭 경로
    rows = []
    for b in it.band.unique():
        for c in ["en_item", "en_other", "random"]:
            g = it[(it.band == b) & (it.cond == c)]
            rows.append(dict(band=b, 조건=c, Δ방해어_minus_Δ정답=fmt(*boot_ci(g.d_gap))))
    md += ["## 4. 방해어(弹簧) 쪽 변화\n",
           "영어 다의어를 거쳐 방해어로 새는 경로가 있다면, 영어를 지울 때 방해어가 정답보다 더 많이 떨어져야 한다(음수).\n",
           md_table(pd.DataFrame(rows)), ""]

    (d / "summary.md").write_text("\n".join(md), "utf-8")
    m.to_csv(d / "item_effects.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
