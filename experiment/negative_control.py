#!/usr/bin/env python3
"""B 음성 대조에서 렌즈 점수 × 간섭 상관이 큰 이유를 쪼개 본다.

  python -m experiment.negative_control --models qwen3-1.7b exaone3.5-2.4b llama3.2-3b
  → results/negative_control.md

B는 한 단어(배)가 두 행으로 들어 있다: (정답 船, 방해어 梨, probe pear) / (정답 梨, 방해어 船, probe ship).
두 행의 입력어 프롬프트는 같으므로, 짝 행의 S는 "같은 프롬프트에서 정답 뜻 영어의 렌즈 로그확률"이다.

  1. 짝 안 / 단어 하나당 한 행: 상관이 같은 프롬프트의 두 뜻 비교에서만 나오는지, 단어 간에도 나오는지
     (한 행 = 국립국어원 기준 우세 뜻이 정답인 행 / 소수 뜻이 정답인 행)
  2. 정답 뜻 영어(짝 행의 S)와 간섭의 상관
     - 렌즈가 '헷갈리는 정도'를 잰다면 두 뜻의 영어 모두 간섭과 양(+)의 상관
     - 렌즈가 '모델이 기운 뜻'을 읽는다면 방해 뜻 영어는 양(+), 정답 뜻 영어는 음(-)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .analyze import ci, corr_ci, md_table
from .config import DATASETS, N_BOOT, RESULTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3-1.7b", "exaone3.5-2.4b", "llama3.2-3b"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    nb = args.n_boot
    dom = pd.read_csv(DATASETS["B"], keep_default_na=False)[["id", "nikl_dom_target"]]
    dom["nikl_dom_target"] = dom.nikl_dom_target.astype(float)

    md = ["# B 음성 대조 분해", "", __doc__.split("\n\n", 1)[1].strip().replace("\n", "  \n"), ""]
    for m in args.models:
        it = pd.read_csv(RESULTS_DIR / m / "item_scores.csv").dropna(subset=["I"])
        a = it[(it.dataset == "A") & (it.task == "translate")]
        b = it[(it.dataset == "B") & (it.task == "translate")].merge(dom, on="id").sort_values(["lens", "id"])
        assert (b.groupby(["lens", "group"]).size() == 2).all(), "B는 단어당 두 행이어야 한다"
        b["S_ans"] = b.groupby(["lens", "group"]).S.transform(lambda s: s.iloc[::-1].values)

        r1, r2 = [], []
        for ln in sorted(set(b.lens)):
            ga, gb = a[a.lens == ln], b[b.lens == ln]
            hi, lo = gb[gb.nikl_dom_target >= 0.5], gb[gb.nikl_dom_target < 0.5]
            d = gb[["S", "S_diff", "I"]] - gb.groupby("group")[["S", "S_diff", "I"]].transform("mean")
            for x in ("S", "S_diff"):
                r1.append({"lens": ln, "x": x, "A": ci(*corr_ci(ga[x], ga.I, ga.group, nb)),
                           "B 전체(60)": ci(*corr_ci(gb[x], gb.I, gb.group, nb)),
                           f"B 우세 뜻 정답({len(hi)})": ci(*corr_ci(hi[x], hi.I, hi.group, nb)),
                           f"B 소수 뜻 정답({len(lo)})": ci(*corr_ci(lo[x], lo.I, lo.group, nb)),
                           "B 짝 안(두 뜻 차이) 부호 일치": float((np.sign(d[x]) == np.sign(d.I)).mean())})
            r2.append({"lens": ln, "r(방해 뜻 영어, I)": ci(*corr_ci(gb.S, gb.I, gb.group, nb)),
                       "r(정답 뜻 영어, I)": ci(*corr_ci(gb.S_ans, gb.I, gb.group, nb)),
                       "r(방해 − 정답, I)": ci(*corr_ci(gb.S - gb.S_ans, gb.I, gb.group, nb))})
        md += [f"## {m}", "", "### 1. A와 B, B를 단어당 한 행으로", "", md_table(pd.DataFrame(r1)), "",
               "### 2. 같은 프롬프트의 정답 뜻 영어 (S 기준)", "", md_table(pd.DataFrame(r2)), ""]
        print(f"[{m}] done")
    out = RESULTS_DIR / "negative_control.md"
    out.write_text("\n".join(md), "utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
