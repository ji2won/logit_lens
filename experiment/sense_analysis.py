#!/usr/bin/env python3
"""A의 뜻별 영어 probe 분석 (S_sense).

spring은 봄(계절)과 용수철 두 뜻이 같은 토큰이라, 렌즈에 spring이 보여도 어느 뜻이 켜졌는지 모른다.
A_sense_probes.csv의 뜻별 단어 묶음으로 방해 뜻(coil bounce mattress)이 얼마나 켜졌는지 따로 읽는다.

  python -m experiment.sense_analysis --model llama3.2-3b
  → results/<model>/sense_summary.md

지표 (문항×프롬프트마다 구한 뒤 프롬프트 4개 평균, 층 구간 안에서 최댓값)
  S_d       방해 뜻 묶음의 렌즈 로그확률
  S_rel     층마다 (방해 뜻 − 정답 뜻) 로그확률. 방해 뜻이 정답 뜻에 비해 얼마나 켜졌나
  S_d_diff  층마다 (입력어 − 대조어 평균) 방해 뜻 로그확률. 입력과 무관하게 나오는 몫을 뺀다
B의 r(S, I)는 같은 구조(방해 뜻 영어 probe)의 음성 대조다. 영어 경유라면 A가 B보다 커야 한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import KEYS, corr_ci, interference, md_table, mid_layers
from .config import RESULTS_DIR

RANGES = {"mid (25–75%)": None, "late (75–97%)": (0.75, 0.97)}


def sense_scores(lens: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    d = lens[(lens.dataset == "A") & lens.layer.between(lo, hi)].copy()
    d["rel"] = d.lp_sense_d - d.lp_sense_t
    k = KEYS + ["lens"]
    inp = d[d.role == "input"]
    s = inp.groupby(k).agg(S_d=("lp_sense_d", "max"), S_rel=("rel", "max"), S_t=("lp_sense_t", "max"))
    ctl = d[d.role == "control"].groupby(k + ["layer"]).lp_sense_d.mean().rename("ctrl")
    j = inp.set_index(k + ["layer"]).lp_sense_d.to_frame().join(ctl, how="inner")
    s["S_d_diff"] = (j.lp_sense_d - j.ctrl).groupby(k).max()
    return s.reset_index()


def b_scores(lens: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    d = lens[(lens.dataset == "B") & (lens.role == "input") & lens.layer.between(lo, hi)]
    return d.groupby(KEYS + ["lens"]).lp_distractor.max().rename("S_d").reset_index()


def run(out_dir: Path) -> str:
    lens = pd.read_csv(out_dir / "lens.csv.gz", low_memory=False)
    if "lp_sense_d" not in lens:
        raise SystemExit("lens.csv.gz에 lp_sense_d가 없음: A를 A_sense_probes.csv와 함께 다시 돌려야 한다")
    I = interference(pd.read_csv(out_dir / "outputs.csv"))
    Ii = I.groupby(["dataset", "task", "id", "group"], as_index=False).I.mean()
    L = int(lens.layer.max())
    lines = [f"# A 뜻별 영어 probe (S_sense)\n\n층 수 {L}. r은 Spearman, 95% CI는 영어 단어(A)/한국어 입력어(B) 묶음 bootstrap.\n"]
    for rname, rng in RANGES.items():
        lo, hi = mid_layers(L, rng)
        S = sense_scores(lens, lo, hi)
        Si = S.groupby(["dataset", "task", "id", "lens"], as_index=False)[["S_d", "S_rel", "S_t", "S_d_diff"]].mean()
        m = Si.merge(Ii, on=["dataset", "task", "id"])
        B = b_scores(lens, lo, hi).groupby(["dataset", "task", "id", "lens"], as_index=False).S_d.mean().merge(Ii, on=["dataset", "task", "id"])
        rows = []
        for (task, ln), g in m.groupby(["task", "lens"]):
            g = g.sort_values("id")
            for x in ("S_d", "S_rel", "S_d_diff"):
                r, a, b = corr_ci(g[x], g.I, g.group)
                rows.append(dict(task=task, lens=ln, x=x, n=len(g), r=r, ci_lo=a, ci_hi=b))
        for ln, g in B[B.task == "translate"].groupby("lens"):
            r, a, b = corr_ci(g.S_d, g.I, g.group)
            rows.append(dict(task="B translate (음성 대조)", lens=ln, x="S_d", n=len(g), r=r, ci_lo=a, ci_hi=b))
        lines.append(f"\n## {rname}: {lo}–{hi}층\n\n### 상관 r(지표, 간섭 I)\n\n" + md_table(pd.DataFrame(rows)))
        # 렌즈별 평균 확률: 정답 뜻 / 방해 뜻 (보정 후 얼마나 남나)
        p = m[m.task == "translate"].groupby("lens")[["S_t", "S_d"]].agg(lambda s: np.exp(s).mean())
        p.columns = ["정답 뜻 p", "방해 뜻 p"]
        p["방해/정답"] = p["방해 뜻 p"] / p["정답 뜻 p"]
        lines.append("\n### 평균 확률 (translate)\n\n" + md_table(p.reset_index()))
    text = "\n".join(lines) + "\n"
    (out_dir / "sense_summary.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir or RESULTS_DIR / args.model.replace("/", "_"))
    print(run(out_dir))


if __name__ == "__main__":
    main()
