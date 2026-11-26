"""A_en: 로짓 렌즈(바닐라)로 읽었을 때 중간층에 영어 probe 단어가 실제로 보이는 A 문항만 고른다.

기준 (기본값, 인자로 바꿀 수 있음)
  - 과제 translate, 입력어 프롬프트, 바닐라 렌즈, 중간층(config.MID_LAYER_RANGE → Llama 7–21층)
  - 프롬프트 버전마다 probe 영어 단어(A_final.en, 예: spring)의 중간층 최대 확률을 구하고
  - 그 값이 --p-min(0.1) 이상인 버전이 --min-versions(3)개 이상인 문항을 남긴다.

출력: A_en.csv = A_final.csv의 해당 행 + 선별 근거 열
  sel_model, sel_pmax_mean(버전 평균 최대 확률), sel_n_versions_p(기준 넘은 버전 수),
  sel_n_versions_top1(중간층 어느 층에서든 top-1이 probe 단어였던 버전 수)

사용: python make_A_en.py --model llama3.2-3b
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiment.config import MID_LAYER_RANGE, RESULTS_DIR, ROOT


def select(lens_file: Path, p_min: float, min_versions: int) -> pd.DataFrame:
    L = pd.read_csv(lens_file, usecols=["dataset", "id", "group", "task", "version", "role",
                                         "lens", "layer", "top1", "lp_poly"])
    n_layers = int(L.layer.max())
    lo, hi = math.ceil(MID_LAYER_RANGE[0] * n_layers), math.floor(MID_LAYER_RANGE[1] * n_layers)
    d = L[(L.dataset == "A") & (L.task == "translate") & (L.role == "input")
          & (L.lens == "vanilla") & L.layer.between(lo, hi)].copy()
    d["top_is_probe"] = d.top1.astype(str).str.strip().str.lower() == d.group.astype(str).str.lower()
    v = d.groupby(["id", "version"]).agg(pmax=("lp_poly", lambda s: float(np.exp(s.max()))),
                                         top=("top_is_probe", "any")).reset_index()
    it = v.groupby("id").agg(sel_pmax_mean=("pmax", "mean"),
                             sel_n_versions_p=("pmax", lambda s: int((s >= p_min).sum())),
                             sel_n_versions_top1=("top", "sum")).reset_index()
    it["keep_en"] = it.sel_n_versions_p >= min_versions
    print(f"중간층 {lo}–{hi}층, 기준 p ≥ {p_min} 버전 ≥ {min_versions}: {it.keep_en.sum()}/{len(it)}문항")
    return it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--p-min", type=float, default=0.1)
    ap.add_argument("--min-versions", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "A_en.csv"))
    args = ap.parse_args()

    it = select(RESULTS_DIR / args.model / "lens.csv.gz", args.p_min, args.min_versions)
    a = pd.read_csv(ROOT / "A_final.csv", keep_default_na=False)
    out = a.merge(it, on="id", how="inner")
    out = out[out.keep_en].drop(columns="keep_en")
    out.insert(out.columns.get_loc("sel_pmax_mean"), "sel_model", args.model)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"→ {args.out} ({len(out)}행, 영어 단어 {out.en.nunique()}개)")


if __name__ == "__main__":
    main()
