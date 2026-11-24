#!/usr/bin/env python3
"""실제 생성에서 맞게 번역한 문항만 남겨 렌즈 점수 × 간섭 상관을 다시 본다.

  python -m experiment.correct_only --models qwen3-1.7b exaone3.5-2.4b llama3.2-3b
  → results/correct_only.md

입력: results/<model>/item_scores.csv (analyze.py), results/<model>/generations.csv (generate.py)
맞음 판정(프롬프트 하나): A = 생성 답이 정답과 정확히 일치, B = 두 뜻(정답·방해어) 중 하나와 일치
문항을 남기는 기준: 프롬프트 4개 중 맞은 수 ≥ k (k = 1, 3, 4). 기본 결과는 k=3.
상관·CI는 analyze.py와 같은 방식(문항 단위 Spearman, 묶음 bootstrap).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .analyze import _boot_index, _spearman, ci, corr_ci, md_table
from .config import N_BOOT, RESULTS_DIR, SEED

THRESHOLDS = {"전체": 0, "≥1/4 맞음": 1, "≥3/4 맞음": 3, "4/4 맞음": 4}


def n_correct(gen: pd.DataFrame) -> pd.DataFrame:
    g = gen[gen.task == "translate"].copy()
    g["ok"] = np.where(g.dataset == "A", g.exact, g.exact | g.distractor)
    return g.groupby(["dataset", "id"], as_index=False).ok.sum().rename(columns={"ok": "n_ok"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3-1.7b", "exaone3.5-2.4b", "llama3.2-3b"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    md = ["# 맞게 번역한 문항만으로 본 상관", "",
          "맞음: A는 생성 답 = 정답(정확 일치), B는 두 뜻 중 하나. 문항은 프롬프트 4개 중 맞은 수로 거른다.",
          "상관은 문항 단위 Spearman, 95% CI는 묶음 bootstrap(analyze.py와 같음).", ""]
    for m in args.models:
        d = RESULTS_DIR / m
        gen = pd.read_csv(d / "generations.csv", keep_default_na=False)
        for c in ("exact", "distractor"):
            gen[c] = gen[c].astype(str) == "True"
        items = pd.read_csv(d / "item_scores.csv").dropna(subset=["I"])
        items = items[items.task == "translate"].merge(n_correct(gen), on=["dataset", "id"], how="left")

        cor, neg = [], []
        for name, k in THRESHOLDS.items():
            sub = items[items.n_ok >= k]
            for (ds, ln), g in sub.groupby(["dataset", "lens"]):
                for x in ("S", "S_diff"):
                    r, lo, hi = corr_ci(g[x], g.I, g.group, args.n_boot)
                    cor.append({"dataset": ds, "lens": ln, "x": x, "문항": name, "n": len(g), "r [CI]": ci(r, lo, hi)})
            for ln in sorted(set(sub.lens)):
                ra = sub[(sub.dataset == "A") & (sub.lens == ln)]
                rb = sub[(sub.dataset == "B") & (sub.lens == ln)]
                if len(ra) < 3 or len(rb) < 3:
                    continue
                rng = np.random.default_rng(SEED)
                da = [_spearman(ra.S.values[i], ra.I.values[i]) for i in _boot_index(ra.group.values, rng, args.n_boot)]
                db = [_spearman(rb.S.values[i], rb.I.values[i]) for i in _boot_index(rb.group.values, rng, args.n_boot)]
                diff = np.array(da) - np.array(db)
                rA, rB = _spearman(ra.S, ra.I), _spearman(rb.S, rb.I)
                neg.append({"문항": name, "lens": ln, "n_A": len(ra), "n_B": len(rb), "r_A": rA, "r_B": rB,
                            "r_A - r_B [CI]": ci(rA - rB, *np.nanpercentile(diff, [2.5, 97.5]))})

        cor = pd.DataFrame(cor)
        wide = cor.pivot_table(index=["dataset", "lens", "x"], columns="문항", values="r [CI]", aggfunc="first")
        wide = wide[[c for c in THRESHOLDS if c in wide.columns]].reset_index()
        ns = cor.drop_duplicates(["dataset", "문항"]).pivot(index="dataset", columns="문항", values="n")
        ns = ns[[c for c in THRESHOLDS if c in ns.columns]]
        md += [f"## {m}", "", "문항 수: " + "; ".join(
            f"{ds} " + ", ".join(f"{c} {int(ns.loc[ds, c])}" for c in ns.columns) for ds in ns.index), "",
               "### 렌즈 점수 × 간섭 (translate)", "", md_table(wide), "",
               "### 음성 대조 (S 기준)", "", md_table(pd.DataFrame(neg)), ""]
        print(f"[{m}] done")
    out = RESULTS_DIR / "correct_only.md"
    out.write_text("\n".join(md), "utf-8")
    print(f"→ {out}")


if __name__ == "__main__":
    main()
