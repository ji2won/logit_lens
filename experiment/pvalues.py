"""상관마다 p값을 붙인다.

python -m experiment.pvalues  → results/pvalues.csv

  p_spearman  Spearman 순위상관의 t 근사 p (scipy). 문항이 서로 독립이라고 가정한다
  p_boot      묶음 bootstrap(analyze.py의 CI와 같은 묶음: A는 영어 단어, B는 한국어 입력어)으로 구한 양측 p
              = 2 × min(P(r* ≤ 0), P(r* ≥ 0)). 하한은 1/n_boot
  차이(r_A − r_B, Δr)는 p_boot만 낸다. r_A − r_B는 A와 B를 따로 재표집, Δr은 같은 표본에서 두 렌즈를 비교
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .analyze import _boot_index, _spearman, mid_layers
from .config import DATASETS, RESULTS_DIR

MODELS = ["qwen3-1.7b", "exaone3.5-2.4b", "llama3.2-3b"]
RANGES = {"mid": "", "late": "late"}


def p_from_boot(bs) -> float:
    bs = np.asarray(bs, float)
    bs = bs[~np.isnan(bs)]
    return float(max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / len(bs)))


def boot_idx(groups, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    return list(_boot_index(np.asarray(groups), rng, n_boot))


def corr_row(x, y, idx):
    x, y = np.asarray(x, float), np.asarray(y, float)
    bs = [_spearman(x[i], y[i]) for i in idx]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(n=len(x), r=_spearman(x, y), ci_lo=lo, ci_hi=hi,
                p_spearman=float(spearmanr(x, y).pvalue), p_boot=p_from_boot(bs)), bs


def run(n_boot: int) -> pd.DataFrame:
    dom = pd.read_csv(DATASETS["B"], keep_default_na=False)[["id", "nikl_dom_target", "nikl_dom_ok"]]
    dom["nikl_dom_target"] = pd.to_numeric(dom.nikl_dom_target, errors="coerce")
    rows = []
    for m in MODELS:
        for rname, sub in RANGES.items():
            it = pd.read_csv(RESULTS_DIR / m / sub / "item_scores.csv").dropna(subset=["I"])
            boots = {}   # (dataset, task, lens, x) → 부트스트랩 r 목록
            for (ds, task), g0 in it.groupby(["dataset", "task"]):
                g0 = g0.sort_values(["lens", "id"])
                ids = g0[g0.lens == g0.lens.iloc[0]].sort_values("id")
                idx = boot_idx(ids.group.values, n_boot)
                for ln, g in g0.groupby("lens"):
                    g = g.sort_values("id")
                    for x in ("S", "S_diff"):
                        res, bs = corr_row(g[x], g.I, idx)
                        boots[(ds, task, ln, x)] = np.asarray(bs)
                        rows.append(dict(model=m, range=rname, analysis="상관", dataset=ds, task=task,
                                         lens=ln, x=x, **res))
                # 판정: 같은 표본에서 Δr = r(보정) − r(바닐라)
                if ds == "A" and task == "translate":
                    for ln in ("mean", "affine"):
                        for x in ("S", "S_diff"):
                            d = boots[(ds, task, ln, x)] - boots[(ds, task, "vanilla", x)]
                            a = it[(it.dataset == ds) & (it.task == task)]
                            r_new = _spearman(a[a.lens == ln].sort_values("id")[x], a[a.lens == ln].sort_values("id").I)
                            r_ref = _spearman(a[a.lens == "vanilla"].sort_values("id")[x], a[a.lens == "vanilla"].sort_values("id").I)
                            lo, hi = np.nanpercentile(d, [2.5, 97.5])
                            rows.append(dict(model=m, range=rname, analysis="Δr (보정 − 바닐라)", dataset=ds,
                                             task=task, lens=ln, x=x, r=r_new - r_ref, ci_lo=lo, ci_hi=hi,
                                             p_boot=p_from_boot(d)))
            # 음성 대조: r_A − r_B (A와 B 독립 재표집)
            for btask in ("translate", "translate_ctx"):
                for ln in ("vanilla", "mean", "affine"):
                    for x in ("S", "S_diff"):
                        ka, kb = ("A", "translate", ln, x), ("B", btask, ln, x)
                        if ka not in boots or kb not in boots:
                            continue
                        ra = [r for r in rows if r.get("analysis") == "상관" and r["model"] == m and r["range"] == rname
                              and (r["dataset"], r["task"], r["lens"], r["x"]) == ka][0]["r"]
                        rb = [r for r in rows if r.get("analysis") == "상관" and r["model"] == m and r["range"] == rname
                              and (r["dataset"], r["task"], r["lens"], r["x"]) == kb][0]["r"]
                        d = boots[ka] - boots[kb]
                        lo, hi = np.nanpercentile(d, [2.5, 97.5])
                        rows.append(dict(model=m, range=rname, analysis="r_A − r_B", dataset="A−B", task=btask,
                                         lens=ln, x=x, r=ra - rb, ci_lo=lo, ci_hi=hi, p_boot=p_from_boot(d)))
            # B 분해: 같은 프롬프트의 정답 뜻 영어(짝 행의 S)와 간섭
            for btask in ("translate", "translate_ctx"):
                b = it[(it.dataset == "B") & (it.task == btask)].merge(dom, on="id").sort_values(["lens", "id"])
                b["S_ans"] = b.groupby(["lens", "group"]).S.transform(lambda s: s.iloc[::-1].values)
                idx = boot_idx(b[b.lens == "vanilla"].group.values, n_boot)
                for ln, g in b.groupby("lens"):
                    res, _ = corr_row(g.S_ans, g.I, idx)
                    rows.append(dict(model=m, range=rname, analysis="B 정답 뜻 영어 × I", dataset="B",
                                     task=btask, lens=ln, x="S_ans", **res))
                # 뜻 빈도: 방해 뜻 우세도 × I (렌즈와 무관하므로 한 번만)
                g = b[(b.lens == "vanilla") & b.nikl_dom_ok.astype(str).eq("True")].dropna(subset=["nikl_dom_target"])
                if len(g) and rname == "mid":
                    res, _ = corr_row(1 - g.nikl_dom_target, g.I, boot_idx(g.group.values, n_boot))
                    rows.append(dict(model=m, range=rname, analysis="B 방해 뜻 우세도 × I", dataset="B",
                                     task=btask, lens="-", x="dominance", **res))
            print(f"[{m} {rname}] done", flush=True)
    # 뜻별 probe (lens.csv.gz에 lp_sense_d가 있는 모델만)
    from .analyze import interference
    from .sense_analysis import RANGES as SR, sense_scores
    for m in MODELS:
        lens = pd.read_csv(RESULTS_DIR / m / "lens.csv.gz", low_memory=False)
        if "lp_sense_d" not in lens or lens.lp_sense_d.isna().all():
            continue
        I = interference(pd.read_csv(RESULTS_DIR / m / "outputs.csv"))
        Ii = I.groupby(["dataset", "task", "id", "group"], as_index=False).I.mean()
        L = int(lens.layer.max())
        for rname, rng in SR.items():
            lo, hi = mid_layers(L, rng)
            S = sense_scores(lens, lo, hi).groupby(["dataset", "task", "id", "lens"], as_index=False)[
                ["S_d", "S_rel", "S_d_diff"]].mean().merge(Ii, on=["dataset", "task", "id"])
            for (task, ln), g in S.groupby(["task", "lens"]):
                g = g.sort_values("id")
                idx = boot_idx(g.group.values, n_boot)
                for x in ("S_d", "S_rel", "S_d_diff"):
                    res, _ = corr_row(g[x], g.I, idx)
                    rows.append(dict(model=m, range=rname, analysis="A 뜻별 probe", dataset="A", task=task,
                                     lens=ln, x=x, **res))
        print(f"[{m} sense] done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    df = run(args.n_boot)
    out = RESULTS_DIR / "pvalues.csv"
    df.to_csv(out, index=False, float_format="%.6g")
    print(f"→ {out} ({len(df)}행)")


if __name__ == "__main__":
    main()
