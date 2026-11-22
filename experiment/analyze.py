#!/usr/bin/env python3
"""run_items.py 결과로 간섭·렌즈 점수·신뢰도·상관·판정을 계산한다.

  python -m experiment.analyze --model llama3.2-3b
  → results/<model>/summary.md, metrics.json, item_scores.csv, 그림(png)

정의
  간섭 I        = log p(방해어 | 입력어 프롬프트) - 평균_c log p(방해어 | 대조어 c 프롬프트)   (최종 출력, 렌즈 무관)
  렌즈 점수 S   = 중간층(25–75%)에서 probe 영어 단어 렌즈 로그확률의 최댓값 (입력어 프롬프트)
                  A: 영어 다의어(spring), B: 방해어 뜻의 영어(pear), C: 정답 영어(sea)
  S_diff        = 층마다 (입력어 - 대조어 평균) 렌즈 로그확률을 구한 뒤 중간층 최댓값 (보조 지표)
  신뢰도        = 프롬프트 4개를 문항 측정의 반복으로 보고 Cronbach's alpha (기준 0.7)
  상관          = 문항별로 4개 프롬프트 평균을 낸 뒤 Spearman. 95% CI는 묶음(group) 단위 bootstrap
  판정(A 번역)  = 보정 렌즈마다 영어 감소율(1 - 평균 확률 비)과 상관 변화(Δr, 같은 bootstrap 표본)로
                  | 크게 줄고 상관 그대로 → 가설 지지 | 거의 안 줆 → 가설에 불리 | 크게 줄고 상관 하락 → 렌즈 신뢰 |
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import BIG_REDUCTION, MID_LAYER_RANGE, N_BOOT, RELIABILITY_MIN, RESULTS_DIR, SEED

PRIMARY = {"A": "lp_poly", "B": "lp_distractor", "C": "lp_target"}
KEYS = ["dataset", "id", "task", "version"]


# ---------------------------------------------------------------- 점수 계산
def mid_layers(n_layers: int) -> tuple[int, int]:
    lo, hi = MID_LAYER_RANGE
    return math.ceil(lo * n_layers), math.floor(hi * n_layers)


def interference(out: pd.DataFrame) -> pd.DataFrame:
    o = out[out.logp_distractor.notna()]
    inp = o[o.role == "input"].set_index(KEYS)
    ctl = o[o.role == "control"].groupby(KEYS).logp_distractor.mean()
    df = inp[["group", "etym", "logp_answer", "logp_distractor"]].join(ctl.rename("logp_distractor_ctrl"), how="inner")
    df["I"] = df.logp_distractor - df.logp_distractor_ctrl
    return df.reset_index()


def lens_scores(lens: pd.DataFrame) -> pd.DataFrame:
    L = int(lens.layer.max())
    lo, hi = mid_layers(L)
    rows = []
    for ds, col in PRIMARY.items():
        d = lens[(lens.dataset == ds) & lens.layer.between(lo, hi)]
        if d.empty or col not in d:
            continue
        k = KEYS + ["lens"]
        inp = d[d.role == "input"]
        s = inp.groupby(k)[col].max().rename("S")
        ctl = d[d.role == "control"].groupby(k + ["layer"])[col].mean().rename("ctrl")
        diff = inp.set_index(k + ["layer"])[col].to_frame().join(ctl, how="inner")
        sd = (diff[col] - diff.ctrl).groupby(k).max().rename("S_diff")
        rows.append(pd.concat([s, sd], axis=1).reset_index())
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 통계
def cronbach_alpha(wide: pd.DataFrame) -> float:
    """행 = 문항, 열 = 프롬프트 버전"""
    w = wide.dropna()
    k = w.shape[1]
    if k < 2 or len(w) < 3:
        return float("nan")
    return k / (k - 1) * (1 - w.var(ddof=1).sum() / w.sum(axis=1).var(ddof=1))


def _spearman(x, y):
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _boot_index(groups: np.ndarray, rng, n_boot: int):
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[i] for i in pick])


def corr_ci(x, y, groups, n_boot=N_BOOT, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    bs = [_spearman(x[i], y[i]) for i in _boot_index(np.asarray(groups), rng, n_boot)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return _spearman(x, y), lo, hi


def paired_delta(x_ref, x_new, y, p_ref, p_new, groups, n_boot=N_BOOT, seed=SEED):
    """같은 bootstrap 표본으로 Δr = r(new) - r(ref)와 감소율 = 1 - mean(p_new)/mean(p_ref)"""
    x_ref, x_new, y = (np.asarray(a, float) for a in (x_ref, x_new, y))
    p_ref, p_new = np.asarray(p_ref, float), np.asarray(p_new, float)
    rng = np.random.default_rng(seed)
    dr, red = [], []
    for i in _boot_index(np.asarray(groups), rng, n_boot):
        dr.append(_spearman(x_new[i], y[i]) - _spearman(x_ref[i], y[i]))
        red.append(1 - p_new[i].mean() / p_ref[i].mean())
    point_dr = _spearman(x_new, y) - _spearman(x_ref, y)
    point_red = 1 - p_new.mean() / p_ref.mean()
    return (point_dr, *np.nanpercentile(dr, [2.5, 97.5])), (point_red, *np.nanpercentile(red, [2.5, 97.5]))


def verdict(red: float, dr_hi: float) -> str:
    if red < BIG_REDUCTION:
        return "거의 안 줄었음 → 그 원인은 영어를 만들지 않았다 (가설에 불리)"
    if dr_hi < 0:
        return "크게 줄고 상관도 떨어짐 → 지운 영어가 진짜 신호였다 (로짓 렌즈를 믿을 만함)"
    return "크게 줄고 상관은 그대로 → 지운 영어는 부풀려진 영어였다 (가설 지지)"


# ---------------------------------------------------------------- 표 출력
def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def ci(p, lo, hi):
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------- 메인 분석
def analyze(out_dir: Path, n_boot=N_BOOT, plots=True) -> dict:
    out = pd.read_csv(out_dir / "outputs.csv")
    lens = pd.read_csv(out_dir / "lens.csv.gz")
    n_layers = int(lens.layer.max())
    lo, hi = mid_layers(n_layers)
    I = interference(out)
    S = lens_scores(lens)
    md, metrics = [f"# 결과 요약 ({out_dir.name})\n",
                   f"층 수 {n_layers}, 렌즈 점수 구간 {lo}–{hi}층, bootstrap {n_boot}회\n"], {}

    # ---- 1. 신뢰도 (프롬프트 4개)
    rel = []
    for (ds, task), g in I.groupby(["dataset", "task"]):
        rel.append(dict(dataset=ds, task=task, measure="간섭 I",
                        alpha=cronbach_alpha(g.pivot(index="id", columns="version", values="I"))))
    for (ds, task, ln), g in S.groupby(["dataset", "task", "lens"]):
        for m in ("S", "S_diff"):
            rel.append(dict(dataset=ds, task=task, measure=f"렌즈 {m} ({ln})",
                            alpha=cronbach_alpha(g.pivot(index="id", columns="version", values=m))))
    rel = pd.DataFrame(rel)
    rel["통과"] = np.where(rel.alpha >= RELIABILITY_MIN, "O", "X")
    md += ["## 1. 신뢰도 (프롬프트 4개, Cronbach's alpha)\n",
           f"기준 {RELIABILITY_MIN}. 미달한 지표의 상관은 '과장'으로 해석하지 않는다.\n", md_table(rel), ""]
    metrics["reliability"] = rel.to_dict("records")

    # ---- 문항 단위 (프롬프트 평균)
    Ii = I.groupby(["dataset", "task", "id", "group", "etym"], as_index=False).I.mean()
    Si = S.groupby(["dataset", "task", "id", "lens"], as_index=False)[["S", "S_diff"]].mean()
    items = Si.merge(Ii, on=["dataset", "task", "id"], how="left")
    items["p"] = np.exp(items.S)
    items.to_csv(out_dir / "item_scores.csv", index=False)

    # ---- 2. 상관
    cor = []
    for (ds, task, ln), g in items.dropna(subset=["I"]).groupby(["dataset", "task", "lens"]):
        for m in ("S", "S_diff"):
            r, a, b = corr_ci(g[m], g.I, g.group, n_boot)
            cor.append(dict(dataset=ds, task=task, lens=ln, x=m, n=len(g), r=r, ci_lo=a, ci_hi=b))
    cor = pd.DataFrame(cor)
    if len(cor):
        md += ["## 2. 렌즈 점수 × 간섭 상관 (Spearman, 문항 단위)\n", md_table(cor), ""]
    metrics["correlation"] = cor.to_dict("records")

    # ---- 3. 판정 (A 번역)
    a = items[(items.dataset == "A") & (items.task == "translate")].dropna(subset=["I"])
    if len(a) and "vanilla" in set(a.lens):
        ref = a[a.lens == "vanilla"].set_index("id")
        rows = []
        for ln in sorted(set(a.lens) - {"vanilla"}):
            new = a[a.lens == ln].set_index("id").loc[ref.index]
            (dr, dlo, dhi), (rd, rlo, rhi) = paired_delta(ref.S, new.S, ref.I, ref.p, new.p, ref.group, n_boot)
            rows.append({"렌즈": ln, "영어 확률(바닐라)": ref.p.mean(), "영어 확률(보정)": new.p.mean(),
                         "감소율 [CI]": ci(rd, rlo, rhi), "Δr [CI]": ci(dr, dlo, dhi), "판정": verdict(rd, dhi)})
        vt = pd.DataFrame(rows)
        md += ["## 3. 판정 (A, 번역 과제)\n",
               f"'크게 줄었음' = 감소율 ≥ {BIG_REDUCTION}. '상관 하락' = Δr의 95% CI 상한 < 0.\n", md_table(vt), ""]
        metrics["verdict"] = vt.to_dict("records")

    # ---- 4. 음성 대조 (B) vs A
    rows = []
    for ln in sorted(set(items.lens)):
        ra = items[(items.dataset == "A") & (items.task == "translate") & (items.lens == ln)].dropna(subset=["I"])
        rb = items[(items.dataset == "B") & (items.task == "translate") & (items.lens == ln)].dropna(subset=["I"])
        if len(ra) < 3 or len(rb) < 3:
            continue
        rng = np.random.default_rng(SEED)
        da = [_spearman(ra.S.values[i], ra.I.values[i]) for i in _boot_index(ra.group.values, rng, n_boot)]
        db = [_spearman(rb.S.values[i], rb.I.values[i]) for i in _boot_index(rb.group.values, rng, n_boot)]
        diff = np.array(da) - np.array(db)
        rows.append({"렌즈": ln, "r_A": _spearman(ra.S, ra.I), "r_B": _spearman(rb.S, rb.I),
                     "r_A - r_B [CI]": ci(_spearman(ra.S, ra.I) - _spearman(rb.S, rb.I),
                                          *np.nanpercentile(diff, [2.5, 97.5]))})
    if rows:
        md += ["## 4. 음성 대조: B(한국어 동음이의어)\n",
               "B의 간섭은 영어와 무관하므로 r_B ≈ 0이어야 한다. r_B도 크면 렌즈 점수가 영어가 아닌 "
               "'헷갈리는 정도'를 재고 있을 가능성이 있다.\n", md_table(pd.DataFrame(rows)), ""]
        metrics["negative_control"] = rows

    # ---- 5. 어종별
    rows = []
    for (ds, task, ln, et), g in items.dropna(subset=["I"]).groupby(["dataset", "task", "lens", "etym"]):
        if task != "translate" or len(g) < 8:
            continue
        r, lo_, hi_ = corr_ci(g.S, g.I, g.group, n_boot)
        rows.append(dict(dataset=ds, lens=ln, etym=et, n=len(g), r=r, ci_lo=lo_, ci_hi=hi_,
                         mean_I=g.I.mean(), mean_p=g.p.mean()))
    if rows:
        md += ["## 5. 어종별 (8문항 이상인 집단만)\n",
               "한자어는 한자음을 거쳐 중국어로 바로 가는 지름길이 있을 수 있다.\n", md_table(pd.DataFrame(rows)), ""]
        metrics["by_etym"] = rows

    # ---- 6. 양성 대조 (A: 영어 중간 단계 프롬프트)
    pa = items[(items.dataset == "A")].dropna(subset=["I"])
    if {"translate", "translate_en"} <= set(pa.task):
        rows = []
        for (task, ln), g in pa.groupby(["task", "lens"]):
            r, lo_, hi_ = corr_ci(g.S, g.I, g.group, n_boot)
            rows.append({"과제": task, "렌즈": ln, "평균 간섭": g.I.mean(), "평균 영어 확률": g.p.mean(),
                         "r [CI]": ci(r, lo_, hi_)})
        md += ["## 6. 양성 대조: 프롬프트에 영어 중간 단계\n",
               "영어 경유를 강제했으므로 translate_en에서 간섭·영어 확률이 커지고 상관이 보여야 한다. "
               "그렇지 않으면 간섭 지표가 영어 경유를 잡지 못하는 것이다.\n", md_table(pd.DataFrame(rows)), ""]
        metrics["positive_control"] = rows

    # ---- 7. C: 무관한 영어가 오르는지
    c = lens[(lens.dataset == "C") & (lens.role == "input")]
    if len(c):
        mid = c[c.layer.between(lo, hi)]
        agg = {k: (k, lambda s: np.exp(s).mean()) for k in ("lp_target", "lp_unrelated", "mass_en", "mass_zh", "mass_ko")
               if k in mid}
        tab = mid.groupby(["task", "lens"]).agg(**agg).reset_index()
        md += ["## 7. C: 중간층 평균 확률 (정답 영어 vs 무관 영어, 언어별 확률 질량)\n",
               "가설 1이 맞다면 바닐라에서는 무관한 영어(chair)와 영어 질량 전체가 높고, 평균 제거 렌즈에서 줄어야 한다.\n",
               md_table(tab, "{:.4f}"), ""]
        metrics["C"] = tab.to_dict("records")

    # ---- 렌즈 적합도
    fit_file = out_dir / "lens_fit.csv"
    if fit_file.exists():
        f = pd.read_csv(fit_file)
        f = f[f.lens.isin(["vanilla", "mean", "affine"]) & f.layer.between(lo, hi)]
        md += ["## 부록: 렌즈 적합도 (held-out KL(최종 || 렌즈), 중간층 평균)\n",
               "아핀 렌즈가 바닐라보다 KL이 작아야 기저 맞춤이 실제로 된 것이다.\n",
               md_table(f.groupby("lens", as_index=False).kl.mean()), ""]

    (out_dir / "summary.md").write_text("\n".join(md), "utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=float), "utf-8")
    if plots:
        make_plots(out_dir, items, lens, lo, hi)
    print(f"[done] → {out_dir / 'summary.md'}")
    return metrics


def make_plots(out_dir: Path, items: pd.DataFrame, lens: pd.DataFrame, lo: int, hi: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 산점도: 데이터셋×과제마다 렌즈별 패널
    for (ds, task), g in items.dropna(subset=["I"]).groupby(["dataset", "task"]):
        order = ["vanilla", "mean", "affine"]
        lens_names = [n for n in order if n in set(g.lens)] + sorted(set(g.lens) - set(order))
        fig, axes = plt.subplots(1, len(lens_names), figsize=(4.2 * len(lens_names), 4), squeeze=False)
        for ax, ln in zip(axes[0], lens_names):
            h = g[g.lens == ln]
            ax.scatter(h.S, h.I, s=14, alpha=0.7)
            ax.set_title(f"{ln}  r={_spearman(h.S, h.I):.2f}")
            ax.set_xlabel("lens score (max mid-layer log p)")
            ax.axhline(0, lw=0.5, c="gray")
        axes[0][0].set_ylabel("interference I")
        fig.suptitle(f"{ds} / {task}")
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{ds}_{task}.png", dpi=120)
        plt.close(fig)

    # 층별 곡선: 입력어 프롬프트에서 probe 확률과 영어 질량
    inp = lens[lens.role == "input"]
    cols = [c for c in inp.columns if c.startswith("lp_") or c == "mass_en"]
    for (ds, task), g in inp.groupby(["dataset", "task"]):
        use = [c for c in cols if g[c].notna().any()]
        fig, axes = plt.subplots(1, len(use), figsize=(4.2 * len(use), 3.6), squeeze=False)
        for ax, c in zip(axes[0], use):
            for ln, h in g.groupby("lens"):
                m = h.groupby("layer")[c].apply(lambda s: np.exp(s).mean())
                ax.plot(m.index, m.values, label=ln)
            ax.axvspan(lo, hi, color="gray", alpha=0.1)
            ax.set_title(c)
            ax.set_xlabel("layer")
        axes[0][0].set_ylabel("mean probability")
        axes[0][-1].legend()
        fig.suptitle(f"{ds} / {task}")
        fig.tight_layout()
        fig.savefig(out_dir / f"layers_{ds}_{task}.png", dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    analyze(Path(args.out_dir or RESULTS_DIR / args.model.replace("/", "_")), args.n_boot)


if __name__ == "__main__":
    main()
