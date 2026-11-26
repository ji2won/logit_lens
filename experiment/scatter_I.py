#!/usr/bin/env python3
"""데이터셋 A(한→중, 바닐라 렌즈): x = 간섭 I, y = 렌즈 점수 4종 산점도.
  python -m experiment.scatter_I --model llama3.2-3b
  → results/<model>/A_en_run/scatter_I_vs_lens.png, scatter_I_vs_lens.csv
I = log p(방해어 | 입력어) - 평균 log p(방해어 | 대조어)  (최종 출력, 렌즈 무관)
S        중간층(7–21) 영어 다의어 probe(spring) 로그확률 최댓값
S_diff   층마다 (입력어 - 대조어 평균) 후 중간층 최댓값
S_d      방해 뜻 영어 묶음(coil, bounce …) probe 최댓값
S_d_diff 방해 뜻 probe의 입력어 - 대조어 차이 최댓값
점수는 프롬프트별로 구한 뒤 4개 평균. r = Spearman, 95% CI = 영어 단어 묶음 bootstrap.
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

from .analyze import corr_ci, mid_layers
from .config import RESULTS_DIR

for f in ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False


def scores(lens, col):
    lo, hi = mid_layers(int(lens.layer.max()))
    d = lens[lens.layer.between(lo, hi)].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    k = ["id", "version"]
    inp = d[d.role == "input"]
    s = inp.groupby(k)[col].max()
    ctl = d[d.role == "control"].groupby(k + ["layer"])[col].mean().rename("ctrl")
    diff = inp.set_index(k + ["layer"])[col].to_frame().join(ctl, how="inner")
    sd = (diff[col] - diff.ctrl).groupby(k).max()
    return pd.DataFrame({"S": s, "S_diff": sd}).groupby("id").mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--exclude", action="store_true", help="외래어(etym=외래어)·정답/방해어 한자 공유 문항 제외")
    a = ap.parse_args()
    R = RESULTS_DIR / a.model
    lens = pd.read_csv(R / "lens.csv.gz", keep_default_na=False, low_memory=False)
    lens = lens[(lens.dataset == "A") & (lens.task == "translate") & (lens.lens == "vanilla")]
    it = pd.read_csv(R / "item_scores.csv")
    it = it[(it.dataset == "A") & (it.task == "translate") & (it.lens == "vanilla")].set_index("id")[["I", "group"]]
    p = scores(lens, "lp_poly")
    q = scores(lens, "lp_sense_d").add_prefix("Sd_")
    df = it.join(p).join(q).rename(columns={"Sd_S": "S_d", "Sd_S_diff": "S_d_diff"})
    ko = pd.read_csv(RESULTS_DIR.parent / "A_final.csv", keep_default_na=False).set_index("id")
    df["ko"] = ko.ko_input
    tag = ""
    if a.exclude:
        drop = ko.index[(ko.etym_ko_input == "외래어") | (ko.zh_shared_chars != "")]
        print("제외:", ", ".join(ko.ko_input[drop]))
        df = df.drop(index=[i for i in drop if i in df.index])
        tag = "_excl"
    df.to_csv(R / "A_en_run" / f"scatter_I_vs_lens{tag}.csv")
    panels = [("S", "영어 다의어 probe (spring)\nlog p 중간층 최댓값"),
              ("S_diff", "영어 다의어 probe\n입력어 - 대조어 (log)"),
              ("S_d", "방해 뜻 영어 probe (coil …)\nlog p 중간층 최댓값"),
              ("S_d_diff", "방해 뜻 영어 probe\n입력어 - 대조어 (log)")]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.4), facecolor="#fcfcfb")
    for ax, (c, lab) in zip(axs.flat, panels):
        g = df[["I", c, "group", "ko"]].dropna()
        r, lo, hi = corr_ci(g.I, g[c], g.group)
        ax.set_facecolor("#fcfcfb")
        ax.scatter(g.I, g[c], s=26, color="#2a78d6", alpha=0.75, edgecolor="#fcfcfb", linewidth=0.8, zorder=3)
        b = np.polyfit(g.I, g[c], 1)
        xs = np.linspace(g.I.min(), g.I.max(), 50)
        ax.plot(xs, np.polyval(b, xs), color="#52514e", lw=1.2, ls="--", zorder=2)
        ax.axvline(0, color="#cfcfcb", lw=0.8, zorder=1)
        for _, row in pd.concat([g.nlargest(4, "I"), g.nlargest(2, c), g.nsmallest(2, "I")]).drop_duplicates().iterrows():
            ax.annotate(row.ko, (row.I, row[c]), xytext=(4, 3), textcoords="offset points", fontsize=8, color="#52514e")
        ax.set_title(f"Spearman r = {r:.2f} [{lo:.2f}, {hi:.2f}],  n = {len(g)}", fontsize=10, color="#0b0b0b", loc="left")
        ax.set_ylabel(lab, fontsize=9, color="#52514e")
        ax.set_xlabel("간섭 I = log p(방해어|입력어) - log p(방해어|대조어)", fontsize=9, color="#52514e")
        ax.grid(color="#e8e8e4", lw=0.6)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#bdbcb6")
        ax.tick_params(colors="#52514e", labelsize=8)
    fig.suptitle(f"{a.model} · 데이터셋 A 한→중 · 바닐라 렌즈: 간섭 I vs 렌즈 점수 (문항 단위)"
                 + (" · 외래어·한자 공유 제외" if a.exclude else ""), fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(R / "A_en_run" / f"scatter_I_vs_lens{tag}.png", dpi=160)
    print(df[["I", "S", "S_diff", "S_d", "S_d_diff"]].describe().round(2))


if __name__ == "__main__":
    main()
