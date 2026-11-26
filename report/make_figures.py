#!/usr/bin/env python3
"""REPORT.md용 요약 그림. 수치는 results/ 아래 summary.md·pvalues.csv에서 옮겼다.
  python report/make_figures.py  → report/figures/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "AppleGothic", "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#a8a7a2", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 10, "savefig.dpi": 160,
})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#a8a7a2"
MODELS = ["Qwen3-1.7B", "EXAONE-3.5-2.4B", "Llama-3.2-3B"]


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(OUT / name)
    plt.close(fig)


# 1. 렌즈 적합도 (held-out KL, 중간층 평균)
kl = {"바닐라": [8.77, 7.82, 7.64], "평균 제거": [12.31, 7.78, 8.06], "아핀": [2.70, 1.86, 1.92]}
fig, ax = plt.subplots(figsize=(6.4, 3.4))
x = np.arange(3); w = 0.26
for i, (k, v) in enumerate(kl.items()):
    b = ax.bar(x + (i - 1) * w, v, w - 0.02, color=C[i], label=k)
    ax.bar_label(b, fmt="%.1f", fontsize=8, color="#52514e", padding=2)
ax.set_xticks(x, MODELS); ax.set_ylabel("KL(최종 ‖ 렌즈), nat")
ax.set_title("그림 1. 렌즈 적합도 — 낮을수록 최종 출력과 가깝다", loc="left", fontsize=11)
ax.legend(frameon=False, ncol=3, loc="upper right")
save(fig, "fig1_lens_kl.png")

# 2. A(주 실험) vs B(음성 대조) 상관, 바닐라 렌즈
p = pd.read_csv(ROOT / "results" / "pvalues.csv")
p = p[(p.range == "mid") & (p.analysis == "상관") & (p.lens == "vanilla")]
keys = ["qwen3-1.7b", "exaone3.5-2.4b", "llama3.2-3b"]
conds = [("A", "translate", "A 번역 (영어 다의어)"),
         ("B", "translate", "B 번역 (한국어 동음이의어, 음성 대조)"),
         ("B", "translate_ctx", "B 예문 조건 (음성 대조)")]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, xm in zip(axes, ["S", "S_diff"]):
    for j, (ds, task, lab) in enumerate(conds):
        r, lo, hi = [], [], []
        for m in keys:
            row = p[(p.model == m) & (p.dataset == ds) & (p.task == task) & (p.x == xm)].iloc[0]
            r.append(row.r); lo.append(row.r - row.ci_lo); hi.append(row.ci_hi - row.r)
        ax.bar(x + (j - 1) * w, r, w - 0.02, color=C[j], label=lab,
               yerr=[lo, hi], error_kw=dict(ecolor="#52514e", elinewidth=1, capsize=2))
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(x, MODELS, fontsize=9)
    ax.set_title(f"x = {xm}", loc="left")
axes[0].set_ylabel("Spearman r (렌즈 점수 × 간섭 I)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9, ncol=3, loc="lower center")
fig.subplots_adjust(bottom=0.2)
fig.suptitle("그림 2. 렌즈 영어 점수는 영어와 무관한 B에서 더 강하게 간섭을 따라간다 (바닐라, 95% CI)",
             x=0.01, ha="left", fontsize=11)
save(fig, "fig2_rA_vs_rB.png", rect=(0, 0.08, 1, 1))

# 3. C 번역 과제 중간층 언어 질량
mass = {  # (영어 단어, 영어 조각 = mass_en - mass_en_word, 중국어)
    "Qwen3-1.7B": {"바닐라": (0.03, 0.25, 0.34), "평균 제거": (0.05, 0.25, 0.59), "아핀": (0.01, 0.01, 0.34)},
    "EXAONE-3.5-2.4B": {"바닐라": (0.32, 0.32, 0.01), "평균 제거": (0.31, 0.33, 0.01), "아핀": (0.01, 0.02, 0.20)},
    "Llama-3.2-3B": {"바닐라": (0.40, 0.31, 0.03), "평균 제거": (0.42, 0.31, 0.02), "아핀": (0.00, 0.02, 0.78)},
}
fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), sharey=True)
for ax, (m, d) in zip(axes, mass.items()):
    lenses = list(d)
    a = np.array([d[k] for k in lenses])
    bot = np.zeros(3)
    for i, (lab, col) in enumerate(zip(["영어 단어", "영어 조각", "중국어"], [C[0], "#9fc3ee", C[1]])):
        ax.bar(lenses, a[:, i], 0.6, bottom=bot, color=col, label=lab, edgecolor="#fcfcfb", linewidth=1.5)
        bot += a[:, i]
    ax.set_title(m, loc="left", fontsize=10); ax.set_ylim(0, 1)
axes[0].set_ylabel("중간층 평균 확률 질량")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9, ncol=3, loc="lower center")
fig.suptitle("그림 3. C 한→중 번역: 평균 제거는 영어를 못 지우고, 아핀은 영어를 지우고 중국어로 채운다",
             x=0.01, ha="left", fontsize=11)
save(fig, "fig3_C_language_mass.png", rect=(0, 0.07, 1, 1))

# 4. 인과 개입: 영어 방향 제거 (Llama, distractor_ablate)
conds4 = ["이 문항 영어 (U)", "다른 문항 영어 (U_oth)", "무작위 (U_rand)"]
mid = {"Δ정답": [(-0.19, -0.28, -0.11), (-0.01, -0.03, 0.01), (-0.02, -0.05, 0.02)],
       "Δ방해어": [(-0.30, -0.45, -0.16), (-0.00, -0.04, 0.04), (-0.01, -0.09, 0.06)],
       "ΔI": [(-0.16, -0.29, -0.04), (0.01, -0.02, 0.04), (-0.05, -0.10, 0.01)]}
late = {"Δ정답": [(-0.12, -0.19, -0.06), (-0.02, -0.03, -0.00), (-0.01, -0.03, 0.02)],
        "Δ방해어": [(-0.58, -0.78, -0.40), (-0.04, -0.07, -0.00), (-0.04, -0.09, 0.01)],
        "ΔI": [(-0.38, -0.56, -0.23), (0.01, -0.02, 0.04), (-0.03, -0.07, 0.00)]}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, (band, d) in zip(axes, [("7–21층에서 제거", mid), ("22–27층에서 제거", late)]):
    for j, c in enumerate(conds4):
        vals = [d[k][j] for k in d]
        r = [v[0] for v in vals]; lo = [v[0] - v[1] for v in vals]; hi = [v[2] - v[0] for v in vals]
        ax.bar(x + (j - 1) * w, r, w - 0.02, color=[C[0], GRAY, "#d3d2cc"][j], label=c,
               yerr=[lo, hi], error_kw=dict(ecolor="#52514e", elinewidth=1, capsize=2))
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.axhline(-1.75, color=C[7], lw=1, ls="--")
    ax.text(2.45, -1.70, "원래 평균 I = 1.75", color="#52514e", fontsize=8, ha="right", va="bottom")
    ax.set_xticks(x, list(d)); ax.set_title(band, loc="left")
axes[0].set_ylabel("개입 후 - 원래 (nat)")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("그림 4. 영어 방향을 지워도 간섭은 원래 크기의 9%(중간층)·22%(후반층)만 줄어든다 (Llama, A 99문항)",
             x=0.01, ha="left", fontsize=11)
save(fig, "fig4_distractor_ablate.png")

# 5. 부풀림: 잔차 몫 vs 렌즈 확률 (Llama A, inflation)
L = [8, 12, 16, 18, 20, 21, 22, 24, 26, 27, 28]
share_en = [0.027, 0.027, 0.056, 0.070, 0.105, 0.125, 0.129, 0.117, 0.094, 0.084, 0.049]
share_zh = [0.018, 0.019, 0.047, 0.058, 0.092, 0.110, 0.116, 0.124, 0.112, 0.105, 0.173]
p_en = [0.000, 0.000, 0.034, 0.142, 0.487, 0.616, 0.657, 0.593, 0.428, 0.294, 0.005]
p_zh = [0.000, 0.000, 0.002, 0.004, 0.009, 0.006, 0.007, 0.015, 0.056, 0.115, 0.530]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
for ax, (ye, yz, t) in zip(axes, [(share_en, share_zh, "잔차에서 차지하는 몫 (share)"),
                                  (p_en, p_zh, "바닐라 렌즈 확률")]):
    ax.plot(L, ye, color=C[0], lw=2, marker="o", ms=4, label="영어")
    ax.plot(L, yz, color=C[1], lw=2, marker="o", ms=4, label="중국어 정답")
    ax.axvspan(20, 24, color="#e6e5e0", zorder=0)
    ax.set_title(t, loc="left"); ax.set_xlabel("층")
axes[0].axhline(0.034, color=GRAY, ls=":", lw=1); axes[0].text(8, 0.038, "무작위 기대값", fontsize=8, color="#52514e")
axes[0].legend(frameon=False, fontsize=9)
axes[1].annotate("20–24층: 몫은 같은데(0.121)\n확률은 0.654 대 0.012", xy=(22, 0.657), xytext=(9.5, 0.45),
                 fontsize=8, color="#52514e", arrowprops=dict(arrowstyle="-", color=GRAY))
fig.suptitle("그림 5. 렌즈 확률은 잔차 속 크기에 비해 영어를 과장한다 (Llama, A, softmax 증폭)",
             x=0.01, ha="left", fontsize=11)
save(fig, "fig5_inflation.png")

# 6. 양성 대조: 같은 제거를 한→중 vs 한→영에 (C)
fig, ax = plt.subplots(figsize=(6.4, 3.4))
tasks = ["한→중", "한→영 (양성 대조)"]
vals = {"렌즈 1위 영어 제거": [(-0.195, -0.255, -0.143), (-1.336, -1.560, -1.123)],
        "다른 영어 제거": [(-0.005, -0.013, 0.004), (0.002, -0.004, 0.010)],
        "무작위 제거": [(-0.008, -0.015, -0.002), (0.000, -0.005, 0.006)]}
xx = np.arange(2)
for j, (k, v) in enumerate(vals.items()):
    r = [a[0] for a in v]
    b = ax.bar(xx + (j - 1) * w, r, w - 0.02, color=[C[0], GRAY, "#d3d2cc"][j], label=k,
               yerr=[[a[0] - a[1] for a in v], [a[2] - a[0] for a in v]],
               error_kw=dict(ecolor="#52514e", elinewidth=1, capsize=2))
    if j == 0:
        ax.bar_label(b, fmt="%.2f", fontsize=8, color="#52514e", padding=3)
ax.axhline(0, color="#52514e", lw=0.8)
ax.set_xticks(xx, tasks); ax.set_ylabel("Δ log p(정답), nat"); ax.set_ylim(-1.8, 0.1)
ax.legend(frameon=False, fontsize=8, loc="lower left")
ax.set_title("그림 6. 같은 제거가 한→영 정답은 크게 떨어뜨린다 (Llama, C)", loc="left", fontsize=11)
save(fig, "fig6_posctrl.png")

# 7. 단일 언어 문맥 간섭
fig, ax = plt.subplots(figsize=(6.4, 3.2))
lab = ["한→중 번역", "중국어 따라 쓰기", "한국어 따라 쓰기"]
v = [(1.75, 1.19, 2.32), (0.99, 0.69, 1.29), (0.28, -0.01, 0.55)]
b = ax.bar(lab, [a[0] for a in v], 0.55, color=C[0],
           yerr=[[a[0] - a[1] for a in v], [a[2] - a[0] for a in v]],
           error_kw=dict(ecolor="#52514e", elinewidth=1, capsize=3))
ax.bar_label(b, fmt="%.2f", fontsize=8, color="#52514e", padding=3)
ax.set_ylabel("평균 간섭 I (nat)")
ax.set_title("그림 7. 간섭은 번역할 때 가장 크다 (Llama, A 99문항)", loc="left", fontsize=11)
save(fig, "fig7_mono_interference.png")

print("saved to", OUT)
