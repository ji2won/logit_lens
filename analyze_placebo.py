"""무관 중국어 단어 대조 실험 분석 (GPU 필요 없음). run_placebo.py 출력을 읽는다.

질문: 이자 → 兴趣처럼 방해어 확률이 대조어보다 오르는 것이 '영어 다의어(interest) 때문'인가,
      아니면 입력어를 넣으면 아무 중국어 단어나 같이 오르는 '일반적인 입력 효과'인가?

같은 공식으로 무관 단어 u에 대해서도 계산한다.
  I(w) = log p(w | 입력어) − 평균 log p(w | 대조어 2개)      (단어 전체 확률, 최종 출력, 템플릿 평균)
  I_real = I(방해어),  I_placebo = I(u), u = 그 문항과 무관한 중국어 단어

무관 단어 고르기 (문항마다)
  후보 = placebo_pool.csv (A의 정답·방해어 + C의 중국어 단어). 아래에 걸리면 제외:
    - 그 문항의 정답·방해어 자신
    - 영어 단어가 같은 문항(쌍 문항: 감방/세포 등)에서 온 단어
    - 그 문항의 정답·방해어와 한자를 하나라도 공유하는 단어
    - 입력어·대조어의 번역어 (A·C 표에서 그 한국어 단어의 중국어)
    - few-shot 예시의 중국어 단어(雨伞 등)
    - --exclude 파일(item,zh)에 적은 단어 (눈으로 보고 의미가 걸리는 것)
  matched = 위를 통과한 것 중 빈도(zipf)가 방해어 ±0.5, 그 모델에서 토큰 수가 방해어와 같은 것.
            5개가 안 되면 zipf ±1.0 → 토큰 수 ±1 순서로 넓힌다(match_level 열에 기록).

    python analyze_placebo.py placebo/llama placebo/exaone_chat --data ../dataset_A/A_final_v2.csv \
        --c ../dataset_C/C_final.csv --pool placebo_pool.csv --out results_placebo
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts  # noqa: E402

HAN = re.compile(r"[一-鿿]")
FOCUS = ["층", "미사", "이자", "감방", "부리", "사냥감"]  # 다른 분 결과에서 확인한 6문항


def exclusions(df, pool, C, extra):
    """문항 id → 제외할 중국어 단어 집합, 제외 이유 표."""
    ko2zh = {}
    for _, r in df.iterrows():
        ko2zh.setdefault(r.ko_input, set()).add(r.zh_answer)
    if C is not None:
        for _, r in C.iterrows():
            ko2zh.setdefault(r.ko, set()).add(r.zh)
            ko2zh.setdefault(r.ko_unrelated, set()).add(r.zh_unrelated)
    shot_zh = {z for tpl in prompts.TEMPLATES.values() for _, z in tpl[0]}
    src_en = {w: set(str(s).split(";")) for w, s in zip(pool.word, pool.src_en)}
    ex, why = {}, []
    for _, r in df.iterrows():
        own = {r.zh_answer, r.zh_distractor}
        chars = set(HAN.findall(r.zh_answer + r.zh_distractor))
        ctrls = [c.strip() for c in str(r.ko_control_final).split(";")]
        trans = set().union(*[ko2zh.get(k, set()) for k in [r.ko_input] + ctrls])
        s = set()
        for w in pool.word:
            reason = None
            if w in own:
                reason = "own"
            elif r.en in src_en[w]:
                reason = "same_en"
            elif chars & set(w):
                reason = "shared_char"
            elif w in trans:
                reason = "translation_of_input_or_ctrl"
            elif w in shot_zh:
                reason = "fewshot"
            elif (r.id, w) in extra:
                reason = "manual"
            if reason:
                s.add(w)
                if reason != "own":
                    why.append(dict(item=r.id, ko_input=r.ko_input, zh=w, reason=reason))
        ex[r.id] = s
    return ex, pd.DataFrame(why)


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def wilcoxon_p(x):
    try:
        from scipy.stats import wilcoxon
        x = np.asarray(x, float)
        x = x[~np.isnan(x) & (x != 0)]
        return float(wilcoxon(x).pvalue) if len(x) >= 5 else np.nan
    except Exception:
        return np.nan


def analyze_one(prefix, df, pool, ex, out_dir, min_correct):
    M = pd.read_csv(prefix + "_prompts.csv")
    S = pd.read_csv(prefix + "_scores.csv.gz")
    tag = os.path.basename(prefix)
    os.makedirs(out_dir, exist_ok=True)
    S = S.merge(M[["pid", "item", "template", "cond"]], on="pid")

    # I(w) 템플릿별: 입력 − 대조어 평균. lp(전체 단어)와 lp_tok1(첫 토큰) 둘 다
    W = S.pivot_table(index=["item", "template", "zh"], columns="cond", values=["lp", "lp_tok1"])
    T = pd.DataFrame(index=W.index)
    for v in ["lp", "lp_tok1"]:
        T["I" if v == "lp" else "I_tok1"] = W[(v, "input")] - (W[(v, "ctrl1")] + W[(v, "ctrl2")]) / 2
    T["lp_input"] = W[("lp", "input")]
    T = T.reset_index()

    # 모델 토큰 수: T1 입력 조건의 단어 토큰 수
    nt = S[(S.cond == "input") & (S.template == "T1")].groupby(["item", "zh"]).ntok.first()
    T = T.merge(nt.rename("ntok").reset_index(), on=["item", "zh"], how="left")

    info = df.set_index("id")
    T["is_dis"] = T.zh.values == info.loc[T.item, "zh_distractor"].values
    T["is_ans"] = T.zh.values == info.loc[T.item, "zh_answer"].values
    T["excluded"] = [w in ex[i] for i, w in zip(T.item, T.zh)]
    zipf = dict(zip(pool.word, pool.zipf))
    T["zipf"] = T.zh.map(zipf)
    T.to_csv(os.path.join(out_dir, "item_word_template_I.csv.gz"), index=False, compression="gzip")

    A = T.groupby(["item", "zh"]).agg(I=("I", "mean"), I_tok1=("I_tok1", "mean"), I_min=("I", "min"),
                                     n_pos_tpl=("I", lambda s: int((s > 0).sum())), lp_input=("lp_input", "mean"),
                                     ntok=("ntok", "first"), zipf=("zipf", "first"), is_dis=("is_dis", "first"),
                                     is_ans=("is_ans", "first"), excluded=("excluded", "first")).reset_index()

    # 정답 여부(입력 조건 greedy)
    if "correct" in M:
        nc = M[M.cond == "input"].groupby("item").correct.sum()
    else:
        nc = pd.Series(np.nan, index=df.id)

    rows = []
    for it, g in A.groupby("item"):
        r = info.loc[it]
        d = g[g.is_dis].iloc[0]
        a_ = g[g.is_ans].iloc[0]
        pl = g[~g.excluded]
        lvl = None
        for lvl, zw, tw in [(0, 0.5, 0), (1, 1.0, 0), (2, 1.0, 1), (3, 9.0, 9)]:
            m = pl[((pl.zipf - r.zipf_zh_distractor).abs() <= zw) & ((pl.ntok - d.ntok).abs() <= tw)]
            if len(m) >= 5:
                break
        pct = float((m.I < d.I).mean() + 0.5 * (m.I == d.I).mean())
        rows.append(dict(
            item=it, ko_input=r.ko_input, en=r.en, zh_answer=r.zh_answer, zh_distractor=r.zh_distractor,
            ctrl=r.ko_control_final, n_correct=nc.get(it, np.nan),
            I_real=d.I, I_real_tok1=d.I_tok1, real_pos_tpl=d.n_pos_tpl, lp_real_input=d.lp_input, ntok_real=d.ntok,
            I_answer=a_.I,
            placebo_all_mean=pl.I.mean(), placebo_all_median=pl.I.median(), placebo_all_sd=pl.I.std(),
            placebo_all_frac_pos=(pl.I > 0).mean(), n_placebo_all=len(pl),
            placebo_matched_mean=m.I.mean(), placebo_matched_median=m.I.median(), placebo_matched_sd=m.I.std(),
            placebo_matched_tok1_mean=m.I_tok1.mean(),
            placebo_matched_p95=m.I.quantile(0.95), n_matched=len(m), match_level=lvl,
            I_adj=d.I - m.I.mean(), pct_rank=pct, z=(d.I - m.I.mean()) / (m.I.std() or np.nan),
            top_placebo=", ".join(f"{w}({v:+.1f})" for w, v in pl.nlargest(3, "I")[["zh", "I"]].values),
        ))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out_dir, "items_placebo.csv"), index=False, encoding="utf-8")

    # 템플릿별 순위(방해어가 무관 단어보다 높은 템플릿 수)
    TT = T[~T.excluded | T.is_dis]
    tpl_rank = []
    for (it, tp), g in TT.groupby(["item", "template"]):
        d = g[g.is_dis].I.iloc[0]
        p = g[~g.is_dis].I
        tpl_rank.append(dict(item=it, template=tp, I_real=d, placebo_mean=p.mean(), pct=(p < d).mean()))
    TR = pd.DataFrame(tpl_rank)
    TR.to_csv(os.path.join(out_dir, "template_rank.csv"), index=False)

    def summ(X):
        return dict(
            n_items=int(len(X)),
            I_real_mean=float(X.I_real.mean()), I_real_ci=boot_ci(X.I_real), I_real_frac_pos=float((X.I_real > 0).mean()),
            placebo_matched_mean=float(X.placebo_matched_mean.mean()), placebo_matched_ci=boot_ci(X.placebo_matched_mean),
            placebo_all_mean=float(X.placebo_all_mean.mean()),
            placebo_frac_pos=float(X.placebo_all_frac_pos.mean()),
            I_adj_mean=float(X.I_adj.mean()), I_adj_ci=boot_ci(X.I_adj), I_adj_frac_pos=float((X.I_adj > 0).mean()),
            I_adj_wilcoxon_p=wilcoxon_p(X.I_adj),
            pct_rank_mean=float(X.pct_rank.mean()), frac_pct95=float((X.pct_rank >= 0.95).mean()),
            frac_pct95_expected_by_chance=0.05,
            corr_real_vs_placebo=float(np.corrcoef(X.I_real, X.placebo_matched_mean)[0, 1]) if len(X) > 2 else np.nan,
            I_answer_mean=float(X.I_answer.mean()),
        )

    subsets = {"all": R}
    if R.n_correct.notna().any():
        subsets[f"correct>={min_correct}/4"] = R[R.n_correct >= min_correct]
    subsets["exclude_shared_char_ans_dis"] = R[[not (set(HAN.findall(a)) & set(HAN.findall(b)))
                                                for a, b in zip(R.zh_answer, R.zh_distractor)]]
    summary = {k: summ(v) for k, v in subsets.items()}
    summary["tpl_frac_real_above_all_placebos_mean"] = float(TR.pct.mean())
    summary["match_levels"] = R.match_level.value_counts().to_dict()
    chk = prefix + "_check.json"
    if os.path.exists(chk):
        summary["sanity"] = {k: v for k, v in json.load(open(chk)).items() if k != "rows"}
    json.dump(summary, open(os.path.join(out_dir, "summary_placebo.json"), "w"), ensure_ascii=False, indent=1,
              default=float)

    # 보고서
    L = [f"# 무관 중국어 단어 대조 실험 — {tag}", ""]
    if "sanity" in summary:
        L += [f"sanity(배치 계산 = 한 개씩 계산): {summary['sanity']}", ""]
    L += ["I = log p(단어|입력어) − 평균 log p(단어|대조어 2개), 단어 전체 확률, 최종 출력, 템플릿 4개 평균.",
          "I_adj = I(방해어) − 평균 I(빈도·토큰 수를 맞춘 무관 단어). pct = 무관 단어 중 방해어보다 I가 낮은 비율.",
          "무관 단어도 평균 I가 크게 양수면 '입력어를 넣으면 중국어 단어가 전반적으로 오르는' 일반 효과가 있다는 뜻이다.", ""]
    L += ["| 문항 묶음 | 문항 수 | I(방해어) 평균 [95% CI] | I(무관, 맞춤) 평균 [CI] | I_adj 평균 [CI] | I_adj>0 비율 | Wilcoxon p | pct≥0.95 비율 | I(정답) 평균 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, s in summary.items():
        if not isinstance(s, dict) or "n_items" not in s:
            continue
        L.append(f"| {k} | {s['n_items']} | {s['I_real_mean']:+.2f} [{s['I_real_ci'][0]:+.2f}, {s['I_real_ci'][1]:+.2f}] "
                 f"| {s['placebo_matched_mean']:+.2f} [{s['placebo_matched_ci'][0]:+.2f}, {s['placebo_matched_ci'][1]:+.2f}] "
                 f"| {s['I_adj_mean']:+.2f} [{s['I_adj_ci'][0]:+.2f}, {s['I_adj_ci'][1]:+.2f}] | {s['I_adj_frac_pos']:.0%} "
                 f"| {s['I_adj_wilcoxon_p']:.2g} | {s['frac_pct95']:.0%} (우연 5%) | {s['I_answer_mean']:+.2f} |")
    s = summary["all"]
    L += ["", f"- 무관 단어(전체 후보) 평균 I: {s['placebo_all_mean']:+.2f}, 무관 단어 중 I>0 비율 평균: {s['placebo_frac_pos']:.0%}",
          f"- 문항별 I(방해어)와 I(무관 평균)의 상관: {s['corr_real_vs_placebo']:.2f}  (높으면 방해어 I의 상당 부분이 일반 효과)",
          f"- 템플릿별로 방해어가 무관 단어보다 높은 비율의 평균: {summary['tpl_frac_real_above_all_placebos_mean']:.0%}",
          f"- 매칭 단계(0 = zipf±0.5·토큰 수 같음): {summary['match_levels']}", ""]

    F = R[R.ko_input.isin(FOCUS)].set_index("ko_input").reindex([f for f in FOCUS if f in set(R.ko_input)])
    if len(F):
        L += ["## 다른 분이 본 6문항", "",
              "| 입력 | 방해어 | I(방해어) | 양수 템플릿 | I(무관 맞춤 평균) | 무관 95%분위 | I_adj | pct | 무관 상위 3 | 첫 토큰 기준 I(방해어) / I(무관) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for ko, r in F.iterrows():
            L.append(f"| {ko} | {r.zh_distractor} | {r.I_real:+.2f} | {r.real_pos_tpl}/4 | {r.placebo_matched_mean:+.2f} "
                     f"| {r.placebo_matched_p95:+.2f} | {r.I_adj:+.2f} | {r.pct_rank:.2f} | {r.top_placebo} "
                     f"| {r.I_real_tok1:+.2f} / {r.placebo_matched_tok1_mean:+.2f} |")
        L.append("")

    L += ["## 문항 전체 (I_adj 내림차순)", "",
          "| 입력 | 정답 | 방해어 | 정답 템플릿 수 | I(방해어) | I(무관 맞춤) | n | I_adj | pct |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.sort_values("I_adj", ascending=False).iterrows():
        L.append(f"| {r.ko_input} | {r.zh_answer} | {r.zh_distractor} | {r.n_correct:.0f} | {r.I_real:+.2f} "
                 f"| {r.placebo_matched_mean:+.2f} | {r.n_matched} | {r.I_adj:+.2f} | {r.pct_rank:.2f} |")
    open(os.path.join(out_dir, "report_placebo.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    plot(R, tag, out_dir)
    print(f"{tag}: I(방해어) {s['I_real_mean']:+.2f}, I(무관) {s['placebo_matched_mean']:+.2f}, I_adj {s['I_adj_mean']:+.2f} "
          f"(p={s['I_adj_wilcoxon_p']:.2g}), pct≥0.95 {s['frac_pct95']:.0%} → {out_dir}")
    return R


def plot(R, tag, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    lim = [min(R.I_real.min(), R.placebo_matched_mean.min()) - 0.5, max(R.I_real.max(), R.placebo_matched_mean.max()) + 0.5]
    ax[0].plot(lim, lim, color="0.7", lw=1)
    ax[0].axhline(0, color="0.85", lw=1)
    ax[0].axvline(0, color="0.85", lw=1)
    ax[0].scatter(R.placebo_matched_mean, R.I_real, s=14, color="#3b6ea8")
    for _, r in R[R.ko_input.isin(FOCUS)].iterrows():
        ax[0].annotate(r.ko_input, (r.placebo_matched_mean, r.I_real), fontsize=8, fontfamily=["NanumGothic", "DejaVu Sans"])
    ax[0].set_xlabel("I(unrelated zh words, matched), item mean")
    ax[0].set_ylabel("I(distractor)")
    ax[0].set_title(f"{tag}: points above diagonal = distractor > unrelated")
    bins = np.linspace(min(R.I_real.min(), R.placebo_matched_mean.min()), max(R.I_real.max(), R.placebo_matched_mean.max()), 30)
    ax[1].hist(R.placebo_matched_mean, bins=bins, alpha=0.6, label="unrelated (matched mean)", color="0.5")
    ax[1].hist(R.I_real, bins=bins, alpha=0.6, label="distractor", color="#c0504d")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].legend()
    ax[1].set_xlabel("I (nats)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "placebo_scatter.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+", help="run_placebo.py의 --out 접두어들")
    ap.add_argument("--data", default="../dataset_A/A_final_v2.csv")
    ap.add_argument("--c", default="../dataset_C/C_final.csv", help="입력·대조어 번역어 제외용. 없으면 ''")
    ap.add_argument("--pool", default="placebo_pool.csv")
    ap.add_argument("--exclude", default="", help="수동 제외 csv (열: item, zh)")
    ap.add_argument("--min-correct", type=int, default=3)
    ap.add_argument("--out", default="results_placebo")
    a = ap.parse_args()

    df = pd.read_csv(a.data)
    pool = pd.read_csv(a.pool)
    C = pd.read_csv(a.c) if a.c and os.path.exists(a.c) else None
    extra = set()
    if a.exclude:
        E = pd.read_csv(a.exclude)
        extra = set(zip(E["item"], E["zh"]))
    ex, why = exclusions(df, pool, C, extra)
    os.makedirs(a.out, exist_ok=True)
    why.to_csv(os.path.join(a.out, "excluded_words.csv"), index=False, encoding="utf-8")
    for p in a.prefixes:
        analyze_one(p, df, pool, ex, os.path.join(a.out, os.path.basename(p)), a.min_correct)


if __name__ == "__main__":
    main()
