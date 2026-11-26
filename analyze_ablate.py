"""run_ablate.py 결과 분석 (CPU). 문항 안 비교: spring 방향을 뺀 효과 − 같은 문항의 무관 영어 방향을 뺀 효과.
 
지표 (문항별로 템플릿 4개 평균을 먼저 낸 뒤)
  Δ        : log p(단어 | 개입) − log p(단어 | clean), 같은 배치 기준. 단어 = 정답(ans) 또는 방해어(dis)
  eff      : Δ_target − mean_k Δ_ctrl_k            (음수 = spring 방향을 뺐을 때 대조보다 더 떨어짐)
  pct      : 대조 K개 중 Δ_ctrl < Δ_target 인 비율 (+동률 0.5). 귀무가설이면 평균 0.5, 작을수록 target이 더 떨어짐
  sel      : (Δdis − Δans)_target − mean_k (Δdis − Δans)_ctrl_k   (음수 = 방해어가 정답보다 더 선택적으로 떨어짐)
  조작 확인 : read 층 logit lens에서 영어 다의어 토큰 log p의 변화 (target vs 대조)
검정: 문항 간 Wilcoxon 부호순위(eff=0, pct=0.5), 평균의 95% bootstrap CI.
 
    python analyze_ablate.py ablate/llama --out results_ablate/llama [--items results_placebo/llama/items_placebo.csv]
"""
import argparse
import json
import os
 
import numpy as np
import pandas as pd
from scipy import stats
 
 
def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return [np.nan, np.nan]
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]
 
 
def wil(x, mu=0.0):
    x = np.asarray(x, float) - mu
    x = x[~np.isnan(x) & (x != 0)]
    if len(x) < 5:
        return np.nan
    return float(stats.wilcoxon(x).pvalue)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix", help="run_ablate.py --out 접두어")
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default="", help="(선택) items_placebo.csv: 문항별 간섭 I와의 관계를 탐색적으로 봄")
    ap.add_argument("--min-correct", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
 
    R = pd.read_csv(a.prefix + "_rows.csv.gz")
    Lz = pd.read_csv(a.prefix + "_lens.csv.gz")
    M = pd.read_csv(a.prefix + "_prompts.csv")
    chk = json.load(open(a.prefix + "_check.json"))
    ctl = pd.read_csv(a.prefix + "_controls.csv")
 
    ncor = M[M.cond == "input"].groupby("item").correct.sum()
    M["n_correct"] = M.item.map(ncor)
    R = R[R.role != "clean"].merge(M[["pid", "item", "en", "template", "cond", "n_correct"]], on="pid")
 
    # 템플릿 평균
    T = (R.groupby(["item", "en", "cond", "n_correct", "dirtype", "config", "kind", "role", "dir_word"])
         [["delta", "pert"]].mean().reset_index())
 
    # 문항별 효과
    recs = []
    keys = ["item", "en", "cond", "n_correct", "dirtype", "config"]
    for k, g in T.groupby(keys):
        row = dict(zip(keys, k))
        d = {}
        for kind in ["ans", "dis"]:
            gk = g[g.kind == kind]
            tgt = gk[gk.role == "target"]
            if tgt.empty:
                continue
            t = tgt.delta.iloc[0]
            row[f"d_target_{kind}"] = t
            row[f"pert_target"] = tgt.pert.iloc[0]
            for role in ["ctrl_ablate", "ctrl_matched"]:
                c = gk[gk.role == role].set_index("dir_word").delta
                tag = role.split("_")[1]
                row[f"d_{tag}_{kind}"] = c.mean()
                row[f"eff_{tag}_{kind}"] = t - c.mean()
                row[f"pct_{tag}_{kind}"] = float((c < t).mean() + 0.5 * (c == t).mean())
                d[(role, kind)] = c
                if kind == "ans":
                    row[f"pert_{tag}"] = gk[gk.role == role].pert.mean()
        for role in ["ctrl_ablate", "ctrl_matched"]:
            tag = role.split("_")[1]
            if (role, "ans") in d and (role, "dis") in d:
                cs = (d[(role, "dis")] - d[(role, "ans")])
                ts = row["d_target_dis"] - row["d_target_ans"]
                row[f"sel_{tag}"] = ts - cs.mean()
                row[f"selpct_{tag}"] = float((cs < ts).mean() + 0.5 * (cs == ts).mean())
        recs.append(row)
    E = pd.DataFrame(recs)
 
    # 조작 확인: lens에서 영어 다의어 log p
    Lz = Lz.merge(M[["pid", "item", "cond", "n_correct"]], on="pid")
    clean = Lz[Lz.role == "clean"].groupby("pid").lens_lp_target.first()  # 배치마다 하나씩 기록됨(값 같음)
    Lz["dlens_target"] = Lz.lens_lp_target - Lz.pid.map(clean)
    Lm = (Lz[Lz.role != "clean"].groupby(["item", "cond", "n_correct", "dirtype", "config", "role"])
          .dlens_target.mean().unstack("role").reset_index())
    Lm["lens_clean"] = Lz[Lz.role == "clean"].groupby("item").lens_lp_target.mean().reindex(Lm.item).values
 
    E.to_csv(os.path.join(a.out, "items_ablate.csv"), index=False, encoding="utf-8")
    Lm.to_csv(os.path.join(a.out, "lens_check.csv"), index=False, encoding="utf-8")
 
    subsets = {f"정답≥{a.min_correct}/4": lambda x: x[x.n_correct >= a.min_correct], "전체": lambda x: x}
    summ = dict(check={k: v for k, v in chk.items() if k != "clean_vs_single"}, tables={})
    lines = [f"# 인과 개입 결과: {chk['model']}", "",
             "정합성 검사: " + ", ".join(f"{k}={chk[k]:.4g}" if isinstance(chk[k], float) else f"{k}={chk[k]}"
                                    for k in ["max_clean_within_batch", "max_rerun", "max_clean_vs_single", "max_resid_after_ablate", "ok"]),
             "- max_clean_within_batch·max_rerun은 0이어야 한다(같은 배치 안 비교가 잡음 없이 됨).",
             "- max_clean_vs_single은 배치 크기가 다른 계산과의 bf16 수치 차이(참고용. 효과는 같은 배치 안에서만 잰다).", "",
             f"대조 단어: 문항당 {ctl.groupby('item').size().iloc[0]}개, 영어 빈도 차 평균 "
             f"{(ctl.zipf_ctrl - ctl.zipf_en).abs().mean():.3f} zipf. "
             + ", ".join(f"cos({c[4:]}) 평균 {ctl[c].mean():.3f}" for c in ctl.columns if c.startswith("cos_")), ""]
 
    def fmt(m, ci, p):
        return f"{m:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] p={p:.2g}"
 
    for sname, sf in subsets.items():
        for cond in sorted(E.cond.unique(), key=lambda c: c != "input"):
            X = sf(E[E.cond == cond])
            if X.empty:
                continue
            n = X.item.nunique()
            lines += [f"## {sname}, 조건 {cond} ({'한국어 입력, 주 실험' if cond == 'input' else '영어 단어 직접 입력, 양성 대조'}), 문항 {n}개", "",
                      "효과 = Δ(target) − 평균 Δ(대조), nats. 음수면 spring 방향을 뺄 때 대조보다 더 떨어짐. pct는 0.5가 귀무.", "",
                      "| 방향 | 설정 | Δ방해어 target | eff 방해어 (ablate 대조) | eff 방해어 (크기 맞춘 대조) | pct 방해어 (크기 맞춤) | Δ정답 target | eff 정답 (크기 맞춘 대조) | sel (방해어−정답, 크기 맞춤) | ‖Δh‖ target / ablate 대조 |",
                      "|---|---|---|---|---|---|---|---|---|---|"]
            for (dt, cf), g in X.groupby(["dirtype", "config"]):
                r = {}
                for c in ["eff_ablate_dis", "eff_matched_dis", "eff_matched_ans", "sel_matched", "d_target_dis", "d_target_ans"]:
                    r[c] = (g[c].mean(), boot_ci(g[c]), wil(g[c]))
                pc = (g.pct_matched_dis.mean(), boot_ci(g.pct_matched_dis), wil(g.pct_matched_dis, 0.5))
                lines.append(f"| {dt} | {cf} | {g.d_target_dis.mean():+.3f} | {fmt(*r['eff_ablate_dis'])} | {fmt(*r['eff_matched_dis'])} | "
                             f"{pc[0]:.2f} p={pc[2]:.2g} | {g.d_target_ans.mean():+.3f} | {fmt(*r['eff_matched_ans'])} | {fmt(*r['sel_matched'])} | "
                             f"{g.pert_target.mean():.1f} / {g.pert_ablate.mean():.1f} |")
                summ["tables"][f"{sname}|{cond}|{dt}|{cf}"] = dict(
                    n=int(n), **{k: dict(mean=v[0], ci=v[1], p=v[2]) for k, v in r.items()},
                    pct_matched_dis=dict(mean=pc[0], ci=pc[1], p=pc[2]),
                    pert_target=float(g.pert_target.mean()), pert_ablate=float(g.pert_ablate.mean()))
            lines.append("")
            # 조작 확인
            Y = sf(Lm[Lm.cond == cond])
            lines += ["조작 확인 (read 층 logit lens, 영어 다의어 토큰 log p 변화, 템플릿·문항 평균)", "",
                      "| 방향 | 설정 | target | ablate 대조 | 크기 맞춘 대조 |", "|---|---|---|---|---|"]
            for (dt, cf), g in Y.groupby(["dirtype", "config"]):
                lines.append(f"| {dt} | {cf} | {g.target.mean():+.2f} | {g.ctrl_ablate.mean():+.2f} | {g.ctrl_matched.mean():+.2f} |")
            lines.append("")
 
    if a.items and os.path.exists(a.items):
        It = pd.read_csv(a.items)[["item", "I_real", "I_adj"]]
        X = E[(E.cond == "input") & (E.n_correct >= a.min_correct)].merge(It, on="item")
        lines += [f"## 탐색: 문항별 간섭 I와 개입 효과 (정답≥{a.min_correct}/4, input)", "",
                  "| 방향 | 설정 | Spearman(I_adj, eff 방해어 크기 맞춤) | n |", "|---|---|---|---|"]
        for (dt, cf), g in X.groupby(["dirtype", "config"]):
            rr = stats.spearmanr(g.I_adj, g.eff_matched_dis)
            lines.append(f"| {dt} | {cf} | {rr.statistic:+.2f} (p={rr.pvalue:.2g}) | {len(g)} |")
        lines.append("")
 
    open(os.path.join(a.out, "report_ablate.md"), "w", encoding="utf-8").write("\n".join(lines))
    json.dump(summ, open(os.path.join(a.out, "summary_ablate.json"), "w"), ensure_ascii=False, indent=1, default=float)
    print("\n".join(lines))
 
 
if __name__ == "__main__":
    main()
 