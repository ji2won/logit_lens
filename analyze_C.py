"""데이터셋 C(일반 단어) 분석: 가설 ①(평균 성분) + 영어 편향 지수 B + 반복 과제.
 
    python analyze_C.py cache/llama_C.pt cache/exaone_chat_C.pt --data ../dataset_C/C_final.csv \
        --wiki-from cache/llama.pt,cache/exaone_chat.pt --affine-langs ko,zh,en --out resultsC
 
지표 (lens별, 프롬프트 마지막 토큰, 중간 층 ℳ 평균 log 확률)
  B_zh  = lp(무관 영어 culture) − lp(무관 중국어 文化)     같은 무관 개념을 영어로 읽을지 중국어로 읽을지.
  B_ko  = lp(무관 영어 culture) − lp(무관 한국어 문화)     입력과 상관없는 개념이므로 0보다 크면
                                                          '무엇을 넣든 영어 쪽으로 기우는' 편향(가설 ①의 대상).
  Spec  = lp(sea | 입력 바다) − lp(sea | 무관 입력)        같은 영어 토큰을 관련 입력과 무관 입력에서 읽은 차이.
                                                          토큰 자체의 사전 확률이 상쇄되므로 '진짜 개념 신호'.
  무관 입력은 다른 문항의 입력어다(C의 무관 단어는 문항끼리의 순열). 그래서 별도 프롬프트 없이 계산된다.
 
판정 (가설 ①)
  logit lens에서 B > 0 이고, 평균 제거 lens에서 B가 줄면(Δ CI 상한 < 0) → 평균 성분이 영어를 부풀림(지지)
  평균 제거 lens에서 B가 안 줄면 → 평균 성분은 원인이 아님(불리)
  어떤 보정에서도 Spec은 유지되어야 정상(관련 영어 신호까지 지우면 보정이 신호를 지운 것)
"""
import argparse
import json
import math
import os
import sys
 
import numpy as np
import pandas as pd
import torch
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import ci, run_lenses, sanity, top_tokens_mid  # noqa: E402
from prompts import SHOTS_A, SHOTS_B  # noqa: E402
 
EXTRA_COLS = {"enU": "en_unrelated", "zhU": "zh_unrelated", "koU": "ko_unrelated", "zhR": "zh"}
 
 
def extra_arrays(C, df):
    meta = C["meta"]
    row = df.set_index("id")
    out = {}
    for name, col in EXTRA_COLS.items():
        table = C["extra_ids"][col]
        words = meta.item.map(row[col]).values
        lists = [table[w]["ids"] for w in words]
        k = max(len(x) for x in lists)
        out[name] = (np.array([(x + [x[0]] * k)[:k] for x in lists]), np.array([[j < len(x) for j in range(k)] for x in lists]))
    return out
 
 
def boot_mean(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan"), [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return float(x.mean()), ci(b)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--data", default="../dataset_C/C_final.csv")
    ap.add_argument("--out", default="resultsC")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wiki-from", default="")
    ap.add_argument("--affine-langs", default="")
    ap.add_argument("--no-tuned", action="store_true")
    ap.add_argument("--min-correct", type=int, default=3, help="'정답 문항만' 집합의 기준 (템플릿 4개 중)")
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.data)
    shot_ko = {k for k, _ in SHOTS_A + SHOTS_B}
    shot_zh = {z for _, z in SHOTS_A + SHOTS_B}
    # few-shot 예시와 겹치는 문항: 자기 단어가 겹치면 정답이 프롬프트에 있고, 무관 단어가 겹치면 B가 부풀려짐
    df["fewshot_self"] = df.ko.isin(shot_ko) | df.zh.isin(shot_zh)
    df["fewshot_unrel"] = df.ko_unrelated.isin(shot_ko) | df.zh_unrelated.isin(shot_zh)
    ko2id = dict(zip(df.ko, df.id))
    df["unrel_item"] = df.ko_unrelated.map(ko2id)
    from run_cache import is_correct
 
    for path in a.caches:
        C = torch.load(path, weights_only=False)
        if a.wiki_from:
            src = a.wiki_from.split(",")[a.caches.index(path)]
            W = torch.load(src, weights_only=False, map_location="cpu")
            assert W["model"] == C["model"], f"--wiki-from 모델이 다름: {W['model']} vs {C['model']}"
            for k in ["wiki_mu", "wiki_mu_by_lang", "wiki_X", "wiki_X_lang", "wiki_n_mu_tokens"]:
                if k in W:
                    C[k] = W[k]
            del W
            print("위키 부분을 가져옴:", src, flush=True)
        meta = C["meta"]
        has = meta.gen.notna()
        meta.loc[has, "correct"] = [is_correct(g, x) for g, x in zip(meta.gen[has], meta.answer[has])]
        print(os.path.basename(path), "정답률:", meta[has].groupby("task").correct.mean().round(3).to_dict(), flush=True)
        L = C["n_layers"]
        mids = list(range(math.ceil(0.25 * L), math.floor(0.75 * L) + 1))
        layers = sorted(set(range(L + 1)))
        R, lens_names, aff = run_lenses(C, a.device, layers, mids, "wiki",
                                        [x for x in a.affine_langs.split(",") if x] or None, None,
                                        tuned=not a.no_tuned, extra=extra_arrays(C, df))
        sd = sanity(C, R, a.device)
        name = os.path.basename(path).rsplit(".", 1)[0]
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        R = R.merge(meta[["pid", "item", "task", "template", "cond", "correct"]], on="pid")
        R["p_lang"] = R.p_latin + R.p_han + R.p_hangul
        cols = ["E", "p_lang", "kl", "en_lp", "enU_lp", "zhU_lp", "koU_lp", "zhR_lp"]
        R.groupby(["task", "lens", "layer"])[cols].mean().reset_index().to_csv(os.path.join(d, "layer_C.csv"), index=False)
 
        Rm = R[R.layer.isin(mids)]
        P = Rm.groupby(["task", "lens", "pid", "item", "template"])[cols].mean().reset_index()
        P["B_zh"] = P.enU_lp - P.zhU_lp
        P["B_ko"] = P.enU_lp - P.koU_lp
        I = P.groupby(["task", "lens", "item"])[cols + ["B_zh", "B_ko"]].mean().reset_index()
        # Spec: 문항 Y의 영어 토큰 en_Y를, Y 입력(관련)에서 읽은 값 − X 입력(Y가 X의 무관 단어)에서 읽은 값
        rel = I.set_index(["task", "lens", "item"]).en_lp
        unr = I.assign(item=I.item.map(dict(zip(df.id, df.unrel_item)))).set_index(["task", "lens", "item"]).enU_lp
        unr = unr[~unr.index.duplicated()]
        I["Spec"] = (rel - unr.reindex(rel.index)).values
        ok = meta[(meta.task == "translate") & (meta.cond == "input")].groupby("item").correct.sum()
        I = I.merge(df[["id", "etym", "sino_same", "fewshot_self", "fewshot_unrel", "unrel_item"]], left_on="item", right_on="id")
        I["n_correct"] = I.item.map(ok)
        I["zhU_whole"] = I.item.map(dict(zip(df.id, [C["extra_ids"]["zh_unrelated"][w]["whole"] for w in df.zh_unrelated])))
        I["koU_whole"] = I.item.map(dict(zip(df.id, [C["extra_ids"]["ko_unrelated"][w]["whole"] for w in df.ko_unrelated])))
        bad = set(df.id[df.fewshot_self])
        inv = dict(zip(df.unrel_item, df.id))  # Y → X (X의 무관 단어가 Y). Y의 Spec은 X 프롬프트를 쓴다
        I["spec_ok"] = ~I.item.isin(bad) & ~I.item.map(inv).isin(bad)
        I.to_csv(os.path.join(d, "items_C.csv"), index=False, encoding="utf-8-sig")
 
        clean = ~I.fewshot_self & ~I.fewshot_unrel
        sets = {"전체(few-shot 겹침 제외)": clean, f"정답 {a.min_correct}/4 이상": clean & (I.n_correct >= a.min_correct)}
        lines = [f"# {C['model']} — 데이터셋 C", "", f"중간 층 ℳ = {mids[0]}–{mids[-1]}", "",
                 "sanity: " + json.dumps(sd, ensure_ascii=False), "",
                 f"문항 {df.id.nunique()}개 중 few-shot 겹침 제외: 자기 단어 {int(df.fewshot_self.sum())}, 무관 단어 {int(df.fewshot_unrel.sum())}", ""]
        summ = dict(model=C["model"], sanity=sd, mids=mids)
        for task in ["translate", "repeat"]:
            for sname, m in sets.items():
                if task == "repeat" and sname != "전체(few-shot 겹침 제외)":
                    continue
                X = I[(I.task == task) & m]
                lines += [f"## {task} · {sname} (문항 {X.item.nunique()})", "",
                          "| lens | B_zh [95% CI] | B_ko [95% CI] | Spec [95% CI] | E | 언어 질량 | KL |",
                          "|---|---|---|---|---|---|---|"]
                base = X[X.lens == "logit"].set_index("item")
                res = {}
                for ln in lens_names:
                    Y = X[X.lens == ln].set_index("item")
                    row = {}
                    for met in ["B_zh", "B_ko", "Spec"]:
                        v = Y[met] if met != "Spec" else Y.loc[Y.spec_ok, met]
                        row[met] = boot_mean(v, a.nboot)
                        if ln != "logit":
                            dv = (Y[met] - base[met].reindex(Y.index))
                            if met == "Spec":
                                dv = dv[Y.spec_ok]
                            row["d_" + met] = boot_mean(dv, a.nboot)
                    row.update(E=float(Y.E.mean()), p_lang=float(Y.p_lang.mean()), kl=float(Y.kl.mean()))
                    res[ln] = row
 
                    def fm(t):
                        return f"{t[0]:.2f} [{t[1][0]:.2f}, {t[1][1]:.2f}]"
                    lines.append(f"| {ln} | {fm(row['B_zh'])} | {fm(row['B_ko'])} | {fm(row['Spec'])} | "
                                 f"{row['E']:.3f} | {row['p_lang']:.3f} | {row['kl']:.2f} |")
                lines += ["", "logit lens 대비 변화 Δ [95% CI] (음수 = 보정 후 감소)", "",
                          "| lens | Δ B_zh | Δ B_ko | Δ Spec |", "|---|---|---|---|"]
                for ln in lens_names[1:]:
                    r = res[ln]
                    lines.append("| " + ln + " | " + " | ".join(
                        f"{r['d_' + k][0]:.2f} [{r['d_' + k][1][0]:.2f}, {r['d_' + k][1][1]:.2f}]" for k in ["B_zh", "B_ko", "Spec"]) + " |")
                # 글자 조각 첫 토큰 문제 점검: 무관 중국어/한국어 첫 토큰이 온전한 글자인 문항만
                w = X[(X.lens.isin(["logit", "mean_wiki", "tuned"])) & X.zhU_whole & X.koU_whole]
                if len(w):
                    lines += ["", f"점검: 무관 단어 첫 토큰이 온전한 글자인 문항만 ({w.item.nunique()}문항) B_zh / B_ko 평균: " +
                              ", ".join(f"{ln} {g.B_zh.mean():.2f} / {g.B_ko.mean():.2f}" for ln, g in w.groupby("lens"))]
                # 한자어 지름길: 한국어 한자어가 중국어와 같은 한자인 문항(sino_same=Y) vs 아닌 문항
                g = X[X.lens.isin(["logit", "tuned"])].groupby(["sino_same", "lens"])[["B_zh", "Spec", "E"]].mean()
                lines += ["", "sino_same(한국어 한자어 = 중국어 같은 한자)별 평균:", "", "```\n" + g.round(3).to_string() + "\n```", ""]
                summ[f"{task}|{sname}"] = res
        tt = top_tokens_mid(C, R, mids)
        lines += ["## 중간 층 판독 1위 토큰 (번역)", ""] + [f"- {ln}: " + ", ".join(f"{t!r}×{n}" for t, n in v) for ln, v in tt.items()]
        open(os.path.join(d, "report_C.md"), "w", encoding="utf-8").write("\n".join(lines))
        json.dump(summ, open(os.path.join(d, "summary_C.json"), "w"), ensure_ascii=False, indent=1, default=str)
        print("저장:", d, flush=True)
 
 
if __name__ == "__main__":
    main()
 