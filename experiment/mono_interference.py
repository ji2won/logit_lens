#!/usr/bin/env python3
"""단일 언어 문맥에서도 방해어 간섭이 생기는가 (데이터셋 A, 번역 없음).

  python -m experiment.mono_interference --model llama3.2-3b --device mps
  python -m experiment.mono_interference --model llama3.2-3b --analyze-only
  → results/<model>/A_en_run/mono_interference/{controls_zh.csv, raw.csv, items.csv, summary.md}

한→중 번역의 간섭 I(층 → 故事)가 '번역하면서 생긴 것'인지 '단어 자체에 붙은 연결'인지 가린다.
  zh  중국어 따라 쓰기: 입력 = 정답 중국어(楼层), 대조 = 대조어의 중국어(楼梯·屋顶).
      I_zh = log p(故事 | 楼层) − 평균 log p(故事 | 대조)
  ko  한국어 따라 쓰기: 입력 = 층, 대조 = 계단·지붕, 방해어 = 방해 뜻의 한국어(이야기).
      I_ko = log p(이야기 | 층) − 평균 log p(이야기 | 대조)
  tr  기준: 원래 한→중 번역 간섭 I (같은 코드로 다시 잰다)
대조어의 중국어는 모델의 한→중 번역(탐욕 생성, v1·v3 중 한자만으로 된 첫 결과)으로 정한다 → controls_zh.csv.
확인용(렌즈): 각 프롬프트 마지막 위치에서 중간층(7–21) 영어 다의어 확률 최댓값.
"""
from __future__ import annotations

import argparse
import math
import re

import numpy as np
import pandas as pd
import torch

from . import prompts
from .ablate_analysis import md_table
from .config import DATASETS, MID_LAYER_RANGE, MODELS, N_BOOT, RESULTS_DIR, SEED, VERSIONS
from .data import load_A
from .distractor_ablate import variants
from .lenses import VanillaLens, lens_logprobs
from .lm import LM

# 중국어 따라 쓰기 (prompts._REPEAT와 같은 형식, 전부 중국어)
_ZH_REPEAT = {
    "v1": ("", lambda s: (f"中文：{s}\n中文：", "")),
    "v2": ("请照抄下面的词。\n", lambda s: (f"{s}→", "")),
    "v3": ("请照抄下面的中文词。\n", lambda s: (f"中文：{s}\n照抄：", "")),
    "v4": ("", lambda s: (f"「{s}」再写一遍：「", "」")),
}
HAN = re.compile(r"^[一-鿿]+$")


def zh_repeat(version, word):
    head, fmt = _ZH_REPEAT[version]
    lines = [fmt(zh)[0] + zh + fmt(zh)[1] for _, zh, _ in prompts.FEWSHOT]
    return head + "\n".join(lines) + "\n" + fmt(word)[0]


@torch.no_grad()
def greedy(lm, prompt, n=6):
    ids = lm.encode(prompt)
    out = []
    for _ in range(n):
        _, lg = lm.run(ids + out)
        out.append(int(lg[-1].argmax()))
    return lm.decode(out)


def control_zh(lm, words, path):
    if path.exists():
        m = dict(pd.read_csv(path, keep_default_na=False)[["ko", "zh"]].values)
        if all(w in m for w in words):
            return m
    rows = []
    for w in sorted(set(words)):
        zh = ""
        for v in ("v1", "v3"):
            g = greedy(lm, prompts.build("translate", v, w))
            g = re.split(r"[\n」\s]", g)[0].strip()
            if HAN.match(g):
                zh = g
                break
        rows.append(dict(ko=w, zh=zh))
    pd.DataFrame(rows).to_csv(path, index=False)
    return {r["ko"]: r["zh"] for r in rows}


def run(args, out):
    items = load_A(DATASETS["A"])
    if args.limit:
        items = items[: args.limit]
    a = pd.read_csv(DATASETS["A"], keep_default_na=False).set_index("id")
    lm = LM.from_pretrained(MODELS[args.model], device=args.device)
    L = lm.n_layers
    layers = list(range(math.ceil(MID_LAYER_RANGE[0] * L), math.floor(MID_LAYER_RANGE[1] * L) + 1))
    print("대조어 중국어 번역", flush=True)
    czh = control_zh(lm, [c for it in items for c in it.controls], out / "controls_zh.csv")

    def score(prompt, cont, W=None):
        p = lm.encode(prompt)
        c = lm.encode(cont, special=False)
        hs, lg = lm.run(p + c)
        lp = torch.log_softmax(lg, -1)
        s = sum(lp[len(p) - 1 + i, t].item() for i, t in enumerate(c))
        lens = np.nan
        if W:
            P = lens_logprobs(lm, VanillaLens(), hs[:, len(p) - 1]).exp()
            lens = P[layers][:, W].sum(1).max().item()
        return s, lens

    rows = []
    for k, it in enumerate(items):
        W = variants(lm, it.en_hint)
        ko_d = a.loc[it.id, "ko_reverse_input"]
        czs = [czh.get(c, "") for c in it.controls]
        for v in VERSIONS:
            for task, inp, ctrls, ans, dist, build in [
                ("tr", it.ko, it.controls, it.zh_answer, it.zh_distractor, lambda w: prompts.build("translate", v, w)),
                ("zh", it.zh_answer, [c for c in czs if c and c != it.zh_answer], it.zh_answer, it.zh_distractor,
                 lambda w: zh_repeat(v, w)),
                ("ko", it.ko, it.controls, it.ko, ko_d, lambda w: prompts.build("repeat", v, w)),
            ]:
                if not ctrls:
                    continue
                ld, lens = score(build(inp), dist, W)
                la, _ = score(build(inp), ans)
                lc = [score(build(c), dist)[0] for c in ctrls]
                rows.append(dict(id=it.id, ko=it.ko, en=it.en_hint, zh=it.zh_answer, zh_d=it.zh_distractor, ko_d=ko_d,
                                 ctrl_zh=";".join(czs), version=v, task=task, lp_dist=ld, lp_ans=la,
                                 lp_dist_ctrl=float(np.mean(lc)), I=ld - float(np.mean(lc)), lens_W_mid=lens))
        if (k + 1) % 10 == 0 or k + 1 == len(items):
            print(f"  {k + 1}/{len(items)}", flush=True)
            pd.DataFrame(rows).to_csv(out / "raw.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "raw.csv", index=False)


def boot(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return f"{x.mean():.2f} [{np.percentile(m, 2.5):.2f}, {np.percentile(m, 97.5):.2f}]"


def analyze(out, model):
    from scipy.stats import spearmanr
    r = pd.read_csv(out / "raw.csv", keep_default_na=False)
    it = r.groupby(["id", "ko", "en", "zh", "zh_d", "ko_d", "ctrl_zh", "task"], as_index=False)[
        ["I", "lp_dist", "lp_ans", "lens_W_mid"]].mean()
    w = it.pivot_table(index=["id", "ko", "en", "zh", "zh_d", "ko_d", "ctrl_zh"], columns="task",
                       values=["I", "lens_W_mid", "lp_ans"]).reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    a = pd.read_csv(RESULTS_DIR.parent / "A_final.csv", keep_default_na=False).set_index("id")
    w["shared"] = [bool(set(z) & set(d)) for z, d in zip(w.zh, w.zh_d)]
    w["etym"] = a.loc[w.id, "etym_ko_input"].values
    w.to_csv(out / "items.csv", index=False)
    md = [f"# 단일 언어 문맥의 방해어 간섭 ({model})\n",
          f"문항 {len(w)}개 × 프롬프트 4개. I = log p(방해어|입력) − 평균 log p(방해어|대조) (nat). 문항 평균, 95% bootstrap CI.\n",
          "- tr: 한→중 번역 (층 → 楼层, 방해어 故事)\n- zh: 중국어 따라 쓰기 (楼层 → 楼层, 방해어 故事, 대조 = 대조어의 중국어)\n"
          "- ko: 한국어 따라 쓰기 (층 → 층, 방해어 = 방해 뜻 한국어 이야기)\n"]
    rows = []
    for t in ["tr", "zh", "ko"]:
        x = w[f"I_{t}"].dropna()
        rows.append(dict(과제=t, n=len(x), 평균_I=boot(x), I양수=f"{(x > 0).mean():.0%}",
                         정답_logp=f"{w[f'lp_ans_{t}'].mean():.2f}", 렌즈_중간층_영어=f"{w[f'lens_W_mid_{t}'].mean():.3f}"))
    md += ["## 1. 과제별 간섭\n", md_table(pd.DataFrame(rows)), ""]
    rows = []
    for name, g in [("전체", w), ("글자 공유 제외", w[~w.shared]), ("한→중 I≥5", w[w.I_tr >= 5]),
                    ("한→중 I≥5, 공유 제외", w[(w.I_tr >= 5) & ~w.shared])]:
        g = g.dropna(subset=["I_tr", "I_zh", "I_ko"])
        rz, pz = spearmanr(g.I_tr, g.I_zh)
        rk, pk = spearmanr(g.I_tr, g.I_ko)
        rows.append(dict(문항=name, n=len(g), I_tr=f"{g.I_tr.mean():.2f}", I_zh=f"{g.I_zh.mean():.2f}", I_ko=f"{g.I_ko.mean():.2f}",
                         ρ_tr_zh=f"{rz:.2f} (p={pz:.3g})", ρ_tr_ko=f"{rk:.2f} (p={pk:.3g})"))
    md += ["## 2. 번역 간섭과 단일 언어 간섭의 관계\n", md_table(pd.DataFrame(rows)), ""]
    top = w.sort_values("I_tr", ascending=False).head(20)[
        ["ko", "en", "zh", "zh_d", "ko_d", "ctrl_zh", "shared", "I_tr", "I_zh", "I_ko", "lens_W_mid_tr", "lens_W_mid_zh"]]
    md += ["## 3. 한→중 간섭이 큰 문항 20개\n", md_table(top.round(2)), ""]
    (out / "summary.md").write_text("\n".join(md), "utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = RESULTS_DIR / args.model / "A_en_run" / "mono_interference"
    out.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        run(args, out)
    analyze(out, args.model)


if __name__ == "__main__":
    main()
