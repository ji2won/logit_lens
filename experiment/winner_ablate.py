#!/usr/bin/env python3
"""렌즈가 중간층에서 1위로 읽는 영어 단어(그 문항의 '영어 승자')를 지워도 최종 출력이 바뀌지 않는가 (A·C 전체, 한→중).

  python -m experiment.winner_ablate --model llama3.2-3b --device mps
  python -m experiment.winner_ablate --model llama3.2-3b --analyze-only
  → results/<model>/A_en_run/winner_ablate/{raw.csv, lens_after.csv, summary.md}

1단계 (깨끗한 실행): 문항×프롬프트마다 16–24층 렌즈 1위 토큰 중 영어 단어(알파벳 3자 이상)를 모으고,
         그중 확률이 가장 큰 것을 '영어 승자'로 고른다. 영어 1위가 한 번도 없으면 승자 없음.
2단계 (제거): 승자의 1토큰 변형(단·복수 × 앞 공백 × 대소문자)의 γ⊙U 방향을 중국어 출력 토큰
         (정답 토큰 + 깨끗한 최종 상위 20)에 직교화한 '영어 고유 조각'을 7–21층 블록 출력, 프롬프트 위치에서 지운다.
  W_orth      그 승자
  other_orth  같은 데이터셋 다른 문항의 승자 (같은 직교화 + 자기 승자에 직교, 차원 맞춤)
  rand_orth   무작위 방향 (같은 직교화, 같은 차원)
측정(렌즈 없음): 정답 전체 log p (teacher forcing), 최종 1위 유지 여부, KL(깨끗한 ‖ 개입 후)
확인(렌즈): 제거 후 16–24층 1위 토큰 → lens_after.csv
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
from .data import load_A, load_C
from .distractor_ablate import LayerAblator, onb, orth, variants
from .lenses import VanillaLens, lens_logprobs
from .lm import LM

WIN_LAYERS = range(16, 25)
CONDS = ["W_orth", "other_orth", "rand_orth"]
_EN = re.compile(r"^[A-Za-z]{3,}$")


def run(args, out):
    items = [("A", it) for it in load_A(DATASETS["A"])] + [("C", it) for it in load_C(DATASETS["C"])]
    if args.limit:
        items = [x for x in items if x[0] == "A"][: args.limit] + [x for x in items if x[0] == "C"][: args.limit]
    lm = LM.from_pretrained(MODELS[args.model], device=args.device)
    abl = LayerAblator(lm)
    L = lm.n_layers
    layers = list(range(math.ceil(MID_LAYER_RANGE[0] * L), math.floor(MID_LAYER_RANGE[1] * L) + 1))
    g = lm.norm.weight.float().cpu()
    Uw = lm.unembed.weight
    gu = lambda ids: Uw[list(ids)].float().cpu() * g
    gen = torch.Generator().manual_seed(SEED)
    d = Uw.shape[1]

    def fwd(p_ids, a_ids, Q):
        abl.Qs = {l: Q.to(lm.device) for l in layers} if Q is not None else {}
        abl.n_prompt = len(p_ids)
        hs, lg = lm.run(p_ids + a_ids)
        abl.Qs = {}
        lp = torch.log_softmax(lg, -1)
        n = len(p_ids)
        ans = sum(lp[n - 1 + i, t].item() for i, t in enumerate(a_ids))
        P = lens_logprobs(lm, VanillaLens(), abl.stack(hs)).exp()
        return lp[n - 1], ans, P

    # 1단계: 깨끗한 실행, 승자 고르기
    clean = {}
    for k, (ds, it) in enumerate(items):
        a_ids = lm.encode(it.zh_answer, special=False)
        for v in VERSIONS:
            p_ids = lm.encode(prompts.build("translate", v, it.ko))
            lp, ans, P = fwd(p_ids, a_ids, None)
            best, bp = None, 0.0
            for l in WIN_LAYERS:
                i = int(P[l].argmax())
                s = lm.decode([i]).strip()
                if _EN.match(s) and P[l, i].item() > bp:
                    best, bp = s, P[l, i].item()
            clean[(it.id, v)] = dict(p_ids=p_ids, a_ids=a_ids, lp=lp.cpu(), ans=ans, P=P.cpu(), win=best, win_p=bp)
        if (k + 1) % 50 == 0:
            print(f"  clean {k + 1}/{len(items)}", flush=True)

    win_ids = {key: (variants(lm, c["win"]) if c["win"] else []) for key, c in clean.items()}
    by_ds = {ds: [it.id for d_, it in items if d_ == ds] for ds in ("A", "C")}
    rows, after = [], []
    for k, (ds, it) in enumerate(items):
        pool = by_ds[ds]
        for v in VERSIONS:
            c = clean[(it.id, v)]
            base = dict(dataset=ds, id=it.id, ko=it.ko, zh=it.zh_answer, version=v, winner=c["win"] or "",
                        win_p=c["win_p"], lp_ans_clean=c["ans"], top_clean=lm.decode([int(c["lp"].argmax())]))
            if not c["win"]:
                rows.append(dict(base, cond="none"))
                continue
            W = win_ids[(it.id, v)]
            against = sorted((set(c["a_ids"]) | set(c["lp"].topk(20).indices.tolist())) - set(W))
            A = gu(against)
            Qw = onb(orth(gu(W), A))
            r = Qw.shape[1]
            # 다른 문항의 승자: 같은 데이터셋에서 반 바퀴 떨어진 곳부터 승자가 있고 단어가 다른 첫 문항
            j0 = pool.index(it.id)
            oW = []
            for s in range(1, len(pool)):
                o = clean[(pool[(j0 + len(pool) // 2 + s) % len(pool)], v)]
                if o["win"] and o["win"].lower() != c["win"].lower():
                    oW = [i for i in variants(lm, o["win"]) if i not in set(W)]
                    break
            Qo = onb(orth(gu(oW), torch.cat([A, gu(W)])), rank=r) if oW else None
            Qr = onb(orth(torch.randn(r, d, generator=gen), A))
            lp0 = c["lp"].to(lm.device)
            top0 = int(lp0.argmax())
            for cond, Q in [("W_orth", Qw), ("other_orth", Qo), ("rand_orth", Qr)]:
                if Q is None or Q.shape[1] == 0:
                    continue
                lp, ans, P = fwd(c["p_ids"], c["a_ids"], Q)
                rows.append(dict(base, cond=cond, rank=Q.shape[1], lp_ans=ans, d_lp_ans=ans - c["ans"],
                                 kl=(lp0.exp() * (lp0 - lp)).sum().item(), top1_same=int(lp.argmax()) == top0,
                                 top_after=lm.decode([int(lp.argmax())]),
                                 lens_win_mid=P[list(WIN_LAYERS)][:, W].sum(1).max().item(),
                                 lens_win_mid_clean=c["P"][list(WIN_LAYERS)][:, W].sum(1).max().item()))
                if cond == "W_orth":
                    for l in WIN_LAYERS:
                        for when, PP in (("clean", c["P"]), ("removed", P.cpu())):
                            i = int(PP[l].argmax())
                            after.append(dict(dataset=ds, id=it.id, ko=it.ko, zh=it.zh_answer, version=v, winner=c["win"],
                                              layer=l, when=when, top1=lm.decode([i]), p=PP[l, i].item(),
                                              p_ans_first=PP[l, c["a_ids"][0]].item()))
        if (k + 1) % 25 == 0 or k + 1 == len(items):
            print(f"  ablate {k + 1}/{len(items)}", flush=True)
            pd.DataFrame(rows).to_csv(out / "raw.csv", index=False)
            pd.DataFrame(after).to_csv(out / "lens_after.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "raw.csv", index=False)
    pd.DataFrame(after).to_csv(out / "lens_after.csv", index=False)


def boot(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return "–"
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return f"{x.mean():.3f} [{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]"


def lang(s):
    s = s.strip()
    if re.search(r"[一-鿿]", s) or "�" in s:
        return "중국어"
    if re.fullmatch(r"[A-Za-z]+", s):
        return "영어"
    return "기타"


def analyze(out, model):
    r = pd.read_csv(out / "raw.csv", keep_default_na=False, low_memory=False)
    for c in ["d_lp_ans", "kl", "lens_win_mid", "lens_win_mid_clean", "win_p"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r["top1_same"] = r.top1_same.astype(str).eq("True")
    md = [f"# 렌즈 1위 영어(영어 승자)를 지우면 최종 출력이 바뀌나 ({model}, 한→중, 렌즈 없음)\n",
          "승자 = 16–24층 렌즈 1위 중 확률이 가장 큰 영어 단어. 제거 = 승자의 영어 고유 조각(중국어 출력 토큰에 직교화), 7–21층, 프롬프트 위치.\n"]
    cov = []
    for ds, g in r[r.cond.isin(["none", "W_orth"])].groupby("dataset"):
        cov.append(dict(데이터셋=ds, 문항x프롬프트=len(g), 영어_승자_있음=f"{(g.cond == 'W_orth').mean():.0%}",
                        승자_평균확률=f"{g[g.cond == 'W_orth'].win_p.mean():.2f}"))
    md += ["## 0. 영어가 1위인 경우가 얼마나 되나\n", md_table(pd.DataFrame(cov)), ""]
    w = r[r.cond != "none"]
    u = w.groupby(["dataset", "id", "cond"], as_index=False)[["d_lp_ans", "kl", "top1_same", "lens_win_mid", "lens_win_mid_clean"]].mean()
    rows = []
    for ds in ["A", "C", "전체"]:
        for cond in CONDS:
            g = u[(u.cond == cond) & ((u.dataset == ds) | (ds == "전체"))]
            rows.append(dict(데이터셋=ds, 조건=cond, 문항=len(g), 렌즈_승자_전=f"{g.lens_win_mid_clean.mean():.3f}",
                             렌즈_승자_후=f"{g.lens_win_mid.mean():.3f}", Δlogp_정답=boot(g.d_lp_ans), KL=boot(g.kl),
                             최종1위_유지=f"{g.top1_same.mean():.1%}"))
    md += ["## 1. 조건별 (문항 단위 평균, 95% bootstrap CI)\n", md_table(pd.DataFrame(rows)), ""]
    p = u.pivot_table(index=["dataset", "id"], columns="cond", values=["d_lp_ans", "kl"]).reset_index()
    rows = []
    for ds in ["A", "C", "전체"]:
        g = p if ds == "전체" else p[p.dataset == ds]
        for b in ["other_orth", "rand_orth"]:
            rows.append(dict(데이터셋=ds, 비교=f"W_orth − {b}", n=int(g[("d_lp_ans", b)].notna().sum()),
                             ΔΔlogp_정답=boot(g[("d_lp_ans", "W_orth")] - g[("d_lp_ans", b)]),
                             ΔKL=boot(g[("kl", "W_orth")] - g[("kl", b)])))
    md += ["## 2. 짝 비교 (음수 = 승자를 지웠을 때 기준선보다 정답이 더 떨어짐)\n", md_table(pd.DataFrame(rows)), ""]
    x = w[w.cond == "W_orth"]
    bins = [(-np.inf, -1), (-1, -0.3), (-0.3, -0.1), (-0.1, 0.1), (0.1, np.inf)]
    lab = ["< −1 (크게 떨어짐)", "−1 ~ −0.3", "−0.3 ~ −0.1", "−0.1 ~ 0.1 (거의 그대로)", "> 0.1 (올라감)"]
    rows = []
    for ds in ["A", "C"]:
        gg = x[x.dataset == ds]
        row = {"데이터셋": ds, "n": len(gg), "최종1위_유지": f"{gg.top1_same.mean():.0%}"}
        for (lo, hi), l in zip(bins, lab):
            row[l] = f"{((gg.d_lp_ans > lo) & (gg.d_lp_ans <= hi)).mean():.0%}"
        rows.append(row)
    md += ["## 3. 문항×프롬프트별 정답 변화 분포 (W_orth, nat)\n", md_table(pd.DataFrame(rows)), ""]
    big = x.sort_values("d_lp_ans").head(12)[["dataset", "ko", "zh", "version", "winner", "d_lp_ans", "top_clean", "top_after"]]
    md += ["## 4. 정답이 가장 많이 떨어진 경우\n", md_table(big.round(2)), ""]
    a = pd.read_csv(out / "lens_after.csv", keep_default_na=False)
    a["lang"] = a.top1.map(lang)
    a["is_win"] = [t.strip().lower()[:4] == wn.lower()[:4] for t, wn in zip(a.top1, a.winner)]
    a["cat"] = np.where(a.is_win, "승자 영어", a.lang.map({"영어": "다른 영어", "중국어": "중국어", "기타": "기타"}))
    t = (pd.crosstab([a.dataset, a.when], a.cat, normalize="index") * 100).round(0).reset_index()
    md += ["## 5. 렌즈 16–24층 1위 구성 (%): 승자를 지우면 무엇이 1위가 되나\n", md_table(t), ""]
    rows = []
    for ds in ["A", "C"]:
        for when in ["clean", "removed"]:
            g = a[(a.dataset == ds) & (a.when == when)]
            rows.append(dict(데이터셋=ds, 상태=when, **{f"L{l}": f"{g[g.layer == l].p_ans_first.mean():.2f}" for l in [18, 20, 21, 22, 24]}))
    md += ["## 6. 렌즈의 중국어 정답 첫 토큰 확률 (층별 평균)\n", md_table(pd.DataFrame(rows)), ""]
    (out / "summary.md").write_text("\n".join(md), "utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="winner_ablate")
    args = ap.parse_args()
    out = RESULTS_DIR / args.model / "A_en_run" / args.out
    out.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        run(args, out)
    analyze(out, args.model)


if __name__ == "__main__":
    main()
