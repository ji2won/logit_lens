#!/usr/bin/env python3
"""로짓 렌즈 부풀림 검사 두 가지 (한→중, Llama).

  python -m experiment.inflation_test --model llama3.2-3b --device mps
  python -m experiment.inflation_test --model llama3.2-3b --analyze-only
  → results/<model>/A_en_run/inflation/{size.csv, posctrl.csv, summary.md}

[1] 크기 대비 확률 (A·C, 깨끗한 실행)
  렌즈 로짓 = (h/rms(h))·(γ⊙U_t) = √d · ‖γ⊙U_t‖ · cos(h, γ⊙U_t).  즉 로짓은 잔차와 토큰 방향의 '코사인'에 비례한다.
  층마다 프롬프트 마지막 위치에서
    share_en    ‖Q_enᵀh‖/‖h‖   Q_en = 렌즈 영어 승자(winner_ablate) 1토큰 변형의 γ⊙U를 중국어 출력 토큰에 직교화한 기저
                                 (winner_ablate가 지운 바로 그 조각)
    share_en_raw  직교화 전 영어 변형 기저
    share_zh    ‖Q_zhᵀh‖/‖h‖   Q_zh = 중국어 정답 토큰들의 γ⊙U 기저
    base_en / base_zh   같은 차원 무작위 부분공간의 기대값 √(r/d)
    p_en, p_zh  렌즈 확률 (영어 변형 합 / 정답 첫 토큰)
    cos1, cos2, p1, p2   렌즈 1·2위 토큰의 코사인과 확률 (작은 코사인 차이가 큰 확률 차이가 되는지)
[2] 한→영 양성 대조 (C)
  같은 제거(정답 영어 단어의 영어 고유 조각, 중국어 출력 토큰에 직교화, 7–21층, 프롬프트 위치)를
  한→중(translate)과 한→영(translate_to_en)에 똑같이 건다. 한→영에서는 그 영어가 출력이므로
  제거 방법이 '필요한 성분'을 잡을 수 있다면 정답 영어 확률이 크게 떨어져야 한다.
  조건: W_orth / other_orth(다른 문항 영어, 차원 맞춤) / rand_orth
"""
from __future__ import annotations

import argparse
import math

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


def setup(args):
    lm = LM.from_pretrained(MODELS[args.model], device=args.device)
    abl = LayerAblator(lm)
    g = lm.norm.weight.float().cpu()
    Uw = lm.unembed.weight
    gu = lambda ids: Uw[list(ids)].float().cpu() * g
    return lm, abl, gu


def part_size(args, out, lm, abl, gu):
    win = pd.read_csv(RESULTS_DIR / args.model / "A_en_run" / "winner_ablate" / "raw.csv", keep_default_na=False, low_memory=False)
    win = win[win.cond == "W_orth"].set_index(["id", "version"]).winner.to_dict()
    items = [("A", it) for it in load_A(DATASETS["A"])] + [("C", it) for it in load_C(DATASETS["C"])]
    if args.limit:
        items = [x for x in items if x[0] == "A"][: args.limit] + [x for x in items if x[0] == "C"][: args.limit]
    G = (lm.unembed.weight.float() * lm.norm.weight.float())            # [V, d] γ⊙U (기기 위)
    Gn = G.norm(dim=1)
    d = G.shape[1]
    rows = []
    abl.Qs = {}
    for k, (ds, it) in enumerate(items):
        a_ids = lm.encode(it.zh_answer, special=False)
        Qz = onb(gu(a_ids)).to(lm.device)
        for v in VERSIONS:
            w = win.get((it.id, v))
            if not w:
                continue
            W = variants(lm, w)
            p_ids = lm.encode(prompts.build("translate", v, it.ko))
            abl.n_prompt = len(p_ids)
            hs, lg = lm.run(p_ids)
            h = hs[:, -1]                                                  # [L+1, d] 정규화 전 잔차
            against = sorted((set(a_ids) | set(lg[-1].topk(20).indices.tolist())) - set(W))
            Qe = onb(orth(gu(W), gu(against))).to(lm.device)
            Qer = onb(gu(W)).to(lm.device)
            P = lens_logprobs(lm, VanillaLens(), h).exp()
            hn = h / h.norm(dim=1, keepdim=True)
            for l in range(h.shape[0]):
                cos = (G @ hn[l]) / Gn                                     # [V] 토큰별 코사인
                pv, iv = P[l].topk(2)
                rows.append(dict(dataset=ds, id=it.id, ko=it.ko, version=v, winner=w, layer=l,
                                 share_en=(Qe.T @ hn[l]).norm().item(), share_en_raw=(Qer.T @ hn[l]).norm().item(),
                                 share_zh=(Qz.T @ hn[l]).norm().item(),
                                 base_en=math.sqrt(Qe.shape[1] / d), base_zh=math.sqrt(Qz.shape[1] / d),
                                 p_en=P[l, W].sum().item(), p_zh=P[l, a_ids[0]].item(),
                                 cos_en=cos[W].max().item(), cos_zh=cos[a_ids[0]].item(),
                                 p1=pv[0].item(), p2=pv[1].item(), cos1=cos[iv[0]].item(), cos2=cos[iv[1]].item(),
                                 cos_max_abs=cos.abs().max().item()))
        if (k + 1) % 50 == 0 or k + 1 == len(items):
            print(f"  size {k + 1}/{len(items)}", flush=True)
    pd.DataFrame(rows).to_csv(out / "size.csv", index=False)
    del G, Gn
    if str(lm.device).startswith("mps"):
        torch.mps.empty_cache()


def part_posctrl(args, out, lm, abl, gu):
    items = load_C(DATASETS["C"])
    if args.limit:
        items = items[: args.limit]
    L = lm.n_layers
    layers = list(range(math.ceil(MID_LAYER_RANGE[0] * L), math.floor(MID_LAYER_RANGE[1] * L) + 1))
    gen = torch.Generator().manual_seed(SEED)
    d = lm.unembed.weight.shape[1]
    n = len(items)
    Wof = {it.id: variants(lm, it.probes["target"]) for it in items}

    def fwd(p_ids, a_ids, Q):
        abl.Qs = {l: Q.to(lm.device) for l in layers} if Q is not None else {}
        abl.n_prompt = len(p_ids)
        hs, lg = lm.run(p_ids + a_ids)
        abl.Qs = {}
        lp = torch.log_softmax(lg, -1)
        m = len(p_ids)
        ans = sum(lp[m - 1 + i, t].item() for i, t in enumerate(a_ids))
        P = lens_logprobs(lm, VanillaLens(), abl.stack(hs)).exp()
        return lp[m - 1], ans, P

    prev = out / "posctrl.csv"
    rows = pd.read_csv(prev, keep_default_na=False).to_dict("records") if args.start and prev.exists() else []
    if rows:
        rows = [r for r in rows if r["id"] in {it.id for it in items[: args.start]}]
    for k, it in enumerate(items):
        if k < args.start:
            continue
        W = Wof[it.id]
        if not W:
            continue
        oth = items[(k + n // 2) % n]
        en = it.probes["target"]
        tasks = {"ko→zh": ("translate", lm.encode(it.zh_answer, special=False)),
                 "ko→en": ("translate_to_en", lm.encode(en, special=False))}
        for v in VERSIONS:
            pz = lm.encode(prompts.build("translate", v, it.ko))
            lpz, _, _ = fwd(pz, [], None)
            against = sorted((set(tasks["ko→zh"][1]) | set(lpz.topk(20).indices.tolist())) - set(W))
            A = gu(against)
            Qw = onb(orth(gu(W), A))
            r = Qw.shape[1]
            oW = [i for i in Wof[oth.id] if i not in set(W)]
            Qo = onb(orth(gu(oW), torch.cat([A, gu(W)])), rank=r) if oW else None
            Qr = onb(orth(torch.randn(r, d, generator=gen), A))
            for tname, (task, a_ids) in tasks.items():
                p_ids = lm.encode(prompts.build(task, v, it.ko))
                lp0, ans0, P0 = fwd(p_ids, a_ids, None)
                top0 = int(lp0.argmax())
                for cond, Q in [("W_orth", Qw), ("other_orth", Qo), ("rand_orth", Qr)]:
                    if Q is None or Q.shape[1] == 0:
                        continue
                    lp, ans, P = fwd(p_ids, a_ids, Q)
                    rows.append(dict(id=it.id, ko=it.ko, en=en, zh=it.zh_answer, version=v, task=tname, cond=cond,
                                     rank=Q.shape[1], lp_ans_clean=ans0, lp_ans=ans, d_lp_ans=ans - ans0,
                                     top1_same=int(lp.argmax()) == top0, top_clean=lm.decode([top0]),
                                     top_after=lm.decode([int(lp.argmax())]),
                                     kl=(lp0.exp() * (lp0 - lp)).sum().item(),
                                     lens_en_mid_clean=P0[layers][:, W].sum(1).max().item(),
                                     lens_en_mid=P[layers][:, W].sum(1).max().item(),
                                     lens_en_L27_clean=P0[27, W].sum().item(), lens_en_L27=P[27, W].sum().item()))
        if lm.device == "mps" or str(lm.device).startswith("mps"):
            torch.mps.empty_cache()
        if (k + 1) % 25 == 0 or k + 1 == n:
            print(f"  posctrl {k + 1}/{n}", flush=True)
            pd.DataFrame(rows).to_csv(out / "posctrl.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "posctrl.csv", index=False)


def boot(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return "–"
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return f"{x.mean():.3f} [{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]"


def analyze(out, model):
    md = [f"# 로짓 렌즈 부풀림 검사 ({model}, 한→중)\n"]
    s = pd.read_csv(out / "size.csv", keep_default_na=False)
    md += ["## 1. 크기 대비 확률\n",
           "share = 잔차 방향(단위벡터)이 그 부분공간에 놓인 길이(0~1). base = 같은 차원 무작위 부분공간의 기대값. "
           "cos = 잔차와 토큰 방향의 코사인(렌즈 로짓은 이것에 비례). 문항×프롬프트 평균.\n"]
    for ds in ["A", "C"]:
        g = s[s.dataset == ds].groupby("layer")[["share_en", "share_en_raw", "share_zh", "base_en", "base_zh",
                                                   "p_en", "p_zh", "cos_en", "cos_zh", "cos1", "cos2", "p1", "p2"]].mean()
        g = g.loc[[8, 12, 16, 18, 20, 21, 22, 24, 26, 27, 28]].reset_index()
        md += [f"### {ds}\n", md_table(g.round(3)), ""]
    mid = s[s.layer.between(20, 24)]
    fin = s[s.layer == 28]
    rows = []
    for name, g in [("20–24층", mid), ("최종층(28)", fin)]:
        rows.append(dict(구간=name, 영어_share=f"{g.share_en.mean():.3f}", 중국어_share=f"{g.share_zh.mean():.3f}",
                         영어_p=f"{g.p_en.mean():.3f}", 중국어_p=f"{g.p_zh.mean():.3f}",
                         렌즈1위_cos=f"{g.cos1.mean():.3f}", 렌즈2위_cos=f"{g.cos2.mean():.3f}",
                         렌즈1위_p=f"{g.p1.mean():.3f}", 렌즈2위_p=f"{g.p2.mean():.3f}"))
    md += ["### 요약: 중간층 vs 최종층 (A·C 합)\n", md_table(pd.DataFrame(rows)), ""]
    # 같은 확률대에서 필요한 코사인: 렌즈 1위 확률 0.5 이상인 경우의 1위 코사인 분포
    rows = []
    for name, g in [("20–24층", mid), ("최종층(28)", fin)]:
        hi = g[g.p1 >= 0.5]
        rows.append(dict(구간=name, n=len(hi), 렌즈1위_p_평균=f"{hi.p1.mean():.3f}",
                         cos1_중앙값=f"{hi.cos1.median():.3f}", cos2_중앙값=f"{hi.cos2.median():.3f}",
                         cos차_중앙값=f"{(hi.cos1 - hi.cos2).median():.3f}"))
    md += ["### 렌즈 1위 확률 ≥ 0.5인 경우: 그 확률을 만드는 코사인\n", md_table(pd.DataFrame(rows)), ""]

    p = pd.read_csv(out / "posctrl.csv", keep_default_na=False)
    p["top1_same"] = p.top1_same.astype(str).eq("True")
    u = p.groupby(["id", "task", "cond"], as_index=False)[["d_lp_ans", "kl", "top1_same", "lens_en_mid_clean", "lens_en_mid",
                                                           "lens_en_L27_clean", "lens_en_L27", "lp_ans_clean"]].mean()
    rows = []
    for t in ["ko→zh", "ko→en"]:
        for c in ["W_orth", "other_orth", "rand_orth"]:
            g = u[(u.task == t) & (u.cond == c)]
            rows.append(dict(과제=t, 조건=c, 문항=len(g), 정답_logp_깨끗=f"{g.lp_ans_clean.mean():.2f}",
                             렌즈_영어_중간_전=f"{g.lens_en_mid_clean.mean():.3f}", 렌즈_영어_중간_후=f"{g.lens_en_mid.mean():.3f}",
                             렌즈_영어_L27_전=f"{g.lens_en_L27_clean.mean():.3f}", 렌즈_영어_L27_후=f"{g.lens_en_L27.mean():.3f}",
                             Δlogp_정답=boot(g.d_lp_ans), KL=boot(g.kl), 최종1위_유지=f"{g.top1_same.mean():.1%}"))
    md += ["## 2. 한→영 양성 대조 (C, 같은 제거를 두 과제에)\n", md_table(pd.DataFrame(rows)), ""]
    w = u.pivot_table(index="id", columns=["task", "cond"], values="d_lp_ans")
    rows = []
    for t in ["ko→zh", "ko→en"]:
        for b in ["other_orth", "rand_orth"]:
            rows.append(dict(과제=t, 비교=f"W_orth − {b}", ΔΔlogp_정답=boot(w[(t, "W_orth")] - w[(t, b)])))
    rows.append(dict(과제="ko→en − ko→zh", 비교="W_orth 효과 차이", ΔΔlogp_정답=boot(w[("ko→en", "W_orth")] - w[("ko→zh", "W_orth")])))
    md += [md_table(pd.DataFrame(rows)), ""]
    x = p[(p.cond == "W_orth")]
    bins = [(-np.inf, -1), (-1, -0.3), (-0.3, -0.1), (-0.1, 0.1), (0.1, np.inf)]
    lab = ["< −1", "−1 ~ −0.3", "−0.3 ~ −0.1", "±0.1", "> 0.1"]
    rows = []
    for t in ["ko→zh", "ko→en"]:
        gg = x[x.task == t]
        rows.append({"과제": t, "n": len(gg), **{l: f"{((gg.d_lp_ans > lo) & (gg.d_lp_ans <= hi)).mean():.0%}" for (lo, hi), l in zip(bins, lab)}})
    md += ["### 문항×프롬프트별 정답 변화 분포 (W_orth, nat)\n", md_table(pd.DataFrame(rows)), ""]
    (out / "summary.md").write_text("\n".join(md), "utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="inflation")
    ap.add_argument("--part", choices=["all", "size", "posctrl"], default="all")
    ap.add_argument("--start", type=int, default=0, help="posctrl을 이 문항 번호부터 이어서 (앞 결과는 posctrl.csv에서 유지)")
    args = ap.parse_args()
    out = RESULTS_DIR / args.model / "A_en_run" / args.out
    out.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        lm, abl, gu = setup(args)
        if args.part in ("all", "size"):
            part_size(args, out, lm, abl, gu)
        if args.part in ("all", "posctrl"):
            part_posctrl(args, out, lm, abl, gu)
    analyze(out, args.model)


if __name__ == "__main__":
    main()
