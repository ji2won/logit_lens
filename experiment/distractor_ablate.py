#!/usr/bin/env python3
"""중간층의 영어 다의어 성분을 지우면 최종 출력의 '방해어' 확률(간섭)이 줄어드는가 (데이터셋 A 전체).

  python -m experiment.distractor_ablate --model llama3.2-3b --device mps
  python -m experiment.distractor_ablate --model llama3.2-3b --analyze-only
  → results/<model>/A_en_run/distractor_ablate/{raw.csv, summary.md, item_effects.csv}

예: 층 → 楼层 과제에서 영어 story 성분을 지웠을 때 故事(방해어)의 log p가 떨어지는가.
간섭 I = log p(방해어 | 입력어) − 평균_c log p(방해어 | 대조어 c). 같은 제거를 대조어 프롬프트에도 걸어 ΔI를 잰다.

제거: hidden_states 7–21층(블록 출력, --layers로 변경), 프롬프트 전 위치, h ← h − QQᵀh
방향 두 종류
  U   출력 어휘 방향: 영어 단어 1토큰 변형(단·복수 × 앞 공백 × 대소문자)의 γ⊙U 기저.
      중국어 직교화: 방해어·정답 토큰 + 깨끗한 입력어 프롬프트 최종 상위 20 토큰의 γ⊙U 행공간을 뺀다.
  M   내부 표현 방향(평균 차이): en→zh 프롬프트에서 영어 단어 위치의 잔차(층별, 프롬프트 4개 평균)
      − A 전체 영어 단어의 평균. 층마다 1차원. 같은 중국어 토큰에 직교화.
  Mk  M을 한국어 입력어 자신의 평균 차이 방향(ko→zh 프롬프트, 입력어 위치, 같은 층)에도 직교화.
      입력어와 공유하는 개념(층=floor)을 빼고 영어 단어에만 있는 성분(story의 '이야기' 뜻·영어 형태)만 지운다.
기준선: *_oth = 다른 문항의 같은 종류 방향(같은 직교화, U는 차원 맞춤), *_rand = 무작위 방향(같은 직교화, 같은 차원).
측정(렌즈 없음): 입력어 프롬프트의 log p(방해어 전체), log p(정답 전체), 대조어 프롬프트의 log p(방해어 전체).
확인용(렌즈): 입력어 프롬프트 마지막 위치에서 중간층 영어 단어 확률 최댓값.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import torch

from . import prompts
from .ablate import decoder_layers
from .ablate_analysis import md_table
from .config import DATASETS, MID_LAYER_RANGE, MODELS, N_BOOT, RESULTS_DIR, SEED, VERSIONS
from .data import load_A
from .lenses import VanillaLens, lens_logprobs
from .lm import LM
from .order_ablate import en2zh_prompt

CONDS = ["U", "U_oth", "U_rand", "M", "M_oth", "M_rand", "Mk", "Mk_oth"]


class LayerAblator:
    """hidden_states 인덱스 l(=l-1번째 블록 출력)에 층별 기저 Q_l로 h ← h − (hQ_l)Q_lᵀ (프롬프트 위치만).
    같은 훅에서 프롬프트 마지막 위치의 (제거 후) 잔차를 저장한다."""

    def __init__(self, lm):
        self.blocks = decoder_layers(lm.model)
        self.Qs, self.n_prompt, self.h = {}, None, [None] * len(self.blocks)
        for i, b in enumerate(self.blocks):
            b.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def hook(module, args, out):
            h = out[0] if isinstance(out, tuple) else out
            n = self.n_prompt or h.shape[1]
            Q = self.Qs.get(i + 1)
            if Q is not None:
                hf = h.float()
                new = hf.clone()
                new[:, :n] -= (hf[:, :n] @ Q) @ Q.T
                h = new.to(h.dtype)
                out = (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
            self.h[i] = h[0, n - 1].float()
            return out if Q is not None else None
        return hook

    def stack(self, hs):
        o = hs[:, self.n_prompt - 1].clone()
        for i in range(len(self.h) - 1):
            o[i + 1] = self.h[i]
        return o


def word_pos(lm, prompt, word):
    """프롬프트 마지막 줄에 있는 word의 마지막 토큰 위치."""
    end = prompt.rfind(word) + len(word)
    return len(lm.encode(prompt[:end])) - 1


def variants(lm, en):
    w = en.strip()
    pl = w[:-1] + "ies" if w.endswith("y") and w[-2:-1] not in "aeiou" else (
        w + "es" if w.endswith(("s", "x", "z", "ch", "sh")) else w + "s")
    ids = set()
    for x in {w, w.lower(), w.capitalize(), pl, pl.capitalize()}:
        for s in (x, " " + x):
            t = lm.encode(s, special=False)
            if len(t) == 1:
                ids.add(t[0])
    return sorted(ids)


def orth(M, A):
    """M [r, d]의 행에서 A [k, d] 행공간을 뺀다."""
    if A is None or not len(A):
        return M
    Qa = torch.linalg.qr(A.T)[0]
    return M - (M @ Qa) @ Qa.T


def onb(M, rank=None):
    U, S, _ = torch.linalg.svd(M.T, full_matrices=False)
    r = int((S > S[0] * 1e-3).sum()) if len(S) and S[0] > 0 else 0
    if rank is not None:
        r = min(r, rank)
    return U[:, :r].contiguous()


@torch.no_grad()
def mean_diff_dirs(lm, abl, items, layers):
    """→ en_dir[id] {l: [d]}, ko_dir[id] {l: [d]} : 단어 위치 잔차(프롬프트 4개 평균) − 전체 평균."""
    abl.Qs, abl.n_prompt = {}, None
    E, K = {}, {}
    for it in items:
        e, k = 0, 0
        for v in VERSIONS:
            p = en2zh_prompt(v, it.en_hint)
            hs, _ = lm.run(lm.encode(p))
            e = e + hs[layers, word_pos(lm, p, it.en_hint)].cpu()
            p = prompts.build("translate", v, it.ko)
            hs, _ = lm.run(lm.encode(p))
            k = k + hs[layers, word_pos(lm, p, it.ko)].cpu()
        E[it.id], K[it.id] = e / len(VERSIONS), k / len(VERSIONS)
    me = torch.stack(list(E.values())).mean(0)
    mk = torch.stack(list(K.values())).mean(0)
    return {i: x - me for i, x in E.items()}, {i: x - mk for i, x in K.items()}


def run(args, out):
    items = load_A(DATASETS["A"])
    if args.limit:
        items = items[: args.limit]
    lm = LM.from_pretrained(MODELS[args.model], device=args.device)
    abl = LayerAblator(lm)
    L = lm.n_layers
    layers = list(range(math.ceil(MID_LAYER_RANGE[0] * L), math.floor(MID_LAYER_RANGE[1] * L) + 1))
    lens_layers = layers   # 렌즈 확인은 늘 중간층(7–21)
    if args.layers:
        a_, b_ = map(int, args.layers.split("-"))
        layers = list(range(a_, b_ + 1))
    conds = args.conds.split(",") if args.conds else CONDS
    g = lm.norm.weight.float().cpu()
    Uw = lm.unembed.weight
    gu = lambda ids: (Uw[list(ids)].float().cpu() * g)
    gen = torch.Generator().manual_seed(SEED)
    d = Uw.shape[1]

    print("평균 차이 방향 계산", flush=True)
    E, K = mean_diff_dirs(lm, abl, items, layers) if any(c.startswith("M") for c in conds) else ({}, {})
    n = len(items)
    W_of = {it.id: variants(lm, it.en_hint) for it in items}

    def fwd(prompt_ids, cont_ids, Qs):
        abl.Qs = {l: Q.to(lm.device) for l, Q in Qs.items()} if Qs else {}
        abl.n_prompt = len(prompt_ids)
        hs, lg = lm.run(prompt_ids + cont_ids)
        abl.Qs = {}
        lp = torch.log_softmax(lg, -1)
        npr = len(prompt_ids)
        s = sum(lp[npr - 1 + i, t].item() for i, t in enumerate(cont_ids))
        return s, lp[npr - 1], hs

    rows = []
    for k, it in enumerate(items):
        j = (k + n // 2) % n
        oth = items[j]
        W = W_of[it.id]
        dist_ids = lm.encode(it.zh_distractor, special=False)
        ans_ids = lm.encode(it.zh_answer, special=False)
        in_prompts = {v: lm.encode(prompts.build("translate", v, it.ko)) for v in VERSIONS}
        # 직교화 대상: 방해어·정답 토큰 + 깨끗한 입력어 최종 상위 20 (프롬프트 4개 합집합)
        against = set(dist_ids) | set(ans_ids)
        for v, p in in_prompts.items():
            _, lp, _ = fwd(p, [], None)
            against |= set(lp.topk(20).indices.tolist())
        A = gu(sorted(against - set(W)))
        Uq = onb(orth(gu(W), A))
        r = Uq.shape[1]
        Uo = onb(orth(gu([i for i in W_of[oth.id] if i not in set(W)]), torch.cat([A, gu(W)])), rank=r)
        Ur = onb(orth(torch.randn(r, d, generator=gen), A))
        Qs = {"U": {l: Uq for l in layers}, "U_oth": {l: Uo for l in layers}, "U_rand": {l: Ur for l in layers},
              "M": {}, "M_oth": {}, "M_rand": {}, "Mk": {}, "Mk_oth": {}}
        for li, l in enumerate(layers if E else []):
            Qs["M"][l] = onb(orth(E[it.id][li][None], A))
            Qs["M_oth"][l] = onb(orth(E[oth.id][li][None], torch.cat([A, E[it.id][li][None]])))
            Qs["M_rand"][l] = onb(orth(torch.randn(1, d, generator=gen), A))
            Qs["Mk"][l] = onb(orth(E[it.id][li][None], torch.cat([A, K[it.id][li][None]])))
            Qs["Mk_oth"][l] = onb(orth(E[oth.id][li][None], torch.cat([A, K[it.id][li][None], E[it.id][li][None]])))
        rank = {c: (np.mean([q.shape[1] for q in Qs[c].values()]) if c in Qs else 0) for c in ["clean"] + conds}
        for v in VERSIONS:
            p = in_prompts[v]
            ctrl = [lm.encode(prompts.build("translate", v, c)) for c in it.controls]
            for c in ["clean"] + conds:
                q = Qs.get(c)
                ld, _, hs = fwd(p, dist_ids, q)
                P = lens_logprobs(lm, VanillaLens(), abl.stack(hs)).exp()
                la, _, _ = fwd(p, ans_ids, q)
                lc = [fwd(cp, dist_ids, q)[0] for cp in ctrl]
                rows.append(dict(id=it.id, ko=it.ko, en=it.en_hint, zh=it.zh_answer, zh_d=it.zh_distractor,
                                 version=v, cond=c, rank=rank[c], lp_dist=ld, lp_ans=la, lp_dist_ctrl=float(np.mean(lc)),
                                 I=ld - float(np.mean(lc)), lens_W_mid=P[lens_layers][:, W].sum(1).max().item()))
        if (k + 1) % 5 == 0 or k + 1 == n:
            print(f"  {k + 1}/{n}", flush=True)
            pd.DataFrame(rows).to_csv(out / "raw.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "raw.csv", index=False)


def boot(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return "–"
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return f"{x.mean():.2f} [{np.percentile(m, 2.5):.2f}, {np.percentile(m, 97.5):.2f}]"


def analyze(out, model, layers="7–21"):
    r = pd.read_csv(out / "raw.csv", keep_default_na=False)
    it = r.groupby(["id", "ko", "en", "zh", "zh_d", "cond"], as_index=False)[
        ["lp_dist", "lp_ans", "lp_dist_ctrl", "I", "lens_W_mid", "rank"]].mean()
    c = it[it.cond == "clean"].set_index("id")
    x = it[it.cond != "clean"].copy()
    for m in ["lp_dist", "lp_ans", "lp_dist_ctrl", "I"]:
        x["d_" + m] = x[m].values - c.loc[x.id, m].values
    x["I0"] = c.loc[x.id, "I"].values
    x["lens0"] = c.loc[x.id, "lens_W_mid"].values
    x["grp_I"] = np.where(x.I0 > 2, "I>2", "I≤2")
    x["grp_lens"] = np.where(x.lens0 >= 0.1, "렌즈 보임(≥0.1)", "렌즈 안 보임")
    x.to_csv(out / "item_effects.csv", index=False)
    ni = c.shape[0]
    md = [f"# 중간층 영어 성분 제거 → 최종 출력의 방해어 확률 ({model}, 렌즈 없음)\n",
          f"문항 {ni}개 × 프롬프트 4개. 제거 = {layers}층, 프롬프트 전 위치. Δ = 개입 후 − 깨끗한 (nat). 문항 평균, 95% bootstrap CI.\n",
          f"깨끗한 상태: 평균 I {c.I.mean():.2f} (I>0 {(c.I > 0).mean():.0%}), log p(방해어) {c.lp_dist.mean():.2f}, "
          f"log p(정답) {c.lp_ans.mean():.2f}, 렌즈 중간층 영어 최대 {c.lens_W_mid.mean():.3f}\n"]
    rows = []
    present = [c for c in CONDS if c in set(x.cond)]
    for cond in present:
        g = x[x.cond == cond]
        rows.append(dict(조건=cond, 차원=round(g["rank"].mean(), 1), 렌즈_중간층_영어=f"{g.lens_W_mid.mean():.3f}",
                         Δlogp_방해어=boot(g.d_lp_dist), Δlogp_방해어_대조=boot(g.d_lp_dist_ctrl),
                         ΔI=boot(g.d_I), Δlogp_정답=boot(g.d_lp_ans)))
    md += ["## 1. 조건별\n", md_table(pd.DataFrame(rows)), ""]
    w = x.pivot_table(index=["id", "grp_I", "grp_lens"], columns="cond", values=["d_lp_dist", "d_I", "d_lp_ans"]).reset_index()
    rows = []
    for a, b in [p for p in [("U", "U_oth"), ("U", "U_rand"), ("M", "M_oth"), ("M", "M_rand"), ("Mk", "Mk_oth")]
                 if p[0] in present and p[1] in present]:
        for name, g in [("전체", w)] + [(k, w[w.grp_I == k]) for k in ["I>2", "I≤2"]] + \
                       [(k, w[w.grp_lens == k]) for k in ["렌즈 보임(≥0.1)", "렌즈 안 보임"]]:
            rows.append(dict(비교=f"{a} − {b}", 문항=name, n=len(g),
                             ΔΔlogp_방해어=boot(g[("d_lp_dist", a)] - g[("d_lp_dist", b)]),
                             ΔΔI=boot(g[("d_I", a)] - g[("d_I", b)]),
                             ΔΔlogp_정답=boot(g[("d_lp_ans", a)] - g[("d_lp_ans", b)])))
    md += ["## 2. 짝 비교 (음수 = 그 단어의 영어 성분을 지웠을 때 기준선보다 더 떨어짐)\n",
           "- ΔΔI가 음수이고 CI가 0을 넘지 않으면: 영어 다의어 성분이 방해어(간섭)를 실어 나른다.\n"
           "- ΔΔlogp_정답도 같이 크게 음수이면 방해어만이 아니라 번역 전체가 무너진 것(특이성 없음).\n",
           md_table(pd.DataFrame(rows)), ""]
    top = x[x.cond.isin([c for c in ["U", "M", "Mk"] if c in present])].pivot_table(index=["id", "ko", "en", "zh_d", "I0"], columns="cond", values="d_I").reset_index()
    top = top.sort_values("I0", ascending=False).head(15)
    md += ["## 3. 깨끗한 간섭이 큰 문항 15개의 ΔI\n", md_table(top.round(2)), ""]
    (out / "summary.md").write_text("\n".join(md), "utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--layers", default=None, help="제거 층 범위 a-b (hidden_states 인덱스). 기본 = 중간층")
    ap.add_argument("--conds", default=None, help="쉼표로 구분한 조건 (기본 전부)")
    args = ap.parse_args()
    out = RESULTS_DIR / args.model / "A_en_run" / (args.out or "distractor_ablate")
    out.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        run(args, out)
    analyze(out, args.model, args.layers or "7–21")


if __name__ == "__main__":
    main()
