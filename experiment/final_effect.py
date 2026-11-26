#!/usr/bin/env python3
"""중간층(7–21층)에서 그 단어의 영어 방향을 지웠을 때, 렌즈 없이 '실제 최종 출력'의 중국어 확률이 바뀌는가.

렌즈 부풀림 vs 영어 경유를 가리는 검사:
  - 영어 경유라면: 중간층 영어를 지우면 최종 출력의 중국어 정답이 기준선(다른 영어·무작위 방향)보다 더 떨어진다.
  - 렌즈 부풀림이라면: 렌즈의 영어는 사라져도 최종 출력은 기준선과 다르지 않다.

  python -m experiment.final_effect --model llama3.2-3b --device mps
  python -m experiment.final_effect --model llama3.2-3b --analyze-only
  → results/<model>/A_en_run/final_effect/{raw.csv, summary.md}

단위·W 토큰 = order_ablate/units.csv (A_en 입력어 28 + 대조어 33).
제거: hidden_states 7–21층(=블록 6–20 출력), 프롬프트 위치만, h ← h − QQᵀh
  W            그 단어 영어 계열(summer, summers …)의 γ⊙U 기저
  W_orth       W를 '중국어 출력 토큰'(깨끗한·W 제거 후 최종 출력 상위 20개 + 정답 토큰)에 직교화
               → 영어와 중국어 방향이 겹쳐 생기는 직접 효과를 뺀다
  other_orth   다른 단위의 W를 같은 중국어 토큰과 자기 W에 직교화, 차원 수를 W_orth와 맞춤 (기준선)
  rand_orth    무작위 방향을 같은 방식으로 직교화, 차원 수 맞춤 (기준선)
측정 (렌즈 없음 = 모델의 실제 최종 출력)
  lp_top      깨끗한 순전파 최종 1위 토큰의 log p
  lp_answer   (입력어만) 데이터셋 중국어 정답 전체의 log p (teacher forcing)
  kl          KL(깨끗한 최종 분포 ‖ 개입 후 최종 분포)
  zh_mass     최종 출력에서 중국어 토큰 확률 합
보조 (렌즈): L27 정답 첫 토큰 확률, 중간층 W 최대 확률(제거가 실제로 됐는지 확인)
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import torch

from . import prompts
from .ablate import Ablator
from .ablate_analysis import md_table
from .config import MID_LAYER_RANGE, MODELS, N_BOOT, RESULTS_DIR, ROOT, SEED, VERSIONS
from .lenses import VanillaLens, lens_logprobs
from .lm import LM
from .order_ablate import basis

CONDS = ["W", "W_orth", "other_orth", "rand_orth"]


class PosCapture:
    """블록 출력을 Ablator 훅 뒤에서, 지정한 위치(pos)만 받는다 (HF hidden_states는 같은 층 제거가 반영 안 됨)."""

    def __init__(self, blocks):
        self.pos, self.h = -1, [None] * len(blocks)
        for i, b in enumerate(blocks):
            b.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def hook(module, args, out):
            h = out[0] if isinstance(out, tuple) else out
            self.h[i] = h[0, self.pos].float()
        return hook

    def stack(self, hs):
        o = hs[:, self.pos].clone()
        for i in range(len(self.h) - 1):
            o[i + 1] = self.h[i]
        return o


def run(args, out):
    base = RESULTS_DIR / args.model / "A_en_run"
    units = pd.read_csv(base / "order_ablate" / "units.csv", keep_default_na=False)
    zh = dict(pd.read_csv(ROOT / "A_final.csv", keep_default_na=False)[["id", "zh_answer"]].values)
    lm = LM.from_pretrained(MODELS[args.model], device=args.device)
    abl = Ablator(lm)
    cap = PosCapture(abl.blocks)
    L = lm.n_layers
    layers = list(range(math.ceil(MID_LAYER_RANGE[0] * L), math.floor(MID_LAYER_RANGE[1] * L) + 1))
    mf = RESULTS_DIR / args.model / "lang_masks.pt"
    zmask = torch.load(mf)["zh"].to(lm.device)
    gen = torch.Generator().manual_seed(SEED)
    n = len(units)
    Wl = [[int(i) for i in s.split()] for s in units.W_ids]
    rows = []

    def forward(ids, n_prompt, Q):
        """→ (최종 log p [V] at 프롬프트 끝, 정답 log p 합 or None, 렌즈 P [L+1, V] at 프롬프트 끝)"""
        if Q is None:
            abl.clear()
        else:
            abl.set(Q, layers)
        abl.n_prompt = n_prompt
        cap.pos = n_prompt - 1
        hs, lg = lm.run(ids)
        abl.clear()
        lp = torch.log_softmax(lg, -1)
        ans = sum(lp[n_prompt - 1 + i, t].item() for i, t in enumerate(ids[n_prompt:])) if len(ids) > n_prompt else None
        # 마지막 hs(정규화 직전)는 lm.run이 전 위치를 주므로 그대로, 나머지 층은 훅 뒤 캡처값
        return lp[n_prompt - 1], ans, lens_logprobs(lm, VanillaLens(), cap.stack(hs)).exp()

    for k, u in units.iterrows():
        if args.limit and k >= args.limit:
            break
        W = Wl[k]
        j = (k + n // 2) % n
        if units.id[j] == u.id:
            j = (j + 1) % n
        oW = [i for i in Wl[j] if i not in set(W)]
        is_input = u.role == "input"
        ans_ids = lm.encode(zh[u.id], special=False) if is_input else []
        Qw, rw = basis(lm, W)
        for v in VERSIONS:
            p_ids = lm.encode(prompts.build("translate", v, u.word))
            ids = p_ids + ans_ids
            npr = len(p_ids)
            res = {"clean": forward(ids, npr, None)}
            top = int(res["clean"][0].argmax())
            res["W"] = forward(ids, npr, Qw.to(lm.device))
            against = {i for c in ("clean", "W") for i in res[c][0].topk(20).indices.tolist()} | set(ans_ids[:1])
            against = sorted(against - set(W))
            Qo, ro = basis(lm, W, against)
            Qx, rx = basis(lm, oW, against + W, rank=ro)
            M = torch.randn(ro, lm.unembed.weight.shape[1], generator=gen)
            A = (lm.unembed.weight[against].float() * lm.norm.weight.float()).cpu()
            Qa = torch.linalg.qr(A.T)[0]
            M = M - (M @ Qa) @ Qa.T
            Qr = torch.linalg.qr(M.T)[0]
            res["W_orth"] = forward(ids, npr, Qo.to(lm.device))
            if rx:
                res["other_orth"] = forward(ids, npr, Qx.to(lm.device))
            res["rand_orth"] = forward(ids, npr, Qr.to(lm.device))
            lp0 = res["clean"][0]
            for c, (lp, ans, P) in res.items():
                rows.append(dict(id=u.id, word=u.word, role=u.role, en=u.en, version=v, cond=c,
                                 rank={"clean": 0, "W": rw, "W_orth": ro, "other_orth": rx, "rand_orth": ro}[c],
                                 lp_top=lp[top].item(), lp_answer=ans,
                                 kl=(lp0.exp() * (lp0 - lp)).sum().item(),
                                 zh_mass=lp.exp()[zmask].sum().item(),
                                 top1_same=int(lp.argmax()) == top,
                                 lens27_ans=P[27, ans_ids[0]].item() if ans_ids else None,
                                 lens_W_mid=P[layers][:, W].sum(1).max().item()))
        if (k + 1) % 10 == 0 or k + 1 == n:
            print(f"  {k + 1}/{n}", flush=True)
            pd.DataFrame(rows).to_csv(out / "raw.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "raw.csv", index=False)


def boot(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if not len(x):
        return "–"
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return f"{x.mean():.3f} [{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]"


def analyze(out):
    r = pd.read_csv(out / "raw.csv")
    key = ["id", "word", "version"]
    c = r[r.cond == "clean"].set_index(key)
    x = r[r.cond != "clean"].copy()
    idx = list(zip(x.id, x.word, x.version))
    for m in ["lp_top", "lp_answer", "zh_mass"]:
        x["d_" + m] = x[m].values - c.loc[idx, m].values
    u = x.groupby(["id", "word", "role", "cond"], as_index=False)[
        ["d_lp_top", "d_lp_answer", "kl", "d_zh_mass", "top1_same", "lens_W_mid", "rank"]].mean()
    cu = c.reset_index().groupby(["id", "word"])[["lp_top", "lp_answer", "zh_mass", "lens_W_mid"]].mean()
    md = ["# 중간층 영어를 지우면 실제 최종 출력의 중국어가 바뀌나 (Llama, 렌즈 없음)\n",
          f"단위 {len(cu)}개 (입력어 {u[u.role == 'input'][['id', 'word']].drop_duplicates().shape[0]}·대조어 "
          f"{u[u.role == 'control'][['id', 'word']].drop_duplicates().shape[0]}). 제거 = 중간층 7–21층, 프롬프트 위치. "
          "Δ = 개입 후 − 깨끗한 (nat). 프롬프트 4개 평균 후 단위 평균, 95% bootstrap CI.\n",
          f"깨끗한 상태: 최종 1위 토큰 log p {cu.lp_top.mean():.3f}, 정답 log p(입력어) {cu.lp_answer.mean():.3f}, "
          f"중국어 질량 {cu.zh_mass.mean():.3f}, 렌즈 중간층 영어 단어 최대확률 {cu.lens_W_mid.mean():.3f}\n"]
    rows = []
    for cond in CONDS:
        g = u[u.cond == cond]
        rows.append(dict(조건=cond, 차원=round(g["rank"].mean(), 1), 렌즈_중간층_영어=f"{g.lens_W_mid.mean():.3f}",
                         Δlog_p_최종1위=boot(g.d_lp_top), Δlog_p_정답_입력어=boot(g[g.role == "input"].d_lp_answer),
                         KL=boot(g.kl), Δ중국어질량=boot(g.d_zh_mass), 최종1위_유지=f"{g.top1_same.mean():.3f}"))
    md += ["## 1. 조건별 (실제 최종 출력)\n", md_table(pd.DataFrame(rows)), ""]
    w = u.pivot_table(index=["id", "word", "role"], columns="cond", values=["d_lp_top", "d_lp_answer", "kl"]).reset_index()
    rows = []
    for a, b in [("W", "other_orth"), ("W_orth", "other_orth"), ("W_orth", "rand_orth"), ("W", "W_orth")]:
        for role in ["전체", "input", "control"]:
            g = w if role == "전체" else w[w.role == role]
            rows.append(dict(비교=f"{a} − {b}", 단위=role, n=len(g),
                             ΔΔlog_p_최종1위=boot(g[("d_lp_top", a)] - g[("d_lp_top", b)]),
                             ΔΔlog_p_정답=boot(g[("d_lp_answer", a)] - g[("d_lp_answer", b)]) if role != "control" else "–",
                             ΔKL=boot(g[("kl", a)] - g[("kl", b)])))
    md += ["## 2. 짝 비교\n",
           "- **W_orth − other_orth**가 핵심: 중국어와 겹치는 성분을 뺀 '영어 고유' 방향을 지웠을 때, 같은 차원의 다른 영어를 지운 것보다 "
           "최종 출력이 더 떨어지면(음수) 영어 경유, 차이가 없으면 렌즈가 보인 영어는 출력에 쓰이지 않는다(부풀림).\n"
           "- W − W_orth = 영어·중국어 방향 겹침 때문에 생긴 직접 효과의 크기.\n",
           md_table(pd.DataFrame(rows)), ""]
    (out / "summary.md").write_text("\n".join(md), "utf-8")
    u.to_csv(out / "unit_effects.csv", index=False)
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    out = RESULTS_DIR / args.model / "A_en_run" / "final_effect"
    out.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        run(args, out)
    analyze(out)


if __name__ == "__main__":
    main()
