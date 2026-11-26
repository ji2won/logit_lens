#!/usr/bin/env python3
"""한국어 단어를 넣고, 그 단어와 직접 대응하는 영어(여름 → summer 계열) 방향을 지운 뒤
층별 렌즈 상위 토큰과 최종 출력이 어떻게 바뀌는지 기록한다.

  python -m experiment.remove_word --model llama3.2-3b --device mps
  → results/<model>/A_en_run/remove_word/{topk.csv, final.csv}

단위와 W 토큰은 order_ablate의 units.csv를 그대로 쓴다(A_en 입력어·대조어 61개).
제거 범위 (블록 출력, 프롬프트 전 위치, h ← h − QQᵀh, Q = γ⊙U[W]의 정규직교 기저. 다른 방향에 직교화하지 않음)
  pre    10층 ~ 확정-1층 (order_ablate와 같은 범위: 확정 전까지만)
  all    10층 ~ 27층 (마지막 블록 직전까지 전부: 모델이 회복할 여지를 없앰)
  *_orth 위와 같되, W 기저를 최종 출력 상위 토큰들(깨끗한·pre·all 각각 상위 --orth-k개의 합집합)의
         γ⊙U 방향에 직교화한다. W와 중국어 정답 방향이 겹쳐서 생기는 직접 효과를 뺀 조건.
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch

from . import prompts
from .ablate import Ablator
from .config import MODELS, RESULTS_DIR, VERSIONS
from .lenses import VanillaLens, lens_logprobs
from .lm import LM
from .order_ablate import ABL_FROM, PostCapture, basis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--from-layer", type=int, default=15, help="이 층부터 상위 토큰 저장")
    ap.add_argument("--orth-k", type=int, default=20, help="직교화에 쓸 최종 출력 상위 토큰 수(조건마다)")
    args = ap.parse_args()
    base = RESULTS_DIR / args.model / "A_en_run"
    out = base / "remove_word"
    out.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(base / "order_ablate" / "units.csv", keep_default_na=False)

    lm = LM.from_pretrained(MODELS[args.model], device=args.device)
    abl = Ablator(lm)
    cap = PostCapture(abl.blocks)
    L = lm.n_layers
    topk, fin = [], []
    for k, u in units.iterrows():
        W = [int(i) for i in u.W_ids.split()]
        Q, r = basis(lm, W)
        Q = Q.to(lm.device)
        for v in VERSIONS:
            ids = lm.encode(prompts.build("translate", v, u.word))
            abl.clear()
            hs, lg = lm.run(ids)
            P = lens_logprobs(lm, VanillaLens(), cap.hs_last(hs)).exp()
            onset = next((l for l in range(15, L) if P[l, W].sum() >= 0.3), None)
            clean_top = int(lg[-1].argmax())
            runs = {"clean": (P, lg)}
            for mode, layers in {"pre": range(ABL_FROM, onset) if onset else None,
                                 "all": range(ABL_FROM, L)}.items():
                if layers is None:
                    continue
                abl.set(Q, layers)
                hs2, lg2 = lm.run(ids)
                abl.clear()
                runs[mode] = (lens_logprobs(lm, VanillaLens(), cap.hs_last(hs2)).exp(), lg2)
            # 중국어 직교화: 깨끗한·제거 후 최종 출력의 상위 토큰(원래 답과 바뀐 답 모두) 방향을 W에서 빼고 다시 지운다.
            # 이렇게 하면 제거가 그 중국어 토큰들의 로짓을 직접 깎거나 올리지 못하고, 앞층 계산을 거친 효과만 남는다.
            against = sorted({i for _, lgm in runs.values() for i in lgm[-1].topk(args.orth_k).indices.tolist()} - set(W))
            Qo, ro = basis(lm, W, against)
            Qo = Qo.to(lm.device)
            for mode in ["pre", "all"]:
                if mode not in runs:
                    continue
                layers = range(ABL_FROM, onset) if mode == "pre" else range(ABL_FROM, L)
                abl.set(Qo, layers)
                hs2, lg2 = lm.run(ids)
                abl.clear()
                runs[mode + "_orth"] = (lens_logprobs(lm, VanillaLens(), cap.hs_last(hs2)).exp(), lg2)
            for mode, (Pm, lgm) in runs.items():
                key = dict(id=u.id, word=u.word, role=u.role, en=u.en, version=v, mode=mode, onset_clean=onset)
                for l in range(args.from_layer, L + 1):
                    pv, iv = Pm[l].topk(args.topk)
                    for rk, (p_, i_) in enumerate(zip(pv.tolist(), iv.tolist()), 1):
                        topk.append(dict(key, layer=l, rank=rk, token=lm.decode([i_]), p=p_))
                lp = torch.log_softmax(lgm[-1], -1)
                pv, iv = lp.exp().topk(5)
                fin.append(dict(key, W_rank=ro if mode.endswith("_orth") else r, n_against=len(against), W_max_mid=Pm[7:22, W].sum(1).max().item(),
                                p_clean_top=lp[clean_top].exp().item(), clean_top=lm.decode([clean_top]),
                                top1=lm.decode([iv[0].item()]),
                                top5=" ".join(f"{lm.decode([i])}:{p:.3f}" for p, i in zip(pv.tolist(), iv.tolist()))))
        if (k + 1) % 10 == 0 or k + 1 == len(units):
            print(f"  {k + 1}/{len(units)}", flush=True)
    pd.DataFrame(topk).to_csv(out / "topk.csv", index=False)
    pd.DataFrame(fin).to_csv(out / "final.csv", index=False)
    print(f"[done] → {out}")


if __name__ == "__main__":
    main()
