#!/usr/bin/env python3
"""A_en 문항만 한→중 번역(translate, 입력어 프롬프트)을 돌려, 층별로 중국어 정답이 렌즈에 얼마나 보이는지 기록한다.

  python -m experiment.zh_trace --model llama3.2-3b --device mps
  → results/<model>/A_en_run/zh_trace.csv   문항×버전×층: 바닐라 렌즈의
        p_zh      중국어 정답 첫 토큰 확률 (lm.score와 같은 토큰화: 정답을 앞 공백 없이 인코딩한 첫 토큰)
        p_en      영어 probe 단어(A.en) 첫 토큰 후보들의 확률 합
        mass_zh / mass_en   어휘 중 중국어 / 영어 토큰 확률 합
        top1      그 층 1위 토큰
     results/<model>/A_en_run/zh_topk.csv   --topk-from 층부터 마지막 층까지 상위 --topk 토큰과 확률
     (role = input | control. control은 ko_control_final 단어를 입력으로 넣은 프롬프트, p_own_en = 그 단어 자신의 영어)
     results/<model>/A_en_run/zh_final.csv  문항×버전: 최종 출력의 p(정답 전체), p(정답 첫 토큰), 1위 토큰
"""
from __future__ import annotations

import argparse
import math

import pandas as pd
import torch

from . import prompts
from .config import DATASETS, MODELS, RESULTS_DIR, ROOT, VERSIONS
from .data import load_A
from .lenses import VanillaLens, language_masks, lens_logprobs, probe_token_ids
from .lm import LM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--items", default=str(ROOT / "A_en.csv"))
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--topk-from", type=float, default=0.25, help="이 비율 층부터 마지막 층까지 상위 토큰 저장")
    ap.add_argument("--roles", nargs="+", default=["input", "control"])
    args = ap.parse_args()

    out = RESULTS_DIR / args.model / "A_en_run"
    out.mkdir(parents=True, exist_ok=True)
    keep = set(pd.read_csv(args.items).id)
    items = [it for it in load_A(DATASETS["A"]) if it.id in keep]

    lm = LM.from_pretrained(MODELS[args.model], device=args.device)
    mf = RESULTS_DIR / args.model / "lang_masks.pt"
    masks = torch.load(mf) if mf.exists() else language_masks(lm)
    lens = VanillaLens()

    rows, fin, topk_rows = [], [], []
    for k, it in enumerate(items, 1):
        zh_first = lm.encode(it.zh_answer, special=False)[0]
        en_ids = probe_token_ids(lm, it.en_hint)[0]                # 입력어의 영어 probe (spring)
        roles = [("input", it.ko, it.en_hint)]
        if "control" in args.roles:
            ens = it.en_controls or [""] * len(it.controls)
            roles += [("control", c, e) for c, e in zip(it.controls, ens)]
        for role, word, own_en in roles:
            own_ids = probe_token_ids(lm, own_en)[0] if own_en else []   # 그 단어 자신의 영어 (여름 → summer)
            for v in VERSIONS:
                prompt = prompts.build("translate", v, word, it.en_hint)
                # 프롬프트 마지막 위치의 잔차는 뒤에 붙는 토큰과 무관(인과 마스크). 대조어에서도
                # p_answer = 입력어의 중국어 정답이 나올 확률(기준선)로 기록한다.
                lp, lp1, nt, hs = lm.score(prompt, it.zh_answer)
                L = lens_logprobs(lm, lens, hs)                       # [L+1, V]
                P = L.exp()
                top = L.argmax(-1).tolist()
                key = dict(id=it.id, ko=it.ko, zh=it.zh_answer, role=role, word=word, version=v)
                lo = math.ceil(args.topk_from * (L.shape[0] - 1))
                for l in range(lo, L.shape[0]):                      # 중간층부터 마지막 층까지 상위 토큰
                    pv, iv = P[l].topk(args.topk)
                    for r, (p_, i_) in enumerate(zip(pv.tolist(), iv.tolist()), 1):
                        topk_rows.append(dict(key, layer=l, rank=r, token=lm.decode([i_]), p=p_))
                for l in range(L.shape[0]):
                    rows.append(dict(key, en=it.en_hint, own_en=own_en, zh_tok=lm.decode([zh_first]), layer=l,
                                     p_zh=P[l, zh_first].item(), p_en=P[l, en_ids].sum().item(),
                                     p_own_en=P[l, own_ids].sum().item() if own_ids else float("nan"),
                                     mass_zh=P[l][masks["zh"].to(P.device)].sum().item(),
                                     mass_en=P[l][masks["en"].to(P.device)].sum().item(),
                                     top1=lm.decode([top[l]])))
                fin.append(dict(key, en=it.en_hint, ntok=nt,
                                p_answer=math.exp(lp), p_answer_first=math.exp(lp1),
                                final_top1=lm.decode([top[-1]])))
        if k % 10 == 0 or k == len(items):
            print(f"  {k}/{len(items)}", flush=True)

    pd.DataFrame(rows).to_csv(out / "zh_trace.csv", index=False)
    pd.DataFrame(fin).to_csv(out / "zh_final.csv", index=False)
    pd.DataFrame(topk_rows).to_csv(out / "zh_topk.csv", index=False)
    print(f"[done] → {out}")


if __name__ == "__main__":
    main()
