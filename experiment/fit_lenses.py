#!/usr/bin/env python3
"""평균 제거 렌즈와 아핀 렌즈의 파라미터를 한·중 위키 문장으로 구한다.

  python -m experiment.fit_lenses --model llama3.2-3b
  → results/<model>/lenses.pt      mu [L+1,d], W [L+1,d,d], b [L+1,d], 층별 lambda
    results/<model>/lens_fit.csv   held-out 문장에서 층별 KL(최종 출력 || 렌즈 출력)

평균 제거: mu_l = 층 l 잔차의 평균 (모든 문장·토큰 위치, BOS와 첫 토큰 제외)
아핀:     h_l → h_L(최종층, 정규화 전) 리지 회귀. 닫힌 해라 학습이 결정적이다.
          W_l = (C_l + lambda * tr(C_l)/d * I)^-1 C_lL,  b_l = mu_L - mu_l W_l
          lambda는 held-out KL이 가장 작은 값을 층마다 고른다.
KL 표는 렌즈가 실제로 최종 출력에 더 가까워졌는지 확인하는 용도다(아핀 KL < 바닐라 KL이어야 의미가 있다).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import torch

from .config import DATA_DIR, MODELS, RESULTS_DIR, SEED
from .lm import LM


def load_sentences(corpus_dir: Path, n_per_lang: int) -> list[str]:
    out = []
    for lang in ("ko", "zh"):
        lines = (corpus_dir / f"wiki_{lang}.jsonl").read_text("utf-8").splitlines()
        out += [json.loads(x)["text"] for x in lines[:n_per_lang]]
    return out


@torch.no_grad()
def collect(lm: LM, sents, max_tokens, max_pos, rng, stats_device, keep=False):
    """충분통계 누적. keep=True면 벡터 자체(held-out 평가용)를 float16으로 모아 돌려준다."""
    n, sx, xx, xy, kept = 0, None, None, None, []
    for s in sents:
        ids = lm.encode(s)[:max_tokens]
        if len(ids) < 4:
            continue
        hs, _ = lm.run(ids)                                   # [L+1, T, d]
        pos = list(range(2, hs.shape[1]))
        if len(pos) > max_pos:
            pos = sorted(rng.sample(pos, max_pos))
        X = hs[:, pos]                                        # [L+1, P, d]
        if keep:
            kept.append(X.half().cpu())
            continue
        X = X.to(stats_device, torch.float64)
        if sx is None:
            L1, _, d = X.shape
            sx = torch.zeros(L1, d, dtype=torch.float64, device=stats_device)
            xx = torch.zeros(L1 - 1, d, d, dtype=torch.float64, device=stats_device)
            xy = torch.zeros(L1 - 1, d, d, dtype=torch.float64, device=stats_device)
        n += X.shape[1]
        sx += X.sum(1)
        Xt = X[:-1].transpose(1, 2)                           # [L, d, P]
        xx.baddbmm_(Xt, X[:-1])
        xy.baddbmm_(Xt, X[-1].expand(X.shape[0] - 1, -1, -1))
    if keep:
        return torch.cat(kept, 1)                             # [L+1, N, d]
    return n, sx, xx, xy


@torch.no_grad()
def kl_to_final(lm: LM, H: torch.Tensor, layer: int, transform, chunk=256) -> float:
    """held-out 벡터에서 KL(최종 출력 || 렌즈 출력)의 평균 (nats)"""
    tot, cnt = 0.0, 0
    for i in range(0, H.shape[1], chunk):
        hl = H[layer, i:i + chunk].to(lm.device).float()
        hf = H[-1, i:i + chunk].to(lm.device).float()
        p = torch.log_softmax(lm.readout(hf), -1)
        q = torch.log_softmax(lm.readout(transform(hl)), -1)
        tot += (p.exp() * (p - q)).sum(-1).sum().item()
        cnt += hl.shape[0]
    return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="config.MODELS의 짧은 이름 또는 HF id")
    ap.add_argument("--corpus-dir", default=str(DATA_DIR))
    ap.add_argument("--n-per-lang", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-pos", type=int, default=32, help="문장당 쓰는 토큰 위치 수 상한")
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--eval-max", type=int, default=3000, help="KL 평가에 쓰는 held-out 벡터 수 상한")
    ap.add_argument("--lambdas", default="1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--stats-device", default=None, help="충분통계 위치(기본: 모델과 같은 장치)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    name = MODELS.get(args.model, args.model)
    out_dir = Path(args.out_dir or RESULTS_DIR / args.model.replace("/", "_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    lm = LM.from_pretrained(name, device=args.device)
    fit(lm, load_sentences(Path(args.corpus_dir), args.n_per_lang), out_dir, args)


def fit(lm: LM, sents: list[str], out_dir: Path, args):
    rng = random.Random(SEED)
    sents = sents[:]
    rng.shuffle(sents)
    n_hold = max(1, int(len(sents) * args.holdout))
    hold, train = sents[:n_hold], sents[n_hold:]
    sd = args.stats_device or lm.device

    n, sx, xx, xy = collect(lm, train, args.max_tokens, args.max_pos, rng, sd)
    H = collect(lm, hold, args.max_tokens, args.max_pos, rng, sd, keep=True)[:, :args.eval_max]
    L1, d = sx.shape
    print(f"[fit] 학습 벡터 {n}개 ({len(train)}문장), held-out {H.shape[1]}개, 층 {L1 - 1}, d={d}")

    mu = sx / n                                               # [L+1, d]
    C = xx / n - mu[:-1, :, None] * mu[:-1, None, :]
    Cxy = xy / n - mu[:-1, :, None] * mu[-1][None, None, :]
    eye = torch.eye(d, dtype=torch.float64, device=sd)

    mu_f = mu.float().to(lm.device)
    W = torch.eye(d).repeat(L1, 1, 1)
    b = torch.zeros(L1, d)
    rows, best_lam = [], []
    lambdas = [float(x) for x in args.lambdas.split(",")]
    for l in range(L1 - 1):
        kl_v = kl_to_final(lm, H, l, lambda h: h)
        kl_m = kl_to_final(lm, H, l, lambda h, l=l: h - mu_f[l])
        rows += [dict(layer=l, lens="vanilla", kl=kl_v), dict(layer=l, lens="mean", kl=kl_m)]
        best = None
        for lam in lambdas:
            reg = lam * torch.trace(C[l]) / d
            Wl = torch.linalg.solve(C[l] + reg * eye, Cxy[l])
            bl = mu[-1] - mu[l] @ Wl
            Wf, bf = Wl.float().to(lm.device), bl.float().to(lm.device)
            kl = kl_to_final(lm, H, l, lambda h: h @ Wf + bf)
            rows.append(dict(layer=l, lens=f"affine_lam{lam:g}", kl=kl))
            if best is None or kl < best[0]:
                best = (kl, lam, Wl.float().cpu(), bl.float().cpu())
        rows.append(dict(layer=l, lens="affine", kl=best[0], lam=best[1]))
        W[l], b[l] = best[2], best[3]
        best_lam.append(best[1])
        print(f"  층 {l:2d}: KL vanilla {kl_v:.3f}  mean {kl_m:.3f}  affine {best[0]:.3f} (lambda {best[1]:g})")

    torch.save({"mu": mu.float().cpu(), "W": W, "b": b, "lambda": best_lam, "n_train_vectors": n},
               out_dir / "lenses.pt")
    pd.DataFrame(rows).to_csv(out_dir / "lens_fit.csv", index=False)
    print(f"[done] → {out_dir / 'lenses.pt'}, {out_dir / 'lens_fit.csv'}")


if __name__ == "__main__":
    main()
