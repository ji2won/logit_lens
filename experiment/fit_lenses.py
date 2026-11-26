#!/usr/bin/env python3
"""평균 제거 렌즈와 아핀 렌즈의 파라미터를 한·중 위키 문장으로 구한다.

  python -m experiment.fit_lenses --model llama3.2-3b
  → results/<model>/lenses.pt      mu [L+1,d], W [L+1,d,d], b [L+1,d], 층별 lambda
    results/<model>/lens_fit.csv   held-out 문장에서 층별 KL(최종 출력 || 렌즈 출력)

평균 제거: mu_l = 층 l 잔차의 평균 (모든 문장·토큰 위치, BOS와 첫 토큰 제외)
아핀:     h_l → h_L(최종층, 정규화 전) 리지 회귀. 닫힌 해라 학습이 결정적이다.
          W_l = (C_l + lambda * tr(C_l)/d * I)^-1 C_lL,  b_l = mu_L - mu_l W_l
          lambda는 held-out KL이 가장 작은 값을 층마다 고른다.
tuned:    T_l(h) = h + h A_l + b_l. 출력 분포 KL(최종 출력 || 렌즈 출력)을 직접 줄이도록 Adam으로 학습한다.
          A_l = 0, b_l = 0(= 바닐라 로짓 렌즈)에서 시작하고, 학습 문장 중 따로 떼어 둔 검증 문장의 KL이
          --tuned-patience epoch 연속 줄지 않으면 멈춘 뒤 검증 KL이 가장 낮았던 값을 쓴다.
          시작점(epoch 0)도 후보에 넣으므로 최악이어도 바닐라 렌즈와 같다.
          아핀(벡터 MSE)과 달리 단어 확률을 맞추므로 판독이 공백·기호로 무너지지 않는다.
KL 표는 렌즈가 실제로 최종 출력에 더 가까워졌는지 확인하는 용도다(아핀·tuned KL < 바닐라 KL이어야 의미가 있다).

이미 lenses.pt가 있는 모델에 tuned만 더하려면:
  python -m experiment.fit_lenses --model llama3.2-3b --only-tuned
  → lenses.pt에 tuned_A, tuned_b를 더하고 lens_fit.csv의 tuned 행만 바꾼다(평균·아핀은 그대로).
"""
from __future__ import annotations

import argparse
import math
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
def collect(lm: LM, sents, max_tokens, max_pos, rng, stats_device, keep=False, layers=None):
    """충분통계 누적. keep=True면 벡터 자체(held-out 평가용)를 float16으로 모아 돌려준다.
    layers: xx, xy를 쌓을 층 번호(기본: 최종층 제외 전부). sx는 항상 모든 층.
            keep=True일 때 layers를 주면 그 층들 + 최종층만 모은다([len(layers)+1, N, d])."""
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
            kept.append((X if layers is None else X[list(layers) + [-1]]).half().cpu())
            continue
        X = X.to(stats_device).to(torch.float64)   # 장치 이동 먼저 (MPS는 float64 미지원)
        if sx is None:
            L1, _, d = X.shape
            layers = list(range(L1 - 1)) if layers is None else list(layers)
            sx = torch.zeros(L1, d, dtype=torch.float64, device=stats_device)
            xx = torch.zeros(len(layers), d, d, dtype=torch.float64, device=stats_device)
            xy = torch.zeros(len(layers), d, d, dtype=torch.float64, device=stats_device)
        n += X.shape[1]
        sx += X.sum(1)
        Xl = X[layers]                                        # [k, P, d]
        Xt = Xl.transpose(1, 2)                               # [k, d, P]
        xx.baddbmm_(Xt, Xl)
        xy.baddbmm_(Xt, X[-1].expand(len(layers), -1, -1))
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


# ---------------------------------------------------------------- tuned lens
def _readout_grad(lm: LM, h: torch.Tensor) -> torch.Tensor:
    """lm.readout과 같지만 기울기가 흐른다(모델 파라미터는 고정)."""
    return lm.unembed(lm.norm(h.to(lm.unembed.weight.dtype))).float()


def _kl_batch(lm: LM, hl, hf, A, b) -> torch.Tensor:
    """배치 평균 KL(최종 출력 || tuned 렌즈 출력). hl, hf [B, d] float32"""
    with torch.no_grad():
        p = torch.log_softmax(lm.readout(hf), -1)
    q = torch.log_softmax(_readout_grad(lm, hl + hl @ A + b), -1)
    return (p.exp() * (p - q)).sum(-1).mean()


def _val_kl(lm: LM, X, Y, A, b, chunk=256) -> float:
    with torch.no_grad():
        tot = 0.0
        for i in range(0, X.shape[0], chunk):
            hl = X[i:i + chunk].to(lm.device).float()
            hf = Y[i:i + chunk].to(lm.device).float()
            tot += _kl_batch(lm, hl, hf, A, b).item() * hl.shape[0]
        return tot / X.shape[0]


def train_tuned_layer(lm: LM, X: torch.Tensor, Y: torch.Tensor, args, layer: int):
    """한 층의 tuned 변환 학습. X: 층 l 벡터 [N, d], Y: 최종층 벡터 [N, d] (문장 순서대로).
    뒤쪽 --tuned-val 비율(=뒤쪽 문장들)을 검증으로 쓴다. → (A [d,d], b [d], 검증 KL 기록)"""
    N, d = X.shape
    n_val = max(1, int(N * args.tuned_val))
    Xt, Yt, Xv, Yv = X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]
    A = torch.zeros(d, d, device=lm.device, requires_grad=True)
    b = torch.zeros(d, device=lm.device, requires_grad=True)
    opt = torch.optim.Adam([A, b], lr=args.tuned_lr)
    g = torch.Generator().manual_seed(SEED + layer)
    best = (_val_kl(lm, Xv, Yv, A, b), 0, A.detach().clone(), b.detach().clone())   # epoch 0 = 바닐라
    hist, bad = [best[0]], 0
    for ep in range(1, args.tuned_epochs + 1):
        perm = torch.randperm(Xt.shape[0], generator=g)
        for i in range(0, len(perm), args.tuned_batch):
            idx = perm[i:i + args.tuned_batch]
            loss = _kl_batch(lm, Xt[idx].to(lm.device).float(), Yt[idx].to(lm.device).float(), A, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        v = _val_kl(lm, Xv, Yv, A, b)
        hist.append(v)
        if v < best[0] - 1e-4:
            best, bad = (v, ep, A.detach().clone(), b.detach().clone()), 0
        else:
            bad += 1
            if bad >= args.tuned_patience:
                break
    return best[2].cpu(), best[3].cpu(), dict(best_epoch=best[1], val_kl=best[0], epochs_run=len(hist) - 1,
                                              val_kl_start=hist[0], converged=bad >= args.tuned_patience)


def fit_tuned(lm: LM, train: list[str], H: torch.Tensor, args, L1: int, d: int):
    """모든 층(최종층 제외)의 tuned 변환. 층 묶음마다 학습 벡터를 모아 메모리를 아낀다.
    H: held-out 벡터 [L+1, N, d] (보고용 KL은 평균·아핀과 같은 kl_to_final로 잰다)."""
    for p in lm.model.parameters():
        p.requires_grad_(False)
    n_sent = min(len(train), math.ceil(args.tuned_max_vectors / args.max_pos))
    sents = train[:n_sent]
    chunk = getattr(args, "layer_chunk", None) or (7 if str(lm.device).startswith("mps") else L1 - 1)
    groups = [list(range(i, min(i + chunk, L1 - 1))) for i in range(0, L1 - 1, chunk)]
    TA, Tb = torch.zeros(L1, d, d), torch.zeros(L1, d)
    rows, info = [], []
    for layers in groups:
        rng = random.Random(SEED + 1)        # 묶음마다 같은 위치를 뽑는다
        X = collect(lm, sents, args.max_tokens, args.max_pos, rng, "cpu", keep=True, layers=layers)
        for j, l in enumerate(layers):
            A, b, st = train_tuned_layer(lm, X[j], X[-1], args, l)
            TA[l], Tb[l] = A, b
            Af, bf = A.to(lm.device), b.to(lm.device)
            kl = kl_to_final(lm, H, l, lambda h: h + h @ Af + bf)
            rows.append(dict(layer=l, lens="tuned", kl=kl, **st))
            info.append(st)
            print(f"  층 {l:2d}: tuned held-out KL {kl:.3f} (검증 {st['val_kl_start']:.3f} → {st['val_kl']:.3f}, "
                  f"epoch {st['best_epoch']}/{st['epochs_run']}{'' if st['converged'] else ', 조기 종료 전 최대 epoch 도달'})",
                  flush=True)
        del X
    print(f"[tuned] 학습 문장 {n_sent}개, 층당 벡터 최대 {n_sent * args.max_pos}개. "
          f"최대 epoch까지 간(수렴 안 한) 층 {sum(not s['converged'] for s in info)}/{len(info)}")
    return TA, Tb, rows


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
    ap.add_argument("--layer-chunk", type=int, default=None,
                    help="xx, xy를 한 번에 쌓을 층 수(기본: MPS는 7, 그 외는 전부). 메모리가 모자랄 때 줄인다")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    # tuned lens
    ap.add_argument("--tuned-epochs", type=int, default=30, help="0이면 tuned lens를 학습하지 않는다")
    ap.add_argument("--tuned-lr", type=float, default=1e-3)
    ap.add_argument("--tuned-batch", type=int, default=128)
    ap.add_argument("--tuned-patience", type=int, default=3)
    ap.add_argument("--tuned-val", type=float, default=0.1, help="학습 벡터 중 검증(조기 종료)용 비율")
    ap.add_argument("--tuned-max-vectors", type=int, default=40000, help="층당 학습 벡터 수 상한")
    ap.add_argument("--only-tuned", action="store_true",
                    help="기존 lenses.pt의 평균·아핀은 그대로 두고 tuned만 학습해 더한다")
    args = ap.parse_args()

    name = MODELS.get(args.model, args.model)
    out_dir = Path(args.out_dir or RESULTS_DIR / args.model.replace("/", "_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    lm = LM.from_pretrained(name, device=args.device)
    sents = load_sentences(Path(args.corpus_dir), args.n_per_lang)
    if args.only_tuned:
        fit_only_tuned(lm, sents, out_dir, args)
    else:
        fit(lm, sents, out_dir, args)


def _split(sents: list[str], holdout: float):
    rng = random.Random(SEED)
    sents = sents[:]
    rng.shuffle(sents)
    n_hold = max(1, int(len(sents) * holdout))
    return sents[:n_hold], sents[n_hold:]


def fit_only_tuned(lm: LM, sents: list[str], out_dir: Path, args):
    params = torch.load(out_dir / "lenses.pt")
    hold, train = _split(sents, args.holdout)
    H = collect(lm, hold, args.max_tokens, args.max_pos, random.Random(SEED + 2), "cpu", keep=True)[:, :args.eval_max]
    L1, _, d = H.shape
    TA, Tb, rows = fit_tuned(lm, train, H, args, L1, d)
    params.update(tuned_A=TA, tuned_b=Tb)
    torch.save(params, out_dir / "lenses.pt")
    fit_file = out_dir / "lens_fit.csv"
    old = pd.read_csv(fit_file) if fit_file.exists() else pd.DataFrame()
    if len(old):
        old = old[old.lens != "tuned"]
    # 평균·아핀 행은 원래 held-out 위치에서 잰 값이므로, 같은 H에서 바닐라 KL을 따로 남겨 tuned와 바로 비교할 수 있게 한다
    van = [dict(layer=l, lens="vanilla_tunedeval", kl=kl_to_final(lm, H, l, lambda h: h)) for l in range(L1 - 1)]
    old = old[old.lens != "vanilla_tunedeval"] if len(old) else old
    pd.concat([old, pd.DataFrame(rows + van)], ignore_index=True).to_csv(fit_file, index=False)
    print(f"[done] tuned 추가 → {out_dir / 'lenses.pt'}, {fit_file}")


def fit(lm: LM, sents: list[str], out_dir: Path, args):
    rng = random.Random(SEED)
    sents = sents[:]
    rng.shuffle(sents)
    n_hold = max(1, int(len(sents) * args.holdout))
    hold, train = sents[:n_hold], sents[n_hold:]
    # MPS(Apple)는 float64를 지원하지 않으므로 충분통계는 CPU에서 누적
    sd = args.stats_device or ("cpu" if str(lm.device).startswith("mps") else lm.device)

    # 층을 묶음으로 나눠 여러 번 훑으며 xx, xy를 쌓는다(float64 [k,d,d]만 메모리에 둠).
    # 매 묶음마다 rng를 같은 상태에서 시작하므로 뽑는 토큰 위치가 같고, 결과는 한 번에 쌓는 것과 같다.
    L1, _, d = lm.run(lm.encode(train[0])[:4])[0].shape        # 층 수+1, 은닉 차원
    chunk = getattr(args, "layer_chunk", None) or (7 if str(lm.device).startswith("mps") else L1 - 1)
    groups = [list(range(i, min(i + chunk, L1 - 1))) for i in range(0, L1 - 1, chunk)]
    rng_start = rng.getstate()

    W = torch.eye(d).repeat(L1, 1, 1)
    b = torch.zeros(L1, d)
    rows, best_lam = [], []
    lambdas = [float(x) for x in args.lambdas.split(",")]
    H = mu = mu_f = eye = None
    for gi, layers in enumerate(groups):
        rng.setstate(rng_start)
        n, sx, xx, xy = collect(lm, train, args.max_tokens, args.max_pos, rng, sd, layers=layers)
        if H is None:   # 첫 묶음 직후 rng 상태 = 원래 학습 수집 직후 상태 → held-out 위치도 원래와 같음
            H = collect(lm, hold, args.max_tokens, args.max_pos, rng, sd, keep=True)[:, :args.eval_max].clone()
            print(f"[fit] 학습 벡터 {n}개 ({len(train)}문장), held-out {H.shape[1]}개, 층 {L1 - 1}, d={d}, "
                  f"층 묶음 {len(groups)}개")
            mu = sx / n                                       # [L+1, d]
            mu_f = mu.float().to(lm.device)
            eye = torch.eye(d, dtype=torch.float64, device=sd)
        # 공분산을 xx, xy 자리에서 층마다 계산(사본·외적 임시값 없음)
        for j, l in enumerate(layers):
            xx[j].div_(n).sub_(torch.outer(mu[l], mu[l]))
            xy[j].div_(n).sub_(torch.outer(mu[l], mu[-1]))
        for j, l in enumerate(layers):
            C, Cxy = xx[j], xy[j]
            kl_v = kl_to_final(lm, H, l, lambda h: h)
            kl_m = kl_to_final(lm, H, l, lambda h, l=l: h - mu_f[l])
            rows += [dict(layer=l, lens="vanilla", kl=kl_v), dict(layer=l, lens="mean", kl=kl_m)]
            best = None
            for lam in lambdas:
                reg = lam * torch.trace(C) / d
                Wl = torch.linalg.solve(C + reg * eye, Cxy)
                bl = mu[-1] - mu[l] @ Wl
                Wf, bf = Wl.float().to(lm.device), bl.float().to(lm.device)
                kl = kl_to_final(lm, H, l, lambda h: h @ Wf + bf)
                rows.append(dict(layer=l, lens=f"affine_lam{lam:g}", kl=kl))
                if best is None or kl < best[0]:
                    best = (kl, lam, Wl.float().cpu(), bl.float().cpu())
            rows.append(dict(layer=l, lens="affine", kl=best[0], lam=best[1]))
            W[l], b[l] = best[2], best[3]
            best_lam.append(best[1])
            print(f"  층 {l:2d}: KL vanilla {kl_v:.3f}  mean {kl_m:.3f}  affine {best[0]:.3f} (lambda {best[1]:g})",
                  flush=True)
        del xx, xy

    params = {"mu": mu.float().cpu(), "W": W, "b": b, "lambda": best_lam, "n_train_vectors": n}
    if getattr(args, "tuned_epochs", 0) > 0:
        TA, Tb, trows = fit_tuned(lm, train, H, args, L1, d)
        params.update(tuned_A=TA, tuned_b=Tb)
        rows += trows
    torch.save(params, out_dir / "lenses.pt")
    pd.DataFrame(rows).to_csv(out_dir / "lens_fit.csv", index=False)
    print(f"[done] → {out_dir / 'lenses.pt'}, {out_dir / 'lens_fit.csv'}")


if __name__ == "__main__":
    main()
