#!/usr/bin/env python3
"""인과 개입: 중간층 잔차에서 영어 단어 방향을 지웠을 때 최종 중국어 출력이 얼마나 무너지는가.

렌즈로 '읽히는' 영어(decodability)와 그 영어가 출력에 '쓰이는' 정도(causal effect)를 따로 잰다.
  부풀림 = 읽히는 양은 큰데, 지워도 출력이 별로 안 바뀌는 경우.

방법
  1) 깨끗한 순전파로 각 프롬프트 마지막 위치의 바닐라 렌즈 분포를 구간(band) 층에서 평균해
     영어 토큰(라틴 2자 이상) 상위 30개를 모은다. 여러 프롬프트에 공통으로 뜨는 토큰(상위 30에
     드는 프롬프트 비율 ≥ --generic-frac)은 '공통 조각'(arious, rani …)으로 분류한다.
  2) 문항별 방향 묶음 (모두 같은 개수 k로 맞춘다)
       en_item     이 문항의 영어: probe 단어(spring)·정답 뜻 묶음(season summer winter)의 토큰
                   + 렌즈 상위 영어 중 공통 조각이 아닌 것. 렌즈가 '읽어 낸' 그 영어.
       en_other    다른 문항의 en_item (영어이긴 하지만 뜻이 무관) → 핵심 대조
       en_generic  공통 조각 상위 k개 (문항과 무관하게 늘 뜨는 영어)
       random      무작위 방향 k개
       ko_input    입력 한국어 단어의 토큰
       zh_answer   중국어 정답 단어의 토큰 (출력 토큰 자체를 지우면 어떻게 되는지 참고용)
     방향은 언임베딩 행에 최종 RMSNorm 이득을 곱한 것(w_t = γ ⊙ U_t; 로짓 = w_t·h/rms)이고,
     QR로 정규직교화한 부분공간 P에 대해 h ← h − P h 를 구간의 모든 층 출력·모든 위치에 적용한다.
  3) 측정: log p(중국어 정답), log p(중국어 방해어) (teacher forcing, 토큰 합),
     프롬프트 마지막 위치 최종 top-1이 바뀌었는지, 지운 성분 비율 ‖Ph‖/‖h‖ (구간 평균, 마지막 위치).

  python -m experiment.ablate --model llama3.2-3b --device mps
  → results/<model>/ablate/ablate_raw.csv, directions.csv, clean_lens.csv
"""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

from . import prompts
from .config import DATASETS, MID_LAYER_RANGE, MODELS, RESULTS_DIR, SEED, VERSIONS
from .data import load
from .lenses import probe_token_ids
from .lm import LM

BANDS = {"mid": MID_LAYER_RANGE, "late": (0.75, 0.97)}
CONDS = ["en_item", "en_other", "en_generic", "random", "ko_input", "zh_answer"]


def band_layers(n_layers, frac):
    return list(range(math.ceil(frac[0] * n_layers), math.floor(frac[1] * n_layers) + 1))


def decoder_layers(model):
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError("디코더 층 목록을 찾지 못함")


class Ablator:
    """hidden_states 인덱스 l(=l-1번째 블록 출력)에 h ← h − (hQ)Qᵀ 를 건다."""

    def __init__(self, lm: LM):
        self.lm = lm
        self.blocks = decoder_layers(lm.model)
        self.Q = None
        self.active = set()
        self.removed = []          # 프롬프트 마지막 위치 ‖Ph‖/‖h‖
        self.n_prompt = None
        for i, b in enumerate(self.blocks):
            b.register_forward_hook(self._make_hook(i + 1))

    def _make_hook(self, hs_index):
        def hook(module, args, out):
            if self.Q is None or hs_index not in self.active:
                return None
            h = out[0] if isinstance(out, tuple) else out
            n = self.n_prompt or h.shape[1]          # 프롬프트 위치에만 건다(정답 토큰 위치는 그대로)
            hf = h.float()
            proj = (hf[:, :n] @ self.Q) @ self.Q.T
            self.removed.append((proj[0, -1].norm() / hf[0, n - 1].norm().clamp_min(1e-6)).item())
            new = hf.clone()
            new[:, :n] -= proj
            new = new.to(h.dtype)
            return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new
        return hook

    def set(self, Q, layers):
        self.Q, self.active, self.removed = Q, set(layers), []

    def clear(self):
        self.Q, self.active, self.removed = None, set(), []


@torch.no_grad()
def score(lm: LM, prompt_ids, cont_ids, abl=None):
    """→ (log p 합, 첫 토큰 log p, 프롬프트 마지막 위치 top-1 id)"""
    if abl is not None:
        abl.n_prompt = len(prompt_ids)
    x = torch.tensor([prompt_ids + cont_ids], device=lm.device)
    logits = lm.model(input_ids=x, use_cache=False).logits[0].float()
    lp = torch.log_softmax(logits, -1)
    n = len(prompt_ids)
    s = sum(lp[n - 1 + i, t].item() for i, t in enumerate(cont_ids))
    return s, lp[n - 1, cont_ids[0]].item(), int(logits[n - 1].argmax())


def subspace(lm: LM, ids, k, gen=None):
    """토큰 id 목록 → 정규직교 기저 [d, k'] (w_t = γ ⊙ U_t). ids=None이면 무작위 k개."""
    W = lm.unembed.weight
    d = W.shape[1]
    if ids is None:
        M = torch.randn(k, d, generator=gen).to(lm.device)
    else:
        ids = list(ids)[:k]
        if not ids:
            return None
        M = W[ids].float()
        g = getattr(lm.norm, "weight", None)
        if g is not None:
            M = M * g.float()
    Q, _ = torch.linalg.qr(M.float().T.cpu())    # [d, k'] (MPS는 QR 미지원일 수 있어 CPU에서)
    return Q.to(lm.device)


def word_ids(lm, words):
    out = []
    for w in words:
        for i in probe_token_ids(lm, w)[0]:
            if i not in out:
                out.append(i)
    return out


def text_ids(lm, word):
    """한국어·중국어 단어: 앞 공백 없는 토큰 열 전체(첫 토큰만 쓰면 조각이 겹칠 수 있어 전부 쓴다)."""
    return list(dict.fromkeys(lm.encode(word, special=False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--versions", nargs="+", default=VERSIONS)
    ap.add_argument("--bands", nargs="+", default=list(BANDS))
    ap.add_argument("--k", type=int, default=12, help="방향 묶음 크기(모든 조건 공통)")
    ap.add_argument("--generic-frac", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or RESULTS_DIR / args.model / "ablate")
    out_dir.mkdir(parents=True, exist_ok=True)
    lm = LM.from_pretrained(MODELS[args.model], device=args.device)
    abl = Ablator(lm)
    L = lm.n_layers
    bands = {b: band_layers(L, BANDS[b]) for b in args.bands}
    print("[bands]", {b: (v[0], v[-1]) for b, v in bands.items()})

    items = load("A", DATASETS["A"])[: args.limit]
    prompts.check_fewshot(items)
    en_mask = torch.load(RESULTS_DIR / args.model / "lang_masks.pt")["en"].to(lm.device)

    # ---- 1) 깨끗한 순전파: 렌즈 상위 영어 토큰
    clean = {}
    rows = []
    for it in items:
        for v in args.versions:
            p = prompts.build("translate", v, it.ko)
            ids = lm.encode(p)
            hs, _ = lm.run(ids)
            for b, lay in bands.items():
                lp = torch.log_softmax(lm.readout(hs[lay, -1]), -1)          # [nb, V]
                pm = lp.exp().mean(0) * en_mask                               # 구간 평균 확률, 영어만
                top = pm.topk(30)
                clean[(it.id, v, b)] = (top.indices.tolist(), top.values.tolist(), lp)
    # 공통 조각
    generic = {}
    for b in bands:
        keys = [k for k in clean if k[2] == b]
        cnt = Counter(i for k in keys for i in clean[k][0])
        generic[b] = [i for i, c in cnt.most_common() if c / len(keys) >= args.generic_frac]
        print(f"[generic {b}] {len(generic[b])}개:", [lm.decode([i]) for i in generic[b][:20]])

    # ---- 2) 방향 묶음
    sense = {}
    sp = Path(DATASETS["A"]).with_name("A_sense_probes.csv")
    if sp.exists():
        sd = pd.read_csv(sp, keep_default_na=False)
        sense = dict(zip(sd.id, sd.en_target_set))
    dir_ids, dir_rows = {}, []
    for it in items:
        probe = word_ids(lm, [it.probes["poly"]] + sense.get(it.id, "").split())
        for v in args.versions:
            for b, lay in bands.items():
                top, vals, lp = clean[(it.id, v, b)]
                lens_pick = [i for i in top if i not in generic[b] and i not in probe]
                en_item = (probe + lens_pick)[: args.k]
                dir_ids[(it.id, v, b, "en_item")] = en_item
                dir_ids[(it.id, v, b, "en_generic")] = generic[b][: len(en_item)]
                dir_ids[(it.id, v, b, "ko_input")] = text_ids(lm, it.ko)
                dir_ids[(it.id, v, b, "zh_answer")] = text_ids(lm, it.zh_answer)
                # 렌즈 decodability: en_item 토큰 확률 합의 구간 최댓값
                dec = lp[:, en_item].exp().sum(-1).max().item()
                dec_probe = lp[:, probe].exp().sum(-1).max().item() if probe else float("nan")
                dir_rows.append(dict(id=it.id, en=it.probes["poly"], ko=it.ko, version=v, band=b,
                                     n_en_item=len(en_item), decod_en_item=dec, decod_probe=dec_probe,
                                     en_item=" | ".join(lm.decode([i]).strip() for i in en_item)))
    # en_other: 문항 순서를 절반 돌려 짝지음(같은 버전·구간)
    n = len(items)
    for j, it in enumerate(items):
        other = items[(j + n // 2) % n]
        for v in args.versions:
            for b in bands:
                k_item = len(dir_ids[(it.id, v, b, "en_item")])
                o = [i for i in dir_ids[(other.id, v, b, "en_item")]
                     if i not in dir_ids[(it.id, v, b, "en_item")]]
                dir_ids[(it.id, v, b, "en_other")] = o[:k_item]
                dir_ids[(it.id, v, b, "_other_id")] = other.id
    pd.DataFrame(dir_rows).to_csv(out_dir / "directions.csv", index=False)

    # ---- 3) 개입
    gen = torch.Generator().manual_seed(SEED)
    raw = []
    t0 = time.time()
    for j, it in enumerate(items):
        ans = lm.encode(it.zh_answer, special=False)
        dis = lm.encode(it.zh_distractor, special=False)
        for v in args.versions:
            pids = lm.encode(prompts.build("translate", v, it.ko))
            abl.clear()
            la0, la10, top0 = score(lm, pids, ans, abl)
            ld0, ld10, _ = score(lm, pids, dis, abl)
            raw.append(dict(id=it.id, version=v, band="none", cond="clean", k=0, lp_answer=la0,
                            lp_answer_first=la10, lp_distractor=ld0, lp_distractor_first=ld10, top1_changed=False, removed_frac=0.0))
            for b, lay in bands.items():
                k_item = len(dir_ids[(it.id, v, b, "en_item")])
                for c in CONDS:
                    if c == "random":
                        Q = subspace(lm, None, k_item, gen)
                    else:
                        Q = subspace(lm, dir_ids[(it.id, v, b, c)], args.k)
                    if Q is None:
                        continue
                    abl.set(Q, lay)
                    la, la1, top = score(lm, pids, ans, abl)
                    rem = sum(abl.removed) / max(len(abl.removed), 1)
                    abl.set(Q, lay)
                    ld, ld1, _ = score(lm, pids, dis, abl)
                    raw.append(dict(id=it.id, version=v, band=b, cond=c, k=Q.shape[1], lp_answer=la,
                                    lp_answer_first=la1, lp_distractor=ld, lp_distractor_first=ld1, top1_changed=top != top0, removed_frac=rem,
                                    other_id=dir_ids[(it.id, v, b, "_other_id")] if c == "en_other" else ""))
            abl.clear()
        if (j + 1) % 10 == 0 or j + 1 == len(items):
            print(f"  {j + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(raw).to_csv(out_dir / "ablate_raw.csv", index=False)
    pd.DataFrame(raw).to_csv(out_dir / "ablate_raw.csv", index=False)
    print(f"[done] → {out_dir}")


if __name__ == "__main__":
    main()
