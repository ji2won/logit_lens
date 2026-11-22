"""세 가지 렌즈와 렌즈 출력에서 영어 점수를 읽는 도구.

모두 읽을 때만 적용하고 모델 계산은 건드리지 않는다. 읽기 = unembed(final_norm(T_l(h)))
  vanilla  T_l(h) = h                          (바닐라 로짓 렌즈)
  mean     T_l(h) = h - mu_l                   (가설 1: 입력과 무관한 평균 성분 제거)
  affine   T_l(h) = h W_l + b_l                (가설 2: 중간층 기저를 최종층 기저로 맞춤)
mu_l, W_l, b_l은 fit_lenses.py가 한·중 위키 문장으로 구한다.
"""
from __future__ import annotations

import re
from pathlib import Path

import torch


class Lens:
    name = "base"

    def transform(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VanillaLens(Lens):
    name = "vanilla"

    def transform(self, layer, h):
        return h


class MeanLens(Lens):
    name = "mean"

    def __init__(self, mu: torch.Tensor, add_final_mean: bool = False):
        self.mu = mu                      # [L+1, d]
        self.add_final_mean = add_final_mean

    def transform(self, layer, h):
        out = h - self.mu[layer].to(h)
        if self.add_final_mean:
            out = out + self.mu[-1].to(h)
        return out


class AffineLens(Lens):
    name = "affine"

    def __init__(self, W: torch.Tensor, b: torch.Tensor):
        self.W, self.b = W, b             # [L+1, d, d], [L+1, d]; 마지막 층은 항등

    def transform(self, layer, h):
        return h @ self.W[layer].to(h) + self.b[layer].to(h)


def load_lenses(names, lens_file: Path | None, device="cpu") -> list[Lens]:
    params = torch.load(lens_file, map_location=device) if lens_file and Path(lens_file).exists() else None
    out = []
    for n in names:
        if n == "vanilla":
            out.append(VanillaLens())
        elif params is None:
            raise FileNotFoundError(f"{n} 렌즈에는 fit_lenses.py 결과({lens_file})가 필요함")
        elif n == "mean":
            out.append(MeanLens(params["mu"]))
        elif n == "mean_shift":
            m = MeanLens(params["mu"], add_final_mean=True)
            m.name = "mean_shift"
            out.append(m)
        elif n == "affine":
            out.append(AffineLens(params["W"], params["b"]))
        else:
            raise ValueError(n)
    return out


@torch.no_grad()
def lens_logprobs(lm, lens: Lens, hs: torch.Tensor) -> torch.Tensor:
    """hs [L+1, d] (한 위치의 전 층 잔차) → 층별 로그확률 [L+1, V]"""
    h = torch.stack([lens.transform(l, hs[l]) for l in range(hs.shape[0])])
    return torch.log_softmax(lm.readout(h), -1)


# ---------- 영어 단어 → 토큰 ----------
def probe_token_ids(lm, word: str) -> tuple[list[int], bool]:
    """영어 단어의 첫 토큰 후보(앞 공백 유무 × 소문자/첫 글자 대문자).
    → (토큰 id 목록, ' word'가 1토큰인지)"""
    ids, single = set(), False
    for w in {word, word.lower(), word.capitalize()}:
        for s in (w, " " + w):
            t = lm.encode(s, special=False)
            if t:
                ids.add(t[0])
            if s == " " + word and len(t) == 1:
                single = True
    return sorted(ids), single


def probe_logprob(logprobs: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """[L+1, V] → [L+1]: 후보 토큰 확률의 합(log)"""
    return torch.logsumexp(logprobs[:, ids], -1)


# ---------- 언어별 어휘 마스크 ----------
_EN = re.compile(r"^[A-Za-z]{2,}$")
_HANGUL = re.compile(r"[가-힣]")
_CJK = re.compile(r"[一-鿿]")


def language_masks(lm) -> dict[str, torch.Tensor]:
    """어휘 전체를 영어(라틴 문자 2자 이상)/한국어(한글 포함)/중국어(한자 포함, 한글 없음)로 분류."""
    V = lm.vocab_size
    en = torch.zeros(V, dtype=torch.bool)
    ko = torch.zeros(V, dtype=torch.bool)
    zh = torch.zeros(V, dtype=torch.bool)
    for i in range(V):
        s = lm.decode([i]).strip()
        if _EN.match(s):
            en[i] = True
        elif _HANGUL.search(s):
            ko[i] = True
        elif _CJK.search(s):
            zh[i] = True
    return {"en": en, "ko": ko, "zh": zh}


def mass(logprobs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[L+1, V] → [L+1]: 마스크 토큰 확률 합(log)"""
    return torch.logsumexp(logprobs[:, mask.to(logprobs.device)], -1)
