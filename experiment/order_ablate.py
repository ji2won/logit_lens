#!/usr/bin/env python3
"""확정 전에 뜨는 영어 이웃(여름 → season, sunshine)이 영어 단어(summer)를 거쳐 나온 것인지,
한국어 입력에서 바로 나온 것인지 양방향 방향 제거로 가린다.

  python -m experiment.order_ablate --model llama3.2-3b --device mps
  python -m experiment.order_ablate --model llama3.2-3b --analyze-only
  → results/<model>/A_en_run/order_ablate/{units.csv, raw.csv, en_input.csv, summary.md}

단위 = (문항, 입력 단어) — A_en의 입력어와 대조어(ko_control_final). 각 단위에서
  W = 그 단어의 영어 계열 토큰 (summer, Summer, summers …)
  N = 확정 전 층(15 ~ 확정-1)에 뜬, W가 아닌 영어 단어 (season, sunshine …)
  확정 층 = 바닐라 렌즈에서 W 확률 합이 0.3을 처음 넘는 층 (깨끗한 순전파, 프롬프트 4개 각각)
조건 (10층 ~ 확정-1층의 블록 출력, 프롬프트 전 위치에서 h ← h − QQᵀh)
  abl_W       W 방향 제거. 단 N 방향에 직교화해서 N의 로짓을 직접 건드리지 않는다
  abl_N       N 방향 제거. W 방향에 직교화
  abl_otherW  다른 단위의 W (현재 W·N에 직교화, 차원 수를 W와 맞춤) — 기준선
  abl_otherN  다른 단위의 N (같은 방식, 차원 수를 N과 맞춤) — 기준선
측정 (바닐라 렌즈, 프롬프트 마지막 위치)
  N_pre   확정-3 ~ 확정-1층 N 확률 합 평균      → abl_W에서 줄면 "summer를 보고 season이 뜬다"
  W_post  확정 ~ 확정+2층 W 확률 합 평균        → abl_N에서 줄면 "season을 거쳐 summer로 좁혀진다"
  onset   W 확률 합이 0.3을 넘는 첫 층 (없으면 28)
  lp_final 깨끗한 순전파의 최종 1위 토큰 log p
보조 관찰: 같은 단위를 영어 단어로 직접 넣은 영→중 프롬프트(en_input.csv)에서 N이 뜨는지.
"""
from __future__ import annotations

import argparse
import math
import re

import numpy as np
import pandas as pd
import torch

from . import prompts
from .ablate import Ablator
from .ablate_analysis import md_table
from .config import DATASETS, MODELS, N_BOOT, RESULTS_DIR, ROOT, SEED, VERSIONS
from .data import load_A
from .lenses import VanillaLens, lens_logprobs
from .lm import LM

LAT = re.compile(r"^[a-z]{3,}$")
ABL_FROM = 10
COND = ["abl_W", "abl_N", "abl_otherW", "abl_otherN"]

# 영어 → 중국어 (보조 관찰). 형식은 prompts._TRANSLATE와 같고 원문 언어만 영어
EN2ZH = {
    "v1": ("", lambda s: f"English：{s}\n中文："),
    "v2": ("다음 영어 단어를 중국어로 번역하세요.\n", lambda s: f"{s}→"),
    "v3": ("请把英语词翻译成中文。\n", lambda s: f"英语：{s}\n中文："),
    "v4": ("", lambda s: f"「{s}」의 중국어는「"),
}
EN2ZH_POST = {"v1": "", "v2": "", "v3": "", "v4": "」"}


def en2zh_prompt(v, word):
    head, fmt = EN2ZH[v]
    return head + "\n".join(fmt(en) + zh + EN2ZH_POST[v] for _, zh, en in prompts.FEWSHOT) + "\n" + fmt(word)


def fam_fn(en):
    en = en.lower()
    stem = en[:max(3, min(len(en) - 1, 5))]
    return lambda x: x.startswith(stem) or (en.startswith(x) and len(x) >= 3)


def single_ids(lm, words):
    """단어 → 1토큰으로 인코딩되는 변형(앞 공백 유무 × 소문자/대문자)의 id. 여러 토큰 단어는 버린다."""
    out = []
    for w in words:
        for v in {w, w.lower(), w.capitalize()}:
            for s in (v, " " + v):
                t = lm.encode(s, special=False)
                if len(t) == 1 and t[0] not in out:
                    out.append(t[0])
    return out


def basis(lm, ids, against=(), rank=None):
    """γ⊙U[ids]의 행 공간에서 against 행 공간을 뺀 정규직교 기저 [d, r]."""
    W = lm.unembed.weight
    g = lm.norm.weight.float()
    M = (W[list(ids)].float() * g).cpu()
    if len(against):
        A = (W[list(against)].float() * g).cpu()
        Qa = torch.linalg.qr(A.T)[0]
        M = M - (M @ Qa) @ Qa.T
    U, S, _ = torch.linalg.svd(M.T, full_matrices=False)
    r = int((S > S[0] * 1e-3).sum()) if len(S) else 0
    if rank is not None:
        r = min(r, rank)
    return U[:, :r].contiguous(), r


def pick_units(topk_file, items, min_n=0.015):
    """zh_topk.csv(깨끗한 렌즈 상위 10)에서 단위별 W 단어, N 단어, 확정 층 후보를 고른다."""
    t = pd.read_csv(topk_file, keep_default_na=False)
    t["tok"] = t.token.str.strip().str.lower().str.lstrip("-")
    own = {}
    for it in items:
        own[(it.id, it.ko)] = it.en_hint
        for c, e in zip(it.controls, it.en_controls or [""] * len(it.controls)):
            own[(it.id, c)] = e
    rows = []
    for (i, w, role), g in t.groupby(["id", "word", "role"], sort=False):
        en = own.get((i, w), "")
        if not en:
            continue
        fam = fam_fn(en)
        dist = {l: gl.groupby("tok").p.sum() / gl.version.nunique() for l, gl in g.groupby("layer")}
        pm = {l: sum(v for k, v in d.items() if fam(k)) for l, d in dist.items()}
        onset = next((l for l in range(15, 28) if pm.get(l, 0) >= 0.3), None)
        if onset is None or not 18 <= onset <= 25:
            continue
        pre = pd.Series(dtype=float)
        for l in range(15, onset):
            d = dist[l]
            pre = pre.add(d[[bool(LAT.match(k)) and not fam(k) for k in d.index]], fill_value=0)
        pre = (pre / (onset - 15)).sort_values(ascending=False)
        pre = pre[pre >= 0.005].head(4)
        if pre.sum() < min_n:
            continue
        stem = en.lower()[:max(3, min(len(en) - 1, 5))]          # W 단어는 어간으로 시작하는 온전한 단어만 ('sum' 같은 조각 제외)
        wwords = sorted({k for d in dist.values() for k in d.index if LAT.match(k) and k.startswith(stem)} | {en.lower()})
        rows.append(dict(id=i, word=w, role=role, en=en, onset_topk=onset, W_words=" ".join(wwords),
                         N_words=" ".join(pre.index), N_mass_topk=pre.sum()))
    return pd.DataFrame(rows)


class PostCapture:
    """블록 출력을 Ablator 훅 '뒤에서' 받는다. HF의 output_hidden_states는 이 모델에서 제거 전 값을 기록하므로
    (같은 층 안의 제거가 안 보임) 렌즈는 이 값으로 읽는다. Ablator보다 나중에 등록해야 한다."""

    def __init__(self, blocks):
        self.h = [None] * len(blocks)
        for i, b in enumerate(blocks):
            b.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def hook(module, args, out):
            h = out[0] if isinstance(out, tuple) else out
            self.h[i] = h[0, -1].float()
        return hook

    def hs_last(self, hs):
        """lm.run의 hs [L+1, T, d] → 마지막 위치 [L+1, d]. 0층(임베딩)과 마지막(정규화 직전)은 lm.run 값."""
        out = hs[:, -1].clone()
        for i in range(len(self.h) - 1):
            out[i + 1] = self.h[i]
        return out


def lens_stats(lm, hs, W_ids, N_ids):
    P = lens_logprobs(lm, VanillaLens(), hs).exp()            # [L+1, V]
    return P[:, W_ids].sum(1).cpu().numpy(), P[:, N_ids].sum(1).cpu().numpy()


def run(args, out):
    items = [it for it in load_A(DATASETS["A"]) if it.id in set(pd.read_csv(ROOT / "A_en.csv").id)]
    units = pick_units(RESULTS_DIR / args.model / "A_en_run" / "zh_topk.csv", items)
    if args.limit:
        units = units.head(args.limit)
    lm = LM.from_pretrained(MODELS[args.model], device=args.device)
    abl = Ablator(lm)
    cap = PostCapture(abl.blocks)
    units["W_ids"] = [single_ids(lm, s.split()) for s in units.W_words]
    units["N_ids"] = [[i for i in single_ids(lm, s.split()) if i not in set(w)] for s, w in zip(units.N_words, units.W_ids)]
    units = units[(units.W_ids.str.len() > 0) & (units.N_ids.str.len() > 0)].reset_index(drop=True)
    n = len(units)
    print(f"[units] {n}개 (입력어 {(units.role == 'input').sum()}, 대조어 {(units.role == 'control').sum()})")

    raw, eni = [], []
    for k, u in units.iterrows():
        o = units.iloc[(k + n // 2) % n]
        if o.id == u.id:
            o = units.iloc[(k + n // 2 + 1) % n]
        own = set(u.W_ids) | set(u.N_ids)
        oW = [i for i in o.W_ids if i not in own]
        oN = [i for i in o.N_ids if i not in own]
        QW, rW = basis(lm, u.W_ids, u.N_ids)
        QN, rN = basis(lm, u.N_ids, u.W_ids)
        QoW, roW = basis(lm, oW, list(own), rank=rW) if oW else (None, 0)
        QoN, roN = basis(lm, oN, list(own), rank=rN) if oN else (None, 0)
        Qs = {"abl_W": (QW, rW), "abl_N": (QN, rN), "abl_otherW": (QoW, roW), "abl_otherN": (QoN, roN)}
        for v in VERSIONS:
            ids = lm.encode(prompts.build("translate", v, u.word))
            abl.clear()
            hs, logits = lm.run(ids)
            Wc, Nc = lens_stats(lm, cap.hs_last(hs), u.W_ids, u.N_ids)
            lpc = torch.log_softmax(logits[-1], -1)
            top = int(lpc.argmax())
            onset = next((l for l in range(15, 28) if Wc[l] >= 0.3), None)
            base = dict(id=u.id, word=u.word, role=u.role, en=u.en, version=v, onset_clean=onset)
            raw.append(dict(base, cond="clean", rank=0, removed=0.0, lp_final=lpc[top].item(),
                            **{f"W{l}": Wc[l] for l in range(29)}, **{f"N{l}": Nc[l] for l in range(29)}))
            if onset is None or onset - 1 < ABL_FROM:
                continue
            layers = list(range(ABL_FROM, onset))
            for c, (Q, r) in Qs.items():
                if Q is None or r == 0:
                    continue
                abl.set(Q.to(lm.device), layers)
                hs2, lg2 = lm.run(ids)
                W2, N2 = lens_stats(lm, cap.hs_last(hs2), u.W_ids, u.N_ids)
                rem = float(np.mean(abl.removed)) if abl.removed else 0.0
                abl.clear()
                raw.append(dict(base, cond=c, rank=r, removed=rem,
                                lp_final=torch.log_softmax(lg2[-1], -1)[top].item(),
                                **{f"W{l}": W2[l] for l in range(29)}, **{f"N{l}": N2[l] for l in range(29)}))
            # 보조: 영어 단어 직접 입력
            hs3, _ = lm.run(lm.encode(en2zh_prompt(v, u.en)))
            W3, N3 = lens_stats(lm, cap.hs_last(hs3), u.W_ids, u.N_ids)
            eni.append(dict(base, **{f"W{l}": W3[l] for l in range(29)}, **{f"N{l}": N3[l] for l in range(29)}))
        if (k + 1) % 10 == 0 or k + 1 == n:
            print(f"  {k + 1}/{n}", flush=True)
            pd.DataFrame(raw).to_csv(out / "raw.csv", index=False)
    units.assign(W_ids=units.W_ids.map(lambda x: " ".join(map(str, x))),
                 N_ids=units.N_ids.map(lambda x: " ".join(map(str, x)))).to_csv(out / "units.csv", index=False)
    pd.DataFrame(raw).to_csv(out / "raw.csv", index=False)
    pd.DataFrame(eni).to_csv(out / "en_input.csv", index=False)


def boot(x, n=N_BOOT, seed=SEED):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return f"{x.mean():.3f} [{np.percentile(m, 2.5):.3f}, {np.percentile(m, 97.5):.3f}]"


def analyze(out):
    r = pd.read_csv(out / "raw.csv", keep_default_na=False)
    r["onset_clean"] = pd.to_numeric(r.onset_clean, errors="coerce")
    r = r[r.onset_clean.notna()].copy()

    def meas(row, which, lo, hi):
        o = int(row.onset_clean)
        return np.mean([float(row[f"{which}{l}"]) for l in range(o + lo, o + hi + 1)])

    r["N_pre"] = r.apply(lambda x: meas(x, "N", -3, -1), axis=1)
    r["W_pre"] = r.apply(lambda x: meas(x, "W", -3, -1), axis=1)
    r["W_post"] = r.apply(lambda x: meas(x, "W", 0, 2), axis=1)
    r["onset"] = r.apply(lambda x: next((l for l in range(15, 28) if float(x[f"W{l}"]) >= 0.3), 28), axis=1)
    key = ["id", "word", "version"]
    c = r[r.cond == "clean"].set_index(key)
    x = r[r.cond != "clean"].copy()
    for m in ["N_pre", "W_pre", "W_post", "onset", "lp_final"]:
        x["d_" + m] = x[m].values - c.loc[list(zip(x.id, x.word, x.version)), m].values
    # N_pre 비율 변화: log(개입/깨끗) — 확률 규모가 단위마다 달라서 비율로도 본다
    x["lr_N_pre"] = np.log(x.N_pre.clip(1e-6).values / c.loc[list(zip(x.id, x.word, x.version)), "N_pre"].clip(1e-6).values)
    x["lr_W_post"] = np.log(x.W_post.clip(1e-6).values / c.loc[list(zip(x.id, x.word, x.version)), "W_post"].clip(1e-6).values)
    u = x.groupby(["id", "word", "role", "cond"], as_index=False)[
        ["d_N_pre", "lr_N_pre", "d_W_pre", "d_W_post", "lr_W_post", "d_onset", "d_lp_final", "removed", "rank"]].mean()
    cu = c.reset_index().groupby(["id", "word"])[["N_pre", "W_pre", "W_post", "onset"]].mean()

    md = ["# 확정 전 영어 이웃은 영어 단어를 거쳐 나오나 (양방향 방향 제거, Llama)\n",
          f"단위 {u[['id', 'word']].drop_duplicates().shape[0]}개 (입력어 {u[u.role == 'input'].word.nunique()}·"
          f"대조어 {u[u.role == 'control'][['id', 'word']].drop_duplicates().shape[0]}). "
          f"제거 구간 {ABL_FROM}층 ~ 확정-1층, 프롬프트 전 위치. 프롬프트 4개 평균 후 단위 평균, 95% bootstrap CI.\n",
          f"깨끗한 상태 평균: N_pre {cu.N_pre.mean():.3f}, W_pre {cu.W_pre.mean():.3f}, W_post {cu.W_post.mean():.3f}, "
          f"확정 층 {cu.onset.mean():.1f}\n",
          "- N_pre = 확정 직전 3개 층의 이웃(N) 확률 합, W_post = 확정부터 3개 층의 단어(W) 확률 합. "
          "lr_ = log(개입 후/깨끗) — 0이면 변화 없음, −0.69면 절반.\n"]
    rows = []
    for cond in COND:
        g = u[u.cond == cond]
        rows.append(dict(조건=cond, n=len(g), 차원=round(g["rank"].mean(), 1), 지운비율=f"{g.removed.mean():.3f}",
                         ΔN_pre=boot(g.d_N_pre), lr_N_pre=boot(g.lr_N_pre), ΔW_post=boot(g.d_W_post),
                         lr_W_post=boot(g.lr_W_post), Δ확정층=boot(g.d_onset), Δlp_final=boot(g.d_lp_final)))
    md += ["## 1. 조건별 효과\n", md_table(pd.DataFrame(rows)), ""]

    w = u.pivot_table(index=["id", "word", "role"], columns="cond",
                      values=["lr_N_pre", "lr_W_post", "d_onset"]).reset_index()
    rows = []
    for role in ["전체", "input", "control"]:
        g = w if role == "전체" else w[w.role == role]
        rows.append(dict(단위=role, n=len(g),
                         W제거가_N에_미친_효과=boot(g[("lr_N_pre", "abl_W")] - g[("lr_N_pre", "abl_otherW")]),
                         N제거가_W에_미친_효과=boot(g[("lr_W_post", "abl_N")] - g[("lr_W_post", "abl_otherN")]),
                         N제거가_확정층에_미친_효과=boot(g[("d_onset", "abl_N")] - g[("d_onset", "abl_otherN")])))
    md += ["## 2. 핵심 비교 (기준선을 뺀 값)\n",
           "- W제거가_N에_미친_효과 = lr_N_pre[abl_W] − lr_N_pre[abl_otherW]. 뚜렷이 음수면 "
           "**영어 단어(summer)가 있어야 이웃(season)이 뜬다** → 이웃은 영어 단어를 거쳐 나온다.\n"
           "- N제거가_W에_미친_효과 = lr_W_post[abl_N] − lr_W_post[abl_otherN]. 뚜렷이 음수면 "
           "**이웃을 거쳐 단어로 좁혀진다**. 확정층 효과는 양수(늦어짐)여야 같은 방향.\n",
           md_table(pd.DataFrame(rows)), ""]

    e = pd.read_csv(out / "en_input.csv", keep_default_na=False)
    if len(e):
        e["onset_clean"] = pd.to_numeric(e.onset_clean, errors="coerce")
        e = e[e.onset_clean.notna()]
        e["N_pre"] = e.apply(lambda x: meas(x, "N", -3, -1), axis=1)
        e["N_max"] = e[[f"N{l}" for l in range(7, 28)]].astype(float).max(1)
        e["N_peak"] = e[[f"N{l}" for l in range(7, 28)]].astype(float).values.argmax(1) + 7
        e["W_first"] = e.apply(lambda x: next((l for l in range(0, 28) if float(x[f"W{l}"]) >= 0.3), 28), axis=1)
        k = c.reset_index()
        k["N_max"] = k[[f"N{l}" for l in range(7, 28)]].astype(float).max(1)
        k["N_peak"] = k[[f"N{l}" for l in range(7, 28)]].astype(float).values.argmax(1) + 7
        ke = k.merge(e, on=key, suffixes=("_ko", "_en"))
        ue = ke.groupby(["id", "word"])[["N_pre_ko", "N_pre_en", "N_max_ko", "N_max_en", "N_peak_ko", "N_peak_en",
                                         "W_first"]].mean()
        md += ["## 3. 보조: 영어 단어를 직접 넣었을 때 (영→중 프롬프트)\n",
               "같은 W·N 토큰을 한국어 입력과 영어 입력에서 비교. W_first = 영어 입력에서 W 확률 합이 0.3을 처음 넘는 층.\n",
               md_table(pd.DataFrame([dict(입력="한국어", N_pre=boot(ue.N_pre_ko), N_max=boot(ue.N_max_ko), N_정점층=boot(ue.N_peak_ko)),
                             dict(입력="영어", N_pre=boot(ue.N_pre_en), N_max=boot(ue.N_max_en), N_정점층=boot(ue.N_peak_en))])),
               f"\n영어 입력에서 W가 0.3을 넘는 첫 층 평균 {ue.W_first.mean():.1f}. "
               f"r(N_max 한국어, 영어) = {pd.Series(ue.N_max_ko).corr(pd.Series(ue.N_max_en), method='spearman'):.2f}\n"]

    w.columns = ["_".join(map(str, cc)).strip("_") for cc in w.columns]
    w.merge(pd.read_csv(out / "units.csv")[["id", "word", "en", "W_words", "N_words"]], on=["id", "word"]).to_csv(
        out / "unit_effects.csv", index=False)
    (out / "summary.md").write_text("\n".join(md), "utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2-3b")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    out = RESULTS_DIR / args.model / "A_en_run" / "order_ablate" if not args.out_dir else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if not args.analyze_only:
        run(args, out)
    analyze(out)


if __name__ == "__main__":
    main()
