#!/usr/bin/env python3
"""run_items.py 결과로 간섭·렌즈 점수·신뢰도·상관·판정을 계산한다.

  python -m experiment.analyze --model llama3.2-3b
  → results/<model>/summary.md, metrics.json, item_scores.csv, 그림(png)

정의
  간섭 I        = log p(방해어 | 입력어 프롬프트) - 평균_c log p(방해어 | 대조어 c 프롬프트)   (최종 출력, 렌즈 무관)
  렌즈 점수 S   = 중간층(25–75%)에서 probe 영어 단어 렌즈 로그확률의 최댓값 (입력어 프롬프트)
                  A: 영어 다의어(spring), B: 방해어 뜻의 영어(pear), C: 정답 영어(sea)
  S_diff        = 층마다 (입력어 - 대조어 평균) 렌즈 로그확률을 구한 뒤 중간층 최댓값 (보조 지표)
  신뢰도        = 프롬프트 4개를 문항 측정의 반복으로 보고 Cronbach's alpha (기준 0.7)
  상관          = 문항별로 4개 프롬프트 평균을 낸 뒤 Spearman. 95% CI는 묶음(group) 단위 bootstrap
  판정(A 번역)  = 보정 렌즈마다 영어 감소율(1 - 평균 확률 비)과 상관 변화(Δr, 같은 bootstrap 표본)로
                  | 크게 줄고 상관 그대로 → 가설 지지 | 거의 안 줆 → 가설에 불리 | 크게 줄고 상관 하락 → 렌즈 신뢰 |
  충실도 관문   = 과제 프롬프트, 중간층 평균에서 보정 렌즈의 KL(최종 || 렌즈) < 바닐라 KL 이고
                  언어 질량(영+중+한) ≥ FIDELITY_MASS_RATIO × 바닐라 언어 질량. 미달한 렌즈는 판정하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import (BIG_REDUCTION, DATASETS, FIDELITY_MASS_RATIO, MID_LAYER_RANGE, N_BOOT, RELIABILITY_MIN,
                     RESULTS_DIR, SEED)

PRIMARY = {"A": "lp_poly", "B": "lp_distractor", "C": "lp_target"}
KEYS = ["dataset", "id", "task", "version"]


# ---------------------------------------------------------------- 점수 계산
def mid_layers(n_layers: int, layer_range=None) -> tuple[int, int]:
    lo, hi = layer_range or MID_LAYER_RANGE
    return math.ceil(lo * n_layers), math.floor(hi * n_layers)


def interference(out: pd.DataFrame) -> pd.DataFrame:
    o = out[out.logp_distractor.notna()]
    inp = o[o.role == "input"].set_index(KEYS)
    ctl = o[o.role == "control"].groupby(KEYS).logp_distractor.mean()
    df = inp[["group", "etym", "logp_answer", "logp_distractor"]].join(ctl.rename("logp_distractor_ctrl"), how="inner")
    df["I"] = df.logp_distractor - df.logp_distractor_ctrl
    return df.reset_index()


def lens_scores(lens: pd.DataFrame, layer_range=None) -> pd.DataFrame:
    L = int(lens.layer.max())
    lo, hi = mid_layers(L, layer_range)
    rows = []
    for ds, col in PRIMARY.items():
        d = lens[(lens.dataset == ds) & lens.layer.between(lo, hi)]
        if d.empty or col not in d:
            continue
        k = KEYS + ["lens"]
        inp = d[d.role == "input"]
        s = inp.groupby(k)[col].max().rename("S")
        ctl = d[d.role == "control"].groupby(k + ["layer"])[col].mean().rename("ctrl")
        diff = inp.set_index(k + ["layer"])[col].to_frame().join(ctl, how="inner")
        sd = (diff[col] - diff.ctrl).groupby(k).max().rename("S_diff")
        rows.append(pd.concat([s, sd], axis=1).reset_index())
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 통계
def cronbach_alpha(wide: pd.DataFrame) -> float:
    """행 = 문항, 열 = 프롬프트 버전"""
    w = wide.dropna()
    k = w.shape[1]
    if k < 2 or len(w) < 3:
        return float("nan")
    return k / (k - 1) * (1 - w.var(ddof=1).sum() / w.sum(axis=1).var(ddof=1))


def _spearman(x, y):
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _boot_index(groups: np.ndarray, rng, n_boot: int):
    uniq = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in uniq]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[i] for i in pick])


def corr_ci(x, y, groups, n_boot=N_BOOT, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    bs = [_spearman(x[i], y[i]) for i in _boot_index(np.asarray(groups), rng, n_boot)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return _spearman(x, y), lo, hi


def paired_delta(x_ref, x_new, y, p_ref, p_new, groups, n_boot=N_BOOT, seed=SEED):
    """같은 bootstrap 표본으로 Δr = r(new) - r(ref)와 감소율 = 1 - mean(p_new)/mean(p_ref)"""
    x_ref, x_new, y = (np.asarray(a, float) for a in (x_ref, x_new, y))
    p_ref, p_new = np.asarray(p_ref, float), np.asarray(p_new, float)
    rng = np.random.default_rng(seed)
    dr, red = [], []
    for i in _boot_index(np.asarray(groups), rng, n_boot):
        dr.append(_spearman(x_new[i], y[i]) - _spearman(x_ref[i], y[i]))
        red.append(1 - p_new[i].mean() / p_ref[i].mean())
    point_dr = _spearman(x_new, y) - _spearman(x_ref, y)
    point_red = 1 - p_new.mean() / p_ref.mean()
    return (point_dr, *np.nanpercentile(dr, [2.5, 97.5])), (point_red, *np.nanpercentile(red, [2.5, 97.5]))


def verdict(red: float, dr_hi: float, base_lo: float = 1.0) -> str:
    """base_lo: 바닐라 렌즈 상관의 95% CI 하한. 0 이하면 비교할 기준 상관이 없으므로 판정하지 않는다."""
    if base_lo <= 0:
        return "판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음"
    if red < BIG_REDUCTION:
        return "거의 안 줄었음 → 그 원인은 영어를 만들지 않았다 (가설에 불리)"
    if dr_hi < 0:
        return "크게 줄고 상관도 떨어짐 → 지운 영어가 진짜 신호였다 (로짓 렌즈를 믿을 만함)"
    return "크게 줄고 상관은 그대로 → 지운 영어는 부풀려진 영어였다 (가설 지지)"


def fidelity(lens: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """과제 프롬프트(입력어)에서 렌즈 충실도, 중간층 평균. 데이터셋×과제×렌즈 한 행.
    kl 열이 없는 이전 실행 결과면 언어 질량만으로 관문을 본다."""
    d = lens[(lens.role == "input") & lens.layer.between(lo, hi)]
    langs = [c for c in ("mass_en", "mass_zh", "mass_ko") if c in d]
    if not langs:
        return pd.DataFrame()
    d = d.assign(lang_mass=sum(np.exp(d[c]) for c in langs))
    has_kl = "kl" in d and d.kl.notna().any()
    agg = dict(lang_mass=("lang_mass", "mean"))
    if has_kl:
        agg["kl"] = ("kl", "mean")
    f = d.groupby(["dataset", "task", "lens"]).agg(**agg).reset_index()
    van = f[f.lens == "vanilla"].set_index(["dataset", "task"])
    rows = []
    for r in f.itertuples(index=False):
        key = (r.dataset, r.task)
        if key not in van.index:
            continue
        v = van.loc[key]
        ratio = r.lang_mass / v.lang_mass
        ok_mass = ratio >= FIDELITY_MASS_RATIO
        ok_kl = (r.kl < v.kl) if has_kl else None
        if r.lens == "vanilla":
            gate = "기준"
        elif has_kl:
            gate = "O" if ok_mass and ok_kl else "X"
        else:
            gate = "O (질량만)" if ok_mass else "X"
        rows.append({"dataset": r.dataset, "task": r.task, "lens": r.lens,
                     "KL(최종||렌즈)": r.kl if has_kl else float("nan"),
                     "바닐라 KL": v.kl if has_kl else float("nan"),
                     "언어 질량": r.lang_mass, "바닐라 대비": ratio, "관문": gate})
    return pd.DataFrame(rows)


def gate_of(fid: pd.DataFrame, ds: str, task: str, ln: str) -> str:
    if fid.empty:
        return "정보 없음"
    g = fid[(fid.dataset == ds) & (fid.task == task) & (fid.lens == ln)]
    return g.관문.iloc[0] if len(g) else "정보 없음"


# ---------------------------------------------------------------- 표 출력
def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = list(df.columns)
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) and not pd.isna(v) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def ci(p, lo, hi):
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


# ---------------------------------------------------------------- 메인 분석
def analyze(out_dir: Path, n_boot=N_BOOT, plots=True, layer_range=None, tag="") -> dict:
    """layer_range: 렌즈 점수 구간(비율). tag가 있으면 결과를 <out_dir>/<tag>/에 쓴다(원래 결과 보존)."""
    src = out_dir
    if tag:
        out_dir = out_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.read_csv(src / "outputs.csv")
    lens = pd.read_csv(src / "lens.csv.gz")
    n_layers = int(lens.layer.max())
    lo, hi = mid_layers(n_layers, layer_range)
    I = interference(out)
    S = lens_scores(lens, layer_range)
    md, metrics = [f"# 결과 요약 ({out_dir.name})\n",
                   f"층 수 {n_layers}, 렌즈 점수 구간 {lo}–{hi}층, bootstrap {n_boot}회\n"], {}

    # ---- 0. 렌즈 충실도 관문 (과제 프롬프트)
    fid = fidelity(lens, lo, hi)
    if len(fid):
        md += ["## 0. 렌즈 충실도 관문 (과제 프롬프트, 중간층 평균, 입력어)\n",
               "보정 렌즈는 위키 held-out KL(부록)이 아니라 실제 과제 프롬프트에서도 최종 출력에 더 가까워야 하고, "
               "판독이 공백·기호로 무너지지 않아야 한다. 관문: KL < 바닐라 KL, 언어 질량(영+중+한) ≥ "
               f"{FIDELITY_MASS_RATIO} × 바닐라. 'X'인 렌즈는 3절 판정에서 '판정 불가'로 처리한다. "
               "KL 열이 비어 있으면 KL을 기록하기 전 실행 결과라 언어 질량만 본 것이다.\n",
               md_table(fid), ""]
        metrics["fidelity"] = fid.to_dict("records")

    # ---- 1. 신뢰도 (프롬프트 4개)
    rel = []
    for (ds, task), g in I.groupby(["dataset", "task"]):
        rel.append(dict(dataset=ds, task=task, measure="간섭 I",
                        alpha=cronbach_alpha(g.pivot(index="id", columns="version", values="I"))))
    for (ds, task, ln), g in S.groupby(["dataset", "task", "lens"]):
        for m in ("S", "S_diff"):
            rel.append(dict(dataset=ds, task=task, measure=f"렌즈 {m} ({ln})",
                            alpha=cronbach_alpha(g.pivot(index="id", columns="version", values=m))))
    rel = pd.DataFrame(rel)
    rel["통과"] = np.where(rel.alpha >= RELIABILITY_MIN, "O", "X")
    md += ["## 1. 신뢰도 (프롬프트 4개, Cronbach's alpha)\n",
           f"기준 {RELIABILITY_MIN}. 미달한 지표의 상관은 '과장'으로 해석하지 않는다.\n", md_table(rel), ""]
    metrics["reliability"] = rel.to_dict("records")

    # ---- 문항 단위 (프롬프트 평균)
    Ii = I.groupby(["dataset", "task", "id", "group", "etym"], as_index=False).I.mean()
    Si = S.groupby(["dataset", "task", "id", "lens"], as_index=False)[["S", "S_diff"]].mean()
    items = Si.merge(Ii, on=["dataset", "task", "id"], how="left")
    items["p"] = np.exp(items.S)
    items.to_csv(out_dir / "item_scores.csv", index=False)

    # ---- 2. 상관
    cor = []
    for (ds, task, ln), g in items.dropna(subset=["I"]).groupby(["dataset", "task", "lens"]):
        for m in ("S", "S_diff"):
            r, a, b = corr_ci(g[m], g.I, g.group, n_boot)
            cor.append(dict(dataset=ds, task=task, lens=ln, x=m, n=len(g), r=r, ci_lo=a, ci_hi=b))
    cor = pd.DataFrame(cor)
    if len(cor):
        md += ["## 2. 렌즈 점수 × 간섭 상관 (Spearman, 문항 단위)\n", md_table(cor), ""]
    metrics["correlation"] = cor.to_dict("records")

    # ---- 3. 판정 (A 번역)
    a = items[(items.dataset == "A") & (items.task == "translate")].dropna(subset=["I"])
    if len(a) and "vanilla" in set(a.lens):
        ref = a[a.lens == "vanilla"].set_index("id")
        rows = []
        for x in ("S", "S_diff"):
            r0, base_lo, base_hi = corr_ci(ref[x], ref.I, ref.group, n_boot)
            for ln in sorted(set(a.lens) - {"vanilla"}):
                new = a[a.lens == ln].set_index("id").loc[ref.index]
                (dr, dlo, dhi), (rd, rlo, rhi) = paired_delta(ref[x], new[x], ref.I, ref.p, new.p, ref.group, n_boot)
                gate = gate_of(fid, "A", "translate", ln)
                v = verdict(rd, dhi, base_lo)
                if gate == "X":
                    v = "판정 불가: 렌즈 충실도 관문 미달 (0절)"
                rows.append({"x": x, "렌즈": ln, "충실도 관문": gate, "바닐라 r [CI]": ci(r0, base_lo, base_hi),
                             "영어 확률(바닐라)": ref.p.mean(), "영어 확률(보정)": new.p.mean(),
                             "감소율 [CI]": ci(rd, rlo, rhi), "Δr [CI]": ci(dr, dlo, dhi),
                             "판정": v})
        vt = pd.DataFrame(rows)
        md += ["## 3. 판정 (A, 번역 과제)\n",
               f"'크게 줄었음' = 감소율 ≥ {BIG_REDUCTION} (S 기준 확률). '상관 하락' = Δr의 95% CI 상한 < 0. "
               "바닐라 상관의 CI 하한이 0 이하이거나, 보정 렌즈가 충실도 관문(0절)을 넘지 못하면 판정하지 않는다.\n",
               md_table(vt), ""]
        metrics["verdict"] = vt.to_dict("records")

    # ---- 4. 음성 대조 (B) vs A. B는 맥락 없음(translate)과 예문(translate_ctx) 두 조건
    rows = []
    for tb in ("translate", "translate_ctx"):
        for ln in sorted(set(items.lens)):
            ra = items[(items.dataset == "A") & (items.task == "translate") & (items.lens == ln)].dropna(subset=["I"])
            rb = items[(items.dataset == "B") & (items.task == tb) & (items.lens == ln)].dropna(subset=["I"])
            if len(ra) < 3 or len(rb) < 3:
                continue
            rng = np.random.default_rng(SEED)
            da = [_spearman(ra.S.values[i], ra.I.values[i]) for i in _boot_index(ra.group.values, rng, n_boot)]
            db = [_spearman(rb.S.values[i], rb.I.values[i]) for i in _boot_index(rb.group.values, rng, n_boot)]
            diff = np.array(da) - np.array(db)
            rows.append({"B 과제": tb, "렌즈": ln, "n_B": len(rb), "B 평균 간섭": rb.I.mean(),
                         "B 평균 영어 확률": rb.p.mean(), "r_A": _spearman(ra.S, ra.I), "r_B": _spearman(rb.S, rb.I),
                         "r_A - r_B [CI]": ci(_spearman(ra.S, ra.I) - _spearman(rb.S, rb.I),
                                              *np.nanpercentile(diff, [2.5, 97.5]))})
    if rows:
        md += ["## 4. 음성 대조: B(한국어 동음이의어)\n",
               "B의 간섭은 영어와 무관하므로 r_B ≈ 0이어야 한다. r_B도 크면 렌즈 점수가 영어가 아닌 "
               "'헷갈리는 정도'를 재고 있을 가능성이 있다. translate_ctx는 예문으로 뜻을 정해 준 조건이다"
               "(시장에서 배/복숭아 한 상자를 샀다). 이때 간섭은 '뜻이 정해졌는데도 같은 표기 때문에 남는 "
               "다른 뜻'이므로 A(한국어로는 뜻이 하나, 영어 다의어로 새는 방해어)와 구조가 맞는다.\n",
               md_table(pd.DataFrame(rows)), ""]
        metrics["negative_control"] = rows

    # ---- 4b. B: 뜻 빈도 비율(NIKL)과 간섭
    rows = []
    b_path = Path(DATASETS["B"])
    if b_path.exists() and "nikl_dom_target" in pd.read_csv(b_path, nrows=0).columns:
        dom = pd.read_csv(b_path, keep_default_na=False)[["id", "nikl_dom_target", "nikl_dom_ok"]]
        dom = dom.rename(columns={"id": "_bid"})
        dom = dom[dom.nikl_dom_ok.astype(str) == "True"]
        dom["dom_distractor"] = 1 - dom.nikl_dom_target.astype(float)
        for tb in ("translate", "translate_ctx"):
            for ln in sorted(set(items.lens)):
                rb = items[(items.dataset == "B") & (items.task == tb) & (items.lens == ln)]
                rb = rb.dropna(subset=["I"]).merge(dom, left_on="id", right_on="_bid")
                if len(rb) < 6:
                    continue
                r1, l1, h1 = corr_ci(rb.dom_distractor, rb.I, rb.group, n_boot)
                r2, l2, h2 = corr_ci(rb.S, rb.I, rb.group, n_boot)
                rows.append({"B 과제": tb, "렌즈": ln, "n": len(rb), "r(방해 뜻 우세도, I) [CI]": ci(r1, l1, h1),
                             "r(S, I) [CI] (같은 문항)": ci(r2, l2, h2)})
    if rows:
        md += ["### 4b. B 뜻 빈도 비율 (국립국어원 어휘 의미 분석 말뭉치)\n",
               "방해 뜻이 흔할수록 간섭이 커야 정상이다(eDom의 한국어 대응). 두 뜻 합이 적은 단어(dom_ok=False)는 뺐다.\n",
               md_table(pd.DataFrame(rows)), ""]
        metrics["negative_control_dom"] = rows

    # ---- 5. 어종별
    rows = []
    for (ds, task, ln, et), g in items.dropna(subset=["I"]).groupby(["dataset", "task", "lens", "etym"]):
        if task not in ("translate", "translate_ctx") or len(g) < 8:
            continue
        r, lo_, hi_ = corr_ci(g.S, g.I, g.group, n_boot)
        rows.append(dict(dataset=ds, task=task, lens=ln, etym=et, n=len(g), r=r, ci_lo=lo_, ci_hi=hi_,
                         mean_I=g.I.mean(), mean_p=g.p.mean()))
    if rows:
        md += ["## 5. 어종별 (8문항 이상인 집단만)\n",
               "한자어는 한자음을 거쳐 중국어로 바로 가는 지름길이 있을 수 있다.\n", md_table(pd.DataFrame(rows)), ""]
        metrics["by_etym"] = rows

    # ---- 6. 양성 대조 (A: 영어 중간 단계 프롬프트)
    pa = items[(items.dataset == "A")].dropna(subset=["I"])
    if {"translate", "translate_en"} <= set(pa.task):
        rows = []
        for (task, ln), g in pa.groupby(["task", "lens"]):
            r, lo_, hi_ = corr_ci(g.S, g.I, g.group, n_boot)
            rows.append({"과제": task, "렌즈": ln, "평균 간섭": g.I.mean(), "평균 영어 확률": g.p.mean(),
                         "r [CI]": ci(r, lo_, hi_)})
        md += ["## 6. 양성 대조: 프롬프트에 영어 중간 단계\n",
               "영어 경유를 강제했으므로 translate_en에서 간섭·영어 확률이 커지고 상관이 보여야 한다. "
               "그렇지 않으면 간섭 지표가 영어 경유를 잡지 못하는 것이다.\n", md_table(pd.DataFrame(rows)), ""]
        metrics["positive_control"] = rows

    # ---- 7. C: 무관한 영어가 오르는지
    c = lens[(lens.dataset == "C") & (lens.role == "input")]
    if len(c):
        mid = c[c.layer.between(lo, hi)]
        agg = {k: (k, lambda s: np.exp(s).mean()) for k in ("lp_target", "lp_unrelated", "mass_en", "mass_en_word", "mass_zh", "mass_ko")
               if k in mid}
        tab = mid.groupby(["task", "lens"]).agg(**agg).reset_index()
        md += ["## 7. C: 중간층 평균 확률 (정답 영어 vs 무관 영어, 언어별 확률 질량)\n",
               "가설 1이 맞다면 바닐라에서는 무관한 영어(chair)와 영어 질량 전체가 높고, 평균 제거 렌즈에서 줄어야 한다.\n",
               md_table(tab, "{:.4f}"), ""]
        metrics["C"] = tab.to_dict("records")

    # ---- 8. 층 구간별 언어 질량 (영어 편향)
    mcols = [c for c in ("mass_en", "mass_en_word", "mass_zh", "mass_ko") if c in lens]
    if mcols:
        q1, q3 = math.ceil(0.25 * n_layers), math.floor(0.75 * n_layers)
        bands = {f"초반 1–{q1 - 1}": (1, q1 - 1), f"중간 {q1}–{q3}": (q1, q3), f"후반 {q3 + 1}–{n_layers - 1}": (q3 + 1, n_layers - 1)}
        inp = lens[lens.role == "input"]
        rows = []
        for band, (b0, b1) in bands.items():
            g = inp[inp.layer.between(b0, b1)]
            for (ds, task, ln), h in g.groupby(["dataset", "task", "lens"]):
                rows.append({"dataset": ds, "task": task, "lens": ln, "band": band,
                             **{c.replace("mass_", ""): np.exp(h[c]).mean() for c in mcols}})
        bt = pd.DataFrame(rows).sort_values(["dataset", "task", "lens"], kind="stable")
        md += ["## 8. 층 구간별 언어 질량 (입력어 프롬프트 평균 확률)\n",
               "`en`은 라틴 문자 2자 이상 토큰 전부(단어 조각 포함), `en_word`는 단어 시작 토큰이면서 영어 단어 목록에 "
               "있는 토큰만 센다. 둘의 차이가 크면 '영어'로 보이던 것의 상당 부분이 단어 조각이다. "
               "구간은 렌즈 점수 구간 설정과 무관하게 전체 층의 1/4, 3/4로 나눈다.\n", md_table(bt, "{:.3f}"), ""]
        metrics["language_mass"] = bt.to_dict("records")

    # ---- 렌즈 적합도
    fit_file = src / "lens_fit.csv"
    if fit_file.exists():
        f = pd.read_csv(fit_file)
        f = f[f.lens.isin(["vanilla", "mean", "affine", "tuned", "vanilla_tunedeval"]) & f.layer.between(lo, hi)]
        md += ["## 부록: 렌즈 적합도 (위키 held-out KL(최종 || 렌즈), 중간층 평균)\n",
               "아핀·tuned 렌즈가 바닐라보다 KL이 작아야 기저 맞춤이 실제로 된 것이다. "
               "`vanilla_tunedeval`은 --only-tuned로 tuned를 따로 학습했을 때 tuned와 같은 held-out 위치에서 잰 바닐라 KL이다. "
               "과제 프롬프트에서의 충실도는 0절에 있다.\n",
               md_table(f.groupby("lens", as_index=False).kl.mean()), ""]

    (out_dir / "summary.md").write_text("\n".join(md), "utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=float), "utf-8")
    if plots:
        make_plots(out_dir, items, lens, lo, hi)
    print(f"[done] → {out_dir / 'summary.md'}")
    return metrics


def make_plots(out_dir: Path, items: pd.DataFrame, lens: pd.DataFrame, lo: int, hi: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 산점도: 데이터셋×과제마다 렌즈별 패널
    for (ds, task), g in items.dropna(subset=["I"]).groupby(["dataset", "task"]):
        order = ["vanilla", "mean", "affine", "tuned"]
        lens_names = [n for n in order if n in set(g.lens)] + sorted(set(g.lens) - set(order))
        fig, axes = plt.subplots(1, len(lens_names), figsize=(4.2 * len(lens_names), 4), squeeze=False)
        for ax, ln in zip(axes[0], lens_names):
            h = g[g.lens == ln]
            ax.scatter(h.S, h.I, s=14, alpha=0.7)
            ax.set_title(f"{ln}  r={_spearman(h.S, h.I):.2f}")
            ax.set_xlabel("lens score (max mid-layer log p)")
            ax.axhline(0, lw=0.5, c="gray")
        axes[0][0].set_ylabel("interference I")
        fig.suptitle(f"{ds} / {task}")
        fig.tight_layout()
        fig.savefig(out_dir / f"scatter_{ds}_{task}.png", dpi=120)
        plt.close(fig)

    # 층별 곡선: 입력어 프롬프트에서 probe 확률과 영어 질량
    inp = lens[lens.role == "input"]
    cols = [c for c in inp.columns if c.startswith("lp_") or c in ("mass_en", "mass_en_word")]
    for (ds, task), g in inp.groupby(["dataset", "task"]):
        use = [c for c in cols if g[c].notna().any()]
        fig, axes = plt.subplots(1, len(use), figsize=(4.2 * len(use), 3.6), squeeze=False)
        for ax, c in zip(axes[0], use):
            for ln, h in g.groupby("lens"):
                m = h.groupby("layer")[c].apply(lambda s: np.exp(s).mean())
                ax.plot(m.index, m.values, label=ln)
            ax.axvspan(lo, hi, color="gray", alpha=0.1)
            ax.set_title(c)
            ax.set_xlabel("layer")
        axes[0][0].set_ylabel("mean probability")
        axes[0][-1].legend()
        fig.suptitle(f"{ds} / {task}")
        fig.tight_layout()
        fig.savefig(out_dir / f"layers_{ds}_{task}.png", dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--layer-range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="렌즈 점수 구간(전체 층 대비 비율). 기본: config.MID_LAYER_RANGE")
    ap.add_argument("--tag", default="", help="결과를 <out-dir>/<tag>/ 아래에 저장")
    args = ap.parse_args()
    analyze(Path(args.out_dir or RESULTS_DIR / args.model.replace("/", "_")), args.n_boot,
            layer_range=args.layer_range, tag=args.tag)


if __name__ == "__main__":
    main()
