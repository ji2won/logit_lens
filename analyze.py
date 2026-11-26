"""캐시 → lens 세 개(+변형)로 판독 → 지표. GPU 있으면 빠르고, CPU도 된다(--layer-stride로 줄이기).

    python analyze.py cache/llama.pt cache/exaone.pt --out results/
    (여러 캐시를 주면 모델 간 공통 정답 문항으로도 한 번 더 계산)

lens
  logit        : softmax(W_U · norm(h))
  mean_wiki    : h − μ_wiki(ℓ)         μ = 위키 문장 절반의 전체 위치 평균 (캐시에 wiki가 있을 때)
  mean_task    : h − μ_task(ℓ)         μ = 같은 영어 단어 문항을 뺀 나머지 프롬프트 마지막 토큰 평균
  affine       : A_ℓ h + b_ℓ ≈ h_L     위키 나머지 절반으로 ridge 회귀 (없으면 문항 상태로 5-fold, 약함)
  affine_nobias: A_ℓ h                 순서 민감도 분석용 (affine → μ 순서)
  tuned        : h + h Δ_ℓ + b_ℓ       위키 같은 절반으로, 최종 출력 분포와의 KL을 줄이도록 학습 (tuned lens 방식,
                                       Δ=0에서 시작 = logit lens에서 출발). ridge affine은 hidden state의 MSE를 줄이므로
                                       중간 층 판독이 '기타' 토큰(구두점 등)으로 쏠릴 수 있어 이걸 기준 lens로 같이 본다.

lens 충실도 (report의 표)
  KL       : KL(최종 출력 ‖ lens 판독), 프롬프트 마지막 토큰. 낮을수록 그 층 판독이 실제 출력에 가깝다
  언어 질량 : p(latin)+p(han)+p(hangul). 이게 작으면 E는 아주 작은 확률끼리의 비율이라 믿기 어렵다
  E_w      : 질량 가중 E = Σp(latin) / Σ언어 질량 (프롬프트·층 합)

출력 (모델별 폴더)
  layer_E.csv     층 × lens × 과제의 문자 체계별 확률 질량, 영어 비중 E
  items.csv       문항별 I(w), I_en(w), lens별 s_L(w), 정답 여부, 공변량
  summary.json    r_L, bootstrap CI, F1, F2, 예측 1 분해, sanity check
  report.md       위 내용을 읽기 쉽게
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats

SCRIPTS = ["latin", "han", "hangul", "other"]
DELTA = 0.1  # "r_L 유지"로 볼 최대 하락폭 (--delta)


# ---------- lens 기본 ----------
def apply_norm(h, spec):
    w = spec["weight"].to(h.device)
    if spec["kind"] == "rms":
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + spec["eps"]) * w
    elif spec["kind"] == "ln":
        h = (h - h.mean(-1, keepdim=True)) / torch.sqrt(h.var(-1, unbiased=False, keepdim=True) + spec["eps"]) * w
        if spec["bias"] is not None:
            h = h + spec["bias"].to(h.device)
    else:
        raise ValueError(f"알 수 없는 norm 종류: {spec['cls']}")
    return h


class Unembed:
    def __init__(self, u, device):
        self.W = u["W"].to(device).float()
        self.b = None if u["b"] is None else u["b"].to(device)
        self.spec = u["norm"]

    def logprobs(self, h):
        z = apply_norm(h.float(), self.spec) @ self.W.T
        if self.b is not None:
            z = z + self.b
        return torch.log_softmax(z, -1)


# ---------- affine ----------
def ridge_fit(X, Y, intercept=True, lams=(1e-3, 1e-2, 1e-1, 1, 10)):
    """X,Y [M,d]. CPU float64로 푼다(GPU 메모리 절약). λ는 80/20 검증 MSE로 고르고(λ × 평균 고유값 스케일),
    보고하는 R²는 학습 80%만으로 푼 모델의 검증 20% 성능(누설 없음). 최종 A,b는 전체로 다시 푼다."""
    X, Y = X.double().cpu(), Y.double().cpu()
    M, d = X.shape
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(0))
    tr, va = perm[: int(M * 0.8)], perm[int(M * 0.8):]
    I = torch.eye(d, dtype=torch.float64)

    def prep(Xt, Yt):
        if intercept:
            mx, my = Xt.mean(0), Yt.mean(0)
        else:
            mx = my = torch.zeros(d, dtype=Xt.dtype)
        Xc = Xt - mx
        G = Xc.T @ Xc
        return G, Xc.T @ (Yt - my), mx, my, G.diagonal().mean()

    def solve(pr, lam):
        G, XY, mx, my, sc = pr
        A = torch.linalg.solve(G + lam * sc * I, XY)
        return A, my - mx @ A

    ptr = prep(X[tr], Y[tr])
    best = None
    for lam in lams:
        A, b = solve(ptr, lam)
        mse = ((X[va] @ A + b - Y[va]) ** 2).mean().item()
        if best is None or mse < best[0]:
            best = (mse, lam, A, b)
    _, lam, A, b = best
    r2 = 1 - ((X[va] @ A + b - Y[va]) ** 2).sum().item() / ((Y[va] - Y[va].mean(0)) ** 2).sum().item()
    A, b = solve(prep(X, Y), lam)
    return A.float(), b.float(), dict(lam=lam, val_mse=best[0], val_r2=r2)


def tuned_fit(X, Y, U, device, epochs=30, bs=128, lr=1e-4, patience=3, seed=0):
    """tuned lens 방식 affine: h' = h + hΔ + b, KL(softmax(U·h_L) ‖ softmax(U·h'))를 줄이도록 Adam.
    Δ=0, b=0에서 시작(= logit lens). 행을 64개 묶음으로 나눠 20%를 검증으로 두고(같은 문장 위치가 양쪽에 섞이지 않게)
    검증 KL이 patience번 안 줄면 멈추고 가장 좋았던 값을 쓴다. 반환: Δ, b, info(검증 KL: tuned / logit lens)."""
    torch.manual_seed(seed)
    M, d = X.shape
    blk = np.arange(M) // 64
    rng = np.random.default_rng(seed)
    vb = rng.permutation(blk.max() + 1)[: max(1, int((blk.max() + 1) * 0.2))]
    va = np.isin(blk, vb)
    tr_idx, va_idx = np.where(~va)[0], np.where(va)[0]
    D = torch.zeros(d, d, device=device, requires_grad=True)
    b = torch.zeros(d, device=device, requires_grad=True)
    opt = torch.optim.Adam([D, b], lr=lr)

    def kl_batch(idx):
        x = X[idx].to(device).float()
        with torch.no_grad():
            lpf = U.logprobs(Y[idx].to(device).float())
        lp = U.logprobs(x + x @ D + b)
        return (lpf.exp() * (lpf - lp)).sum(-1)

    def val_kl(use_lens=True):
        tot = 0.0
        with torch.no_grad():
            for s in range(0, len(va_idx), bs):
                idx = va_idx[s:s + bs]
                x = X[idx].to(device).float()
                lpf = U.logprobs(Y[idx].to(device).float())
                h = x + x @ D + b if use_lens else x
                tot += (lpf.exp() * (lpf - U.logprobs(h))).sum().item()
        return tot / len(va_idx)

    kl_logit = val_kl(False)
    best = (val_kl(True), D.detach().clone(), b.detach().clone(), 0)
    bad = 0
    for ep in range(epochs):
        perm = rng.permutation(tr_idx)
        for s in range(0, len(perm), bs):
            loss = kl_batch(perm[s:s + bs]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = val_kl(True)
        if v < best[0] - 1e-4:
            best, bad = (v, D.detach().clone(), b.detach().clone(), ep + 1), 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best[1].cpu(), best[2].cpu(), dict(val_kl_tuned=best[0], val_kl_logit=kl_logit, epochs=best[3])


# ---------- 통계 ----------
def partial_spearman(x, y, Z):
    """순위 변환 후 공변량 순위로 OLS 잔차를 구해 Pearson. Z: [n,k] 또는 None"""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if Z is not None and Z.shape[1]:
        Zr = np.column_stack([np.ones(len(x))] + [stats.rankdata(z) for z in Z.T])
        rx = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
        ry = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_groups(df, group, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = df[group].values
    ug = np.unique(g)
    idx = {k: np.where(g == k)[0] for k in ug}
    out = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug), replace=True)
        out.append(fn(df.iloc[np.concatenate([idx[k] for k in pick])]))
    return np.array(out)


def ci(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)] if a.ndim == 1 else a
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ---------- 메인 계산 ----------
def run_lenses(C, device, layers, mids, affine_src, affine_langs=None, mu_langs=None, tuned=True, extra=None):
    """extra: {이름: (IDX [N,k] int, MASK [N,k] bool)} — 프롬프트마다 추가로 읽을 토큰 묶음. 기록에 '{이름}_lp'로 남는다."""
    meta = C["meta"]
    H = C["hs_last"]  # [N,L+1,d] fp16
    N, L1, d = H.shape
    L = L1 - 1
    U = Unembed(C["unembed"], device)
    script = np.array(C["vocab_script"])
    Smask = torch.tensor(np.stack([script == s for s in SCRIPTS], 1), dtype=torch.float32, device=device)  # [V,4]
    en_ids = C["en_ids"]
    ans1 = meta.ans_ids.map(lambda s: json.loads(s)[0]).values
    dis1 = meta.dis_ids.map(lambda s: json.loads(s)[0]).values
    global EN_IDX, EN_MASK
    k = max(len(v) for v in en_ids.values())
    EN_IDX = np.array([(en_ids[e] + [en_ids[e][0]] * k)[:k] for e in meta.en.values])
    EN_MASK = np.array([[j < len(en_ids[e]) for j in range(k)] for e in meta.en.values])

    # μ_task: 같은 영어 단어(en) 프롬프트를 뺀 평균 (translate 과제 마지막 토큰)
    tr = torch.from_numpy((meta.task == "translate").values.copy())
    Hf = H.float()
    tot = Hf[tr].sum(0)
    ntot = int(tr.sum())
    g_sum, g_n = {}, {}
    for en in meta.en.unique():
        m = torch.from_numpy((meta.en.values == en).copy()) & tr
        g_sum[en], g_n[en] = Hf[m].sum(0), int(m.sum())

    def mu_task_for(rows):
        return torch.stack([(tot - g_sum[e]) / (ntot - g_n[e]) for e in meta.en.values[rows]])  # [n,L+1,d]

    lens_names = ["logit", "mean_task"]
    has_wiki = "wiki_mu" in C
    wsel = None
    if has_wiki:
        XL = np.array(C.get("wiki_X_lang", ["?"] * len(C["wiki_X"])))
        wsel = np.ones(len(XL), bool) if not affine_langs else np.isin(XL, affine_langs)
        assert wsel.sum() > 0, f"affine 학습 언어 {affine_langs}의 문장이 캐시에 없음 (있는 언어: {sorted(set(XL))})"
        wsel = torch.from_numpy(wsel)
        mu_w = C["wiki_mu"]
        if mu_langs and "wiki_mu_by_lang" in C:
            parts = [C["wiki_mu_by_lang"][k] for k in mu_langs if k in C["wiki_mu_by_lang"]]
            assert parts, f"μ 언어 {mu_langs}가 캐시에 없음"
            mu_w = sum(p[0] for p in parts) / sum(p[1] for p in parts)
        print("affine 학습 표본:", int(wsel.sum()), "언어:", affine_langs or "전부", "| μ 언어:", mu_langs or "전부", flush=True)
    if has_wiki:
        lens_names.append("mean_wiki")
    lens_names += ["affine", "affine_nobias"]
    do_tuned = tuned and has_wiki and affine_src == "wiki"
    if do_tuned:
        lens_names.append("tuned")
        torch.backends.cuda.matmul.allow_tf32 = True

    # affine 학습
    aff, aff_info = {}, {}
    for l in layers:
        if has_wiki and affine_src == "wiki":
            X = C["wiki_X"][wsel][:, l].float()
            Y = C["wiki_X"][wsel][:, L].float()
            A, b, info = ridge_fit(X, Y, True)
            A0, b0, info0 = ridge_fit(X, Y, False)
            aff[l] = ("global", A, b, A0, b0)
            if do_tuned:
                if l == L:
                    Dt, bt, tinfo = torch.zeros(d, d), torch.zeros(d), dict(val_kl_tuned=0.0, val_kl_logit=0.0, epochs=0)
                else:
                    Dt, bt, tinfo = tuned_fit(C["wiki_X"][wsel][:, l], C["wiki_X"][wsel][:, L], U, device)
                aff[l] = aff[l] + (Dt, bt)
                info["tuned"] = tinfo
                print(f"tuned ℓ={l}: 검증 KL {tinfo['val_kl_logit']:.3f}(logit) → {tinfo['val_kl_tuned']:.3f}(tuned), "
                      f"{tinfo['epochs']} epoch", flush=True)
        else:  # 문항 상태로 5-fold (en 단위) — 학습 표본이 적어 약함
            ens = meta.en.unique()
            rng = np.random.default_rng(0)
            fold_of = dict(zip(rng.permutation(ens), np.arange(len(ens)) % 5))
            folds = meta.en.map(fold_of).values
            per = {}
            for f in range(5):
                trn = folds != f
                X = Hf[trn][:, l]
                Y = Hf[trn][:, L]
                A, b, info = ridge_fit(X, Y, True, lams=(1, 10, 100, 1000))
                A0, b0, info0 = ridge_fit(X, Y, False, lams=(1, 10, 100, 1000))
                per[f] = (A, b, A0, b0)
            aff[l] = ("fold", per, folds)
        aff_info[l] = dict(intercept=info, nobias=info0, tuned=info.pop("tuned", None))

    rec = []  # (pid, layer, lens) -> stats
    bs = 64
    for l in layers:
        for s in range(0, N, bs):
            rows = np.arange(s, min(N, s + bs))
            h = Hf[rows, l].to(device)
            variants = {"logit": h, "mean_task": h - mu_task_for(rows)[:, l].to(device)}
            if has_wiki:
                variants["mean_wiki"] = h - mu_w[l].to(device)
            if aff[l][0] == "global":
                _, A, b, A0, b0 = aff[l][:5]
                variants["affine"] = h @ A.to(device) + b.to(device)
                variants["affine_nobias"] = h @ A0.to(device) + b0.to(device)
                if do_tuned:
                    Dt, bt = aff[l][5:]
                    variants["tuned"] = h + h @ Dt.to(device) + bt.to(device)
            else:
                _, per, folds = aff[l]
                ha, hn = torch.empty_like(h), torch.empty_like(h)
                for f, (A, b, A0, b0) in per.items():
                    m = torch.tensor(folds[rows] == f, device=device)
                    if m.any():
                        ha[m] = h[m] @ A.to(device) + b.to(device)
                        hn[m] = h[m] @ A0.to(device) + b0.to(device)
                variants["affine"], variants["affine_nobias"] = ha, hn
            ei = torch.tensor(EN_IDX[rows], device=device)
            em = torch.tensor(EN_MASK[rows], device=device)
            ai = torch.tensor(ans1[rows], device=device)[:, None]
            di = torch.tensor(dis1[rows], device=device)[:, None]
            lpf = U.logprobs(Hf[rows, L].to(device))  # 실제 최종 출력 분포
            pf = lpf.exp()
            for name, hv in variants.items():
                lp = U.logprobs(hv)  # [n,V]
                kl = (pf * (lpf - lp)).sum(-1).cpu().numpy()
                mass = (lp.exp() @ Smask).cpu().numpy()  # [n,4]
                top = lp.max(-1)
                en_lp = torch.logsumexp(lp.gather(1, ei).masked_fill(~em, -float("inf")), 1).cpu().numpy()
                a1 = lp.gather(1, ai)[:, 0].cpu().numpy()
                xlp = {}
                for xn, (XI, XM) in (extra or {}).items():
                    xi = torch.tensor(XI[rows], device=device)
                    xm = torch.tensor(XM[rows], device=device)
                    xlp[xn] = torch.logsumexp(lp.gather(1, xi).masked_fill(~xm, -float("inf")), 1).cpu().numpy()
                d1 = lp.gather(1, di)[:, 0].cpu().numpy()
                ti, tv = top.indices.cpu().numpy(), top.values.cpu().numpy()
                for k, pid in enumerate(rows):
                    rec.append(dict(pid=int(pid), layer=l, lens=name,
                                    **{f"p_{sc}": float(mass[k, j]) for j, sc in enumerate(SCRIPTS)},
                                    en_lp=float(en_lp[k]), ans1_lp=float(a1[k]), dis1_lp=float(d1[k]),
                                    top1=int(ti[k]), top1_lp=float(tv[k]), kl=float(kl[k]),
                                    **{f"{xn}_lp": float(v[k]) for xn, v in xlp.items()}))
        print(f"layer {l}/{L} 완료", flush=True)
    R = pd.DataFrame(rec)
    R["E"] = R.p_latin / (R.p_latin + R.p_han + R.p_hangul)
    return R, lens_names, {int(k): v for k, v in aff_info.items()}


def sanity(C, R, device):
    meta, L = C["meta"], C["n_layers"]
    fin = R[(R.layer == L) & (R.lens == "logit")].set_index("pid").sort_index()
    ft = C["final_top"].numpy()
    match = float((fin.top1.values == ft[fin.index, 0].astype(int)).mean())
    dlp = float(np.abs(fin.top1_lp.values - ft[fin.index, 1]).max())
    # μ=0 평균 제거 lens == logit lens
    U = Unembed(C["unembed"], device)
    h = C["hs_last"][:8, L // 2].float().to(device)
    same = float((U.logprobs(h - torch.zeros_like(h)) - U.logprobs(h)).abs().max())
    return dict(final_layer_top1_match=match, final_layer_max_abs_dlogp=dlp, mu0_equals_logit_maxdiff=same,
                # 16비트 저장 반올림 때문에 1·2위가 거의 동점인 프롬프트에서 순서가 바뀔 수 있어 약간의 여유를 둠
                ok=bool(match > 0.97 and dlp < 0.1 and same == 0.0))


def item_table(C, R, lens_names, mids, covars):
    meta = C["meta"]
    tr = meta[meta.task == "translate"]
    key = ["item", "en", "template"]
    piv = tr.pivot_table(index=key, columns="cond", values="lp_dis", aggfunc="first")
    piv_a = tr.pivot_table(index=key, columns="cond", values="lp_ans", aggfunc="first")
    inp = tr[tr.cond == "input"].set_index(key)
    T = pd.DataFrame(index=piv.index)
    T["I"] = piv["input"] - (piv["ctrl1"] + piv["ctrl2"]) / 2
    T["I_en"] = piv["en"] - (piv["ctrl1"] + piv["ctrl2"]) / 2
    T["lp_ans_input"] = piv_a["input"]
    T["correct"] = inp["correct"].astype(bool)
    T["pid"] = inp["pid"]
    Rm = R[R.layer.isin(mids)]
    for ln in lens_names:
        s = Rm[Rm.lens == ln].groupby("pid").en_lp.max()
        T[f"s_{ln}"] = T.pid.map(s)
        T[f"E_{ln}"] = T.pid.map(Rm[Rm.lens == ln].groupby("pid").E.mean())
    T = T.reset_index()
    T["ntok_en_model"] = T.en.map(C["ntok_en"])
    return T


def analyze_items(T, lens_names, covars, min_correct, keep_items=None, nboot=2000, boot_group="en"):
    templates = sorted(T.template.unique())
    ok = T.groupby("item").correct.sum()
    items = ok[ok >= min_correct].index if keep_items is None else pd.Index(keep_items)
    T = T[T.item.isin(items)]
    # 문자열 열(어원, 묶음 단위)은 평균에서 빠지므로 묶음 키로 넣어 남긴다
    T = T.copy()
    gk = ["item", "en"]
    for c in [boot_group, "etym_ko_input"]:
        if c in T and c not in gk:
            T[c] = T[c].fillna("?")
            gk.append(c)
    agg = T.groupby(gk).mean(numeric_only=True).reset_index()
    cov = [c for c in covars if c in agg]
    Z = agg[cov].values if cov else None
    res = dict(n_items=int(len(agg)), n_en=int(agg.en.nunique()), covariates=cov)

    # 예측 2: r_L
    r = {ln: partial_spearman(agg[f"s_{ln}"], agg.I, Z) for ln in lens_names}

    def fn(sub):
        Zs = sub[cov].values if cov else None
        return [partial_spearman(sub[f"s_{ln}"], sub.I, Zs) for ln in lens_names]

    B = boot_groups(agg, boot_group if boot_group in agg else "en", fn, nboot)
    res["r_L"] = {ln: dict(r=r[ln], ci=ci(B[:, i])) for i, ln in enumerate(lens_names)}
    res["r_L_diff_vs_logit"] = {ln: dict(d=r[ln] - r["logit"], ci=ci(B[:, i] - B[:, 0]))
                               for i, ln in enumerate(lens_names) if ln != "logit"}
    res["r_L_raw_spearman"] = {ln: float(stats.spearmanr(agg[f"s_{ln}"], agg.I)[0]) for ln in lens_names}

    # F1 양성 대조: 영어 입력의 간섭이 한국어 입력보다 큰가
    w = stats.wilcoxon(agg.I_en, agg.I)
    res["F1_positive"] = dict(mean_I=float(agg.I.mean()), mean_I_en=float(agg.I_en.mean()),
                              frac_I_en_gt_I=float((agg.I_en > agg.I).mean()), wilcoxon_p=float(w.pvalue),
                              mean_I_gt0_frac=float((agg.I > 0).mean()))
    # F1 어원별
    if "etym_ko_input" in agg:
        res["by_etym"] = {}
        for g, sub in agg.groupby(agg.etym_ko_input.fillna("?").str.replace(r"^AMBIG.*", "AMBIG", regex=True)):
            if len(sub) >= 8:
                Zs = sub[cov].values if cov else None
                res["by_etym"][g] = dict(n=int(len(sub)), mean_I=float(sub.I.mean()),
                                         r_L={ln: partial_spearman(sub[f"s_{ln}"], sub.I, Zs) for ln in lens_names})

    # F2 split-half (템플릿 4개를 2:2로 나누는 3가지 평균)
    if len(templates) >= 4:
        splits = [(templates[:2], templates[2:]), ([templates[0], templates[2]], [templates[1], templates[3]]),
                  ([templates[0], templates[3]], [templates[1], templates[2]])]
        rel = {}
        for var in ["I"] + [f"s_{ln}" for ln in lens_names]:
            rs = []
            for a, b in splits:
                xa = T[T.template.isin(a)].groupby("item")[var].mean()
                xb = T[T.template.isin(b)].groupby("item")[var].mean()
                rr = stats.spearmanr(xa, xb.loc[xa.index])[0]
                rs.append(2 * rr / (1 + rr))
            rel[var] = float(np.mean(rs))
        res["F2_reliability_spearman_brown"] = rel
        res["F2_disattenuated_r_L"] = {ln: (r[ln] / math.sqrt(rel["I"] * rel[f"s_{ln}"])
                                            if rel["I"] > 0 and rel[f"s_{ln}"] > 0 else float("nan"))
                                       for ln in lens_names}
        res["F2_interpretable"] = {ln: bool(rel["I"] >= 0.7 and rel[f"s_{ln}"] >= 0.7) for ln in lens_names}
        per_t = {}
        for t in templates:
            st = T[T.template == t].merge(agg[["item"] + cov], on="item", suffixes=("", "_a")) if cov else T[T.template == t]
            Zt = st[cov].values if cov else None
            per_t[t] = {ln: partial_spearman(st[f"s_{ln}"], st.I, Zt) for ln in lens_names}
        res["F2_r_L_per_template"] = per_t
        res["F2_lens_rank_per_template"] = {t: sorted(v, key=lambda k: -v[k]) for t, v in per_t.items()}

    # 불일치 문항 (lens 높고 간섭 낮음 / 반대), logit lens 기준
    rk = pd.DataFrame({"item": agg.item, "en": agg.en, "rank_s": stats.rankdata(agg.s_logit) / len(agg),
                       "rank_I": stats.rankdata(agg.I) / len(agg)})
    rk["gap"] = rk.rank_s - rk.rank_I
    res["mismatch_high_lens_low_I"] = rk.nlargest(8, "gap")[["item", "en"]].values.tolist()
    res["mismatch_low_lens_high_I"] = rk.nsmallest(8, "gap")[["item", "en"]].values.tolist()

    # μ 몫과 빈도의 겹침 (6절 주의)
    for mu in [m for m in lens_names if m.startswith("mean")]:
        if "zipf_en" in agg:
            res[f"mu_share_vs_zipf_en_{mu}"] = float(stats.spearmanr(agg.s_logit - agg[f"s_{mu}"], agg.zipf_en)[0])
    return res, agg


def decompose_E(R, meta, mids, items, lens_names):
    out = {}
    for task in ["translate", "repeat"]:
        pids = meta[(meta.task == task) & (meta.cond == "input") & meta.item.isin(items)].pid
        sub = R[R.pid.isin(pids) & R.layer.isin(mids)]
        E = sub.groupby("lens").E.mean().to_dict()
        d = dict(E=E)
        g = sub.assign(p_lang=sub.p_latin + sub.p_han + sub.p_hangul).groupby("lens")
        # 단어 단위 지표 (번역 과제만, 대조 조건이 있을 때):
        #   gap  = lp(영어 단어 첫 토큰) − lp(정답 첫 토큰)          중간 층에서 영어 단어가 정답보다 얼마나 앞서는지
        #   spec = lp(영어 단어 | 입력어) − lp(영어 단어 | 대조어 평균) 같은 토큰, 관련 입력 vs 무관 입력 (사전 확률 상쇄)
        if task == "translate" and {"ctrl1", "ctrl2"} <= set(meta.cond):
            mm = meta[(meta.task == task) & meta.item.isin(items)][["pid", "item", "template", "cond"]]
            W = R[R.pid.isin(mm.pid) & R.layer.isin(mids)].merge(mm, on="pid")
            W = W.groupby(["lens", "item", "template", "cond"])[["en_lp", "ans1_lp"]].mean().reset_index()
            piv = W.pivot_table(index=["lens", "item", "template"], columns="cond", values="en_lp")
            piv_a = W.pivot_table(index=["lens", "item", "template"], columns="cond", values="ans1_lp")
            it = pd.DataFrame({"en_lp": piv["input"], "ans1_lp": piv_a["input"],
                               "gap": piv["input"] - piv_a["input"],
                               "spec": piv["input"] - (piv["ctrl1"] + piv["ctrl2"]) / 2}).groupby(["lens", "item"]).mean()
            rng = np.random.default_rng(0)
            wl = {}
            for ln, x in it.groupby(level=0):
                row = {}
                for k in ["en_lp", "ans1_lp", "gap", "spec"]:
                    v = x[k].values
                    bs = rng.choice(v, (1000, len(v))).mean(1)
                    row[k] = (float(v.mean()), ci(bs))
                    if ln != "logit":
                        dv = v - it.loc["logit"][k].reindex(x.index.get_level_values(1)).values
                        row["d_" + k] = (float(dv.mean()), ci(rng.choice(dv, (1000, len(dv))).mean(1)))
                wl[ln] = row
            d["word_level"] = wl
        d["fidelity"] = {ln: dict(E=float(x.E.mean()), E_w=float(x.p_latin.sum() / x.p_lang.sum()),
                                  p_lang=float(x.p_lang.mean()), kl=float(x.kl.mean()), kl_median=float(x.kl.median()))
                         for ln, x in g}
        for mu in [m for m in lens_names if m.startswith("mean")]:
            d[f"order_mu_then_affine[{mu}]"] = dict(mu_share=E["logit"] - E[mu], affine_share=E[mu] - E["affine"],
                                                    remainder=E["affine"])
        d["order_affine_then_mu"] = dict(affine_share=E["logit"] - E["affine_nobias"],
                                         mu_share=E["affine_nobias"] - E["affine"], remainder=E["affine"])
        out[task] = d
    return out


def write_report(path, model, res_all, sanity_d, aff_info, mids, L, top_tokens=None):
    def f(x):
        return "nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"
    lines = [f"# {model}", "", f"중간 층 ℳ = {mids[0]}–{mids[-1]} (전체 {L}층의 25–75%)", ""]
    lines += ["## sanity check", "", "```", json.dumps(sanity_d, ensure_ascii=False, indent=1), "```", ""]
    al = [f"ℓ={l}: R²={v['intercept']['val_r2']:.3f} (λ={v['intercept']['lam']})" for l, v in aff_info.items() if l in mids]
    lines += ["affine 검증 R² (중간 층): " + ", ".join(al), ""]
    tl = [f"ℓ={l}: {v['tuned']['val_kl_logit']:.2f}→{v['tuned']['val_kl_tuned']:.2f}"
          for l, v in aff_info.items() if l in mids and v.get("tuned")]
    if tl:
        lines += ["tuned 검증 KL (위키, logit lens → tuned, 중간 층): " + ", ".join(tl), ""]
    if top_tokens:
        lines += ["### 중간 층 판독 1위 토큰 (번역, 입력 조건, 자주 나온 순)", ""]
        for ln, toks in top_tokens.items():
            lines.append(f"- {ln}: " + ", ".join(f"{t!r}×{n}" for t, n in toks))
        lines.append("")
    for name, (res, dec) in res_all.items():
        lines += [f"## 문항 집합: {name} (문항 {res['n_items']}, 영어 단어 {res['n_en']})", ""]
        lines += ["### 예측 2: lens 점수 s_L과 간섭 I의 부분 Spearman (공변량: " + ", ".join(res["covariates"]) + ")", "",
                  "| lens | r_L | 95% CI | logit 대비 Δ | Δ 95% CI | 신뢰도 보정 r |", "|---|---|---|---|---|---|"]
        for ln, v in res["r_L"].items():
            dd = res["r_L_diff_vs_logit"].get(ln)
            dis = res.get("F2_disattenuated_r_L", {}).get(ln)
            lines.append(f"| {ln} | {f(v['r'])} | [{f(v['ci'][0])}, {f(v['ci'][1])}] | "
                         f"{f(dd['d']) if dd else '-'} | {('[' + f(dd['ci'][0]) + ', ' + f(dd['ci'][1]) + ']') if dd else '-'} | {f(dis)} |")
        lines += ["", f"판정 참고(6절): Δ = r(보정 lens) − r(logit). Δ CI 상한 < 0 → 'r_L 하락'(기각 방향). "
                  f"Δ CI 하한 > −{DELTA} → 'r_L 유지'(지지 방향). 그 사이는 판단 불가. F1·F2를 통과했을 때만 해석.", ""]
        for ln, dd in res["r_L_diff_vs_logit"].items():
            if dd["ci"][1] < 0:
                verdict = "r_L 하락 (기각 방향)"
            elif dd["ci"][0] > -DELTA:
                verdict = f"r_L 유지 (지지 방향, Δ CI 하한 > −{DELTA})"
            else:
                verdict = f"판단 불가 (Δ CI가 0과 −{DELTA}를 모두 포함, 표본 부족)"
            lines.append(f"- {ln}: {verdict}")
        p = res["F1_positive"]
        lines += ["", "### F1 간섭 점수 타당성", "",
                  f"- 양성 대조: 평균 I(한국어 입력) = {f(p['mean_I'])}, 평균 I_en(영어 입력) = {f(p['mean_I_en'])}, "
                  f"I_en > I 비율 = {f(p['frac_I_en_gt_I'])}, Wilcoxon p = {p['wilcoxon_p']:.2g}. "
                  f"I > 0 비율 = {f(p['mean_I_gt0_frac'])}",
                  "- 음성 대조(한국어 다의어 '배' 등)는 데이터셋 A에 없음 → D3 필요", ""]
        if res.get("by_etym"):
            lines += ["어원별 (문항 8개 이상인 집단):", "", "| 어원 | n | 평균 I | " + " | ".join(res["r_L"]) + " |",
                      "|---|---|---|" + "---|" * len(res["r_L"])]
            for g, v in res["by_etym"].items():
                lines.append(f"| {g} | {v['n']} | {f(v['mean_I'])} | " + " | ".join(f(v['r_L'][ln]) for ln in res["r_L"]) + " |")
            lines.append("")
        if "F2_reliability_spearman_brown" in res:
            lines += ["### F2 신뢰도 (split-half, Spearman–Brown)", "", "| 변수 | 신뢰도 | 0.7 이상 |", "|---|---|---|"]
            for k, v in res["F2_reliability_spearman_brown"].items():
                lines.append(f"| {k} | {f(v)} | {'Y' if v >= 0.7 else 'N'} |")
            lines += ["", "템플릿별 lens 순위: " + "; ".join(f"{t}: {' > '.join(v)}" for t, v in res["F2_lens_rank_per_template"].items()), ""]
        lines += ["불일치 (logit lens 높음·간섭 낮음): " + ", ".join(f"{a}({b})" for a, b in res["mismatch_high_lens_low_I"]),
                  "", "불일치 (logit lens 낮음·간섭 높음): " + ", ".join(f"{a}({b})" for a, b in res["mismatch_low_lens_high_I"]), ""]
        for k, v in res.items():
            if k.startswith("mu_share_vs_zipf"):
                lines.append(f"- {k}: Spearman = {f(v)} (높으면 μ 몫이 빈도와 겹쳐 μ 판정이 약해짐)")
        lines += ["", "### 예측 1: 중간 층 영어 비중 E 분해 (ℳ 평균, 입력 조건)", ""]
        for task, d in dec.items():
            lines.append(f"**{task}** E: " + ", ".join(f"{k}={f(v)}" for k, v in d["E"].items()))
            if "fidelity" in d:
                lines += ["", "| lens | E | E_w (질량 가중) | 언어 질량 | KL(최종‖lens) 평균 | KL 중앙값 |", "|---|---|---|---|---|---|"]
                for ln, v in d["fidelity"].items():
                    lines.append(f"| {ln} | {f(v['E'])} | {f(v['E_w'])} | {f(v['p_lang'])} | {f(v['kl'])} | {f(v['kl_median'])} |")
                lines.append("")
            if "word_level" in d:
                lines += ["단어 단위 (번역, ℳ 평균 log 확률). gap = 영어 단어 − 정답 첫 토큰, spec = 영어 단어(입력어) − 영어 단어(대조어). "
                          "Δ는 logit lens 대비, [95% CI]는 문항 bootstrap", "",
                          "| lens | 영어 단어 | 정답 | gap | spec | Δ 영어 단어 | Δ spec |", "|---|---|---|---|---|---|---|"]
                for ln, v in d["word_level"].items():
                    c = lambda t: f"{t[0]:.2f} [{t[1][0]:.2f}, {t[1][1]:.2f}]"
                    lines.append(f"| {ln} | {v['en_lp'][0]:.2f} | {v['ans1_lp'][0]:.2f} | {c(v['gap'])} | {c(v['spec'])} | "
                                 f"{c(v['d_en_lp']) if 'd_en_lp' in v else '-'} | {c(v['d_spec']) if 'd_spec' in v else '-'} |")
                lines.append("")
            for k, v in d.items():
                if k not in ("E", "fidelity", "word_level"):
                    lines.append(f"- {k}: " + ", ".join(f"{a}={f(b)}" for a, b in v.items()))
            lines.append("")
    open(path, "w", encoding="utf-8").write("\n".join(lines))


def top_tokens_mid(C, R, mids, k=12):
    """ℳ 층에서 lens별 판독 1위 토큰 빈도 (번역·입력 조건). 토크나이저를 오프라인 캐시에서 못 읽으면 id로 표시."""
    try:
        from transformers import AutoTokenizer
        kw = dict(trust_remote_code=True, local_files_only=True)
        if C.get("revision"):
            kw["revision"] = C["revision"]
        tok = AutoTokenizer.from_pretrained(C["model"], **kw)
        dec = lambda i: tok.decode([i])
    except Exception as e:  # noqa: BLE001
        print("토크나이저를 못 읽어 토큰 id로 표시:", type(e).__name__, flush=True)
        dec = lambda i: f"#{i}"
    sub = R[(R.task == "translate") & (R.cond == "input") & R.layer.isin(mids)]
    return {ln: [(dec(int(t)), int(n)) for t, n in x.top1.value_counts().head(k).items()] for ln, x in sub.groupby("lens")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--data", default="../dataset_A/A_final_v2.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-correct", type=int, default=3, help="템플릿 4개 중 몇 개 이상 맞힌 문항만 쓸지")
    ap.add_argument("--covars", default="zipf_en,zipf_ko_input,zipf_zh_distractor")
    ap.add_argument("--affine-src", default="wiki", choices=["wiki", "items"])
    ap.add_argument("--affine-langs", default="", help="affine 학습에 쓸 위키 언어 (예: ko,zh,en). 비우면 캐시에 있는 전부")
    ap.add_argument("--mu-langs", default="", help="μ_wiki에 쓸 위키 언어. 비우면 전부")
    ap.add_argument("--layer-stride", type=int, default=1, help="CPU에서 느리면 2~4. ℳ 층은 항상 포함")
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--delta", type=float, default=0.1, help="'r_L 유지'로 볼 최대 하락폭")
    ap.add_argument("--no-tuned", action="store_true", help="KL로 학습하는 tuned lens를 건너뜀 (GPU 시간 절약)")
    ap.add_argument("--wiki-from", default="", help="위키 부분(μ, affine·tuned 학습 표본)을 가져올 캐시. caches와 같은 순서로 쉼표 구분 "
                                                   "(예: 데이터셋 B 캐시에 A 캐시의 위키를 씀)")
    ap.add_argument("--boot-group", default="en", help="bootstrap 묶음 단위. B는 ko_input (같은 한국어 단어의 두 방향을 함께 뽑음)")
    a = ap.parse_args()
    global DELTA
    DELTA = a.delta
    os.makedirs(a.out, exist_ok=True)
    data = pd.read_csv(a.data)
    keep_cols = [c for c in ["id", "ko_input", "etym_ko_input", "zipf_en", "zipf_ko_input", "zipf_zh_distractor", "zipf_zh_answer",
                             "draft_flags", "direction"] if c in data]
    covars = [c for c in a.covars.split(",") if c]

    per_model = {}
    for path in a.caches:
        C = torch.load(path, weights_only=False)
        if a.wiki_from:
            src = a.wiki_from.split(",")[a.caches.index(path)]
            W = torch.load(src, weights_only=False, map_location="cpu")
            assert W["model"] == C["model"], f"--wiki-from 모델이 다름: {W['model']} vs {C['model']}"
            for k in ["wiki_mu", "wiki_mu_by_lang", "wiki_X", "wiki_X_lang", "wiki_n_mu_tokens"]:
                if k in W:
                    C[k] = W[k]
            del W
            print("위키 부분을 가져옴:", src, flush=True)
        # 정답 여부는 저장된 출력(gen)으로 최신 채점 기준(run_cache.is_correct)에 따라 다시 매긴다
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from run_cache import is_correct
        mt = C["meta"]
        has = mt.gen.notna() if "gen" in mt else pd.Series(False, index=mt.index)
        old_acc = mt.loc[has, "correct"].astype(bool).mean()
        mt.loc[has, "correct"] = [is_correct(g, a_) for g, a_ in zip(mt.gen[has], mt.answer[has])]
        print(os.path.basename(path), "정답률 재채점:", round(float(old_acc), 3), "→",
              mt[has].groupby("task").correct.mean().round(3).to_dict(), flush=True)
        L = C["n_layers"]
        mids = list(range(math.ceil(0.25 * L), math.floor(0.75 * L) + 1))
        layers = sorted(set(range(0, L + 1, a.layer_stride)) | set(mids) | {L})
        R, lens_names, aff_info = run_lenses(C, a.device, layers, mids, a.affine_src,
                                                [x for x in a.affine_langs.split(",") if x] or None,
                                                [x for x in a.mu_langs.split(",") if x] or None, tuned=not a.no_tuned)
        sd = sanity(C, R, a.device)
        T = item_table(C, R, lens_names, mids, covars).merge(data[keep_cols], left_on="item", right_on="id", how="left")
        name = os.path.basename(path).rsplit(".", 1)[0]
        per_model[name] = dict(C=C, R=R, T=T, lens=lens_names, mids=mids, L=L, sanity=sd, aff=aff_info)
        print(name, "sanity", sd)

    # 모델 간 공통 정답 문항
    common = None
    for v in per_model.values():
        ok = v["T"].groupby("item").correct.sum()
        s = set(ok[ok >= a.min_correct].index)
        common = s if common is None else common & s

    for name, v in per_model.items():
        d = os.path.join(a.out, name)
        os.makedirs(d, exist_ok=True)
        meta = v["C"]["meta"]
        R = v["R"].merge(meta[["pid", "item", "task", "template", "cond"]], on="pid")
        layer_E = R.groupby(["task", "cond", "lens", "layer"])[["p_latin", "p_han", "p_hangul", "p_other", "E", "en_lp", "ans1_lp", "dis1_lp"]].mean().reset_index()
        layer_E.to_csv(os.path.join(d, "layer_E.csv"), index=False)
        v["T"].to_csv(os.path.join(d, "items.csv"), index=False, encoding="utf-8-sig")
        meta.drop(columns=["prompt"]).to_csv(os.path.join(d, "prompts_scored.csv"), index=False, encoding="utf-8-sig")
        res_all, summ = {}, dict(model=v["C"]["model"], sanity=v["sanity"], affine=v["aff"], mids=v["mids"])
        sets = {"own_correct": None}
        if len(per_model) > 1:
            sets["common_correct"] = sorted(common)
        for sname, keep in sets.items():
            res, agg = analyze_items(v["T"], v["lens"], covars, a.min_correct, keep, a.nboot, a.boot_group)
            items = agg.item.unique()
            dec = decompose_E(v["R"], meta, v["mids"], items, v["lens"])
            res_all[sname] = (res, dec)
            summ[sname] = dict(items=res, E_decomposition=dec)
            agg.to_csv(os.path.join(d, f"items_agg_{sname}.csv"), index=False, encoding="utf-8-sig")
        top_tokens = top_tokens_mid(v["C"], R, v["mids"])
        summ["top_tokens_mid"] = top_tokens
        json.dump(summ, open(os.path.join(d, "summary.json"), "w"), ensure_ascii=False, indent=1, default=str)
        write_report(os.path.join(d, "report.md"), v["C"]["model"], res_all, v["sanity"], v["aff"], v["mids"], v["L"], top_tokens)
        print("저장:", d)


if __name__ == "__main__":
    main()
