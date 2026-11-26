"""인과 개입 실험 (GPU): 중간 층에서 영어 다의어 방향(예: spring)을 빼면 정답·방해어 확률이 어떻게 바뀌나.
 
질문
  봄 → 春天 번역에서 방해어 弹簧(spring의 '용수철' 뜻)이 오르는 것이 영어 spring을 거쳐서라면,
  중간 층 residual에서 spring 방향을 없앴을 때 log p(弹簧)가 '무관 영어 단어 방향을 똑같이 뺀 것'보다 더 떨어져야 한다.
  비교는 문항 안에서 한다(같은 프롬프트, 같은 배치, 같은 층·위치). 그래서 문항 간 상관(예측 2)의 검정력 문제가 없다.
 
방향 (dirtype)
  unembed  : g ⊙ (W_U[t] − mean W_U).  t = " "+영어단어의 첫 토큰, g = 최종 RMSNorm 가중치.
             logit lens에서 그 영어 토큰 점수를 만드는 residual 방향 (공통 성분은 뺌).
  meandiff : 층별로  mean_템플릿 h_ℓ(영어 단어 w를 직접 넣은 프롬프트, 마지막 위치)
                   − 모든 영어 단어에 대한 같은 값의 평균.   (en 조건 프롬프트에서 뽑은 '개념' 방향)
 
개입 (행 종류 role, 모두 단위 벡터 d̂ 기준)
  target       : h ← h − (h·d̂_T) d̂_T                (spring 성분 제거)
  ctrl_ablate  : h ← h − (h·d̂_c) d̂_c                (대조 단어 성분 제거. 같은 조작, 다른 단어)
  ctrl_matched : h ← h − k d̂_c,  k = 같은 배치의 target 행이 그 층·위치에서 뺀 양 (h_T·d̂_T)
                 → 층마다 ‖Δh‖가 target과 같다. 그래서 (dirtype, 설정) 한 묶음(1+2K행)은 항상 같은 배치에 넣는다.
  clean        : 개입 없음
대조 단어: 데이터셋 C의 영어 단어(en, en_unrelated) 중 데이터셋 A·B의 영어 단어가 아닌 것,
           Llama 토큰 1개인 것, 문항의 영어 다의어와 빈도(zipf)가 가장 가까운 K개.
 
개입 설정 (--configs, 이름:층:위치)
  층 ℓ = ℓ번째 블록에 들어가는 residual (= HF hidden_states[ℓ], run_cache.py의 ℓ와 같은 번호).
  위치 all = 모든 위치(few-shot·입력어·단어 토큰 포함), last = 프롬프트 마지막 위치만.
 
수치 정합성 (이전 placebo 실행의 batch vs single 차이 문제 때문에)
  - 한 번의 forward = 같은 토큰열(프롬프트+단어)을 행 수만큼 복사한 배치. 길이가 모두 같아서 패딩이 없다.
  - 배치마다 맨 앞에 clean 행을 두고, 효과는 '같은 배치 안의 clean 행'과의 차이로만 잰다.
  - 마지막 배치도 clean 행으로 채워 배치 크기를 항상 같게 한다.
  - check.json: (a) 같은 배치 안 clean 행끼리 차이(0이어야 함), (b) clean 행 vs 한 개씩(run_cache.py 방식) 차이,
                (c) 같은 배치를 두 번 돌린 차이(결정성), (d) 개입 후 d̂ 성분이 0이 됐는지.
 
    python run_ablate.py --model meta-llama/Llama-3.2-3B --data ../dataset_A/A_final_v2.csv \
        --ctrl-src ../dataset_C/C_final.csv --exclude-en ../dataset_B/B_final.csv --out ablate/llama
 
출력 (--out 접두어)
  <out>_rows.csv.gz   : 프롬프트 × 단어(ans/dis) × 행. lp(단어 전체 log p), delta(= lp − 같은 배치 clean lp), pert(‖Δh‖, 마지막 위치)
  <out>_lens.csv.gz   : 프롬프트 × 행. 개입 뒤 read 층(기본 22)에서 logit lens로 읽은 target/자기 방향 단어 log p (조작 확인)
  <out>_prompts.csv   : 프롬프트 정보, greedy 출력, 정답 여부
  <out>_controls.csv  : 문항별 대조 단어와 빈도, 방향 코사인
  <out>_check.json    : 정합성 검사
"""
import argparse
import json
import os
import sys
import time
 
import numpy as np
import pandas as pd
import torch
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts  # noqa: E402
import run_cache as rc  # noqa: E402
 
 
# ---------------- 개입 훅 ----------------
class Intervener:
    """디코더 블록 입력(pre-hook)에서 행마다 다른 방향을 뺀다.
    plan[ℓ] = dict(D=[B,d] 자기 방향(비활성 행은 0), src=[B] 뺄 양을 가져올 행(자기 자신 또는 같은 묶음의 target 행),
                   matched=[B] bool, active=[B] bool), pos=[B,T] 위치 마스크."""
 
    def __init__(self, layers):
        self.layers = layers
        self.plan = None
        self.pos = None
        self.last = None      # 프롬프트 마지막 위치 index
        self.pert = None      # [B] 마지막 위치에서 뺀 양의 제곱합
        self.resid = None     # [B] 개입 뒤 남은 자기 방향 성분 |h'·d̂| 최댓값 (검사용)
        for i, m in enumerate(layers):
            m.register_forward_pre_hook(self._mk(i), with_kwargs=True)
 
    def _mk(self, li):
        def hook(mod, args, kwargs):
            if self.plan is None or li not in self.plan:
                return None
            pl = self.plan[li]
            if args:
                h = args[0]
            else:
                h = kwargs["hidden_states"]
            dev = h.device
            D = pl["D"].to(dev)
            src = pl["src"].to(dev)
            act = pl["active"].to(dev)
            mat = pl["matched"].to(dev)
            pos = self.pos[:, : h.shape[1]].to(dev)
            hf = h.float()
            p_own = torch.einsum("btd,bd->bt", hf, D) * pos * act[:, None].float()
            # matched 행: 같은 배치 안 target 행(src)이 이 층·위치에서 뺀 양을 그대로 자기 방향으로 뺀다
            coef = p_own[src]
            hn = hf - coef[..., None] * D[:, None, :]
            # 비활성 행은 원래 텐서 그대로(비트 단위로 같게)
            out = torch.where(act[:, None, None], hn.to(h.dtype), h)
            self.pert += (coef[:, self.last] ** 2).cpu()
            r = torch.einsum("btd,bd->bt", out.float(), D)[:, self.last].abs()
            self.resid = torch.maximum(self.resid, torch.where(act & ~mat, r, torch.zeros_like(r)).cpu())
            if args:
                return (out,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = out
            return args, kwargs
        return hook
 
 
class Capture:
    """read 층 블록 출력의 마지막 위치를 저장 (조작 확인용)."""
 
    def __init__(self, mod):
        self.on = False
        self.h = None
        self.last = None
        mod.register_forward_hook(self)
 
    def __call__(self, mod, inp, out):
        if self.on:
            o = out[0] if isinstance(out, tuple) else out
            self.h = o[:, self.last].detach().float().cpu()
 
 
class Slicer:
    """lm_head 입력을 단어 예측 위치 [n-1, n-1+L)로만 자른다 (로짓 메모리 절약)."""
 
    def __init__(self, mod):
        self.s = None
        self.L = None
        mod.register_forward_pre_hook(self)
 
    def __call__(self, mod, args):
        if self.s is None:
            return None
        return (args[0][:, self.s: self.s + self.L],) + tuple(args[1:])
 
 
def decoder_layers(model):
    for p in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        m = rc.getattr_path(model, p)
        if m is not None:
            return list(m)
    raise RuntimeError("디코더 블록 목록을 못 찾음")
 
 
def parse_configs(s):
    out = []
    for c in s.split(","):
        name, ls, pos = c.split(":")
        if "-" in ls:
            a, b = map(int, ls.split("-"))
            layers = list(range(a, b + 1))
        else:
            layers = [int(x) for x in ls.split("+")]
        assert pos in ("all", "last")
        out.append(dict(name=name, layers=layers, pos=pos))
    return out
 
 
def first_tok(R, w):
    e = R.enc(" " + w, special=False)
    return e[0], len(e)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--data", default="../dataset_A/A_final_v2.csv")
    ap.add_argument("--ctrl-src", default="../dataset_C/C_final.csv", help="대조 영어 단어 후보 (en, en_unrelated + zipf)")
    ap.add_argument("--exclude-en", default="../dataset_B/B_final.csv", help="후보에서 뺄 영어 단어가 든 파일들(쉼표). A는 자동 제외")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=8, help="문항당 대조 단어 수")
    ap.add_argument("--configs", default="W:7-21:all,W:7-21:last,L10:10:all,L16:16:all")
    ap.add_argument("--dirtypes", default="unembed,meandiff")
    ap.add_argument("--conds", default="input,en", help="input = 한국어 입력(주 실험), en = 영어 단어 직접 입력(양성 대조)")
    ap.add_argument("--templates", default="T1,T2,T3,T4")
    ap.add_argument("--read-layer", type=int, default=22, help="조작 확인: 이 hidden_states 번호(= 블록 read-1 출력)에서 lens로 읽음")
    ap.add_argument("--chunk", type=int, default=72, help="한 forward의 행 수(clean 1개 포함). 메모리 부족이면 줄일 것")
    ap.add_argument("--no-greedy", action="store_true")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-mem", type=float, default=-1)
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="디버그: 문항 수 제한")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"torch {torch.__version__} | device={a.device}", flush=True)
    if a.device == "cpu" and not a.allow_cpu:
        sys.exit("GPU가 안 잡힘. CPU로 돌리려면 --allow-cpu")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
 
    df = pd.read_csv(a.data)
    if a.limit:
        df = df.head(a.limit)
    configs = parse_configs(a.configs)
    dirtypes = a.dirtypes.split(",")
    templates = a.templates.split(",")
 
    gm = a.gpu_mem
    if gm < 0:
        gm = 0.0
        if a.device.startswith("cuda"):
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"GPU: {torch.cuda.get_device_name(0)} {total:.1f}GiB", flush=True)
            if total < 16:
                gm = max(1.0, total - 2.0)
    R = rc.Runner(a.model, a.device, getattr(torch, a.dtype), gm, a.revision)
    R.chat = a.chat
    sp = not a.chat
    model = R.model
    blocks = decoder_layers(model)
    nL = len(blocks)
    assert all(0 <= l < nL for c in configs for l in c["layers"]), f"층 번호는 0–{nL-1}"
    assert 1 <= a.read_layer <= nL
    I = Intervener(blocks)
    cap = Capture(blocks[a.read_layer - 1])
    slicer = Slicer(model.get_output_embeddings())
 
    W = rc.real_param(model, model.get_output_embeddings(), "weight").float()  # [V,d] CPU
    nspec = rc.norm_spec(model, R.norm)
    g = nspec["weight"].float()
    Wmean = W.mean(0)
 
    def lens_logp(h):  # h [B,d] CPU float → [B,V] log p (최종 norm + W_U)
        if nspec["kind"] == "rms":
            x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + nspec["eps"])
        else:
            x = torch.nn.functional.layer_norm(h, (h.shape[-1],), eps=nspec["eps"])
        x = x * g
        if nspec["bias"] is not None:
            x = x + nspec["bias"]
        return torch.log_softmax(x @ W.T, -1)
 
    # ---------------- 대조 단어 후보 ----------------
    excl = {w.lower() for w in df.en}
    for f in [x for x in a.exclude_en.split(",") if x]:
        if os.path.exists(f):
            e = pd.read_csv(f)
            for c in [c for c in e.columns if c.startswith("en")]:
                excl |= {str(w).lower() for w in e[c].dropna()}
    C = pd.read_csv(a.ctrl_src)
    cand = pd.concat([C[["en", "zipf_en"]].rename(columns={"zipf_en": "zipf"}),
                      C[["en_unrelated", "zipf_en_unrelated"]].rename(columns={"en_unrelated": "en", "zipf_en_unrelated": "zipf"})])
    cand = cand.dropna().drop_duplicates("en")
    cand = cand[~cand.en.str.lower().isin(excl) & cand.en.str.match(r"^[a-z]+$")]
    cand["tok"], cand["ntok"] = zip(*[first_tok(R, w) for w in cand.en])
    cand = cand[cand.ntok == 1].reset_index(drop=True)
    print(f"대조 단어 후보 {len(cand)}개 (A·B 영어 제외, 토큰 1개)", flush=True)
 
    ctrl_of = {}
    for _, r in df.iterrows():
        dz = (cand.zipf - r.zipf_en).abs().values + rng.uniform(0, 1e-6, len(cand))
        ctrl_of[r.id] = list(cand.en.values[np.argsort(dz)[: a.K]])
    used_words = sorted(set(df.en) | {w for v in ctrl_of.values() for w in v})
    tok_of = {w: first_tok(R, w) for w in used_words}
 
    # ---------------- 방향 ----------------
    mid_layers = sorted({l for c in configs for l in c["layers"]})
    dirs = {}  # (dirtype, word) -> [nL, d] 단위벡터 (unembed는 층마다 같음)
    if "unembed" in dirtypes:
        for w in used_words:
            u = g * (W[tok_of[w][0]] - Wmean)
            dirs[("unembed", w)] = (u / u.norm()).expand(nL, -1).contiguous()
    if "meandiff" in dirtypes:
        t0 = time.time()
        Hs = {}
        for w in used_words:
            acc = []
            for t in templates:
                pr, _ = prompts._fill(prompts.TEMPLATES[t], w)
                if a.chat:
                    pr = prompts.to_chat(pd.DataFrame([dict(task="translate", prompt=pr, sep="")]), R.tok).prompt.iloc[0]
                with torch.no_grad():
                    out = model(input_ids=torch.tensor([R.enc(pr, special=sp)], device=R.device), output_hidden_states=True)
                acc.append(torch.stack([out.hidden_states[l][0, -1].float().cpu() for l in range(nL)]))
            Hs[w] = torch.stack(acc).mean(0)  # [nL, d]
        mu = torch.stack(list(Hs.values())).mean(0)
        for w in used_words:
            v = Hs[w] - mu
            dirs[("meandiff", w)] = v / v.norm(dim=-1, keepdim=True)
        print(f"meandiff 방향 {len(used_words)}단어 × {len(templates)}템플릿  {time.time()-t0:.0f}s", flush=True)
 
    crec = []
    for _, r in df.iterrows():
        for c in ctrl_of[r.id]:
            row = dict(item=r.id, en=r.en, zipf_en=r.zipf_en, ctrl=c, zipf_ctrl=float(cand.set_index("en").zipf[c]))
            for dt in dirtypes:
                cs = (dirs[(dt, r.en)][mid_layers] * dirs[(dt, c)][mid_layers]).sum(-1)
                row[f"cos_{dt}"] = float(cs.mean())
            crec.append(row)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame(crec).to_csv(a.out + "_controls.csv", index=False, encoding="utf-8")
 
    # ---------------- 행 설계 ----------------
    def rows_for(en, ctrls):
        """[(dirtype, config, role, word)]. 맨 앞은 clean."""
        rs = []
        for dt in dirtypes:
            for c in configs:
                rs.append((dt, c["name"] + ":" + c["pos"], "target", en))
                rs += [(dt, c["name"] + ":" + c["pos"], "ctrl_ablate", w) for w in ctrls]
                rs += [(dt, c["name"] + ":" + c["pos"], "ctrl_matched", w) for w in ctrls]
        return rs
 
    cfg_by = {c["name"] + ":" + c["pos"]: c for c in configs}
    d_model = W.shape[1]
 
    def make_plan(rs, en, T, n):
        """rs: 개입 행 목록(clean 행 미포함). 배치 = [clean] + rs. 위치 마스크는 길이 T, 프롬프트 길이 n."""
        B = len(rs) + 1
        plan = {}
        pos = torch.zeros(B, T)
        for li in sorted({l for _, cf, _, _ in rs for l in cfg_by[cf]["layers"]}):
            plan[li] = dict(D=torch.zeros(B, d_model), src=torch.arange(B),
                            matched=torch.zeros(B, dtype=torch.bool), active=torch.zeros(B, dtype=torch.bool))
        tgt_row = {}
        for j, (dt, cf, role, w) in enumerate(rs, start=1):
            if role == "target":
                tgt_row[(dt, cf)] = j
        for j, (dt, cf, role, w) in enumerate(rs, start=1):
            if dt == "_pad":  # 배치 크기 맞춤용 clean 행
                continue
            c = cfg_by[cf]
            if c["pos"] == "all":
                pos[j] = 1
            else:
                pos[j, n - 1] = 1
            for li in c["layers"]:
                plan[li]["D"][j] = dirs[(dt, w)][li]
                plan[li]["active"][j] = True
                if role == "ctrl_matched":
                    plan[li]["src"][j] = tgt_row[(dt, cf)]  # 묶음이 배치 안에 함께 있어야 함 (KeyError면 chunk 설정 오류)
                    plan[li]["matched"][j] = True
        return plan, pos
 
    @torch.no_grad()
    def run_chunk(p, wids, rs, en, want_lens):
        n = len(p)
        T = n + len(wids)
        plan, pos = make_plan(rs, en, T, n)
        B = len(rs) + 1
        x = torch.tensor([p + wids] * B, device=R.device)
        I.plan, I.pos, I.last = plan, pos, n - 1
        I.pert = torch.zeros(B)
        I.resid = torch.zeros(B)
        cap.on, cap.last = want_lens, n - 1
        slicer.s, slicer.L = n - 1, len(wids)
        try:
            out = model(input_ids=x, use_cache=False)
        finally:
            I.plan = None
            slicer.s = None
            cap.on = False
        lp = torch.log_softmax(out.logits.float(), -1)
        t = torch.tensor(wids, device=lp.device)
        per = lp[:, : len(wids)].gather(2, t[None, :, None].expand(B, -1, 1))[..., 0]  # [B, L]
        return per.sum(1).cpu(), per[:, 0].cpu(), I.pert.sqrt().clone(), I.resid.clone(), (cap.h if want_lens else None)
 
    # ---------------- 프롬프트 ----------------
    P = prompts.build(df, templates=templates, with_repeat=False)
    P = P[P.cond.isin(a.conds.split(","))].reset_index(drop=True)
    if a.chat:
        P = prompts.to_chat(P, R.tok)
    zipf_of = df.set_index("id").zipf_en.to_dict()
    print(f"프롬프트 {len(P)}개, 문항당 행 {1 + len(rows_for('x', ['c'] * a.K))}개, chunk {a.chunk}", flush=True)
 
    recs, lrec, meta = [], [], []
    chk = dict(clean_within_batch=[], clean_vs_single=[], rerun=[], resid_after_ablate=0.0)
    t0 = time.time()
    for i, r in P.iterrows():
        p = R.enc(r.prompt, special=sp)
        ctrls = ctrl_of[r["item"]]
        rs_all = rows_for(r.en, ctrls)
        # (dirtype, 설정) 묶음 단위로 배치를 채운다. 배치마다 clean 행 최소 2개(맨 앞 + 끝 패딩)
        gsize = 1 + 2 * len(ctrls)
        per_chunk = ((a.chunk - 2) // gsize) * gsize
        assert per_chunk > 0, f"--chunk는 {gsize + 2} 이상이어야 함"
        chunks = [rs_all[s: s + per_chunk] for s in range(0, len(rs_all), per_chunk)]
        for kind, word in [("ans", r.answer), ("dis", r.distractor)]:
            _, wids, joint = R.split_ids(r.prompt, r.sep, word)
            for rs in chunks:
                npad = a.chunk - 1 - len(rs)  # 끝을 clean 행으로 채워 배치 크기를 항상 a.chunk로
                rs_real = rs
                tot, f1, pert, resid, hl = run_chunk_pad(run_chunk, p, wids, rs_real, npad, r.en,
                                                         want_lens=(kind == "ans"))
                base = tot[0].item()
                for j, row in enumerate(rs_real, start=1):
                    dt, cf, role, w = row
                    recs.append((r.pid, kind, dt, cf, role, w, tot[j].item(), tot[j].item() - base,
                                 f1[j].item() - f1[0].item(), pert[j].item(), len(wids)))
                recs.append((r.pid, kind, "-", "-", "clean", "-", base, 0.0, 0.0, 0.0, len(wids)))
                chk["clean_within_batch"].append(float((tot[len(rs_real) + 1:] - tot[0]).abs().max()))
                chk["resid_after_ablate"] = max(chk["resid_after_ablate"], float(resid.max()))
                if hl is not None:
                    LP = lens_logp(hl)
                    tt = tok_of[r.en][0]
                    for j, row in enumerate([None] + rs_real):
                        own = tok_of[row[3]][0] if row else tt
                        lrec.append((r.pid, row[0] if row else "-", row[1] if row else "-", row[2] if row else "clean",
                                     row[3] if row else "-", LP[j, tt].item(), LP[j, own].item()))
            if i < 4:
                ref, _, _, _ = R.score(p, wids)
                base = [x for x in recs if x[0] == r.pid and x[1] == kind and x[4] == "clean"][0][6]
                chk["clean_vs_single"].append(dict(pid=int(r.pid), kind=kind, batch=base, single=ref, diff=abs(base - ref)))
                if kind == "ans":
                    rs = chunks[0]
                    a1 = run_chunk_pad(run_chunk, p, wids, rs, a.chunk - 1 - len(rs), r.en, False)[0]
                    a2 = run_chunk_pad(run_chunk, p, wids, rs, a.chunk - 1 - len(rs), r.en, False)[0]
                    chk["rerun"].append(float((a1 - a2).abs().max()))
        m = dict(pid=r.pid, item=r["item"], en=r.en, zipf_en=zipf_of[r["item"]], template=r.template, cond=r.cond,
                 word=r.word, answer=r.answer, distractor=r.distractor, n_prompt_tok=len(p))
        if r.cond == "input" and not a.no_greedy:
            gtxt = R.greedy(p)
            m.update(gen=gtxt, correct=rc.is_correct(gtxt, r.answer))
        meta.append(m)
        if i % 20 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(P)}  {el:.0f}s  (남은 시간 약 {el/(i+1)*(len(P)-i-1)/60:.0f}분)", flush=True)
 
    cols = ["pid", "kind", "dirtype", "config", "role", "dir_word", "lp", "delta", "delta_tok1", "pert", "ntok"]
    pd.DataFrame(recs, columns=cols).to_csv(a.out + "_rows.csv.gz", index=False, encoding="utf-8", compression="gzip")
    pd.DataFrame(lrec, columns=["pid", "dirtype", "config", "role", "dir_word", "lens_lp_target", "lens_lp_own"]).to_csv(
        a.out + "_lens.csv.gz", index=False, encoding="utf-8", compression="gzip")
    pd.DataFrame(meta).to_csv(a.out + "_prompts.csv", index=False, encoding="utf-8")
    summ = dict(model=a.model, chat=a.chat, configs=a.configs, dirtypes=dirtypes, K=a.K, read_layer=a.read_layer,
                minutes=(time.time() - t0) / 60,
                max_clean_within_batch=max(chk["clean_within_batch"] or [0.0]),
                max_clean_vs_single=max([c["diff"] for c in chk["clean_vs_single"]] or [0.0]),
                max_rerun=max(chk["rerun"] or [0.0]), max_resid_after_ablate=chk["resid_after_ablate"],
                clean_vs_single=chk["clean_vs_single"])
    summ["ok"] = bool(summ["max_clean_within_batch"] == 0 and summ["max_rerun"] == 0 and summ["max_resid_after_ablate"] < 0.5)
    json.dump(summ, open(a.out + "_check.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "clean_vs_single"}, ensure_ascii=False))
    print(f"저장: {a.out}_rows.csv.gz 등  {(time.time()-t0)/60:.0f}분")
 
 
def run_chunk_pad(run_chunk, p, wids, rs, npad, en, want_lens):
    """배치 크기를 고정하려고 끝에 비활성(clean) 행 npad개를 붙인다.
    비활성 행은 dirtype/config가 필요 없으므로 첫 행을 복사하되 '활성' 표시를 끈 행으로 만든다."""
    if npad == 0:
        return run_chunk(p, wids, rs, en, want_lens)
    return run_chunk(p, wids, rs + [("_pad",) + rs[0][1:]] * npad, en, want_lens)
 
 
if __name__ == "__main__":
    main()
 