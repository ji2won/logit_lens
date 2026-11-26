"""모델 실행·캐시 (GPU). 데이터셋 A 프롬프트 + (선택) 위키 문장.
 
저장하는 것 (cache_<모델>.pt 하나)
  meta        : 프롬프트 표 + 모델 결과(정답 여부, 정답·방해어 전체 로그확률 등)
  hs_last     : [N, L+1, d] fp16. 프롬프트 마지막 토큰의 hidden state.
                0 = 임베딩, ℓ = ℓ번째 블록 출력, L = 최종 norm 직전(pre-norm) residual.
                (HF의 hidden_states[-1]은 Llama에서 이미 norm이 적용된 값이라 섞으면 lens가 틀어진다.
                 그래서 L번째는 최종 norm 입력을 hook으로 따로 받는다.)
  unembed     : W_U, lm_head bias, 최종 norm 가중치·종류·eps  → GPU 없이 lens 계산 가능
  final_top   : 마지막 위치 실제 모델 출력의 top-1 id와 log p (sanity check용)
  wiki_mu     : [L+1, d] 위키 문장 절반(짝수 줄)의 전체 위치 평균        (--wiki 줄 때)
  wiki_X      : [M, L+1, d] fp16, 나머지 절반(홀수 줄)에서 뽑은 위치들  (affine 학습용)
  vocab_script: 토큰별 문자 체계 (latin/han/hangul/other)
 
    python run_cache.py --model meta-llama/Llama-3.2-3B --data ../dataset_A/A_final_v2.csv \
        --wiki wiki_ko_zh.txt --out cache/llama.pt
"""
import argparse
import json
import math
import random
import re
import sys
import time
 
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import prompts  # noqa: E402
 
NORM_PATHS = ["model.norm", "transformer.ln_f", "model.final_layernorm", "transformer.norm", "gpt_neox.final_layer_norm"]
 
 
def getattr_path(obj, path):
    for p in path.split("."):
        if not hasattr(obj, p):
            return None
        obj = getattr(obj, p)
    return obj
 
 
def find_final_norm(model):
    for p in NORM_PATHS:
        m = getattr_path(model, p)
        if m is not None and isinstance(m, torch.nn.Module):
            return p, m
    raise RuntimeError("최종 norm 모듈을 못 찾음. NORM_PATHS에 경로를 추가할 것")
 
 
def real_param(model, mod, pname):
    """모듈 파라미터의 실제 값. 일부 층을 CPU로 내린(offload) 경우 모듈의 가중치는 'meta'(빈 껍데기)이고
    실제 값은 accelerate hook의 weights_map에 있다. 거기도 없으면 체크포인트 파일에서 직접 읽는다."""
    p = getattr(mod, pname, None)
    if p is None:
        return None
    if p.device.type != "meta":
        return p.detach().cpu()
    hook = getattr(mod, "_hf_hook", None)
    for h in getattr(hook, "hooks", [hook]):
        wm = getattr(h, "weights_map", None)
        if wm is not None:
            try:
                return wm[pname].detach().cpu()
            except KeyError:
                pass
    # 체크포인트 safetensors에서 찾기 (묶인 lm_head는 embed_tokens 가중치)
    import glob
    import os
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    full = next(n for n, m in model.named_modules() if m is mod) + "." + pname
    cands = [full, full.replace("model.", "", 1)]
    if pname == "weight" and "lm_head" in full:
        cands += [k for k in ["model.embed_tokens.weight", "transformer.wte.weight"]]
    path = model.config._name_or_path
    d = path if os.path.isdir(path) else snapshot_download(path, allow_patterns=["*.safetensors"])
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        with safe_open(f, "pt") as sf:
            for k in cands:
                if k in sf.keys():
                    return sf.get_tensor(k)
    raise RuntimeError(f"{full} 값을 찾지 못함")
 
 
def norm_spec(model, mod):
    name = type(mod).__name__
    kind = "rms" if "rms" in name.lower() else ("ln" if isinstance(mod, torch.nn.LayerNorm) or "layernorm" in name.lower() else "unknown")
    eps = getattr(mod, "eps", None) or getattr(mod, "variance_epsilon", None) or getattr(mod, "epsilon", None) or 1e-5
    w = real_param(model, mod, "weight").float()
    b = real_param(model, mod, "bias")
    return dict(kind=kind, cls=name, eps=float(eps), weight=w, bias=None if b is None else b.float())
 
 
# ---------- 토큰 문자 체계 ----------
def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}
 
 
def token_bytes(tok, ids):
    inv = _bytes_to_unicode()
    out = []
    for t in tok.convert_ids_to_tokens(ids):
        if t is not None and all(ch in inv for ch in t):
            out.append(bytes(inv[ch] for ch in t))
        else:  # sentencepiece 등
            out.append((t or "").replace("▁", " ").encode("utf-8", "replace"))
    return out
 
 
def script_of(b: bytes) -> str:
    """토큰 바이트 → latin / han / hangul / other. 글자 조각은 UTF-8 선두 바이트로 판정."""
    s = b.decode("utf-8", "ignore")
    if re.search(r"[一-鿿㐀-䶿]", s):
        return "han"
    if re.search(r"[가-힣ᄀ-ᇿ㄰-㆏]", s):
        return "hangul"
    if re.search(r"[A-Za-z]", s):
        return "latin"
    st = b.lstrip(b" ")
    if st and 0xE4 <= st[0] <= 0xE9:   # U+4E00–U+9FFF 선두 바이트
        return "han"
    if st and 0xEA <= st[0] <= 0xED:   # U+AC00–U+D7A3 선두 바이트
        return "hangul"
    return "other"
 
 
# ---------- 본체 ----------
class Runner:
    def __init__(self, name, device, dtype, gpu_mem_gib=0.0, revision=None):
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, revision=revision)
        if device.startswith("cuda") and gpu_mem_gib > 0:
            # GPU 메모리가 모델보다 작을 때: 들어가는 만큼만 GPU에, 나머지 층은 CPU에 두고 돌림(accelerate).
            # 가중치 값은 그대로라 hidden state·확률은 같고 속도만 느려진다.
            self.model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=dtype, trust_remote_code=True, revision=revision, device_map="auto",
                max_memory={0: f"{gpu_mem_gib}GiB", "cpu": "48GiB"}).eval()
            print("층 배치:", sorted(set(map(str, self.model.hf_device_map.values()))), flush=True)
            device = "cuda:0"
        else:
            self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, trust_remote_code=True, revision=revision).to(device).eval()
        self.device = device
        self.norm_path, self.norm = find_final_norm(self.model)
        self._cap = None
        self.norm.register_forward_hook(lambda m, i, o: setattr(self, "_cap", i[0].detach()))
 
    def enc(self, text, special=True):
        return self.tok(text, add_special_tokens=special).input_ids
 
    def split_ids(self, prompt, sep, word):
        """프롬프트 id와 이어질 단어 id. 합쳐서 토큰화했을 때 접두가 같으면 그걸 쓰고,
        아니면(따옴표+한자가 한 토큰으로 붙는 경우 등) 단어를 따로 토큰화한다."""
        sp = not getattr(self, "chat", False)  # chat template 문자열에는 이미 특수 토큰이 들어 있음
        p = self.enc(prompt, special=sp)
        f = self.enc(prompt + sep + word, special=sp)
        if f[: len(p)] == p and len(f) > len(p):
            return p, f[len(p):], True
        return p, self.enc(sep + word, special=False), False
 
    @torch.no_grad()
    def forward(self, ids):
        x = torch.tensor([ids], device=self.device)
        out = self.model(input_ids=x, output_hidden_states=True)
        hs = list(out.hidden_states)
        hs[-1] = self._cap  # pre-norm 최종 residual
        return out.logits[0].float(), torch.stack([h[0].to(out.logits.device) for h in hs], 0)  # [T,V], [L+1,T,d]
 
    @torch.no_grad()
    def score(self, p, w):
        logits, hs = self.forward(p + w)
        lp = torch.log_softmax(logits, -1)
        n = len(p)
        toks = torch.tensor(w, device=lp.device)
        per = lp[n - 1 : n - 1 + len(w)].gather(1, toks[:, None])[:, 0]
        return per.sum().item(), per[0].item(), lp[n - 1], hs[:, n - 1]
 
    @torch.no_grad()
    def greedy(self, p, n=8):
        out = self.model.generate(torch.tensor([p], device=self.device), max_new_tokens=n, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id)
        return self.tok.decode(out[0, len(p):], skip_special_tokens=True)
 
 
def first_word(gen):
    """출력 첫 줄에서 따옴표·공백을 걷어낸 첫 단어 (한자/한글이 이어지는 부분)."""
    g = gen.strip().split("\n")[0].strip().strip('"“”「」').strip()
    # 대화 형식에서 모델이 예시 형식을 따라 "한국어: 봄 봄", "中文: 春天"처럼 이름표를 붙여 답하는 경우
    g = re.sub(r"^(한국어|중국어|中文|汉语|中国语)\s*[:：=]\s*", "", g).strip().strip('"“”「」').strip()
    m = re.match(r"[\u4e00-\u9fff\uac00-\ud7a3]+", g)
    return m.group(0) if m else g
 
 
def is_correct_strict(gen, answer):
    return first_word(gen).startswith(answer)
 
 
def to_simplified(w):
    """번체 → 간체 (鶴 → 鹤, 鐵 → 铁). zhconv가 없으면 그대로."""
    try:
        import zhconv
        return zhconv.convert(w, "zh-cn")
    except ImportError:
        return w
 
 
def is_correct(gen, answer):
    """정답이거나, 정답의 앞부분만 낸 경우(春天 → 春, 星星 → 星, 胸部 → 胸)도 정답.
    번체로 답한 경우는 간체로 바꿔서 비교한다(鶴 → 鹤). 일본식 한자(鉄)는 오답으로 남는다.
    한 글자 단어가 같은 뜻인 경우가 많고, 이 실험의 정답 판정은 '그 단어를 아는가'만 거르면 되기 때문."""
    w = to_simplified(first_word(gen))
    return bool(w) and (w.startswith(answer) or answer.startswith(w))
 
 
def add_wiki(cache, R, a):
    """위키 문장 상태를 캐시에 넣는다. 파일 한 줄 = '언어\t문장' (언어 표시가 없으면 '?').
    짝수 줄 → μ(언어별 합계로 저장), 홀수 줄 → affine 학습용 위치 표본(언어 표시 함께 저장).
    analyze.py에서 --affine-langs / --mu-langs로 어떤 언어 문장을 쓸지 고를 수 있다."""
    t0 = time.time()
    lines = [l.rstrip("\n") for l in open(a.wiki, encoding="utf-8") if l.strip()]
    mu = {}
    X, XL = [], []
    for j, line in enumerate(lines):
        lang, text = line.split("\t", 1) if "\t" in line else ("?", line)
        ids = R.enc(text.strip())[: a.wiki_max_tok]
        if len(ids) < 4:
            continue
        _, hs = R.forward(ids)
        hs = hs[:, 1:].float()  # 0번(BOS/첫 토큰, attention sink) 제외
        if j % 2 == 0:
            s_, n_ = mu.get(lang, (0, 0))
            mu[lang] = (s_ + hs.sum(1).double().cpu(), n_ + hs.shape[1])
        else:
            pos = random.sample(range(hs.shape[1]), min(a.wiki_pos, hs.shape[1]))
            X.append(hs[:, pos].permute(1, 0, 2).to(torch.float16).cpu())
            XL += [lang] * len(pos)
        if j % 200 == 0:
            print(f"wiki {j}/{len(lines)} {time.time()-t0:.0f}s", flush=True)
    tot = sum(v[0] for v in mu.values())
    n = sum(v[1] for v in mu.values())
    cache["wiki_mu"] = (tot / n).float()
    cache["wiki_mu_by_lang"] = {k: (v[0].float(), v[1]) for k, v in mu.items()}
    cache["wiki_X"] = torch.cat(X)
    cache["wiki_X_lang"] = XL
    cache["wiki_n_mu_tokens"] = n
    print("wiki 언어별 μ 토큰 수:", {k: v[1] for k, v in mu.items()},
          "affine 표본 수:", pd.Series(XL).value_counts().to_dict(), flush=True)
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None, help="모델 저장소의 특정 커밋 (EXAONE은 transformers 4.x용 옛 코드가 필요)")
    ap.add_argument("--data", default="../dataset_A/A_final_v2.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wiki", default="", help="한 줄에 한 문장. 짝수 줄 → μ, 홀수 줄 → affine 학습")
    ap.add_argument("--wiki-only", action="store_true",
                    help="문항은 다시 돌리지 않고, --out에 있는 기존 캐시의 위키 부분만 새로 계산해 덮어씀")
    ap.add_argument("--chat", action="store_true",
                    help="Instruct 모델용: 프롬프트를 대화 형식(chat template)으로 감싸고 지시문을 붙임")
    ap.add_argument("--wiki-max-tok", type=int, default=64)
    ap.add_argument("--wiki-pos", type=int, default=8, help="affine용으로 문장당 뽑을 위치 수")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="디버그: 문항 수 제한")
    ap.add_argument("--allow-cpu", action="store_true", help="GPU가 안 잡혀도 CPU로 돌림 (매우 느림)")
    ap.add_argument("--gpu-mem", type=float, default=-1,
                    help="GPU에 올릴 최대 GiB. -1이면 GPU 크기를 보고 자동(모델이 안 들어가면 일부 층을 CPU로), 0이면 전부 GPU")
    a = ap.parse_args()
    print(f"torch {torch.__version__} | device={a.device} | CUDA 사용 가능={torch.cuda.is_available()}", flush=True)
    if a.device == "cpu" and not a.allow_cpu:
        sys.exit("GPU가 안 잡혀서 멈춤. torch CUDA 빌드와 서버 드라이버가 맞는지 확인할 것 (README 참고). "
                 "그래도 CPU로 돌리려면 --allow-cpu")
    random.seed(0)
 
    df = pd.read_csv(a.data)
    if a.limit:
        df = df.head(a.limit)
    P = prompts.build(df) if "ko_control_final" in df else prompts.build_C(df)
    gm = a.gpu_mem
    if gm < 0:
        gm = 0.0
        if a.device.startswith("cuda"):
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"GPU: {torch.cuda.get_device_name(0)} {total:.1f}GiB", flush=True)
            if total < 16:  # 3B bf16 ≈ 6.4GiB + 활성값 여유
                gm = max(1.0, total - 1.5)
    R = Runner(a.model, a.device, getattr(torch, a.dtype), gm, a.revision)
    tok = R.tok
    t0 = time.time()
    # 판독용 가중치를 먼저 꺼내 둠 (실패하면 긴 실행 전에 바로 멈추도록)
    W = R.model.get_output_embeddings()
    Wb = real_param(R.model, W, "bias")
    unembed = dict(W=real_param(R.model, W, "weight").to(torch.float16), b=None if Wb is None else Wb.float(),
                   norm=norm_spec(R.model, R.norm), norm_path=R.norm_path)
    print("W_U", tuple(unembed["W"].shape), "norm", unembed["norm"]["cls"], unembed["norm"]["kind"], flush=True)
 
    if a.wiki_only:
        cache = torch.load(a.out, weights_only=False, map_location="cpu")
        assert cache["model"] == a.model, f"캐시 모델({cache['model']})과 --model이 다름"
        add_wiki(cache, R, a)
        torch.save(cache, a.out)
        print(f"위키 부분만 갱신: {a.out}  {time.time()-t0:.0f}s")
        return
 
    R.chat = a.chat
    if a.chat:
        P = prompts.to_chat(P, tok)
        print("대화 형식 예시:\n" + P.prompt.iloc[0] + "⟨답⟩", flush=True)
 
    recs, hs_last, top = [], [], []
    for i, r in P.iterrows():
        p, wa, ja = R.split_ids(r.prompt, r.sep, r.answer)
        _, wd, jd = R.split_ids(r.prompt, r.sep, r.distractor)
        la, la1, lp_last, h = R.score(p, wa)
        ld, ld1, _, _ = R.score(p, wd)
        rec = dict(lp_ans=la, lp_ans_tok1=la1, ntok_ans=len(wa), lp_dis=ld, lp_dis_tok1=ld1, ntok_dis=len(wd),
                   joint_tok_ans=ja, joint_tok_dis=jd, ans_ids=json.dumps(wa), dis_ids=json.dumps(wd))
        if r.cond == "input":
            g = R.greedy(p)
            rec.update(gen=g, correct=is_correct(g, r.answer), correct_strict=is_correct_strict(g, r.answer))
        recs.append(rec)
        hs_last.append(h.to(torch.float16).cpu())
        v, ix = lp_last.max(-1)
        top.append((ix.item(), v.item()))
        if i % 100 == 0:
            print(f"{i}/{len(P)} {time.time()-t0:.0f}s", flush=True)
    meta = pd.concat([P.reset_index(drop=True), pd.DataFrame(recs)], axis=1)
 
    # 영어 단어 토큰 (lens 점수용): 변형 4개의 첫 토큰 id
    en_ids = {}
    for en in df.en.unique():
        ids = set()
        for v in [en, " " + en, en.capitalize(), " " + en.capitalize()]:
            e = R.enc(v, special=False)
            if e:
                ids.add(e[0])
        en_ids[en] = sorted(ids)
    ntok_en = {en: len(R.enc(" " + en, special=False)) for en in df.en.unique()}
 
    # 데이터셋 C: 무관 단어(영·중·한)의 첫 토큰 id. whole = 첫 토큰이 글자 조각이 아닌 온전한 글자인지
    extra_ids = {}
    for col in ["en_unrelated", "zh_unrelated", "ko_unrelated", "zh"]:
        if col not in df:
            continue
        extra_ids[col] = {}
        for w in df[col].dropna().unique():
            vs = [w, " " + w, w.capitalize(), " " + w.capitalize()] if col.startswith("en") else [w, " " + w]
            ids = sorted({e[0] for e in (R.enc(v, special=False) for v in vs) if e})
            first = R.enc(w, special=False)[:1]
            b = token_bytes(tok, first)[0] if first else b""
            try:
                whole = bool(b.decode("utf-8").strip())
            except UnicodeDecodeError:
                whole = False
            extra_ids[col][w] = dict(ids=ids, whole=whole)
 
    V = unembed["W"].shape[0]
    vb = token_bytes(tok, list(range(V)))
    vocab_script = [script_of(b) for b in vb]
 
    cache = dict(model=a.model, revision=getattr(a, "revision", None), chat=a.chat, meta=meta, hs_last=torch.stack(hs_last), unembed=unembed,
                 final_top=torch.tensor(top), en_ids=en_ids, ntok_en=ntok_en, vocab_script=vocab_script,
                 n_layers=hs_last[0].shape[0] - 1, extra_ids=extra_ids)
 
    if a.wiki:
        add_wiki(cache, R, a)
 
    torch.save(cache, a.out)
    acc = meta[meta.cond == "input"].groupby("task").correct.mean().to_dict()
    print(f"저장: {a.out}  N={len(meta)}  정답률={acc}  {time.time()-t0:.0f}s")
 
 
if __name__ == "__main__":
    main()
 