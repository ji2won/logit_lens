"""무관 중국어 단어 대조 실험 (GPU). 최종 출력만 본다(lens 없음).

같은 프롬프트(데이터셋 A, 템플릿 T1–T4, 조건 input / ctrl1 / ctrl2)에서
후보 목록(placebo_pool.csv)의 중국어 단어 전부에 대해 '단어 전체' 로그확률을 잰다.
  log p(단어 | 프롬프트) = 단어 토큰 로그확률의 합 (teacher forcing). 첫 토큰만 쓰지 않는다.
방해어·정답도 후보 목록 안에 들어 있으므로, 진짜 방해어와 무관 단어가 정확히 같은 방법으로 계산된다.

속도: 프롬프트 하나에 후보 단어들을 한 배치로 붙여서(오른쪽 패딩) 한 번에 돌린다.
lm_head에 들어가는 위치를 단어 구간으로 잘라서 로짓 메모리를 줄인다(6GB GPU + CPU offload용).

    python run_placebo.py --model meta-llama/Llama-3.2-3B --data ../dataset_A/A_final_v2.csv \
        --pool placebo_pool.csv --out placebo/llama
    python run_placebo.py --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct --revision <커밋> --chat ... --out placebo/exaone_chat

출력 (--out 접두어)
  <out>_prompts.csv : 프롬프트별 정보(문항, 템플릿, 조건, 입력 단어, greedy 출력, 정답 여부)
  <out>_scores.csv.gz : 프롬프트 × 후보 단어별 lp(전체), lp_tok1(첫 토큰), ntok, joint
  <out>_check.json : 같은 값을 run_cache.py 방식(한 개씩)으로 다시 잰 것과의 차이(sanity)
"""
import argparse
import json
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts  # noqa: E402
import run_cache as rc  # noqa: E402


class Slicer:
    """lm_head 입력을 [:, s:s+L] 위치로 자른다. 쓰지 않을 때는 s=None."""

    def __init__(self, mod):
        self.s = None
        self.L = None
        mod.register_forward_pre_hook(self)

    def __call__(self, mod, args):
        if self.s is None:
            return None
        h = args[0]
        return (h[:, self.s: self.s + self.L],) + tuple(args[1:])


def word_ids(R, prompt_ids, prompt, sep, word):
    """run_cache.Runner.split_ids와 같은 규칙. 프롬프트 id는 미리 받아서 다시 토큰화하지 않는다."""
    sp = not getattr(R, "chat", False)
    f = R.enc(prompt + sep + word, special=sp)
    n = len(prompt_ids)
    if f[:n] == prompt_ids and len(f) > n:
        return f[n:], True
    return R.enc(sep + word, special=False), False


@torch.no_grad()
def score_batch(R, slicer, p, ws, pad_id):
    """프롬프트 p 뒤에 단어 ws[i]를 붙인 것들을 한 배치로. 반환: (전체 lp, 첫 토큰 lp) 리스트."""
    n = len(p)
    L = max(len(w) for w in ws)
    x = torch.full((len(ws), n + L), pad_id, dtype=torch.long)
    for i, w in enumerate(ws):
        x[i, :n] = torch.tensor(p)
        x[i, n:n + len(w)] = torch.tensor(w)
    slicer.s, slicer.L = n - 1, L
    try:
        out = R.model(input_ids=x.to(R.device), use_cache=False)
    finally:
        slicer.s = None
    lp = torch.log_softmax(out.logits.float(), -1)  # [B, L, V]: 위치 j = 단어 j번째 토큰 예측
    tot, first = [], []
    for i, w in enumerate(ws):
        t = torch.tensor(w, device=lp.device)
        per = lp[i, : len(w)].gather(1, t[:, None])[:, 0]
        tot.append(per.sum().item())
        first.append(per[0].item())
    return tot, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--data", default="../dataset_A/A_final_v2.csv")
    ap.add_argument("--pool", default="placebo_pool.csv")
    ap.add_argument("--out", required=True, help="출력 접두어, 예: placebo/llama")
    ap.add_argument("--conds", default="input,ctrl1,ctrl2", help="en을 더하면 영어 입력 조건도 잼")
    ap.add_argument("--templates", default="T1,T2,T3,T4")
    ap.add_argument("--batch", type=int, default=64, help="한 번에 붙이는 후보 단어 수. 메모리 부족이면 자동으로 반씩 줄임")
    ap.add_argument("--no-greedy", action="store_true", help="입력 조건 greedy 생성(정답 여부) 생략")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-mem", type=float, default=-1, help="run_cache.py와 같음. -1 자동")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="디버그: 문항 수 제한")
    a = ap.parse_args()
    print(f"torch {torch.__version__} | device={a.device}", flush=True)
    if a.device == "cpu" and not a.allow_cpu:
        sys.exit("GPU가 안 잡힘. CPU로 돌리려면 --allow-cpu")

    df = pd.read_csv(a.data)
    if a.limit:
        df = df.head(a.limit)
    pool = pd.read_csv(a.pool)
    words = list(pool.word)
    miss = sorted(set(df.zh_distractor) - set(words))
    assert not miss, f"방해어가 후보 목록에 없음: {miss}. make_placebo_pool.py를 같은 데이터로 다시 돌릴 것"

    P = prompts.build(df, templates=a.templates.split(","), with_repeat=False)
    P = P[P.cond.isin(a.conds.split(","))].reset_index(drop=True)

    gm = a.gpu_mem
    if gm < 0:
        gm = 0.0
        if a.device.startswith("cuda"):
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            print(f"GPU: {torch.cuda.get_device_name(0)} {total:.1f}GiB", flush=True)
            if total < 16:
                gm = max(1.0, total - 2.0)  # run_cache.py보다 0.5GiB 더 비워 둠(배치 활성값)
    R = rc.Runner(a.model, a.device, getattr(torch, a.dtype), gm, a.revision)
    R.chat = a.chat
    if a.chat:
        P = prompts.to_chat(P, R.tok)
    slicer = Slicer(R.model.get_output_embeddings())
    pad_id = R.tok.pad_token_id if R.tok.pad_token_id is not None else R.tok.eos_token_id

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"프롬프트 {len(P)}개 × 후보 {len(words)}개, 배치 {a.batch}", flush=True)
    print("프롬프트 예시:\n" + P.prompt.iloc[0] + "⟨단어⟩", flush=True)

    t0 = time.time()
    bs = a.batch
    recs, meta = [], []
    for i, r in P.iterrows():
        sp = not a.chat
        p = R.enc(r.prompt, special=sp)
        ids = [word_ids(R, p, r.prompt, r.sep, w) for w in words]
        lps, lp1s = [], []
        j = 0
        while j < len(words):
            chunk = [w for w, _ in ids[j: j + bs]]
            try:
                t, f = score_batch(R, slicer, p, chunk, pad_id)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                print(f"메모리 부족 → 배치 {bs}", flush=True)
                continue
            lps += t
            lp1s += f
            j += len(chunk)
        for w, (wi, joint), lp, lp1 in zip(words, ids, lps, lp1s):
            recs.append((r.pid, w, lp, lp1, len(wi), joint))
        m = dict(pid=r.pid, item=r["item"], en=r.en, template=r.template, cond=r.cond, word=r.word,
                 answer=r.answer, distractor=r.distractor, n_prompt_tok=len(p))
        if r.cond == "input" and not a.no_greedy:
            g = R.greedy(p)
            m.update(gen=g, correct=rc.is_correct(g, r.answer))
        meta.append(m)
        if i % 50 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(P)}  {el:.0f}s  (남은 시간 약 {el/(i+1)*(len(P)-i-1)/60:.0f}분)", flush=True)

    M = pd.DataFrame(meta)
    S = pd.DataFrame(recs, columns=["pid", "zh", "lp", "lp_tok1", "ntok", "joint"])
    M.to_csv(a.out + "_prompts.csv", index=False, encoding="utf-8")
    S.to_csv(a.out + "_scores.csv.gz", index=False, encoding="utf-8", compression="gzip")

    # sanity: 처음 몇 프롬프트의 방해어·정답을 run_cache.py 방식(한 개씩, 패딩 없음)으로 다시 재서 비교
    chk = []
    for _, r in P.head(6).iterrows():
        p = R.enc(r.prompt, special=not a.chat)
        for w in [r.distractor, r.answer]:
            wi, _ = word_ids(R, p, r.prompt, r.sep, w)
            ref, _, _, _ = R.score(p, wi)
            got = S[(S.pid == r.pid) & (S.zh == w)].lp.iloc[0]
            chk.append(dict(pid=int(r.pid), zh=w, batch=float(got), single=float(ref), diff=float(abs(got - ref))))
    md = max(c["diff"] for c in chk)
    json.dump(dict(max_abs_diff=md, ok=bool(md < 0.1), rows=chk, model=a.model, chat=a.chat,
                   minutes=(time.time() - t0) / 60), open(a.out + "_check.json", "w"), ensure_ascii=False, indent=1)
    print(f"sanity 배치 vs 한 개씩 최대 차이 {md:.4f} ({'OK' if md < 0.1 else '확인 필요'})")
    print(f"저장: {a.out}_prompts.csv, {a.out}_scores.csv.gz  {(time.time()-t0)/60:.0f}분")


if __name__ == "__main__":
    main()
