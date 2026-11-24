#!/usr/bin/env python3
"""입력어 프롬프트에서 모델이 실제로 생성하는 번역을 greedy로 뽑아 정답·방해어와 비교한다.

  python -m experiment.generate --model llama3.2-3b --device mps
  → results/<model>/generations.csv   문항×과제×프롬프트마다 생성 원문, 잘라낸 답, 판정
    results/<model>/generation_summary.md  데이터셋·과제별 정확도

outputs.csv는 정답·방해어의 확률만 기록하므로, 모델이 실제로 무엇을 내는지(다른 동의어, 형식 이탈 등)는
여기서 본다. 대조어 프롬프트는 생성하지 않는다.

판정 (답 = 생성문에서 첫 줄, v4는 」 앞까지 자른 뒤 앞뒤 공백·구두점 제거)
  exact        답 == 정답
  distractor   답 == 방해어
  contains_*   답에 정답/방해어가 들어 있음 (春天来了 같은 경우)
B는 맥락이 없으므로 두 뜻(정답·방해어) 중 하나면 맞은 것으로 따로 센다(either).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from . import prompts
from .config import DATASETS, MODELS, RESULTS_DIR, VERSIONS
from .data import load
from .lm import LM

TASKS = {"A": ["translate", "translate_en"], "B": ["translate"]}
_STRIP = " \t\r\n。，、．.,:：;；!?！？\"'“”‘’「」『』()（）[]【】"


def cut(text: str, version: str) -> str:
    t = text.split("\n", 1)[0]
    if version == "v4":
        t = t.split("」", 1)[0]
    return t.strip(_STRIP)


@torch.no_grad()
def greedy(lm: LM, prompt: str, max_new: int) -> str:
    ids = torch.tensor([lm.encode(prompt)], device=lm.device)
    out = lm.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                            do_sample=False, pad_token_id=lm.tok.pad_token_id or lm.tok.eos_token_id)
    return lm.tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)


def judge(df: pd.DataFrame, special=()) -> pd.DataFrame:
    """raw에서 답을 잘라 정답·방해어와 비교한다. special: 지울 특수 토큰 문자열(이전 실행 결과 재채점용)."""
    raw = df.raw.fillna("").astype(str)
    for t in special:
        raw = raw.str.replace(t, "\n", regex=False)
    df = df.assign(answer=[cut(r, v) for r, v in zip(raw, df.version)])
    df["exact"] = df.answer == df.zh_answer
    df["distractor"] = df.answer == df.zh_distractor
    df["contains_answer"] = [a in g for a, g in zip(df.zh_answer, df.answer)]
    df["contains_distractor"] = [d in g for d, g in zip(df.zh_distractor, df.answer)]
    return df


def run(lm: LM, datasets, versions, max_new: int, out_dir: Path):
    rows = []
    for ds, items in datasets.items():
        for k, it in enumerate(items, 1):
            for task in TASKS[ds]:
                for v in versions:
                    raw = greedy(lm, prompts.build(task, v, it.ko, it.en_hint), max_new)
                    rows.append(dict(dataset=ds, id=it.id, group=it.group, etym=it.etym, task=task, version=v,
                                     word=it.ko, zh_answer=it.zh_answer, zh_distractor=it.zh_distractor, raw=raw))
            if k % 10 == 0 or k == len(items):
                print(f"  [{ds}] {k}/{len(items)}", flush=True)
    df = judge(pd.DataFrame(rows))
    df.to_csv(out_dir / "generations.csv", index=False)
    return df


def summarize(df: pd.DataFrame, model: str) -> str:
    out = [f"# 생성 정확도 ({model})", "",
           "greedy 생성, 입력어 프롬프트만. 답 = 첫 줄(v4는 」 앞)에서 앞뒤 구두점 제거.", "",
           "| dataset | task | n | 정답 일치 | 방해어 일치 | 정답 포함 | 방해어 포함 | 그 외 |",
           "|---|---|---|---|---|---|---|---|"]
    for (ds, task), g in df.groupby(["dataset", "task"]):
        other = ~(g.contains_answer | g.contains_distractor)
        out.append(f"| {ds} | {task} | {len(g)} | {g.exact.mean():.3f} | {g.distractor.mean():.3f} | "
                   f"{g.contains_answer.mean():.3f} | {g.contains_distractor.mean():.3f} | {other.mean():.3f} |")
    b = df[df.dataset == "B"]
    if len(b):
        out += ["", f"B 두 뜻 중 하나와 일치: {(b.exact | b.distractor).mean():.3f} (n={len(b)})"]
    out += ["", "## 프롬프트 버전별 정답 일치 (A translate)", "", "| version | 정답 일치 | 방해어 일치 |", "|---|---|---|"]
    for v, g in df[(df.dataset == "A") & (df.task == "translate")].groupby("version"):
        out.append(f"| {v} | {g.exact.mean():.3f} | {g.distractor.mean():.3f} |")
    miss = df[(df.dataset == "A") & (df.task == "translate") & ~df.contains_answer]
    out += ["", "## 정답이 안 나온 답 (A translate, 많은 순 상위 30)", "", "| 입력어 | 정답 | 방해어 | 생성 답 | 횟수 |",
            "|---|---|---|---|---|"]
    top = miss.groupby(["word", "zh_answer", "zh_distractor", "answer"]).size().sort_values(ascending=False).head(30)
    for (w, a, d, g), n in top.items():
        out.append(f"| {w} | {a} | {d} | {g or '(빈 답)'} | {n} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(TASKS))
    ap.add_argument("--versions", nargs="+", default=VERSIONS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--rescore", action="store_true", help="생성은 하지 않고 기존 generations.csv를 다시 채점")
    args = ap.parse_args()

    out_dir = Path(args.out_dir or RESULTS_DIR / args.model.replace("/", "_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.rescore:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODELS.get(args.model, args.model), trust_remote_code=True)
        df = judge(pd.read_csv(out_dir / "generations.csv", keep_default_na=False), tok.all_special_tokens)
        df.to_csv(out_dir / "generations.csv", index=False)
        (out_dir / "generation_summary.md").write_text(summarize(df, args.model), "utf-8")
        print(f"[rescore] {len(df)}개 → {out_dir / 'generation_summary.md'}")
        return
    datasets = {ds: load(ds, DATASETS[ds])[:args.limit] for ds in args.datasets if Path(DATASETS[ds]).exists()}
    lm = LM.from_pretrained(MODELS.get(args.model, args.model), device=args.device)
    df = run(lm, datasets, args.versions, args.max_new, out_dir)
    (out_dir / "generation_summary.md").write_text(summarize(df, args.model), "utf-8")
    print(f"[done] {len(df)}개 → {out_dir / 'generation_summary.md'}")


if __name__ == "__main__":
    main()
