#!/usr/bin/env python3
"""문항마다 최종 출력 확률(간섭 계산용)과 층별 렌즈 출력(렌즈 점수 계산용)을 기록한다.

  python -m experiment.run_items --model llama3.2-3b
  → results/<model>/outputs.csv   문항×과제×프롬프트×(입력어|대조어)마다
                                   log p(정답), log p(방해어) (중국어 단어 전체 토큰의 합)
    results/<model>/lens.csv.gz   위 각 프롬프트의 마지막 위치에서, 렌즈×층마다
                                   lp_<probe>: 영어 probe 단어의 렌즈 로그확률
                                   mass_en/ko/zh: 어휘 중 영어/한국어/중국어 토큰 확률 합(log)
                                   top1: 렌즈 출력 1위 토큰
    results/<model>/probes.csv    영어 probe 단어의 토큰화와 1토큰 여부

렌즈는 정답이 시작되기 직전 위치(프롬프트 마지막 토큰)를 읽는다.
대조어 프롬프트도 같은 probe(여름 프롬프트에서 spring)를 읽어 두므로, 분석에서
"입력어 - 대조어" 렌즈 점수도 계산할 수 있다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from . import prompts
from .config import DATASETS, LENSES, MODELS, RESULTS_DIR, TASKS, VERSIONS
from .data import Item, load
from .lenses import language_masks, lens_logprobs, load_lenses, mass, probe_logprob, probe_token_ids
from .lm import LM


def run_all(lm: LM, datasets: dict[str, list[Item]], tasks: dict[str, list[str]], versions, lenses, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_file = out_dir / "lang_masks.pt"
    if mask_file.exists():
        masks = torch.load(mask_file)
    else:
        masks = language_masks(lm)
        torch.save(masks, mask_file)

    # 영어 probe 토큰
    words = sorted({w for items in datasets.values() for it in items for w in it.probes.values()})
    probe_ids, probe_rows = {}, []
    for w in words:
        ids, single = probe_token_ids(lm, w)
        probe_ids[w] = ids
        probe_rows.append(dict(word=w, token_ids=" ".join(map(str, ids)),
                               tokens=" | ".join(repr(lm.decode([i])) for i in ids), single_token=single))
    pd.DataFrame(probe_rows).to_csv(out_dir / "probes.csv", index=False)

    out_rows, lens_rows = [], []
    n_layers = lm.n_layers
    for ds, items in datasets.items():
        prompts.check_fewshot(items)
        for k, it in enumerate(items, 1):
            for task in tasks[ds]:
                for v in versions:
                    roles = [("input", it.ko)]
                    if task != "repeat":
                        roles += [("control", c) for c in it.controls]
                    for role, w in roles:
                        prompt = prompts.build(task, v, w, it.en_hint)
                        key = dict(dataset=ds, id=it.id, group=it.group, etym=it.etym, task=task,
                                   version=v, role=role, word=w)
                        rec, hs = dict(key), None
                        answer = w if task == "repeat" else it.zh_answer
                        distractor = None if task == "repeat" else it.zh_distractor
                        if role == "input":
                            lp, lp1, nt, hs = lm.score(prompt, answer)
                            rec.update(logp_answer=lp, logp_answer_first=lp1, ntok_answer=nt)
                        if distractor:
                            lp, lp1, nt, hs_d = lm.score(prompt, distractor)
                            rec.update(logp_distractor=lp, logp_distractor_first=lp1, ntok_distractor=nt)
                            hs = hs if hs is not None else hs_d
                        out_rows.append(rec)

                        for lens in lenses:
                            lp_all = lens_logprobs(lm, lens, hs)
                            cols = {f"lp_{name}": probe_logprob(lp_all, probe_ids[word]).tolist()
                                    for name, word in it.probes.items()}
                            cols.update({f"mass_{lang}": mass(lp_all, m).tolist() for lang, m in masks.items()})
                            top1 = lp_all.argmax(-1).tolist()
                            for l in range(n_layers + 1):
                                row = dict(key, lens=lens.name, layer=l, top1=lm.decode([top1[l]]))
                                row.update({c: vals[l] for c, vals in cols.items()})
                                lens_rows.append(row)
            if k % 10 == 0 or k == len(items):
                print(f"  [{ds}] {k}/{len(items)}", flush=True)

    pd.DataFrame(out_rows).to_csv(out_dir / "outputs.csv", index=False)
    pd.DataFrame(lens_rows).to_csv(out_dir / "lens.csv.gz", index=False)
    print(f"[done] {len(out_rows)} 프롬프트, 렌즈 행 {len(lens_rows)} → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--versions", nargs="+", default=VERSIONS)
    ap.add_argument("--lenses", nargs="+", default=LENSES)
    ap.add_argument("--limit", type=int, default=None, help="데이터셋마다 앞 N문항만 (빠른 점검용)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir or RESULTS_DIR / args.model.replace("/", "_"))
    datasets = {}
    for ds in args.datasets:
        path = DATASETS[ds]
        if not Path(path).exists():
            print(f"[skip] {ds}: {path} 없음")
            continue
        datasets[ds] = load(ds, path)[:args.limit]
    lm = LM.from_pretrained(MODELS.get(args.model, args.model), device=args.device)
    lenses = load_lenses(args.lenses, out_dir / "lenses.pt", device=args.device)
    run_all(lm, datasets, TASKS, args.versions, lenses, out_dir)


if __name__ == "__main__":
    main()
