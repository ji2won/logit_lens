#!/usr/bin/env python3
"""일부 데이터셋·과제만 따로 돌린 결과(run_items --out-dir)를 모델의 기본 결과에 합친다.

  python -m experiment.merge_runs --model llama3.2-3b --run ctx_run
  → results/<model>/outputs.csv, lens.csv.gz를 갱신
    (따로 돌린 결과에 있는 (dataset, task) 조합은 기존 행을 지우고 새 행으로 바꾼다)

probes.csv는 단어별로 합친다. 기존 파일은 덮어쓰므로 먼저 백업해 둔다(run_ctx_mac.sh가 pre_ctx_backup/에 둠).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import RESULTS_DIR


def merge(base: Path, run: Path):
    for name in ("outputs.csv", "lens.csv.gz"):
        old, new = pd.read_csv(base / name), pd.read_csv(run / name)
        combos = set(map(tuple, new[["dataset", "task"]].drop_duplicates().values))
        keep = ~pd.Series(list(zip(old.dataset, old.task)), index=old.index).isin(combos)
        out = pd.concat([old[keep], new], ignore_index=True)
        out.to_csv(base / name, index=False)
        print(f"[{name}] 기존 {len(old)} - 교체 {int((~keep).sum())} + 새 {len(new)} = {len(out)}  {sorted(combos)}")
    p_old, p_new = pd.read_csv(base / "probes.csv"), pd.read_csv(run / "probes.csv")
    pd.concat([p_old, p_new]).drop_duplicates("word", keep="last").sort_values("word").to_csv(base / "probes.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", required=True, help="results/<model>/ 아래 따로 돌린 결과 폴더 이름")
    args = ap.parse_args()
    base = RESULTS_DIR / args.model
    merge(base, base / args.run)


if __name__ == "__main__":
    main()
