#!/usr/bin/env bash
# B 예문 조건(translate_ctx)만 Mac(MPS)에서 돌린다. 기존 결과는 pre_ctx_backup/에 보관한다.
#   bash run_ctx_mac.sh                      # 세 모델
#   MODELS="llama3.2-3b" bash run_ctx_mac.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-/opt/anaconda3/envs/logit/bin/python}"
MODELS="${MODELS:-llama3.2-3b qwen3-1.7b exaone3.5-2.4b}"

for M in $MODELS; do
  D="results/$M"
  echo "=== $M $(date)"
  mkdir -p "$D/pre_ctx_backup"
  for f in outputs.csv lens.csv.gz probes.csv generations.csv generation_summary.md; do
    [ -f "$D/pre_ctx_backup/$f" ] || cp "$D/$f" "$D/pre_ctx_backup/$f"
  done
  "$PY" -m experiment.run_items --model "$M" --device mps --datasets B --tasks translate_ctx \
        --out-dir "$D/ctx_run" --lens-file "$D/lenses.pt"
  "$PY" -m experiment.generate --model "$M" --device mps --datasets B --tasks translate_ctx
done
echo "ALL DONE $(date)"
