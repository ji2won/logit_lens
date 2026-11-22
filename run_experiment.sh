#!/usr/bin/env bash
# 로짓 렌즈 실험 전체 실행 (GPU 서버)
#   bash run_experiment.sh                       # 기본 모델 llama3.2-3b
#   MODEL=exaone3.5-2.4b bash run_experiment.sh
#   LIMIT=5 bash run_experiment.sh               # 데이터셋마다 5문항만 (빠른 점검)
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-llama3.2-3b}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-}"

python -c "import torch, transformers, datasets, scipy, matplotlib" 2>/dev/null || pip install -r requirements.txt

# 1. 한·중 위키 문장 (한 번만)
[ -f data/wiki_ko.jsonl ] && [ -f data/wiki_zh.jsonl ] || python -m experiment.prep_corpus --n 1000

# 2. 평균 제거·아핀 렌즈 적합 (모델마다 한 번)
[ -f "results/$MODEL/lenses.pt" ] || python -m experiment.fit_lenses --model "$MODEL" --device "$DEVICE"

# 3. 문항 실행 (A/B/C 중 파일이 있는 것만)
python -m experiment.run_items --model "$MODEL" --device "$DEVICE" ${LIMIT:+--limit "$LIMIT"}

# 4. 분석
python -m experiment.analyze --model "$MODEL"
echo "결과: results/$MODEL/summary.md"
