#!/usr/bin/env bash
# 로짓 렌즈 실험 전체 실행 (GPU 서버)
#   bash run_experiment.sh                       # 기본 모델 llama3.2-3b
#   MODEL=exaone3.5-2.4b bash run_experiment.sh
#   LIMIT=5 bash run_experiment.sh               # 데이터셋마다 5문항만 (빠른 점검)
#   FIT_ARGS="--layer-chunk 8" bash run_experiment.sh   # 렌즈 적합 옵션 추가
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-llama3.2-3b}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-}"
FIT_ARGS="${FIT_ARGS:-}"   # fit_lenses에 더 넘길 옵션 (예: "--layer-chunk 8 --tuned-epochs 50")

python -c "import torch, transformers, datasets, scipy, matplotlib" 2>/dev/null || pip install -r requirements.txt

# 1. 한·중 위키 문장 (한 번만)
[ -f data/wiki_ko.jsonl ] && [ -f data/wiki_zh.jsonl ] || python -m experiment.prep_corpus --n 1000

# 2. 평균 제거·아핀·tuned 렌즈 적합 (모델마다 한 번)
#    lenses.pt가 이미 있는데 tuned가 없으면(이전 실행) tuned만 학습해 더한다
LENS_FILE="results/$MODEL/lenses.pt"
if [ ! -f "$LENS_FILE" ]; then
  python -m experiment.fit_lenses --model "$MODEL" --device "$DEVICE" $FIT_ARGS
elif ! python -c "import sys, torch; sys.exit(0 if 'tuned_A' in torch.load('$LENS_FILE', map_location='cpu') else 1)"; then
  python -m experiment.fit_lenses --model "$MODEL" --device "$DEVICE" --only-tuned $FIT_ARGS
fi

# 3. 문항 실행 (A/B/C 중 파일이 있는 것만)
python -m experiment.run_items --model "$MODEL" --device "$DEVICE" ${LIMIT:+--limit "$LIMIT"}

# 4. 분석
python -m experiment.analyze --model "$MODEL"
python -m experiment.analyze --model "$MODEL" --layer-range 0.75 0.97 --tag late
echo "결과: results/$MODEL/summary.md"
