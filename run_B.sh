#!/usr/bin/env bash
# 묶음 B 선별 파이프라인 (GPU 서버 등 어디서든 실행 가능, GPU 불필요)
#   bash run_B.sh                 # 기본
#   STDICT_API_KEY=... ETYM=stdict bash run_B.sh                       # 사전 동형어 수까지
#   TOKENIZERS=meta-llama/Llama-3.2-3B,LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct bash run_B.sh   # 토큰 수까지
set -euo pipefail
cd "$(dirname "$0")"

CEDICT="${CEDICT:-../cedict_ts.u8}"
ETYM="${ETYM:-none}"
TOKENIZERS="${TOKENIZERS:-}"

python -c "import wordfreq, pandas, requests" 2>/dev/null || pip install -r requirements.txt

if [ ! -f "$CEDICT" ]; then
  echo "[info] CC-CEDICT 다운로드 → $CEDICT"
  curl -fL -o /tmp/cedict.zip https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.zip
  unzip -p /tmp/cedict.zip cedict_ts.u8 > "$CEDICT"
fi

extra=()
[ -n "$TOKENIZERS" ] && extra+=(--tokenizers "$TOKENIZERS")
python annotate_B.py B_candidates_draft.csv -o B_candidates_annotated.csv --etym-api "$ETYM" ${extra[@]+"${extra[@]}"}
python review_B.py --cedict "$CEDICT"
python filter_B.py
