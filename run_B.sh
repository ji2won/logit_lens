#!/usr/bin/env bash
# 묶음 B 선별 파이프라인 (GPU 서버 등 어디서든 실행 가능, GPU 불필요)
#   bash run_B.sh                 # 기본
#   STDICT_API_KEY=... ETYM=stdict bash run_B.sh                       # 사전 동형어 수까지
#   TOKENIZERS=meta-llama/Llama-3.2-3B,LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct bash run_B.sh   # 토큰 수까지
#   NIKL="/path/NXLS*.json /path/SXLS*.json" bash run_B.sh   # 뜻 빈도 비율(국립국어원 어휘 의미 분석 말뭉치)
#   DOM_RANGE="0.1 0.9" ...                                    # 뜻 비율이 극단인 단어 제외(기본은 열만 붙임)
set -euo pipefail
cd "$(dirname "$0")"

CEDICT="${CEDICT:-../cedict_ts.u8}"
ETYM="${ETYM:-none}"
TOKENIZERS="${TOKENIZERS:-}"
NIKL="${NIKL:-}"
DOM_RANGE="${DOM_RANGE:-}"

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
# shellcheck disable=SC2086
[ -n "$NIKL" ] && python nikl_sense_freq.py $NIKL
fargs=()
[ -n "$DOM_RANGE" ] && fargs+=(--dom-range $DOM_RANGE)
python filter_B.py ${fargs[@]+"${fargs[@]}"}
