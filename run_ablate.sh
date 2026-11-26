#!/bin/bash
#SBATCH -J lensAblate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH -w moana-u1
#SBATCH -t 1-0
#SBATCH -o /data/alsjiwon/logitlens/exp_A/logs/%x_%j.out
# 인과 개입: 중간 층에서 영어 다의어 방향(spring 등)을 빼고 정답·방해어 log p 변화를 봄.
# 대조 = 빈도 맞춘 무관 영어 단어 방향 K=8개 (같은 크기로 빼는 것 포함). 문항 안 비교.
# 먼저 문항 2개로 시험(메모리·정합성) → 통과하면 전체. Llama 약 1–2시간 예상(6GB GPU + CPU offload).
# 사용: cd /data/alsjiwon/logitlens/exp_A && mkdir -p logs ablate && sbatch run_ablate.sh
#       EXAONE도 돌리려면: sbatch --export=ALL,WITH_EXAONE=1 run_ablate.sh
set -e
source /data/alsjiwon/env.sh
export HF_HUB_OFFLINE=1
EXP=/data/alsjiwon/logitlens/exp_A
DA=/data/alsjiwon/logitlens/dataset_A/A_final_v2.csv
DB=/data/alsjiwon/logitlens/dataset_B/B_final.csv
DC=/data/alsjiwon/logitlens/dataset_C/C_final.csv
EXAONE_REV=e949c91dec92095908d34e6b560af77dd0c993f8
CHUNK=${CHUNK:-72}   # 메모리 부족(OutOfMemoryError)이면 sbatch --export=ALL,CHUNK=38 처럼 줄일 것 (최소 19)
cd $EXP
mkdir -p $EXP/ablate $EXP/logs
nvidia-smi --query-gpu=name,memory.total --format=csv || true
COMMON="--data $DA --ctrl-src $DC --exclude-en $DB --chunk $CHUNK"
 
# 1) 시험 실행 (문항 2개)
python $EXP/run_ablate.py --model meta-llama/Llama-3.2-3B $COMMON --limit 2 --out $EXP/ablate/smoke_llama
python - <<PY
import json; c = json.load(open("$EXP/ablate/smoke_llama_check.json"))
print("시험 실행 정합성:", {k: v for k, v in c.items() if k != "clean_vs_single"})
assert c["ok"], "정합성 검사 실패 → 전체 실행 중단. check.json을 올려 주세요"
PY
 
# 2) 전체 (Llama, 주 모델)
[ -f $EXP/ablate/llama_rows.csv.gz ] || python $EXP/run_ablate.py --model meta-llama/Llama-3.2-3B $COMMON --out $EXP/ablate/llama
python $EXP/analyze_ablate.py $EXP/ablate/llama --out $EXP/results_ablate/llama --items $EXP/results_placebo/llama/items_placebo.csv || true
 
# 3) (선택) EXAONE, 대화 형식
if [ -n "$WITH_EXAONE" ]; then
  [ -f $EXP/ablate/exaone_chat_rows.csv.gz ] || python $EXP/run_ablate.py --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct \
      --revision $EXAONE_REV --chat $COMMON \
      --configs W:8-22:all,W:8-22:last,L11:11:all,L17:17:all --read-layer 23 --out $EXP/ablate/exaone_chat
  python $EXP/analyze_ablate.py $EXP/ablate/exaone_chat --out $EXP/results_ablate/exaone_chat \
      --items $EXP/results_placebo/exaone_chat/items_placebo.csv || true
fi
 
tar czf $EXP/results_ablate.tgz --exclude='smoke_*' -C $EXP results_ablate ablate
echo "완료: $EXP/results_ablate.tgz 를 올려 주세요"
 