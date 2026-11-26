#!/bin/bash
#SBATCH -J lensPlacebo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH -w moana-u1
#SBATCH -t 1-0
#SBATCH -o /data/alsjiwon/logitlens/exp_A/logs/%x_%j.out
# 무관 중국어 단어 대조 실험: "입력어 때문에 아무 상관 없는 중국어 단어도 같이 오르나?"
# 데이터셋 A 프롬프트(T1–T4 × 입력/대조어 2개)에서 후보 377개 단어의 전체 확률을 재고, 방해어와 같은 I를 계산.
# GPU 6GB(RTX A2000)면 run_cache.py처럼 일부 층을 CPU로 내려서 돌림(자동). 모델당 1–3시간 예상.
# 사용: cd /data/alsjiwon/logitlens/exp_A && mkdir -p logs placebo && sbatch run_placebo.sh
set -e
source /data/alsjiwon/env.sh
export HF_HUB_OFFLINE=1
EXP=/data/alsjiwon/logitlens/exp_A
DA=/data/alsjiwon/logitlens/dataset_A/A_final_v2.csv
DC=/data/alsjiwon/logitlens/dataset_C/C_final.csv
EXAONE_REV=e949c91dec92095908d34e6b560af77dd0c993f8
cd $EXP
mkdir -p $EXP/placebo $EXP/logs
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# 후보 단어 목록 (이미 올렸으면 그대로 씀)
[ -f $EXP/placebo_pool.csv ] || python $EXP/make_placebo_pool.py --a $DA --c $DC --out $EXP/placebo_pool.csv

[ -f $EXP/placebo/llama_scores.csv.gz ] || python $EXP/run_placebo.py --model meta-llama/Llama-3.2-3B \
    --data $DA --pool $EXP/placebo_pool.csv --out $EXP/placebo/llama
[ -f $EXP/placebo/exaone_chat_scores.csv.gz ] || python $EXP/run_placebo.py --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct \
    --revision $EXAONE_REV --chat --data $DA --pool $EXP/placebo_pool.csv --out $EXP/placebo/exaone_chat

python $EXP/analyze_placebo.py $EXP/placebo/llama $EXP/placebo/exaone_chat \
    --data $DA --c $DC --pool $EXP/placebo_pool.csv --out $EXP/results_placebo
tar czf $EXP/results_placebo.tgz -C $EXP results_placebo placebo/llama_prompts.csv placebo/exaone_chat_prompts.csv \
    placebo/llama_check.json placebo/exaone_chat_check.json placebo/llama_scores.csv.gz placebo/exaone_chat_scores.csv.gz
echo "완료: $EXP/results_placebo.tgz 를 올려 주세요"
