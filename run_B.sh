#!/bin/bash
#SBATCH -J prj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH -w moana-u1
#SBATCH -t 0-4
#SBATCH -o /data/alsjiwon/logitlens/exp_A/logs/%x_%j.out
# 데이터셋 B('배' 음성 대조): 문항 캐시 → 분석. 위키 부분(μ, affine·tuned 학습)은 A 캐시의 것을 그대로 씀
# 먼저 로그인 노드에서: cd ../dataset_B && python make_B_v1.py B_final.csv B_final_v1.csv
set -e
source /data/alsjiwon/env.sh
export HF_HUB_OFFLINE=1
EXP=/data/alsjiwon/logitlens/exp_A
DATA=/data/alsjiwon/logitlens/dataset_B/B_final_v1.csv
EXAONE_REV=e949c91dec92095908d34e6b560af77dd0c993f8
cd $EXP
[ -f $DATA ] || { echo "$DATA 없음. dataset_B에서 make_B_v1.py를 먼저 돌릴 것"; exit 1; }
 
[ -f $EXP/cache/llama_B.pt ] || python $EXP/run_cache.py --model meta-llama/Llama-3.2-3B --data $DATA --out $EXP/cache/llama_B.pt
[ -f $EXP/cache/exaone_chat_B.pt ] || python $EXP/run_cache.py --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct --revision $EXAONE_REV \
    --chat --data $DATA --out $EXP/cache/exaone_chat_B.pt
 
python $EXP/analyze.py $EXP/cache/llama_B.pt $EXP/cache/exaone_chat_B.pt --data $DATA --out $EXP/resultsB \
    --wiki-from $EXP/cache/llama.pt,$EXP/cache/exaone_chat.pt --affine-langs ko,zh,en --boot-group ko_input
 