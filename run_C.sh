#!/bin/bash
#SBATCH -J lensC
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH -w moana-u1
#SBATCH -t 0-4
#SBATCH -o /data/alsjiwon/logitlens/exp_A/logs/%x_%j.out
# 데이터셋 C(일반 단어): 가설 ①(평균 성분), 영어 편향 지수 B, 반복 과제. 문항 캐시 → analyze_C.py
# 위키 부분(μ, affine·tuned 학습)은 A 캐시의 것을 그대로 씀
set -e
source /data/alsjiwon/env.sh
export HF_HUB_OFFLINE=1
EXP=/data/alsjiwon/logitlens/exp_A
DATA=/data/alsjiwon/logitlens/dataset_C/C_final.csv
EXAONE_REV=e949c91dec92095908d34e6b560af77dd0c993f8
cd $EXP
[ -f $DATA ] || { echo "$DATA 없음. C_final.csv를 dataset_C/에 올릴 것"; exit 1; }
grep -q "build_C" $EXP/prompts.py || { echo "prompts.py가 옛 버전"; exit 1; }
grep -q "extra_ids" $EXP/run_cache.py || { echo "run_cache.py가 옛 버전"; exit 1; }
 
[ -f $EXP/cache/llama_C.pt ] || python $EXP/run_cache.py --model meta-llama/Llama-3.2-3B --data $DATA --out $EXP/cache/llama_C.pt
[ -f $EXP/cache/exaone_chat_C.pt ] || python $EXP/run_cache.py --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct --revision $EXAONE_REV \
    --chat --data $DATA --out $EXP/cache/exaone_chat_C.pt
 
python $EXP/analyze_C.py $EXP/cache/llama_C.pt $EXP/cache/exaone_chat_C.pt --data $DATA --out $EXP/resultsC \
    --wiki-from $EXP/cache/llama.pt,$EXP/cache/exaone_chat.pt --affine-langs ko,zh,en
 