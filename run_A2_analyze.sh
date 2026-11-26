#!/bin/bash
#SBATCH -J lensA2an
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH -w moana-u1
#SBATCH -t 0-3
#SBATCH -o /data/alsjiwon/logitlens/exp_A/logs/%x_%j.out
# 2차: 분석만 다시 (캐시는 run_A2.sh에서 이미 만들어짐)
set -e
source /data/alsjiwon/env.sh
export HF_HUB_OFFLINE=1
EXP=/data/alsjiwon/logitlens/exp_A
DATA=/data/alsjiwon/logitlens/dataset_A/A_final_v2.csv
cd $EXP
C="$EXP/cache/llama.pt $EXP/cache/exaone_chat.pt"
python $EXP/analyze.py $C --data $DATA --out $EXP/results2 --affine-langs ko,zh,en
python $EXP/analyze.py $C --data $DATA --out $EXP/results2_affine_kozh --affine-langs ko,zh
python $EXP/analyze.py $C --data $DATA --out $EXP/results2_min2 --affine-langs ko,zh,en --min-correct 2
