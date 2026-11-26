#!/bin/bash
#SBATCH -J lensBmin2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH -w moana-u1
#SBATCH -t 1-0
#SBATCH -o /data/alsjiwon/logitlens/exp_A/logs/%x_%j.out
# run_AB4_analyze.sh의 마지막 줄(B, 정답 기준 2/4)만 따로
set -e
source /data/alsjiwon/env.sh
export HF_HUB_OFFLINE=1
EXP=/data/alsjiwon/logitlens/exp_A
cd $EXP
B="$EXP/cache/llama_B.pt $EXP/cache/exaone_chat_B.pt"
DB=/data/alsjiwon/logitlens/dataset_B/B_final_v1.csv
WB="--wiki-from $EXP/cache/llama.pt,$EXP/cache/exaone_chat.pt --boot-group ko_input"
python $EXP/analyze.py $B --data $DB --out $EXP/resultsB_min2 --affine-langs ko,zh,en --min-correct 2 $WB
