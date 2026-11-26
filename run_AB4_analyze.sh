#!/bin/bash
#SBATCH -J AB4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH -w moana-u1
#SBATCH -t 0-24
#SBATCH -o /data/alsjiwon/logitlens/exp_A/logs/%x_%j.out
# 4차 분석 (캐시 재사용): 단어 단위 예측 1 표, 어원별 표, 정답 기준 2/4로 문항 늘리기 — A와 B
set -e
source /data/alsjiwon/env.sh
export HF_HUB_OFFLINE=1
EXP=/data/alsjiwon/logitlens/exp_A
cd $EXP
grep -q "word_level" $EXP/analyze.py || { echo "analyze.py가 옛 버전"; exit 1; }
A="$EXP/cache/llama.pt $EXP/cache/exaone_chat.pt"
B="$EXP/cache/llama_B.pt $EXP/cache/exaone_chat_B.pt"
DA=/data/alsjiwon/logitlens/dataset_A/A_final_v2.csv
DB=/data/alsjiwon/logitlens/dataset_B/B_final_v1.csv
WB="--wiki-from $EXP/cache/llama.pt,$EXP/cache/exaone_chat.pt --boot-group ko_input"
python $EXP/analyze.py $A --data $DA --out $EXP/results4 --affine-langs ko,zh,en
python $EXP/analyze.py $A --data $DA --out $EXP/results4_min2 --affine-langs ko,zh,en --min-correct 2
python $EXP/analyze.py $B --data $DB --out $EXP/resultsB_min2 --affine-langs ko,zh,en --min-correct 2 $WB
 