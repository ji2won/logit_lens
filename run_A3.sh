#!/bin/bash
#SBATCH -J lensA3an
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH -w moana-u1
#SBATCH -t 0-4
#SBATCH -o /data/alsjiwon/logitlens/exp_A/logs/%x_%j.out
# 3차: 분석만 다시 (캐시는 그대로). 바뀐 점
#  - tuned lens 추가: 최종 출력과의 KL을 줄이도록 학습한 affine (ridge affine의 '기타 토큰 쏠림' 점검)
#  - lens 충실도 표: KL(최종‖lens), 언어 질량, 질량 가중 E
#  - 중간 층 판독 1위 토큰
set -e
source /data/alsjiwon/env.sh
export HF_HUB_OFFLINE=1
EXP=/data/alsjiwon/logitlens/exp_A
DATA=/data/alsjiwon/logitlens/dataset_A/A_final_v2.csv
cd $EXP
C="$EXP/cache/llama.pt $EXP/cache/exaone_chat.pt"
python $EXP/analyze.py $C --data $DATA --out $EXP/results3 --affine-langs ko,zh,en
python $EXP/analyze.py $C --data $DATA --out $EXP/results3_kozh --affine-langs ko,zh
 