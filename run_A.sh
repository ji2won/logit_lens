#!/usr/bin/bash
#SBATCH -J prj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH -w moana-u1
#SBATCH -t 0-4
#SBATCH -o /data/alsjiwon/logitlens/exp_A/logs/runA.out
#SBATCH -e /data/alsjiwon/logitlens/exp_A/logs/runA.err

echo "Job started at: $(date) on $(hostname)"
cd /data/alsjiwon/logitlens
source /data/alsjiwon/miniconda3/etc/profile.d/conda.sh
conda activate logitlens
set -e
source /data/alsjiwon/env.sh   # setup_env.sh가 만든 내 폴더 안 환경 (HF 토큰도 여기 캐시 위치에서 읽음)
# 계산 노드는 인터넷이 불안정함. 두 모델은 이미 캐시에 받아져 있으니 오프라인으로 읽음
# (모델을 새로 받아야 하면 이 줄을 지우거나, 로그인 노드에서 먼저 받아 둘 것)
export HF_HUB_OFFLINE=1
EXP=/data/alsjiwon/logitlens/exp_A
DATA=/data/alsjiwon/logitlens/dataset_A/A_final_v2.csv
cd $EXP
# EXAONE 저장소의 최신 코드는 transformers 5용이라, transformers 4.46에 맞는 2024-12-11 커밋으로 고정
EXAONE_REV=e949c91dec92095908d34e6b560af77dd0c993f8
mkdir -p $EXP/cache $EXP/results
[ -f $EXP/wiki_ko_zh.txt ] || python $EXP/fetch_wiki.py --n 1000 --data $DATA --out $EXP/wiki_ko_zh.txt
[ -f $EXP/cache/llama.pt ]  || python $EXP/run_cache.py --model meta-llama/Llama-3.2-3B --data $DATA --wiki $EXP/wiki_ko_zh.txt --out $EXP/cache/llama.pt
[ -f $EXP/cache/exaone.pt ] || python $EXP/run_cache.py --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct --revision $EXAONE_REV --data $DATA --wiki $EXP/wiki_ko_zh.txt --out $EXP/cache/exaone.pt
python $EXP/analyze.py $EXP/cache/llama.pt $EXP/cache/exaone.pt --data $DATA --out $EXP/results


echo "Job finished at: $(date)"


