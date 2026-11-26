#!/bin/bash
#SBATCH -J lensA2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH -w moana-u1
#SBATCH -t 0-4
#SBATCH -o /data/alsjiwon/logitlens/exp_A/logs/%x_%j.out
# 2차 실행 (1차 결과 반영)
#  1) 위키 문장에 영어를 넣어 affine·μ를 다시 계산 (Llama는 문항 재실행 없이 위키 부분만)
#  2) EXAONE은 대화 형식(--chat)으로 문항부터 다시 (기존 cache/exaone.pt는 그대로 둠)
#  3) 분석 세 가지: 기본(affine 3개 언어) / 민감도(affine 한·중만, 1차와 같은 조건) / 정답 기준 2/4
# 먼저 로그인 노드에서(인터넷 필요):
#   source /data/alsjiwon/env.sh && pip install zhconv
#   cd /data/alsjiwon/logitlens/exp_A && python fetch_wiki.py --n 1000 --langs ko,zh,en --data ../dataset_A/A_final_v2.csv --out wiki_ko_zh_en.txt
# 그다음: sbatch run_A2.sh
echo "Job started at: $(date) on $(hostname)"
set -e
source /data/alsjiwon/env.sh
export HF_HUB_OFFLINE=1
EXP=/data/alsjiwon/logitlens/exp_A
DATA=/data/alsjiwon/logitlens/dataset_A/A_final_v2.csv
WIKI=$EXP/wiki_ko_zh_en.txt
EXAONE_REV=e949c91dec92095908d34e6b560af77dd0c993f8
cd $EXP
[ -f $WIKI ] || { echo "$WIKI 없음. 로그인 노드에서 fetch_wiki.py를 먼저 돌릴 것"; exit 1; }
python -c "import zhconv" || { echo "zhconv 없음. 로그인 노드에서 pip install zhconv"; exit 1; }

python $EXP/run_cache.py --model meta-llama/Llama-3.2-3B --data $DATA --wiki $WIKI --wiki-only --out $EXP/cache/llama.pt
[ -f $EXP/cache/exaone_chat.pt ] || python $EXP/run_cache.py --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct --revision $EXAONE_REV \
    --chat --data $DATA --wiki $WIKI --out $EXP/cache/exaone_chat.pt

C="$EXP/cache/llama.pt $EXP/cache/exaone_chat.pt"
python $EXP/analyze.py $C --data $DATA --out $EXP/results2 --affine-langs ko,zh,en
python $EXP/analyze.py $C --data $DATA --out $EXP/results2_affine_kozh --affine-langs ko,zh
python $EXP/analyze.py $C --data $DATA --out $EXP/results2_min2 --affine-langs ko,zh,en --min-correct 2


echo "Job finished at: $(date)"


