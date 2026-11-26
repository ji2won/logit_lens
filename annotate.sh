#!/usr/bin/bash
#SBATCH -J annotA
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=29G
#SBATCH -p batch_ugrad_advisor_x
#SBATCH -A ugrad_advisor_x
#SBATCH -w moana-u1
#SBATCH -t 0-4
#SBATCH -o /data/alsjiwon/logitlens/annotate.out
#SBATCH -e /data/alsjiwon/logitlens/annotate.err

echo "Job started at: $(date) on $(hostname)"
cd /data/alsjiwon/logitlens
source /data/alsjiwon/miniconda3/etc/profile.d/conda.sh
conda activate logitlens
export STDICT_API_KEY=$(cat /data/alsjiwon/.stdict_key)

python annotate_A.py A_candidates_draft.csv \
  -o A_candidates_annotated.csv --etym-api stdict

echo "Job finished at: $(date)"