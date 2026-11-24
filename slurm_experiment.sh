#!/usr/bin/bash
# GPU 서버(MOANA, Slurm)에서 로짓 렌즈 실험 돌리기
#   마스터 노드에서 제출 (conda 환경·캐시 경로는 server_env.sh가 잡아줌):
#     mkdir -p logs
#     MODEL=qwen3-1.7b sbatch slurm_experiment.sh
#     MODEL=exaone3.5-2.4b sbatch slurm_experiment.sh
#     LIMIT=5 MODEL=qwen3-1.7b sbatch slurm_experiment.sh   # 빠른 점검
# (-p/-w는 명령줄 sbatch -p ... -w ... 로 바꿀 수 있음)
#SBATCH -J logit-lens
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-y5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# 캐시·임시 파일을 /data로 보내고 conda 환경(logit)을 켠다
set +u; source server_env.sh; set -u

echo "host=$(hostname)  python=$(which python)  model=${MODEL:-llama3.2-3b}"
nvidia-smi --query-gpu=name,memory.total --format=csv

bash run_experiment.sh
exit 0
