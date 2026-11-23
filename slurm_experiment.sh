#!/usr/bin/bash
# 세라프(Slurm)에서 로짓 렌즈 실험 돌리기
#   마스터 노드에서, 쓸 conda 환경을 activate한 뒤 제출 (sbatch가 그 PATH를 물려받음):
#     conda activate logit
#     mkdir -p logs
#     sbatch -p <파티션> -w <노드> slurm_experiment.sh
#     MODEL=exaone3.5-2.4b sbatch -p <파티션> -w <노드> slurm_experiment.sh
#     LIMIT=5 MODEL=qwen3-1.7b sbatch -p <파티션> -w <노드> slurm_experiment.sh   # 빠른 점검
#SBATCH -J logit-lens
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# 모델 캐시는 /home이 아니라 /data에 둔다
export HF_HOME="${HF_HOME:-/data/$USER/hf_cache}"
mkdir -p "$HF_HOME"

echo "host=$(hostname)  python=$(which python)  model=${MODEL:-llama3.2-3b}"
nvidia-smi --query-gpu=name,memory.total --format=csv

bash run_experiment.sh
exit 0
