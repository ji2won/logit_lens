#!/usr/bin/bash
# tuned lens 추가 후 세 모델 재실행 (MOANA, Slurm 배열 작업: 모델 하나당 작업 하나)
#   마스터 노드에서 제출:
#     mkdir -p logs
#     sbatch slurm_tuned.sh                      # 세 모델 전부 (0=llama, 1=exaone, 2=qwen)
#     sbatch --array=2 slurm_tuned.sh            # qwen만
#     LIMIT=5 sbatch --array=2 slurm_tuned.sh    # 빠른 점검 (데이터셋마다 5문항)
#
# 하는 일 (모델마다):
#   1. results/<모델>/ 의 기존 결과를 pre_tuned_backup/ 으로 복사
#   2. lenses.pt가 있으면 tuned만 학습해 더하고(--only-tuned), 없으면 렌즈 전부 적합
#   3. run_items 전체 재실행 (렌즈 4개, 과제 프롬프트 KL 열 포함)
#   4. analyze 기본(25–75%) + late(75–97%)
#   5. tuned 수렴 요약을 로그 끝에 출력
#
# 노드에 GPU가 하나뿐이면 세 작업이 차례로 돈다. 동시에 돌리지 않으려면 sbatch --array=0-2%1
#SBATCH -J logit-tuned
#SBATCH -p batch_ce_ugrad
#SBATCH -w moana-y5
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -t 1-0
#SBATCH -o logs/tuned-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

MODELS=(llama3.2-3b exaone3.5-2.4b qwen3-1.7b)
export MODEL="${MODEL:-${MODELS[${SLURM_ARRAY_TASK_ID:-0}]}}"
# tuned 학습 벡터를 층 8개씩 모아 CPU 메모리를 아낀다 (전부 모으면 Llama에서 ~14GB)
export FIT_ARGS="${FIT_ARGS:---layer-chunk 8}"

set +u; source server_env.sh; set -u

echo "host=$(hostname)  python=$(which python)  model=$MODEL  fit_args=$FIT_ARGS"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 1. 기존 결과 백업 (하위 실행 폴더·백업 폴더는 빼고 맨 위 파일과 late/만)
R="results/$MODEL"
if [ -d "$R" ] && [ -z "${LIMIT:-}" ]; then
  B="$R/pre_tuned_backup"
  if [ -e "$B" ]; then B="$B-$(date +%Y%m%d-%H%M%S)"; fi
  mkdir -p "$B"
  find "$R" -maxdepth 1 -type f ! -name lenses.pt ! -name lang_masks.pt -exec cp -p {} "$B"/ \;
  [ -d "$R/late" ] && cp -rp "$R/late" "$B"/
  [ -f "$R/lenses.pt" ] && cp -p "$R/lenses.pt" "$B"/lenses_pre_tuned.pt
  echo "[backup] → $B"
fi
if [ ! -f "$R/lenses.pt" ]; then
  echo "[warn] $R/lenses.pt 없음 → 평균·아핀·tuned를 전부 새로 적합합니다 (기존 결과와 아핀 수치가 조금 다를 수 있음)"
fi

# 2–4.
bash run_experiment.sh

# 5. tuned 수렴 요약
python - <<EOF
import pandas as pd
f = pd.read_csv("$R/lens_fit.csv")
v = f[f.lens == "vanilla"].set_index("layer").kl
t = f[f.lens == "tuned"].set_index("layer")
print("\n[tuned 요약] 층별 held-out KL (vanilla → tuned)")
print(pd.DataFrame({"vanilla": v, "tuned": t.kl, "best_epoch": t.get("best_epoch"),
                    "converged": t.get("converged")}).to_string())
if "converged" in t:
    print(f"수렴 안 한 층: {int((~t.converged.astype(bool)).sum())}/{len(t)}  (많으면 FIT_ARGS에 --tuned-epochs 늘려 재실행)")
EOF
echo "결과: $R/summary.md (0절 충실도 관문, 3절 판정)"
exit 0
