# GPU 서버(MOANA) 환경 설정: 캐시·임시 파일을 전부 /data로 보낸다.
# 노드 로컬 디스크(/, /home, /tmp)는 꽉 차 있을 수 있으니 쓰지 않는다.
#   srun으로 들어간 뒤:  source server_env.sh
#   sbatch는 slurm_experiment.sh가 알아서 불러온다.
D="/data/$USER"
export TMPDIR="$D/tmp"
export XDG_CACHE_HOME="$D/cache"          # pip, triton, matplotlib 등 ~/.cache 대신
export PIP_CACHE_DIR="$D/cache/pip"
export HF_HOME="$D/hf_cache"              # 모델·데이터셋 다운로드
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME"

# conda 환경 켜기 (없으면 건너뜀)
if [ -f "$D/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$D/anaconda3/etc/profile.d/conda.sh"
  conda activate logit 2>/dev/null || echo "[server_env] conda 환경 logit 없음"
fi
unset D
