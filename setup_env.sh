#!/bin/bash
# 공용 서버용 환경 설치: 모든 것을 내 폴더(/data/alsjiwon) 안에만 둔다. sudo 불필요, 다른 프로젝트에서도 재사용.
#   - conda 본체:    이미 있는 /data/alsjiwon/miniconda3 사용 (없을 때만 Miniforge를 내 폴더에 설치)
#   - 파이썬 환경:   $ROOT/envs/lens   (python 3.11. 기존 logitlens 환경은 건드리지 않음)
#   - 캐시:          $ROOT/.cache/{pip,conda_pkgs,huggingface,tmp}  (홈 ~/.cache 대신)
# 사용:  bash setup_env.sh               (한 번만)
#        source /data/alsjiwon/env.sh    (접속할 때마다, sbatch 안에서도)
set -e
ROOT=/data/alsjiwon
ENV=$ROOT/envs/lens
mkdir -p $ROOT/envs $ROOT/.cache/pip $ROOT/.cache/huggingface $ROOT/.cache/tmp $ROOT/.cache/conda_pkgs
export PIP_CACHE_DIR=$ROOT/.cache/pip TMPDIR=$ROOT/.cache/tmp CONDA_PKGS_DIRS=$ROOT/.cache/conda_pkgs

# 1) conda 찾기 (PATH에 없어도 내 폴더의 설치본을 직접 씀)
CONDA=""
for c in $ROOT/miniconda3 $ROOT/anaconda3 $ROOT/miniforge3; do
  [ -f $c/etc/profile.d/conda.sh ] && { CONDA=$c; break; }
done
if [ -z "$CONDA" ]; then
  CONDA=$ROOT/miniforge3
  curl -fsSL -o $ROOT/.cache/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash $ROOT/.cache/miniforge.sh -b -p $CONDA   # -b: ~/.bashrc 수정 안 함
  rm -f $ROOT/.cache/miniforge.sh
fi
echo "conda: $CONDA"
source $CONDA/etc/profile.d/conda.sh

# 2) 환경 만들기. 앞서 실패한 venv 잔해(pip 없는 반쪽 폴더)가 있으면 지움
if [ -d $ENV ] && [ ! -x $ENV/bin/pip ]; then rm -rf $ENV; fi
# conda-forge 채널만 씀: Anaconda 기본 채널(repo.anaconda.com)의 이용약관 동의가 필요 없음
[ -x $ENV/bin/python ] || conda create -y --prefix $ENV --override-channels -c conda-forge python=3.11

# 나중에 pip install을 해도 torch·transformers 버전이 바뀌지 않게 고정 파일을 둠
cat > $ENV/pip-constraints.txt <<EOF
torch==2.5.1+cu121
transformers==4.46.3
EOF

cat > $ROOT/env.sh <<EOF
export PIP_CONSTRAINT=$ENV/pip-constraints.txt
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121
export PIP_CACHE_DIR=$ROOT/.cache/pip
export CONDA_PKGS_DIRS=$ROOT/.cache/conda_pkgs
export HF_HOME=$ROOT/.cache/huggingface
export TMPDIR=$ROOT/.cache/tmp
source $CONDA/etc/profile.d/conda.sh
conda activate $ENV
EOF
source $ROOT/env.sh

# 3) 패키지
python -m pip install -U pip
# 서버 GPU 드라이버가 CUDA 12.2라 cu121 빌드 torch 2.5.1을 씀 (기본 PyPI 휠은 더 새 드라이버가 필요)
# EXAONE-3.5 모델 코드(trust_remote_code)가 transformers 5.x와 안 맞아서 4.46.3으로 고정
# 한 명령에 같이 고정해야 다른 패키지가 torch를 최신으로 다시 올리지 않음
pip uninstall -y torch triton $(pip list 2>/dev/null | awk '/^nvidia-/{print $1}') || true
pip install "torch==2.5.1+cu121" "transformers==4.46.3" accelerate datasets pandas scipy "numpy<2.3" tiktoken huggingface_hub \
    --extra-index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| CUDA 사용 가능:", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
echo
echo "완료. 다음: source $ROOT/env.sh && hf auth login   (토큰은 $ROOT/.cache/huggingface에 저장됨)"
