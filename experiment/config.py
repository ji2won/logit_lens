"""실험 전역 설정."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"          # 위키 문장 등 (prep_corpus.py가 만든다)
RESULTS_DIR = ROOT / "results"    # 모델별 결과

# 짧은 이름 → HF 모델 id
MODELS = {
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",
    "exaone3.5-2.4b": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}

# 데이터셋 파일과 각 데이터셋에서 돌릴 과제
DATASETS = {
    "A": ROOT / "A_final.csv",
    "B": ROOT / "B_final.csv",
    "C": ROOT / "C_final.csv",
}
TASKS = {
    "A": ["translate", "translate_en"],   # translate_en = 양성 대조(영어 중간 단계를 프롬프트에 넣음)
    "B": ["translate"],                   # 음성 대조
    "C": ["translate", "repeat"],         # 무관한 영어가 오르는지 + 반복 과제
}
VERSIONS = ["v1", "v2", "v3", "v4"]       # 프롬프트 4가지
LENSES = ["vanilla", "mean", "affine"]

# 렌즈 점수를 취하는 중간층 구간(전체 층 수 대비 비율, 양 끝 포함)
MID_LAYER_RANGE = (0.25, 0.75)

# 판정 기준
RELIABILITY_MIN = 0.7          # 프롬프트 4개에 걸친 Cronbach's alpha
BIG_REDUCTION = 0.5            # 보정 후 영어 확률이 바닐라 대비 50% 이상 줄면 "크게 줄었음"
N_BOOT = 2000
SEED = 0
