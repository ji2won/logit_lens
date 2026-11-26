"""캐시에서 번역 정답 여부 확인: 템플릿별 정답률과 틀린 출력 예시.
    python inspect_gen.py cache/llama.pt [translate|repeat]
"""
import sys
import torch
import pandas as pd

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from run_cache import is_correct  # noqa: E402

C = torch.load(sys.argv[1], weights_only=False, map_location="cpu")
task = sys.argv[2] if len(sys.argv) > 2 else "translate"   # translate 또는 repeat
m = C["meta"]
t = m[(m.task == task) & (m.cond == "input")].copy()
t["correct"] = t.correct.astype(bool)
t["correct_new"] = [is_correct(g, a) for g, a in zip(t.gen, t.answer)]
print("새 채점 기준(앞부분 일치 허용)으로 다시 매긴 정답률:", t.groupby("template").correct_new.mean().round(3).to_dict())
ok2 = t.groupby("item").correct_new.sum()
print("  문항별 맞힌 템플릿 수:", ok2.value_counts().sort_index().to_dict())
print("템플릿별 정답률:", t.groupby("template").correct.mean().round(3).to_dict())
ok = t.groupby("item").correct.sum()
print("문항별 맞힌 템플릿 수 분포:", ok.value_counts().sort_index().to_dict())
pd.set_option("display.width", 200, "display.max_colwidth", 30)
w = t[~t.correct][["template", "word", "answer", "gen"]]
w = w.assign(gen=w.gen.str.replace("\n", "⏎"))
print(f"\n틀린 출력 {len(w)}개 중 60개:")
print(w.head(60).to_string(index=False))
