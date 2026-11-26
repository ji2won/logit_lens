"""정답·방해어 첫 토큰 겹침 확인 (Llama-3.2, EXAONE-3.5).
 
A_final_tok.csv의 토큰 열은 HF 접속이 막힌 환경에서 만들었다
(Llama: Meta 공식 llama3 tokenizer.model + tiktoken, EXAONE: tokenizer.json의
vocab/merges로 BPE를 직접 재현). 이 스크립트는 실제 HF 토크나이저로 같은 값을
다시 계산해서 표와 다른 곳이 있는지 확인한다.
 
    pip install transformers
    hf auth login            # Llama-3.2는 gated
    python check_first_token.py A_final_tok.csv
 
출력: 형태(bare/sp)별 첫 토큰 겹침 행, 그리고 표의 토큰 id와 다른 행 수.
"""
import json
import sys
 
import pandas as pd
from transformers import AutoTokenizer
 
MODELS = {
    "llama": "meta-llama/Llama-3.2-3B",
    "exaone": "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
}
FORMS = {"bare": "", "sp": " "}  # bare: 프롬프트가 `中文: "`처럼 끝날 때, sp: `中文:` 뒤에 공백+단어
 
path = sys.argv[1] if len(sys.argv) > 1 else "A_final_tok.csv"
d = pd.read_csv(path)
 
for m, name in MODELS.items():
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    for form, pre in FORMS.items():
        ids = {
            role: [tok.encode(pre + w, add_special_tokens=False) for w in d[col]]
            for role, col in [("ans", "zh_answer"), ("dis", "zh_distractor")]
        }
        clash = [a[0] == b[0] for a, b in zip(ids["ans"], ids["dis"])]
        rows = d.loc[clash, ["id", "zh_answer", "zh_distractor"]]
        print(f"[{m} {form}] 첫 토큰 겹침 {sum(clash)}행")
        for _, r in rows.iterrows():
            print("   ", r.id, r.zh_answer, r.zh_distractor)
        # 표와 비교
        col = f"tok1id_{m}_ans_{form}"
        if col in d:
            mism = 0
            for role in ("ans", "dis"):
                mism += sum(int(x) != y[0] for x, y in zip(d[f"tok1id_{m}_{role}_{form}"], ids[role]))
            if m == "exaone":
                for role in ("ans", "dis"):
                    mism += sum(json.loads(x) != y for x, y in zip(d[f"tokids_exaone_{role}_{form}"], ids[role]))
            print(f"    표와 다른 값: {mism}개 (0이어야 정상)")
 