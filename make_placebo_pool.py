"""무관 중국어 단어(플라시보) 후보 목록 만들기. GPU 필요 없음.

질문: 입력어(이자)를 넣었을 때 방해어(兴趣)만 오르는가, 아니면 아무 중국어 단어나 같이 오르는가?
→ 방해어 대신 '무관 단어'를 넣고 같은 I를 계산해 비교한다.

후보 = 데이터셋 A의 정답·방해어 전부 + 데이터셋 C의 중국어 단어(zh, zh_unrelated) 전부.
모두 빈도(zipf)가 이미 있는 단어라 빈도 맞추기가 쉽다.
모델은 후보 전부를 모든 프롬프트에서 점수 매기고(run_placebo.py), 문항별로 어떤 후보를 '무관'으로 볼지는
분석 단계(analyze_placebo.py)에서 정한다. 그래서 여기서는 문항과 무관한 목록 하나만 만든다.

    python make_placebo_pool.py --a ../dataset_A/A_final_v2.csv --c ../dataset_C/C_final.csv --out placebo_pool.csv
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="../dataset_A/A_final_v2.csv")
    ap.add_argument("--c", default="../dataset_C/C_final.csv")
    ap.add_argument("--out", default="placebo_pool.csv")
    a = ap.parse_args()

    A = pd.read_csv(a.a)
    rows = []
    for _, r in A.iterrows():
        rows.append(dict(word=r.zh_answer, zipf=r.zipf_zh_answer, src="A_answer", src_id=r.id, src_en=r.en))
        rows.append(dict(word=r.zh_distractor, zipf=r.zipf_zh_distractor, src="A_distractor", src_id=r.id, src_en=r.en))
    if a.c:
        C = pd.read_csv(a.c)
        zipf_of = dict(zip(C.zh, C.zipf_zh))  # zh_unrelated는 다른 문항의 zh라 빈도가 같은 표에 있음
        for _, r in C.iterrows():
            rows.append(dict(word=r.zh, zipf=r.zipf_zh, src="C_zh", src_id=r.id, src_en=r.en))
            rows.append(dict(word=r.zh_unrelated, zipf=zipf_of.get(r.zh_unrelated), src="C_unrelated", src_id=r.id,
                             src_en=r.en_unrelated))
    P = pd.DataFrame(rows).dropna(subset=["word"])
    P["word"] = P.word.str.strip()
    # 같은 단어가 여러 번 나오면 하나로 합치고 출처를 모두 남긴다(제외 규칙에 씀)
    g = P.groupby("word", sort=False)
    out = pd.DataFrame(dict(
        zipf=g.zipf.max(),
        src=g.src.agg(lambda s: ";".join(sorted(set(s)))),
        src_id=g.src_id.agg(lambda s: ";".join(sorted(set(map(str, s))))),
        src_en=g.src_en.agg(lambda s: ";".join(sorted(set(map(str, s))))),
    )).reset_index()
    out["nchar"] = out.word.str.len()
    out.to_csv(a.out, index=False, encoding="utf-8")
    print(f"후보 {len(out)}개 → {a.out}")
    print("글자 수:", out.nchar.value_counts().sort_index().to_dict())
    print("zipf 없음:", int(out.zipf.isna().sum()))


if __name__ == "__main__":
    main()
