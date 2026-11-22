#!/usr/bin/env python3
"""
묶음 B(한국어 동음이의어) 후보 표에 자동 검사·부착을 수행한다.

B 문항 구조: 한국어 입력어 하나(배)에 뜻이 둘(선박/과일). 단어마다 2행을 만들어
한 행은 뜻 1을 정답·뜻 2를 방해어로, 다른 행은 그 반대로 둔다. 프롬프트에는 맥락을 주지 않는다.

  1) 중국어 정답/방해어 글자 공유               → ZH_SHARE (B에서는 제외 사유)
  2) 영어 번역 검사: 두 뜻이 같은 영어(EN_SAME), 여러 단어(EN_MULTIWORD),
     묶음 A의 영어 다의어와 겹침(EN_IN_A)
  3) wordfreq Zipf (en/ko/zh). 목록을 직접 조회한다(MeCab 미사용, filter_A.py와 같은 방식)
  4) 대조어 검사: 빈도 차(CTRL_FREQ_GAP), 목록 밖(CTRL_FREQ_ZERO),
     데이터셋의 다른 입력어와 겹침(CTRL_IN_DATASET), 다른 행과 중복(CTRL_DUP)
  5) 어종: hanja_* 열로 고유어쌍/한자어쌍/혼합 분류
  6) (선택) 표준국어대사전·우리말샘 API: 입력어 동형어 수(2 미만이면 KO_NOT_HOMOGRAPH),
     대조어 동형어 여부(CTRL_HOMOGRAPH)
  7) (선택) 토크나이저 기준 토큰 수

사용법
  pip install wordfreq requests
  python annotate_B.py B_candidates_draft.csv -o B_candidates_annotated.csv --etym-api none
  # 동형어 수까지: export STDICT_API_KEY=... ; --etym-api stdict
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

from annotate_A import API, EtymClient, etym_for, load_tokenizers, ntok, shared_chars, split_multi

HERE = Path(__file__).resolve().parent
_fd: dict[str, dict] = {}


def zipf(word: str, lang: str) -> float:
    """wordfreq 목록 직접 조회. 목록에 없으면 0.0."""
    import wordfreq  # noqa: WPS433
    if lang not in _fd:
        _fd[lang] = wordfreq.get_frequency_dict(lang)
    w = (word or "").strip()
    if lang == "en":
        w = w.lower()
    f = _fd[lang].get(w)
    return round(math.log10(f * 1e9), 2) if f else 0.0


def word_etym(h1: str, h2: str) -> str:
    if h1 and h2:
        return "한자어쌍"
    if not h1 and not h2:
        return "고유어쌍"
    return "혼합"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--a-table", default=str(HERE / "A_candidates_annotated.csv"),
                    help="EN_IN_A·CTRL_IN_DATASET 검사용 묶음 A 표")
    ap.add_argument("--etym-api", choices=["urimalsaem", "stdict", "none"], default="none")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--cache", default=str(HERE / "etym_cache.json"))
    ap.add_argument("--rare-zipf", type=float, default=2.5)
    ap.add_argument("--freq-gap", type=float, default=1.0)
    ap.add_argument("--tokenizers", default="")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output or inp.with_name(inp.stem.replace("_draft", "") + "_annotated.csv"))
    rows = list(csv.DictReader(open(inp, encoding="utf-8-sig")))

    a_en, a_ko = set(), set()
    if Path(args.a_table).exists():
        for r in csv.DictReader(open(args.a_table, encoding="utf-8-sig")):
            a_en.add(r["en"].lower())
            a_ko.update(w for w in (r["ko_input"], r["ko_reverse_input"]) if w)
    else:
        print(f"[info] {args.a_table} 없음 → EN_IN_A 검사 생략", file=sys.stderr)

    b_ko = {r["ko_input"] for r in rows}
    ctrl_count = Counter(c for r in rows for c in split_multi(r["ko_control"]))

    client = None
    if args.etym_api != "none":
        key = args.api_key or os.environ.get(API[args.etym_api]["env"])
        if key:
            client = EtymClient(args.etym_api, key, Path(args.cache))
        else:
            print(f"[info] {API[args.etym_api]['env']} 없음 → 동형어 검사 생략", file=sys.stderr)
    toks = load_tokenizers([t for t in args.tokenizers.split(",") if t.strip()]) if args.tokenizers else {}
    etym_memo: dict[str, dict] = {}

    def etym(w):
        if w not in etym_memo:
            etym_memo[w] = etym_for(client, w)
        return etym_memo[w]

    for i, r in enumerate(rows, 1):
        flags = []
        # 1) 중국어 글자 공유
        r["zh_shared_chars"] = shared_chars(r["zh_answer"], r["zh_distractor"])
        if r["zh_shared_chars"]:
            flags.append("ZH_SHARE")

        # 2) 영어
        et, ed = r["en_target"].strip().lower(), r["en_distractor"].strip().lower()
        if et == ed:
            flags.append("EN_SAME")
        if " " in et or " " in ed:
            flags.append("EN_MULTIWORD")
        if et in a_en or ed in a_en:
            flags.append("EN_IN_A")

        # 3) 빈도
        r["zipf_en_target"] = zipf(et, "en")
        r["zipf_en_distractor"] = zipf(ed, "en")
        r["zipf_ko_input"] = zi = zipf(r["ko_input"], "ko")
        r["zipf_zh_answer"] = zipf(r["zh_answer"], "zh")
        r["zipf_zh_distractor"] = zd = zipf(r["zh_distractor"], "zh")
        if zd < args.rare_zipf:
            flags.append("RARE_DIST")
        if zi == 0:
            flags.append("KO_FREQ_ZERO")

        # 4) 대조어
        ctl = split_multi(r["ko_control"])
        cz = [zipf(c, "ko") for c in ctl]
        r["zipf_ko_control"] = ";".join(str(z) for z in cz)
        if any(z == 0 for z in cz):
            flags.append("CTRL_FREQ_ZERO")
        if zi and any(z and abs(z - zi) > args.freq_gap for z in cz):
            flags.append("CTRL_FREQ_GAP")
        if any(c in b_ko or c in a_ko for c in ctl):
            flags.append("CTRL_IN_DATASET")
        if any(ctrl_count[c] > 1 for c in ctl):
            flags.append("CTRL_DUP")

        # 5) 어종
        r["word_etym"] = word_etym(r["hanja_target"], r["hanja_distractor"])
        if len(r["ko_input"]) == 1:
            flags.append("KO_MONOSYLLABLE")

        # 6) 사전 동형어 (선택)
        if client is not None:
            e = etym(r["ko_input"])
            r["homograph_n_ko_input"] = e.get("homograph_n", "")
            r["entries_ko_input"] = e.get("entries", "")
            if isinstance(e.get("homograph_n"), int) and e["homograph_n"] < 2:
                flags.append("KO_NOT_HOMOGRAPH")
            ch = [etym(c).get("homograph_n", "") for c in ctl]
            r["homograph_n_ko_control"] = ";".join(str(x) for x in ch)
            if any(isinstance(x, int) and x > 1 for x in ch):
                flags.append("CTRL_HOMOGRAPH")
            if i % 20 == 0:
                client.save()

        # 7) 토큰 수 (선택)
        for short, tok in toks.items():
            for col in ("en_target", "en_distractor"):
                r[f"ntok_{col}_{short}"] = n = ntok(tok, " " + r[col])
                if n > 1:
                    flags.append(f"EN_MULTITOK_{short}")
            r[f"ntok_zh_answer_{short}"] = ntok(tok, r["zh_answer"])

        r["auto_flags"] = ";".join(dict.fromkeys(flags))

    if client is not None:
        client.save()

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    c = Counter(fl for r in rows for fl in r["auto_flags"].split(";") if fl)
    print(f"[done] {len(rows)}행 ({len(b_ko)}단어) → {out}")
    print(f"       자동 플래그: {dict(c)}")
    print(f"       어종: {dict(Counter(r['word_etym'] for r in rows))}")


if __name__ == "__main__":
    main()
