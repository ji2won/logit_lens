#!/usr/bin/env python3
"""
묶음 B 검토 열 생성. 자동 사전 조회 + 사람 판단(아래 사전들)을 열로 붙인다.

  1) CC-CEDICT: zh_answer, zh_distractor 표제어 조회. 있으면 OK, 없으면 NOT_FOUND가 기본값.
     뜻이 어긋나거나 조합어인 경우는 J에 사람이 판정을 적는다.
  2) 영어 다의어(EN_POLY): 영어 번역이 흔한 다른 뜻을 가져 중간층 영어 토큰이 애매해지는 경우
  3) 한국어 뜻 빈도(KO_SENSE): 두 뜻 중 하나가 일상에서 드물게 쓰이는 경우
  4) 제3의 뜻(OTHER_SENSE): 다른 뜻이 두 뜻만큼 흔하거나 더 흔한 경우
  5) 고유명사(PROPER)
  6) 대조어 대체(S): annotate 결과에서 CTRL_FREQ_GAP / CTRL_FREQ_ZERO / CTRL_DUP / CTRL_IN_DATASET가 뜬 행.
     첫 실행 뒤 출력되는 목록을 보고 채운다. 제안어는 실행 시 Zipf 차 1.0 이내인지 검사한다.

판정 값은 A와 같다: Y = 문제 있음(제외), ? = 경계(제외), N 또는 빈칸 = 통과.

사용법
  python review_B.py --cedict ../cedict_ts.u8
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# ---------- 1. CEDICT 판정 덮어쓰기 ----------
# 단어 -> (판정, 대체 제안, 메모). 판정: OK / COMPOSITIONAL / PARTIAL / MISMATCH / NOT_FOUND / NOT_IN_GLOSS
# 첫 실행 뒤 NOT_FOUND 목록과 뜻풀이를 보고 채운다.
J: dict[str, tuple[str, str, str]] = {
    # 예: '灭火':('OK','','to extinguish a fire. 명사 소화(消火)와 품사 차이'),
}

# ---------- 2. 영어 다의어 ----------
# 영어 단어 -> (판정, 이유)
EN_POLY = {
    'fan': ('Y', '팬(admirer). 한국어 외래어 팬과도 겹침'),
    'swallow': ('Y', '삼키다(동사)'),
    'lot': ('Y', 'a lot(많음), 땅 한 구획'),
    'fly': ('Y', '날다(동사), 바지 지퍼'),
    'minister': ('Y', '목사(성직자)'),
    'spectacle': ('Y', 'spectacles(안경)'),
    'order': ('Y', '명령, 순서, 질서'),
    'spell': ('Y', '철자를 말하다, 한동안(a spell)'),
    'branch': ('Y', '지점·부서'),
    'stake': ('Y', '말뚝, 화형'),
    'reason': ('Y', '이유(cause). 이성 뜻과 같은 단어'),
    'futures': ('Y', 'future(미래)의 복수형과 같음'),
    'mouse': ('Y', '컴퓨터 마우스'),
    'temple': ('?', '관자놀이'),
    'stream': ('?', '스트리밍, 흐름'),
    'bridge': ('?', '카드 게임 브리지, 치아 브리지'),
    'past': ('?', '전치사·형용사 용법이 대부분'),
}

# ---------- 3. 한국어 뜻이 일상에서 드문 경우 ----------
# (입력어, 뜻 열 값) -> (판정, 이유). 해당 뜻이 정답이든 방해어든 그 단어의 두 행 모두 제외된다.
KO_SENSE = {
    ('수술', '꽃의 기관'): ('?', '수술(stamen)은 교과서 용어. 일상 빈도 낮음'),
    ('공포', '법령 발표'): ('?', '公布는 뉴스·법령 문체에서만 쓰임'),
    ('이해', '이익과 손해'): ('?', '利害는 대부분 이해관계 합성어로 쓰임'),
    ('의지', '기댐'): ('?', '依支는 대부분 동사 의지하다로 쓰임'),
    ('주의', '사상 체계'): ('?', '主義는 대부분 -주의 접미 용법'),
    ('동화', '같아짐'): ('?', '同化는 학술어'),
    ('선물', '선물 거래'): ('?', 'A에서도 경계(?) 판정'),
    ('실례', '실제 예'): ('?', '實例는 문어체'),
    ('비', '빗자루'): ('?', '단독 비보다 빗자루가 흔함'),
    ('양', '분량'): ('N', '양이 많다 등 흔함'),
    ('고장', '지역'): ('N', '우리 고장 등 흔함'),
}

# ---------- 4. 제3의 뜻이 두 뜻만큼 흔한 경우 ----------
OTHER_SENSE = {
    '김': ('Y', '성씨 김이 압도적으로 흔함'),
    '연기': ('Y', '延期(미룸)도 일상어'),
    '비': ('Y', '碑·比 및 접두 용법'),
    '양': ('?', '兩, -양(孃)'),
    '가지': ('?', '몇 가지(종류)가 매우 흔함'),
    '배': ('N', '신체 배도 흔하지만 선박/과일 대비 측정 가능. 메모만'),
    '감': ('N', '옷감은 합성어로 주로 쓰임'),
    '사기': ('N', '沙器(사기그릇)는 합성어로 주로 쓰임'),
    '의사': ('N', '義士는 드묾'),
    '전기': ('N', '前期는 문어'),
    '동기': ('N', '冬期 드묾'),
    '사자': ('N', '死者는 문어'),
    '부인': ('N', '夫人은 같은 아내 뜻 계열'),
}

# ---------- 5. 고유명사 ----------
PROPER = {'Paris'}

# ---------- 6. 대조어 대체 ----------
# id -> ([(단어, 어종)], 메모). 바꿀 대조어 자리에 순서대로 들어간다(filter_B.py).
S: dict[str, tuple[list[tuple[str, str]], str]] = {
    # 예: 'B001': ([('여객선','한자어')], '버스(5.1) 대체'),
}


def load_cedict(path: Path) -> dict[str, list[str]]:
    """cedict_ts.u8 → {간체: [병음: 뜻풀이, ...]}"""
    idx: dict[str, list[str]] = {}
    for line in path.read_text("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        trad, simp, rest = line.split(" ", 2)
        pinyin = rest[rest.index("[") + 1:rest.index("]")]
        gloss = rest[rest.index("/") + 1:].strip("/").replace("/", " / ")
        idx.setdefault(simp, []).append(f"{pinyin}: {gloss}")
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(HERE / "B_candidates_annotated.csv"))
    ap.add_argument("--output", default=str(HERE / "B_candidates_reviewed.csv"))
    ap.add_argument("--cedict", default=str(HERE.parent / "cedict_ts.u8"))
    ap.add_argument("--freq-gap", type=float, default=1.0)
    args = ap.parse_args()

    import wordfreq
    fd = wordfreq.get_frequency_dict("ko")
    zf = lambda w: round(math.log10(fd[w] * 1e9), 2) if w in fd else 0.0  # noqa: E731

    d = pd.read_csv(args.input, keep_default_na=False)
    idx = load_cedict(Path(args.cedict))

    # 1. CEDICT
    def ced(w):
        g = idx.get(w)
        default = ('OK', '', '') if g else ('NOT_FOUND', '', '')
        st, fix, note = J.get(w, default)
        if fix and fix not in idx:
            note += ' [제안어도 CEDICT 없음]'
        return ('Y' if g else 'N'), (' || '.join(g))[:160] if g else '', st, fix, note

    for col, pre in [('zh_answer', 'cedict_answer'), ('zh_distractor', 'cedict_distractor')]:
        res = d[col].map(ced)
        for k, name in enumerate(['found', 'gloss', 'ok', 'fix', 'note']):
            d[f'{pre}_{name}'] = res.map(lambda x, k=k: x[k])

    # 2~5. 사람 판단
    def worst(*vals):
        return 'Y' if 'Y' in vals else ('?' if '?' in vals else ('N' if 'N' in vals else ''))

    ep = [(EN_POLY.get(r.en_target.lower(), ('', '')), EN_POLY.get(r.en_distractor.lower(), ('', '')))
          for r in d.itertuples()]
    d['en_poly'] = [worst(a[0], b[0]) for a, b in ep]
    d['en_poly_which'] = [' / '.join(f'{w}: {x[1]}' for w, x in ((r.en_target, a), (r.en_distractor, b)) if x[1])
                          for r, (a, b) in zip(d.itertuples(), ep)]

    ks = [(KO_SENSE.get((r.ko_input, r.sense_target), ('', '')), KO_SENSE.get((r.ko_input, r.sense_distractor), ('', '')))
          for r in d.itertuples()]
    d['ko_sense_rare'] = [worst(a[0], b[0]) for a, b in ks]
    d['ko_sense_rare_which'] = [' / '.join(x[1] for x in (a, b) if x[1]) for a, b in ks]

    d['other_sense_dominant'] = d.ko_input.map(lambda w: OTHER_SENSE.get(w, ('', ''))[0])
    d['other_sense_which'] = d.ko_input.map(lambda w: OTHER_SENSE.get(w, ('', ''))[1])
    d['proper_noun'] = [('Y' if r.en_target in PROPER or r.en_distractor in PROPER else '') for r in d.itertuples()]

    # 6. 대조어 대체 제안
    need = d.auto_flags.str.contains('CTRL_FREQ_GAP|CTRL_FREQ_ZERO|CTRL_DUP|CTRL_IN_DATASET')
    for c in ['ctrl_suggest', 'ctrl_suggest_zipf', 'ctrl_suggest_etym', 'ctrl_suggest_note']:
        d[c] = ''
    for i, r in d[need].iterrows():
        words, note = S.get(r['id'], ([], 'TODO'))
        d.at[i, 'ctrl_suggest'] = ';'.join(w for w, _ in words)
        d.at[i, 'ctrl_suggest_zipf'] = ';'.join(str(zf(w)) for w, _ in words)
        d.at[i, 'ctrl_suggest_etym'] = ';'.join(e for _, e in words)
        d.at[i, 'ctrl_suggest_note'] = note
        for w, _ in words:
            assert zf(w) > 0 and abs(zf(w) - float(r['zipf_ko_input'])) <= args.freq_gap, (r['id'], w, zf(w))

    d.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'[done] {d.shape} → {args.output}')
    for c in ['cedict_answer_ok', 'cedict_distractor_ok', 'en_poly', 'ko_sense_rare', 'other_sense_dominant']:
        print(' ', c, d[c].value_counts().to_dict())

    # 사람이 채울 목록
    nf = sorted(set(d.loc[d.cedict_answer_ok == 'NOT_FOUND', 'zh_answer']))
    print(f'\n[J에 판정 필요] CEDICT NOT_FOUND {len(nf)}개: {" ".join(nf)}')
    todo = d[need & (d.ctrl_suggest_note == 'TODO')]
    print(f'[S에 대조어 필요] {len(todo)}행')
    for r in todo.itertuples():
        print(f'  {r.id} {r.ko_input}({r.sense_target}) 입력 {r.zipf_ko_input} | '
              f'대조어 {r.ko_control} = {r.zipf_ko_control} | {r.auto_flags}')


if __name__ == '__main__':
    main()
