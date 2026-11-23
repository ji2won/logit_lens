#!/usr/bin/env python3
"""
묶음 B 기준 적용 → B_candidates_reviewed.csv에 keep/exclude_reason 기록, B_final.csv 생성.

단계 (조금이라도 애매하면 뺀다)
  1. 중국어:  CEDICT 판정이 OK/COMPOSITIONAL이 아님, 정답·방해어 글자 공유(ZH_SHARE)
  2. 영어:    두 뜻이 같은 영어(EN_SAME), 여러 단어(EN_MULTIWORD), 묶음 A와 겹침(EN_IN_A),
              영어 다의어 Y/?
  3. 한국어:  한 뜻이 일상에서 드묾 Y/?, 제3의 뜻이 우세 Y/?, 고유명사,
              사전상 동형어가 아님(API를 돌린 경우)
  4. 빈도:    입력어가 wordfreq 목록 밖, 대체 후에도 대조어가 0·차 1.0 초과·중복·데이터셋 단어
  5. 짝:      같은 입력어의 다른 행이 빠지면 함께 뺀다(두 뜻을 대칭으로 비교하기 위해)

뜻 빈도(선택): B_sense_freq.csv(nikl_sense_freq.py 결과)가 있으면 nikl_* 열을 붙인다.
  --dom-range LO HI 를 주면 dom_ok인 행 중 dom_target이 범위를 벗어나는 단어를 뺀다(DOM_EXTREME).
  기본은 빼지 않고 열만 붙여 분석에서 공변량으로 쓴다.

사용법
  python filter_B.py
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OKS = {'OK', 'COMPOSITIONAL'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reviewed', default=str(HERE / 'B_candidates_reviewed.csv'))
    ap.add_argument('--final', default=str(HERE / 'B_final.csv'))
    ap.add_argument('--a-table', default=str(HERE / 'A_candidates_annotated.csv'))
    ap.add_argument('--freq-gap', type=float, default=1.0)
    ap.add_argument('--sense-freq', default=str(HERE / 'B_sense_freq.csv'))
    ap.add_argument('--dom-range', type=float, nargs=2, default=None, metavar=('LO', 'HI'))
    args = ap.parse_args()

    import wordfreq
    fd = wordfreq.get_frequency_dict('ko')
    zf = lambda w: round(math.log10(fd[w] * 1e9), 2) if w in fd else 0.0  # noqa: E731

    d = pd.read_csv(args.reviewed, keep_default_na=False)
    # 이전 실행 결과 열은 다시 계산
    d = d.drop(columns=[c for c in ['exclude_reason', 'keep', 'ko_control_final', 'zipf_ko_control_final',
                                    'ko_control_changed'] if c in d.columns or c.startswith('nikl_')])
    d = d.drop(columns=[c for c in d.columns if c.startswith('nikl_')])
    if Path(args.sense_freq).exists():
        sf = pd.read_csv(args.sense_freq, keep_default_na=False)
        d = d.merge(sf[['id'] + [c for c in sf.columns if c.startswith('nikl_')]], on='id', how='left')
        print(f'[sense] {args.sense_freq} 병합 (dom_ok {int((d.nikl_dom_ok.astype(str) == "True").sum())}행)')
    elif args.dom_range:
        raise SystemExit('--dom-range에는 B_sense_freq.csv가 필요함 (nikl_sense_freq.py 먼저 실행)')

    dataset_words = set(d.ko_input)
    if Path(args.a_table).exists():
        a = pd.read_csv(args.a_table, keep_default_na=False)
        dataset_words |= set(a.ko_input) | set(a.ko_reverse_input)
    dataset_words.discard('')

    # ---------- 최종 대조어 ----------
    def bad_slots(words, zi):
        cnt = Counter(w for ws in d.ko_control.str.split(';') for w in ws)
        return [zf(w) == 0 or abs(zf(w) - zi) > args.freq_gap or w in dataset_words or cnt[w] > 1
                for w in words]

    fin, finz, chg = [], [], []
    for r in d.itertuples():
        orig = r.ko_control.split(';')
        zi = float(getattr(r, 'zipf_ko_sense', r.zipf_ko_input))
        sug = [w for w in r.ctrl_suggest.split(';') if w]
        new = [sug.pop(0) if bad and sug else w for w, bad in zip(orig, bad_slots(orig, zi))]
        fin.append(';'.join(new))
        finz.append(';'.join(str(zf(w)) for w in new))
        chg.append('Y' if new != orig else 'N')
    d['ko_control_final'], d['zipf_ko_control_final'], d['ko_control_changed'] = fin, finz, chg

    # 최종 대조어 중복은 대체 후 기준으로 다시 센다
    final_cnt = Counter(w for ws in d.ko_control_final.str.split(';') for w in ws)

    # ---------- 제외 사유 ----------
    def reasons(r):
        out = []
        fl = set(r.auto_flags.split(';'))
        # 1. 중국어
        if r.cedict_answer_ok not in OKS:
            out.append(f'CEDICT_ANSWER_{r.cedict_answer_ok}')
        if r.cedict_distractor_ok not in OKS:
            out.append(f'CEDICT_DISTRACTOR_{r.cedict_distractor_ok}')
        if 'ZH_SHARE' in fl:
            out.append('ZH_SHARE')
        # 2. 영어
        out += [f for f in ('EN_SAME', 'EN_MULTIWORD', 'EN_IN_A') if f in fl]
        if r.en_poly == 'Y':
            out.append('EN_POLY')
        if r.en_poly == '?':
            out.append('EN_POLY_BORDERLINE')
        # 3. 한국어
        if r.ko_sense_rare == 'Y':
            out.append('KO_SENSE_RARE')
        if r.ko_sense_rare == '?':
            out.append('KO_SENSE_BORDERLINE')
        if r.other_sense_dominant == 'Y':
            out.append('OTHER_SENSE_DOMINANT')
        if r.other_sense_dominant == '?':
            out.append('OTHER_SENSE_BORDERLINE')
        if r.proper_noun == 'Y':
            out.append('PROPER_NOUN')
        if 'KO_NOT_HOMOGRAPH' in fl:
            out.append('KO_NOT_HOMOGRAPH')
        # 4. 빈도
        if float(r.zipf_ko_input) == 0:
            out.append('FREQ_INPUT_ZERO')
        else:
            zi = float(getattr(r, 'zipf_ko_sense', r.zipf_ko_input))
            for w, z in zip(r.ko_control_final.split(';'), r.zipf_ko_control_final.split(';')):
                z = float(z)
                if z == 0:
                    out.append('FREQ_CTRL_ZERO')
                elif abs(z - zi) > args.freq_gap:
                    out.append('CTRL_FREQ_GAP')
                if w in dataset_words:
                    out.append('CTRL_IN_DATASET')
                if final_cnt[w] > 1:
                    out.append('CTRL_DUP')
        # 4b. 뜻 빈도 비율(선택)
        if args.dom_range and str(r.nikl_dom_ok) == 'True':
            lo, hi = args.dom_range
            if not lo <= float(r.nikl_dom_target) <= hi:
                out.append('DOM_EXTREME')
        return ';'.join(dict.fromkeys(out))

    d['exclude_reason'] = d.apply(reasons, axis=1)

    # 5. 짝
    dropped_words = set(d.loc[d.exclude_reason != '', 'ko_input'])
    pair = d.ko_input.isin(dropped_words) & (d.exclude_reason == '')
    d.loc[pair, 'exclude_reason'] = 'PAIR_DROPPED'
    d['keep'] = d.exclude_reason.eq('').map({True: 'Y', False: 'N'})

    d.to_csv(args.reviewed, index=False, encoding='utf-8-sig')
    k = d[d.keep == 'Y']
    k.to_csv(args.final, index=False, encoding='utf-8-sig')

    # ---------- 요약 ----------
    steps = [
        ('1. 중국어', 'CEDICT|ZH_SHARE'),
        ('2. 영어', 'EN_'),
        ('3. 한국어', 'KO_SENSE|OTHER_SENSE|PROPER|KO_NOT_HOMOGRAPH'),
        ('4. 빈도·대조어·뜻 비율', 'FREQ_|CTRL_|DOM_'),
        ('5. 짝', 'PAIR_DROPPED'),
    ]
    print(f'[done] {len(d)}행({d.ko_input.nunique()}단어) → 남은 {len(k)}행({k.ko_input.nunique()}단어)')
    print(f'       어종: {k.groupby("word_etym").ko_input.nunique().to_dict()}  대조어 바뀐 행: {(k.ko_control_changed == "Y").sum()}')
    print('\n| 단계 | 해당 행 | 새로 빠진 행 |')
    print('|---|---|---|')
    gone = pd.Series(False, index=d.index)
    for name, pat in steps:
        hit = d.exclude_reason.str.contains(pat)
        print(f'| {name} | {hit.sum()} | {(hit & ~gone).sum()} |')
        gone |= hit
    print('\n사유별:', d.exclude_reason.str.split(';').explode().replace('', pd.NA).dropna().value_counts().to_dict())
    print('\n제외 단어:')
    for w, g in d[d.keep == 'N'].groupby('ko_input', sort=False):
        print(f'  {w}: ' + ' | '.join(f'{r.id} {r.exclude_reason}' for r in g.itertuples()))


if __name__ == '__main__':
    main()
