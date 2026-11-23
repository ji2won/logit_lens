#!/usr/bin/env python3
"""
국립국어원 어휘 의미 분석 말뭉치(NIKL_LS)로 B 단어의 두 뜻 출현 빈도를 센다.

eDom(영어 다의어의 '두 의미 빈도 비율')에 대응하는 한국어 쪽 통제 변수를 만든다.
말뭉치의 sense_id는 우리말샘 뜻 번호(동형어를 통틀어 이어 매김)라서, 어떤 번호가
B의 어느 뜻인지는 사람이 예문을 보고 정해 B_sense_map.csv에 적어 둔다.

  dom_target = (n_target + 0.5) / (n_target + n_distractor + 1)    # 0.5 평활
  dom_ok     = 두 뜻 합계가 --min-count 이상일 때만 True (그 아래는 비율을 믿지 않음)
  share_target = (n_target + 0.5) / (n_word + 1)   # n_word = 그 형태의 명사 출현 전체(모든 뜻)
  share_ok   = n_word가 --min-count 이상
  wordfreq 빈도는 형태 전체의 빈도이므로, 뜻별 빈도는 zipf + log10(share_target)로 근사한다(review_B.py).

출력
  B_sense_freq.csv  B 후보 행마다 n_target, n_distractor, dom_target, dom_ok
  --dump 를 주면 단어별로 말뭉치의 모든 명사 뜻 번호·빈도·예문을 출력(대응표 작성용)

사용법
  python nikl_sense_freq.py NXLS*.json SXLS*.json
  python nikl_sense_freq.py NXLS*.json SXLS*.json --dump > senses.txt
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def count(paths, words):
    """→ Counter[(form, sense_id)], 예문 dict. 명사(NN*)만 센다."""
    cnt, ex = collections.Counter(), collections.defaultdict(list)
    for p in paths:
        d = json.load(open(p, encoding='utf-8'))
        for doc in d['document']:
            for s in doc.get('sentence', []):
                for w in s.get('WSD', []):
                    if w['form'] in words and w['pos'].startswith('NN') and w['pos'] != 'NNP':
                        k = (w['form'], w['sense_id'])
                        cnt[k] += 1
                        if len(ex[k]) < 2:
                            b = w['begin']
                            ex[k].append(s['form'][max(0, b - 15):b + 15].replace('\n', ' '))
    return cnt, ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('corpus', nargs='+', help='NIKL_LS JSON 파일들')
    ap.add_argument('--draft', default=str(HERE / 'B_candidates_draft.csv'))
    ap.add_argument('--map', default=str(HERE / 'B_sense_map.csv'))
    ap.add_argument('-o', '--out', default=str(HERE / 'B_sense_freq.csv'))
    ap.add_argument('--min-count', type=int, default=10)
    ap.add_argument('--dump', action='store_true')
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.draft, encoding='utf-8-sig')))
    words = {r['ko_input'] for r in rows}
    cnt, ex = count(args.corpus, words)

    if args.dump:
        for wd in sorted(words):
            print(f'== {wd}')
            for k in sorted(k for k in cnt if k[0] == wd):
                print(f'   {k[1]} n={cnt[k]} | ' + ' / '.join(ex[k]))
        return

    n_word = collections.Counter()
    for (form, _), c in cnt.items():
        n_word[form] += c
    smap = {}
    for m in csv.DictReader(open(args.map, encoding='utf-8-sig')):
        ids = [i for i in m['nikl_ids'].split(';') if i]
        smap[(m['ko'], m['sense'])] = sum(cnt[(m['ko'], i)] for i in ids)
    missing = sorted({(r['ko_input'], r[c]) for r in rows for c in ('sense_target', 'sense_distractor')}
                     - set(smap))
    if missing:
        raise SystemExit(f'B_sense_map.csv에 없는 뜻: {missing}')

    out = []
    for r in rows:
        nt = smap[(r['ko_input'], r['sense_target'])]
        nd = smap[(r['ko_input'], r['sense_distractor'])]
        out.append({'id': r['id'], 'ko_input': r['ko_input'], 'sense_target': r['sense_target'],
                    'sense_distractor': r['sense_distractor'], 'nikl_n_target': nt,
                    'nikl_n_distractor': nd, 'nikl_dom_target': round((nt + 0.5) / (nt + nd + 1), 3),
                    'nikl_dom_ok': nt + nd >= args.min_count, 'nikl_n_word': n_word[r['ko_input']],
                    'nikl_share_target': round((nt + 0.5) / (n_word[r['ko_input']] + 1), 4),
                    'nikl_share_ok': n_word[r['ko_input']] >= args.min_count})
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    words_ok = sorted({o['ko_input'] for o in out if o['nikl_dom_ok']})
    both = sorted({o['ko_input'] for o in out if o['nikl_n_target'] and o['nikl_n_distractor']})
    print(f'[nikl] {len(words)}단어 → {args.out}')
    print(f'  두 뜻 모두 1회 이상: {len(both)}단어  {" ".join(both)}')
    print(f'  두 뜻 합 {args.min_count}회 이상(dom_ok): {len(words_ok)}단어  {" ".join(words_ok)}')


if __name__ == '__main__':
    main()
