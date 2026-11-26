# 묶음 A (영어 다의어) 후보 표 · 자동 부착 스크립트

## 파일
| 파일 | 내용 |
|---|---|
| `A_candidates_draft.csv` | 후보 초안 213행 (영어 단어 191개). A 파이프라인 2단계까지 |
| `annotate_A.py` | 글자 공유 검사 · wordfreq 빈도 · 어원(우리말샘/표준국어대사전) · (선택) 토큰 수 부착 |
| `A_candidates_annotated_noetym.csv` | 위 스크립트를 어원 없이(`--etym-api none`) 돌린 결과 |

## 표 열
| 열 | 뜻 |
|---|---|
| `id`, `en` | 문항 id, 영어 다의어 |
| `direction` | `fwd` 기본 방향 / `rev` 방향 뒤집은 예시(용수철→春 등 19개) / `alt` 같은 단어의 다른 입력·방해어 버전 3개 |
| `sense_target`, `ko_input`, `zh_answer` | 목표 의미, 한국어 입력, 중국어 정답 |
| `sense_distractor`, `zh_distractor` | 방해 의미, 중국어 방해어 |
| `ko_reverse_input` | 방해 의미의 한국어 단어. 방향을 뒤집을 때 입력으로 씀 |
| `ko_control` | 대조어 후보 2개(`;` 구분). I(w)=log p(방해어\|입력) − log p(방해어\|대조어)의 분모 쪽 |
| `draft_flags` | 초안 작성 시 붙인 사람용 플래그(아래) |
| `note` | 메모 |

bootstrap은 `en` 단위로 묶어서 리샘플링하면 된다(`rev`, `alt` 행도 같은 `en`).

### 초안 플래그 (`draft_flags`, 작성자 판단, 검수 필요)
- `KO_POLY` / `KO_POLY_REV` / `KO_POLY_CTRL`: 한국어 입력 / 역방향 입력 / 대조어가 다의어로 보임 → 7절 필터 대상
- `ZH_SHARE`: 중국어 정답·방해어 글자 공유 예상(스크립트가 다시 확인). `ZH_POLY`: 중국어 쪽에도 다의성 있음
- `REL`: 두 의미가 어원적으로 이어진 다의(polysemy)에 가까움. 동음이의(homonymy)만 원하면 제외 후보. `REL?`는 애매
- `LOAN` / `LOAN_REV`: 한국어 입력/역방향 입력이 영어 외래어(펀치, 팬 등). 입력에 영어 음이 새므로 따로 표시
- `RARE`, `RARE_DIST`: 단어 자체 또는 방해어가 드묾. `MULTIWORD(_REV)`: 여러 단어 표현
- `WEAK`: 명사-명사 대응이 약함(제외 후보). `SLANG`: 방해 의미가 속어

## 스크립트
```bash
pip install "wordfreq[cjk]" requests          # 토큰 수까지: transformers
export URIMALSAEM_API_KEY=발급받은키           # 표준국어대사전이면 STDICT_API_KEY
python annotate_A.py A_candidates_draft.csv -o A_candidates_annotated.csv --etym-api urimalsaem
# 선택: --tokenizers meta-llama/Llama-3.2-3B,LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
```
- 키가 없으면 어원만 건너뛰고 나머지는 돈다. API 응답은 `etym_cache.json`에 저장되어 재실행 시 다시 부르지 않는다.
- Llama-3.2 토크나이저는 gated라 `huggingface-cli login`이 필요하다.

### 추가되는 열
- `zh_shared_chars`: 정답·방해어가 공유하는 한자
- `zipf_*`: wordfreq Zipf 빈도 (en / ko 입력 / ko 역방향 / ko 대조어들 / zh 정답 / zh 방해어)
- `etym_ko_input`, `origin_ko_input`: 고유어 / 한자어 / 외래어 / 혼종어, 원어 표기. 동형어끼리 종류가 다르면 `AMBIG:…`
- `homograph_n_ko_input`, `entries_ko_input`: 같은 표기 명사 표제어 수와 목록(번호:어원(원어):뜻풀이 앞부분). 한국어 다의어 필터 보조
- `etym_ko_reverse`, `homograph_n_ko_reverse`, `etym_ko_control`
- `ntok_*`: 모델별 토큰 수(`ntok_en_*`는 앞에 공백을 붙인 " spring" 기준)
- `auto_flags`: `ZH_SHARE`, `RARE_DIST`(방해어 Zipf < 2.5), `CTRL_FREQ_GAP`(입력·대조어 Zipf 차 > 1.0), `KO_FREQ_ZERO`, `KO_HOMOGRAPH(_REV)`, `ETYM_AMBIG`, `CTRL_ETYM_DIFF`, `EN_MULTITOK_<모델>`. 기준값은 `--rare-zipf`, `--freq-gap`으로 조절

## 주의
- 중국어 번역과 대조어는 초안이다. CC-CEDICT 확인과 사람 검수가 필요하다.
- wordfreq 한국어는 합성어(나무껍질, 강둑 등)를 목록에 갖고 있지 않아 0이 나온다(`KO_FREQ_ZERO`, 31행). 이 경우 빈도 통제는 다른 자료로 보완하거나 해당 문항을 빼야 한다.
- 동형어가 여러 개일 때 대표 어원은 첫 표제어 것을 쓴다. 사원(寺院/社員)처럼 둘 다 한자어면 `AMBIG`가 안 뜨므로 `homograph_n`을 같이 볼 것.
- 어원 API 파서는 응답 구조를 재귀로 훑어 `original_language`, `language_type` 필드를 찾는다. 작성 환경에서 API 서버에 접속할 수 없어 실제 응답으로는 시험하지 못했고, 모의 응답으로만 확인했다. 처음 돌릴 때 몇 단어의 `entries_ko_input`을 눈으로 확인할 것.
- 영어 우세도(eDom norms)는 아직 붙이지 않았다(본실험 단계).
