# 묶음 C(일반 단어) 문항 설계와 선별 기준

A·B가 다의어·동음이의어로 **간섭**을 보는 묶음이라면, C는 다의성이 없는 **평범한 번역 문항**이다. 대조어도 방해어도 없다. 두 가지에 쓴다.

1. **가설 1(늘 영어로 기우는 성분)**: 렌즈 중간층에서 정답 영어(바다 → sea)만 오르는지, 입력과 무관한 영어(culture)까지 오르는지 본다. 무관한 영어까지 오르면 "입력 단어를 영어로 생각한다"가 아니라 "잔차에 영어 쪽 성분이 늘 있다"는 쪽이다. 평균 제거 렌즈에서 이 차이가 어떻게 바뀌는지가 핵심이다.
2. **반복 과제**(`repeat`): 영어가 필요 없는 과제에서도 영어가 오르는지 본다.

원칙은 A·B와 같이 **조금이라도 애매하면 뺀다**이다.

> 상태: 후보 221개 → 최종 **204개** (`C_final.csv`). wordfreq·CC-CEDICT·세 모델 토크나이저로 실제 검사함.

## 파일

| 파일 | 내용 |
|---|---|
| `make_C_draft.py` | 후보 목록(손으로 작성) → `C_candidates_draft.csv` |
| `build_C.py` | 자동 검사 + 사람 판단 + 무관 단어 짝짓기 → `C_candidates_annotated.csv`(전체, 제외 사유 포함), `C_final.csv` |

```bash
python make_C_draft.py
python build_C.py --cedict ../cedict_ts.u8 --tokenizers
```

## 열 (`C_final.csv`)

| 열 | 내용 |
|---|---|
| `ko`, `zh`, `en` | 입력어, 정답 중국어, 정답 뜻의 영어(렌즈 probe `target`) |
| `en_unrelated`, `zh_unrelated`, `ko_unrelated` | 무관 단어. 다른 C 문항 하나를 그대로 가져온 것(아래 "짝짓기") |
| `etym`, `hanja` | 고유어 99 / 한자어 92 / 혼종어 9 / 모호 4. **손으로 적음**(사전 API 미사용). 모호는 한자 어원설이 있지만 사전마다 표기가 갈리는 감자·당근·배추·상추 |
| `sino_same` | 한자어의 한자가 중국어 정답과 같은 글자인가(번체→간체 변환 포함). Y 51개. 한자음을 거쳐 중국어로 바로 갈 수 있으므로 분석 때 나눠 본다 |
| `category`, `category_unrelated` | 자연·동물·신체·음식·물건·장소·사람·식물·추상 |
| `zipf_ko/en/zh`, `zipf_en_unrelated` | wordfreq Zipf |
| `ntok_en_llama/exaone/qwen` | `' ' + en`의 토큰 수. 1토큰: Llama 180, EXAONE 189, Qwen 180개. 세 모델 모두 여러 토큰인 것은 15개. A와 같이 빼지 않고 기록만 한다(렌즈 점수는 첫 토큰 후보 확률을 쓴다) |

`load_C`는 `id, ko, zh, en, en_unrelated, zh_unrelated, etym`만 읽고, 나머지 열은 분석용 메타데이터다.

## 후보를 쓸 때 미리 뺀 것

- **A·B와 겹치는 단어**: 입력어·대조어·영어 전부. 다시 `build_C.py`가 검사한다(`OVERLAP_A/B`).
- **흔한 동형어가 있는 한국어**: 눈·배·차·밤·말·다리·연기·정부·산(酸)·강(講)·문·벽·돈 등. 한 음절어는 동형어가 사실상 없는 것(물·꿀·꿈·피·왕·혀·뺨 등)만 넣었다.
- **외래어**: 빵·버스·컴퓨터·펭귄처럼 한국어 표기가 외래어이거나, 연어→三文鱼(salmon 음차)처럼 중국어가 영어 음차인 것. 표기 자체가 영어를 거치므로 "영어가 오르는가"를 볼 수 없다.
- **영어 다의어**: fly·bear·bank·watch·spring·letter·glass·park·train 등. 중간층에서 영어가 떠도 어느 뜻인지 알 수 없다.
- **영어가 여러 단어**인 것: sweet potato·soy sauce·post office.
- **few-shot 단어**: 학교·거울·친구·연필·창문(및 그 중국어·영어).

## 선별 단계 (`build_C.py`)

| 단계 | 확인하는 것 | 제외 사유 코드 | 제외 수 |
|---|---|---|---|
| 1. 겹침 | A·B 단어, few-shot | `OVERLAP_A`, `OVERLAP_B`, `FEWSHOT` | 0 |
| 2. 빈도 | 한국어·영어 Zipf ≥ 3.0, 중국어가 목록에 있는가 | `FREQ_KO_LOW`, `FREQ_EN_LOW`, `FREQ_ZH_ZERO` | 11 (눈사람·허파·분필·불가사리·살구·무당벌레 / 배꼽·손톱·콧구멍·지우개) |
| 3. 중국어 | CC-CEDICT에 있고 뜻풀이에 그 영어가 있는가 | `CEDICT_NOT_FOUND`, `CEDICT_EN_MISS` | 0 (자동 불일치 12개는 사람이 확인해 모두 OK, 아래 J) |
| 4. 영어 다의어 | 사람 판단 `EN_POLY` | `EN_POLY(_BORDERLINE)` | 3 (wound, desert, age) |
| 5. 한국어 동형어 | 사람 판단 `KO_HOMOGRAPH` | `KO_HOMOGRAPH(_BORDERLINE)` | 4 (도로, 선물, 공주, 보석) |

### 사람 판단 (`build_C.py` 맨 위의 사전)

- `EN_POLY`: wound(wind의 과거분사), desert(동사 '버리다'), age(시대).
- `KO_HOMOGRAPH`: 도로(부사 '되돌려'), 선물(先物 거래), 공주(지명 公州), 보석(保釋).
- `J`(CEDICT 덮어쓰기): 뜻풀이의 단어가 영어와 글자만 다른 경우. 乌龟(tortoise/turtle), 蚯蚓(earthworm), 斧头(ax), 照片(photograph), 宝石(gem), 作家(author), 树叶(tree leaves), 数学(mathematics), 그리고 동사·형용사로만 풀이됐지만 명사로도 쓰는 悲伤·愤怒·诞生·结婚.

## 무관 단어 짝짓기

무관 단어를 따로 고르지 않고 **C 문항끼리 순열로 배정**한다. 그러면 무관 영어 204개와 정답 영어 204개가 같은 단어 집합이 되어 빈도·토큰 수·품사 분포가 자동으로 같아진다. "정답 영어가 무관 영어보다 높다"는 차이를 단어 자체의 성질이 아니라 입력과의 관계로만 설명할 수 있다.

- 비용: 영어 Zipf 차. 헝가리안 알고리즘(`scipy.optimize.linear_sum_assignment`)으로 총 차이를 최소화.
- 금지: 자기 자신, 같은 범주(고양이–개구리 X), 중국어 글자 공유, 한쪽 영어가 다른 쪽을 포함(sun/sunflower), 영어 앞 세 글자가 같음(첫 토큰이 겹칠 수 있음).
- 결과: Zipf 차 평균 0.03, 최대 0.72.

## 한계

- 어종은 사전 API 없이 손으로 적었다. A처럼 우리말샘 원어 필드로 다시 확인하면 좋다(`annotate_A.py`의 `EtymClient` 재사용 가능).
- 한국어 동형어 판정도 사람 판단이다. B처럼 국립국어원 의미 분석 말뭉치로 뜻 비율을 보지는 않았다.
- 추상어(40개)는 구체어보다 번역이 덜 고정적이다(행복 → 幸福/快乐). `category`로 나눠 볼 수 있다.
