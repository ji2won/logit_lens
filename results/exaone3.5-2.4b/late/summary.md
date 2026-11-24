# 결과 요약 (late)

층 수 30, 렌즈 점수 구간 23–29층, bootstrap 2000회

## 1. 신뢰도 (프롬프트 4개, Cronbach's alpha)

기준 0.7. 미달한 지표의 상관은 '과장'으로 해석하지 않는다.

| dataset | task | measure | alpha | 통과 |
|---|---|---|---|---|
| A | translate | 간섭 I | 0.966 | O |
| A | translate_en | 간섭 I | 0.963 | O |
| B | translate | 간섭 I | 0.987 | O |
| B | translate_ctx | 간섭 I | 0.982 | O |
| A | translate | 렌즈 S (affine) | 0.993 | O |
| A | translate | 렌즈 S_diff (affine) | 0.994 | O |
| A | translate | 렌즈 S (mean) | 0.990 | O |
| A | translate | 렌즈 S_diff (mean) | 0.993 | O |
| A | translate | 렌즈 S (vanilla) | 0.991 | O |
| A | translate | 렌즈 S_diff (vanilla) | 0.993 | O |
| A | translate_en | 렌즈 S (affine) | 0.970 | O |
| A | translate_en | 렌즈 S_diff (affine) | 0.985 | O |
| A | translate_en | 렌즈 S (mean) | 0.934 | O |
| A | translate_en | 렌즈 S_diff (mean) | 0.986 | O |
| A | translate_en | 렌즈 S (vanilla) | 0.922 | O |
| A | translate_en | 렌즈 S_diff (vanilla) | 0.987 | O |
| B | translate | 렌즈 S (affine) | 0.991 | O |
| B | translate | 렌즈 S_diff (affine) | 0.992 | O |
| B | translate | 렌즈 S (mean) | 0.986 | O |
| B | translate | 렌즈 S_diff (mean) | 0.988 | O |
| B | translate | 렌즈 S (vanilla) | 0.986 | O |
| B | translate | 렌즈 S_diff (vanilla) | 0.988 | O |
| B | translate_ctx | 렌즈 S (affine) | 0.986 | O |
| B | translate_ctx | 렌즈 S_diff (affine) | 0.984 | O |
| B | translate_ctx | 렌즈 S (mean) | 0.984 | O |
| B | translate_ctx | 렌즈 S_diff (mean) | 0.984 | O |
| B | translate_ctx | 렌즈 S (vanilla) | 0.984 | O |
| B | translate_ctx | 렌즈 S_diff (vanilla) | 0.983 | O |

## 2. 렌즈 점수 × 간섭 상관 (Spearman, 문항 단위)

| dataset | task | lens | x | n | r | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| A | translate | affine | S | 99 | 0.012 | -0.188 | 0.218 |
| A | translate | affine | S_diff | 99 | 0.025 | -0.185 | 0.230 |
| A | translate | mean | S | 99 | 0.047 | -0.143 | 0.236 |
| A | translate | mean | S_diff | 99 | -0.033 | -0.233 | 0.170 |
| A | translate | vanilla | S | 99 | 0.047 | -0.150 | 0.237 |
| A | translate | vanilla | S_diff | 99 | -0.045 | -0.242 | 0.161 |
| A | translate_en | affine | S | 99 | -0.094 | -0.293 | 0.109 |
| A | translate_en | affine | S_diff | 99 | 0.083 | -0.131 | 0.283 |
| A | translate_en | mean | S | 99 | -0.003 | -0.198 | 0.194 |
| A | translate_en | mean | S_diff | 99 | 0.019 | -0.168 | 0.213 |
| A | translate_en | vanilla | S | 99 | -0.007 | -0.213 | 0.201 |
| A | translate_en | vanilla | S_diff | 99 | 0.001 | -0.185 | 0.196 |
| B | translate | affine | S | 60 | 0.677 | 0.525 | 0.810 |
| B | translate | affine | S_diff | 60 | 0.770 | 0.645 | 0.866 |
| B | translate | mean | S | 60 | 0.735 | 0.643 | 0.816 |
| B | translate | mean | S_diff | 60 | 0.805 | 0.742 | 0.859 |
| B | translate | vanilla | S | 60 | 0.734 | 0.640 | 0.819 |
| B | translate | vanilla | S_diff | 60 | 0.805 | 0.744 | 0.856 |
| B | translate_ctx | affine | S | 60 | 0.595 | 0.431 | 0.722 |
| B | translate_ctx | affine | S_diff | 60 | 0.742 | 0.631 | 0.832 |
| B | translate_ctx | mean | S | 60 | 0.654 | 0.492 | 0.777 |
| B | translate_ctx | mean | S_diff | 60 | 0.742 | 0.622 | 0.840 |
| B | translate_ctx | vanilla | S | 60 | 0.605 | 0.417 | 0.752 |
| B | translate_ctx | vanilla | S_diff | 60 | 0.728 | 0.583 | 0.839 |

## 3. 판정 (A, 번역 과제)

'크게 줄었음' = 감소율 ≥ 0.5 (S 기준 확률). '상관 하락' = Δr의 95% CI 상한 < 0. 바닐라 상관의 CI 하한이 0 이하이면 판정하지 않는다.

| x | 렌즈 | 바닐라 r [CI] | 영어 확률(바닐라) | 영어 확률(보정) | 감소율 [CI] | Δr [CI] | 판정 |
|---|---|---|---|---|---|---|---|
| S | affine | 0.047 [-0.150, 0.237] | 0.390 | 0.167 | 0.572 [0.464, 0.665] | -0.035 [-0.129, 0.059] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |
| S | mean | 0.047 [-0.150, 0.237] | 0.390 | 0.416 | -0.066 [-0.104, -0.034] | 0.001 [-0.037, 0.039] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |
| S_diff | affine | -0.045 [-0.242, 0.161] | 0.390 | 0.167 | 0.572 [0.464, 0.665] | 0.070 [-0.035, 0.192] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |
| S_diff | mean | -0.045 [-0.242, 0.161] | 0.390 | 0.416 | -0.066 [-0.104, -0.034] | 0.012 [-0.014, 0.041] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |

## 4. 음성 대조: B(한국어 동음이의어)

B의 간섭은 영어와 무관하므로 r_B ≈ 0이어야 한다. r_B도 크면 렌즈 점수가 영어가 아닌 '헷갈리는 정도'를 재고 있을 가능성이 있다. translate_ctx는 예문으로 뜻을 정해 준 조건이다(시장에서 배/복숭아 한 상자를 샀다). 이때 간섭은 '뜻이 정해졌는데도 같은 표기 때문에 남는 다른 뜻'이므로 A(한국어로는 뜻이 하나, 영어 다의어로 새는 방해어)와 구조가 맞는다.

| B 과제 | 렌즈 | n_B | B 평균 간섭 | B 평균 영어 확률 | r_A | r_B | r_A - r_B [CI] |
|---|---|---|---|---|---|---|---|
| translate | affine | 60 | 9.466 | 0.099 | 0.012 | 0.677 | -0.665 [-0.926, -0.396] |
| translate | mean | 60 | 9.466 | 0.235 | 0.047 | 0.735 | -0.687 [-0.896, -0.476] |
| translate | vanilla | 60 | 9.466 | 0.219 | 0.047 | 0.734 | -0.687 [-0.908, -0.469] |
| translate_ctx | affine | 60 | 5.042 | 0.010 | 0.012 | 0.595 | -0.584 [-0.828, -0.321] |
| translate_ctx | mean | 60 | 5.042 | 0.031 | 0.047 | 0.654 | -0.607 [-0.842, -0.346] |
| translate_ctx | vanilla | 60 | 5.042 | 0.029 | 0.047 | 0.605 | -0.558 [-0.808, -0.281] |

### 4b. B 뜻 빈도 비율 (국립국어원 어휘 의미 분석 말뭉치)

방해 뜻이 흔할수록 간섭이 커야 정상이다(eDom의 한국어 대응). 두 뜻 합이 적은 단어(dom_ok=False)는 뺐다.

| B 과제 | 렌즈 | n | r(방해 뜻 우세도, I) [CI] | r(S, I) [CI] (같은 문항) |
|---|---|---|---|---|
| translate | affine | 38 | 0.444 [0.067, 0.721] | 0.628 [0.401, 0.816] |
| translate | mean | 38 | 0.444 [0.067, 0.721] | 0.723 [0.607, 0.832] |
| translate | vanilla | 38 | 0.444 [0.067, 0.721] | 0.708 [0.576, 0.828] |
| translate_ctx | affine | 38 | 0.317 [-0.029, 0.613] | 0.523 [0.294, 0.707] |
| translate_ctx | mean | 38 | 0.317 [-0.029, 0.613] | 0.608 [0.400, 0.769] |
| translate_ctx | vanilla | 38 | 0.317 [-0.029, 0.613] | 0.540 [0.266, 0.755] |

## 5. 어종별 (8문항 이상인 집단만)

한자어는 한자음을 거쳐 중국어로 바로 가는 지름길이 있을 수 있다.

| dataset | task | lens | etym | n | r | ci_lo | ci_hi | mean_I | mean_p |
|---|---|---|---|---|---|---|---|---|---|
| A | translate | affine | 고유어 | 34 | -0.202 | -0.559 | 0.204 | 0.708 | 0.116 |
| A | translate | affine | 모호 | 12 | -0.189 | -0.814 | 0.518 | 0.916 | 0.259 |
| A | translate | affine | 외래어 | 13 | 0.115 | -0.507 | 0.720 | 2.945 | 0.280 |
| A | translate | affine | 한자어 | 35 | 0.231 | -0.135 | 0.555 | 1.531 | 0.167 |
| A | translate | mean | 고유어 | 34 | -0.106 | -0.412 | 0.226 | 0.708 | 0.429 |
| A | translate | mean | 모호 | 12 | -0.175 | -0.842 | 0.528 | 0.916 | 0.501 |
| A | translate | mean | 외래어 | 13 | -0.093 | -0.709 | 0.622 | 2.945 | 0.490 |
| A | translate | mean | 한자어 | 35 | 0.265 | -0.110 | 0.574 | 1.531 | 0.391 |
| A | translate | vanilla | 고유어 | 34 | -0.091 | -0.413 | 0.249 | 0.708 | 0.395 |
| A | translate | vanilla | 모호 | 12 | -0.231 | -0.834 | 0.470 | 0.916 | 0.493 |
| A | translate | vanilla | 외래어 | 13 | -0.011 | -0.631 | 0.669 | 2.945 | 0.452 |
| A | translate | vanilla | 한자어 | 35 | 0.213 | -0.203 | 0.546 | 1.531 | 0.368 |
| B | translate | affine | 고유어쌍 | 12 | 0.804 | 0.681 | 0.943 | 7.592 | 0.108 |
| B | translate | affine | 한자어쌍 | 40 | 0.653 | 0.457 | 0.831 | 10.088 | 0.114 |
| B | translate | affine | 혼합 | 8 | 0.690 | -0.105 | 1.000 | 9.163 | 0.011 |
| B | translate | mean | 고유어쌍 | 12 | 0.811 | 0.681 | 0.978 | 7.592 | 0.268 |
| B | translate | mean | 한자어쌍 | 40 | 0.748 | 0.635 | 0.847 | 10.088 | 0.248 |
| B | translate | mean | 혼합 | 8 | 0.548 | -0.263 | 1.000 | 9.163 | 0.118 |
| B | translate | vanilla | 고유어쌍 | 12 | 0.783 | 0.638 | 0.907 | 7.592 | 0.232 |
| B | translate | vanilla | 한자어쌍 | 40 | 0.734 | 0.600 | 0.846 | 10.088 | 0.240 |
| B | translate | vanilla | 혼합 | 8 | 0.524 | -0.263 | 1.000 | 9.163 | 0.093 |
| B | translate_ctx | affine | 고유어쌍 | 12 | 0.692 | 0.314 | 0.820 | 3.802 | 0.011 |
| B | translate_ctx | affine | 한자어쌍 | 40 | 0.580 | 0.390 | 0.756 | 5.829 | 0.012 |
| B | translate_ctx | affine | 혼합 | 8 | 0.119 | -0.579 | 0.737 | 2.970 | 0.000 |
| B | translate_ctx | mean | 고유어쌍 | 12 | 0.776 | 0.496 | 0.943 | 3.802 | 0.058 |
| B | translate_ctx | mean | 한자어쌍 | 40 | 0.714 | 0.548 | 0.846 | 5.829 | 0.029 |
| B | translate_ctx | mean | 혼합 | 8 | -0.143 | -0.842 | 0.579 | 2.970 | 0.002 |
| B | translate_ctx | vanilla | 고유어쌍 | 12 | 0.713 | 0.394 | 0.886 | 3.802 | 0.054 |
| B | translate_ctx | vanilla | 한자어쌍 | 40 | 0.657 | 0.451 | 0.831 | 5.829 | 0.027 |
| B | translate_ctx | vanilla | 혼합 | 8 | -0.167 | -0.842 | 0.737 | 2.970 | 0.001 |

## 6. 양성 대조: 프롬프트에 영어 중간 단계

영어 경유를 강제했으므로 translate_en에서 간섭·영어 확률이 커지고 상관이 보여야 한다. 그렇지 않으면 간섭 지표가 영어 경유를 잡지 못하는 것이다.

| 과제 | 렌즈 | 평균 간섭 | 평균 영어 확률 | r [CI] |
|---|---|---|---|---|
| translate | affine | 1.331 | 0.167 | 0.012 [-0.188, 0.218] |
| translate | mean | 1.331 | 0.416 | 0.047 [-0.143, 0.236] |
| translate | vanilla | 1.331 | 0.390 | 0.047 [-0.150, 0.237] |
| translate_en | affine | 2.949 | 0.268 | -0.094 [-0.293, 0.109] |
| translate_en | mean | 2.949 | 0.623 | -0.003 [-0.198, 0.194] |
| translate_en | vanilla | 2.949 | 0.589 | -0.007 [-0.213, 0.201] |

## 8. 층 구간별 언어 질량 (입력어 프롬프트 평균 확률)

`en`은 라틴 문자 2자 이상 토큰 전부(단어 조각 포함), `en_word`는 단어 시작 토큰이면서 영어 단어 목록에 있는 토큰만 센다. 둘의 차이가 크면 '영어'로 보이던 것의 상당 부분이 단어 조각이다. 구간은 렌즈 점수 구간 설정과 무관하게 전체 층의 1/4, 3/4로 나눈다.

| dataset | task | lens | band | en | en_word | zh | ko |
|---|---|---|---|---|---|---|---|
| A | translate | affine | 초반 1–7 | 0.050 | 0.018 | 0.184 | 0.113 |
| A | translate | affine | 중간 8–22 | 0.027 | 0.009 | 0.200 | 0.011 |
| A | translate | affine | 후반 23–29 | 0.143 | 0.108 | 0.258 | 0.040 |
| A | translate | mean | 초반 1–7 | 0.314 | 0.162 | 0.006 | 0.129 |
| A | translate | mean | 중간 8–22 | 0.638 | 0.309 | 0.012 | 0.274 |
| A | translate | mean | 후반 23–29 | 0.774 | 0.592 | 0.020 | 0.164 |
| A | translate | vanilla | 초반 1–7 | 0.334 | 0.179 | 0.006 | 0.128 |
| A | translate | vanilla | 중간 8–22 | 0.641 | 0.323 | 0.014 | 0.273 |
| A | translate | vanilla | 후반 23–29 | 0.754 | 0.565 | 0.024 | 0.169 |
| A | translate_en | affine | 초반 1–7 | 0.051 | 0.017 | 0.202 | 0.104 |
| A | translate_en | affine | 중간 8–22 | 0.028 | 0.011 | 0.219 | 0.012 |
| A | translate_en | affine | 후반 23–29 | 0.185 | 0.140 | 0.234 | 0.034 |
| A | translate_en | mean | 초반 1–7 | 0.323 | 0.165 | 0.006 | 0.129 |
| A | translate_en | mean | 중간 8–22 | 0.634 | 0.308 | 0.012 | 0.274 |
| A | translate_en | mean | 후반 23–29 | 0.821 | 0.652 | 0.015 | 0.127 |
| A | translate_en | vanilla | 초반 1–7 | 0.343 | 0.183 | 0.006 | 0.130 |
| A | translate_en | vanilla | 중간 8–22 | 0.639 | 0.323 | 0.013 | 0.274 |
| A | translate_en | vanilla | 후반 23–29 | 0.799 | 0.624 | 0.018 | 0.135 |
| B | translate | affine | 초반 1–7 | 0.051 | 0.018 | 0.186 | 0.112 |
| B | translate | affine | 중간 8–22 | 0.028 | 0.010 | 0.202 | 0.012 |
| B | translate | affine | 후반 23–29 | 0.132 | 0.101 | 0.382 | 0.054 |
| B | translate | mean | 초반 1–7 | 0.317 | 0.163 | 0.006 | 0.130 |
| B | translate | mean | 중간 8–22 | 0.642 | 0.307 | 0.012 | 0.273 |
| B | translate | mean | 후반 23–29 | 0.696 | 0.546 | 0.046 | 0.219 |
| B | translate | vanilla | 초반 1–7 | 0.336 | 0.180 | 0.006 | 0.128 |
| B | translate | vanilla | 중간 8–22 | 0.644 | 0.321 | 0.013 | 0.272 |
| B | translate | vanilla | 후반 23–29 | 0.682 | 0.526 | 0.050 | 0.222 |
| B | translate_ctx | affine | 초반 1–7 | 0.055 | 0.023 | 0.133 | 0.195 |
| B | translate_ctx | affine | 중간 8–22 | 0.034 | 0.013 | 0.235 | 0.015 |
| B | translate_ctx | affine | 후반 23–29 | 0.123 | 0.100 | 0.345 | 0.069 |
| B | translate_ctx | mean | 초반 1–7 | 0.318 | 0.181 | 0.006 | 0.126 |
| B | translate_ctx | mean | 중간 8–22 | 0.634 | 0.334 | 0.012 | 0.276 |
| B | translate_ctx | mean | 후반 23–29 | 0.695 | 0.560 | 0.029 | 0.236 |
| B | translate_ctx | vanilla | 초반 1–7 | 0.337 | 0.197 | 0.006 | 0.126 |
| B | translate_ctx | vanilla | 중간 8–22 | 0.637 | 0.347 | 0.013 | 0.276 |
| B | translate_ctx | vanilla | 후반 23–29 | 0.676 | 0.539 | 0.034 | 0.241 |

## 부록: 렌즈 적합도 (held-out KL(최종 || 렌즈), 중간층 평균)

아핀 렌즈가 바닐라보다 KL이 작아야 기저 맞춤이 실제로 된 것이다.

| lens | kl |
|---|---|
| affine | 0.744 |
| mean | 4.374 |
| vanilla | 3.944 |
