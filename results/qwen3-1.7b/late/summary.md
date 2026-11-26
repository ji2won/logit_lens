# 결과 요약 (late)

층 수 28, 렌즈 점수 구간 21–27층, bootstrap 2000회

## 1. 신뢰도 (프롬프트 4개, Cronbach's alpha)

기준 0.7. 미달한 지표의 상관은 '과장'으로 해석하지 않는다.

| dataset | task | measure | alpha | 통과 |
|---|---|---|---|---|
| A | translate | 간섭 I | 0.960 | O |
| A | translate_en | 간섭 I | 0.975 | O |
| B | translate | 간섭 I | 0.979 | O |
| B | translate_ctx | 간섭 I | 0.983 | O |
| A | translate | 렌즈 S (affine) | 0.975 | O |
| A | translate | 렌즈 S_diff (affine) | 0.985 | O |
| A | translate | 렌즈 S (mean) | 0.985 | O |
| A | translate | 렌즈 S_diff (mean) | 0.990 | O |
| A | translate | 렌즈 S (vanilla) | 0.988 | O |
| A | translate | 렌즈 S_diff (vanilla) | 0.990 | O |
| A | translate_en | 렌즈 S (affine) | 0.973 | O |
| A | translate_en | 렌즈 S_diff (affine) | 0.986 | O |
| A | translate_en | 렌즈 S (mean) | 0.971 | O |
| A | translate_en | 렌즈 S_diff (mean) | 0.990 | O |
| A | translate_en | 렌즈 S (vanilla) | 0.975 | O |
| A | translate_en | 렌즈 S_diff (vanilla) | 0.988 | O |
| B | translate | 렌즈 S (affine) | 0.974 | O |
| B | translate | 렌즈 S_diff (affine) | 0.977 | O |
| B | translate | 렌즈 S (mean) | 0.978 | O |
| B | translate | 렌즈 S_diff (mean) | 0.985 | O |
| B | translate | 렌즈 S (vanilla) | 0.982 | O |
| B | translate | 렌즈 S_diff (vanilla) | 0.985 | O |
| B | translate_ctx | 렌즈 S (affine) | 0.965 | O |
| B | translate_ctx | 렌즈 S_diff (affine) | 0.974 | O |
| B | translate_ctx | 렌즈 S (mean) | 0.983 | O |
| B | translate_ctx | 렌즈 S_diff (mean) | 0.990 | O |
| B | translate_ctx | 렌즈 S (vanilla) | 0.989 | O |
| B | translate_ctx | 렌즈 S_diff (vanilla) | 0.987 | O |
| C | repeat | 렌즈 S (affine) | 0.945 | O |
| C | repeat | 렌즈 S_diff (affine) | nan | X |
| C | repeat | 렌즈 S (mean) | 0.966 | O |
| C | repeat | 렌즈 S_diff (mean) | nan | X |
| C | repeat | 렌즈 S (vanilla) | 0.970 | O |
| C | repeat | 렌즈 S_diff (vanilla) | nan | X |
| C | translate | 렌즈 S (affine) | 0.961 | O |
| C | translate | 렌즈 S_diff (affine) | nan | X |
| C | translate | 렌즈 S (mean) | 0.981 | O |
| C | translate | 렌즈 S_diff (mean) | nan | X |
| C | translate | 렌즈 S (vanilla) | 0.978 | O |
| C | translate | 렌즈 S_diff (vanilla) | nan | X |
| C | translate_to_en | 렌즈 S (affine) | 0.966 | O |
| C | translate_to_en | 렌즈 S_diff (affine) | nan | X |
| C | translate_to_en | 렌즈 S (mean) | 0.967 | O |
| C | translate_to_en | 렌즈 S_diff (mean) | nan | X |
| C | translate_to_en | 렌즈 S (vanilla) | 0.975 | O |
| C | translate_to_en | 렌즈 S_diff (vanilla) | nan | X |

## 2. 렌즈 점수 × 간섭 상관 (Spearman, 문항 단위)

| dataset | task | lens | x | n | r | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| A | translate | affine | S | 99 | 0.180 | -0.030 | 0.367 |
| A | translate | affine | S_diff | 99 | 0.266 | 0.073 | 0.446 |
| A | translate | mean | S | 99 | 0.119 | -0.072 | 0.299 |
| A | translate | mean | S_diff | 99 | 0.266 | 0.070 | 0.445 |
| A | translate | vanilla | S | 99 | 0.036 | -0.165 | 0.223 |
| A | translate | vanilla | S_diff | 99 | 0.221 | 0.028 | 0.398 |
| A | translate_en | affine | S | 99 | 0.328 | 0.160 | 0.480 |
| A | translate_en | affine | S_diff | 99 | 0.450 | 0.275 | 0.602 |
| A | translate_en | mean | S | 99 | 0.255 | 0.075 | 0.427 |
| A | translate_en | mean | S_diff | 99 | 0.451 | 0.274 | 0.596 |
| A | translate_en | vanilla | S | 99 | 0.227 | 0.040 | 0.402 |
| A | translate_en | vanilla | S_diff | 99 | 0.457 | 0.283 | 0.601 |
| B | translate | affine | S | 60 | 0.603 | 0.447 | 0.718 |
| B | translate | affine | S_diff | 60 | 0.849 | 0.740 | 0.914 |
| B | translate | mean | S | 60 | 0.770 | 0.677 | 0.847 |
| B | translate | mean | S_diff | 60 | 0.883 | 0.806 | 0.930 |
| B | translate | vanilla | S | 60 | 0.751 | 0.651 | 0.826 |
| B | translate | vanilla | S_diff | 60 | 0.849 | 0.753 | 0.915 |
| B | translate_ctx | affine | S | 60 | 0.640 | 0.489 | 0.766 |
| B | translate_ctx | affine | S_diff | 60 | 0.881 | 0.771 | 0.936 |
| B | translate_ctx | mean | S | 60 | 0.728 | 0.552 | 0.840 |
| B | translate_ctx | mean | S_diff | 60 | 0.878 | 0.777 | 0.931 |
| B | translate_ctx | vanilla | S | 60 | 0.720 | 0.571 | 0.826 |
| B | translate_ctx | vanilla | S_diff | 60 | 0.828 | 0.694 | 0.901 |

## 3. 판정 (A, 번역 과제)

'크게 줄었음' = 감소율 ≥ 0.5 (S 기준 확률). '상관 하락' = Δr의 95% CI 상한 < 0. 바닐라 상관의 CI 하한이 0 이하이면 판정하지 않는다.

| x | 렌즈 | 바닐라 r [CI] | 영어 확률(바닐라) | 영어 확률(보정) | 감소율 [CI] | Δr [CI] | 판정 |
|---|---|---|---|---|---|---|---|
| S | affine | 0.036 [-0.165, 0.223] | 0.037 | 0.002 | 0.939 [0.896, 0.975] | 0.144 [0.009, 0.273] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |
| S | mean | 0.036 [-0.165, 0.223] | 0.037 | 0.040 | -0.075 [-0.464, 0.373] | 0.083 [-0.012, 0.188] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |
| S_diff | affine | 0.221 [0.028, 0.398] | 0.037 | 0.002 | 0.939 [0.896, 0.975] | 0.044 [-0.031, 0.125] | 크게 줄고 상관은 그대로 → 지운 영어는 부풀려진 영어였다 (가설 지지) |
| S_diff | mean | 0.221 [0.028, 0.398] | 0.037 | 0.040 | -0.075 [-0.464, 0.373] | 0.045 [-0.021, 0.113] | 거의 안 줄었음 → 그 원인은 영어를 만들지 않았다 (가설에 불리) |

## 4. 음성 대조: B(한국어 동음이의어)

B의 간섭은 영어와 무관하므로 r_B ≈ 0이어야 한다. r_B도 크면 렌즈 점수가 영어가 아닌 '헷갈리는 정도'를 재고 있을 가능성이 있다. translate_ctx는 예문으로 뜻을 정해 준 조건이다(시장에서 배/복숭아 한 상자를 샀다). 이때 간섭은 '뜻이 정해졌는데도 같은 표기 때문에 남는 다른 뜻'이므로 A(한국어로는 뜻이 하나, 영어 다의어로 새는 방해어)와 구조가 맞는다.

| B 과제 | 렌즈 | n_B | B 평균 간섭 | B 평균 영어 확률 | r_A | r_B | r_A - r_B [CI] |
|---|---|---|---|---|---|---|---|
| translate | affine | 60 | 16.435 | 0.000 | 0.180 | 0.603 | -0.423 [-0.664, -0.191] |
| translate | mean | 60 | 16.435 | 0.005 | 0.119 | 0.770 | -0.651 [-0.864, -0.439] |
| translate | vanilla | 60 | 16.435 | 0.004 | 0.036 | 0.751 | -0.715 [-0.926, -0.496] |
| translate_ctx | affine | 60 | 10.965 | 0.000 | 0.180 | 0.640 | -0.460 [-0.706, -0.219] |
| translate_ctx | mean | 60 | 10.965 | 0.001 | 0.119 | 0.728 | -0.608 [-0.839, -0.366] |
| translate_ctx | vanilla | 60 | 10.965 | 0.002 | 0.036 | 0.720 | -0.684 [-0.918, -0.450] |

### 4b. B 뜻 빈도 비율 (국립국어원 어휘 의미 분석 말뭉치)

방해 뜻이 흔할수록 간섭이 커야 정상이다(eDom의 한국어 대응). 두 뜻 합이 적은 단어(dom_ok=False)는 뺐다.

| B 과제 | 렌즈 | n | r(방해 뜻 우세도, I) [CI] | r(S, I) [CI] (같은 문항) |
|---|---|---|---|---|
| translate | affine | 38 | 0.259 [-0.103, 0.528] | 0.672 [0.503, 0.796] |
| translate | mean | 38 | 0.259 [-0.103, 0.528] | 0.753 [0.607, 0.867] |
| translate | vanilla | 38 | 0.259 [-0.103, 0.528] | 0.719 [0.574, 0.838] |
| translate_ctx | affine | 38 | 0.381 [-0.014, 0.666] | 0.652 [0.369, 0.830] |
| translate_ctx | mean | 38 | 0.381 [-0.014, 0.666] | 0.723 [0.437, 0.872] |
| translate_ctx | vanilla | 38 | 0.381 [-0.014, 0.666] | 0.685 [0.416, 0.854] |

## 5. 어종별 (8문항 이상인 집단만)

한자어는 한자음을 거쳐 중국어로 바로 가는 지름길이 있을 수 있다.

| dataset | task | lens | etym | n | r | ci_lo | ci_hi | mean_I | mean_p |
|---|---|---|---|---|---|---|---|---|---|
| A | translate | affine | 고유어 | 34 | 0.173 | -0.197 | 0.507 | 3.206 | 0.001 |
| A | translate | affine | 모호 | 12 | 0.587 | -0.000 | 0.926 | 4.234 | 0.000 |
| A | translate | affine | 외래어 | 13 | 0.214 | -0.483 | 0.760 | 3.947 | 0.008 |
| A | translate | affine | 한자어 | 35 | 0.185 | -0.159 | 0.491 | 5.195 | 0.003 |
| A | translate | mean | 고유어 | 34 | 0.255 | -0.100 | 0.566 | 3.206 | 0.004 |
| A | translate | mean | 모호 | 12 | 0.231 | -0.345 | 0.702 | 4.234 | 0.001 |
| A | translate | mean | 외래어 | 13 | 0.231 | -0.413 | 0.754 | 3.947 | 0.117 |
| A | translate | mean | 한자어 | 35 | -0.006 | -0.338 | 0.315 | 5.195 | 0.066 |
| A | translate | vanilla | 고유어 | 34 | 0.095 | -0.290 | 0.442 | 3.206 | 0.008 |
| A | translate | vanilla | 모호 | 12 | -0.070 | -0.609 | 0.500 | 4.234 | 0.007 |
| A | translate | vanilla | 외래어 | 13 | -0.121 | -0.756 | 0.462 | 3.947 | 0.146 |
| A | translate | vanilla | 한자어 | 35 | 0.010 | -0.316 | 0.337 | 5.195 | 0.040 |
| B | translate | affine | 고유어쌍 | 12 | 0.566 | 0.165 | 0.872 | 14.550 | 0.000 |
| B | translate | affine | 한자어쌍 | 40 | 0.662 | 0.497 | 0.789 | 17.199 | 0.000 |
| B | translate | affine | 혼합 | 8 | 0.500 | -0.263 | 0.842 | 15.438 | 0.000 |
| B | translate | mean | 고유어쌍 | 12 | 0.720 | 0.316 | 0.935 | 14.550 | 0.002 |
| B | translate | mean | 한자어쌍 | 40 | 0.777 | 0.657 | 0.867 | 17.199 | 0.007 |
| B | translate | mean | 혼합 | 8 | 0.619 | -0.105 | 1.000 | 15.438 | 0.000 |
| B | translate | vanilla | 고유어쌍 | 12 | 0.727 | 0.324 | 0.921 | 14.550 | 0.005 |
| B | translate | vanilla | 한자어쌍 | 40 | 0.758 | 0.642 | 0.852 | 17.199 | 0.004 |
| B | translate | vanilla | 혼합 | 8 | 0.810 | 0.289 | 1.000 | 15.438 | 0.003 |
| B | translate_ctx | affine | 고유어쌍 | 12 | 0.636 | 0.511 | 0.935 | 11.518 | 0.000 |
| B | translate_ctx | affine | 한자어쌍 | 40 | 0.670 | 0.459 | 0.812 | 11.339 | 0.000 |
| B | translate_ctx | affine | 혼합 | 8 | 0.381 | -0.263 | 0.800 | 8.263 | 0.000 |
| B | translate_ctx | mean | 고유어쌍 | 12 | 0.790 | 0.338 | 0.978 | 11.518 | 0.003 |
| B | translate_ctx | mean | 한자어쌍 | 40 | 0.750 | 0.561 | 0.874 | 11.339 | 0.001 |
| B | translate_ctx | mean | 혼합 | 8 | 0.405 | -0.684 | 0.974 | 8.263 | 0.000 |
| B | translate_ctx | vanilla | 고유어쌍 | 12 | 0.825 | 0.655 | 0.978 | 11.518 | 0.003 |
| B | translate_ctx | vanilla | 한자어쌍 | 40 | 0.730 | 0.543 | 0.856 | 11.339 | 0.002 |
| B | translate_ctx | vanilla | 혼합 | 8 | 0.643 | -0.211 | 0.974 | 8.263 | 0.000 |

## 6. 양성 대조: 프롬프트에 영어 중간 단계

영어 경유를 강제했으므로 translate_en에서 간섭·영어 확률이 커지고 상관이 보여야 한다. 그렇지 않으면 간섭 지표가 영어 경유를 잡지 못하는 것이다.

| 과제 | 렌즈 | 평균 간섭 | 평균 영어 확률 | r [CI] |
|---|---|---|---|---|
| translate | affine | 4.147 | 0.002 | 0.180 [-0.030, 0.367] |
| translate | mean | 4.147 | 0.040 | 0.119 [-0.072, 0.299] |
| translate | vanilla | 4.147 | 0.037 | 0.036 [-0.165, 0.223] |
| translate_en | affine | 9.321 | 0.029 | 0.328 [0.160, 0.480] |
| translate_en | mean | 9.321 | 0.167 | 0.255 [0.075, 0.427] |
| translate_en | vanilla | 9.321 | 0.159 | 0.227 [0.040, 0.402] |

## 7. C: 중간층 평균 확률 (정답 영어 vs 무관 영어, 언어별 확률 질량)

가설 1이 맞다면 바닐라에서는 무관한 영어(chair)와 영어 질량 전체가 높고, 평균 제거 렌즈에서 줄어야 한다.

| task | lens | lp_target | lp_unrelated | mass_en | mass_en_word | mass_zh | mass_ko |
|---|---|---|---|---|---|---|---|
| repeat | affine | 0.0003 | 0.0000 | 0.0033 | 0.0006 | 0.0022 | 0.9480 |
| repeat | mean | 0.0633 | 0.0000 | 0.2807 | 0.0155 | 0.1362 | 0.3885 |
| repeat | vanilla | 0.0253 | 0.0000 | 0.0796 | 0.0083 | 0.1414 | 0.4364 |
| translate | affine | 0.0042 | 0.0000 | 0.0057 | 0.0041 | 0.9917 | 0.0001 |
| translate | mean | 0.0391 | 0.0000 | 0.0717 | 0.0655 | 0.9274 | 0.0000 |
| translate | vanilla | 0.0263 | 0.0000 | 0.0411 | 0.0372 | 0.9534 | 0.0001 |
| translate_to_en | affine | 0.4130 | 0.0005 | 0.6890 | 0.1249 | 0.0492 | 0.0004 |
| translate_to_en | mean | 0.5395 | 0.0000 | 0.8597 | 0.6446 | 0.1302 | 0.0000 |
| translate_to_en | vanilla | 0.4924 | 0.0002 | 0.7666 | 0.5858 | 0.1442 | 0.0001 |

## 8. 층 구간별 언어 질량 (입력어 프롬프트 평균 확률)

`en`은 라틴 문자 2자 이상 토큰 전부(단어 조각 포함), `en_word`는 단어 시작 토큰이면서 영어 단어 목록에 있는 토큰만 센다. 둘의 차이가 크면 '영어'로 보이던 것의 상당 부분이 단어 조각이다. 구간은 렌즈 점수 구간 설정과 무관하게 전체 층의 1/4, 3/4로 나눈다.

| dataset | task | lens | band | en | en_word | zh | ko |
|---|---|---|---|---|---|---|---|
| A | translate | affine | 초반 1–6 | 0.068 | 0.017 | 0.264 | 0.224 |
| A | translate | affine | 중간 7–21 | 0.020 | 0.006 | 0.354 | 0.030 |
| A | translate | affine | 후반 22–27 | 0.003 | 0.002 | 0.994 | 0.000 |
| A | translate | mean | 초반 1–6 | 0.242 | 0.021 | 0.484 | 0.005 |
| A | translate | mean | 중간 7–21 | 0.293 | 0.039 | 0.587 | 0.006 |
| A | translate | mean | 후반 22–27 | 0.048 | 0.043 | 0.952 | 0.000 |
| A | translate | vanilla | 초반 1–6 | 0.532 | 0.011 | 0.080 | 0.002 |
| A | translate | vanilla | 중간 7–21 | 0.274 | 0.024 | 0.337 | 0.002 |
| A | translate | vanilla | 후반 22–27 | 0.038 | 0.033 | 0.954 | 0.000 |
| A | translate_en | affine | 초반 1–6 | 0.061 | 0.015 | 0.255 | 0.181 |
| A | translate_en | affine | 중간 7–21 | 0.014 | 0.004 | 0.379 | 0.026 |
| A | translate_en | affine | 후반 22–27 | 0.019 | 0.011 | 0.978 | 0.000 |
| A | translate_en | mean | 초반 1–6 | 0.279 | 0.028 | 0.471 | 0.003 |
| A | translate_en | mean | 중간 7–21 | 0.248 | 0.039 | 0.627 | 0.003 |
| A | translate_en | mean | 후반 22–27 | 0.145 | 0.131 | 0.853 | 0.000 |
| A | translate_en | vanilla | 초반 1–6 | 0.557 | 0.010 | 0.088 | 0.002 |
| A | translate_en | vanilla | 중간 7–21 | 0.258 | 0.020 | 0.411 | 0.002 |
| A | translate_en | vanilla | 후반 22–27 | 0.111 | 0.099 | 0.880 | 0.000 |
| B | translate | affine | 초반 1–6 | 0.069 | 0.017 | 0.264 | 0.223 |
| B | translate | affine | 중간 7–21 | 0.021 | 0.006 | 0.371 | 0.031 |
| B | translate | affine | 후반 22–27 | 0.000 | 0.000 | 0.996 | 0.000 |
| B | translate | mean | 초반 1–6 | 0.246 | 0.022 | 0.481 | 0.005 |
| B | translate | mean | 중간 7–21 | 0.277 | 0.038 | 0.611 | 0.006 |
| B | translate | mean | 후반 22–27 | 0.005 | 0.004 | 0.995 | 0.000 |
| B | translate | vanilla | 초반 1–6 | 0.540 | 0.011 | 0.077 | 0.002 |
| B | translate | vanilla | 중간 7–21 | 0.273 | 0.023 | 0.370 | 0.002 |
| B | translate | vanilla | 후반 22–27 | 0.010 | 0.008 | 0.982 | 0.001 |
| B | translate_ctx | affine | 초반 1–6 | 0.073 | 0.022 | 0.218 | 0.291 |
| B | translate_ctx | affine | 중간 7–21 | 0.017 | 0.006 | 0.259 | 0.069 |
| B | translate_ctx | affine | 후반 22–27 | 0.001 | 0.000 | 0.996 | 0.000 |
| B | translate_ctx | mean | 초반 1–6 | 0.259 | 0.021 | 0.452 | 0.004 |
| B | translate_ctx | mean | 중간 7–21 | 0.321 | 0.061 | 0.570 | 0.002 |
| B | translate_ctx | mean | 후반 22–27 | 0.018 | 0.016 | 0.982 | 0.000 |
| B | translate_ctx | vanilla | 초반 1–6 | 0.619 | 0.008 | 0.051 | 0.002 |
| B | translate_ctx | vanilla | 중간 7–21 | 0.367 | 0.017 | 0.303 | 0.002 |
| B | translate_ctx | vanilla | 후반 22–27 | 0.012 | 0.010 | 0.981 | 0.000 |
| C | repeat | affine | 초반 1–6 | 0.077 | 0.021 | 0.209 | 0.249 |
| C | repeat | affine | 중간 7–21 | 0.041 | 0.012 | 0.110 | 0.260 |
| C | repeat | affine | 후반 22–27 | 0.003 | 0.001 | 0.001 | 0.971 |
| C | repeat | mean | 초반 1–6 | 0.268 | 0.027 | 0.424 | 0.004 |
| C | repeat | mean | 중간 7–21 | 0.356 | 0.058 | 0.409 | 0.005 |
| C | repeat | mean | 후반 22–27 | 0.299 | 0.016 | 0.148 | 0.451 |
| C | repeat | vanilla | 초반 1–6 | 0.560 | 0.011 | 0.059 | 0.003 |
| C | repeat | vanilla | 중간 7–21 | 0.349 | 0.024 | 0.105 | 0.002 |
| C | repeat | vanilla | 후반 22–27 | 0.091 | 0.010 | 0.163 | 0.509 |
| C | translate | affine | 초반 1–6 | 0.068 | 0.017 | 0.265 | 0.223 |
| C | translate | affine | 중간 7–21 | 0.020 | 0.006 | 0.345 | 0.030 |
| C | translate | affine | 후반 22–27 | 0.006 | 0.005 | 0.992 | 0.000 |
| C | translate | mean | 초반 1–6 | 0.244 | 0.021 | 0.482 | 0.005 |
| C | translate | mean | 중간 7–21 | 0.295 | 0.046 | 0.585 | 0.006 |
| C | translate | mean | 후반 22–27 | 0.067 | 0.062 | 0.932 | 0.000 |
| C | translate | vanilla | 초반 1–6 | 0.532 | 0.011 | 0.079 | 0.002 |
| C | translate | vanilla | 중간 7–21 | 0.277 | 0.026 | 0.342 | 0.002 |
| C | translate | vanilla | 후반 22–27 | 0.041 | 0.038 | 0.955 | 0.000 |
| C | translate_to_en | affine | 초반 1–6 | 0.080 | 0.018 | 0.217 | 0.239 |
| C | translate_to_en | affine | 중간 7–21 | 0.053 | 0.007 | 0.081 | 0.026 |
| C | translate_to_en | affine | 후반 22–27 | 0.765 | 0.142 | 0.056 | 0.000 |
| C | translate_to_en | mean | 초반 1–6 | 0.304 | 0.032 | 0.388 | 0.006 |
| C | translate_to_en | mean | 중간 7–21 | 0.460 | 0.097 | 0.377 | 0.002 |
| C | translate_to_en | mean | 후반 22–27 | 0.865 | 0.644 | 0.127 | 0.000 |
| C | translate_to_en | vanilla | 초반 1–6 | 0.610 | 0.008 | 0.040 | 0.002 |
| C | translate_to_en | vanilla | 중간 7–21 | 0.445 | 0.048 | 0.114 | 0.002 |
| C | translate_to_en | vanilla | 후반 22–27 | 0.787 | 0.611 | 0.145 | 0.000 |

## 부록: 렌즈 적합도 (held-out KL(최종 || 렌즈), 중간층 평균)

아핀 렌즈가 바닐라보다 KL이 작아야 기저 맞춤이 실제로 된 것이다.

| lens | kl |
|---|---|
| affine | 0.890 |
| mean | 7.728 |
| vanilla | 3.712 |
