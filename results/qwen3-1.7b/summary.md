# 결과 요약 (qwen3-1.7b)

층 수 28, 렌즈 점수 구간 7–21층, bootstrap 2000회

## 1. 신뢰도 (프롬프트 4개, Cronbach's alpha)

기준 0.7. 미달한 지표의 상관은 '과장'으로 해석하지 않는다.

| dataset | task | measure | alpha | 통과 |
|---|---|---|---|---|
| A | translate | 간섭 I | 0.960 | O |
| A | translate_en | 간섭 I | 0.975 | O |
| B | translate | 간섭 I | 0.979 | O |
| B | translate_ctx | 간섭 I | 0.983 | O |
| A | translate | 렌즈 S (affine) | 0.966 | O |
| A | translate | 렌즈 S_diff (affine) | 0.950 | O |
| A | translate | 렌즈 S (mean) | 0.960 | O |
| A | translate | 렌즈 S_diff (mean) | 0.984 | O |
| A | translate | 렌즈 S (vanilla) | 0.953 | O |
| A | translate | 렌즈 S_diff (vanilla) | 0.984 | O |
| A | translate_en | 렌즈 S (affine) | 0.964 | O |
| A | translate_en | 렌즈 S_diff (affine) | 0.956 | O |
| A | translate_en | 렌즈 S (mean) | 0.931 | O |
| A | translate_en | 렌즈 S_diff (mean) | 0.967 | O |
| A | translate_en | 렌즈 S (vanilla) | 0.919 | O |
| A | translate_en | 렌즈 S_diff (vanilla) | 0.969 | O |
| B | translate | 렌즈 S (affine) | 0.963 | O |
| B | translate | 렌즈 S_diff (affine) | 0.960 | O |
| B | translate | 렌즈 S (mean) | 0.939 | O |
| B | translate | 렌즈 S_diff (mean) | 0.982 | O |
| B | translate | 렌즈 S (vanilla) | 0.905 | O |
| B | translate | 렌즈 S_diff (vanilla) | 0.982 | O |
| B | translate_ctx | 렌즈 S (affine) | 0.973 | O |
| B | translate_ctx | 렌즈 S_diff (affine) | 0.948 | O |
| B | translate_ctx | 렌즈 S (mean) | 0.910 | O |
| B | translate_ctx | 렌즈 S_diff (mean) | 0.981 | O |
| B | translate_ctx | 렌즈 S (vanilla) | 0.879 | O |
| B | translate_ctx | 렌즈 S_diff (vanilla) | 0.982 | O |
| C | repeat | 렌즈 S (affine) | 0.963 | O |
| C | repeat | 렌즈 S_diff (affine) | nan | X |
| C | repeat | 렌즈 S (mean) | 0.865 | O |
| C | repeat | 렌즈 S_diff (mean) | nan | X |
| C | repeat | 렌즈 S (vanilla) | 0.808 | O |
| C | repeat | 렌즈 S_diff (vanilla) | nan | X |
| C | translate | 렌즈 S (affine) | 0.926 | O |
| C | translate | 렌즈 S_diff (affine) | nan | X |
| C | translate | 렌즈 S (mean) | 0.970 | O |
| C | translate | 렌즈 S_diff (mean) | nan | X |
| C | translate | 렌즈 S (vanilla) | 0.947 | O |
| C | translate | 렌즈 S_diff (vanilla) | nan | X |
| C | translate_to_en | 렌즈 S (affine) | 0.962 | O |
| C | translate_to_en | 렌즈 S_diff (affine) | nan | X |
| C | translate_to_en | 렌즈 S (mean) | 0.977 | O |
| C | translate_to_en | 렌즈 S_diff (mean) | nan | X |
| C | translate_to_en | 렌즈 S (vanilla) | 0.969 | O |
| C | translate_to_en | 렌즈 S_diff (vanilla) | nan | X |

## 2. 렌즈 점수 × 간섭 상관 (Spearman, 문항 단위)

| dataset | task | lens | x | n | r | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| A | translate | affine | S | 99 | -0.112 | -0.292 | 0.090 |
| A | translate | affine | S_diff | 99 | 0.315 | 0.124 | 0.488 |
| A | translate | mean | S | 99 | 0.098 | -0.113 | 0.315 |
| A | translate | mean | S_diff | 99 | 0.331 | 0.145 | 0.491 |
| A | translate | vanilla | S | 99 | -0.033 | -0.228 | 0.165 |
| A | translate | vanilla | S_diff | 99 | 0.297 | 0.108 | 0.464 |
| A | translate_en | affine | S | 99 | 0.024 | -0.141 | 0.200 |
| A | translate_en | affine | S_diff | 99 | 0.506 | 0.334 | 0.639 |
| A | translate_en | mean | S | 99 | 0.279 | 0.077 | 0.453 |
| A | translate_en | mean | S_diff | 99 | 0.571 | 0.410 | 0.705 |
| A | translate_en | vanilla | S | 99 | 0.141 | -0.053 | 0.330 |
| A | translate_en | vanilla | S_diff | 99 | 0.515 | 0.344 | 0.655 |
| B | translate | affine | S | 60 | 0.194 | -0.033 | 0.422 |
| B | translate | affine | S_diff | 60 | 0.720 | 0.594 | 0.809 |
| B | translate | mean | S | 60 | 0.635 | 0.484 | 0.755 |
| B | translate | mean | S_diff | 60 | 0.835 | 0.738 | 0.906 |
| B | translate | vanilla | S | 60 | 0.406 | 0.185 | 0.598 |
| B | translate | vanilla | S_diff | 60 | 0.804 | 0.695 | 0.877 |
| B | translate_ctx | affine | S | 60 | 0.128 | -0.117 | 0.372 |
| B | translate_ctx | affine | S_diff | 60 | 0.723 | 0.544 | 0.837 |
| B | translate_ctx | mean | S | 60 | 0.555 | 0.331 | 0.710 |
| B | translate_ctx | mean | S_diff | 60 | 0.808 | 0.676 | 0.890 |
| B | translate_ctx | vanilla | S | 60 | 0.372 | 0.140 | 0.573 |
| B | translate_ctx | vanilla | S_diff | 60 | 0.741 | 0.573 | 0.856 |

## 3. 판정 (A, 번역 과제)

'크게 줄었음' = 감소율 ≥ 0.5 (S 기준 확률). '상관 하락' = Δr의 95% CI 상한 < 0. 바닐라 상관의 CI 하한이 0 이하이면 판정하지 않는다.

| x | 렌즈 | 바닐라 r [CI] | 영어 확률(바닐라) | 영어 확률(보정) | 감소율 [CI] | Δr [CI] | 판정 |
|---|---|---|---|---|---|---|---|
| S | affine | -0.033 [-0.228, 0.165] | 0.003 | 0.000 | 0.936 [0.866, 0.966] | -0.079 [-0.309, 0.152] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |
| S | mean | -0.033 [-0.228, 0.165] | 0.003 | 0.006 | -1.054 [-2.670, 0.519] | 0.131 [-0.018, 0.275] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |
| S_diff | affine | 0.297 [0.108, 0.464] | 0.003 | 0.000 | 0.936 [0.866, 0.966] | 0.017 [-0.101, 0.144] | 크게 줄고 상관은 그대로 → 지운 영어는 부풀려진 영어였다 (가설 지지) |
| S_diff | mean | 0.297 [0.108, 0.464] | 0.003 | 0.006 | -1.054 [-2.670, 0.519] | 0.034 [-0.041, 0.107] | 거의 안 줄었음 → 그 원인은 영어를 만들지 않았다 (가설에 불리) |

## 4. 음성 대조: B(한국어 동음이의어)

B의 간섭은 영어와 무관하므로 r_B ≈ 0이어야 한다. r_B도 크면 렌즈 점수가 영어가 아닌 '헷갈리는 정도'를 재고 있을 가능성이 있다. translate_ctx는 예문으로 뜻을 정해 준 조건이다(시장에서 배/복숭아 한 상자를 샀다). 이때 간섭은 '뜻이 정해졌는데도 같은 표기 때문에 남는 다른 뜻'이므로 A(한국어로는 뜻이 하나, 영어 다의어로 새는 방해어)와 구조가 맞는다.

| B 과제 | 렌즈 | n_B | B 평균 간섭 | B 평균 영어 확률 | r_A | r_B | r_A - r_B [CI] |
|---|---|---|---|---|---|---|---|
| translate | affine | 60 | 16.435 | 0.000 | -0.112 | 0.194 | -0.306 [-0.606, 0.001] |
| translate | mean | 60 | 16.435 | 0.007 | 0.098 | 0.635 | -0.537 [-0.790, -0.276] |
| translate | vanilla | 60 | 16.435 | 0.002 | -0.033 | 0.406 | -0.439 [-0.725, -0.136] |
| translate_ctx | affine | 60 | 10.965 | 0.000 | -0.112 | 0.128 | -0.240 [-0.542, 0.071] |
| translate_ctx | mean | 60 | 10.965 | 0.002 | 0.098 | 0.555 | -0.457 [-0.732, -0.160] |
| translate_ctx | vanilla | 60 | 10.965 | 0.001 | -0.033 | 0.372 | -0.404 [-0.691, -0.106] |

### 4b. B 뜻 빈도 비율 (국립국어원 어휘 의미 분석 말뭉치)

방해 뜻이 흔할수록 간섭이 커야 정상이다(eDom의 한국어 대응). 두 뜻 합이 적은 단어(dom_ok=False)는 뺐다.

| B 과제 | 렌즈 | n | r(방해 뜻 우세도, I) [CI] | r(S, I) [CI] (같은 문항) |
|---|---|---|---|---|
| translate | affine | 38 | 0.259 [-0.103, 0.528] | 0.171 [-0.180, 0.474] |
| translate | mean | 38 | 0.259 [-0.103, 0.528] | 0.611 [0.370, 0.792] |
| translate | vanilla | 38 | 0.259 [-0.103, 0.528] | 0.448 [0.171, 0.672] |
| translate_ctx | affine | 38 | 0.381 [-0.014, 0.666] | 0.098 [-0.289, 0.451] |
| translate_ctx | mean | 38 | 0.381 [-0.014, 0.666] | 0.603 [0.303, 0.786] |
| translate_ctx | vanilla | 38 | 0.381 [-0.014, 0.666] | 0.405 [0.148, 0.635] |

## 5. 어종별 (8문항 이상인 집단만)

한자어는 한자음을 거쳐 중국어로 바로 가는 지름길이 있을 수 있다.

| dataset | task | lens | etym | n | r | ci_lo | ci_hi | mean_I | mean_p |
|---|---|---|---|---|---|---|---|---|---|
| A | translate | affine | 고유어 | 34 | -0.460 | -0.695 | -0.192 | 3.206 | 0.000 |
| A | translate | affine | 모호 | 12 | 0.182 | -0.426 | 0.781 | 4.234 | 0.000 |
| A | translate | affine | 외래어 | 13 | 0.121 | -0.542 | 0.714 | 3.947 | 0.001 |
| A | translate | affine | 한자어 | 35 | -0.024 | -0.316 | 0.276 | 5.195 | 0.000 |
| A | translate | mean | 고유어 | 34 | 0.229 | -0.104 | 0.525 | 3.206 | 0.000 |
| A | translate | mean | 모호 | 12 | -0.203 | -0.821 | 0.533 | 4.234 | 0.000 |
| A | translate | mean | 외래어 | 13 | 0.346 | -0.339 | 0.838 | 3.947 | 0.008 |
| A | translate | mean | 한자어 | 35 | 0.007 | -0.330 | 0.362 | 5.195 | 0.015 |
| A | translate | vanilla | 고유어 | 34 | -0.117 | -0.450 | 0.239 | 3.206 | 0.001 |
| A | translate | vanilla | 모호 | 12 | -0.203 | -0.749 | 0.528 | 4.234 | 0.001 |
| A | translate | vanilla | 외래어 | 13 | 0.187 | -0.390 | 0.777 | 3.947 | 0.006 |
| A | translate | vanilla | 한자어 | 35 | 0.072 | -0.252 | 0.409 | 5.195 | 0.005 |
| B | translate | affine | 고유어쌍 | 12 | -0.399 | -0.826 | -0.145 | 14.550 | 0.000 |
| B | translate | affine | 한자어쌍 | 40 | 0.338 | 0.046 | 0.608 | 17.199 | 0.000 |
| B | translate | affine | 혼합 | 8 | 0.333 | -0.263 | 0.842 | 15.438 | 0.000 |
| B | translate | mean | 고유어쌍 | 12 | 0.580 | -0.166 | 0.878 | 14.550 | 0.001 |
| B | translate | mean | 한자어쌍 | 40 | 0.638 | 0.453 | 0.786 | 17.199 | 0.010 |
| B | translate | mean | 혼합 | 8 | 0.762 | -0.105 | 1.000 | 15.438 | 0.000 |
| B | translate | vanilla | 고유어쌍 | 12 | 0.203 | -0.565 | 0.676 | 14.550 | 0.001 |
| B | translate | vanilla | 한자어쌍 | 40 | 0.410 | 0.114 | 0.645 | 17.199 | 0.002 |
| B | translate | vanilla | 혼합 | 8 | 0.524 | -0.105 | 0.842 | 15.438 | 0.001 |
| B | translate_ctx | affine | 고유어쌍 | 12 | -0.350 | -0.782 | 0.209 | 11.518 | 0.000 |
| B | translate_ctx | affine | 한자어쌍 | 40 | 0.265 | -0.038 | 0.546 | 11.339 | 0.000 |
| B | translate_ctx | affine | 혼합 | 8 | 0.524 | -0.105 | 1.000 | 8.263 | 0.000 |
| B | translate_ctx | mean | 고유어쌍 | 12 | 0.503 | -0.065 | 0.863 | 11.518 | 0.001 |
| B | translate_ctx | mean | 한자어쌍 | 40 | 0.607 | 0.355 | 0.793 | 11.339 | 0.002 |
| B | translate_ctx | mean | 혼합 | 8 | 0.333 | -0.842 | 0.974 | 8.263 | 0.000 |
| B | translate_ctx | vanilla | 고유어쌍 | 12 | 0.336 | -0.309 | 0.837 | 11.518 | 0.001 |
| B | translate_ctx | vanilla | 한자어쌍 | 40 | 0.384 | 0.104 | 0.613 | 11.339 | 0.001 |
| B | translate_ctx | vanilla | 혼합 | 8 | 0.452 | -0.105 | 1.000 | 8.263 | 0.001 |

## 6. 양성 대조: 프롬프트에 영어 중간 단계

영어 경유를 강제했으므로 translate_en에서 간섭·영어 확률이 커지고 상관이 보여야 한다. 그렇지 않으면 간섭 지표가 영어 경유를 잡지 못하는 것이다.

| 과제 | 렌즈 | 평균 간섭 | 평균 영어 확률 | r [CI] |
|---|---|---|---|---|
| translate | affine | 4.147 | 0.000 | -0.112 [-0.292, 0.090] |
| translate | mean | 4.147 | 0.006 | 0.098 [-0.113, 0.315] |
| translate | vanilla | 4.147 | 0.003 | -0.033 [-0.228, 0.165] |
| translate_en | affine | 9.321 | 0.000 | 0.024 [-0.141, 0.200] |
| translate_en | mean | 9.321 | 0.016 | 0.279 [0.077, 0.453] |
| translate_en | vanilla | 9.321 | 0.007 | 0.141 [-0.053, 0.330] |

## 7. C: 중간층 평균 확률 (정답 영어 vs 무관 영어, 언어별 확률 질량)

가설 1이 맞다면 바닐라에서는 무관한 영어(chair)와 영어 질량 전체가 높고, 평균 제거 렌즈에서 줄어야 한다.

| task | lens | lp_target | lp_unrelated | mass_en | mass_en_word | mass_zh | mass_ko |
|---|---|---|---|---|---|---|---|
| repeat | affine | 0.0001 | 0.0001 | 0.0407 | 0.0123 | 0.1099 | 0.2596 |
| repeat | mean | 0.0022 | 0.0001 | 0.3561 | 0.0584 | 0.4088 | 0.0045 |
| repeat | vanilla | 0.0004 | 0.0002 | 0.3491 | 0.0237 | 0.1055 | 0.0024 |
| translate | affine | 0.0001 | 0.0001 | 0.0202 | 0.0059 | 0.3446 | 0.0304 |
| translate | mean | 0.0045 | 0.0000 | 0.2955 | 0.0455 | 0.5852 | 0.0058 |
| translate | vanilla | 0.0023 | 0.0003 | 0.2772 | 0.0261 | 0.3421 | 0.0023 |
| translate_to_en | affine | 0.0071 | 0.0006 | 0.0527 | 0.0072 | 0.0812 | 0.0260 |
| translate_to_en | mean | 0.0364 | 0.0001 | 0.4597 | 0.0974 | 0.3769 | 0.0016 |
| translate_to_en | vanilla | 0.0262 | 0.0007 | 0.4445 | 0.0476 | 0.1135 | 0.0016 |

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
| affine | 2.702 |
| mean | 12.307 |
| vanilla | 8.774 |
