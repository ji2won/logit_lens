# 결과 요약 (llama3.2-3b)

층 수 28, 렌즈 점수 구간 7–21층, bootstrap 2000회

## 1. 신뢰도 (프롬프트 4개, Cronbach's alpha)

기준 0.7. 미달한 지표의 상관은 '과장'으로 해석하지 않는다.

| dataset | task | measure | alpha | 통과 |
|---|---|---|---|---|
| A | translate | 간섭 I | 0.979 | O |
| A | translate_en | 간섭 I | 0.982 | O |
| B | translate | 간섭 I | 0.988 | O |
| B | translate_ctx | 간섭 I | 0.983 | O |
| A | translate | 렌즈 S (affine) | 0.980 | O |
| A | translate | 렌즈 S_diff (affine) | 0.991 | O |
| A | translate | 렌즈 S (mean) | 0.994 | O |
| A | translate | 렌즈 S_diff (mean) | 0.995 | O |
| A | translate | 렌즈 S (vanilla) | 0.995 | O |
| A | translate | 렌즈 S_diff (vanilla) | 0.995 | O |
| A | translate_en | 렌즈 S (affine) | 0.970 | O |
| A | translate_en | 렌즈 S_diff (affine) | 0.983 | O |
| A | translate_en | 렌즈 S (mean) | 0.971 | O |
| A | translate_en | 렌즈 S_diff (mean) | 0.990 | O |
| A | translate_en | 렌즈 S (vanilla) | 0.971 | O |
| A | translate_en | 렌즈 S_diff (vanilla) | 0.990 | O |
| B | translate | 렌즈 S (affine) | 0.972 | O |
| B | translate | 렌즈 S_diff (affine) | 0.986 | O |
| B | translate | 렌즈 S (mean) | 0.985 | O |
| B | translate | 렌즈 S_diff (mean) | 0.987 | O |
| B | translate | 렌즈 S (vanilla) | 0.986 | O |
| B | translate | 렌즈 S_diff (vanilla) | 0.986 | O |
| B | translate_ctx | 렌즈 S (affine) | 0.949 | O |
| B | translate_ctx | 렌즈 S_diff (affine) | 0.976 | O |
| B | translate_ctx | 렌즈 S (mean) | 0.969 | O |
| B | translate_ctx | 렌즈 S_diff (mean) | 0.980 | O |
| B | translate_ctx | 렌즈 S (vanilla) | 0.972 | O |
| B | translate_ctx | 렌즈 S_diff (vanilla) | 0.980 | O |
| C | repeat | 렌즈 S (affine) | 0.959 | O |
| C | repeat | 렌즈 S_diff (affine) | nan | X |
| C | repeat | 렌즈 S (mean) | 0.972 | O |
| C | repeat | 렌즈 S_diff (mean) | nan | X |
| C | repeat | 렌즈 S (vanilla) | 0.970 | O |
| C | repeat | 렌즈 S_diff (vanilla) | nan | X |
| C | translate | 렌즈 S (affine) | 0.972 | O |
| C | translate | 렌즈 S_diff (affine) | nan | X |
| C | translate | 렌즈 S (mean) | 0.995 | O |
| C | translate | 렌즈 S_diff (mean) | nan | X |
| C | translate | 렌즈 S (vanilla) | 0.996 | O |
| C | translate | 렌즈 S_diff (vanilla) | nan | X |

## 2. 렌즈 점수 × 간섭 상관 (Spearman, 문항 단위)

| dataset | task | lens | x | n | r | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| A | translate | affine | S | 99 | -0.152 | -0.341 | 0.055 |
| A | translate | affine | S_diff | 99 | 0.128 | -0.077 | 0.311 |
| A | translate | mean | S | 99 | -0.077 | -0.279 | 0.145 |
| A | translate | mean | S_diff | 99 | 0.195 | -0.000 | 0.373 |
| A | translate | vanilla | S | 99 | -0.096 | -0.302 | 0.122 |
| A | translate | vanilla | S_diff | 99 | 0.211 | 0.018 | 0.388 |
| A | translate_en | affine | S | 99 | -0.141 | -0.319 | 0.064 |
| A | translate_en | affine | S_diff | 99 | 0.262 | 0.077 | 0.435 |
| A | translate_en | mean | S | 99 | -0.086 | -0.291 | 0.132 |
| A | translate_en | mean | S_diff | 99 | 0.271 | 0.102 | 0.428 |
| A | translate_en | vanilla | S | 99 | -0.091 | -0.290 | 0.126 |
| A | translate_en | vanilla | S_diff | 99 | 0.289 | 0.115 | 0.445 |
| B | translate | affine | S | 60 | 0.484 | 0.317 | 0.649 |
| B | translate | affine | S_diff | 60 | 0.737 | 0.603 | 0.842 |
| B | translate | mean | S | 60 | 0.705 | 0.588 | 0.797 |
| B | translate | mean | S_diff | 60 | 0.789 | 0.719 | 0.851 |
| B | translate | vanilla | S | 60 | 0.699 | 0.572 | 0.795 |
| B | translate | vanilla | S_diff | 60 | 0.780 | 0.699 | 0.844 |
| B | translate_ctx | affine | S | 60 | 0.151 | -0.091 | 0.411 |
| B | translate_ctx | affine | S_diff | 60 | 0.695 | 0.521 | 0.821 |
| B | translate_ctx | mean | S | 60 | 0.626 | 0.472 | 0.755 |
| B | translate_ctx | mean | S_diff | 60 | 0.701 | 0.539 | 0.815 |
| B | translate_ctx | vanilla | S | 60 | 0.559 | 0.385 | 0.711 |
| B | translate_ctx | vanilla | S_diff | 60 | 0.713 | 0.565 | 0.823 |

## 3. 판정 (A, 번역 과제)

'크게 줄었음' = 감소율 ≥ 0.5 (S 기준 확률). '상관 하락' = Δr의 95% CI 상한 < 0. 바닐라 상관의 CI 하한이 0 이하이면 판정하지 않는다.

| x | 렌즈 | 바닐라 r [CI] | 영어 확률(바닐라) | 영어 확률(보정) | 감소율 [CI] | Δr [CI] | 판정 |
|---|---|---|---|---|---|---|---|
| S | affine | -0.096 [-0.302, 0.122] | 0.229 | 0.011 | 0.954 [0.903, 0.987] | -0.056 [-0.213, 0.093] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |
| S | mean | -0.096 [-0.302, 0.122] | 0.229 | 0.244 | -0.064 [-0.125, -0.008] | 0.019 [-0.018, 0.058] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |
| S_diff | affine | 0.211 [0.018, 0.388] | 0.229 | 0.011 | 0.954 [0.903, 0.987] | -0.083 [-0.243, 0.065] | 크게 줄고 상관은 그대로 → 지운 영어는 부풀려진 영어였다 (가설 지지) |
| S_diff | mean | 0.211 [0.018, 0.388] | 0.229 | 0.244 | -0.064 [-0.125, -0.008] | -0.016 [-0.049, 0.013] | 거의 안 줄었음 → 그 원인은 영어를 만들지 않았다 (가설에 불리) |

## 4. 음성 대조: B(한국어 동음이의어)

B의 간섭은 영어와 무관하므로 r_B ≈ 0이어야 한다. r_B도 크면 렌즈 점수가 영어가 아닌 '헷갈리는 정도'를 재고 있을 가능성이 있다. translate_ctx는 예문으로 뜻을 정해 준 조건이다(시장에서 배/복숭아 한 상자를 샀다). 이때 간섭은 '뜻이 정해졌는데도 같은 표기 때문에 남는 다른 뜻'이므로 A(한국어로는 뜻이 하나, 영어 다의어로 새는 방해어)와 구조가 맞는다.

| B 과제 | 렌즈 | n_B | B 평균 간섭 | B 평균 영어 확률 | r_A | r_B | r_A - r_B [CI] |
|---|---|---|---|---|---|---|---|
| translate | affine | 60 | 8.326 | 0.002 | -0.152 | 0.484 | -0.636 [-0.885, -0.382] |
| translate | mean | 60 | 8.326 | 0.240 | -0.077 | 0.705 | -0.782 [-1.009, -0.534] |
| translate | vanilla | 60 | 8.326 | 0.224 | -0.096 | 0.699 | -0.795 [-1.024, -0.544] |
| translate_ctx | affine | 60 | 4.459 | 0.000 | -0.152 | 0.151 | -0.303 [-0.608, 0.021] |
| translate_ctx | mean | 60 | 4.459 | 0.054 | -0.077 | 0.626 | -0.703 [-0.947, -0.447] |
| translate_ctx | vanilla | 60 | 4.459 | 0.052 | -0.096 | 0.559 | -0.655 [-0.917, -0.390] |

### 4b. B 뜻 빈도 비율 (국립국어원 어휘 의미 분석 말뭉치)

방해 뜻이 흔할수록 간섭이 커야 정상이다(eDom의 한국어 대응). 두 뜻 합이 적은 단어(dom_ok=False)는 뺐다.

| B 과제 | 렌즈 | n | r(방해 뜻 우세도, I) [CI] | r(S, I) [CI] (같은 문항) |
|---|---|---|---|---|
| translate | affine | 38 | 0.523 [0.198, 0.747] | 0.440 [0.223, 0.657] |
| translate | mean | 38 | 0.523 [0.198, 0.747] | 0.714 [0.542, 0.833] |
| translate | vanilla | 38 | 0.523 [0.198, 0.747] | 0.702 [0.517, 0.832] |
| translate_ctx | affine | 38 | 0.420 [0.021, 0.721] | 0.244 [-0.080, 0.544] |
| translate_ctx | mean | 38 | 0.420 [0.021, 0.721] | 0.586 [0.362, 0.761] |
| translate_ctx | vanilla | 38 | 0.420 [0.021, 0.721] | 0.526 [0.282, 0.725] |

## 5. 어종별 (8문항 이상인 집단만)

한자어는 한자음을 거쳐 중국어로 바로 가는 지름길이 있을 수 있다.

| dataset | task | lens | etym | n | r | ci_lo | ci_hi | mean_I | mean_p |
|---|---|---|---|---|---|---|---|---|---|
| A | translate | affine | 고유어 | 34 | -0.054 | -0.406 | 0.312 | 1.448 | 0.002 |
| A | translate | affine | 모호 | 12 | -0.573 | -0.899 | 0.065 | 1.715 | 0.006 |
| A | translate | affine | 외래어 | 13 | 0.341 | -0.346 | 0.842 | 1.962 | 0.053 |
| A | translate | affine | 한자어 | 35 | -0.196 | -0.496 | 0.143 | 1.949 | 0.006 |
| A | translate | mean | 고유어 | 34 | -0.137 | -0.481 | 0.219 | 1.448 | 0.172 |
| A | translate | mean | 모호 | 12 | -0.462 | -0.872 | 0.238 | 1.715 | 0.226 |
| A | translate | mean | 외래어 | 13 | 0.286 | -0.389 | 0.765 | 1.962 | 0.432 |
| A | translate | mean | 한자어 | 35 | 0.003 | -0.359 | 0.370 | 1.949 | 0.277 |
| A | translate | vanilla | 고유어 | 34 | -0.160 | -0.493 | 0.214 | 1.448 | 0.175 |
| A | translate | vanilla | 모호 | 12 | -0.510 | -0.851 | 0.149 | 1.715 | 0.214 |
| A | translate | vanilla | 외래어 | 13 | 0.302 | -0.356 | 0.780 | 1.962 | 0.416 |
| A | translate | vanilla | 한자어 | 35 | 0.011 | -0.354 | 0.380 | 1.949 | 0.243 |
| B | translate | affine | 고유어쌍 | 12 | 0.357 | 0.086 | 0.818 | 7.112 | 0.001 |
| B | translate | affine | 한자어쌍 | 40 | 0.508 | 0.294 | 0.701 | 8.349 | 0.002 |
| B | translate | affine | 혼합 | 8 | 0.595 | 0.390 | 1.000 | 10.033 | 0.001 |
| B | translate | mean | 고유어쌍 | 12 | 0.839 | 0.681 | 1.000 | 7.112 | 0.239 |
| B | translate | mean | 한자어쌍 | 40 | 0.705 | 0.524 | 0.827 | 8.349 | 0.253 |
| B | translate | mean | 혼합 | 8 | 0.714 | 0.600 | 1.000 | 10.033 | 0.180 |
| B | translate | vanilla | 고유어쌍 | 12 | 0.853 | 0.696 | 1.000 | 7.112 | 0.233 |
| B | translate | vanilla | 한자어쌍 | 40 | 0.685 | 0.499 | 0.825 | 8.349 | 0.235 |
| B | translate | vanilla | 혼합 | 8 | 0.667 | 0.585 | 1.000 | 10.033 | 0.157 |
| B | translate_ctx | affine | 고유어쌍 | 12 | 0.168 | -0.342 | 0.579 | 4.876 | 0.001 |
| B | translate_ctx | affine | 한자어쌍 | 40 | 0.186 | -0.180 | 0.504 | 4.725 | 0.000 |
| B | translate_ctx | affine | 혼합 | 8 | -0.119 | -1.000 | 0.400 | 2.505 | 0.000 |
| B | translate_ctx | mean | 고유어쌍 | 12 | 0.643 | 0.122 | 0.917 | 4.876 | 0.177 |
| B | translate_ctx | mean | 한자어쌍 | 40 | 0.643 | 0.453 | 0.792 | 4.725 | 0.028 |
| B | translate_ctx | mean | 혼합 | 8 | 0.548 | 0.390 | 0.842 | 2.505 | 0.000 |
| B | translate_ctx | vanilla | 고유어쌍 | 12 | 0.713 | 0.130 | 0.942 | 4.876 | 0.175 |
| B | translate_ctx | vanilla | 한자어쌍 | 40 | 0.523 | 0.325 | 0.714 | 4.725 | 0.026 |
| B | translate_ctx | vanilla | 혼합 | 8 | 0.405 | -0.263 | 0.842 | 2.505 | 0.000 |

## 6. 양성 대조: 프롬프트에 영어 중간 단계

영어 경유를 강제했으므로 translate_en에서 간섭·영어 확률이 커지고 상관이 보여야 한다. 그렇지 않으면 간섭 지표가 영어 경유를 잡지 못하는 것이다.

| 과제 | 렌즈 | 평균 간섭 | 평균 영어 확률 | r [CI] |
|---|---|---|---|---|
| translate | affine | 1.746 | 0.011 | -0.152 [-0.341, 0.055] |
| translate | mean | 1.746 | 0.244 | -0.077 [-0.279, 0.145] |
| translate | vanilla | 1.746 | 0.229 | -0.096 [-0.302, 0.122] |
| translate_en | affine | 4.394 | 0.035 | -0.141 [-0.319, 0.064] |
| translate_en | mean | 4.394 | 0.517 | -0.086 [-0.291, 0.132] |
| translate_en | vanilla | 4.394 | 0.487 | -0.091 [-0.290, 0.126] |

## 7. C: 중간층 평균 확률 (정답 영어 vs 무관 영어, 언어별 확률 질량)

가설 1이 맞다면 바닐라에서는 무관한 영어(chair)와 영어 질량 전체가 높고, 평균 제거 렌즈에서 줄어야 한다.

| task | lens | lp_target | lp_unrelated | mass_en | mass_en_word | mass_zh | mass_ko |
|---|---|---|---|---|---|---|---|
| repeat | affine | 0.0020 | 0.0005 | 0.0580 | 0.0070 | 0.4726 | 0.1457 |
| repeat | mean | 0.0320 | 0.0001 | 0.6606 | 0.2810 | 0.0289 | 0.0138 |
| repeat | vanilla | 0.0217 | 0.0001 | 0.6567 | 0.2761 | 0.0329 | 0.0139 |
| translate | affine | 0.0031 | 0.0001 | 0.0165 | 0.0043 | 0.7773 | 0.0003 |
| translate | mean | 0.1206 | 0.0000 | 0.7286 | 0.4228 | 0.0229 | 0.0097 |
| translate | vanilla | 0.1003 | 0.0000 | 0.7131 | 0.3999 | 0.0292 | 0.0100 |

## 8. 층 구간별 언어 질량 (입력어 프롬프트 평균 확률)

`en`은 라틴 문자 2자 이상 토큰 전부(단어 조각 포함), `en_word`는 단어 시작 토큰이면서 영어 단어 목록에 있는 토큰만 센다. 둘의 차이가 크면 '영어'로 보이던 것의 상당 부분이 단어 조각이다. 구간은 렌즈 점수 구간 설정과 무관하게 전체 층의 1/4, 3/4로 나눈다.

| dataset | task | lens | band | en | en_word | zh | ko |
|---|---|---|---|---|---|---|---|
| A | translate | affine | 초반 1–6 | 0.036 | 0.006 | 0.649 | 0.019 |
| A | translate | affine | 중간 7–21 | 0.016 | 0.004 | 0.778 | 0.000 |
| A | translate | affine | 후반 22–27 | 0.113 | 0.073 | 0.738 | 0.000 |
| A | translate | mean | 초반 1–6 | 0.369 | 0.155 | 0.014 | 0.006 |
| A | translate | mean | 중간 7–21 | 0.716 | 0.403 | 0.024 | 0.010 |
| A | translate | mean | 후반 22–27 | 0.917 | 0.843 | 0.057 | 0.001 |
| A | translate | vanilla | 초반 1–6 | 0.396 | 0.159 | 0.016 | 0.007 |
| A | translate | vanilla | 중간 7–21 | 0.702 | 0.381 | 0.030 | 0.011 |
| A | translate | vanilla | 후반 22–27 | 0.891 | 0.804 | 0.065 | 0.002 |
| A | translate_en | affine | 초반 1–6 | 0.026 | 0.003 | 0.714 | 0.004 |
| A | translate_en | affine | 중간 7–21 | 0.022 | 0.005 | 0.759 | 0.000 |
| A | translate_en | affine | 후반 22–27 | 0.175 | 0.112 | 0.688 | 0.000 |
| A | translate_en | mean | 초반 1–6 | 0.375 | 0.153 | 0.015 | 0.007 |
| A | translate_en | mean | 중간 7–21 | 0.716 | 0.401 | 0.023 | 0.011 |
| A | translate_en | mean | 후반 22–27 | 0.933 | 0.880 | 0.045 | 0.001 |
| A | translate_en | vanilla | 초반 1–6 | 0.399 | 0.156 | 0.016 | 0.008 |
| A | translate_en | vanilla | 중간 7–21 | 0.704 | 0.382 | 0.029 | 0.011 |
| A | translate_en | vanilla | 후반 22–27 | 0.914 | 0.851 | 0.053 | 0.001 |
| B | translate | affine | 초반 1–6 | 0.035 | 0.006 | 0.656 | 0.019 |
| B | translate | affine | 중간 7–21 | 0.013 | 0.002 | 0.792 | 0.000 |
| B | translate | affine | 후반 22–27 | 0.053 | 0.040 | 0.842 | 0.000 |
| B | translate | mean | 초반 1–6 | 0.369 | 0.155 | 0.014 | 0.006 |
| B | translate | mean | 중간 7–21 | 0.721 | 0.412 | 0.022 | 0.010 |
| B | translate | mean | 후반 22–27 | 0.904 | 0.857 | 0.064 | 0.002 |
| B | translate | vanilla | 초반 1–6 | 0.395 | 0.159 | 0.016 | 0.007 |
| B | translate | vanilla | 중간 7–21 | 0.706 | 0.389 | 0.028 | 0.010 |
| B | translate | vanilla | 후반 22–27 | 0.859 | 0.786 | 0.090 | 0.003 |
| B | translate_ctx | affine | 초반 1–6 | 0.030 | 0.004 | 0.624 | 0.024 |
| B | translate_ctx | affine | 중간 7–21 | 0.018 | 0.003 | 0.750 | 0.000 |
| B | translate_ctx | affine | 후반 22–27 | 0.067 | 0.055 | 0.858 | 0.001 |
| B | translate_ctx | mean | 초반 1–6 | 0.367 | 0.156 | 0.014 | 0.006 |
| B | translate_ctx | mean | 중간 7–21 | 0.718 | 0.398 | 0.021 | 0.010 |
| B | translate_ctx | mean | 후반 22–27 | 0.925 | 0.877 | 0.054 | 0.001 |
| B | translate_ctx | vanilla | 초반 1–6 | 0.395 | 0.160 | 0.016 | 0.007 |
| B | translate_ctx | vanilla | 중간 7–21 | 0.707 | 0.382 | 0.027 | 0.010 |
| B | translate_ctx | vanilla | 후반 22–27 | 0.887 | 0.818 | 0.074 | 0.003 |
| C | repeat | affine | 초반 1–6 | 0.041 | 0.007 | 0.574 | 0.026 |
| C | repeat | affine | 중간 7–21 | 0.058 | 0.007 | 0.473 | 0.146 |
| C | repeat | affine | 후반 22–27 | 0.027 | 0.005 | 0.043 | 0.796 |
| C | repeat | mean | 초반 1–6 | 0.359 | 0.156 | 0.014 | 0.006 |
| C | repeat | mean | 중간 7–21 | 0.661 | 0.281 | 0.029 | 0.014 |
| C | repeat | mean | 후반 22–27 | 0.637 | 0.416 | 0.032 | 0.231 |
| C | repeat | vanilla | 초반 1–6 | 0.390 | 0.162 | 0.016 | 0.007 |
| C | repeat | vanilla | 중간 7–21 | 0.657 | 0.276 | 0.033 | 0.014 |
| C | repeat | vanilla | 후반 22–27 | 0.612 | 0.368 | 0.041 | 0.212 |
| C | translate | affine | 초반 1–6 | 0.036 | 0.006 | 0.651 | 0.020 |
| C | translate | affine | 중간 7–21 | 0.016 | 0.004 | 0.777 | 0.000 |
| C | translate | affine | 후반 22–27 | 0.106 | 0.072 | 0.732 | 0.001 |
| C | translate | mean | 초반 1–6 | 0.370 | 0.156 | 0.014 | 0.006 |
| C | translate | mean | 중간 7–21 | 0.729 | 0.423 | 0.023 | 0.010 |
| C | translate | mean | 후반 22–27 | 0.915 | 0.853 | 0.056 | 0.001 |
| C | translate | vanilla | 초반 1–6 | 0.396 | 0.159 | 0.016 | 0.007 |
| C | translate | vanilla | 중간 7–21 | 0.713 | 0.400 | 0.029 | 0.010 |
| C | translate | vanilla | 후반 22–27 | 0.885 | 0.812 | 0.073 | 0.002 |

## 부록: 렌즈 적합도 (held-out KL(최종 || 렌즈), 중간층 평균)

아핀 렌즈가 바닐라보다 KL이 작아야 기저 맞춤이 실제로 된 것이다.

| lens | kl |
|---|---|
| affine | 1.916 |
| mean | 8.060 |
| vanilla | 7.640 |
