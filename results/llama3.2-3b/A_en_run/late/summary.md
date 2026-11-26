# 결과 요약 (late)

층 수 28, 렌즈 점수 구간 21–27층, bootstrap 2000회

## 1. 신뢰도 (프롬프트 4개, Cronbach's alpha)

기준 0.7. 미달한 지표의 상관은 '과장'으로 해석하지 않는다.

| dataset | task | measure | alpha | 통과 |
|---|---|---|---|---|
| A | translate | 간섭 I | 0.979 | O |
| A | translate_en | 간섭 I | 0.980 | O |
| B | translate | 간섭 I | 0.988 | O |
| B | translate_ctx | 간섭 I | 0.983 | O |
| A | translate | 렌즈 S (affine) | 0.991 | O |
| A | translate | 렌즈 S_diff (affine) | 0.993 | O |
| A | translate | 렌즈 S (mean) | 0.970 | O |
| A | translate | 렌즈 S_diff (mean) | 0.997 | O |
| A | translate | 렌즈 S (vanilla) | 0.972 | O |
| A | translate | 렌즈 S_diff (vanilla) | 0.996 | O |
| A | translate_en | 렌즈 S (affine) | 0.993 | O |
| A | translate_en | 렌즈 S_diff (affine) | 0.991 | O |
| A | translate_en | 렌즈 S (mean) | 0.976 | O |
| A | translate_en | 렌즈 S_diff (mean) | 0.997 | O |
| A | translate_en | 렌즈 S (vanilla) | 0.979 | O |
| A | translate_en | 렌즈 S_diff (vanilla) | 0.997 | O |
| B | translate | 렌즈 S (affine) | 0.991 | O |
| B | translate | 렌즈 S_diff (affine) | 0.990 | O |
| B | translate | 렌즈 S (mean) | 0.990 | O |
| B | translate | 렌즈 S_diff (mean) | 0.985 | O |
| B | translate | 렌즈 S (vanilla) | 0.990 | O |
| B | translate | 렌즈 S_diff (vanilla) | 0.984 | O |
| B | translate_ctx | 렌즈 S (affine) | 0.987 | O |
| B | translate_ctx | 렌즈 S_diff (affine) | 0.987 | O |
| B | translate_ctx | 렌즈 S (mean) | 0.983 | O |
| B | translate_ctx | 렌즈 S_diff (mean) | 0.979 | O |
| B | translate_ctx | 렌즈 S (vanilla) | 0.984 | O |
| B | translate_ctx | 렌즈 S_diff (vanilla) | 0.981 | O |
| C | repeat | 렌즈 S (affine) | 0.960 | O |
| C | repeat | 렌즈 S_diff (affine) | nan | X |
| C | repeat | 렌즈 S (mean) | 0.977 | O |
| C | repeat | 렌즈 S_diff (mean) | nan | X |
| C | repeat | 렌즈 S (vanilla) | 0.975 | O |
| C | repeat | 렌즈 S_diff (vanilla) | nan | X |
| C | translate | 렌즈 S (affine) | 0.994 | O |
| C | translate | 렌즈 S_diff (affine) | nan | X |
| C | translate | 렌즈 S (mean) | 0.992 | O |
| C | translate | 렌즈 S_diff (mean) | nan | X |
| C | translate | 렌즈 S (vanilla) | 0.992 | O |
| C | translate | 렌즈 S_diff (vanilla) | nan | X |

## 2. 렌즈 점수 × 간섭 상관 (Spearman, 문항 단위)

| dataset | task | lens | x | n | r | ci_lo | ci_hi |
|---|---|---|---|---|---|---|---|
| A | translate | affine | S | 44 | 0.227 | -0.112 | 0.507 |
| A | translate | affine | S_diff | 44 | 0.326 | 0.012 | 0.566 |
| A | translate | mean | S | 44 | -0.102 | -0.370 | 0.187 |
| A | translate | mean | S_diff | 44 | 0.128 | -0.183 | 0.414 |
| A | translate | vanilla | S | 44 | -0.118 | -0.380 | 0.174 |
| A | translate | vanilla | S_diff | 44 | 0.142 | -0.166 | 0.423 |
| A | translate_en | affine | S | 44 | 0.223 | -0.106 | 0.505 |
| A | translate_en | affine | S_diff | 44 | 0.336 | 0.013 | 0.591 |
| A | translate_en | mean | S | 44 | -0.226 | -0.507 | 0.079 |
| A | translate_en | mean | S_diff | 44 | 0.122 | -0.173 | 0.386 |
| A | translate_en | vanilla | S | 44 | -0.217 | -0.497 | 0.089 |
| A | translate_en | vanilla | S_diff | 44 | 0.158 | -0.120 | 0.401 |
| B | translate | affine | S | 60 | 0.736 | 0.630 | 0.827 |
| B | translate | affine | S_diff | 60 | 0.805 | 0.703 | 0.882 |
| B | translate | mean | S | 60 | 0.757 | 0.676 | 0.829 |
| B | translate | mean | S_diff | 60 | 0.799 | 0.716 | 0.858 |
| B | translate | vanilla | S | 60 | 0.750 | 0.662 | 0.829 |
| B | translate | vanilla | S_diff | 60 | 0.783 | 0.700 | 0.844 |
| B | translate_ctx | affine | S | 60 | 0.538 | 0.340 | 0.707 |
| B | translate_ctx | affine | S_diff | 60 | 0.690 | 0.536 | 0.811 |
| B | translate_ctx | mean | S | 60 | 0.682 | 0.540 | 0.805 |
| B | translate_ctx | mean | S_diff | 60 | 0.756 | 0.618 | 0.853 |
| B | translate_ctx | vanilla | S | 60 | 0.649 | 0.492 | 0.786 |
| B | translate_ctx | vanilla | S_diff | 60 | 0.760 | 0.622 | 0.854 |

## 3. 판정 (A, 번역 과제)

'크게 줄었음' = 감소율 ≥ 0.5 (S 기준 확률). '상관 하락' = Δr의 95% CI 상한 < 0. 바닐라 상관의 CI 하한이 0 이하이면 판정하지 않는다.

| x | 렌즈 | 바닐라 r [CI] | 영어 확률(바닐라) | 영어 확률(보정) | 감소율 [CI] | Δr [CI] | 판정 |
|---|---|---|---|---|---|---|---|
| S | affine | -0.118 [-0.380, 0.174] | 0.683 | 0.175 | 0.743 [0.638, 0.845] | 0.345 [-0.043, 0.700] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |
| S | mean | -0.118 [-0.380, 0.174] | 0.683 | 0.708 | -0.038 [-0.066, -0.011] | 0.015 [-0.066, 0.095] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |
| S_diff | affine | 0.142 [-0.166, 0.423] | 0.683 | 0.175 | 0.743 [0.638, 0.845] | 0.184 [-0.030, 0.381] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |
| S_diff | mean | 0.142 [-0.166, 0.423] | 0.683 | 0.708 | -0.038 [-0.066, -0.011] | -0.015 [-0.070, 0.045] | 판정 불가: 바닐라 렌즈에서 상관이 유의하지 않아 '상관 유지/하락'을 가릴 수 없음 |

## 4. 음성 대조: B(한국어 동음이의어)

B의 간섭은 영어와 무관하므로 r_B ≈ 0이어야 한다. r_B도 크면 렌즈 점수가 영어가 아닌 '헷갈리는 정도'를 재고 있을 가능성이 있다. translate_ctx는 예문으로 뜻을 정해 준 조건이다(시장에서 배/복숭아 한 상자를 샀다). 이때 간섭은 '뜻이 정해졌는데도 같은 표기 때문에 남는 다른 뜻'이므로 A(한국어로는 뜻이 하나, 영어 다의어로 새는 방해어)와 구조가 맞는다.

| B 과제 | 렌즈 | n_B | B 평균 간섭 | B 평균 영어 확률 | r_A | r_B | r_A - r_B [CI] |
|---|---|---|---|---|---|---|---|
| translate | affine | 60 | 8.326 | 0.040 | 0.227 | 0.736 | -0.509 [-0.847, -0.216] |
| translate | mean | 60 | 8.326 | 0.295 | -0.102 | 0.757 | -0.859 [-1.139, -0.553] |
| translate | vanilla | 60 | 8.326 | 0.282 | -0.118 | 0.750 | -0.868 [-1.148, -0.565] |
| translate_ctx | affine | 60 | 4.459 | 0.005 | 0.227 | 0.538 | -0.311 [-0.685, 0.012] |
| translate_ctx | mean | 60 | 4.459 | 0.085 | -0.102 | 0.682 | -0.784 [-1.077, -0.467] |
| translate_ctx | vanilla | 60 | 4.459 | 0.083 | -0.118 | 0.649 | -0.767 [-1.060, -0.432] |

### 4b. B 뜻 빈도 비율 (국립국어원 어휘 의미 분석 말뭉치)

방해 뜻이 흔할수록 간섭이 커야 정상이다(eDom의 한국어 대응). 두 뜻 합이 적은 단어(dom_ok=False)는 뺐다.

| B 과제 | 렌즈 | n | r(방해 뜻 우세도, I) [CI] | r(S, I) [CI] (같은 문항) |
|---|---|---|---|---|
| translate | affine | 38 | 0.523 [0.198, 0.747] | 0.745 [0.606, 0.838] |
| translate | mean | 38 | 0.523 [0.198, 0.747] | 0.763 [0.650, 0.854] |
| translate | vanilla | 38 | 0.523 [0.198, 0.747] | 0.739 [0.630, 0.830] |
| translate_ctx | affine | 38 | 0.420 [0.021, 0.721] | 0.552 [0.301, 0.740] |
| translate_ctx | mean | 38 | 0.420 [0.021, 0.721] | 0.706 [0.541, 0.833] |
| translate_ctx | vanilla | 38 | 0.420 [0.021, 0.721] | 0.669 [0.466, 0.829] |

## 5. 어종별 (8문항 이상인 집단만)

한자어는 한자음을 거쳐 중국어로 바로 가는 지름길이 있을 수 있다.

| dataset | task | lens | etym | n | r | ci_lo | ci_hi | mean_I | mean_p |
|---|---|---|---|---|---|---|---|---|---|
| A | translate | affine | 고유어 | 11 | 0.518 | -0.159 | 0.889 | 1.218 | 0.038 |
| A | translate | affine | 외래어 | 10 | 0.382 | -0.346 | 0.925 | 2.744 | 0.479 |
| A | translate | affine | 한자어 | 16 | -0.050 | -0.627 | 0.463 | 2.319 | 0.091 |
| A | translate | mean | 고유어 | 11 | -0.664 | -0.972 | -0.097 | 1.218 | 0.641 |
| A | translate | mean | 외래어 | 10 | -0.164 | -0.901 | 0.529 | 2.744 | 0.824 |
| A | translate | mean | 한자어 | 16 | -0.006 | -0.487 | 0.481 | 2.319 | 0.705 |
| A | translate | vanilla | 고유어 | 11 | -0.545 | -0.877 | 0.061 | 1.218 | 0.614 |
| A | translate | vanilla | 외래어 | 10 | -0.139 | -0.869 | 0.538 | 2.744 | 0.796 |
| A | translate | vanilla | 한자어 | 16 | -0.065 | -0.564 | 0.429 | 2.319 | 0.670 |
| B | translate | affine | 고유어쌍 | 12 | 0.902 | 0.771 | 0.978 | 7.112 | 0.002 |
| B | translate | affine | 한자어쌍 | 40 | 0.696 | 0.558 | 0.813 | 8.349 | 0.060 |
| B | translate | affine | 혼합 | 8 | 0.857 | 0.682 | 1.000 | 10.033 | 0.002 |
| B | translate | mean | 고유어쌍 | 12 | 0.832 | 0.696 | 0.972 | 7.112 | 0.240 |
| B | translate | mean | 한자어쌍 | 40 | 0.743 | 0.633 | 0.843 | 8.349 | 0.326 |
| B | translate | mean | 혼합 | 8 | 0.714 | 0.600 | 1.000 | 10.033 | 0.223 |
| B | translate | vanilla | 고유어쌍 | 12 | 0.839 | 0.696 | 1.000 | 7.112 | 0.245 |
| B | translate | vanilla | 한자어쌍 | 40 | 0.732 | 0.621 | 0.837 | 8.349 | 0.307 |
| B | translate | vanilla | 혼합 | 8 | 0.738 | 0.600 | 1.000 | 10.033 | 0.207 |
| B | translate_ctx | affine | 고유어쌍 | 12 | 0.811 | 0.239 | 0.986 | 4.876 | 0.001 |
| B | translate_ctx | affine | 한자어쌍 | 40 | 0.516 | 0.278 | 0.718 | 4.725 | 0.007 |
| B | translate_ctx | affine | 혼합 | 8 | 0.024 | -0.842 | 0.737 | 2.505 | 0.000 |
| B | translate_ctx | mean | 고유어쌍 | 12 | 0.874 | 0.435 | 0.992 | 4.876 | 0.191 |
| B | translate_ctx | mean | 한자어쌍 | 40 | 0.656 | 0.517 | 0.777 | 4.725 | 0.070 |
| B | translate_ctx | mean | 혼합 | 8 | 0.048 | -0.800 | 1.000 | 2.505 | 0.000 |
| B | translate_ctx | vanilla | 고유어쌍 | 12 | 0.860 | 0.388 | 0.992 | 4.876 | 0.182 |
| B | translate_ctx | vanilla | 한자어쌍 | 40 | 0.615 | 0.443 | 0.778 | 4.725 | 0.069 |
| B | translate_ctx | vanilla | 혼합 | 8 | 0.143 | -0.580 | 0.737 | 2.505 | 0.000 |

## 6. 양성 대조: 프롬프트에 영어 중간 단계

영어 경유를 강제했으므로 translate_en에서 간섭·영어 확률이 커지고 상관이 보여야 한다. 그렇지 않으면 간섭 지표가 영어 경유를 잡지 못하는 것이다.

| 과제 | 렌즈 | 평균 간섭 | 평균 영어 확률 | r [CI] |
|---|---|---|---|---|
| translate | affine | 1.778 | 0.175 | 0.227 [-0.112, 0.507] |
| translate | mean | 1.778 | 0.708 | -0.102 [-0.370, 0.187] |
| translate | vanilla | 1.778 | 0.683 | -0.118 [-0.380, 0.174] |
| translate_en | affine | 3.038 | 0.287 | 0.223 [-0.106, 0.505] |
| translate_en | mean | 3.038 | 0.819 | -0.226 [-0.507, 0.079] |
| translate_en | vanilla | 3.038 | 0.797 | -0.217 [-0.497, 0.089] |

## 7. C: 중간층 평균 확률 (정답 영어 vs 무관 영어, 언어별 확률 질량)

가설 1이 맞다면 바닐라에서는 무관한 영어(chair)와 영어 질량 전체가 높고, 평균 제거 렌즈에서 줄어야 한다.

| task | lens | lp_target | lp_unrelated | mass_en | mass_en_word | mass_zh | mass_ko |
|---|---|---|---|---|---|---|---|
| repeat | affine | 0.0072 | 0.0002 | 0.0348 | 0.0067 | 0.0793 | 0.7312 |
| repeat | mean | 0.1933 | 0.0000 | 0.6649 | 0.4415 | 0.0323 | 0.1996 |
| repeat | vanilla | 0.1442 | 0.0000 | 0.6353 | 0.3843 | 0.0411 | 0.1834 |
| translate | affine | 0.0629 | 0.0000 | 0.0984 | 0.0664 | 0.7438 | 0.0005 |
| translate | mean | 0.4911 | 0.0000 | 0.9255 | 0.8674 | 0.0493 | 0.0012 |
| translate | vanilla | 0.4588 | 0.0000 | 0.8981 | 0.8276 | 0.0643 | 0.0014 |

## 8. 층 구간별 언어 질량 (입력어 프롬프트 평균 확률)

`en`은 라틴 문자 2자 이상 토큰 전부(단어 조각 포함), `en_word`는 단어 시작 토큰이면서 영어 단어 목록에 있는 토큰만 센다. 둘의 차이가 크면 '영어'로 보이던 것의 상당 부분이 단어 조각이다. 구간은 렌즈 점수 구간 설정과 무관하게 전체 층의 1/4, 3/4로 나눈다.

| dataset | task | lens | band | en | en_word | zh | ko |
|---|---|---|---|---|---|---|---|
| A | translate | affine | 초반 1–6 | 0.036 | 0.006 | 0.649 | 0.019 |
| A | translate | affine | 중간 7–21 | 0.016 | 0.004 | 0.789 | 0.000 |
| A | translate | affine | 후반 22–27 | 0.154 | 0.100 | 0.721 | 0.000 |
| A | translate | mean | 초반 1–6 | 0.368 | 0.155 | 0.014 | 0.006 |
| A | translate | mean | 중간 7–21 | 0.721 | 0.416 | 0.023 | 0.010 |
| A | translate | mean | 후반 22–27 | 0.933 | 0.890 | 0.047 | 0.001 |
| A | translate | vanilla | 초반 1–6 | 0.396 | 0.160 | 0.016 | 0.007 |
| A | translate | vanilla | 중간 7–21 | 0.706 | 0.391 | 0.029 | 0.010 |
| A | translate | vanilla | 후반 22–27 | 0.912 | 0.856 | 0.058 | 0.001 |
| A | translate_en | affine | 초반 1–6 | 0.026 | 0.003 | 0.714 | 0.004 |
| A | translate_en | affine | 중간 7–21 | 0.024 | 0.007 | 0.763 | 0.000 |
| A | translate_en | affine | 후반 22–27 | 0.230 | 0.150 | 0.654 | 0.000 |
| A | translate_en | mean | 초반 1–6 | 0.375 | 0.153 | 0.015 | 0.007 |
| A | translate_en | mean | 중간 7–21 | 0.720 | 0.412 | 0.022 | 0.011 |
| A | translate_en | mean | 후반 22–27 | 0.933 | 0.891 | 0.046 | 0.000 |
| A | translate_en | vanilla | 초반 1–6 | 0.400 | 0.156 | 0.016 | 0.008 |
| A | translate_en | vanilla | 중간 7–21 | 0.708 | 0.392 | 0.028 | 0.011 |
| A | translate_en | vanilla | 후반 22–27 | 0.915 | 0.865 | 0.057 | 0.001 |
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
| affine | 0.664 |
| mean | 4.772 |
| vanilla | 4.015 |
