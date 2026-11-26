# A 뜻별 영어 probe (S_sense)

층 수 28. r은 Spearman, 95% CI는 영어 단어(A)/한국어 입력어(B) 묶음 bootstrap.


## mid (25–75%): 7–21층

### 상관 r(지표, 간섭 I)

| task | lens | x | n | r | ci_lo | ci_hi |
|---|---|---|---|---|---|---|
| translate | affine | S_d | 99 | -0.011 | -0.207 | 0.185 |
| translate | affine | S_rel | 99 | -0.012 | -0.206 | 0.183 |
| translate | affine | S_d_diff | 99 | 0.390 | 0.211 | 0.557 |
| translate | mean | S_d | 99 | 0.178 | -0.040 | 0.372 |
| translate | mean | S_rel | 99 | 0.089 | -0.131 | 0.299 |
| translate | mean | S_d_diff | 99 | 0.358 | 0.173 | 0.527 |
| translate | vanilla | S_d | 99 | 0.216 | 0.005 | 0.419 |
| translate | vanilla | S_rel | 99 | 0.027 | -0.183 | 0.214 |
| translate | vanilla | S_d_diff | 99 | 0.367 | 0.179 | 0.539 |
| translate_en | affine | S_d | 99 | -0.051 | -0.252 | 0.144 |
| translate_en | affine | S_rel | 99 | -0.018 | -0.236 | 0.177 |
| translate_en | affine | S_d_diff | 99 | 0.506 | 0.329 | 0.658 |
| translate_en | mean | S_d | 99 | 0.425 | 0.242 | 0.587 |
| translate_en | mean | S_rel | 99 | 0.257 | 0.032 | 0.455 |
| translate_en | mean | S_d_diff | 99 | 0.458 | 0.297 | 0.592 |
| translate_en | vanilla | S_d | 99 | 0.455 | 0.274 | 0.607 |
| translate_en | vanilla | S_rel | 99 | 0.254 | 0.028 | 0.452 |
| translate_en | vanilla | S_d_diff | 99 | 0.445 | 0.277 | 0.590 |
| B translate (음성 대조) | affine | S_d | 60 | 0.484 | 0.317 | 0.649 |
| B translate (음성 대조) | mean | S_d | 60 | 0.705 | 0.588 | 0.797 |
| B translate (음성 대조) | vanilla | S_d | 60 | 0.699 | 0.572 | 0.795 |

### 평균 확률 (translate)

| lens | 정답 뜻 p | 방해 뜻 p | 방해/정답 |
|---|---|---|---|
| affine | 0.014 | 0.001 | 0.050 |
| mean | 0.308 | 0.000 | 0.001 |
| vanilla | 0.276 | 0.000 | 0.001 |

## late (75–97%): 21–27층

### 상관 r(지표, 간섭 I)

| task | lens | x | n | r | ci_lo | ci_hi |
|---|---|---|---|---|---|---|
| translate | affine | S_d | 99 | 0.185 | -0.025 | 0.375 |
| translate | affine | S_rel | 99 | 0.234 | 0.016 | 0.437 |
| translate | affine | S_d_diff | 99 | 0.418 | 0.221 | 0.585 |
| translate | mean | S_d | 99 | 0.266 | 0.037 | 0.465 |
| translate | mean | S_rel | 99 | 0.172 | -0.054 | 0.380 |
| translate | mean | S_d_diff | 99 | 0.454 | 0.272 | 0.598 |
| translate | vanilla | S_d | 99 | 0.231 | -0.010 | 0.446 |
| translate | vanilla | S_rel | 99 | 0.207 | -0.024 | 0.417 |
| translate | vanilla | S_d_diff | 99 | 0.474 | 0.305 | 0.612 |
| translate_en | affine | S_d | 99 | 0.393 | 0.198 | 0.560 |
| translate_en | affine | S_rel | 99 | 0.443 | 0.247 | 0.607 |
| translate_en | affine | S_d_diff | 99 | 0.436 | 0.238 | 0.589 |
| translate_en | mean | S_d | 99 | 0.533 | 0.342 | 0.677 |
| translate_en | mean | S_rel | 99 | 0.505 | 0.319 | 0.655 |
| translate_en | mean | S_d_diff | 99 | 0.512 | 0.343 | 0.651 |
| translate_en | vanilla | S_d | 99 | 0.461 | 0.249 | 0.628 |
| translate_en | vanilla | S_rel | 99 | 0.510 | 0.317 | 0.666 |
| translate_en | vanilla | S_d_diff | 99 | 0.485 | 0.315 | 0.631 |
| B translate (음성 대조) | affine | S_d | 60 | 0.736 | 0.630 | 0.827 |
| B translate (음성 대조) | mean | S_d | 60 | 0.757 | 0.676 | 0.829 |
| B translate (음성 대조) | vanilla | S_d | 60 | 0.750 | 0.662 | 0.829 |

### 평균 확률 (translate)

| lens | 정답 뜻 p | 방해 뜻 p | 방해/정답 |
|---|---|---|---|
| affine | 0.040 | 0.000 | 0.001 |
| mean | 0.313 | 0.000 | 0.000 |
| vanilla | 0.291 | 0.000 | 0.000 |
