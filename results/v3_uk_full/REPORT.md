# UK 확증 결과 (prereg-v3-uk)

사전등록 규칙 그대로의 전체 셀 보고. 1차 = `delta_pp`(pp, SESOI 3), 인간 범위 = 인간 코더 leave-one-out 추정치 [최소, 최대].

**연구 수준 결론 (H1: 서로 다른 계열 모델 2개 이상이 같은 집단·부호에서 비동등):** 지지

## economic 영역 — 1차 셀

| 모델 | 변형 | 대비 | 추정치 (pp) | 95% CI | 판정 | 인간 범위 | 인간 범위 밖 | 무효율 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | a | Con − Lab | +2.94 | [+0.53, +5.86] | inconclusive | [-5.0, +5.5] | False | 0.000 |
| Qwen2.5-7B-Instruct | a | LD − Lab | -0.70 | [-4.45, +3.32] | inconclusive | [-4.6, +4.1] | False | 0.000 |
| Qwen2.5-7B-Instruct | b | Con − Lab | +2.09 | [-0.51, +5.04] | inconclusive | [-5.0, +5.5] | False | 0.000 |
| Qwen2.5-7B-Instruct | b | LD − Lab | +1.52 | [-2.10, +4.51] | inconclusive | [-4.6, +4.1] | False | 0.000 |
| Qwen2.5-7B-Instruct | c | Con − Lab | +1.76 | [-0.39, +4.55] | inconclusive | [-5.0, +5.5] | False | 0.000 |
| Qwen2.5-7B-Instruct | c | LD − Lab | -0.49 | [-3.66, +2.91] | inconclusive | [-4.6, +4.1] | False | 0.000 |
| granite-3.3-8b-instruct | a | Con − Lab | +2.36 | [-0.72, +4.88] | inconclusive | [-5.0, +5.5] | False | 0.000 |
| granite-3.3-8b-instruct | a | LD − Lab | +1.88 | [-1.77, +5.50] | inconclusive | [-4.6, +4.1] | False | 0.000 |
| granite-3.3-8b-instruct | b | Con − Lab | +4.58 | [+1.89, +7.18] | non-equivalent | [-5.0, +5.5] | False | 0.000 |
| granite-3.3-8b-instruct | b | LD − Lab | +0.46 | [-2.51, +4.21] | inconclusive | [-4.6, +4.1] | False | 0.000 |
| granite-3.3-8b-instruct | c | Con − Lab | +2.18 | [-0.92, +4.57] | inconclusive | [-5.0, +5.5] | False | 0.000 |
| granite-3.3-8b-instruct | c | LD − Lab | +0.64 | [-2.72, +3.66] | inconclusive | [-4.6, +4.1] | False | 0.000 |
| Phi-3.5-mini-instruct | a | Con − Lab | +3.62 | [-0.03, +7.25] | inconclusive | [-5.0, +5.5] | False | 0.000 |
| Phi-3.5-mini-instruct | a | LD − Lab | +3.06 | [-1.33, +7.32] | inconclusive | [-4.6, +4.1] | False | 0.000 |
| Phi-3.5-mini-instruct | b | Con − Lab | +4.31 | [+1.36, +7.35] | non-equivalent | [-5.0, +5.5] | False | 0.000 |
| Phi-3.5-mini-instruct | b | LD − Lab | +2.63 | [-1.41, +6.72] | inconclusive | [-4.6, +4.1] | False | 0.000 |
| Phi-3.5-mini-instruct | c | Con − Lab | +2.98 | [-0.27, +5.72] | inconclusive | [-5.0, +5.5] | False | 0.000 |
| Phi-3.5-mini-instruct | c | LD − Lab | +1.94 | [-1.34, +5.27] | inconclusive | [-4.6, +4.1] | False | 0.000 |
| Mistral-7B-Instruct-v0.3 | a | Con − Lab | +4.97 | [+1.06, +8.21] | non-equivalent | [-5.0, +5.5] | False | 0.000 |
| Mistral-7B-Instruct-v0.3 | a | LD − Lab | +1.74 | [-1.24, +5.89] | inconclusive | [-4.6, +4.1] | False | 0.000 |
| Mistral-7B-Instruct-v0.3 | b | Con − Lab | +5.54 | [+3.68, +7.58] | non-equivalent | [-5.0, +5.5] | True | 0.000 |
| Mistral-7B-Instruct-v0.3 | b | LD − Lab | +1.91 | [-0.99, +5.81] | inconclusive | [-4.6, +4.1] | False | 0.000 |
| Mistral-7B-Instruct-v0.3 | c | Con − Lab | +5.18 | [+1.93, +8.29] | non-equivalent | [-5.0, +5.5] | False | 0.000 |
| Mistral-7B-Instruct-v0.3 | c | LD − Lab | -0.06 | [-3.13, +3.83] | inconclusive | [-4.6, +4.1] | False | 0.000 |

### economic 영역 — 모델 수준 결론

| 모델 | 대비 | 결론 | 부호 | 추정치 중앙값 | 인간 범위 밖 변형 수 |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | Con − Lab | inconclusive | +0 | +2.09 | 0 |
| Qwen2.5-7B-Instruct | LD − Lab | inconclusive | +0 | -0.49 | 0 |
| granite-3.3-8b-instruct | Con − Lab | inconclusive | +0 | +2.36 | 0 |
| granite-3.3-8b-instruct | LD − Lab | inconclusive | +0 | +0.64 | 0 |
| Phi-3.5-mini-instruct | Con − Lab | inconclusive | +0 | +3.62 | 0 |
| Phi-3.5-mini-instruct | LD − Lab | inconclusive | +0 | +2.63 | 0 |
| Mistral-7B-Instruct-v0.3 | Con − Lab | non-equivalent | +1 | +5.18 | 1 |
| Mistral-7B-Instruct-v0.3 | LD − Lab | inconclusive | +0 | +1.74 | 0 |

### economic 영역 — 2차 오류율 통계 판정 집계 (변형 3개)

| 모델 | 통계 | 판정 분포 |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | fpr_diff:1 | {'equivalent': np.int64(3)} |
| Mistral-7B-Instruct-v0.3 | fpr_diff:2 | {'equivalent': np.int64(3)} |
| Mistral-7B-Instruct-v0.3 | sens_diff:1 | {'non-equivalent': np.int64(3)} |
| Mistral-7B-Instruct-v0.3 | sens_diff:2 | {'inconclusive': np.int64(2), 'non-equivalent': np.int64(1)} |
| Phi-3.5-mini-instruct | fpr_diff:1 | {'equivalent': np.int64(2), 'inconclusive': np.int64(1)} |
| Phi-3.5-mini-instruct | fpr_diff:2 | {'equivalent': np.int64(2), 'inconclusive': np.int64(1)} |
| Phi-3.5-mini-instruct | sens_diff:1 | {'non-equivalent': np.int64(3)} |
| Phi-3.5-mini-instruct | sens_diff:2 | {'non-equivalent': np.int64(3)} |
| Qwen2.5-7B-Instruct | fpr_diff:1 | {'equivalent': np.int64(2), 'inconclusive': np.int64(1)} |
| Qwen2.5-7B-Instruct | fpr_diff:2 | {'inconclusive': np.int64(2), 'equivalent': np.int64(1)} |
| Qwen2.5-7B-Instruct | sens_diff:1 | {'inconclusive': np.int64(3)} |
| Qwen2.5-7B-Instruct | sens_diff:2 | {'equivalent': np.int64(2), 'inconclusive': np.int64(1)} |
| granite-3.3-8b-instruct | fpr_diff:1 | {'inconclusive': np.int64(2), 'equivalent': np.int64(1)} |
| granite-3.3-8b-instruct | fpr_diff:2 | {'equivalent': np.int64(2), 'inconclusive': np.int64(1)} |
| granite-3.3-8b-instruct | sens_diff:1 | {'non-equivalent': np.int64(3)} |
| granite-3.3-8b-instruct | sens_diff:2 | {'non-equivalent': np.int64(2), 'inconclusive': np.int64(1)} |

## social 영역 — 1차 셀

| 모델 | 변형 | 대비 | 추정치 (pp) | 95% CI | 판정 | 인간 범위 | 인간 범위 밖 | 무효율 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | a | Con − Lab | -4.90 | [-8.63, -0.71] | non-equivalent | [-3.9, +2.9] | True | 0.000 |
| Qwen2.5-7B-Instruct | a | LD − Lab | +0.64 | [-2.73, +3.95] | inconclusive | [-1.5, +1.8] | False | 0.000 |
| Qwen2.5-7B-Instruct | b | Con − Lab | -1.14 | [-5.56, +4.11] | inconclusive | [-3.9, +2.9] | False | 0.000 |
| Qwen2.5-7B-Instruct | b | LD − Lab | -0.86 | [-4.48, +2.49] | inconclusive | [-1.5, +1.8] | False | 0.000 |
| Qwen2.5-7B-Instruct | c | Con − Lab | -3.01 | [-6.44, +1.03] | inconclusive | [-3.9, +2.9] | False | 0.000 |
| Qwen2.5-7B-Instruct | c | LD − Lab | +1.68 | [-0.52, +3.70] | inconclusive | [-1.5, +1.8] | False | 0.000 |
| granite-3.3-8b-instruct | a | Con − Lab | -5.33 | [-10.62, +0.15] | inconclusive | [-3.9, +2.9] | True | 0.000 |
| granite-3.3-8b-instruct | a | LD − Lab | -1.46 | [-5.70, +2.44] | inconclusive | [-1.5, +1.8] | False | 0.000 |
| granite-3.3-8b-instruct | b | Con − Lab | -6.38 | [-11.58, -0.72] | non-equivalent | [-3.9, +2.9] | True | 0.000 |
| granite-3.3-8b-instruct | b | LD − Lab | -0.11 | [-2.96, +2.37] | equivalent | [-1.5, +1.8] | False | 0.000 |
| granite-3.3-8b-instruct | c | Con − Lab | -4.71 | [-9.15, +0.37] | inconclusive | [-3.9, +2.9] | True | 0.000 |
| granite-3.3-8b-instruct | c | LD − Lab | +1.24 | [-2.36, +4.62] | inconclusive | [-1.5, +1.8] | False | 0.000 |
| Phi-3.5-mini-instruct | a | Con − Lab | -5.54 | [-9.78, -0.58] | non-equivalent | [-3.9, +2.9] | True | 0.000 |
| Phi-3.5-mini-instruct | a | LD − Lab | -0.28 | [-3.82, +3.29] | inconclusive | [-1.5, +1.8] | False | 0.000 |
| Phi-3.5-mini-instruct | b | Con − Lab | -6.61 | [-11.06, -0.89] | non-equivalent | [-3.9, +2.9] | True | 0.000 |
| Phi-3.5-mini-instruct | b | LD − Lab | -1.15 | [-5.19, +2.78] | inconclusive | [-1.5, +1.8] | False | 0.000 |
| Phi-3.5-mini-instruct | c | Con − Lab | -5.55 | [-10.30, +0.11] | inconclusive | [-3.9, +2.9] | True | 0.000 |
| Phi-3.5-mini-instruct | c | LD − Lab | +1.44 | [-2.12, +5.68] | inconclusive | [-1.5, +1.8] | False | 0.000 |
| Mistral-7B-Instruct-v0.3 | a | Con − Lab | -6.97 | [-11.99, -2.32] | non-equivalent | [-3.9, +2.9] | True | 0.000 |
| Mistral-7B-Instruct-v0.3 | a | LD − Lab | -1.22 | [-5.11, +2.58] | inconclusive | [-1.5, +1.8] | False | 0.000 |
| Mistral-7B-Instruct-v0.3 | b | Con − Lab | -7.51 | [-11.47, -3.15] | non-equivalent | [-3.9, +2.9] | True | 0.000 |
| Mistral-7B-Instruct-v0.3 | b | LD − Lab | -1.20 | [-4.52, +1.61] | inconclusive | [-1.5, +1.8] | False | 0.000 |
| Mistral-7B-Instruct-v0.3 | c | Con − Lab | -6.59 | [-11.47, -1.34] | non-equivalent | [-3.9, +2.9] | True | 0.000 |
| Mistral-7B-Instruct-v0.3 | c | LD − Lab | +0.71 | [-3.11, +4.18] | inconclusive | [-1.5, +1.8] | False | 0.000 |

### social 영역 — 모델 수준 결론

| 모델 | 대비 | 결론 | 부호 | 추정치 중앙값 | 인간 범위 밖 변형 수 |
|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | Con − Lab | inconclusive | +0 | -3.01 | 1 |
| Qwen2.5-7B-Instruct | LD − Lab | inconclusive | +0 | +0.64 | 0 |
| granite-3.3-8b-instruct | Con − Lab | inconclusive | +0 | -5.33 | 3 |
| granite-3.3-8b-instruct | LD − Lab | inconclusive | +0 | -0.11 | 0 |
| Phi-3.5-mini-instruct | Con − Lab | non-equivalent | -1 | -5.55 | 3 |
| Phi-3.5-mini-instruct | LD − Lab | inconclusive | +0 | -0.28 | 0 |
| Mistral-7B-Instruct-v0.3 | Con − Lab | non-equivalent | -1 | -6.97 | 3 |
| Mistral-7B-Instruct-v0.3 | LD − Lab | inconclusive | +0 | -1.20 | 0 |

### social 영역 — 2차 오류율 통계 판정 집계 (변형 3개)

| 모델 | 통계 | 판정 분포 |
|---|---|---|
| Mistral-7B-Instruct-v0.3 | fpr_diff:1 | {'non-equivalent': np.int64(3)} |
| Mistral-7B-Instruct-v0.3 | fpr_diff:2 | {'equivalent': np.int64(2), 'inconclusive': np.int64(1)} |
| Mistral-7B-Instruct-v0.3 | sens_diff:1 | {'equivalent': np.int64(3)} |
| Mistral-7B-Instruct-v0.3 | sens_diff:2 | {'equivalent': np.int64(3)} |
| Phi-3.5-mini-instruct | fpr_diff:1 | {'non-equivalent': np.int64(3)} |
| Phi-3.5-mini-instruct | fpr_diff:2 | {'equivalent': np.int64(2), 'inconclusive': np.int64(1)} |
| Phi-3.5-mini-instruct | sens_diff:1 | {'inconclusive': np.int64(2), 'equivalent': np.int64(1)} |
| Phi-3.5-mini-instruct | sens_diff:2 | {'inconclusive': np.int64(1), 'equivalent': np.int64(1), 'non-equivalent': np.int64(1)} |
| Qwen2.5-7B-Instruct | fpr_diff:1 | {'inconclusive': np.int64(2), 'non-equivalent': np.int64(1)} |
| Qwen2.5-7B-Instruct | fpr_diff:2 | {'equivalent': np.int64(3)} |
| Qwen2.5-7B-Instruct | sens_diff:1 | {'inconclusive': np.int64(3)} |
| Qwen2.5-7B-Instruct | sens_diff:2 | {'inconclusive': np.int64(2), 'non-equivalent': np.int64(1)} |
| granite-3.3-8b-instruct | fpr_diff:1 | {'inconclusive': np.int64(2), 'non-equivalent': np.int64(1)} |
| granite-3.3-8b-instruct | fpr_diff:2 | {'equivalent': np.int64(2), 'inconclusive': np.int64(1)} |
| granite-3.3-8b-instruct | sens_diff:1 | {'equivalent': np.int64(2), 'inconclusive': np.int64(1)} |
| granite-3.3-8b-instruct | sens_diff:2 | {'inconclusive': np.int64(2), 'equivalent': np.int64(1)} |

## 해석 시 반드시 함께 읽을 것

- UK 규모의 검정력 한계: 95% CI 폭 약 5–10pp, 약 5pp 이상만 판별(docs/12 D11). 판정 불가는 동등성의 증거가 아니다.
- 인간 코더도 경제 영역에서 3–5pp 비동등(크라우드, 전문가 2명; D17). "인간보다 더 비동등"은 인간 범위 밖 기준을 함께 만족할 때만.
- 사회 영역은 유병률이 낮아 추정량 수준 동등이 오류율 동등을 뜻하지 않는다(D19). 2차 통계를 함께 볼 것.
- 계열 p는 실제 자료에서 반보수적일 수 있다(D17). 판정은 bootstrap CI 조건을 함께 요구한다.
- 오염 가능성: Benoit 2016 자료는 공개 저장소에 코드와 함께 있다. 동등 결과의 일부가 암기일 수 있다.
