# 03 · Estimand-Aware Validation (EAV) — 절차 명세 v0.2

## 연구자가 따라 할 절차 (논문의 "recommendation box")

1. **Estimand를 사전 선언한다.** $\theta=w^\top\beta$ (정당 격차, 추세, 정당×시간 등). 설계행렬 $X$와 가중치 $w$로 적는다.
2. **Regime을 선언한다.** Plug-in(교정 없이 대입)인가, Correction(DSL/PPI)인가.
3. **Audit를 설계한다.** 무작위 또는 **정당 층화** 표본. Correction regime이면 소수정당을 과대표집한다 ($n_g\propto\sqrt{v_g}$, 02_theory P4 따름정리). 목표 n ≥ 300 (S2 근거).
4. **Selection fold와 inference fold를 나눈다** (K-fold cross-fitting 권장).
5. Selection fold에서 각 후보 모델의 **error regression** $e_m\sim X$를 적합한다.
   - Plug-in: `eav_structural` 손실 최소 모델
   - Correction: `eav_efficiency` 손실(= $w^\top\hat V_m w$) 최소 모델
6. **불변성 진단을 보고한다.** 대비 estimand들 사이 최적 모델이 바뀌는지, 정당별 FPR/FNR 차이 검정.
7. Inference fold(또는 cross-fit)에서 DSL/PPI로 최종 추정과 CI를 낸다.
8. **Estimand Validation Card**를 부록에 싣는다 (`templates/estimand_validation_card.md`).

## 기준의 정의 (코드: `src/eav/audit.py`)

| 이름 | 손실 (작을수록 좋음) | 쓰는 곳 |
|---|---|---|
| `accuracy`, `f1`, `balanced_accuracy` | 1 − 지표 | 기준선 |
| `eav_plugin_naive` | $(w^\top\hat c_m)^2$ | 기준선 (EAV-direct) |
| `eav_plugin` | $(w^\top\hat c_m)^2 - w^\top\hat V_m w$ | **사용 금지** (ADR-0004), 기록용 |
| `eav_structural` | $\tilde b_m^2$, $\tilde b_m=b^{inv}_m+\frac{\hat\tau^2}{\hat\tau^2+\hat V_m}(\hat b_m-b^{inv}_m)$, $b^{inv}_m=-(\widehat{FPR}_m+\widehat{FNR}_m)\hat\theta_H$ | Plug-in 기본값 |
| `eav_efficiency` | $w^\top\hat V_m w$ | Correction 기본값 |

$\hat\tau^2=\max\{0,\ \overline{(\hat b_m-b^{inv}_m)^2-\hat V_m}\}$: 후보 모델들 사이 "불변성으로 설명되지 않는 편향"의 분산을 적률법으로 추정한 값이다. 차등 오류가 없으면 $\hat\tau^2\to0$이 되어 Youden 기준으로 수렴하고, 크면 직접 추정으로 이동한다. **연구자가 불변성 검정을 따로 할 필요 없이 자료가 가중치를 정한다**는 점이 방법론적 매력이다.

## 알려진 약점과 대응 계획

| 약점 | 대응 |
|---|---|
| 후보 모델 간 오류 상관(같은 어려운 문서에서 함께 틀림) — 시뮬레이션은 조건부 독립 가정 | 실제 LLM 예측으로 S2 재현; 문서 난이도 random effect를 넣은 DGP 추가 (S4) |
| $\hat\tau^2$ 적률 추정은 M=5–8이면 불안정 | 모델 수 민감도; 계층 베이즈 대안 |
| 수준 estimand(prevalence)에는 structural 정의 불가 | direct + quantification 문헌(ACC/PACC) 기준선 추가 |
| 선택과 교정에 같은 audit 사용 | cross-fitting; coverage 시뮬레이션 (S5) |
| 선택 대신 혼합(MoE-PPI)이 더 효율적일 수 있음 | 비용 조정 비교: 혼합은 M개 모델을 전체 코퍼스에 적용해야 함 |

## 예정 시뮬레이션

- **S4** 상관된 오류 + 문서 난이도
- **S5** 선택 후 교정 추론의 coverage (split vs cross-fit vs naive reuse)
- **S6** 층화 audit 배분 (균등 vs 비례 vs Neyman)
- **S7** EAV-selection vs MoE-PPI, 코퍼스 라벨링 비용을 반영한 비교
