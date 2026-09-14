# 00 · 적합성 · 발전가능성 · 게재가능성 평가

> 평가일 2026-09-14. 근거: 문헌 검증(원 메모 인용 10건), 2024–2026 경쟁연구 스캔, 이론 명제의 수치 검증(`tests/`), 시뮬레이션 S1–S3.
> 확률은 **주관적 판단**이며, 학술지 공식 통계가 아니다.

## 결론 한 줄

**조건부 GO.** 원 메모의 형태("최고 F1 모델 ≠ 최고 estimand 모델" + 이진 toy theorem + EAV = 추정 MSE 최소화)는 **이미 알려진 결과와 겹치고, EAV 핵심 추정량이 시뮬레이션에서 실패한다.** 반면 **두 regime 분리 + $1/\pi_g$ 효율 결과 + 불변성 수축형 EAV**로 재구성하면 Political Analysis급 기여가 남는다. 판정은 파일럿(2026년 11월 게이트)이 내린다.

## 1. 점수표

| 차원 | 원 메모 그대로 | 재구성(v0.2) 후 | 근거 |
|---|---|---|---|
| 문제의 중요성 | 8 | 8.5 | LLM 측정이 AJPS급 연구 관행에 진입 (Benoit et al.; Egami et al. cond. acc.) |
| 참신성 | **5** | 7 | 불변 오분류→Youden은 Bross 1954; "분산이 중요"는 MoE-PPI/MultiPPI 2026; 경험적 문제제기는 Baumann 2025 |
| 이론적 깊이 | 4 | 7 | 이진 toy → P0 error regression, P3 switching 진단, P4 $1/\pi_g$ |
| 방법의 실효성 | 3 | 6 (조건부) | debiased EAV는 실패. structural/efficiency EAV는 n≳200–400이고 차등 오류가 있을 때만 이긴다 (S2) |
| 실현가능성 | 8 | 8 | 새 human coding 불필요, 파일럿은 소형 GPU/API로 가능 |
| 정치학적 payoff | 4 | 6→8 | Manifesto 적용에서 실제 결론이 바뀌는지에 달림 (미확인) |
| 선점 시간 압박 | — | **높음** | 2026년 1–9월에만 인접 논문 6편 이상 (01_positioning §2) |
| **종합** | **5.5** | **7.5** (파일럿 GO 시 8+) | |

원 메모의 자체 평가 "8.5–9/10"은 **낙관적**이다. 그 차이는 대부분 (i) 역학의 오분류 문헌, (ii) 2026년 PPI 다중 predictor 문헌을 반영하지 않은 데서 온다.

## 2. 적합성 (Fit)

**Political Analysis — 적합 (높음).** 방법론 + 정치 텍스트 측정 + 소프트웨어라는 PA의 핵심 영역이다. Grimmer–Stewart(2013), Mikhaylov et al.(2012), Halterman–Keith(2026)의 계보에 그대로 놓인다. PA 심사자가 반드시 물을 세 질문과 현재 답:

| 심사자 질문 | 현재 답 | 상태 |
|---|---|---|
| "그냥 balanced accuracy 쓰면 되지 않나?" | 불변이면 맞다(P1, S1에서 100%). 차등 오류가 있으면 BA도 25–30%로 떨어진다. EAV-structural은 불변이면 BA로 수렴한다 | ✅ 답 있음 |
| "DSL로 교정하면 모델 선택은 무관하지 않나?" | 편향은 무관하지만 분산은 아니다. 소수정당 대비에서는 정확도 기준 선택이 유효표본을 최대 28% 잃는다 (P4, S1) | ✅ 답 있음 |
| "실제 정치학 결론이 바뀐 사례가 있나?" | 없음 | ❌ **파일럿과 Manifesto 적용이 해결해야 함** |

**AJPS — 조건부 적합.** 방법론만으로는 부족하다. "LLM이 정당을 공평하게 측정하는가"라는 **실질적 발견**(예: 급진우파 정당 텍스트에서 체계적 차등 오류 → 이슈 소유 결론 변화)이 동반되어야 한다.

**대안 학술지.** PSRM(짧은 방법론 + 적용), *Communication Methods and Measures*(TeBlunthuis·Laurer 계보), *Sociological Methods & Research*. P4만 떼어 통계·ML 학회에 내는 것은 MultiPPI 계열과 정면 경쟁이라 비추천한다.

## 3. 발전가능성 (Development potential)

**연구 프로그램으로 3편이 가능하다.**

| 편 | 핵심 | 목표 | 선결 조건 |
|---|---|---|---|
| **Flagship** | 두 regime 이론(P0–P4) + EAV + Baumann 파일럿 + CMP 적용 | PA | 파일럿 GO |
| **Substantive** | "Do LLMs Measure Parties Equally?" CMP에서 정당가족별 차등 오류와 이슈 소유·양극화 결론의 민감도 (오염 통제) | AJPS / BJPolS / CPS | Flagship의 CMP 파이프라인 |
| **Tool / standard** | `eav` 패키지(Python/R) + *Estimand Validation Card* 보고 표준 | PSRM letter / JOSS | 코드 안정화 |

**확장 축 (우선순위 순):**
1. **층화 audit 설계**: $n_g\propto\sqrt{v_g}$. 비용이 거의 들지 않는 실무 권고다. Ye et al. 2026, Stratified PPI와의 차별화가 필요하다.
2. **다국어 = 집단**: 언어별 차등 오류는 비교정치 연구에서 치명적이다 (Benoit et al.의 21개 언어 설정).
3. 다범주·연속 척도(RILE, logit scale).
4. 선택 vs 혼합(MoE-PPI)의 비용-효율 경계.
5. 프롬프트×모델 상호작용 (Baumann과 연결).

## 4. 게재가능성 (주관적 확률)

| 시나리오 | PA 최종 게재 | AJPS | PSRM/CMM/SMR |
|---|---|---|---|
| A. 원 메모 수준 (toy theorem + Baumann 파일럿만) | ≤ 10% | < 5% | 25–35% |
| B. v0.2 이론 + S1–S7 + 파일럿 GO, CMP 적용 없음 | 20–30% | < 10% | 45–55% |
| C. B + CMP 적용에서 **결론이 바뀐 실제 사례** + 오염 통제 + 패키지 | **35–45%** | 15–25% (실질 발견이 강할 때) | 60%+ |
| D. 파일럿 PIVOT-A (BA가 EAV와 대등) | 10–15% | — | 40–50% (지표 논문) |

확률을 가장 크게 움직이는 변수는 순서대로 다음과 같다. ① 실제 정치 결론 변화 사례(C vs B: +15pp). ② 2027년 상반기 전 경쟁 논문의 추가 선점 여부. ③ EAV가 실제 LLM 오류 상관 구조에서도 n≈300에서 이기는지(S4 + 파일럿).

## 5. 원 메모에서 반드시 고칠 것 (심각도 순)

1. **EAV 추정량**: "bias² + variance 최소화"의 불편추정은 잡음이 큰 모델을 고른다 (S2: δ=0에서 BA보다 초과오차 4–26배). → structural 수축으로 교체 (ADR-0004).
2. **"작은 audit에서도 EAV가 이긴다"**: 현재 근거 없음. n≤100에서는 라벨 지표가 낫다. → 주장을 "n≳200–400, 차등 오류 조건"으로 한정.
3. **EAV→DSL 논리 긴장**: 교정하면 편향 기준 선택이 무의미해진다. → 두 regime 분리 (02_theory §0).
4. **P(m_F = m_θ)를 GO/KILL 기준으로 사용**: argmin은 δ=0.1에서도 84% 뒤집히는 불안정한 통계량이다 (S3). → 상대 regret + bootstrap (ADR-0002).
5. **Prop. 1을 신규 정리로 제시**: Bross/Copeland와 동일하고 BA로 해소된다. → "도구"로 격하. 신규성은 P3·P4에 둔다.
6. **인용 오류 3건 + 미확인 2건** (01_positioning §1).
7. **Baumann 예측값 사용 가정**: 미공개. → 재생성 비용을 계획에 반영.
8. **CMP gold의 잡음과 오염**: 원 메모에 없음. → 05_data_plan.

## 6. 가장 강한 한 문장 (재구성 후)

> *Whether an LLM is a good measurement instrument depends on the estimand and on how its labels enter the analysis: without correction, the best model minimizes the projection of its errors onto the estimand's design; with correction, it minimizes their variance — and for comparisons involving small parties, neither is the model with the highest accuracy.*
