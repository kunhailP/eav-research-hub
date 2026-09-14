# 06 · Publication strategy & reviewer attack matrix

## 1. 가제와 초록 초안 (Flagship, PA)

**Title:** *The Best Classifier Is Not the Best Instrument: Estimand-Aware Selection of Language Models for Political Measurement*

**Abstract (draft v0.1, ~150 words).** Political scientists increasingly measure concepts in text with large language models, choosing among candidate models by their accuracy or F1 on a small human-coded set. We show that this choice is generally misaligned with the quantities researchers publish. When model labels are substituted into the corpus, the relevant loss is the projection of a model's errors onto the estimand's design; when labels are bias-corrected with human audits, it is the variance of that projection. Group comparisons make both diverge from accuracy: party-specific error can reverse the sign of a party gap at 94.5% accuracy, and the efficiency of a corrected gap estimate weights each party's errors by the inverse of its share, so accuracy-based selection systematically fails for small parties. We propose Estimand-Aware Validation, a selection rule that shrinks toward measurement invariance, and evaluate it on [N] benchmark tasks and the Manifesto Project corpus.

## 2. 투고 경로

| 순서 | 대상 | 시점 | 비고 |
|---|---|---|---|
| 0 | arXiv 선공개 (이론 + 시뮬레이션 + 파일럿) | 파일럿 GO 후 ~8주 | **선점 방어.** 2026년 경쟁 속도 때문에 필수 |
| 1 | PolMeth / APSA / EPSA 발표 | 다음 사이클 | 마감일은 공고에서 확인 |
| 2 | **Political Analysis** | CMP 적용 완료 후 | 재현 패키지(Dataverse) 필수: 텍스트 제외, 스크립트와 ID만 |
| 2′ | AJPS (Substantive 편) | Flagship 이후 | 실질 발견이 강할 때만 |
| 3 | 거절 시: PSRM → CMM | | 원고 구조를 모듈화해 짧은 판본 준비 |

## 3. 심사자 공격 매트릭스

| # | 예상 공격 | 강도 | 준비된 답 | 보강 작업 |
|---|---|---|---|---|
| R1 | "Balanced accuracy / Youden을 쓰면 끝나는 문제" | 높음 | P1은 인정. 차등 오류 하에서는 BA도 실패 (S1: 25–30%). EAV-structural은 불변 시 BA로 수렴 | 실제 자료에서 BA vs EAV |
| R2 | "Baumann et al.과 무엇이 다른가" | 높음 | 그들은 결론 오류를 측정. 우리는 선택 목적함수 + 이론 + 규칙 | 차별화 표 (01_positioning) |
| R3 | "DSL/PPI면 선택은 무관" | 높음 | P4: 분산은 무관하지 않고, 1/π 가중 | S5 coverage |
| R4 | "MoE-PPI처럼 섞으면 되지 왜 고르나" | 중–높음 | 코퍼스 라벨링 비용 M배. 해석 가능성 | S7 비용 조정 비교 |
| R5 | "시뮬레이션 DGP가 임의적" | 중 | 실제 LLM 오류로 재현 (파일럿). 오류 상관 DGP | S4 |
| R6 | "CMP gold도 오류투성이" | 중 | human-coding estimand 정의 + 민감도 | 05_data_plan B |
| R7 | "LLM이 CMP를 외웠다" | 중 | 시간 holdout + memorization probe | 05_data_plan B |
| R8 | "모델 버전이 사라져 재현 불가" | 중 | open-weight 주분석, 원응답 캐시 | models.lock.json |
| R9 | "선택과 교정에 같은 audit → winner's curse" | 중 | cross-fitting | S5 |
| R10 | "n=300 audit는 비현실적" | 낮–중 | 대부분 LLM validation 연구가 이미 수백 건. 층화로 절감 | S6 |
| R11 | "정치학적으로 무엇이 바뀌나" | **결정적** | 현재 답 없음 | **CMP 적용이 해결** |

## 4. 저자 구성 권고

- 통계 파트(P0–P4 증명, cross-fitting coverage) 공저자 1명: PPI/semiparametric 배경.
- 비교정치 파트(CMP 정당가족, 이슈 소유 해석) 공저자 1명: 실질 기여를 AJPS급으로 끌어올리는 역할.
- 원저자는 이론 설계, 파이프라인, 집필을 맡는다.
