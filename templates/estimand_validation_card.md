# Estimand Validation Card

> LLM으로 측정한 변수를 쓰는 논문의 부록에 싣는 1쪽 보고 양식. (제안 표준, v0.1)

| 항목 | 기입 |
|---|---|
| 개념 / 라벨 정의 | |
| **Target estimand** θ (식과 설계행렬 X, 가중치 w) | |
| **Regime** | plug-in / correction (DSL · PPI · 기타) |
| 후보 모델 (snapshot ID) 과 프롬프트 해시 | |
| Audit 표본: 크기 n, 추출 방식 (무작위 / 정당 층화), 집단별 n_g | |
| Selection / inference 분할 방식 | split / K-fold cross-fit |
| 모델별 accuracy · F1 · balanced accuracy | 표 |
| 모델별 **estimand 오류** $w^\top\hat c_m$ (SE) | 표 |
| 모델별 **PPI 분산** $w^\top\hat V_m w$ | 표 |
| 집단별 FPR / FNR, 불변성 진단 ($\hat\tau^2$, 대비 간 최적 모델 일치 여부) | |
| 선택 규칙과 선택된 모델 | |
| 최종 θ̂, CI, 방법 | |
| 민감도: 다른 후보 모델로 추정 시 θ̂ 범위 | |
