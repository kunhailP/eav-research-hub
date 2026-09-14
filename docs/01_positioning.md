# 01 · Positioning & literature map (검증 상태 포함)

> 2026-09-14 기준. 인용은 `lit/references.bib`. "검증" 열: ✔ 원문·출판사 확인, ◐ 초록 수준, ✎ 원 메모의 인용 오류를 수정함.

## 1. 원 메모 인용의 수정 사항

| 원 메모 | 수정 |
|---|---|
| Korobeynikova et al. (2025) | ✎ **Alizadeh, Kubli, Samei, …, Korobeynikova, Gilardi (2025)**, *J. Computational Social Science* 8(1). Korobeynikova는 8명 중 7번째 저자 |
| Baumann: "F1 0.93인데 risk 50%인 사례" | ✎ 원문: "several tasks with F1 scores exceeding 0.93 still exhibit hacking risks above 50%" |
| Egami et al. AJPS | ✔ **conditionally accepted** (출판 전), 2025 Gosnell Prize ✔ |
| Baumann repo에서 예측값 사용 | ✎ 예측값 미공개 → 재생성 필요 |
| Benoit et al. (2026, AJPS) | ◐ DOI 10.1111/ajps.70050 확인. 연도·권호는 미확인 → "Early View"로 인용. 모델은 Claude/GPT/Gemini 앙상블, 21개 언어 235개 선언문, 6개 차원, 전문가 조사와 r≈.87–.92 |
| Laurer et al. | ✔ *CMM* 19(1) **2025** (온라인 2024) |
| Heseltine & Clemm von Hohenberg | ✔ *Research & Politics* 11(1), 2024 |
| TeBlunthuis, Hase & Chan | ✔ *CMM* 18(3), 2024 |

## 2. 선점 위협 순위 (원 메모에 없던 2026년 논문 포함)

| # | 논문 | 무엇을 선점하나 | 위협 | 대응 문장 |
|---|---|---|---|---|
| 1 | Baumann et al. 2025, arXiv 2509.08825 | "높은 F1 ≠ 안정적 결론"의 경험적 증거, best-performer 선택 (+4pp) | 높음 (a) | 우리는 **선택 목적함수**를 바꾸고, 왜 어긋나는지 이론(P0–P4)을 준다 |
| 2 | Gu, Kong & Xia 2026, arXiv 2604.27892 (MoE-PPI) | 여러 predictor를 **분산 최소화** 가중으로 결합 | 높음 (d) | 선택은 코퍼스 라벨링 비용이 1/M. 우리의 $1/\pi_g$ 결과는 집단 대비에 특화됨. S7에서 직접 비교 |
| 3 | Cowen-Breen et al. 2026, arXiv 2603.27414 (MultiPPI); Brawand et al. 2026, arXiv 2605.08429 | 여러 LLM proxy에 예산 배분 | 중–높음 (c,d) | 이들은 estimand 일반의 효율. 우리는 plug-in regime과 정치 집단 대비, 불변성 진단 |
| 4 | Kotte 2026, arXiv 2606.12426 | 클래스별 오류 차이로 prevalence가 우연히 맞는 사례, 결론 뒤집힘 진단 | 중 (a) | 진단은 있으나 선택 규칙과 이론이 없음 |
| 5 | Wang, Hunt, Tang & Joseph 2026, arXiv 2609.07987 (**9월 7일**) | "fidelity는 substitutability의 필요·충분조건이 아니다" 프레이밍 | 중 (프레이밍) | 설문 시뮬레이션 영역. 텍스트 분류·집단 대비로 차별화. **프레이밍 문장이 겹치므로 서론에서 명시 인용** |
| 6 | Chen, Lu, Li, Guo & Li 2026, arXiv 2601.05420; Chen, Guo & Li 2026, arXiv 2603.16041 | Rogan–Gladen 교정과 PPI 통일, PPI power 공식 | 중 (b,d) | 우리 P4는 이들의 분산식을 집단 대비로 특화하고, 선택 문제로 전환 |
| 7 | Egami et al. 2023 NeurIPS; AJPS cond. acc. | DSL 하에서 예측 품질은 효율에만 영향 | 중 (d) | 우리는 "그렇다면 어떤 모델이 효율적인가"에 답함 |
| 8 | Ludwig, Mullainathan & Rambachan, arXiv 2412.07031 | 추정 용도에는 validation 표본 필요 | 낮–중 | 일반 원칙 → 구체적 기준 |
| 9 | Camuffo et al. 2026, arXiv 2601.02370 | 공변량과 상관된 오류는 정확도와 무관하게 편향 | 중 (a,b) | 프로토콜만 있음. 선택 이론 없음 |
| 10 | Fong & Tyler 2021 PA; Knox, Lucas & Cho 2022 ARPS; TeBlunthuis et al. 2024 | 예측 변수 편향과 교정 | 낮–중 (b) | 고전 배경 |
| 11 | Ye, Lyu & Tao 2026, arXiv 2604.12497 | segment별 AI 정확도에 따라 human label 배분 | 중 (P4 따름정리) | 층화 audit 결과와 겹침 → 인용 후 정치 집단 적용으로 한정 |
| 12 | Vallejo Vera & Driggers 2025 HSSC; Lerner & Yvon 2026 arXiv 2606.05937 | LLM 라벨이 **정당 단서**에 따라 변함 | **기회** | 차등 오류 가정의 경험적 근거로 활용 |
| 13 | Halterman & Keith 2026 PA (Codebook LLMs) | CMP 오염 위험 | **기회/필수 인용** | 오염 설계 (05_data_plan) |
| 14 | Ornstein, Blasingame & Truscott 2024 ("How to Train Your Stochastic Parrot"; 학술지 확인 필요) | **같은 Benoit 2016 UK 문장**에 GPT-3/4를 적용해 선언문 수준 추정치를 비교 | 중 (자료 중복) | 선언문 수준 일치를 보였을 뿐, 정당 비교·차등 오류는 다루지 않음. 공개된 GPT-3 라벨은 추가 '모델'로 재사용 가능 |

**비선점 확인 (고전):** Bross 1954 *Biometrics*; Copeland et al. 1977 *AJE* (불변 오분류 → 영가설 방향 감쇠, 차등 → 양방향). Rogan & Gladen 1978 (prevalence 교정)은 원문 확인이 필요하다.

## 3. 무엇이 여전히 새로운가 (논문에서 주장할 수 있는 것)

1. **Estimand별 선택 규칙 (EAV-structural / EAV-efficiency)**. 후보 LLM 중 하나를 estimand 위험으로 고르는 명시적 규칙을 제시한 논문은 확인되지 않았다.
2. **P4: 집단 대비 효율에서의 $1/\pi_g$ 가중.** 소수정당 연구에서 정확도 기반 선택이 체계적으로 실패하는 이유.
3. **P3: estimand switching = 차등 측정오류의 진단.** 불변성 하에서는 모든 대비의 최적 모델이 같다.
4. **Error-regression 통일 틀 (P0).** plug-in 편향과 correction 분산을 하나의 회귀로 계산한다. 연구자가 바로 쓸 수 있다.
5. **정치학적 적용:** 정당 간 차등 오류가 실제 CMP 기반 결론을 바꾸는가 (오염 통제 포함).

**주장하면 안 되는 것:** "F1이 높아도 inference가 틀린다"(Baumann, Egami), "불변 오분류는 감쇠, 차등 오분류는 부호 반전"(Bross/Copeland), "PPI에서는 분산이 중요"(MoE-PPI, MultiPPI).

## 3b. AJPS 트랙 차별점 스캔 (2026-09-14)

주장 C1–C6은 `paper/ajps/outline.md`의 기여 목록이다. 판정은 원문 확인 2편(Benoit et al. AJPS, Benoit & Laver WEP), 요약 도구 확인 2편, 나머지는 초록 수준이다.

| 주장 | 판정 | 가장 가까운 선행연구 | 남는 차별점 |
|---|---|---|---|
| **C1** 상관·정확도 검증은 집단별 오류를 못 본다 | **부분 선점 → 기여에서 격하 (B3: Bland–Altman으로 자명)** | **Jankowski & Huber 2023 PA "When Correlation Is Not Enough"** (지도학습 포퓰리즘 점수의 정당명·언어 인공물) · Egami et al. 2023 · Kotte 2026 · Stolwijk et al. 2025 | r과 집단 간 편향의 **형식적 한계식**(P6) · 기계 코더의 측정 비동등성/DIF 틀 · 정당가족·국가 비교에 적용 |
| **C2** 정당·언어별 차등 오류 | **부분 선점** | Vallejo Vera & Driggers 2025(단서) · Lerner & Yvon 2026(극우 텍스트 perplexity) · Caliskan et al. 2025(극단 위치 축소) · Weidmann et al. 2026 PS(국가별 방향성 편향) · Le Mens & Gallego 2025(언어 차이 가능성만 언급) | CMP 인간 코드 대비 **정당가족 × 언어 × 모델** 오류를 직접 추정 · 단서와 오염의 분리 |
| **C3** 소규모 audit로 왜곡을 사전 예측 | **새로움 (인접 연구 있음)** | DSL(Egami et al.)은 사후 교정 · Mehrotra, Visokay & Gligorić 2026은 집단 표적 표집 · Messing 2026 | 추정량 설계에 오류를 회귀해 **전체 코딩 전에** 특정 비교의 왜곡을 예측 |
| **C4** 같은 사양 재분석 (인간 vs LLM) | **비교정당정치에서 새로움** | Baumann et al. 2025는 설정 변이(CMP 과제 없음) | Abou-Chadi & Krause 사양·동서 비교·녹색 대조군에 대한 LLM 재분석은 발견되지 않음 |
| **C5** 정당 단서 무작위 조작 | **대체로 선점** | **Vallejo Vera & Driggers 2025** (오스트리아 선언문 문장에 정당 단서 무작위 부여) · Benoit et al. 2026(비무작위 익명화) | **확장으로만 제시**: 다언어·다정당가족, 단서 × 정당가족 상호작용, 비교 추정량 편향으로의 연결 |
| **C6** 전문가 vs 크라우드 기준선 | **격하 (자체 점검 A1)** | Benoit et al. 2016, Horn 2019, Marquardt et al. 2025 | UK 사례의 차이는 대부분 **감쇠**(기울기 0.62)와 선언문 고유 잡음이고 정당 수준 이동은 ±0.03. 비동등성 사례가 아니다. 인간 코더 간 잡음 기준선(noise floor)으로만 사용 |

**가장 가까운 논문: Benoit, De Marchi, Laver, Laver & Ma (AJPS 2026).** 같은 학술지다. 원문 대조 결과(docs/12 §B3), 그들은 상관만 보지 않고 **분포를 비교해** 6개 차원 중 4개에서 편향 "none"(전문가를 대리 정답으로 삼아)이라고 했다. 이것을 LLM이 아닌 선언문 내용의 구조적 차이로 해석했고, r = .99는 GPT-4o 한정 **누출** 점검이었다. 우리 논문은 그들의 검증을 틀렸다고 하지 않는다. **그들이 목표로 하지 않은 집단 조건부 동등성**을, 같은 텍스트·같은 구성개념 기준에서 검정한다.

**안전한 포지셔닝 문장 (초안):**
1. Benoit et al. (2026) validate LLM party positions against expert surveys with correlations, distribution comparisons, Manifesto Project triangulation and a leakage check, all pooled over parties and countries, which their purpose did not require otherwise; we test the group-conditional equivalence that comparative claims need, and we show that pooled agreement cannot rule out between-group bias of up to X SD.
2. Vallejo Vera and Driggers (2025) show that randomly assigned party cues shift LLM labels of Austrian statements; we extend the design across languages and party families and translate cue-induced error into bias in the comparative estimands party scholars report.
3. Unlike Baumann et al. (2025), who show that configuration choices can flip conclusions, we hold the specification fixed and isolate differential error as the mechanism in canonical Manifesto-based findings.
4. Extending Jankowski and Huber's (2023) warning that correlation is not enough, we formalise the problem as measurement non-equivalence of machine coders; unlike post-hoc correction (Egami et al. 2023) or group-targeted sampling (Mehrotra et al. 2026), our audit forecasts the distortion of a given comparison before full coding.
5. Le Mens and Gallego (2025) note that cross-language accuracy differences can bias downstream analyses; we measure that bias, show when it reverses comparisons, and benchmark it against expert–crowd divergence.

**인용 금지:** Törnberg arXiv 2603.13891 (2026년 8월 철회).

## 4. 계보 지도

```
Forman 2008 / Hopkins–King 2010     classification ≠ quantification
        │
Bross 1954 / Copeland 1977          misclassification: invariant → attenuation, differential → any direction
        │
Benoit–Laver–Mikhaylov 2009/2012    political text = measurement with error
        │
Wang–McCormick–Leek 2020 · PPI 2023 · DSL 2023 · TeBlunthuis 2024     prediction ≠ valid inference
        │
Baumann 2025 · Kotte 2026 · Camuffo 2026                              high accuracy ≠ right conclusions (empirical)
        │
MoE-PPI 2026 · MultiPPI 2026 · Chen 2026                              many predictors: variance, not accuracy
        │
        ▼
THIS PROJECT   which instrument for WHICH political estimand?  (P0–P4 + EAV + CMP application)
```

## 5. 모니터링

- 매주 월요일: arXiv `stat.ME`, `cs.CL`, `stat.ML`에서 검색어 {prediction-powered + multiple, LLM annotation + selection, measurement invariance + language model, manifesto + LLM}을 확인한다. 새 위협은 이 표에 행을 추가하고 `docs/decisions/`에 영향 평가를 남긴다.
