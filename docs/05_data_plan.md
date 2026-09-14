# 05 · Data plan

## A. Baumann et al. benchmark (파일럿)

- 출처: `github.com/joebaumann/llmhacking` (최종 갱신 2026-07-06 확인).
- 공개 범위: 전처리 CSV(id, ground truth, text, metadata)와 생성·평가 코드. **LLM 예측 13M건은 미공개** → 재생성하거나 저자에게 요청한다 (요청 메일은 병행하되 기다리지 않음).
- 등록 필요 task 7개(에세이 6, BES issue_survey)는 파일럿에서 제외한다.
- 가설의 집단 분할: 키워드 분할과 메타데이터 분할(48.1%). repo의 `get_groups()`를 재사용한다.

## B. Manifesto Project (메인 적용)

- 버전: **Dataset MPDS2026a, Corpus 2026-1 (2026-09-04 공개)**. 규모 수치(정당 1,429 / 선거 884 / 문서 5,330)는 자동 요약으로 확인한 값이므로 인용 전 페이지에서 수기로 재확인한다.
- 접근: API 키 필요 (프로필 페이지에서 발급). R `manifestoR` 또는 REST API.
- **이용약관: 서면 허가 없는 재배포 금지.** → 재현 패키지에는 텍스트를 넣지 않는다. (manifesto_id, 문장 인덱스, 코드, 스크립트)만 넣고 사용자가 자기 키로 내려받게 한다. 파생 LLM 라벨의 공개 가능 여부는 Manifesto Project에 서면으로 문의한다 (M2 이전).
- **Gold의 잡음:** quasi-sentence 코딩 자체에 체계적 오분류가 있다 (Mikhaylov, Laver & Benoit 2012, PA). 따라서 estimand를 "훈련된 CMP 코더가 산출하는 값"으로 정의한다. 코더 신뢰도 자료나 재코딩 표본이 있으면 민감도 분석에 쓴다.

### 오염(contamination) 대응 — 설계의 핵심

Halterman & Keith (2026, PA)는 Manifesto 코퍼스를 LLM 학습 오염 위험이 가장 큰 자료로 지목했다. 선언문과 codebook은 공개되어 있지만, 문장 단위 코드는 등록 API 뒤에 있다.

1. **시간 holdout:** 각 모델의 학습 cutoff 이후 공개된 선언문(2026a/2026-1 추가분 포함)만으로 만든 부분표본에서 모든 결과를 재현한다.
2. **Memorization probe:** codebook 없이 문장만 주고 CMP 코드를 맞히는 비율을 우연 수준과 비교한다.
3. 오염이 모델마다 다르면 그 자체가 차등 오류의 원천이다. 따라서 오염 지표를 공변량으로 보고한다.

### 후보 estimand (정치학적으로 의미 있는 것만)

| 이름 | 정의 | 정치학적 질문 |
|---|---|---|
| 이민 주목도 격차 | per601_2/per602_2 계열 비중의 급진우파 − 사민주의 정당 차이 | 급진우파의 이슈 소유 |
| 환경 주목도 격차의 추세 | per501 비중의 녹색 − 주류 정당 격차 × 시간 | 이슈 확산(contagion) |
| 정당가족×시기 상호작용 | 2008 이전·이후 경제 좌우 주목도 | 금융위기 이후 재편 |

범주 코드와 정당가족 정의는 **분석 전에 코드북 버전과 함께 고정**한다.

## C. 저장·재현 규칙

- `data/raw/` (원응답 캐시, gitignore), `data/derived/` (라벨 CSV, 공개 가능한 것만).
- 모든 LLM 호출은 (model snapshot, prompt hash, temperature, 날짜)를 함께 기록한다.
- 난수 seed는 스크립트 상수로 둔다. 결과 CSV는 `make sims`로 재생성 가능해야 한다.
