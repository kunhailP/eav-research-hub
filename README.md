# EAV Research Hub

**Estimand-Aware Validation of LLM political text classifiers.**

> *Researchers validate language models on labels, but publish claims about estimands.*
> 목표: 정치학자가 LLM을 **자신이 추정하려는 정치적 양(estimand)** 기준으로 고르게 하는 이론·방법·증거를 만들고, Political Analysis에 투고한다.

## 현재 상태 (2026-09-14)

| 항목 | 상태 |
|---|---|
| 판정 | **조건부 GO** — 파일럿 게이트 G1 (2026-11-22)에서 확정 ([평가](docs/00_assessment.md)) |
| 이론 | v0.2: P0–P4, 수치 예시 전부 테스트 통과 ([theory](docs/02_theory.md)) |
| 시뮬레이션 | S1, S2, S3 완료 · S4–S7 예정 |
| 파일럿 | 사전등록 초안 완료 · 분석 파이프라인 합성 자료로 검증 · **라벨 생성 전** |
| **목표 (ADR-0006)** | **AJPS**: *Can We Compare Parties Measured by Language Models?* · PA 방법론 논문은 분리 ([설계](docs/09_ajps_substantive_design.md), [원고 뼈대](paper/ajps/outline.md)) |
| 판정 자료 | Benoit 2016 UK 문장당 전문가 4–8명 코딩 확보 · Baumann CMP 표본은 영어 번역본 |
| 도구 | `label/label.py` (mock · vLLM · Claude sync/batch, 캐시·재개·정당 단서 조건) · `killtest/differential_error.py` |
| 로컬 추론 | RTX 3090 + vLLM 0.29 (`/root/venvs/vllm`) · Qwen2.5-7B 스모크 테스트 통과 |
| 예비 발견 | 전문가 vs 크라우드: 선언문 상관 r=.96인데 New Labour 이동·1997 LD–Lab 결론이 다름 ([09 §3d](docs/09_ajps_substantive_design.md)) |
| AJPS 정당성 | 조건부 충분 · Benoit et al. 2026 대응 전략 ([11](docs/11_ajps_case_and_engagement.md)) |
| **Red team (2026-09-14)** | 자체 점검 + 독립 검토자 3인 → 치명 결함 다수 발견, **v3 설계 전환** ([12](docs/12_red_team.md), ADR-0011). 기존 킬테스트 ① 사전등록 폐기 |
| v3 파이프라인 (2026-09-14) | `src/eav/latent.py`(다평정자 잠재계층) · `src/eav/v3.py`(모형 기반 DIF, 군집 bootstrap, 모수적 귀무 max-T, TOST 3분류) · `v3/run_uk.py`(pilot → lock → full) · `v3/lock.py` · 테스트 25개 통과. 합성 UK형 세계: 귀무 거짓 비동등 0/20, 검정력 10/20 |
| 자료 | UK Benoit 2016 코더 수준 행렬 완료 · **PImPo 확보(242개 선언문, 개별 코더 응답 없음, 텍스트는 API 키 필요)** ([데이터 카드](docs/data_cards/pimpo.md)) |
| 오픈소스 모델 | vLLM 4개 모델, v3 11개 선택지 제약 디코딩 스모크 통과 |
| 보정 (인간 코더만) | 모의 LLM 표적: 귀무 거짓 비동등 0/25 · 동등 판정 24/25 · 검정력 25/25. 인간 leave-one-out: 크라우드·전문가 2명이 3–5pp 비동등 → 인간 범위 비교 기준 추가 (docs/12 D17) |
| 형식 파일럿 | 선택지 순서 무작위화로 위치 편향 제거. 적격: Qwen2.5-7B · Mistral-7B · Granite-3.3-8B · Phi-3.5-mini / 탈락: OLMo-2 (D15–D16) |
| **사전등록 LOCK** | `prereg-v3-uk` · 2026-09-14T05:50Z · 파일 15개 해시 ([LOCKS](docs/decisions/LOCKS.md)). 두 영역 모두 모의 귀무 거짓 비동등 0/25 확인 후 |
| **확증 실행** | 2026-09-14 05:51Z 시작 · `bash v3/run_full_uk.sh` · 로그 `results/v3_uk_full/run_full_uk.log` · 라벨링 약 3시간 + 판정 약 1시간 |
| **PImPo (2026-09-14)** | Manifesto API로 텍스트 232,397문장 확보(매칭 100%) · 공식 MPDS2024a 정당가족(급진우파 15,152문장) · 크라우드 표 수 복원(D20) · 프롬프트 3종 · 가중 설계 모듈 `src/eav/v3_weighted.py` · 실행기 `v3/run_pimpo.py` · 사전등록 초안 [14](docs/14_prereg_pimpo_draft.md) · 보정(편향 보정판): 귀무 거짓 비동등 0/15 · 검정력 11/15 · SESOI 1pp |
| **PImPo LOCK · 확증 실행** | `prereg-v3-pimpo` LOCK(16개 파일) · 확증 모델 Qwen · OLMo · Granite · Phi (Mistral 탈락, D27) · 25,419문장 × 4 × 3 라벨링 중 · 로그 `results/v3_pimpo_full/run_full_pimpo.log` |
| 다음 한 가지 | UK 판정 결과 해석(`python v3/report_uk.py`) · PImPo 판정 후 해석 · API 모델 추가(키 필요) |

## 지금까지 알게 된 것 (시뮬레이션, 8개 후보 모델)

- **불변 오류이면** balanced accuracy가 정당 격차 편향 최적 모델을 100% 고른다. F1은 65–73%, accuracy는 54–63%다. → 비교 상대는 F1이 아니라 BA다.
- **차등 오류이면** 어떤 라벨 지표든 25–30%로 떨어지고(무작위 12.5%), 편향 최적 모델 대비 regret은 2–4pp다(참 격차 15pp).
- **DSL/PPI로 교정하면** 소수정당(10%) 격차 연구에서 accuracy 기준 선택은 유효표본을 최대 28% 잃는다 (P4, 1/π 가중).
- **EAV-structural / EAV-efficiency**는 audit n≈200–400 이상, 차등 오류가 있을 때 라벨 지표를 이긴다. n≤100에서는 이기지 못한다.
- 편향 불편추정(B̂²−V̂)은 선택기로 **실패**한다 (ADR-0004).

## 데이터 접근 (저장소에 포함하지 않음)

재배포가 금지되거나 제한된 자료는 이 저장소에 없다. `data/` 전체와 실행 로그는 `.gitignore`로 제외한다.

| 자료 | 받는 곳 | 준비 스크립트 |
|---|---|---|
| Benoit et al. (2016) UK 코더 자료 | github.com/kbenoit/CSTA-APSR | `data_prep/benoit2016.py`, `data_prep/benoit2016_multirater.py` |
| PImPo (Lehmann & Zobel 2018) | manifesto-project.wzb.eu/datasets/pimpo | `data_prep/pimpo.py`, `data_prep/pimpo_multirater.py` |
| Manifesto Corpus 텍스트 · MPDS2024a | Manifesto Project API (개인 API 키 필요) | `data_prep/manifesto_texts.py`, `data_prep/pimpo_sentences.py` |
| Benoit et al. (2026) 재현 자료 | Harvard Dataverse doi:10.7910/DVN/XY1FFE | — |

API 키는 저장소 밖(`~/.config/manifesto/api_key`)에 두고, 스크립트에는 환경변수로만 넘긴다. 사전등록 잠금 해시(`docs/decisions/LOCKS.md`)에는 제외된 자료 파일의 해시도 포함되므로, 자료를 같은 방법으로 다시 만들면 잠금을 검증할 수 있다.

## 지도

| 문서 | 내용 |
|---|---|
| [00_assessment](docs/00_assessment.md) | 적합성 · 발전가능성 · 게재가능성 평가, 원 메모 수정 목록 |
| [01_positioning](docs/01_positioning.md) | 문헌 지도, 인용 검증, 2026 선점 위협 13건, 주장 가능한 신규성 |
| [02_theory](docs/02_theory.md) | 두 regime, error regression, P0–P4 |
| [03_eav_method](docs/03_eav_method.md) | EAV 절차와 기준 정의, 예정 시뮬레이션 |
| [04_pilot_prereg](docs/04_pilot_prereg.md) | 파일럿 사전등록: 적격 기준, 통계량, GO/PIVOT/KILL |
| [05_data_plan](docs/05_data_plan.md) | Baumann, Manifesto (약관, 오염, gold 잡음) |
| [06_publication_strategy](docs/06_publication_strategy.md) | 초록 초안, 투고 경로, 심사자 공격 R1–R11 |
| [07_roadmap](docs/07_roadmap.md) | 마일스톤과 게이트 |
| [08_risk_register](docs/08_risk_register.md) | 위험 K1–K10 |
| [09_ajps_substantive_design](docs/09_ajps_substantive_design.md) | AJPS 경로: 정치학 질문, 논문 구조, 재현 대상, 2주 kill test |
| [decisions/ADR-log](docs/decisions/ADR-log.md) | 의사결정 기록 |
| [templates/](templates/) | Estimand Validation Card, 실험 카드, 주간 메모 |
| [lit/references.bib](lit/references.bib) | 검증된 참고문헌 (CHECK 표시 확인 필요) |

## 코드

```
src/eav/misclass.py   집단별 오분류의 모집단 대수 (P1–P4)
src/eav/audit.py      error regression + 선택 기준 (accuracy, F1, BA, EAV-*)
src/eav/simulate.py   후보 모델·audit 표본 생성 DGP
sims/s1_…, s2_…, s3_… 시뮬레이션 (results/ 에 CSV·그림)
pilot/analyze.py      task 무관 파일럿 분석: model×estimand 행렬, regret, budget 곡선
tests/                모든 명제의 수치 예시 + 파이프라인 end-to-end
```

```bash
pip install numpy pandas matplotlib pytest
make test         # 명제 검증 + 파이프라인 테스트
make sims         # S1–S3 재생성 (~40초)
make pilot-demo   # 합성 UK-manifesto 자료로 파일럿 분석 시연
```

실제 파일럿 입력 형식: `gold.csv (doc_id, y, 메타데이터)`, `preds.csv (doc_id, model, yhat)`, `spec.json (estimand 목록)`. 형식은 `pilot/analyze.py` 상단 docstring 참조.

## 운영 원칙

1. **죽이기 먼저.** 모든 단계에 사전 정의된 KILL/PIVOT 조건이 있다.
2. **사전등록은 결과보다 먼저 고정한다.** 변경은 ADR로만 한다.
3. **주장은 01_positioning §3의 범위 안에서만 한다.** Bross, Baumann, PPI 문헌이 이미 말한 것을 신규로 주장하지 않는다.
4. **재현성:** seed 고정, 모델 snapshot 기록, 원응답 캐시, `make`로 재생성.
5. **매주 월요일 arXiv 모니터링.** 이 분야는 2026년에 월 단위로 움직인다.
