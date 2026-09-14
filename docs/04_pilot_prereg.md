# 04 · Pilot v0 사전등록 (Baumann benchmark)

> 잠금 규칙: 이 문서는 **라벨 생성 전에** 태그(`prereg-pilot-v0`)로 고정한다. 이후 변경은 ADR로만 하고, 원문은 남긴다.
> 목적: 논문을 **증명하는 것이 아니라 가능한 한 빨리 죽이는 것.**

## Step 0 — 자료 적격성 확인 (라벨 생성 전, 반나절)

`joebaumann/llmhacking` 저장소에는 **13M 예측값이 공개되어 있지 않다** (확인됨: 코드, ground truth, text, metadata만 있음). 따라서 라벨은 재생성해야 한다. 등록이 필요한 7개 task(에세이 6개 + BES)는 제외한다.

후보 정치 task와 확인할 집단 변수:

| Task (repo) | 원자료 | 확인할 집단 변수 | 우선순위 |
|---|---|---|---|
| manifestos (economic / social ideology, issue) | Benoit et al. 크라우드 코딩 UK 선언문 문장 1987–2010 | **정당**, **연도** | 1 |
| tone | Carlson & Montgomery, 미 상원 정치광고 (859건) | 정당, 연도 | 2 |
| topic | Egami et al., 의회 법안 economy vs not (10k) | 발의자 정당 | 2 |
| ideology_news | Baly et al., AllSides 기사 | 매체 성향 | 3 |
| Gilardi replication (tweets/news) | Gilardi et al. 2023 | 확인 필요 | 3 |

**적격 기준 (사전 고정):** (a) 정치적으로 의미 있는 집단 변수가 있고 두 수준 각각 gold 문서 ≥ 150, (b) 이진화한 라벨의 양성 비율이 5–95%, (c) 이진화 규칙을 task 단위로 사전에 적는다 (예: economic ideology → "right" vs 그 외).
적격 task가 **3개 미만이면** → Step 0에서 PIVOT: Manifesto Project API로 직접 파일럿을 한다 (05_data_plan.md).

## 설계

- **모델 (6–8개, 버전 고정):** open-weight 4–5개가 주 분석이다 (재현성). 예: Llama, Qwen, Mistral, Gemma 계열의 소형과 중형. API 2–3개가 보조 분석이다 (예: `claude-sonnet-5`, `claude-haiku-4-5-20251001`, 타사 1개). 실행일에 정확한 snapshot ID를 `pilot/models.lock.json`에 기록한다.
- **프롬프트:** task당 Baumann repo의 기본 프롬프트 **하나로 고정**, temperature 0. (prompt 변이는 Baumann의 주제이므로 이 파일럿에서는 제거한다.)
- **무효 응답:** 음성 클래스로 코딩하고 모델별 무효율을 보고한다 (`invalid_policy: negative`). 민감도 분석은 `drop`.
- **Estimand (task당 최대 4개):** prevalence, 정당 격차(최대 두 정당), 연도 추세, 정당×연도 상호작용.
- **분석 코드:** `pilot/analyze.py` (합성 자료로 검증 완료: `python pilot/analyze.py --demo --out results/pilot_demo`).

## 통계량

**1차 (confirmatory)**
- **F1 선택의 상대 regret**: $\mathrm{RR}=\dfrac{|B_{m_{F1}}|-\min_m|B_m|}{|\theta_H|}$. 문서 bootstrap으로 5–95% 구간을 낸다.
- **Held-out 곡선**: audit n ∈ {100, 200, 400, 800}에서 선택하고 나머지 gold 문서로 평가한다. Plug-in regime은 `mean_excess_rel_error`로 `f1`·`balanced_accuracy`·`eav_structural`을 비교한다. Correction regime은 `mean_ess_loss`로 `accuracy`·`balanced_accuracy`·`eav_efficiency`를 비교한다.
- **작은 estimand 규칙**: $|\theta_H| < 2\,\mathrm{SE}(\theta_H)$이면 상대 regret이 불안정하다. 이 경우 절대 regret(pp)만 보고하고 판정 분모에서 뺀다 (`relative_regret_defined`). 합성 demo에서 격차 2.8pp인 LD–Lab 대비의 상대오차가 폭주해 이 규칙을 추가했다.

**2차 (exploratory)**
- 부호 반전 (F1 선택 모델), P(argmin 일치)와 **oracle의 bootstrap 안정성**, Kendall τ, 정당별 FPR/FNR 차이, balanced accuracy 선택의 regret.

> P(m_F = m_θ)를 1차 통계량으로 쓰지 않는다: S1/S3에서 argmin 일치는 미세한 차이에도 뒤집히는 불안정한 통계량이다 (ADR-0002).

## 판정 규칙 (사전 고정)

Task–estimand 쌍(대비·추세·상호작용만; prevalence 제외)을 단위로 한다.

| 판정 | 조건 |
|---|---|
| **GO (full paper)** | (i) 쌍의 ≥ 1/3에서 RR ≥ 0.25이고 bootstrap 5% 하한 > 0.10, **그리고** (ii) 적어도 한 정치 task에서 n=400일 때 `eav_structural` 또는 `eav_efficiency`가 F1과 BA보다 held-out 초과오차가 20% 이상 작다 |
| **PIVOT-A (지표 논문)** | (i)은 성립하지만 BA가 EAV와 대등하다 → "F1 대신 Youden 기준" 짧은 논문 (PSRM/CMM). Manifesto 단계로 가지 않는다 |
| **PIVOT-B (효율 논문)** | plug-in regret은 작지만 `eav_efficiency`의 이득이 크다 → P4 중심의 correction-regime 논문 |
| **KILL** | 쌍의 < 15%에서만 RR ≥ 0.25, 부호 반전 없음, 두 regime 모두 EAV 이득 < 10% |

## 비용과 산출물

- 호출 수 ≈ Σ_task(문서 수) × 모델 수. 실행 전 토큰 수로 비용을 추산해 `pilot/cost_estimate.md`에 기록한다. 모든 원응답은 `data/raw/`에 캐시한다 (gitignore).
- 산출물: `results/pilot_v0/<task>/{matrix,regret,budget}.csv`, `fig_matrix.png`, 1쪽 판정 메모 (`docs/decisions/`).
