# 11 · AJPS 정당성과 Benoit et al. (2026) 대응 전략

> ⚠️ **2026-09-14 red team 이후 일부 대체됨** (ADR-0011): §3.1 허수아비화 수정 완료. §3.2 대상 E는 DIF 검정이 아니라 구성개념 격차·결측 분석(보조)으로 격하. §2·§5의 AJPS 전망은 docs/12 §C4로 대체.


> 작성 2026-09-14. 관련 결정: ADR-0010. 원칙은 **"반박 논문"이 아니라 "비교 측정의 동등성 원칙" 논문**으로 쓴다는 것이다.

## 1. 왜 PA가 아니라 AJPS인가 — 네 가지 논거

| 논거 | 내용 | 뒷받침 |
|---|---|---|
| **① 대상이 기법이 아니라 실질 추론 관행이다** | LLM 측정은 이미 비교정치의 실질 논쟁을 판정하는 데 쓰인다. AJPS는 그 관행을 가능하게 한 논문(Benoit et al. 2026)을 실었다. 그 측정이 **어떤 비교에는 쓸 수 있고 어떤 비교에는 못 쓰는지**를 아는 것은 AJPS 독자에게 직접 필요한 지식이다 | Benoit et al. 2026 AJPS · Werner & Habersack 2025 BJPolS · Benoit & Laver 2026 WEP |
| **② 개념적 기여가 일반적이다** | 기계 코더의 측정 동등성(DIF)은 선언문을 넘어 연설·소셜미디어·V-Dem식 코딩 등 LLM으로 만든 모든 비교 지표에 적용된다. 설문 연구의 동등성 전통(King et al. 2004 APSR)을 텍스트 측정으로 옮긴다 | P6 한계식 · 인간 기준선(C6) |
| **③ 실질 논쟁의 결론이 걸려 있다** | 급진우파 전염 효과, 동·서유럽 차이, 이슈 소유는 현재 진행 중인 논쟁이다. 결론이 측정도구에 따라 갈린다면 그 자체가 실질적 발견이다 | 사례 A·B·C · Gessler & Hutter 2026 |
| **④ 누구나 쓸 수 있는 해법** | 소규모 audit으로 특정 비교의 왜곡을 사전 예측하고 교정한다 | C3 · Estimand Validation Card |

## 2. 정당성은 조건부다 — 결과별 행선지

| 결과 | 논문의 성격 | 적합 학술지 |
|---|---|---|
| 차등 오류가 있고, **실질 결론 1개 이상이 도구에 따라 바뀌며**, 사전 예측이 맞음 | 실질 + 방법 | **AJPS** |
| 차등 오류는 있지만 결론은 유지됨 | "이 조건에서는 안전하다" | PA / PSRM |
| 동등성이 성립함 | 최초의 동등성 검정 (Benoit et al.에 우호적 결과) | PA / PSRM |
| 인간 도구끼리도 크게 갈리고, LLM은 그 범위 안에 있음 | "gold 없는 비교 측정" | AJPS 가능 (C6가 중심) |

**AJPS 편집자가 물을 질문:** "Benoit et al.에 대한 코멘트인가?" → 아니다. 그들은 대상 중 하나이고, 같은 원리를 전문가 조사·크라우드·CMP에도 적용한다. 이 삼중 도구 틀이 논문을 코멘트가 아닌 일반 기여로 만든다.

## 3. Benoit et al. (2026)에 대해 어떻게 말할 것인가

**3.1 그들의 주장을 정확히 분리한다** (원문 PDF 2025-07-20판 대조, docs/12 §B3)
| 그들이 실제로 한 것 | 우리의 입장 |
|---|---|
| (a) 18개 점수 앙상블(요약 3모델 × 채점 3모델 × zero/few-shot)과 전문가 조사의 상관이 "often ranging from 0.87 to 0.92". 환경 .82, 분권 .49. MP 척도와 삼각검증, 재현 실행, open-weight 모델 재현, 연립정부 예측타당도까지 | **다투지 않는다.** 검증의 폭을 구체적으로 인정한다 |
| (b) "notwithstanding high correlations" 편향 가능성을 스스로 제기하고 **분포를 비교**했다. 분권에 강한 편향, 환경에 약간, 나머지 4개는 "none"이며, 이를 LLM이 아닌 선언문 내용의 "structural bias"로 해석 | 분포 비교도 **집단을 합친 비교**다. 집단 조건부 검정은 그들이 **목표로 하지 않았다** → 우리가 한다. "그들이 상관만 봤다"고 쓰지 않는다 |
| (c) 정당명 가림 점검: **데이터 누출 점검**, GPT-4o 요약·GPT-4o 채점·zero-shot, 실제 가림은 요약의 49%, r = .99 | 단서 효과 검정이 아니었다. 그들 자료의 가림 쌍 1,192개를 가족별로 보면 급진우파 −0.03(SE .03) 등 **대체로 그들에게 유리하다**(검토자 재계산, 탐색적). 요약 단계에도 정당명이 보였으므로 완전한 단서 검정은 아니다 |
| (d) 앙상블 평균이 가장 좋은 결과를 냈다고 보고 (편향 상쇄를 주장하지는 않음) | P7(공유 편향은 평균으로 사라지지 않음)은 **우리 자신의 명제**로만 제시한다 |

**3.2 그들의 자료로 사전등록 재분석 (대상 E, 실행 가능성 확인됨)**
- 자료: Dataverse doi:10.7910/DVN/XY1FFE. LLM 점수 52,056행(GPT-4o · Claude 3.5 · Gemini 1.5 Pro × zero/few-shot × 정당·연립 조건), 전문가 평균(CHES · Benoit-Laver · Laver-Hunt), CMP 변수(parfam 포함).
- 규모 (2026-09-14 확인, **집단별 잔차는 계산하지 않음**): 선언문 235개 · 21개국(동유럽 68개) · 1989–2019 · LLM과 전문가가 모두 있는 선언문×이슈 1,307칸 · 급진우파 27 · 녹색 24 · 이슈 6개(세금·지출, 사회, EU, 이민, 환경, 분권).
- **사전 지정 대비 (LOCK 전 추가·삭제 가능, LOCK 후 고정):**
  1. 이민: 민족주의/급진우파(70) − 보수(60)
  2. 이민: 급진우파(70) − 기독교민주(50)
  3. 환경: 생태(10) − 사회민주(30)
  4. 세금·지출: 사회주의/좌파(20) − 보수(60)
  5. 사회: 급진우파(70) − 자유(40)
  6. EU: 동유럽 − 서유럽
  7. 6개 이슈 전체: 동유럽 − 서유럽 (다중비교 FDR)
- **통계량:** 대비마다 Δ_LLM − Δ_전문가 (전문가 점수의 선언문 간 SD 단위, 선언문 bootstrap CI). 같은 이슈의 전체 r을 나란히 보고한다(P6의 실증). 앙상블과 개별 모델을 각각 계산한다(P7). CMP 기반 척도(Lowe et al. 2011 logit)를 **세 번째 도구**로 넣는다.
- **해석 규칙:** 전문가 조사도 하나의 도구다. 세 도구의 비교 결론이 **일치하는지**를 1차로 보고한다. "누가 맞나"는 주장하지 않는다.
- **결과가 동등성을 지지하면 그대로 보고한다.** 그들의 결론을 처음으로 제대로 검정한 결과가 되고, 우리 틀의 유용성도 함께 보여준다.

**3.3 어조 규칙**
- 쓰지 않는 표현: "their results are biased", "they overlooked", "flawed validation".
- 쓰는 표현: "pooled agreement statistics answer a different question from the one comparative claims require"; "we test the equivalence that their design did not target."
- 투고 전에 저자들에게 사전 공유하고 의견을 요청한다. 반영한 내용은 감사의 글에 적는다.

**3.4 예상 반론과 답**
| 반론 (저자들이 심사자로 올 경우) | 답 |
|---|---|
| 전문가 조사는 평판을, LLM은 텍스트를 잰다 (구성개념 차이) | **맞다(검토자 판정).** LLM–전문가 차이를 LLM 오류로 부르지 않는다. DIF는 같은 텍스트·같은 구성개념 기준(PImPo, UK 전문가 문장 코딩)에서만 검정하고, 전문가 조사와의 차이는 부호를 사전에 정한 구성개념 격차로 따로 분석한다 |
| 요약 후 채점(문서 수준)은 문장 분류와 다른 파이프라인이다 | **맞다.** 두 파이프라인을 별도 측정도구로 취급한다. 프런티어 모델 재실행은 **그들의 프롬프트와 절차를 그대로** 쓴다 |
| 이민 점수 결측이 집단마다 다르다 | **확인됨:** 이민 결측률은 급진우파 18%, 보수 41%, 사민 36% (전체 실행 기준). 완전사례 비교는 편향된다 → 결측 자체를 결과로 모형화하고 경계(bounds)를 보고한다 |
| 모델이 이미 구식이다 (GPT-4o 등) | 그래서 현재 프런티어 모델(여러 공급사)로 같은 대비를 반복한다 → "구식 모델의 문제인가, 구조적 문제인가"를 판정한다 |
| 다중비교로 인한 우연 | 대비를 사전 지정하고 FDR로 보정한다. 대조군(생태−사민 환경)은 뒤집히지 않아야 한다 |

## 4. 논문 서론 포지셔닝 문단 (초안, 영어)

> Benoit et al. (2026) show that ensembles of large language models recover party positions that agree closely with expert surveys across 21 languages. They also compare score distributions against expert benchmarks, cautioning that bias can persist "notwithstanding high correlations," and they check for data leakage by masking party names. These checks pool over parties and countries, as their purpose did not require otherwise. Comparative research, however, compares radical-right with mainstream parties, East with West, and parties before and after electoral shocks, and such comparisons require that each group be measured with the same error, a property neither pooled correlations nor pooled distributions were designed to establish. We test it where the text and the construct are held fixed, using sentence-level human codings of the same manifesto units, and we treat divergence from expert surveys, which rate parties rather than texts, as a separate question about constructs.

## 5. 이 트랙의 판정

**현재 판단: AJPS 정당성은 "조건부로 충분"하다.** 논거 ①–④는 자료 없이도 성립한다. 그러나 AJPS 수준을 결정하는 것은 §2 표의 첫 줄, 즉 **사전 지정한 실질 결론이 도구에 따라 바뀌는가**다. 대상 E(그들 자료 재분석)는 LLM 호출 없이 며칠 안에 이 질문에 대한 첫 증거를 준다. 따라서 **킬테스트 ①과 함께 가장 먼저 사전등록하고 실행할 대상**이다.
