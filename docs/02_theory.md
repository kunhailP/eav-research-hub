# 02 · Theory memo v0.2

> 상태: 명제 P0–P4는 증명 스케치 완료, 모든 수치 예시는 `tests/test_theory.py`로 검증됨.
> v0.1(원 메모) 대비 변경점은 맨 끝 "v0.1 → v0.2 수정 사항" 참조.

## 0. 두 개의 regime

LLM 라벨이 연구에 들어가는 방식은 두 가지이며, **"좋은 모델"의 정의가 regime마다 다르다.**

| Regime | 최종 추정 | 모델의 오류가 들어가는 곳 | 올바른 선택 기준 |
|---|---|---|---|
| **Plug-in** | 코퍼스 전체에 $\hat Y_m$을 대입해 $\hat\theta_m$ 계산 (현재 관행 대부분) | **편향** | estimand 편향의 크기 |
| **Correction** | DSL / PPI처럼 human audit으로 교정 | 편향은 제거됨 → **분산** | estimand 추정량의 분산 (필요 human label 수) |

원 메모는 plug-in regime만 다뤘고, EAV로 모델을 고른 뒤 DSL을 쓰자고 제안했다. 그런데 DSL을 쓰면 편향은 어차피 교정되므로, **correction regime에서는 "편향이 작은 모델"을 고를 이유가 사라진다.** 이 긴장을 해소하는 것이 v0.2의 핵심이다.

## 1. 통일 틀: error regression (P0)

$\theta = w^\top\beta$, $\beta = \Sigma^{-1}E[XY]$, $\Sigma = E[XX^\top]$ (OLS estimand: 집단 비율, 정당 격차, 추세, 정당×시간 상호작용이 모두 특수한 경우).
모델 $m$의 오류를 $e_m = \hat Y_m - Y$라 하자.

**P0 (Projection).**
(i) Plug-in 편향: $\;\tilde\theta_m - \theta = w^\top \Sigma^{-1} E[X e_m]$.
(ii) Correction regime ($N\to\infty$, 단순무작위 audit $n$): $\;n\,\mathrm{Var}(\hat\theta^{PPI}_m) \to w^\top\Sigma^{-1}E\!\left[XX^\top (e_m - X^\top c_m)^2\right]\Sigma^{-1}w$, 여기서 $c_m=\Sigma^{-1}E[Xe_m]$.
(iii) Prediction loss(0–1): $E[e_m^2]$.

*증명 스케치.* OLS의 선형성: $\Sigma^{-1}E[X\hat Y_m] = \beta + \Sigma^{-1}E[Xe_m]$. PPI 추정량은 $\tilde\beta_{m,N} - \hat c_{m,n}$이고, $N\to\infty$에서 분산은 audit에서 $e_m$을 $X$에 회귀한 계수의 sandwich 분산이다. ∎

**함의.** 정확도는 오류의 **노름**이고, plug-in estimand loss는 오류의 **설계행렬 위 사영(선형 범함수)**이며, correction loss는 그 사영의 **잔차 분산**이다. 세 개의 서로 다른 범함수가 같은 순위를 줄 이유가 일반적으로 없다. 운영상으로는 **"모델의 오류를 estimand의 설계행렬에 회귀하라"** 한 줄로 요약된다. 계수는 plug-in 기준, 표준오차는 correction 기준이 된다 (`src/eav/audit.py::error_regression`).

> DSL(Horvitz–Thompson 형 pseudo-outcome)은 $\mathrm{Var}(Y-f)$ 대신 $E[(Y-f)^2]$을 쓴다. P4의 $1/\pi_g$ 가중 결론은 바뀌지 않는다.

## 2. 이진 라벨, 두 집단

$p_g=P(Y{=}1|G{=}g)$, $\pi_g=P(G{=}g)$, $\alpha_{mg}$=FPR, $\beta_{mg}$=FNR, $\Delta=p_1-p_0$.

$$\tilde p_{mg}=(1-\beta_{mg})p_g+\alpha_{mg}(1-p_g),\qquad B_m=[\alpha_{m1}(1-p_1)-\beta_{m1}p_1]-[\alpha_{m0}(1-p_0)-\beta_{m0}p_0].$$

### P1 — 측정 불변이면 기준은 Youden's J (알려진 결과, 도구로 사용)

$\alpha_{mg}\equiv\alpha_m$, $\beta_{mg}\equiv\beta_m$이면 $\tilde\Delta_m=(1-\alpha_m-\beta_m)\Delta$. 따라서 $|B_m|=(\alpha_m+\beta_m)|\Delta|$이고, 최적 모델은 **Youden's J = balanced accuracy의 선형변환**을 최대화하는 모델이다. 정확도($\bar p$로 가중)와 F1은 이것과 어긋날 수 있다.

*예시 (검증됨).* $\pi=(.75,.25)$, $p=(.05,.25)$ ⇒ $\bar p=.10$, $\Delta=.20$.
A: $(\alpha,\beta)=(.01,.40)$ → 오분류율 .049, $|B|=.082$. B: $(.05,.10)$ → 오분류율 .055, $|B|=.030$.
더 정확한 A가 정당 격차는 2.7배 더 틀린다. **다만 balanced accuracy는 B를 고른다.**

> ⚠️ 이 결과는 Bross (1954), Copeland et al. (1977)의 비차등 오분류 감쇠와 동일하다. **새로운 기여로 주장하면 안 된다.** 기여로 쓸 수 있는 것은 "정치 텍스트 분야의 기본 지표(F1)가 불변 조건에서도 틀린 지표"라는 실무적 지적뿐이다.
> S1 결과: δ=0에서 balanced accuracy는 편향 최적 모델을 **100%** 고르지만, accuracy는 54–63%, F1은 65–73%에 그친다.

### P2 — 차등 오류는 부호를 뒤집는다

$\Delta>0$일 때 $B_m<-\Delta$이면 $\tilde\Delta_m<0$.
*예시 (검증됨).* $p=(.45,.55)$, $\pi=(.5,.5)$. 집단1: $\beta=.10,\alpha=0$, 집단0: $\alpha=.10,\beta=0$ ⇒ 정확도 94.5%, $\tilde\Delta=-.01$ (참값 +.10).

S1 결과: 차등 오류 δ≥0.5에서 accuracy/F1/BA 중 무엇으로 골라도 편향 최적 모델과의 일치율은 25–30%(무작위 선택 12.5%)이다. 선택된 모델이 정당 격차를 50% 이상 왜곡할 확률은 6–29%다.

### P3 — Estimand switching은 차등 측정오류의 진단 신호다 (신규)

**명제.** 오류율이 estimand의 설계변수 $X$와 독립이면(측정 불변), $E[\hat Y_m|X]=\alpha_m+J_mE[Y|X]$이므로 절편을 제외한 모든 OLS 계수가 $J_m$배로 축소된다. 따라서 **모든 대비·추세·상호작용 estimand에 대해 편향 최적 모델이 동일하다.**
**대우.** 서로 다른 대비 estimand에서 최적 모델이 바뀌면, 적어도 하나의 집단 변수에 대해 측정 불변이 깨져 있다.
**예외.** 수준 estimand(전체 prevalence)는 불변 조건에서도 $\alpha(1-\bar p)-\beta\bar p$의 상쇄 때문에 최적 모델이 달라진다 (Forman 2008의 quantification 문제).

S3 결과(3개 정당, 8개 모델):

| δ | 세 대비에서 최적 모델 동일 | prevalence-최적 = R–L-최적 | 4개 estimand의 서로 다른 최적 모델 수 |
|---|---|---|---|
| 0 | **1.000** | 0.159 | 1.84 |
| 0.1 | 0.160 | 0.145 | 2.80 |
| 0.5 | 0.076 | 0.158 | 3.10 |

> ⚠️ **설계상 경고.** δ=0.1이라는 아주 작은 차등만 있어도 argmin은 84% 확률로 바뀐다. 후보 모델들이 비슷한 품질이면 argmin 일치율은 **잡음에 극도로 민감한 통계량**이다. "GPT는 prevalence, Claude는 격차에 최적" 같은 결과만으로는 약하다. 반드시 **regret의 크기**와 **bootstrap 안정성**을 함께 보고해야 한다 (ADR-0002).

### P4 — Correction regime: 소수정당의 오류가 1/π로 증폭된다 (신규)

PPI 정당 격차 추정량의 분산:

$$n\,\mathrm{Var}(\hat\Delta^{PPI}_m) = \sum_{g}\frac{\mathrm{err}_{mg}-b_{mg}^2}{\pi_g},\qquad b_{mg}=\tilde p_{mg}-p_g .$$

반면 오분류율은 $\sum_g \pi_g\,\mathrm{err}_{mg}$이다. **정확도는 소수 집단의 오류를 $\pi_g$로 할인하고, 격차 추정의 효율은 $1/\pi_g$로 할증한다.** 상대 가중치 차이는 $1/\pi_g^2$이다 (π=0.1이면 100배).

*예시 (검증됨).* $\pi=(.9,.1)$, $p=(.2,.2)$, 집단 내 FPR=FNR.
A: 집단 오류 3% / 20% → 정확도 95.3%, 분산 1.889. B: 6% / 8% → 정확도 93.8%, 분산 0.842.
**덜 정확한 B는 같은 정밀도에 A의 45% human label만 필요하다.**

S1 결과: π₁=0.5이면 accuracy가 분산 최적 모델을 96–98% 고른다(가중치가 대칭이므로). π₁=0.1, δ=1이면 36%까지 떨어지고 **유효표본의 28%를 잃는다.**

> 선행연구 위치: "PPI에서는 정확도가 아니라 분산이 중요하다"는 Gu et al. (2026, MoE-PPI), Cowen-Breen et al. (2026, MultiPPI), Chen et al. (2026)에 이미 있다. **정치 집단 대비에서 $1/\pi_g$ 가중이 생기고, 그래서 소수정당 연구에서 정확도 기반 선택이 체계적으로 실패한다**는 결과는 확인된 선행연구가 없다 (01_positioning.md).

**따름정리 (audit 설계).** $\sum_g n_g=n$ 제약에서 $\sum_g v_g/n_g$를 최소화하면 $n_g\propto\sqrt{v_g}$ (Neyman 배분)이다. 소수정당을 과대표집하는 층화 audit이 최적이다. Stratified PPI (Fisch et al. 2024), Ye et al. (2026)과의 관계를 명시해야 한다.

### P6 — 상관 기반 검증은 집단 비교의 왜곡을 보지 못한다 (신규, AJPS 논문의 핵심)

문서(선언문) 수준의 인간 코딩 점수 $x$, LLM 점수 $m = x + cG + e$ ($G$: 정당가족·언어 이진 지표, 비중 $\pi$, $e$: 독립 잡음)라 하자. LLM으로 잰 집단 격차의 편향은 정확히 $c$다. 그런데

$$\mathrm{corr}(x,m)\;\approx\;1-\frac{c^2\,\pi(1-\pi)(1-\rho^2_{xG})+\sigma_e^2}{2\sigma_x^2}.$$

**따름정리.** 검증 상관이 $r^*$ 이상이라는 사실은 $|c|\le\sigma_x\sqrt{2(1-r^*)/[\pi(1-\pi)(1-\rho^2_{xG})]}$ 크기의 격차 편향과 양립한다. 잡음이 없고 두 집단 크기가 같을 때 $r^*=.95$면 **0.63 SD**, $r^*=.90$이면 **0.89 SD**까지 탐지되지 않는다.

*수치 검증 (`tests/test_theory.py`).* $\pi=.5$, 참 격차 +0.20 SD, $c=-0.25$, $\sigma_e=.1$ → $r=0.987$ (근사식 0.987), LLM 격차 −0.05 SD.

**함의.** 선언문 전체를 섞은(pooled) 상관은 대부분 **집단 내부 분산**으로 결정된다. 집단 격차는 그 분산에 비해 작다. 따라서 높은 상관은 비교 연구에 필요한 측정 동등성(measurement equivalence)의 증거가 아니다. 설문조사의 차등문항기능(DIF; King et al. 2004 APSR)과 같은 구조의 문제를 기계 측정에 옮긴 것이다.
**확장 과제.** 집단별 척도 차이($m=a_g+b_gx+e$)가 있으면 격차 편향은 $(a_1-a_0)+(b_1-1)\mu_1-(b_0-1)\mu_0$가 된다. 이 경우의 상관 한계식을 유도한다.

### P7 — 앙상블은 공유된 차등 오류를 제거하지 못한다

모델 $m=1..M$의 집단 대비 편향을 $b_m = \bar b + u_m$으로 분해하자. $\bar b$는 모델들이 공유하는 편향, $u_m$은 모델 고유 편향이다($\sum u_m=0$). 모델별 추정의 잡음을 $\varepsilon_m$이라 하면, 평균 점수로 만든 앙상블 대비의 오차는 $\bar b + \frac1M\sum\varepsilon_m$이다. **평균화는 분산($\varepsilon$)과 모델 고유 편향($u$)은 줄이지만 공유 편향 $\bar b$는 그대로 둔다.** 공급사가 달라도 학습 자료, 정당에 대한 공통 평판, 번역 품질을 공유하면 $\bar b\neq0$일 수 있다.
**검정:** 앙상블 편향과 개별 모델 편향의 평균이 같다는 항등식을 이용해 $\bar b$를 추정한다. 공급사 간 편향의 상관이 높을수록 앙상블의 이득은 분산 감소에 그친다.

## 3. 유한 audit에서의 선택 (P5, 시뮬레이션 근거)

| 기준 | 정의 | S2 결과 요약 |
|---|---|---|
| EAV-plugin (debiased) | $\hat B^2-\hat V$ (편향² 불편추정) | ❌ **실패.** $\hat V$가 큰(잡음 많은) 모델에 유리하다. δ=0에서 BA보다 오차가 4–26배 크다. 기각 (ADR-0004) |
| EAV-direct | $\hat B^2$ | 큰 n에서만 우수 |
| **EAV-structural** | 불변성 예측 편향 $-(\widehat{FPR}+\widehat{FNR})\hat\theta_H$ 쪽으로 경험적 베이즈 수축 | ✅ δ=0에서 BA와 거의 동등. δ=1이면 n≥200에서 라벨 지표를 이긴다 (n=400: 초과오차 2.11pp vs accuracy 3.24pp) |
| **EAV-efficiency** | $\hat V$ (error regression SE²) | ✅ δ≥0.5, n≥200에서 accuracy를 확실히 이긴다 (δ=1, n=400: 유효표본 손실 6% vs 17%) |

**정직한 한계.** n≤100에서는 라벨 지표(accuracy/F1/BA)가 EAV보다 낫거나 같다. **EAV는 "작은 audit에서도 이긴다"가 아니라 "n≈200–400 이상에서, 차등 오류가 있을 때 이긴다"로 주장해야 한다.** 원 메모의 "작은 human-label budget에서도 EAV가 F1을 이긴다"는 현재 근거로는 지지되지 않는다.

## 4. 남은 이론 과제

1. P3의 부분적 역: switching의 **크기**(regret)와 차등 오류 크기의 정량 관계 (bound).
2. 선택 후 추론: 같은 audit로 선택과 교정을 하면 winner's curse가 생긴다 → cross-fitting (Zrnic & Candès 2024) 하에서 coverage 보장.
3. 다범주/연속 척도(RILE, logit scale; Lowe et al. 2011), 다국어(언어를 G로).
4. **선택 vs 혼합**: MoE-PPI의 분산최적 가중 결합과 비교. 선택의 장점은 코퍼스 전체에 모델을 1개만 돌린다는 비용 측면이다 (M배 절감).
5. Gold의 잡음: CMP human coding 자체의 오분류 (Mikhaylov et al. 2012) → estimand를 "human-coding estimand"로 정의하거나 다중 코더 자료로 민감도 분석.

## v0.1 → v0.2 수정 사항

- P1 예시에 실제로 가능한 $(\pi,p)$를 명시했다. 불변 조건의 역전은 BA로 해소되는 **알려진 결과**로 격하했다.
- "EAV = bias² + variance 최소화" → debiased 추정은 선택기로 실패하므로 structural 수축으로 교체했다.
- EAV→DSL 파이프라인의 논리 긴장(교정하면 편향은 무관) → 두 regime 분리, P4 추가.
- "estimand별 최적 모델이 다르다"를 1차 증거로 쓰던 계획 → argmin 불안정성 때문에 regret 기반으로 전환했다.
