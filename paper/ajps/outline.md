# AJPS manuscript — skeleton v0.1

> Status: pre-registration stage. Every `[RESULT]` is a placeholder to be filled only after the preregistered analyses run. Do not write results before they exist.
> Target length: AJPS limit (check current author guidelines) · Supplementary appendix holds proofs, simulations, prompts, and the full model × target multiverse.

## Title
**Can We Compare Parties Measured by Language Models?**

## Abstract (draft, ~170 words) — SUPERSEDED pending v3 design (docs/12): the correlation bound is a known result and must not be the headline; H2/H4 must become signed-magnitude equivalence tests; DIF only against same-text, same-construct human codes
Political scientists increasingly measure parties' issue emphases and positions with large language models, validating these measures by their correlation with human or expert codings. Yet most substantive claims in comparative politics are comparisons — between party families, countries, and periods — and such claims require that measurement error be equivalent across the groups being compared. We show that correlation-based validation cannot detect violations of this requirement: a validation correlation of 0.95 is compatible with a bias in a between-group comparison of more than half a standard deviation, enough to reverse its sign. Drawing on the logic of differential item functioning, we identify why language models may measure some parties and languages differently, and derive audit-based forecasts of how much a given comparison will be distorted. In preregistered replications of three published findings based on Manifesto Project human codings, [RESULT]. We propose estimand-aware validation, a simple procedure that tests the comparison a study actually makes, and show that it [RESULT].

## 1. Introduction (draft)

Large language models have entered the core measurement workflow of comparative politics. They now code party manifestos, legislative speeches and campaign messages at a scale no team of human coders can match, and recent work shows that their scores correlate highly with expert judgments (Benoit et al. 2026; Halterman and Keith 2026). The standard for accepting such a measure has become a high correlation with human codings or a high accuracy on a held-out set.

These validation statistics answer a different question from the one most comparative research asks. A study of mainstream-party responses to the radical right does not use the level of anti-immigration rhetoric in a manifesto; it uses the *difference* between party families, or the *change* in that difference after a radical-right breakthrough. A cross-national study compares issue emphases across countries whose manifestos are written in different languages. Such comparisons are valid only if the instrument errs in the same way for each group being compared — the requirement survey researchers call measurement equivalence, and whose violation they call differential item functioning (King et al. 2004).

[Positioning paragraph — engage directly: Benoit et al. (2026) compare LLM and expert score distributions and find no bias on four of six dimensions relative to experts as a surrogate for ground truth, and a GPT-4o leakage check finds masking party names barely changes scores (r = .99). These are pooled checks, as their purpose required. Jankowski and Huber (2023) warned that correlation is not enough for supervised classifiers; Vallejo Vera and Driggers (2025) showed that party cues move LLM labels. What remains unknown is whether LLM measures are *equivalent across the groups comparative research compares*, and whether non-equivalence changes published-style conclusions. See docs/01_positioning.md §3b for the verified differentiation.]

We make three contributions. First, we show analytically that correlation-based validation is blind to exactly this violation. Because a pooled correlation is dominated by variation within groups, an LLM measure that shifts one party family by a quarter of a standard deviation can correlate at 0.99 with human codings while reversing the sign of the party-family gap (Proposition 1). Second, we identify mechanisms that make such differential error plausible for language models — uneven performance across languages, the coded and euphemistic rhetoric of some party families, sensitivity to party cues, and training-data contamination — and show how a small human audit can forecast, before a full corpus is coded, how much a given comparison will be distorted. Third, we test these forecasts in preregistered replications of three findings based on Manifesto Project human codings: a cross-national comparison, the contested effect of radical-right success on mainstream parties, and a large party-family difference that should be robust. [RESULT: one paragraph, written after analysis.]

We conclude with a procedure, estimand-aware validation, that replaces "does the model agree with humans?" with "does the model reproduce the comparison my study makes?", and with conditions under which LLM-based comparative measurement can be trusted.

## 2. Why validation by agreement fails for comparisons
- 2.1 Setup: document-level human score x, LLM score m, group G (party family, language)
- 2.2 Proposition 1 (P6 in docs/02_theory.md): the correlation–bias bound; table of undetectable bias by r*
- 2.3 From sentences to manifestos: sentence-level group-specific FPR/FNR aggregate to manifesto-level shifts (error regression, P0)
- 2.4 Accuracy and F1 share the blind spot; balanced accuracy suffices only under invariance (P1)

## 3. Why language models may measure groups differently
- Language resources (cross-national comparisons are confounded with language)
- Coded, implicit and euphemistic rhetoric (radical right; Lerner & Yvon 2026 perplexity evidence)
- Party cues (Vallejo Vera & Driggers 2025)
- Contamination (Halterman & Keith 2026) — can differ across countries and periods

## 4. Research design
- 4.1 Preregistration: targets selected by a mechanical rule before any LLM output is seen
- 4.2 Replication targets A (cross-national), B (radical-right contagion), C (placebo)
- 4.3 Models: 6–8 open-weight (local) + API models, pinned versions, one fixed prompt per target, temperature 0
- 4.4 Audit-based distortion forecasts (≈300 sentences per group), registered before full coding
- 4.5 Adjudication: expert/crowd multi-coded sentences; blind self-coding with partial double coding; post-cutoff manifestos for contamination
- 4.6 Hypotheses
  - **H1** Differential error: for ≥1 model, group-specific FPR/FNR differ beyond sampling error (language or party family)
  - **H2** Correlation blindness: models with manifesto-level r ≥ .90 still produce group-contrast bias ≥ .25 SD for ≥1 target
  - **H3** Forecastability: audit forecasts predict realized distortions (sign agreement; rank correlation across model × target)
  - **H4** Differential robustness: placebo conclusion (C) holds for all models; contested conclusions (A/B) change sign or significance for ≥1 model that passes conventional validation
  - **H5** Remedy: estimand-aware validation plus DSL/PPI correction recovers human-coded estimates within their confidence intervals
  - **H6** Human benchmark: LLM–expert differences in comparative conclusions are no larger than crowd–expert differences for the same sentences (Benoit et al. 2016 multi-coded data)

> ~~Candidate opening example (UK expert vs crowd)~~ — WITHDRAWN (docs/12 A1, B1-8): the gap is mostly attenuation (slope 0.62) plus manifesto-specific noise; party-level shift ≈ ±0.03. Use only as a human noise floor, never as evidence of non-equivalence.

## 5. Results
- Fig. 1 Validation correlation vs. comparison bias, every model × target (the blind spot, empirically)
- Fig. 2 Group-specific error rates by language and party family
- Fig. 3 Forecast vs. realized distortion
- Fig. 4 Multiverse of substantive conclusions for A, B, C across models
- Table 1 Replication estimates: human coding vs. each model vs. corrected

## 6. What researchers should do
- Estimand-aware validation in five steps; the Estimand Validation Card (appendix)
- Stratified audits that oversample small parties and non-English languages

## 7. Conclusion

## Appendix plan
A proofs · B simulations (S1–S3) · C prompts and model versions · D full multiverse · E contamination checks · F adjudication protocol and coder agreement

## References to verify before submission
King, Murray, Salomon & Tandon (2004) APSR — DIF / anchoring vignettes · Benoit et al. (AJPS, year/volume) · Halterman & Keith (2026) PA · Vallejo Vera & Driggers (2025) · Lerner & Yvon (2026) · replication targets (docs/09)
