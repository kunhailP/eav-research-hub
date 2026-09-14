# AJPS manuscript — skeleton v0.2

> Status (2026-09-14): preregistered studies partly run. **The research question is fixed.** The abstract and introduction may state only claims marked "established" in the evidence ledger (README); everything else stays `[RESULT]` until its preregistered analysis has run. v0.1 headline claims that were demoted or withdrawn are listed at the end with reasons.
> Target length: AJPS limit (check current author guidelines) · Supplementary appendix holds proofs, calibrations, prompts, model revisions, and every preregistered table.

## Title
**Can We Compare Parties Measured by Language Models?**

## Abstract (draft v0.2, ~170 words)
Political scientists increasingly measure parties' issue emphases and positions with large language models and validate these measures by their agreement with human or expert codings. Yet the core claims of comparative politics are comparisons between party families, countries and periods, and such claims require that the instrument err in the same way in every group compared. We test this requirement directly, against human codings of the same texts, with preregistered equivalence tests that benchmark language models against the disagreement among human coders. In British manifestos, small open-weight models err differently by party and distort a party comparison beyond the range of human coders. In a preregistered reanalysis of Benoit et al. (2026), a frontier-model ensemble and expert surveys show no robust divergence across eleven prespecified comparisons. [RESULT: same-text tests of radical-right versus mainstream parties in fourteen countries; current frontier models; party-cue experiment; model-size ladder.] [RESULT: whether a small human audit tells researchers in advance which comparisons are safe.]

## 1. Introduction (draft)

Large language models have entered the core measurement workflow of comparative politics. They now code party manifestos, legislative speeches and campaign messages at a scale no team of human coders can match, and recent work shows that their scores agree closely with expert judgments (Benoit et al. 2026; Halterman and Keith 2026). The standard for accepting such a measure has become high agreement with human codings or high accuracy on a held-out set.

These validation statistics answer a different question from the one most comparative research asks. A study of mainstream-party responses to the radical right does not use the level of anti-immigration rhetoric in a manifesto; it uses the *difference* between party families, or the *change* in that difference after a radical-right breakthrough. A cross-national study compares issue emphases across countries whose manifestos are written in different languages. Such comparisons are valid only if the instrument errs in the same way for each group being compared — the requirement survey researchers call measurement equivalence, and whose violation they call differential item functioning (King et al. 2004).

[Positioning paragraph — engage directly and accurately (docs/11 §3.3 tone rules): Benoit et al. (2026) validate with correlations, distribution comparisons, Manifesto Project triangulation, replication runs, open-weight models, and a leakage check; these are pooled over parties and countries, as their purpose required. Jankowski and Huber (2023) warned that correlation is not enough for supervised classifiers; Vallejo Vera and Driggers (2025) showed that randomly assigned party cues move LLM labels. What remains unknown is whether LLM measures are equivalent across the groups comparative research compares. See docs/01_positioning.md §3b.]

[Why it may fail for capable models: stronger models know more about parties, so reputation can leak into text judgments — random error may fall while group-specific error does not (E7, E8).]

We make three contributions. First, a test of measurement equivalence for machine coders at the level of the comparison a study reports: group-specific error estimated against human codings of the same texts, a human noise floor from leave-one-out coders, and three-way calls (non-equivalent, equivalent, inconclusive) whose false-call rates are calibrated before any model output is seen. Second, evidence on when the requirement holds: it fails for small open-weight models in British manifestos, it is not contradicted for a frontier ensemble's document-level positions in Benoit et al.'s data, and [RESULT: same-text radical-right comparisons, frontier models, cue mechanism, size ladder]. Third, a procedure a researcher can run before coding a full corpus: a small human audit that forecasts the distortion of a given comparison [RESULT: whether it beats conventional accuracy thresholds].

We conclude with conditions under which LLM-based comparative measurement can be trusted, stated in terms a researcher can check.

## 2. Framework
- 2.1 Setup: texts, human codings, LLM codings, groups (party family, language); the comparison as the estimand
- 2.2 Error regression (P0, docs/02): plug-in distortion = projection of the model's error on the estimand's design; group-specific sensitivity and false-positive rate determine the distortion of a share comparison
- 2.3 Balanced accuracy suffices only under invariance (P1); pooled agreement cannot rule out between-group bias (known bound, stated briefly with Bland–Altman, not as a contribution)
- 2.4 Benchmarks: human coders disagree too — leave-one-out human range; construct mismatch when the benchmark rates parties (expert surveys) rather than texts

## 3. Why language models may measure groups differently
- Language resources (cross-national comparisons are confounded with language)
- Coded, implicit and euphemistic rhetoric (radical right; Lerner & Yvon 2026)
- Party cues and reputation (Vallejo Vera & Driggers 2025), plausibly stronger in more knowledgeable models
- Contamination (Halterman & Keith 2026)

## 4. Research design (every study preregistered and locked before outputs were compared with humans)
| Study | Benchmark | Groups | Models | Prereg | Status |
|---|---|---|---|---|---|
| UK manifestos 1987–2010 | same sentences, 8 experts + crowd | Con / Lab / LD | 4 open-weight 7–8B | docs/13 | done (E1) |
| PImPo, 14 countries | same sentences, crowd | radical right / mainstream | 4 open-weight | docs/14 | running (E5) |
| Benoit et al. (2026) reanalysis | expert surveys (party ratings) | party families, East/West | frontier ensemble + 13 configurations | docs/16 | done (E3) |
| Audit forecast | same as UK / PImPo | as above | as above | docs/18 | UK exploratory (E4), PImPo pending |
| Party-name masking / swapping | same sentences | as PImPo | open-weight (+ frontier) | to register | planned (E7) |
| Model-size ladder | same sentences | as UK / PImPo | Qwen2.5 0.5B–14B | to register | planned (E8) |
| Frontier models, same text | same sentences | as UK / PImPo | current API models | to register | planned, separate (E6) |

Hypotheses as registered: H1 (UK, PImPo: model-level non-equivalence in ≥ 2 model families), H_E1 (Benoit reanalysis: robust divergence in ≥ 1 of 11 contrasts across the reliability grid), F1–F2 (audit forecast vs balanced-accuracy threshold: discrimination and model selection).

## 5. Results (write only from the ledger)
- Fig. 1 Group-specific error rates by model and group, against the human range (UK, PImPo, ladder, frontier)
- Fig. 2 Comparison distortion with calibrated three-way calls, by study
- Fig. 3 Benoit reanalysis: eleven contrasts across the reliability grid, with triangulation
- Fig. 4 Audit forecast vs full-benchmark distortion; rules' discrimination and regret
- Table 1 Evidence ledger: claim, status, study, preregistration

## 6. What researchers should do
- Test the comparison you report, not pooled agreement; the Estimand Validation Card (appendix)
- Audits that oversample small parties and languages; audit size from the calibration results

## 7. Conclusion

## Appendix plan
A proofs · B calibrations of every decision rule (false-call rates, power) · C prompts, option shuffling, model revisions · D full preregistered tables · E deviations and bugs (docs/12 §D) · F data access and lock verification

## Demoted or withdrawn from v0.1 (do not reintroduce without new evidence)
- "Correlation of 0.95/0.99 compatible with a sign reversal" as the headline (Proposition 1) — known result (Bland–Altman); kept only as background in §2.3.
- "Preregistered replications of three published findings" — not what was run; the design is the studies in §4.
- H2 (correlation blindness as an existence claim) and H4 (≥ 1 model changes significance) — existence claims; replaced by calibrated equivalence tests.
- H6 (LLM vs crowd–expert differences) as a hypothesis — kept only as the human noise floor.
- UK expert-vs-crowd opening example — attenuation, not non-equivalence (docs/12 A1).

## References to verify before submission
King, Murray, Salomon & Tandon (2004) APSR · Benoit, De Marchi, Laver, Laver & Ma (2026) AJPS · Halterman & Keith (2026) PA · Vallejo Vera & Driggers (2025) · Lerner & Yvon (2026) · Jankowski & Huber (2023) PA · Lehmann & Zobel (2018) PImPo · Benoit et al. (2016) APSR
