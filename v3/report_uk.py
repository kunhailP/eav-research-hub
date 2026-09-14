"""Report the confirmatory UK results exactly as preregistered (docs/13 §4–§5), written before the results were read.

Every cell is shown, not a selection: area x model x paraphrase x party contrast, with the primary
estimate (pp), 95% CI, call, whether it lies outside the human leave-one-out range, and the secondary
error-rate calls. Model-level and study-level conclusions follow the locked rules.

  python v3/report_uk.py            # writes results/v3_uk_full/REPORT.md
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v3_uk_full"
PARTY = {"delta_pp:1": "Con − Lab", "delta_pp:2": "LD − Lab", "sens_diff:1": "Con − Lab", "sens_diff:2": "LD − Lab",
         "fpr_diff:1": "Con − Lab", "fpr_diff:2": "LD − Lab"}


def main():
    cells = pd.read_csv(OUT / "cells.csv")
    models = pd.read_csv(OUT / "model_conclusions.csv")
    study = json.loads((OUT / "study_conclusion.json").read_text())
    lines = ["# UK 확증 결과 (prereg-v3-uk)", "",
             "사전등록 규칙 그대로의 전체 셀 보고. 1차 = `delta_pp`(pp, SESOI 3), 인간 범위 = 인간 코더 leave-one-out 추정치 [최소, 최대].", ""]
    lines += [f"**연구 수준 결론 (H1: 서로 다른 계열 모델 2개 이상이 같은 집단·부호에서 비동등):** {'지지' if study['H1_supported'] else '지지되지 않음'}", ""]

    prim = cells[cells["stat"].str.startswith("delta_pp")].copy()
    prim["contrast"] = prim["stat"].map(PARTY)
    for area in ["economic", "social"]:
        lines += [f"## {area} 영역 — 1차 셀", "",
                  "| 모델 | 변형 | 대비 | 추정치 (pp) | 95% CI | 판정 | 인간 범위 | 인간 범위 밖 | 무효율 |", "|---|---|---|---|---|---|---|---|---|"]
        for r in prim[prim["area"] == area].sort_values(["model", "paraphrase", "stat"]).itertuples():
            human = f"[{r.human_min:+.1f}, {r.human_max:+.1f}]" if pd.notna(r.human_min) else "—"
            lines.append(f"| {r.model.split('/')[-1]} | {r.paraphrase} | {r.contrast} | {r.estimate:+.2f} | [{r.lo95:+.2f}, {r.hi95:+.2f}] | "
                         f"{r.call} | {human} | {bool(r.exceeds_human_range) if pd.notna(r.exceeds_human_range) else '—'} | {r.invalid_rate:.3f} |")
        lines += ["", f"### {area} 영역 — 모델 수준 결론", "", "| 모델 | 대비 | 결론 | 부호 | 추정치 중앙값 | 인간 범위 밖 변형 수 |", "|---|---|---|---|---|---|"]
        for r in models[models["area"] == area].sort_values(["model", "stat"]).itertuples():
            lines.append(f"| {r.model.split('/')[-1]} | {PARTY[r.stat]} | {r.conclusion} | {r.sign:+d} | {r.median_estimate:+.2f} | {r.paraphrases_outside_human_range} |")
        sec = cells[(cells["area"] == area) & ~cells["stat"].str.startswith("delta_pp")]
        tally = sec.groupby([sec["model"].str.split("/").str[-1], "stat"])["call"].agg(lambda s: dict(s.value_counts())).reset_index()
        lines += ["", f"### {area} 영역 — 2차 오류율 통계 판정 집계 (변형 3개)", "", "| 모델 | 통계 | 판정 분포 |", "|---|---|---|"]
        lines += [f"| {r.model} | {r.stat} | {r.call} |" for r in tally.itertuples()]
        lines.append("")
    lines += ["## 해석 시 반드시 함께 읽을 것", "",
              "- UK 규모의 검정력 한계: 95% CI 폭 약 5–10pp, 약 5pp 이상만 판별(docs/12 D11). 판정 불가는 동등성의 증거가 아니다.",
              "- 인간 코더도 경제 영역에서 3–5pp 비동등(크라우드, 전문가 2명; D17). \"인간보다 더 비동등\"은 인간 범위 밖 기준을 함께 만족할 때만.",
              "- 사회 영역은 유병률이 낮아 추정량 수준 동등이 오류율 동등을 뜻하지 않는다(D19). 2차 통계를 함께 볼 것.",
              "- 계열 p는 실제 자료에서 반보수적일 수 있다(D17). 판정은 bootstrap CI 조건을 함께 요구한다.",
              "- 오염 가능성: Benoit 2016 자료는 공개 저장소에 코드와 함께 있다. 동등 결과의 일부가 암기일 수 있다."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT / 'REPORT.md'} ({len(prim)} primary cells)")


if __name__ == "__main__":
    sys.exit(main())
