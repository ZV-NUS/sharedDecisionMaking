"""Review generated experimental results for evidence-supported claims.

This script is intentionally conservative. It checks whether the current
repository outputs support the core paper claim that the proposed authority
correction and MPC-lite closed-loop controller improve the reference authority
pipeline in the validated scenarios. It does not certify superiority over
external baselines that are not present in the repository.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
TABLE_DIR = RESULT_DIR / "tables"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rollouts(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    payload = text.split("=", 1)[1].strip().rstrip(";")
    return json.loads(payload)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def check(condition: bool, passed: list[str], failed: list[str], message: str) -> None:
    (passed if condition else failed).append(message)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    passed: list[str] = []
    failed: list[str] = []
    cautions: list[str] = []

    rl_metrics = load_json(ROOT / "checkpoints" / "rl_authority" / "test_eval_metrics.json")
    rollouts = load_rollouts(
        ROOT / "outputs" / "shared_authority_validation" / "shared_authority_rollouts.js"
    )
    method_table = load_csv(TABLE_DIR / "method_level_metrics.csv")
    scenario_table = load_csv(TABLE_DIR / "scenario_validation_metrics.csv")

    ref = rl_metrics["reference_authority"]
    rl = rl_metrics["rl_authority"]

    check(
        rl["reward_mean"] > ref["reward_mean"],
        passed,
        failed,
        "RL authority improves mean reward relative to the fuzzy/reference authority prior.",
    )
    check(
        rl["shared_risk_mean"] < ref["shared_risk_mean"],
        passed,
        failed,
        "RL authority reduces mean shared risk relative to the reference authority prior.",
    )
    check(
        rl["shared_risk_max"] < ref["shared_risk_max"],
        passed,
        failed,
        "RL authority reduces worst-case shared risk relative to the reference authority prior.",
    )
    check(
        rl["comfort_cost"] < ref["comfort_cost"],
        passed,
        failed,
        "RL authority reduces the comfort cost relative to the reference authority prior.",
    )

    controller_collisions = [case["metrics"].get("controller_collision", False) for case in rollouts["cases"]]
    check(
        not any(controller_collisions),
        passed,
        failed,
        "The proposed controller has no reported collision in all configured validation cases.",
    )

    method_rows = {row["method"]: row for row in method_table}
    proposed_clearance = float(method_rows["Proposed MPC"]["mean_min_clearance"])
    reference_clearance = float(method_rows["Reference authority"]["mean_min_clearance"])
    check(
        proposed_clearance > reference_clearance,
        passed,
        failed,
        "The proposed MPC trajectory improves average minimum clearance over the reference-authority trajectory.",
    )

    scenario_improved = 0
    scenario_equal_or_better = 0
    for case in rollouts["cases"]:
        prop = float(case["controller_ego"]["min_clearance_m"])
        ref_clear = float(case["reference_ego"]["min_clearance_m"])
        if prop > ref_clear:
            scenario_improved += 1
        if prop >= ref_clear:
            scenario_equal_or_better += 1
    check(
        scenario_equal_or_better >= 5,
        passed,
        failed,
        "The proposed controller is not worse than the reference trajectory in at least five of seven scenarios.",
    )

    if scenario_improved < len(scenario_table):
        cautions.append(
            f"Per-scenario clearance is improved in {scenario_improved}/{len(scenario_table)} cases; "
            "therefore, the manuscript should claim average or task-level improvement rather than universal dominance."
        )

    human_record_clearance = float(method_rows["Human record"]["mean_min_clearance"])
    if proposed_clearance <= human_record_clearance:
        cautions.append(
            "The proposed controller does not exceed the recorded human trajectory in average minimum clearance. "
            "This is acceptable for a shared-control validation but should not be written as dominance over human driving."
        )

    if rl["efficiency_cost"] > ref["efficiency_cost"]:
        cautions.append(
            "The RL authority has a higher efficiency cost than the reference authority in the available test file. "
            "Discuss this as a safety-comfort tradeoff unless additional tuning results are added."
        )

    status = "PASS" if not failed else "FAIL"
    report = [
        "# Review Agent Report",
        "",
        f"Overall decision: **{status}**",
        "",
        "## Checked Evidence",
        "",
        "- Source files: RL authority test metrics, shared-authority rollouts, and generated result tables.",
        "- Claim scope: superiority of the proposed authority-correction and MPC-lite pipeline over the reference-authority pipeline under the current validation configuration.",
        "- Excluded scope: external baselines, unseen datasets, and universal dominance over human-recorded trajectories.",
        "",
        "## Passed Checks",
        "",
    ]
    report.extend([f"- {item}" for item in passed] or ["- None."])
    report.extend(["", "## Failed Checks", ""])
    report.extend([f"- {item}" for item in failed] or ["- None."])
    report.extend(["", "## Cautions for Manuscript Writing", ""])
    report.extend([f"- {item}" for item in cautions] or ["- None."])
    report.extend(
        [
            "",
            "## Numeric Summary",
            "",
            f"- Reference reward: {ref['reward_mean']:.4f}; RL reward: {rl['reward_mean']:.4f}.",
            f"- Reference mean risk: {ref['shared_risk_mean']:.4f}; RL mean risk: {rl['shared_risk_mean']:.4f}.",
            f"- Reference worst risk: {ref['shared_risk_max']:.4f}; RL worst risk: {rl['shared_risk_max']:.4f}.",
            f"- Reference comfort cost: {ref['comfort_cost']:.4f}; RL comfort cost: {rl['comfort_cost']:.4f}.",
            f"- Average minimum clearance, reference authority: {reference_clearance:.2f} m; proposed MPC: {proposed_clearance:.2f} m.",
            f"- Scenarios with proposed clearance greater than reference clearance: {scenario_improved}/{len(scenario_table)}.",
            "",
            "## Recommendation",
            "",
            "Use the generated figures as evidence for an ablation-oriented claim: the proposed RL authority correction and adaptive MPC-lite execution improve the reference authority pipeline in the current validation set. Avoid claiming superiority over all human or external baselines until additional experiments are added.",
        ]
    )

    (RESULT_DIR / "review_agent_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (RESULT_DIR / "review_agent_report.json").write_text(
        json.dumps(
            {
                "status": status,
                "passed": passed,
                "failed": failed,
                "cautions": cautions,
                "scenario_improved_over_reference": scenario_improved,
                "scenario_count": len(scenario_table),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Review agent decision: {status}")
    print(f"Report: {RESULT_DIR / 'review_agent_report.md'}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
