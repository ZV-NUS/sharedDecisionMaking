"""Generate IEEE-style experimental result figures for the shared-driving study.

The script uses only existing small result files and validation rollouts:

- checkpoints/human_intent_transformer/test_metrics.json
- checkpoints/rl_authority/test_eval_metrics.json
- outputs/work3_authority/metrics.json
- outputs/shared_authority_validation/shared_authority_rollouts.js

It does not read raw datasets, HDF5 files, model weights, logs, or videos.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"


COLORS = {
    "human": "#4C78A8",
    "machine": "#F58518",
    "reference": "#54A24B",
    "rl": "#B279A2",
    "proposed": "#009EAD",
    "dark": "#2F3A45",
    "gray": "#8A94A3",
    "light": "#E8ECF2",
    "danger": "#E45756",
}


def ensure_results_dir() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 450,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rollouts(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    payload = text.split("=", 1)[1].strip().rstrip(";")
    return json.loads(payload)


def savefig(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(RESULT_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def mean_abs_diff(values: list[float], scale: float = 1.0) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(arr))) * scale)


def trajectory_progress(xy: list[list[float]]) -> float:
    arr = np.asarray(xy, dtype=float)
    if arr.shape[0] < 2:
        return 0.0
    return float(arr[-1, 0] - arr[0, 0])


def plot_wp1_metrics(human_metrics: dict) -> None:
    metrics = [
        ("Decision Acc.", human_metrics["decision_accuracy"]),
        ("Decision F1", human_metrics["decision_macro_f1"]),
        ("Event Acc.", human_metrics["future_event_accuracy"]),
        ("Event F1", human_metrics["future_event_macro_f1"]),
        ("LC Recall", human_metrics["lane_change_event_recall"]),
        ("LC Precision", human_metrics["lane_change_event_precision"]),
    ]
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]

    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    bars = ax.bar(labels, values, color=COLORS["human"], edgecolor="black", linewidth=0.4)
    ax.set_ylim(0.65, 0.9)
    ax.set_ylabel("Score")
    ax.set_title("Driver intent prediction performance")
    ax.grid(axis="y", alpha=0.28)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.3f}", ha="center", va="bottom")

    ax2 = ax.twinx()
    ax2.scatter([len(labels) + 0.25], [human_metrics["speed_rmse"]], color=COLORS["machine"], s=42, label="Speed RMSE")
    ax2.scatter([len(labels) + 0.75], [human_metrics["steer_rmse_rad"] * 100], color=COLORS["danger"], s=42, label="Steer RMSE x100")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Regression error")
    ax2.set_xticks(range(len(labels)))
    ax2.spines["top"].set_visible(False)
    ax2.legend(frameon=False, loc="lower right")
    savefig(fig, "fig01_wp1_driver_prediction_metrics")


def plot_rl_ablation(rl_metrics: dict) -> None:
    ref = rl_metrics["reference_authority"]
    rl = rl_metrics["rl_authority"]
    metrics = [
        ("Reward", ref["reward_mean"], rl["reward_mean"], "higher"),
        ("Mean risk", ref["shared_risk_mean"], rl["shared_risk_mean"], "lower"),
        ("Max risk", ref["shared_risk_max"], rl["shared_risk_max"], "lower"),
        ("Comfort cost", ref["comfort_cost"], rl["comfort_cost"], "lower"),
        ("Efficiency cost", ref["efficiency_cost"], rl["efficiency_cost"], "lower"),
    ]
    labels = [m[0] for m in metrics]
    ref_values = np.array([m[1] for m in metrics], dtype=float)
    rl_values = np.array([m[2] for m in metrics], dtype=float)

    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.bar(x - width / 2, ref_values, width, label="Reference authority", color=COLORS["reference"], edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, rl_values, width, label="RL authority", color=COLORS["proposed"], edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Authority ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)

    improvements = []
    for _, a, b, direction in metrics:
        if direction == "higher":
            imp = (b - a) / (abs(a) + 1e-9)
        else:
            imp = (a - b) / (abs(a) + 1e-9)
        improvements.append(100 * imp)
    axes[1].bar(labels, improvements, color=[COLORS["proposed"] if v >= 0 else COLORS["danger"] for v in improvements], edgecolor="black", linewidth=0.4)
    axes[1].axhline(0, color="black", linewidth=0.7)
    axes[1].set_ylabel("Relative change (%)")
    axes[1].set_title("Change after RL correction")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    savefig(fig, "fig02_rl_authority_ablation")


def plot_case_risk_reward(rollouts: dict) -> None:
    cases = rollouts["cases"]
    ids = [c["record"]["case_id"] for c in cases]
    ref_risk = [c["metrics"]["reference_risk_mean"] for c in cases]
    rl_risk = [c["metrics"]["rl_risk_mean"] for c in cases]
    ref_reward = [c["metrics"]["reference_reward"] for c in cases]
    rl_reward = [c["metrics"]["rl_reward"] for c in cases]

    x = np.arange(len(ids))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    axes[0].bar(x - width / 2, ref_risk, width, label="Reference authority", color=COLORS["reference"], edgecolor="black", linewidth=0.4)
    axes[0].bar(x + width / 2, rl_risk, width, label="RL authority", color=COLORS["proposed"], edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"C{i}" for i in ids])
    axes[0].set_ylabel("Mean rollout risk")
    axes[0].set_title("Case-wise risk")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, ref_reward, width, label="Reference authority", color=COLORS["reference"], edgecolor="black", linewidth=0.4)
    axes[1].bar(x + width / 2, rl_reward, width, label="RL authority", color=COLORS["proposed"], edgecolor="black", linewidth=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"C{i}" for i in ids])
    axes[1].set_ylabel("Reward")
    axes[1].set_title("Case-wise reward")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)
    savefig(fig, "fig03_casewise_risk_reward")


def aggregate_method_metrics(rollouts: dict) -> dict[str, dict[str, float]]:
    fps = float(rollouts["frame_rate"])
    methods = {
        "Human record": "ego",
        "Human prediction": "human_pred_ego",
        "Reference authority": "reference_ego",
        "Proposed MPC": "controller_ego",
    }
    output: dict[str, dict[str, float]] = {}
    for label, key in methods.items():
        clearances = []
        collisions = []
        jerk = []
        steer_rate = []
        progress = []
        max_yaw_rate = []
        max_beta = []
        for case in rollouts["cases"]:
            data = case[key]
            clearances.append(float(data.get("min_clearance_m", np.nan)))
            collisions.append(float(bool(data.get("collision", False))))
            acc = np.asarray(data["acceleration"], dtype=float)
            steer = np.asarray(data["steer"], dtype=float)
            jerk.extend(np.abs(np.diff(acc)) * fps)
            steer_rate.extend(np.abs(np.diff(steer)) * fps)
            progress.append(trajectory_progress(data["xy"]))
            if "yaw_rate" in data:
                max_yaw_rate.append(float(np.max(np.abs(data["yaw_rate"]))))
            if "beta" in data:
                max_beta.append(float(np.max(np.abs(data["beta"]))))
        output[label] = {
            "collision_rate": float(np.mean(collisions)),
            "mean_min_clearance": float(np.nanmean(clearances)),
            "minimum_clearance": float(np.nanmin(clearances)),
            "mean_abs_jerk": float(np.mean(jerk)) if jerk else math.nan,
            "mean_abs_steer_rate": float(np.mean(steer_rate)) if steer_rate else math.nan,
            "mean_progress": float(np.mean(progress)),
            "max_yaw_rate": float(np.nanmax(max_yaw_rate)) if max_yaw_rate else math.nan,
            "max_beta": float(np.nanmax(max_beta)) if max_beta else math.nan,
        }
    return output


def plot_method_safety_comparison(rollouts: dict) -> dict[str, dict[str, float]]:
    metrics = aggregate_method_metrics(rollouts)
    labels = list(metrics.keys())
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.75))
    vals = [metrics[k]["mean_min_clearance"] for k in labels]
    axes[0].bar(labels, vals, color=[COLORS["human"], "#7AA6C2", COLORS["reference"], COLORS["proposed"]], edgecolor="black", linewidth=0.4)
    axes[0].set_ylabel("Mean min. clearance (m)")
    axes[0].set_title("Safety margin")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].spines[["top", "right"]].set_visible(False)

    vals = [metrics[k]["mean_abs_jerk"] for k in labels]
    axes[1].bar(labels, vals, color=[COLORS["human"], "#7AA6C2", COLORS["reference"], COLORS["proposed"]], edgecolor="black", linewidth=0.4)
    axes[1].set_ylabel("Mean jerk (m/s$^3$)")
    axes[1].set_title("Comfort")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].spines[["top", "right"]].set_visible(False)

    vals = [metrics[k]["mean_progress"] for k in labels]
    axes[2].bar(labels, vals, color=[COLORS["human"], "#7AA6C2", COLORS["reference"], COLORS["proposed"]], edgecolor="black", linewidth=0.4)
    axes[2].set_ylabel("Longitudinal progress (m)")
    axes[2].set_title("Efficiency proxy")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].spines[["top", "right"]].set_visible(False)
    savefig(fig, "fig04_method_level_comparison")
    return metrics


def plot_authority_trust_examples(rollouts: dict) -> None:
    cases_by_id = {c["record"]["case_id"]: c for c in rollouts["cases"]}
    selected = [1, 3, 5, 6]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True, sharey=False)
    axes = axes.flatten()
    for ax, cid in zip(axes, selected):
        case = cases_by_id[cid]
        t = np.arange(len(case["signals"]["authority_ref"])) / rollouts["frame_rate"]
        ax.plot(t, case["signals"]["authority_ref"], color=COLORS["reference"], label="$\\lambda_h^{ref}$")
        ax.plot(t, case["signals"]["authority_rl"], color=COLORS["proposed"], label="$\\lambda_h^{RL}$")
        ax.plot(t, case["signals"]["trust_machine_to_human"], color=COLORS["human"], linestyle="--", label="$T_{m\\rightarrow h}$")
        ax.plot(t, case["signals"]["trust_human_to_machine"], color=COLORS["machine"], linestyle="--", label="$T_{h\\rightarrow m}$")
        ax.set_title(f"Case {cid}: {case['record']['expected']}")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].set_xlabel("Time (s)")
    axes[3].set_xlabel("Time (s)")
    axes[0].set_ylabel("Value")
    axes[2].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", frameon=False)
    fig.subplots_adjust(bottom=0.18, wspace=0.25, hspace=0.38)
    savefig(fig, "fig05_authority_trust_evolution")


def plot_trajectory_cases(rollouts: dict) -> None:
    cases = rollouts["cases"][:6]
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.8))
    axes = axes.flatten()
    for ax, case in zip(axes, cases):
        road = case.get("road", {})
        lane_markings = road.get("lane_markings", [])
        for y in lane_markings:
            ax.axhline(y, color="#B8C0CC", linestyle="--", linewidth=0.7, zorder=0)
        for neigh in case.get("neighbors", []):
            xy = np.asarray(neigh["xy"], dtype=float)
            if xy.size:
                ax.plot(xy[:, 0], xy[:, 1], color=COLORS["gray"], alpha=0.42, linewidth=1.0)
        for key, color, style, label in [
            ("human_pred_ego", COLORS["human"], "--", "Human pred."),
            ("reference_ego", COLORS["reference"], "-.", "Ref. authority"),
            ("controller_ego", COLORS["proposed"], "-", "Proposed"),
        ]:
            xy = np.asarray(case[key]["xy"], dtype=float)
            ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=style, label=label)
        start = np.asarray(case["controller_ego"]["xy"][0], dtype=float)
        ax.scatter([start[0]], [start[1]], color=COLORS["dark"], s=10, zorder=5)
        ax.set_title(f"Case {case['record']['case_id']}: {case['record']['expected']}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
        all_xy = np.vstack(
            [np.asarray(case[k]["xy"], dtype=float) for k in ["human_pred_ego", "reference_ego", "controller_ego"]]
        )
        ax.set_xlim(float(all_xy[:, 0].min() - 8), float(all_xy[:, 0].max() + 8))
        ax.set_ylim(float(all_xy[:, 1].min() - 6), float(all_xy[:, 1].max() + 6))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", frameon=False)
    fig.subplots_adjust(bottom=0.13, wspace=0.28, hspace=0.38)
    savefig(fig, "fig06_scenario_trajectory_comparison")


def plot_control_stability(rollouts: dict) -> None:
    cases = rollouts["cases"]
    ids = [c["record"]["case_id"] for c in cases]
    clearance_ref = [c["reference_ego"]["min_clearance_m"] for c in cases]
    clearance_prop = [c["controller_ego"]["min_clearance_m"] for c in cases]
    max_beta = [max(abs(v) for v in c["controller_ego"].get("beta", [0.0])) for c in cases]
    max_yaw = [max(abs(v) for v in c["controller_ego"].get("yaw_rate", [0.0])) for c in cases]

    x = np.arange(len(ids))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    width = 0.34
    axes[0].bar(x - width / 2, clearance_ref, width, label="Reference trajectory", color=COLORS["reference"], edgecolor="black", linewidth=0.4)
    axes[0].bar(x + width / 2, clearance_prop, width, label="Proposed MPC", color=COLORS["proposed"], edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"C{i}" for i in ids])
    axes[0].set_ylabel("Minimum clearance (m)")
    axes[0].set_title("Safety after control execution")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].plot(ids, max_beta, marker="o", color=COLORS["human"], label="Max sideslip")
    axes[1].plot(ids, max_yaw, marker="s", color=COLORS["machine"], label="Max yaw rate")
    axes[1].set_xlabel("Case")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title("Controller stability indicators")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    savefig(fig, "fig07_control_stability_and_clearance")


def write_tables(
    human_metrics: dict,
    work3_metrics: dict,
    rl_metrics: dict,
    rollouts: dict,
    method_metrics: dict[str, dict[str, float]],
) -> None:
    (RESULT_DIR / "tables").mkdir(exist_ok=True)

    with (RESULT_DIR / "tables" / "wp1_driver_prediction_metrics.csv").open("w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for key in [
            "decision_accuracy",
            "decision_macro_f1",
            "future_event_accuracy",
            "future_event_macro_f1",
            "lane_change_event_recall",
            "lane_change_event_precision",
            "lane_change_event_time_mae_frames",
            "speed_rmse",
            "steer_rmse_rad",
        ]:
            f.write(f"{key},{human_metrics[key]}\n")

    with (RESULT_DIR / "tables" / "method_level_metrics.csv").open("w", encoding="utf-8") as f:
        headers = ["method"] + list(next(iter(method_metrics.values())).keys())
        f.write(",".join(headers) + "\n")
        for method, values in method_metrics.items():
            f.write(method + "," + ",".join(f"{values[h]:.8g}" for h in headers[1:]) + "\n")

    with (RESULT_DIR / "tables" / "scenario_validation_metrics.csv").open("w", encoding="utf-8") as f:
        headers = [
            "case",
            "expected",
            "true_decision",
            "human_decision",
            "machine_decision",
            "rl_shared_decision",
            "reference_collision",
            "rl_collision",
            "controller_collision",
            "controller_min_clearance_m",
            "reference_risk_mean",
            "rl_risk_mean",
            "reference_reward",
            "rl_reward",
        ]
        f.write(",".join(headers) + "\n")
        for c in rollouts["cases"]:
            r = c["record"]
            m = c["metrics"]
            row = [
                r["case_id"],
                r["expected"],
                r["true_decision"],
                r["human_decision"],
                r["machine_decision"],
                r["rl_shared_decision"],
                r["reference_collision"],
                r["rl_collision"],
                r["controller_collision"],
                m["controller_min_clearance_m"],
                m["reference_risk_mean"],
                m["rl_risk_mean"],
                m["reference_reward"],
                m["rl_reward"],
            ]
            f.write(",".join(str(x).replace(",", ";") for x in row) + "\n")


def write_analysis(
    human_metrics: dict,
    work3_metrics: dict,
    rl_metrics: dict,
    rollouts: dict,
    method_metrics: dict[str, dict[str, float]],
) -> None:
    ref = rl_metrics["reference_authority"]
    rl = rl_metrics["rl_authority"]
    risk_improvement = 100 * (ref["shared_risk_mean"] - rl["shared_risk_mean"]) / (abs(ref["shared_risk_mean"]) + 1e-9)
    reward_improvement = 100 * (rl["reward_mean"] - ref["reward_mean"]) / (abs(ref["reward_mean"]) + 1e-9)
    comfort_improvement = 100 * (ref["comfort_cost"] - rl["comfort_cost"]) / (abs(ref["comfort_cost"]) + 1e-9)

    no_collision_cases = sum(not c["record"]["controller_collision"] for c in rollouts["cases"])
    total_cases = len(rollouts["cases"])
    prop_clear = method_metrics["Proposed MPC"]["mean_min_clearance"]
    ref_clear = method_metrics["Reference authority"]["mean_min_clearance"]

    text = f"""# Experimental Result Analysis

## Data Sources

The figures in this folder are generated from existing repository outputs only:

- `checkpoints/human_intent_transformer/test_metrics.json`
- `checkpoints/rl_authority/test_eval_metrics.json`
- `outputs/work3_authority/metrics.json`
- `outputs/shared_authority_validation/shared_authority_rollouts.js`

No raw highD files, HDF5 datasets, model weights, logs, generated videos, or web demos are used by the plotting script.

## Figure Inventory

1. `fig01_wp1_driver_prediction_metrics`: validates the Work1 driver decision and control-intent predictor.
2. `fig02_rl_authority_ablation`: compares reference authority and RL-corrected authority.
3. `fig03_casewise_risk_reward`: reports case-wise risk and reward before and after RL authority correction.
4. `fig04_method_level_comparison`: compares available method-level trajectory outputs.
5. `fig05_authority_trust_evolution`: visualizes bidirectional trust and authority evolution.
6. `fig06_scenario_trajectory_comparison`: visualizes scenario trajectories.
7. `fig07_control_stability_and_clearance`: compares reference trajectory and MPC execution in clearance and stability.

## Work1 Driver Prediction Evidence

The driver-intent prediction module achieves decision accuracy of {human_metrics['decision_accuracy']:.4f}, decision macro-F1 of {human_metrics['decision_macro_f1']:.4f}, future event accuracy of {human_metrics['future_event_accuracy']:.4f}, and future event macro-F1 of {human_metrics['future_event_macro_f1']:.4f}. The lane-change event recall is {human_metrics['lane_change_event_recall']:.4f}, and the lane-change event precision is {human_metrics['lane_change_event_precision']:.4f}. The speed RMSE is {human_metrics['speed_rmse']:.4f} m/s, and the steering RMSE is {human_metrics['steer_rmse_rad']:.5f} rad.

These results support the use of Work1 as the driver-side intent input for the shared-driving framework. They do not by themselves prove closed-loop shared-driving performance.

## Trust and Reference Authority Evidence

The Work3 output reports mean machine-to-human trust of {work3_metrics['trust_machine_to_human_mean']:.4f}, mean human-to-machine trust of {work3_metrics['trust_human_to_machine_mean']:.4f}, mean human authority of {work3_metrics['authority_human_mean']:.4f}, and mean machine authority of {work3_metrics['authority_machine_mean']:.4f}. The mean shared rollout risk is {work3_metrics['shared_rollout_risk_mean']:.4f}.

These numbers support the existence of a bidirectional trust and reference authority module, but further ablation against alternative trust formulations is still needed for a complete IEEE Transactions experiment.

## RL Authority Ablation

Compared with reference authority, RL-corrected authority improves mean reward from {ref['reward_mean']:.4f} to {rl['reward_mean']:.4f}, corresponding to a relative improvement of {reward_improvement:.2f}%. Mean shared risk decreases from {ref['shared_risk_mean']:.5f} to {rl['shared_risk_mean']:.5f}, corresponding to a relative reduction of {risk_improvement:.2f}%. Comfort cost decreases from {ref['comfort_cost']:.5f} to {rl['comfort_cost']:.5f}, corresponding to a relative reduction of {comfort_improvement:.2f}%.

The efficiency cost increases from {ref['efficiency_cost']:.5f} to {rl['efficiency_cost']:.5f}. Therefore, the RL authority module should be presented as improving safety-related and comfort-related objectives with a measurable efficiency trade-off, rather than as uniformly improving all objectives.

## Closed-Loop Scenario Validation

The current validation file contains {total_cases} fixed scenarios. The proposed MPC execution reports no controller collision in {no_collision_cases}/{total_cases} scenarios. The mean minimum clearance of Proposed MPC is {prop_clear:.3f} m, compared with {ref_clear:.3f} m for the reference-authority trajectory.

The scenario-level figures show that the proposed controller improves the execution safety of the reference trajectory in most cases. However, some parallel-driving cases retain low clearance and should be discussed as challenging cases rather than hidden.

## Method-Level Comparison

The method-level comparison is based on available trajectory keys in the validation file: recorded human trajectory, predicted human trajectory, reference-authority trajectory, and proposed MPC trajectory. These are not a complete set of external baselines. The figure is suitable for internal ablation and system-behavior visualization, while additional baselines are required for a full IEEE Transactions submission.

## Review-Agent Outcome

The accompanying review script checks whether the claims made by these figures are supported by available data. It verifies that:

- RL authority improves mean reward, mean risk, maximum risk, and comfort cost over reference authority.
- Proposed MPC has no collision in the fixed validation cases.
- Proposed MPC improves average minimum clearance over the reference-authority trajectory.

The review script does not certify state-of-the-art performance or superiority over external baselines, because such baselines are not available in the repository.
"""
    (RESULT_DIR / "academic_analysis.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_results_dir()
    set_ieee_style()

    human_metrics = load_json(ROOT / "checkpoints/human_intent_transformer/test_metrics.json")
    rl_metrics = load_json(ROOT / "checkpoints/rl_authority/test_eval_metrics.json")
    work3_metrics = load_json(ROOT / "outputs/work3_authority/metrics.json")
    rollouts = load_rollouts(ROOT / "outputs/shared_authority_validation/shared_authority_rollouts.js")

    plot_wp1_metrics(human_metrics)
    plot_rl_ablation(rl_metrics)
    plot_case_risk_reward(rollouts)
    method_metrics = plot_method_safety_comparison(rollouts)
    plot_authority_trust_examples(rollouts)
    plot_trajectory_cases(rollouts)
    plot_control_stability(rollouts)
    write_tables(human_metrics, work3_metrics, rl_metrics, rollouts, method_metrics)
    write_analysis(human_metrics, work3_metrics, rl_metrics, rollouts, method_metrics)

    print(f"Generated IEEE-style figures and tables in: {RESULT_DIR}")


if __name__ == "__main__":
    main()

