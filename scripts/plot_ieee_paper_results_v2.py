"""Generate IEEE Transactions-style result figures for the shared-driving study.

The script uses only compact result artifacts already present in the repository:

- checkpoints/human_intent_transformer/test_metrics.json
- checkpoints/machine_intent_policy/test_metrics.json
- checkpoints/rl_authority/test_eval_metrics.json
- outputs/work3_authority/metrics.json
- outputs/shared_authority_validation/shared_authority_rollouts.js

It does not read raw highD files, HDF5 datasets, model weights, logs, or videos.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures"
TABLE_DIR = RESULT_DIR / "tables"

COLORS = {
    "human": "#2878B5",
    "machine": "#F28E2B",
    "trust": "#59A14F",
    "authority": "#9467BD",
    "proposed": "#00A6B4",
    "reference": "#4E79A7",
    "risk": "#D62728",
    "obstacle": "#9AA4AF",
    "road": "#F5F7FA",
    "edge": "#2C3440",
    "grid": "#D7DEE8",
}


def setup() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 160,
            "savefig.dpi": 500,
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.55,
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


def save_figure(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def trajectory_array(vehicle: dict) -> np.ndarray:
    return np.asarray(vehicle["xy"], dtype=float)


def vehicle_metrics(cases: list[dict], key: str) -> dict[str, float]:
    clearances, collision, jerk, steer_rate, progress = [], [], [], [], []
    yaw_rate, beta = [], []
    fps = 25.0
    for case in cases:
        veh = case[key]
        xy = trajectory_array(veh)
        clearances.append(float(veh.get("min_clearance_m", np.nan)))
        collision.append(float(bool(veh.get("collision", False))))
        progress.append(float(xy[-1, 0] - xy[0, 0]))
        acc = np.asarray(veh.get("acceleration", []), dtype=float)
        steer = np.asarray(veh.get("steer", []), dtype=float)
        if len(acc) > 1:
            jerk.extend(np.abs(np.diff(acc)) * fps)
        if len(steer) > 1:
            steer_rate.extend(np.abs(np.diff(steer)) * fps)
        if "yaw_rate" in veh:
            yaw_rate.append(float(np.nanmax(np.abs(veh["yaw_rate"]))))
        if "beta" in veh:
            beta.append(float(np.nanmax(np.abs(veh["beta"]))))
    return {
        "collision_rate": float(np.mean(collision)),
        "mean_min_clearance_m": float(np.nanmean(clearances)),
        "worst_clearance_m": float(np.nanmin(clearances)),
        "mean_abs_jerk": float(np.mean(jerk)) if jerk else math.nan,
        "mean_abs_steer_rate": float(np.mean(steer_rate)) if steer_rate else math.nan,
        "mean_progress_m": float(np.mean(progress)),
        "max_abs_yaw_rate": float(np.nanmax(yaw_rate)) if yaw_rate else math.nan,
        "max_abs_sideslip": float(np.nanmax(beta)) if beta else math.nan,
    }


def draw_arrow(ax, x1, y1, x2, y2, color="#4B5563") -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def draw_box(ax, xy, w, h, title, body, fc, ec="#253044") -> None:
    rect = Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=0.9)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h * 0.68, title, ha="center", va="center", weight="bold", fontsize=8.2)
    ax.text(xy[0] + w / 2, xy[1] + h * 0.34, body, ha="center", va="center", fontsize=7.2, linespacing=1.12)


def fig01_framework() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.85))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Cross-Level Human-Machine Shared Driving Framework", pad=8, weight="bold")

    draw_box(ax, (0.35, 5.0), 2.3, 1.1, "Work1", "Driver decision and\ncontrol-intent prediction", "#EAF3FB")
    draw_box(ax, (3.35, 5.0), 2.3, 1.1, "Work2", "Automation-side tactical\nintent generation", "#FFF3E4")
    draw_box(ax, (6.35, 5.0), 2.3, 1.1, "Work3-A", "Bidirectional trust and\nreference authority", "#ECF6EC")
    draw_box(ax, (9.35, 5.0), 2.3, 1.1, "Work3-B", "Transformer RL authority\ncorrection", "#F3ECFA")
    draw_box(ax, (3.25, 2.9), 2.6, 1.05, "Intent Fusion", "Authority-weighted trajectory,\nsteering, and acceleration", "#F7F9FC")
    draw_box(ax, (6.45, 2.9), 2.6, 1.05, "Work4", "Adaptive robust MPC-lite\nwith risk-aware weights", "#E5F8FA")
    draw_box(ax, (4.9, 0.75), 3.3, 1.05, "Traffic Environment", "Controlled ego vehicle,\nsurrounding vehicles, and road state", "#F2F4F7")

    for x1, x2 in [(2.65, 3.35), (5.65, 6.35), (8.65, 9.35)]:
        draw_arrow(ax, x1, 5.55, x2, 5.55)
    draw_arrow(ax, 10.5, 5.0, 8.2, 4.0)
    draw_arrow(ax, 1.5, 5.0, 4.55, 4.0)
    draw_arrow(ax, 4.65, 5.0, 4.65, 4.0)
    draw_arrow(ax, 6.35, 5.35, 5.85, 3.65)
    draw_arrow(ax, 9.35, 5.35, 8.0, 3.95)
    draw_arrow(ax, 5.85, 3.42, 6.45, 3.42)
    draw_arrow(ax, 7.75, 2.9, 6.55, 1.82)
    draw_arrow(ax, 6.55, 1.8, 6.55, 2.86)
    ax.text(10.2, 3.35, "final control\ncommands", ha="left", va="center", fontsize=7.2, color=COLORS["edge"])
    draw_arrow(ax, 9.05, 3.42, 10.1, 3.42)
    ax.text(0.45, 3.65, "traffic state feedback", ha="left", va="center", fontsize=7.2, color=COLORS["edge"])
    draw_arrow(ax, 4.9, 1.3, 0.85, 4.95)
    save_figure(fig, "fig01_overall_framework")


def road_limits(case: dict) -> tuple[float, float, float, float]:
    arrays = [trajectory_array(case[k]) for k in ["ego", "human_pred_ego", "reference_ego", "controller_ego"]]
    for n in case.get("neighbors", []):
        arrays.append(trajectory_array(n))
    xy = np.vstack(arrays)
    return float(xy[:, 0].min() - 18), float(xy[:, 0].max() + 18), float(xy[:, 1].min() - 8), float(xy[:, 1].max() + 8)


def draw_vehicle(ax, x, y, length, width, color, label=None, alpha=0.95, z=5) -> None:
    rect = Rectangle((x - length / 2, y - width / 2), length, width, facecolor=color, edgecolor="white", linewidth=0.45, alpha=alpha, zorder=z)
    ax.add_patch(rect)
    if label:
        ax.text(x, y, label, ha="center", va="center", color="white", fontsize=6.4, zorder=z + 1)


def draw_road(ax, case: dict) -> None:
    xmin, xmax, ymin, ymax = road_limits(case)
    ax.set_facecolor("#2E3742")
    road = case.get("road", {})
    bounds = road.get("lane_markings", []) or road.get("upper_lane_markings", []) or []
    if bounds:
        for i, y in enumerate(bounds):
            style = "-" if i in (0, len(bounds) - 1) else (0, (8, 7))
            lw = 1.4 if i in (0, len(bounds) - 1) else 0.95
            ax.plot([xmin, xmax], [y, y], color="#E7ECF2", linestyle=style, linewidth=lw, alpha=0.9, zorder=1)
    else:
        lane_width = float(case.get("lane_width_m", 3.5) or 3.5)
        for y in np.arange(math.floor(ymin / lane_width) * lane_width, ymax + lane_width, lane_width):
            ax.plot([xmin, xmax], [y, y], color="#E7ECF2", linestyle=(0, (8, 7)), linewidth=0.95, alpha=0.8, zorder=1)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig02_scenario_overview(rollouts: dict) -> None:
    cases = rollouts["cases"][:6]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.0))
    axes = axes.ravel()
    for ax, case in zip(axes, cases):
        draw_road(ax, case)
        for n in case.get("neighbors", []):
            xy = trajectory_array(n)
            ax.plot(xy[:, 0], xy[:, 1], color="#AAB3BD", linewidth=0.8, alpha=0.35, zorder=2)
            draw_vehicle(ax, xy[0, 0], xy[0, 1], n.get("length", 4.6), n.get("width", 1.8), COLORS["obstacle"], None, 0.9, 4)
        paths = [
            ("human", "human_pred_ego", COLORS["human"], 1.35),
            ("reference", "reference_ego", COLORS["reference"], 1.35),
            ("proposed", "controller_ego", COLORS["proposed"], 1.8),
        ]
        for label, key, color, lw in paths:
            xy = trajectory_array(case[key])
            ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=lw, label=label, zorder=6)
        ego0 = trajectory_array(case["ego"])[0]
        draw_vehicle(ax, ego0[0], ego0[1], case["ego"].get("length", 4.6), case["ego"].get("width", 1.8), "#1FBFD1", "ego", 1.0, 8)
        rec = case["record"]
        ax.set_title(
            f"C{rec['case_id']}: H/M/RL={rec['human_decision']}/{rec['machine_decision']}/{rec['rl_shared_decision']}",
            loc="left",
            pad=2,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Scenario-Level Closed-Loop Validation", weight="bold", y=0.995)
    fig.subplots_adjust(bottom=0.08, top=0.93, hspace=0.28, wspace=0.12)
    save_figure(fig, "fig02_scenario_overview")


def fig03_wp1_prediction(human: dict, machine: dict) -> None:
    cls_names = ["Decision\nAcc.", "Decision\nMacro-F1", "Event\nAcc.", "Event\nMacro-F1", "LC\nRecall", "LC\nPrecision"]
    cls_vals = [
        human["decision_accuracy"],
        human["decision_macro_f1"],
        human["future_event_accuracy"],
        human["future_event_macro_f1"],
        human["lane_change_event_recall"],
        human["lane_change_event_precision"],
    ]
    reg_names = ["Speed RMSE\n(m/s)", "Steering RMSE\n(rad)", "Event-Time MAE\n(frames)"]
    event_mae = human.get("event_time_mae_frames", human.get("lane_change_event_time_mae_frames", np.nan))
    reg_vals = [human["speed_rmse"], human["steer_rmse_rad"], event_mae]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [2.4, 1.25, 1.25]})
    axes[0].bar(np.arange(len(cls_vals)), cls_vals, color=COLORS["human"], edgecolor="black", linewidth=0.35)
    axes[0].set_xticks(np.arange(len(cls_vals)))
    axes[0].set_xticklabels(cls_names)
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Driver tactical prediction")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(np.arange(len(reg_vals)), reg_vals, color=["#7BAFD4", "#9BC3E6", "#BCD7EF"], edgecolor="black", linewidth=0.35)
    axes[1].set_xticks(np.arange(len(reg_vals)))
    axes[1].set_xticklabels(reg_names)
    axes[1].set_title("Control-intent error")
    axes[1].grid(axis="y", alpha=0.25)
    if machine:
        labels = ["Machine\nDecision", "Machine\nEvent"]
        vals = [machine.get("decision_accuracy", np.nan), machine.get("future_event_accuracy", np.nan)]
        axes[2].bar(np.arange(2), vals, color=COLORS["machine"], edgecolor="black", linewidth=0.35)
        axes[2].set_xticks(np.arange(2))
        axes[2].set_xticklabels(labels)
        axes[2].set_ylim(0, 1.02)
        axes[2].set_title("Automation intent")
        axes[2].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Work1/Work2 Intent Prediction Evidence", weight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig03_intent_prediction_evidence")


def fig04_decision_conflict(rollouts: dict) -> None:
    mapping = {"L": -1, "S": 0, "R": 1}
    cases = rollouts["cases"]
    ids = [f"C{c['record']['case_id']}" for c in cases]
    mat = np.array(
        [
            [mapping[c["record"]["human_decision"]] for c in cases],
            [mapping[c["record"]["machine_decision"]] for c in cases],
            [mapping[c["record"]["rl_shared_decision"]] for c in cases],
        ],
        dtype=float,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.35, 1.0]})
    im = axes[0].imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["Driver", "Automation", "Shared"])
    axes[0].set_xticks(np.arange(len(ids)))
    axes[0].set_xticklabels(ids)
    axes[0].set_title("Tactical decision conflict")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            label = {-1: "L", 0: "S", 1: "R"}[int(mat[i, j])]
            axes[0].text(j, i, label, ha="center", va="center", weight="bold", color="white")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels(["L", "S", "R"])
    risk_ref = [c["metrics"]["reference_risk_mean"] for c in cases]
    risk_rl = [c["metrics"]["rl_risk_mean"] for c in cases]
    x = np.arange(len(cases))
    axes[1].plot(x, risk_ref, "-o", label="Reference", color=COLORS["reference"], markersize=3)
    axes[1].plot(x, risk_rl, "-s", label="RL shared", color=COLORS["proposed"], markersize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ids)
    axes[1].set_ylabel("Mean shared risk")
    axes[1].set_title("Conflict-resolution risk")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "fig04_decision_conflict_resolution")


def fig05_trust_authority(rollouts: dict) -> None:
    cases = rollouts["cases"][:6]
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.0), sharex=False)
    axes = axes.ravel()
    for ax, case in zip(axes, cases):
        t = np.arange(len(case["signals"]["authority_ref"])) / 25.0
        ax.plot(t, case["signals"]["trust_machine_to_human"], color=COLORS["human"], label="M-to-H trust")
        ax.plot(t, case["signals"]["trust_human_to_machine"], color=COLORS["machine"], label="H-to-M trust")
        ax.plot(t, case["signals"]["authority_ref"], color=COLORS["reference"], linestyle="--", label="Reference authority")
        ax.plot(t, case["signals"]["authority_rl"], color=COLORS["proposed"], label="RL authority")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"Case {case['record']['case_id']}", loc="left", pad=2)
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-2].set_xlabel("Time (s)")
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_ylabel("Value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Bidirectional Trust and Authority Evolution", weight="bold", y=0.995)
    fig.subplots_adjust(bottom=0.10, top=0.94, hspace=0.42, wspace=0.25)
    save_figure(fig, "fig05_trust_authority_evolution")


def fig06_ablation(rl_metrics: dict, rollouts: dict) -> None:
    ref = rl_metrics["reference_authority"]
    rl = rl_metrics["rl_authority"]
    labels = ["Reference\nauthority", "RL-corrected\nauthority"]
    metrics = [
        ("Mean reward", [ref["reward_mean"], rl["reward_mean"]], True),
        ("Mean risk", [ref["shared_risk_mean"], rl["shared_risk_mean"]], False),
        ("Worst risk", [ref["shared_risk_max"], rl["shared_risk_max"]], False),
        ("Comfort cost", [ref["comfort_cost"], rl["comfort_cost"]], False),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = np.arange(len(metrics))
    ref_vals = [m[1][0] for m in metrics]
    rl_vals = [m[1][1] for m in metrics]
    width = 0.34
    axes[0].bar(x - width / 2, ref_vals, width, label="Reference prior", color=COLORS["reference"], edgecolor="black", linewidth=0.35)
    axes[0].bar(x + width / 2, rl_vals, width, label="RL correction", color=COLORS["proposed"], edgecolor="black", linewidth=0.35)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m[0] for m in metrics], rotation=18)
    axes[0].set_title("Authority correction ablation")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    cases = rollouts["cases"]
    cids = [f"C{c['record']['case_id']}" for c in cases]
    clearance_ref = [c["reference_ego"]["min_clearance_m"] for c in cases]
    clearance_prop = [c["controller_ego"]["min_clearance_m"] for c in cases]
    x2 = np.arange(len(cases))
    axes[1].bar(x2 - width / 2, clearance_ref, width, label="Before MPC", color=COLORS["reference"], edgecolor="black", linewidth=0.35)
    axes[1].bar(x2 + width / 2, clearance_prop, width, label="After MPC-lite", color=COLORS["proposed"], edgecolor="black", linewidth=0.35)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(cids)
    axes[1].set_ylabel("Minimum clearance (m)")
    axes[1].set_title("Control execution ablation")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Supported Ablation Results", weight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "fig06_ablation_study")


def fig07_method_comparison(methods: dict[str, dict[str, float]]) -> None:
    names = ["Reference\nauthority", "Proposed\nMPC-lite"]
    keys = ["Reference authority", "Proposed MPC-lite"]
    clearance = [methods[k]["mean_min_clearance_m"] for k in keys]
    worst_clearance = [methods[k]["worst_clearance_m"] for k in keys]
    progress = [methods[k]["mean_progress_m"] for k in keys]
    jerk = [methods[k]["mean_abs_jerk"] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    x = np.arange(len(names))
    bar_colors = [COLORS["reference"], COLORS["proposed"]]
    axes[0].bar(x, clearance, color=bar_colors, edgecolor="black", linewidth=0.35)
    axes[0].set_ylabel("Mean minimum clearance (m)")
    axes[0].set_title("Safety margin")
    axes[1].bar(x, worst_clearance, color=bar_colors, edgecolor="black", linewidth=0.35)
    axes[1].set_ylabel("Worst clearance (m)")
    axes[1].set_title("Worst-case safety")
    axes[2].scatter(progress[0], jerk[0], color=COLORS["reference"], s=48, label="Reference authority")
    axes[2].scatter(progress[1], jerk[1], color=COLORS["proposed"], s=58, label="Proposed MPC-lite")
    axes[2].set_xlabel("Mean progress (m)")
    axes[2].set_ylabel("Mean absolute jerk")
    axes[2].set_title("Efficiency-comfort trade-off")
    axes[2].legend(frameon=False, loc="best")
    for ax in axes[:2]:
        ax.set_xticks(x)
        ax.set_xticklabels(names)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Supported Method Comparison", weight="bold", y=1.03)
    fig.tight_layout()
    save_figure(fig, "fig07_method_comparison")


def fig08_safety_efficiency(rollouts: dict) -> None:
    cases = rollouts["cases"]
    ids = [f"C{c['record']['case_id']}" for c in cases]
    x = np.arange(len(cases))
    ref_reward = [c["metrics"]["reference_reward"] for c in cases]
    rl_reward = [c["metrics"]["rl_reward"] for c in cases]
    ref_risk = [c["metrics"]["reference_risk_mean"] for c in cases]
    rl_risk = [c["metrics"]["rl_risk_mean"] for c in cases]
    clearance = [c["controller_ego"]["min_clearance_m"] for c in cases]
    progress = [trajectory_array(c["controller_ego"])[-1, 0] - trajectory_array(c["controller_ego"])[0, 0] for c in cases]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    width = 0.34
    axes[0, 0].bar(x - width / 2, ref_risk, width, label="Reference", color=COLORS["reference"], edgecolor="black", linewidth=0.35)
    axes[0, 0].bar(x + width / 2, rl_risk, width, label="RL shared", color=COLORS["proposed"], edgecolor="black", linewidth=0.35)
    axes[0, 0].set_ylabel("Mean risk")
    axes[0, 0].set_title("Risk reduction")
    axes[0, 1].bar(x - width / 2, ref_reward, width, label="Reference", color=COLORS["reference"], edgecolor="black", linewidth=0.35)
    axes[0, 1].bar(x + width / 2, rl_reward, width, label="RL shared", color=COLORS["proposed"], edgecolor="black", linewidth=0.35)
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].set_title("Task reward")
    axes[1, 0].bar(x, clearance, color=COLORS["proposed"], edgecolor="black", linewidth=0.35)
    axes[1, 0].set_ylabel("Minimum clearance (m)")
    axes[1, 0].set_title("Closed-loop safety margin")
    axes[1, 1].bar(x, progress, color=COLORS["machine"], edgecolor="black", linewidth=0.35)
    axes[1, 1].set_ylabel("Progress (m)")
    axes[1, 1].set_title("Closed-loop efficiency")
    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(ids)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    fig.suptitle("Safety and Efficiency Evaluation", weight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "fig08_safety_efficiency_metrics")


def fig09_mpc_control(rollouts: dict) -> None:
    cases = rollouts["cases"]
    selected = [cases[1], cases[4], cases[5], cases[6]] if len(cases) >= 7 else cases[:4]
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 6.6), sharex=False)
    for row, case in enumerate(selected):
        t = np.arange(len(case["controller_ego"]["speed"])) / 25.0
        ctrl = case["controller_ego"]
        ref = case["reference_ego"]
        axes[row, 0].plot(t, ctrl["speed"], color=COLORS["proposed"], label="MPC-lite")
        axes[row, 0].plot(t, ref["speed"], color=COLORS["reference"], linestyle="--", label="Reference")
        axes[row, 0].set_ylabel(f"C{case['record']['case_id']}\nSpeed")
        axes[row, 1].plot(t, ctrl["acceleration"], color=COLORS["proposed"], label="Acceleration")
        axes[row, 1].plot(t, ctrl["steer"], color=COLORS["machine"], label="Steering")
        axes[row, 2].plot(t, ctrl.get("yaw_rate", [0] * len(t)), color=COLORS["risk"], label="Yaw rate")
        axes[row, 2].plot(t, ctrl.get("beta", [0] * len(t)), color=COLORS["authority"], label="Sideslip")
        for col in range(3):
            axes[row, col].grid(alpha=0.22)
            axes[row, col].spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title("Speed tracking")
    axes[0, 1].set_title("Control command")
    axes[0, 2].set_title("Stability response")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    axes[0, 2].legend(frameon=False)
    fig.suptitle("Authority-Aware MPC-Lite Control Execution", weight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, "fig09_mpc_lite_control_performance")


def write_tables_and_analysis(human: dict, machine: dict, rl: dict, work3: dict, rollouts: dict, method_metrics: dict) -> None:
    wp1_rows = [
        {"metric": k, "value": v}
        for k, v in human.items()
        if isinstance(v, (int, float))
    ]
    write_csv(TABLE_DIR / "wp1_driver_prediction_metrics.csv", wp1_rows)
    method_rows = []
    for method, vals in method_metrics.items():
        row = {"method": method}
        row.update(vals)
        method_rows.append(row)
    write_csv(TABLE_DIR / "method_comparison_metrics.csv", method_rows)
    case_rows = []
    for c in rollouts["cases"]:
        r = c["record"]
        m = c["metrics"]
        case_rows.append(
            {
                "case": r["case_id"],
                "expected": r["expected"],
                "human_decision": r["human_decision"],
                "machine_decision": r["machine_decision"],
                "shared_decision": r["rl_shared_decision"],
                "controller_collision": r["controller_collision"],
                "controller_min_clearance_m": c["controller_ego"]["min_clearance_m"],
                "reference_mean_risk": m["reference_risk_mean"],
                "rl_mean_risk": m["rl_risk_mean"],
                "reference_reward": m["reference_reward"],
                "rl_reward": m["rl_reward"],
                "authority_ref_mean": m["authority_ref_mean"],
                "authority_rl_mean": m["authority_rl_mean"],
            }
        )
    write_csv(TABLE_DIR / "scenario_level_metrics.csv", case_rows)

    ref = rl["reference_authority"]
    rl_auth = rl["rl_authority"]
    prop = method_metrics["Proposed MPC-lite"]
    ref_traj = method_metrics["Reference authority"]
    collision_free = sum(not c["record"]["controller_collision"] for c in rollouts["cases"])
    analysis = f"""# IEEE-Style Experimental Result Analysis

## Data Sources

The results are generated from compact repository outputs only. Raw highD files, HDF5 datasets, checkpoints, logs, and videos are not read by the plotting script.

## Generated Figures

- `fig01_overall_framework`: cross-level shared-driving framework.
- `fig02_scenario_overview`: closed-loop traffic scenario visualization.
- `fig03_intent_prediction_evidence`: Work1 driver-intent and Work2 automation-intent evidence.
- `fig04_decision_conflict_resolution`: tactical conflict and risk resolution.
- `fig05_trust_authority_evolution`: bidirectional trust and authority evolution.
- `fig06_ablation_study`: supported authority and control-execution ablations.
- `fig07_method_comparison`: method-level multi-objective comparison.
- `fig08_safety_efficiency_metrics`: scenario-level safety and efficiency metrics.
- `fig09_mpc_lite_control_performance`: MPC-lite speed, command, and stability response.

## Work1 Driver Prediction

The driver-side predictor reports decision accuracy of {human['decision_accuracy']:.4f}, decision macro-F1 of {human['decision_macro_f1']:.4f}, future-event accuracy of {human['future_event_accuracy']:.4f}, and future-event macro-F1 of {human['future_event_macro_f1']:.4f}. Lane-change event recall and precision are {human['lane_change_event_recall']:.4f} and {human['lane_change_event_precision']:.4f}, respectively. The lane-change event-time MAE is {human.get('lane_change_event_time_mae_frames', float('nan')):.3f} frames. The speed RMSE is {human['speed_rmse']:.4f} m/s and the steering RMSE is {human['steer_rmse_rad']:.5f} rad.

## Trust, Authority, and RL Correction

The trust module reports mean machine-to-human trust of {work3['trust_machine_to_human_mean']:.4f} and mean human-to-machine trust of {work3['trust_human_to_machine_mean']:.4f}. The RL authority correction improves the mean reward from {ref['reward_mean']:.4f} to {rl_auth['reward_mean']:.4f}, reduces mean shared risk from {ref['shared_risk_mean']:.4f} to {rl_auth['shared_risk_mean']:.4f}, and reduces worst-case shared risk from {ref['shared_risk_max']:.4f} to {rl_auth['shared_risk_max']:.4f}.

## Closed-Loop Control Validation

The validation set contains {len(rollouts['cases'])} fixed scenarios. The proposed MPC-lite execution reports no collision in {collision_free}/{len(rollouts['cases'])} scenarios. Its mean minimum clearance is {prop['mean_min_clearance_m']:.3f} m, compared with {ref_traj['mean_min_clearance_m']:.3f} m for the reference-authority trajectory.

## Interpretation

The available evidence supports an ablation-oriented conclusion: the proposed RL-corrected authority and adaptive MPC-lite execution improve the reference-authority pipeline on the main safety-oriented metrics. The figures should not be used to claim superiority over external baselines because such baselines are not available in the current repository outputs.

## Limitations

- Some scenario-level indicators are not uniformly improved in every case.
- The efficiency cost of RL authority is slightly higher than that of the reference prior in the available test file.
- Additional external baselines and driver-in-the-loop experiments are still required for a complete IEEE Transactions submission.
"""
    (RESULT_DIR / "academic_analysis.md").write_text(analysis, encoding="utf-8")


def make_review_report(rl: dict, rollouts: dict, method_metrics: dict) -> None:
    checks: list[tuple[str, bool]] = []
    ref = rl["reference_authority"]
    rl_auth = rl["rl_authority"]
    checks.append(("RL authority improves mean reward over the reference authority prior.", rl_auth["reward_mean"] > ref["reward_mean"]))
    checks.append(("RL authority reduces mean shared risk.", rl_auth["shared_risk_mean"] < ref["shared_risk_mean"]))
    checks.append(("RL authority reduces worst-case shared risk.", rl_auth["shared_risk_max"] < ref["shared_risk_max"]))
    checks.append(("RL authority reduces comfort cost.", rl_auth["comfort_cost"] < ref["comfort_cost"]))
    checks.append(("The proposed controller has no reported collision in all validation cases.", not any(c["record"]["controller_collision"] for c in rollouts["cases"])))
    checks.append(
        (
            "The proposed MPC-lite trajectory improves average minimum clearance over the reference trajectory.",
            method_metrics["Proposed MPC-lite"]["mean_min_clearance_m"] > method_metrics["Reference authority"]["mean_min_clearance_m"],
        )
    )
    figure_files = sorted(FIG_DIR.glob("fig*.png"))
    checks.append(("All nine required figure groups are generated.", len(figure_files) >= 9))
    status = "PASS" if all(v for _, v in checks) else "FAIL"
    cautions = [
        "The review scope is limited to repository-available outputs.",
        "Do not claim superiority over external baselines until additional experiments are added.",
        "Efficiency trade-offs should be discussed explicitly.",
    ]
    lines = ["# Review Agent Report", "", f"Overall decision: **{status}**", "", "## Checks", ""]
    for msg, ok in checks:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] {msg}")
    lines.extend(["", "## Cautions", ""])
    lines.extend([f"- {c}" for c in cautions])
    (RESULT_DIR / "review_agent_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULT_DIR / "review_agent_report.json").write_text(
        json.dumps({"status": status, "checks": [{"message": m, "passed": v} for m, v in checks], "cautions": cautions}, indent=2),
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError("Review agent failed. See results/review_agent_report.md")


def main() -> None:
    setup()
    human = load_json(ROOT / "checkpoints" / "human_intent_transformer" / "test_metrics.json")
    machine = load_json(ROOT / "checkpoints" / "machine_intent_policy" / "test_metrics.json")
    rl = load_json(ROOT / "checkpoints" / "rl_authority" / "test_eval_metrics.json")
    work3 = load_json(ROOT / "outputs" / "work3_authority" / "metrics.json")
    rollouts = load_rollouts(ROOT / "outputs" / "shared_authority_validation" / "shared_authority_rollouts.js")

    method_metrics = {
        "Human record": vehicle_metrics(rollouts["cases"], "ego"),
        "Human prediction": vehicle_metrics(rollouts["cases"], "human_pred_ego"),
        "Reference authority": vehicle_metrics(rollouts["cases"], "reference_ego"),
        "Proposed MPC-lite": vehicle_metrics(rollouts["cases"], "controller_ego"),
    }

    fig01_framework()
    fig02_scenario_overview(rollouts)
    fig03_wp1_prediction(human, machine)
    fig04_decision_conflict(rollouts)
    fig05_trust_authority(rollouts)
    fig06_ablation(rl, rollouts)
    fig07_method_comparison(method_metrics)
    fig08_safety_efficiency(rollouts)
    fig09_mpc_control(rollouts)
    write_tables_and_analysis(human, machine, rl, work3, rollouts, method_metrics)
    make_review_report(rl, rollouts, method_metrics)

    readme = """# Figure Generation Workflow

Run:

```powershell
python scripts/plot_ieee_paper_results_v2.py
```

Outputs:

- `results/figures`: PNG, PDF, and SVG figures.
- `results/tables`: CSV tables used by the figures.
- `results/academic_analysis.md`: paper-oriented result analysis.
- `results/review_agent_report.md`: automatic evidence check.

All labels are in English and are intended for IEEE Transactions-style manuscript preparation.
"""
    (RESULT_DIR / "figure_generation_readme.md").write_text(readme, encoding="utf-8")
    print(f"Generated IEEE-style result package in {RESULT_DIR}")


if __name__ == "__main__":
    main()
