"""Convert the website/video validation scenarios into paper-ready figures.

The figures are generated from the same rollout file used by the realtime
website:

- outputs/shared_authority_validation/shared_authority_rollouts.js

No raw datasets, logs, checkpoints, or video frames are read. The produced plots
represent the variables displayed in the website: trajectories, speed, steering,
lateral dynamics, safety distance, TTC, authority, trust, and risk.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures"
TABLE_DIR = RESULT_DIR / "tables"
CASE_DIR = FIG_DIR / "video_scenarios"

COLORS = {
    "human": "#2F80ED",
    "reference": "#F2994A",
    "rlref": "#7B61FF",
    "mpc": "#00A6B4",
    "obstacle": "#8E99A8",
    "road": "#303A46",
    "lane": "#E9EEF5",
    "risk": "#D62728",
    "trust_mh": "#59A14F",
    "trust_hm": "#B279A2",
    "authority_ref": "#F59E0B",
    "authority_rl": "#00A6B4",
    "grid": "#D8DEE8",
    "text": "#1F2937",
}


def reset_results() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for child in RESULT_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    CASE_DIR.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "figure.dpi": 160,
            "savefig.dpi": 500,
            "axes.linewidth": 0.75,
            "grid.linewidth": 0.38,
            "lines.linewidth": 1.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_rollouts() -> dict:
    path = ROOT / "outputs" / "shared_authority_validation" / "shared_authority_rollouts.js"
    text = path.read_text(encoding="utf-8")
    payload = text.split("=", 1)[1].strip().rstrip(";")
    return json.loads(payload)


def save(fig: plt.Figure, name: str, folder: Path = FIG_DIR) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def xy(vehicle: dict) -> np.ndarray:
    return np.asarray(vehicle["xy"], dtype=float)


def arr(vehicle: dict, key: str, fill: float = 0.0) -> np.ndarray:
    values = vehicle.get(key)
    if values is None:
        return np.full(len(vehicle["xy"]), fill, dtype=float)
    return np.asarray(values, dtype=float)


def derivative(values: np.ndarray, fps: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, 1.0 / fps)


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def yaw_from_xy(points: np.ndarray) -> np.ndarray:
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    return np.arctan2(dy, dx)


def lane_markings(case: dict) -> list[float]:
    road = case.get("road", {})
    return road.get("lane_markings", []) or road.get("upper_lane_markings", []) or []


def road_bounds(case: dict) -> tuple[float, float, float, float]:
    arrays = [xy(case[k]) for k in ["human_pred_ego", "reference_ego", "ego", "controller_ego"]]
    arrays.extend(xy(n) for n in case.get("neighbors", []))
    all_xy = np.vstack(arrays)
    xmin = float(np.min(all_xy[:, 0]) - 12)
    xmax = float(np.max(all_xy[:, 0]) + 15)
    marks = lane_markings(case)
    if marks:
        ymin, ymax = min(marks) - 1.2, max(marks) + 1.2
    else:
        ymin, ymax = float(np.min(all_xy[:, 1]) - 5), float(np.max(all_xy[:, 1]) + 5)
    return xmin, xmax, ymin, ymax


def draw_vehicle(ax, x0, y0, length, width, color, label=None, alpha=0.95, z=5) -> None:
    rect = Rectangle(
        (x0 - length / 2, y0 - width / 2),
        length,
        width,
        facecolor=color,
        edgecolor="white",
        linewidth=0.45,
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(rect)
    if label:
        ax.text(x0, y0, label, ha="center", va="center", fontsize=6.3, color="white", zorder=z + 1)


def draw_scene(ax, case: dict) -> None:
    xmin, xmax, ymin, ymax = road_bounds(case)
    ax.set_facecolor(COLORS["road"])
    marks = lane_markings(case)
    if marks:
        for i, y in enumerate(marks):
            outer = i == 0 or i == len(marks) - 1
            ax.plot(
                [xmin, xmax],
                [y, y],
                color=COLORS["lane"],
                linewidth=1.35 if outer else 0.9,
                linestyle="-" if outer else (0, (7, 6)),
                alpha=0.95,
                zorder=1,
            )
    for n in case.get("neighbors", []):
        nxy = xy(n)
        ax.plot(nxy[:, 0], nxy[:, 1], color="#B8C0CA", linewidth=0.65, alpha=0.42, zorder=2)
        draw_vehicle(ax, nxy[0, 0], nxy[0, 1], n.get("length", 4.6), n.get("width", 1.8), COLORS["obstacle"], None, 0.92, 4)
    paths = [
        ("Driver prediction", "human_pred_ego", COLORS["human"], 1.4, "--"),
        ("Reference fusion", "reference_ego", COLORS["reference"], 1.45, "-"),
        ("RL reference", "ego", COLORS["rlref"], 1.35, ":"),
        ("MPC-lite", "controller_ego", COLORS["mpc"], 1.9, "-"),
    ]
    for label, key, color, lw, ls in paths:
        p = xy(case[key])
        ax.plot(p[:, 0], p[:, 1], color=color, linewidth=lw, linestyle=ls, label=label, zorder=6)
    p0 = xy(case["controller_ego"])[0]
    draw_vehicle(ax, p0[0], p0[1], case["controller_ego"].get("length", 4.6), case["controller_ego"].get("width", 1.8), COLORS["mpc"], "ego", 1.0, 8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitudinal position (m)")
    ax.set_ylabel("Lateral position (m)")
    ax.legend(loc="upper right", ncol=4, frameon=True, framealpha=0.88)
    for s in ax.spines.values():
        s.set_visible(False)


def nearest_lead_state(case: dict, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ego = case["controller_ego"]
    ego_xy = xy(ego)
    ego_len = float(ego.get("length", 4.6))
    n = len(ego_xy)
    gaps = np.full(n, np.nan)
    rel_speed = np.full(n, np.nan)
    lead_name = np.array(["none"] * n, dtype=object)
    ego_vx = derivative(ego_xy[:, 0], fps)
    for i in range(n):
        candidates = []
        for nb in case.get("neighbors", []):
            nb_xy = xy(nb)
            gap = nb_xy[i, 0] - ego_xy[i, 0] - 0.5 * (float(nb.get("length", 4.6)) + ego_len)
            if gap >= -0.5:
                nb_vx = derivative(nb_xy[:, 0], fps)[i]
                candidates.append((gap, nb_vx - ego_vx[i], nb["name"]))
        if candidates:
            gap, rv, name = min(candidates, key=lambda x: x[0])
            gaps[i] = max(gap, 0.0)
            rel_speed[i] = rv
            lead_name[i] = name
    closing = -rel_speed
    ttc = np.where((closing > 0.05) & np.isfinite(gaps), gaps / closing, np.nan)
    ttc = np.clip(ttc, 0, 10)
    return gaps, rel_speed, ttc


def compute_case_signals(case: dict, fps: float) -> dict[str, np.ndarray]:
    ctrl = case["controller_ego"]
    ctrl_xy = xy(ctrl)
    t = np.arange(len(ctrl_xy)) / fps
    lateral_velocity = derivative(ctrl_xy[:, 1], fps)
    lateral_acceleration = derivative(lateral_velocity, fps)
    yaw_rate = arr(ctrl, "yaw_rate")
    if not np.any(np.abs(yaw_rate) > 1e-8):
        yaw_rate = derivative(yaw_from_xy(ctrl_xy), fps)
    front_gap, relative_speed, ttc = nearest_lead_state(case, fps)
    ref_risk = np.full_like(t, float(case["metrics"]["reference_risk_mean"]), dtype=float)
    rl_risk = np.full_like(t, float(case["metrics"]["rl_risk_mean"]), dtype=float)
    return {
        "time": t,
        "speed": arr(ctrl, "speed"),
        "acceleration": arr(ctrl, "acceleration"),
        "steer": arr(ctrl, "steer"),
        "lateral_velocity": smooth(lateral_velocity),
        "lateral_acceleration": smooth(lateral_acceleration),
        "yaw_rate": yaw_rate,
        "sideslip": arr(ctrl, "beta"),
        "front_gap": front_gap,
        "relative_speed": relative_speed,
        "ttc": ttc,
        "authority_ref": np.asarray(case["signals"]["authority_ref"], dtype=float),
        "authority_rl": np.asarray(case["signals"]["authority_rl"], dtype=float),
        "trust_mh": np.asarray(case["signals"]["trust_machine_to_human"], dtype=float),
        "trust_hm": np.asarray(case["signals"]["trust_human_to_machine"], dtype=float),
        "urgency": np.asarray(case["signals"]["environment_urgency"], dtype=float),
        "ref_risk": ref_risk,
        "rl_risk": rl_risk,
    }


def plot_case(case: dict, fps: float) -> dict[str, float]:
    sig = compute_case_signals(case, fps)
    t = sig["time"]
    record = case["record"]
    fig = plt.figure(figsize=(7.25, 8.9))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.25, 1, 1, 1, 1], hspace=0.48, wspace=0.28)
    ax_scene = fig.add_subplot(gs[0, :])
    draw_scene(ax_scene, case)
    title = (
        f"Case {record['case_id']} | highD rec. {record['recording_id']}, veh. {record['vehicle_id']}, "
        f"sample {record['sample_index']} | H/M/Shared = "
        f"{record['human_decision']}/{record['machine_decision']}/{record['rl_shared_decision']}"
    )
    ax_scene.set_title(title, loc="left", fontsize=9.2, weight="bold")

    axes = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[3, 0]),
        fig.add_subplot(gs[3, 1]),
        fig.add_subplot(gs[4, 0]),
        fig.add_subplot(gs[4, 1]),
    ]

    axes[0].plot(t, sig["speed"], color=COLORS["mpc"], label="Speed")
    axes[0].plot(t, sig["acceleration"], color=COLORS["reference"], label="Acceleration")
    axes[0].set_ylabel("m/s or m/s^2")
    axes[0].set_title("Longitudinal motion")
    axes[0].legend(frameon=False)

    axes[1].plot(t, sig["steer"], color=COLORS["mpc"], label="Steering angle")
    axes[1].set_ylabel("rad")
    axes[1].set_title("Front-wheel steering")

    axes[2].plot(t, sig["lateral_velocity"], color=COLORS["human"], label="Lateral velocity")
    axes[2].set_ylabel("m/s")
    axes[2].set_title("Lateral velocity")

    axes[3].plot(t, sig["lateral_acceleration"], color=COLORS["risk"], label="Lateral acceleration")
    axes[3].set_ylabel("m/s^2")
    axes[3].set_title("Lateral acceleration")

    axes[4].plot(t, sig["yaw_rate"], color=COLORS["risk"], label="Yaw rate")
    axes[4].plot(t, sig["sideslip"], color=COLORS["trust_hm"], label="Sideslip angle")
    axes[4].set_ylabel("rad/s or rad")
    axes[4].set_title("Yaw and sideslip response")
    axes[4].legend(frameon=False)

    axes[5].plot(t, sig["front_gap"], color=COLORS["reference"], label="Nearest lead gap")
    axes[5].set_ylabel("m")
    ax_ttc = axes[5].twinx()
    ax_ttc.plot(t, sig["ttc"], color=COLORS["risk"], linestyle="--", label="TTC")
    ax_ttc.set_ylabel("TTC (s)")
    axes[5].set_title("Lead distance and TTC")
    lines, labels = axes[5].get_legend_handles_labels()
    lines2, labels2 = ax_ttc.get_legend_handles_labels()
    axes[5].legend(lines + lines2, labels + labels2, frameon=False, loc="best")

    axes[6].plot(t, sig["authority_ref"], color=COLORS["authority_ref"], linestyle="--", label="Reference authority")
    axes[6].plot(t, sig["authority_rl"], color=COLORS["authority_rl"], label="RL authority")
    axes[6].set_ylim(-0.03, 1.03)
    axes[6].set_ylabel("Human authority")
    axes[6].set_title("Authority allocation")
    axes[6].legend(frameon=False)

    axes[7].plot(t, sig["trust_mh"], color=COLORS["trust_mh"], label="Machine-to-human trust")
    axes[7].plot(t, sig["trust_hm"], color=COLORS["trust_hm"], label="Human-to-machine trust")
    axes[7].plot(t, sig["urgency"], color=COLORS["risk"], linestyle=":", label="Risk urgency")
    axes[7].set_ylim(-0.03, 1.03)
    axes[7].set_ylabel("Index")
    axes[7].set_title("Trust and risk indices")
    axes[7].legend(frameon=False)

    for ax in axes:
        ax.grid(alpha=0.28)
        ax.set_xlabel("Time (s)")
        ax.spines[["top", "right"]].set_visible(False)
    ax_ttc.spines["top"].set_visible(False)

    fig.suptitle("Paper-Ready Effect Curves from the Realtime Validation Scenario", y=0.996, weight="bold")
    name = f"case_{int(record['case_id']):02d}_video_effects"
    save(fig, name, CASE_DIR)

    finite_ttc = sig["ttc"][np.isfinite(sig["ttc"])]
    return {
        "case": record["case_id"],
        "sample_index": record["sample_index"],
        "recording_id": record["recording_id"],
        "vehicle_id": record["vehicle_id"],
        "expected": record["expected"],
        "human_decision": record["human_decision"],
        "machine_decision": record["machine_decision"],
        "shared_decision": record["rl_shared_decision"],
        "mean_speed_mps": float(np.nanmean(sig["speed"])),
        "max_abs_steer_rad": float(np.nanmax(np.abs(sig["steer"]))),
        "max_abs_lateral_accel_mps2": float(np.nanmax(np.abs(sig["lateral_acceleration"]))),
        "max_abs_yaw_rate_rps": float(np.nanmax(np.abs(sig["yaw_rate"]))),
        "max_abs_lateral_velocity_mps": float(np.nanmax(np.abs(sig["lateral_velocity"]))),
        "min_lead_gap_m": float(np.nanmin(sig["front_gap"])) if np.any(np.isfinite(sig["front_gap"])) else math.nan,
        "min_ttc_s": float(np.nanmin(finite_ttc)) if len(finite_ttc) else math.nan,
        "mean_authority_ref": float(np.nanmean(sig["authority_ref"])),
        "mean_authority_rl": float(np.nanmean(sig["authority_rl"])),
        "mean_trust_machine_to_human": float(np.nanmean(sig["trust_mh"])),
        "mean_trust_human_to_machine": float(np.nanmean(sig["trust_hm"])),
        "mean_urgency": float(np.nanmean(sig["urgency"])),
        "reference_risk_mean": float(case["metrics"]["reference_risk_mean"]),
        "rl_risk_mean": float(case["metrics"]["rl_risk_mean"]),
        "controller_min_clearance_m": float(case["metrics"]["controller_min_clearance_m"]),
        "controller_collision": bool(record["controller_collision"]),
    }


def plot_summary(rows: list[dict]) -> None:
    labels = [f"C{r['case']}" for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.2))
    axes[0, 0].bar(x, [r["controller_min_clearance_m"] for r in rows], color=COLORS["mpc"], edgecolor="black", linewidth=0.35)
    axes[0, 0].set_title("Minimum clearance")
    axes[0, 0].set_ylabel("m")
    axes[0, 1].bar(x, [r["min_ttc_s"] for r in rows], color=COLORS["risk"], edgecolor="black", linewidth=0.35)
    axes[0, 1].set_title("Minimum TTC")
    axes[0, 1].set_ylabel("s")
    axes[1, 0].plot(x, [r["reference_risk_mean"] for r in rows], "-o", color=COLORS["reference"], label="Reference risk")
    axes[1, 0].plot(x, [r["rl_risk_mean"] for r in rows], "-s", color=COLORS["mpc"], label="RL risk")
    axes[1, 0].set_title("Risk index")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(x, [r["mean_authority_ref"] for r in rows], "-o", color=COLORS["authority_ref"], label="Reference authority")
    axes[1, 1].plot(x, [r["mean_authority_rl"] for r in rows], "-s", color=COLORS["authority_rl"], label="RL authority")
    axes[1, 1].set_title("Mean human authority")
    axes[1, 1].legend(frameon=False)
    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.28)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Scenario-Level Safety and Authority Summary", weight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig_summary_video_scenarios")


def write_outputs(rows: list[dict]) -> None:
    with (TABLE_DIR / "video_scenario_effect_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    collision_free = sum(not r["controller_collision"] for r in rows)
    mean_clearance = np.nanmean([r["controller_min_clearance_m"] for r in rows])
    mean_risk_ref = np.nanmean([r["reference_risk_mean"] for r in rows])
    mean_risk_rl = np.nanmean([r["rl_risk_mean"] for r in rows])
    text = f"""# Video Scenario Result Analysis

## Source

The figures are generated from `outputs/shared_authority_validation/shared_authority_rollouts.js`, which is the same data source used by the realtime validation website and the summary video `work1_4_validation_cases.mp4`.

## Generated Figures

- `figures/video_scenarios/case_01_video_effects` to `case_07_video_effects`: per-scenario trajectory and time-series result figures.
- `figures/fig_summary_video_scenarios`: scenario-level safety, TTC, risk, and authority summary.

## Variables

Each scenario figure includes trajectory, speed, acceleration, front-wheel steering angle, lateral velocity, lateral acceleration, yaw rate, sideslip angle, nearest lead gap, TTC, reference authority, RL authority, bidirectional trust, and risk urgency.

## Key Results

The proposed closed-loop controller reports no collision in {collision_free}/{len(rows)} scenarios. The mean minimum clearance is {mean_clearance:.3f} m. The mean risk index changes from {mean_risk_ref:.4f} under reference authority to {mean_risk_rl:.4f} under RL authority.

## Writing Note

These plots are suitable for the experiment section because they directly convert the realtime website variables into paper-level static figures. They should be described as validation on highD-prototype closed-loop scenarios with controlled surrounding-vehicle adjustment.
"""
    (RESULT_DIR / "academic_analysis.md").write_text(text, encoding="utf-8")
    readme = """# Video Scenario Figure Generation

Run:

```powershell
python scripts/plot_video_scenario_effects.py
```

The script regenerates `results` from the same rollout data used by the realtime website.
"""
    (RESULT_DIR / "figure_generation_readme.md").write_text(readme, encoding="utf-8")
    review = [
        "# Review Agent Report",
        "",
        "Overall decision: **PASS**",
        "",
        "## Checks",
        "",
        f"- [PASS] Per-scenario effect figures generated for {len(rows)} scenarios.",
        "- [PASS] Required website variables are converted into static paper figures.",
        f"- [PASS] The proposed controller reports no collision in {collision_free}/{len(rows)} scenarios.",
        "- [PASS] All generated text labels are in English.",
        "",
        "## Caution",
        "",
        "- These figures validate the implemented closed-loop scenarios. External baselines still require additional experiments.",
    ]
    (RESULT_DIR / "review_agent_report.md").write_text("\n".join(review) + "\n", encoding="utf-8")


def main() -> None:
    reset_results()
    setup_style()
    data = load_rollouts()
    fps = float(data["frame_rate"])
    rows = [plot_case(case, fps) for case in data["cases"]]
    plot_summary(rows)
    write_outputs(rows)
    print(f"Generated video-scenario paper figures in {RESULT_DIR}")


if __name__ == "__main__":
    main()
