"""Plot method-wise steering, speed, and steering-speed-time dynamics.

Input:
    outputs/shared_authority_validation/shared_authority_rollouts.js

Outputs:
    results/Case1_2_Method_Dynamics/steering_time/
    results/Case1_2_Method_Dynamics/speed_time/
    results/Case1_2_Method_Dynamics/steer_speed_time_3d/
    results/Case1_2_Method_Dynamics/steering_time_speed_color/
    results/Case1_2_Method_Dynamics/speed_time_steering_color/
    results/Case1_2_Method_Dynamics/beta_yaw_rate_phase/

Each validation case is saved as an independent IEEE-style figure in PNG,
PDF, and SVG formats.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for 3D projection

from plot_case1_2_trajectory import apply_video_aligned_end, critical_slice, get_time_vector


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_JS = ROOT / "outputs" / "shared_authority_validation" / "shared_authority_rollouts.js"
OUT_DIR = ROOT / "results" / "Case1_2_Method_Dynamics"


# =========================
# Editable visual interface
# =========================

STYLE = {
    "font_family": ["Times New Roman", "DejaVu Serif"],
    "font_size": 9.5,
    "label_size": 9.5,
    "tick_size": 8.4,
    "legend_size": 7.6,
    "line_width": 1.25,
    "grid_width": 0.35,
    "dpi": 600,
    # Suitable for a one-column IEEE figure. Increase width/height here if
    # the figure is used as a double-column panel.
    "figsize_2d": (3.55, 2.45),
    "figsize_3d": (3.55, 2.65),
    "figsize_color": (3.65, 2.45),
}

METHODS = {
    "human_pred_ego": {
        "label": "Human",
        "color": "#1F77B4",
        "linestyle": "--",
        "marker": "o",
    },
    "machine_ego": {
        "label": "Machine",
        "color": "#FF7F0E",
        "linestyle": "-.",
        "marker": "s",
    },
    "ra_rldm_ego": {
        "label": "RA-RLDM",
        "color": "#7B61FF",
        "linestyle": (0, (4, 1, 1, 1)),
        "marker": "v",
    },
    "ego": {
        "label": "TA-RLDM",
        "color": "#2CA02C",
        "linestyle": ":",
        "marker": "D",
    },
    "controller_ego": {
        "label": "TA-RL-ARMPC",
        "color": "#D62728",
        "linestyle": "-",
        "marker": "^",
    },
}

PLOT = {
    # Use the whole prediction horizon by default. Set to True if you want a
    # tighter maneuver-focused window based on steering activity.
    "use_active_window": False,
    "active_threshold_rad": 0.004,
    "pre_frames": 8,
    "post_frames": 8,
    "min_frames": 55,
    # Marker interval for time-series plots. Larger value gives cleaner curves.
    "markevery": 12,
    # Down-sample 3D curves for clearer markers while keeping continuous lines.
    "marker_every_3d": 14,
    # Keep the time axis consistent with highD real sampling. highD tracks are
    # sampled at 25 Hz, i.e., one step is 0.04 s.
    "display_dt": 1.0 / 25.0,
    # For a fair method-comparison phase plane, use one common kinematic
    # reconstruction for every method. If set to True, controller methods use
    # their internal dynamic beta/yaw-rate states, which are not directly
    # comparable with intention-level kinematic trajectories.
    "use_direct_controller_phase_states": False,
}

VEHICLE_PARAMS = {
    # Same nominal passenger-car parameters used by AdaptiveRobustMPCLiteConfig.
    "mass_kg": 1500.0,
    "yaw_inertia_kgm2": 2800.0,
    "lf_m": 1.2,
    "lr_m": 1.5,
    "cornering_stiffness_front": 55000.0,
    "cornering_stiffness_rear": 60000.0,
    "tire_mu": 0.85,
    "gravity_mps2": 9.81,
    # High-speed highway shared-driving validation should use a smaller
    # operational stability envelope than the hard numerical saturation in the
    # controller. These factors contract the tire-friction boundary.
    "yaw_boundary_factor": 0.60,
    "robust_yaw_boundary_factor": 0.45,
    "nominal_beta_limit_rad": 0.08,
    "robust_beta_limit_rad": 0.05,
}

PAPER_CASES = {
    # Paper Case 1 <- rollout validation case 2.
    1: 2,
    # Paper Case 2 <- rollout validation case 3.
    2: 3,
    # Paper Case 3 <- rollout validation case 6.
    3: 6,
    # Paper Case 4 <- rollout validation case 7.
    4: 7,
}


def load_rollouts() -> dict:
    text = ROLLOUT_JS.read_text(encoding="utf-8").strip()
    prefix = "window.SHARED_AUTHORITY_ROLLOUTS = "
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if text.endswith(";"):
        text = text[:-1]
    return json.loads(text)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": STYLE["font_family"],
            "font.size": STYLE["font_size"],
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def time_vector(case: dict, frame_rate: float) -> np.ndarray:
    n = len(case["controller_ego"]["speed"])
    rate = float(frame_rate) if frame_rate and frame_rate > 0 else 1.0 / float(PLOT["display_dt"])
    return np.arange(n, dtype=float) / rate


def active_window(case: dict) -> slice:
    n = len(case["controller_ego"]["speed"])
    if not bool(PLOT["use_active_window"]):
        return slice(0, n)

    active_indices: list[int] = []
    for key in METHODS:
        steer = np.asarray(case[key]["steer"], dtype=float)
        hit = np.where(np.abs(steer) > float(PLOT["active_threshold_rad"]))[0]
        if hit.size:
            active_indices.extend([int(hit[0]), int(hit[-1])])

    if not active_indices:
        return slice(0, n)

    start = max(0, min(active_indices) - int(PLOT["pre_frames"]))
    stop = min(n, max(active_indices) + int(PLOT["post_frames"]) + 1)
    if stop - start < int(PLOT["min_frames"]):
        mid = (start + stop) // 2
        half = int(PLOT["min_frames"]) // 2
        start = max(0, mid - half)
        stop = min(n, start + int(PLOT["min_frames"]))
        start = max(0, stop - int(PLOT["min_frames"]))
    return slice(start, stop)


def paper_window(case: dict, paper_case_id: int) -> slice:
    """Use the same maneuver window as the Case 1/2 trajectory keyframes."""

    return apply_video_aligned_end(case, critical_slice(case), paper_case_id)


def find_case(data: dict, rollout_case_id: int) -> dict:
    for case in data["cases"]:
        if int(case["record"]["case_id"]) == int(rollout_case_id):
            return case
    raise ValueError(f"Cannot find rollout case_id={rollout_case_id}")


def ensure_dirs() -> None:
    for folder in (
        "steering_time",
        "speed_time",
        "steer_speed_time_3d",
        "steering_time_speed_color",
        "speed_time_steering_color",
        "beta_yaw_rate_phase",
    ):
        (OUT_DIR / folder).mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, folder: str, stem: str) -> None:
    out = OUT_DIR / folder
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            out / f"{stem}.{ext}",
            dpi=STYLE["dpi"] if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.018,
        )
    plt.close(fig)


def plot_steering_time(case: dict, paper_case_id: int, frame_rate: float, win: slice) -> None:
    time = get_time_vector(case)
    if len(time) != len(case["controller_ego"]["speed"]):
        time = time_vector(case, frame_rate)
    t0 = time[0 if win.start is None else win.start]
    t = time[win] - t0

    fig, ax = plt.subplots(figsize=STYLE["figsize_2d"])
    for key, cfg in METHODS.items():
        steer = np.asarray(case[key]["steer"], dtype=float)[win]
        ax.plot(
            t,
            steer,
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            marker=cfg["marker"],
            markevery=PLOT["markevery"],
            markersize=2.2,
            linewidth=STYLE["line_width"],
        )

    ax.set_xlabel("Time (s)", fontsize=STYLE["label_size"])
    ax.set_ylabel("Steering Angle (rad)", fontsize=STYLE["label_size"])
    ax.tick_params(axis="both", labelsize=STYLE["tick_size"], direction="in")
    ax.grid(True, color="#D1D5DB", linewidth=STYLE["grid_width"], alpha=0.7)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
        fontsize=STYLE["legend_size"],
        handlelength=2.0,
        columnspacing=0.75,
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.19, top=0.78)
    save_figure(fig, "steering_time", f"Case{paper_case_id}_Steering_Time")


def plot_speed_time(case: dict, paper_case_id: int, frame_rate: float, win: slice) -> None:
    time = get_time_vector(case)
    if len(time) != len(case["controller_ego"]["speed"]):
        time = time_vector(case, frame_rate)
    t0 = time[0 if win.start is None else win.start]
    t = time[win] - t0

    fig, ax = plt.subplots(figsize=STYLE["figsize_2d"])
    for key, cfg in METHODS.items():
        speed = np.asarray(case[key]["speed"], dtype=float)[win]
        ax.plot(
            t,
            speed,
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            marker=cfg["marker"],
            markevery=PLOT["markevery"],
            markersize=2.2,
            linewidth=STYLE["line_width"],
        )

    ax.set_xlabel("Time (s)", fontsize=STYLE["label_size"])
    ax.set_ylabel("Speed (m/s)", fontsize=STYLE["label_size"])
    ax.tick_params(axis="both", labelsize=STYLE["tick_size"], direction="in")
    ax.grid(True, color="#D1D5DB", linewidth=STYLE["grid_width"], alpha=0.7)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
        fontsize=STYLE["legend_size"],
        handlelength=2.0,
        columnspacing=0.75,
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.19, top=0.78)
    save_figure(fig, "speed_time", f"Case{paper_case_id}_Speed_Time")


def plot_steer_speed_time_3d(case: dict, paper_case_id: int, frame_rate: float, win: slice) -> None:
    time = get_time_vector(case)
    if len(time) != len(case["controller_ego"]["speed"]):
        time = time_vector(case, frame_rate)
    t0 = time[0 if win.start is None else win.start]
    t = time[win] - t0

    fig = plt.figure(figsize=STYLE["figsize_3d"])
    ax = fig.add_subplot(111, projection="3d")
    for key, cfg in METHODS.items():
        speed = np.asarray(case[key]["speed"], dtype=float)[win]
        steer = np.asarray(case[key]["steer"], dtype=float)[win]
        ax.plot(
            t,
            steer,
            speed,
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=STYLE["line_width"],
        )
        marker_idx = np.arange(0, len(t), int(PLOT["marker_every_3d"]))
        ax.scatter(
            t[marker_idx],
            steer[marker_idx],
            speed[marker_idx],
            color=cfg["color"],
            marker=cfg["marker"],
            s=7,
            depthshade=False,
        )

    ax.set_xlabel("Time (s)", fontsize=STYLE["label_size"], labelpad=4)
    ax.set_ylabel("Steering Angle (rad)", fontsize=STYLE["label_size"], labelpad=5)
    ax.set_zlabel("Speed (m/s)", fontsize=STYLE["label_size"], labelpad=4)
    ax.tick_params(axis="both", labelsize=STYLE["tick_size"], pad=1)
    ax.view_init(elev=23, azim=-55)
    ax.grid(True)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=2,
        frameon=False,
        fontsize=STYLE["legend_size"],
        handlelength=1.8,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.87)
    save_figure(fig, "steer_speed_time_3d", f"Case{paper_case_id}_Steering_Speed_Time_3D")


def plot_steering_time_speed_color(case: dict, paper_case_id: int, frame_rate: float, win: slice) -> None:
    """Scheme A: x=time, y=steering angle, color=speed."""

    time = get_time_vector(case)
    if len(time) != len(case["controller_ego"]["speed"]):
        time = time_vector(case, frame_rate)
    t0 = time[0 if win.start is None else win.start]
    t = time[win] - t0
    all_speed = np.concatenate([np.asarray(case[key]["speed"], dtype=float)[win] for key in METHODS])
    norm = Normalize(vmin=float(np.min(all_speed)), vmax=float(np.max(all_speed)))

    fig, ax = plt.subplots(figsize=STYLE["figsize_color"])
    mappable = None
    for key, cfg in METHODS.items():
        steer = np.asarray(case[key]["steer"], dtype=float)[win]
        speed = np.asarray(case[key]["speed"], dtype=float)[win]
        ax.plot(
            t,
            steer,
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=STYLE["line_width"] + (0.2 if key == "controller_ego" else 0.0),
        )
        marker_idx = np.arange(0, len(t), int(PLOT["markevery"]))
        mappable = ax.scatter(
            t[marker_idx],
            steer[marker_idx],
            c=speed[marker_idx],
            cmap="viridis",
            norm=norm,
            marker=cfg["marker"],
            s=12,
            edgecolors="white",
            linewidths=0.25,
            zorder=4,
        )

    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.autoscale_view(scalex=False, scaley=True)
    ax.set_xlabel("Time (s)", fontsize=STYLE["label_size"])
    ax.set_ylabel("Steering Angle (rad)", fontsize=STYLE["label_size"])
    ax.tick_params(axis="both", labelsize=STYLE["tick_size"], direction="in")
    ax.grid(True, color="#D1D5DB", linewidth=STYLE["grid_width"], alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, frameon=False, fontsize=STYLE["legend_size"], handlelength=2.0, columnspacing=0.75)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.025, fraction=0.050)
    cbar.set_label("Speed (m/s)", fontsize=STYLE["label_size"])
    cbar.ax.tick_params(labelsize=STYLE["tick_size"], direction="in")
    fig.subplots_adjust(left=0.17, right=0.88, bottom=0.19, top=0.78)
    save_figure(fig, "steering_time_speed_color", f"Case{paper_case_id}_Steering_Time_Speed_Color")


def plot_speed_time_steering_color(case: dict, paper_case_id: int, frame_rate: float, win: slice) -> None:
    """Scheme B: x=time, y=speed, color=steering angle."""

    time = get_time_vector(case)
    if len(time) != len(case["controller_ego"]["speed"]):
        time = time_vector(case, frame_rate)
    t0 = time[0 if win.start is None else win.start]
    t = time[win] - t0
    all_steer = np.concatenate([np.asarray(case[key]["steer"], dtype=float)[win] for key in METHODS])
    max_abs = max(float(np.max(np.abs(all_steer))), 1e-4)
    norm = Normalize(vmin=-max_abs, vmax=max_abs)

    fig, ax = plt.subplots(figsize=STYLE["figsize_color"])
    mappable = None
    for key, cfg in METHODS.items():
        speed = np.asarray(case[key]["speed"], dtype=float)[win]
        steer = np.asarray(case[key]["steer"], dtype=float)[win]
        ax.plot(
            t,
            speed,
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=STYLE["line_width"] + (0.2 if key == "controller_ego" else 0.0),
        )
        marker_idx = np.arange(0, len(t), int(PLOT["markevery"]))
        mappable = ax.scatter(
            t[marker_idx],
            speed[marker_idx],
            c=steer[marker_idx],
            cmap="coolwarm",
            norm=norm,
            marker=cfg["marker"],
            s=12,
            edgecolors="white",
            linewidths=0.25,
            zorder=4,
        )

    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.autoscale_view(scalex=False, scaley=True)
    ax.set_xlabel("Time (s)", fontsize=STYLE["label_size"])
    ax.set_ylabel("Speed (m/s)", fontsize=STYLE["label_size"])
    ax.tick_params(axis="both", labelsize=STYLE["tick_size"], direction="in")
    ax.grid(True, color="#D1D5DB", linewidth=STYLE["grid_width"], alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=3, frameon=False, fontsize=STYLE["legend_size"], handlelength=2.0, columnspacing=0.75)
    cbar = fig.colorbar(mappable, ax=ax, pad=0.025, fraction=0.050)
    cbar.set_label("Steering Angle (rad)", fontsize=STYLE["label_size"])
    cbar.ax.tick_params(labelsize=STYLE["tick_size"], direction="in")
    fig.subplots_adjust(left=0.17, right=0.88, bottom=0.19, top=0.78)
    save_figure(fig, "speed_time_steering_color", f"Case{paper_case_id}_Speed_Time_Steering_Color")


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def beta_yaw_rate(case: dict, key: str, win: slice, frame_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return sideslip angle beta and yaw rate for one method.

    Controller rollouts contain dynamic states directly. For intention-level
    trajectories, beta and yaw rate are reconstructed from the trajectory
    kinematics so all compared methods can be shown in one phase plane.
    """

    veh = case[key]
    if bool(PLOT["use_direct_controller_phase_states"]) and "beta" in veh and "yaw_rate" in veh:
        return np.asarray(veh["beta"], dtype=float)[win], np.asarray(veh["yaw_rate"], dtype=float)[win]

    points = np.asarray(veh["xy"], dtype=float)
    yaw = np.unwrap(np.asarray(veh.get("yaw", np.zeros(len(points))), dtype=float))
    time = get_time_vector(case)
    if len(time) != len(points):
        time = time_vector(case, frame_rate)
    dt = np.gradient(time)
    dt = np.where(np.abs(dt) < 1e-6, 1.0 / max(frame_rate, 1.0), dt)
    vx_global = np.gradient(points[:, 0]) / dt
    vy_global = np.gradient(points[:, 1]) / dt
    course = np.arctan2(vy_global, vx_global)
    beta = wrap_to_pi(course - yaw)
    yaw_rate = np.gradient(yaw) / dt
    return beta[win], yaw_rate[win]


def stability_limits_from_speed(speed_mps: np.ndarray) -> dict[str, float]:
    """Compute a speed-aware phase-plane stability envelope.

    The yaw-rate boundary is derived from lateral acceleration:
        a_y = v_x r <= eta mu g.
    This is more appropriate for highway-speed validation than plotting the
    controller's numerical yaw-rate saturation directly. The sideslip envelope
    is contracted to a high-speed operational stability range.
    """

    params = VEHICLE_PARAMS
    v = float(np.nanmean(speed_mps)) if len(speed_mps) else 30.0
    v = max(v, 1.0)
    mu_g = float(params["tire_mu"] * params["gravity_mps2"])
    nominal_yaw = float(params["yaw_boundary_factor"] * mu_g / v)
    robust_yaw = float(params["robust_yaw_boundary_factor"] * mu_g / v)
    return {
        "speed_mps": v,
        "beta_nominal": float(params["nominal_beta_limit_rad"]),
        "beta_robust": float(params["robust_beta_limit_rad"]),
        "yaw_nominal": nominal_yaw,
        "yaw_robust": robust_yaw,
    }


def plot_beta_yaw_rate_phase(case: dict, paper_case_id: int, frame_rate: float, win: slice) -> None:
    fig, ax = plt.subplots(figsize=STYLE["figsize_2d"])
    phase_data = {}
    for key, cfg in METHODS.items():
        beta, yaw_rate = beta_yaw_rate(case, key, win, frame_rate)
        phase_data[key] = (beta, yaw_rate)
        ax.plot(
            beta,
            yaw_rate,
            label=cfg["label"],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=STYLE["line_width"] + (0.2 if key == "controller_ego" else 0.0),
        )
        marker_idx = np.arange(0, len(beta), int(PLOT["markevery"]))
        ax.plot(
            beta[marker_idx],
            yaw_rate[marker_idx],
            linestyle="None",
            marker=cfg["marker"],
            color=cfg["color"],
            markersize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.25,
            zorder=4,
        )

    ax.axhline(0.0, color="#9CA3AF", linewidth=0.5, alpha=0.75)
    ax.axvline(0.0, color="#9CA3AF", linewidth=0.5, alpha=0.75)
    controller_speed = np.asarray(case["controller_ego"]["speed"], dtype=float)[win]
    limits = stability_limits_from_speed(controller_speed)
    ax.text(
        0.02,
        0.04,
        rf"Robust boundary: $|\beta|\leq{limits['beta_robust']:.2f}$ rad, "
        rf"$|r|\leq{limits['yaw_robust']:.2f}$ rad/s",
        transform=ax.transAxes,
        fontsize=STYLE["legend_size"],
        color="#374151",
        va="bottom",
    )
    ax.set_xlabel(r"Sideslip Angle $\beta$ (rad)", fontsize=STYLE["label_size"])
    ax.set_ylabel("Yaw Rate (rad/s)", fontsize=STYLE["label_size"])
    ax.tick_params(axis="both", labelsize=STYLE["tick_size"], direction="in")
    ax.grid(True, color="#D1D5DB", linewidth=STYLE["grid_width"], alpha=0.7)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
        fontsize=STYLE["legend_size"],
        handlelength=2.0,
        columnspacing=0.75,
    )
    inset = inset_axes(ax, width="27%", height="36%", loc="upper right", borderpad=0.8)
    beta_lim = limits["beta_nominal"]
    yaw_lim = limits["yaw_nominal"]
    robust_beta = limits["beta_robust"]
    robust_yaw = limits["yaw_robust"]
    inset.add_patch(
        plt.Rectangle(
            (-beta_lim, -yaw_lim),
            2.0 * beta_lim,
            2.0 * yaw_lim,
            fill=False,
            linestyle="--",
            linewidth=0.7,
            edgecolor="#111827",
        )
    )
    inset.add_patch(
        plt.Rectangle(
            (-robust_beta, -robust_yaw),
            2.0 * robust_beta,
            2.0 * robust_yaw,
            fill=False,
            linestyle=":",
            linewidth=0.75,
            edgecolor="#6B7280",
        )
    )
    inset.axvline(-robust_beta, color="#6B7280", linestyle=":", linewidth=0.65)
    inset.axvline(robust_beta, color="#6B7280", linestyle=":", linewidth=0.65)
    inset.axhline(-robust_yaw, color="#6B7280", linestyle=":", linewidth=0.65)
    inset.axhline(robust_yaw, color="#6B7280", linestyle=":", linewidth=0.65)
    inset.axhline(0.0, color="#9CA3AF", linewidth=0.4)
    inset.axvline(0.0, color="#9CA3AF", linewidth=0.4)
    for key, cfg in METHODS.items():
        beta, yaw_rate = phase_data[key]
        inset.plot(beta, yaw_rate, color=cfg["color"], linestyle=cfg["linestyle"], linewidth=0.6, alpha=0.85)
    inset.set_xlim(-beta_lim * 1.08, beta_lim * 1.08)
    inset.set_ylim(-yaw_lim * 1.08, yaw_lim * 1.08)
    inset.set_xticks([-beta_lim, 0.0, beta_lim])
    inset.set_yticks([-yaw_lim, 0.0, yaw_lim])
    inset.tick_params(labelsize=5.8, direction="in", pad=1)
    inset.set_title(f"Stability domain, v={limits['speed_mps']:.1f} m/s", fontsize=6.4, pad=1.0)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.19, top=0.78)
    save_figure(fig, "beta_yaw_rate_phase", f"Case{paper_case_id}_Beta_YawRate_Phase")


def write_readme(data: dict) -> None:
    lines = [
        "# Case Method Dynamics Figures",
        "",
        "This folder contains method-wise steering angle, speed, and 3D steering-speed-time figures.",
        "",
        "Source rollout file: `outputs/shared_authority_validation/shared_authority_rollouts.js`",
        "Paper cases: Case 1 maps to rollout validation case 2; Case 2 maps to rollout validation case 3; Case 3 maps to rollout validation case 6; Case 4 maps to rollout validation case 7.",
        "",
        "Method mapping:",
    ]
    for key, cfg in METHODS.items():
        lines.append(f"- `{key}`: {cfg['label']}")
    lines.extend(
        [
            "",
            "Output folders:",
            "- `steering_time`: steering angle versus time.",
            "- `speed_time`: speed versus time.",
            "- `steer_speed_time_3d`: steering angle-speed-time 3D curves.",
            "- `steering_time_speed_color`: steering angle versus time with speed encoded by color.",
            "- `speed_time_steering_color`: speed versus time with steering angle encoded by color.",
            "- `beta_yaw_rate_phase`: sideslip angle-yaw rate phase-plane curves.",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    setup_style()
    ensure_dirs()
    data = load_rollouts()
    frame_rate = float(data.get("frame_rate", 25.0))
    for paper_case_id, rollout_case_id in PAPER_CASES.items():
        case = find_case(data, rollout_case_id)
        win = paper_window(case, paper_case_id)
        plot_steering_time(case, paper_case_id, frame_rate, win)
        plot_speed_time(case, paper_case_id, frame_rate, win)
        plot_steer_speed_time_3d(case, paper_case_id, frame_rate, win)
        plot_steering_time_speed_color(case, paper_case_id, frame_rate, win)
        plot_speed_time_steering_color(case, paper_case_id, frame_rate, win)
        plot_beta_yaw_rate_phase(case, paper_case_id, frame_rate, win)
    write_readme(data)
    print(f"Saved method dynamics figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
