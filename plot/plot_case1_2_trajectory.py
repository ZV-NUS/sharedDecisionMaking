"""Generate six key-frame figures for Case 1 & 2 trajectory visualization.

This script reads the SAME rollout data as plot_case1_2_trajectory.py:

outputs/shared_authority_validation/shared_authority_rollouts.js

Purpose
-------
Instead of stacking many vehicle snapshots in one static trajectory figure,
this script follows the common paper-style visualization: each case is shown
with six time-consistent key frames. In every subfigure, the ego vehicle and
surrounding vehicles are drawn at the SAME timestamp, with the timestamp
annotated as T=...s. This avoids the false-collision impression caused by
multi-time overlay.

Case mapping
------------
Paper Case 1 <- rollout/video Case 2
Paper Case 2 <- rollout/video Case 3

Trajectory mapping
------------------
human                  <- human_pred_ego
machine                <- machine_ego
Proposed               <- controller_ego
Proposed without ARMPC <- ego
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_JS = ROOT / "outputs" / "shared_authority_validation" / "shared_authority_rollouts.js"
OUT_DIR = ROOT / "results" / "Case1_2_Trajectory_Figures" / "keyframes"


# =========================
# Editable visual interface
# =========================

STYLE = {
    "font_family": ["Times New Roman", "DejaVu Serif"],
    "font_size": 8,
    "figure_dpi": 160,
    "save_dpi": 600,
    # 3 columns x 2 rows, similar to the reference paper style.
    "figsize": (11.4, 5.0),
    # Standalone panels keep the same local view style as the combined figure.
    "single_panel_figsize": (4.25, 2.25),
    # Standalone legend is exported as one horizontal row.
    "legend_figsize": (8.8, 0.55),
    "axis_label_size": 8,
    "tick_size": 7,
    "legend_size": 7,
    "timestamp_size": 7.4,
    "panel_label_size": 9,
}

COLORS = {
    "road": "#5E5D5D",
    "lane": "#E8EEF6",
    "lane_minor": "#C9D1DB",
    "obstacle": "#8F98A3",
    "obstacle_edge": "#F8FAFC",
    # Use high-contrast trajectory colors to make trajectory differences clearer.
    "human": "#1F77B4",
    "machine": "#FF7F0E",
    "ra_rldm": "#7B61FF",
    "ta_rldm": "#2CA02C",
    "proposed": "#D62728",
    "without_armpc": "#2CA02C",
}

ROAD = {
    "outer_line_width": 0.70,
    "inner_line_width": 0.42,
    "inner_dash": (0, (7, 6)),
    "lane_alpha": 0.82,
    "background": True,
}

VEHICLES = {
    "ego_length": 4.6,
    "ego_width": 1.8,
    "use_background_data_size": True,
    "edge_width": 0.45,
    "draw_heading_triangle": True,
    # In each key frame, draw all method-specific ego vehicles at the same
    # timestamp. The vehicle fill color is consistent with its trajectory.
    "ego_snapshot_keys": ["human_pred_ego", "machine_ego", "ra_rldm_ego", "ego", "controller_ego"],
    "ego_alpha": 0.96,
    "obstacle_alpha": 0.78,
}

TRAJECTORIES = {
    "human_pred_ego": {
        "label": "Human",
        "color": COLORS["human"],
        "linestyle": "--",
        "linewidth": 1.05,
        "marker": "o",
        "markevery": 8,
        "markersize": 2.2,
        "zorder": 8,
    },
    "machine_ego": {
        "label": "Machine",
        "color": COLORS["machine"],
        "linestyle": "-.",
        "linewidth": 1.05,
        "marker": "s",
        "markevery": 8,
        "markersize": 2.2,
        "zorder": 9,
    },
    "ra_rldm_ego": {
        "label": "RA-RLDM",
        "color": COLORS["ra_rldm"],
        "linestyle": "-.",
        "linewidth": 1.05,
        "marker": "v",
        "markevery": 8,
        "markersize": 2.2,
        "zorder": 10,
    },
    "controller_ego": {
        "label": "TA-RL-ARMPC",
        "color": COLORS["proposed"],
        "linestyle": "-",
        "linewidth": 1.35,
        "marker": "^",
        "markevery": 8,
        "markersize": 2.6,
        "zorder": 11,
    },
    "ego": {
        "label": "TA-RLDM",
        "color": COLORS["ta_rldm"],
        "linestyle": ":",
        "linewidth": 1.15,
        "marker": "D",
        "markevery": 8,
        "markersize": 2.2,
        "zorder": 10,
    },
}

LAYOUT = {
    # Local window around the current ego position in each panel.
    # This makes the figure look like video snapshots instead of a long trajectory plot.
    "local_x_before": 18.0,
    "local_x_after": 42.0,
    "y_margin": 0.35,
    "case_local_windows": {
    # Keep the same x-span as Case 1: 6 + 54 = 60 m.
    # Shift the Case 2 local window forward to keep front vehicles visible.
        2: {"local_x_before": 6.0, "local_x_after": 54.0},
    },

    # Case-specific vertical margin.
    # Only enlarge Case 2 vertical view to avoid flat-looking panels.
    "case_y_outer_margins": {
        2: 1.45,
    },
    "match_browser_y_direction": True,
    # Use auto aspect for paper keyframes. With equal metric aspect, a
    # 60-m longitudinal window and a multi-lane lateral range become too flat
    # after LaTeX scaling.
    "axis_aspect": "equal",
    "show_axes": False,
    "show_legend": True,
    "legend_ncol": 5,
    "subplots_adjust": {
        "top": 0.90,
        "bottom": 0.24,
        "left": 0.04,
        "right": 0.99,
        "hspace": 0.70,
        "wspace": 0.04,
    },
    # Keep Case 1 unchanged. Paper Case 2 has surrounding vehicles farther
    # ahead than the default local view, so it needs a moderately wider forward
    # true-position window; avoid also including the far rear vehicle here,
    # because that makes the paper panel too flat and visually inconsistent.
    "case_local_windows": {
        2: {"local_x_before": 18.0, "local_x_after": 54.0},
    },
}

LANE_VIEW = {
    # Show only three lanes around the vehicles in the current local panel.
    # Empty lanes with no ego/surrounding vehicles are removed.
    "enabled": True,
    "num_lanes": 3,
    # Extra vertical margin outside the selected three lanes.
    "outer_margin": 0.85,
}

PAPER_CASES = {
    1: {"rollout_case_id": 2, "title": "Case 1"},
    2: {"rollout_case_id": 3, "title": "Case 2"},
    3: {"rollout_case_id": 6, "title": "Case 3"},
    4: {"rollout_case_id": 7, "title": "Case 4"},
}

# Video-aligned endpoints from your observation.
# Paper Case 1 <- video/rollout Case 2: ego endpoint around 290 m.
# Paper Case 2 <- video/rollout Case 3: ego endpoint around 420 m.
END_X_BY_PAPER_CASE = {
    1: 290.0,
    2: 420.0,
}
END_X_EGO_KEY = "controller_ego"

WINDOW = {
    "enabled": True,
    "lateral_threshold_m": 0.35,
    "pre_frames": 12,
    "post_frames": 4,
    "min_window_frames": 55,
}

KEYFRAMES = {
    # Six key frames, like panels (a)-(f) or (e)-(j) in the reference figure.
    "count": 6,
    # Uniformly sample the visible maneuver window. The last panel (f) uses
    # the final visible timestamp to show the final state explicitly.
    "ratios": [0.00, 0.20, 0.40, 0.60, 0.80, 1.00],
    # If the rollout has no explicit time vector, estimate time from the highD
    # sampling rate. highD tracks are sampled at 25 Hz, i.e., dt = 0.04 s.
    "default_dt": 1.0 / 25.0,
    # If True, the displayed T starts from the beginning of the visible window.
    # If False, the displayed T is absolute rollout time.
    "relative_time": True,
    # Draw short local trajectory around the current frame for each method.
    "draw_local_trajectory": True,
    "trajectory_frames_before": 18,
    "trajectory_frames_after": 24,
    # Draw surrounding vehicle traces in the local panel.
    "draw_surrounding_traces": True,
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": STYLE["font_family"],
            "font.size": STYLE["font_size"],
            "axes.labelsize": STYLE["axis_label_size"],
            "xtick.labelsize": STYLE["tick_size"],
            "ytick.labelsize": STYLE["tick_size"],
            "legend.fontsize": STYLE["legend_size"],
            "figure.dpi": STYLE["figure_dpi"],
            "savefig.dpi": STYLE["save_dpi"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_cases() -> list[dict]:
    text = ROLLOUT_JS.read_text(encoding="utf-8")
    payload = text.split("=", 1)[1].strip().rstrip(";")
    return json.loads(payload)["cases"]


def get_case(cases: list[dict], case_id: int) -> dict:
    for case in cases:
        if int(case["record"]["case_id"]) == case_id:
            return case
    raise KeyError(f"Cannot find rollout case {case_id}")


def xy(vehicle: dict) -> np.ndarray:
    return np.asarray(vehicle["xy"], dtype=float)


def yaw_series(vehicle: dict) -> np.ndarray:
    if "yaw" in vehicle:
        return np.asarray(vehicle["yaw"], dtype=float)
    points = xy(vehicle)
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    return np.unwrap(np.arctan2(dy, dx))


def lane_markings(case: dict) -> list[float]:
    road = case.get("road", {})
    return road.get("lane_markings", []) or road.get("upper_lane_markings", []) or []


def critical_slice(case: dict) -> slice:
    if not WINDOW["enabled"]:
        n = len(case["controller_ego"]["xy"])
        return slice(0, n)

    n = len(case["controller_ego"]["xy"])
    active = []
    for key in TRAJECTORIES.keys():
        points = xy(case[key])
        lateral_change = np.abs(points[:, 1] - points[0, 1])
        indices = np.where(lateral_change > WINDOW["lateral_threshold_m"])[0]
        if len(indices):
            active.extend([int(indices[0]), int(indices[-1])])

    if not active:
        return slice(0, n)

    start = max(0, min(active) - WINDOW["pre_frames"])
    end = min(n, max(active) + WINDOW["post_frames"] + 1)

    if end - start < WINDOW["min_window_frames"]:
        mid = (start + end) // 2
        half = WINDOW["min_window_frames"] // 2
        start = max(0, mid - half)
        end = min(n, start + WINDOW["min_window_frames"])
        start = max(0, end - WINDOW["min_window_frames"])

    return slice(start, end)


def apply_video_aligned_end(case: dict, win: slice, paper_case_id: int) -> slice:
    """Trim the visible window using a video-aligned ego longitudinal endpoint."""
    if paper_case_id not in END_X_BY_PAPER_CASE:
        return win
    if END_X_EGO_KEY not in case:
        return win

    ego_points = xy(case[END_X_EGO_KEY])
    target_x = float(END_X_BY_PAPER_CASE[paper_case_id])

    start = 0 if win.start is None else int(win.start)
    current_stop = len(ego_points) if win.stop is None else int(win.stop)
    current_stop = max(start + 2, min(current_stop, len(ego_points)))

    full_x = ego_points[start:, 0]
    if len(full_x) < 2:
        return win

    if full_x[-1] >= full_x[0]:
        hit = np.where(full_x >= target_x)[0]
    else:
        hit = np.where(full_x <= target_x)[0]

    if len(hit):
        new_stop = start + int(hit[0]) + 1
    else:
        new_stop = current_stop

    new_stop = max(start + 2, min(new_stop, len(ego_points)))
    return slice(start, new_stop)


def get_time_vector(case: dict) -> np.ndarray:
    """Try to obtain time from rollout data; otherwise use default dt."""
    n = len(case["controller_ego"]["xy"])

    candidate_keys = ["t", "time", "times", "timestamp", "timestamps"]
    for key in candidate_keys:
        if key in case:
            arr = np.asarray(case[key], dtype=float).reshape(-1)
            if len(arr) == n:
                return arr

    record = case.get("record", {})
    for key in candidate_keys:
        if key in record:
            arr = np.asarray(record[key], dtype=float).reshape(-1)
            if len(arr) == n:
                return arr

    return np.arange(n, dtype=float) * float(KEYFRAMES["default_dt"])


def keyframe_indices(case: dict, win: slice) -> list[int]:
    start = 0 if win.start is None else int(win.start)
    stop = len(case["controller_ego"]["xy"]) if win.stop is None else int(win.stop)
    last = max(start, stop - 1)

    ratios = KEYFRAMES.get("ratios", None)
    if ratios is None or len(ratios) != int(KEYFRAMES["count"]):
        ratios = np.linspace(0.05, 0.90, int(KEYFRAMES["count"]))

    indices = [int(round(start + float(r) * max(0, last - start))) for r in ratios]
    indices = [max(start, min(i, last)) for i in indices]

    # Keep unique sorted indices. If duplicates happen due to short windows,
    # fill with evenly spaced alternatives.
    unique = sorted(set(indices))
    if len(unique) < int(KEYFRAMES["count"]):
        fill = np.linspace(start, last, int(KEYFRAMES["count"]), dtype=int).tolist()
        unique = sorted(set(unique + fill))[: int(KEYFRAMES["count"])]

    return unique[: int(KEYFRAMES["count"])]


def selected_lane_marks(case: dict, xlim: tuple[float, float], idx: int | None = None) -> list[float]:
    """Select only three lane boundaries around lanes occupied in the panel.

    The original road may contain more lanes than needed. For readability,
    this function keeps only three adjacent lanes that contain ego/surrounding
    vehicles in the current local x-window. Empty lanes outside this region are
    removed from the plot.
    """
    marks = sorted(lane_markings(case))
    if not LANE_VIEW.get("enabled", True) or len(marks) < 4:
        return marks

    vehicle_y = []

    def collect_vehicle_y(vehicle: dict) -> None:
        pts = xy(vehicle)
        if idx is not None:
            j = max(0, min(int(idx), len(pts) - 1))
            if xlim[0] <= pts[j, 0] <= xlim[1]:
                vehicle_y.append(float(pts[j, 1]))
        else:
            keep = (pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1])
            if np.any(keep):
                vehicle_y.extend([float(v) for v in pts[keep, 1]])

    for key in TRAJECTORIES.keys():
        collect_vehicle_y(case[key])
    for neighbor in case.get("neighbors", []):
        collect_vehicle_y(neighbor)

    if vehicle_y:
        y_ref = float(np.median(vehicle_y))
    else:
        y_ref = float(0.5 * (marks[0] + marks[-1]))

    lane_centers = [(marks[i] + marks[i + 1]) / 2.0 for i in range(len(marks) - 1)]
    center_lane = int(np.argmin(np.abs(np.asarray(lane_centers) - y_ref)))

    n_lanes = min(int(LANE_VIEW.get("num_lanes", 3)), len(marks) - 1)
    start_lane = center_lane - n_lanes // 2
    start_lane = max(0, min(start_lane, len(marks) - 1 - n_lanes))
    end_lane = start_lane + n_lanes

    return marks[start_lane : end_lane + 1]


def draw_road(ax, case: dict, xlim: tuple[float, float], idx: int | None = None) -> None:
    if ROAD["background"]:
        ax.set_facecolor(COLORS["road"])

    marks = selected_lane_marks(case, xlim, idx)

    if marks:
        base_margin = float(LANE_VIEW.get("outer_margin", 0.25))

        case_margin = None
        if hasattr(ax, "_paper_case_id"):
            case_margin = LAYOUT.get("case_y_outer_margins", {}).get(int(ax._paper_case_id), None)

        margin = float(case_margin) if case_margin is not None else base_margin

        ylim = (
            float(min(marks) - margin),
            float(max(marks) + margin),
        )

    else:
        # fallback from all ego trajectories
        all_points = np.vstack([xy(case[k]) for k in TRAJECTORIES.keys()])
        ylim = (
            float(np.min(all_points[:, 1]) - LAYOUT["y_margin"]),
            float(np.max(all_points[:, 1]) + LAYOUT["y_margin"]),
        )

    for i, y in enumerate(marks):
        outer = i == 0 or i == len(marks) - 1
        ax.plot(
            xlim,
            [y, y],
            color=COLORS["lane"] if outer else COLORS["lane_minor"],
            linestyle="-" if outer else ROAD["inner_dash"],
            linewidth=ROAD["outer_line_width"] if outer else ROAD["inner_line_width"],
            alpha=ROAD["lane_alpha"],
            zorder=1,
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if LAYOUT["match_browser_y_direction"]:
        ax.invert_yaxis()
    ax.set_aspect(LAYOUT["axis_aspect"], adjustable="box")

    if LAYOUT["show_axes"]:
        ax.set_xlabel("Longitudinal position (m)")
        ax.set_ylabel("Lateral position (m)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_vehicle_patch(
    ax,
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
    color: str,
    alpha: float,
    zorder: int,
) -> None:
    rect = Rectangle(
        (-length / 2.0, -width / 2.0),
        length,
        width,
        facecolor=color,
        edgecolor=COLORS["obstacle_edge"],
        linewidth=VEHICLES["edge_width"],
        alpha=alpha,
        zorder=zorder,
    )
    rect.set_transform(Affine2D().rotate(yaw).translate(x, y) + ax.transData)
    ax.add_patch(rect)

    if VEHICLES["draw_heading_triangle"]:
        triangle_points = np.array(
            [
                [length / 2.0, 0.0],
                [length * 0.28, width * 0.28],
                [length * 0.28, -width * 0.28],
            ]
        )
        tri = Polygon(
            triangle_points,
            closed=True,
            facecolor=COLORS["obstacle_edge"],
            edgecolor="none",
            alpha=min(1.0, alpha + 0.10),
            zorder=zorder + 1,
        )
        tri.set_transform(Affine2D().rotate(yaw).translate(x, y) + ax.transData)
        ax.add_patch(tri)


def add_vehicle(ax, vehicle: dict, idx: int, length: float, width: float, color: str, alpha: float, zorder: int) -> None:
    points = xy(vehicle)
    yaw = yaw_series(vehicle)
    idx = max(0, min(int(idx), len(points) - 1))
    draw_vehicle_patch(
        ax,
        points[idx, 0],
        points[idx, 1],
        yaw[idx],
        length,
        width,
        color,
        alpha,
        zorder,
    )


def draw_local_trajectories(ax, case: dict, idx: int, xlim: tuple[float, float]) -> None:
    if not KEYFRAMES["draw_local_trajectory"]:
        return

    n = len(case["controller_ego"]["xy"])
    start = max(0, idx - int(KEYFRAMES["trajectory_frames_before"]))
    stop = min(n, idx + int(KEYFRAMES["trajectory_frames_after"]) + 1)
    local_win = slice(start, stop)

    for key, cfg in TRAJECTORIES.items():
        points = xy(case[key])[local_win]
        # Keep the local curve inside the panel x-window.
        keep = (points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1])
        points = points[keep]
        if len(points) < 2:
            continue
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=cfg["linewidth"],
            marker=cfg.get("marker", None),
            markevery=cfg.get("markevery", None),
            markersize=cfg.get("markersize", 2.0),
            markerfacecolor=cfg["color"],
            markeredgecolor="white",
            markeredgewidth=0.25,
            zorder=cfg["zorder"],
        )


def draw_surrounding_traces(ax, case: dict, idx: int, xlim: tuple[float, float]) -> None:
    if not KEYFRAMES["draw_surrounding_traces"]:
        return

    n = len(case["controller_ego"]["xy"])
    start = max(0, idx - int(KEYFRAMES["trajectory_frames_before"]))
    stop = min(n, idx + int(KEYFRAMES["trajectory_frames_after"]) + 1)
    local_win = slice(start, stop)

    for neighbor in case.get("neighbors", []):
        points = xy(neighbor)[local_win]
        keep = (points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1])
        points = points[keep]
        if len(points) < 2:
            continue
        ax.plot(points[:, 0], points[:, 1], color=COLORS["obstacle"], linewidth=0.55, alpha=0.40, zorder=2)


def draw_current_vehicles(ax, case: dict, idx: int) -> None:
    for neighbor in case.get("neighbors", []):
        length = float(neighbor.get("length", VEHICLES["ego_length"])) if VEHICLES["use_background_data_size"] else VEHICLES["ego_length"]
        width = float(neighbor.get("width", VEHICLES["ego_width"])) if VEHICLES["use_background_data_size"] else VEHICLES["ego_width"]
        add_vehicle(ax, neighbor, idx, length, width, COLORS["obstacle"], VEHICLES["obstacle_alpha"], 4)

    for ego_key in VEHICLES["ego_snapshot_keys"]:
        cfg = TRAJECTORIES[ego_key]
        add_vehicle(
            ax,
            case[ego_key],
            idx,
            VEHICLES["ego_length"],
            VEHICLES["ego_width"],
            cfg["color"],
            VEHICLES["ego_alpha"],
            cfg["zorder"] + 5,
        )


def annotate_panel(ax, case_id: int, panel_id: int, idx: int, time_vec: np.ndarray, win: slice) -> None:
    start = 0 if win.start is None else int(win.start)
    t = float(time_vec[idx])
    if KEYFRAMES["relative_time"]:
        t -= float(time_vec[start])

    ax.text(
        0.90,
        0.17,
        f"T={t:.1f}s",
        transform=ax.transAxes,
        color="#222222",
        fontsize=STYLE["timestamp_size"],
        ha="center",
        va="center",
        zorder=30,
        bbox={
            "boxstyle": "square,pad=0.22",
            "facecolor": "white",
            "edgecolor": "white",
            "linewidth": 0.4,
            "alpha": 0.96,
        },
    )


def annotate_panel_label(ax, panel_id: int) -> None:
    ax.text(
        0.5,
        -0.25,
        f"({chr(ord('a') + panel_id)})",
        transform=ax.transAxes,
        color="black",
        fontsize=STYLE["panel_label_size"],
        ha="center",
        va="top",
        zorder=40,
        clip_on=False,
    )


def legend_handles_labels():
    handles = []
    labels = []

    for key, cfg in TRAJECTORIES.items():
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=cfg["color"],
                linestyle=cfg["linestyle"],
                linewidth=cfg["linewidth"],
                marker=cfg.get("marker", None),
                markersize=cfg.get("markersize", 2.0),
                markerfacecolor=cfg["color"],
                markeredgecolor="white",
                markeredgewidth=0.25,
            )
        )
        labels.append(cfg["label"])

    handles.append(
        Rectangle((0, 0), 1, 1, facecolor=COLORS["obstacle"], edgecolor=COLORS["obstacle_edge"], alpha=VEHICLES["obstacle_alpha"])
    )
    labels.append("Surrounding Vehicles")
    return handles, labels

def draw_keyframe_panel(
    ax,
    rollout_case: dict,
    paper_case_id: int,
    panel_id: int,
    idx: int,
    time_vec: np.ndarray,
    win: slice,
) -> None:
    ax._paper_case_id = int(paper_case_id)
    ego_points = xy(rollout_case[END_X_EGO_KEY])
    cx = float(ego_points[idx, 0])
    case_window = LAYOUT.get("case_local_windows", {}).get(int(paper_case_id), {})
    local_x_before = float(case_window.get("local_x_before", LAYOUT["local_x_before"]))
    local_x_after = float(case_window.get("local_x_after", LAYOUT["local_x_after"]))
    xlim = (cx - local_x_before, cx + local_x_after)

    draw_road(ax, rollout_case, xlim, idx)
    draw_surrounding_traces(ax, rollout_case, idx, xlim)
    draw_local_trajectories(ax, rollout_case, idx, xlim)
    draw_current_vehicles(ax, rollout_case, idx)
    annotate_panel(ax, paper_case_id, panel_id, idx, time_vec, win)


def save_single_keyframe_panels(
    paper_case_id: int,
    rollout_case: dict,
    indices: list[int],
    time_vec: np.ndarray,
    win: slice,
) -> None:
    case_dir = OUT_DIR / f"Case{paper_case_id}_panels"
    case_dir.mkdir(parents=True, exist_ok=True)

    for panel_id, idx in enumerate(indices):
        fig, ax = plt.subplots(figsize=STYLE["single_panel_figsize"])
        draw_keyframe_panel(ax, rollout_case, paper_case_id, panel_id, idx, time_vec, win)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.98)
        name = f"Case{paper_case_id}_Panel_{chr(ord('a') + panel_id)}"
        for ext in ("png", "pdf", "svg"):
            fig.savefig(
                case_dir / f"{name}.{ext}",
                bbox_inches="tight",
                dpi=STYLE["save_dpi"] if ext == "png" else None,
            )
        plt.close(fig)


def save_standalone_legend() -> None:
    handles, labels = legend_handles_labels()
    fig = plt.figure(figsize=STYLE["legend_figsize"])
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        handlelength=2.6,
        columnspacing=1.25,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"Trajectory_Keyframes_Legend.{ext}",
            bbox_inches="tight",
            dpi=STYLE["save_dpi"] if ext == "png" else None,
        )
    plt.close(fig)


def draw_case_keyframes(paper_case_id: int, rollout_case: dict) -> None:
    win = critical_slice(rollout_case)
    win = apply_video_aligned_end(rollout_case, win, paper_case_id)
    indices = keyframe_indices(rollout_case, win)
    time_vec = get_time_vector(rollout_case)

    fig, axes = plt.subplots(2, 3, figsize=STYLE["figsize"])
    axes = axes.reshape(-1)

    for panel_id, (ax, idx) in enumerate(zip(axes, indices)):
        draw_keyframe_panel(ax, rollout_case, paper_case_id, panel_id, idx, time_vec, win)
        annotate_panel_label(ax, panel_id)

    if LAYOUT["show_legend"]:
        handles, labels = legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.085),
            ncol=LAYOUT["legend_ncol"],
            frameon=False,
            handlelength=2.5,
            columnspacing=1.2,
        )

    fig.subplots_adjust(**LAYOUT["subplots_adjust"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"Case{paper_case_id}_Trajectory_Keyframes"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=STYLE["save_dpi"] if ext == "png" else None)
    plt.close(fig)

    save_single_keyframe_panels(paper_case_id, rollout_case, indices, time_vec, win)
    print(f"Saved keyframe figure: {OUT_DIR / (name + '.png')}")


def write_readme() -> None:
    lines = [
        "# Case 1 & 2 Trajectory Keyframe Figures",
        "",
        f"Source rollout file: `{ROLLOUT_JS}`",
        "",
        "This visualization uses six time-consistent key frames per case.",
        "In every subfigure, the ego vehicle and surrounding vehicles are drawn at the same timestamp.",
        "This avoids the false-collision impression caused by overlaying multiple vehicle snapshots in a single static trajectory plot.",
        "",
        "In addition to the combined 2x3 overview figures, each key frame is exported as an independent panel:",
        "",
        "- `Case1_panels/Case1_Panel_a-f.{png,pdf,svg}`.",
        "- `Case2_panels/Case2_Panel_a-f.{png,pdf,svg}`.",
        "- `Trajectory_Keyframes_Legend.{png,pdf,svg}` is a standalone one-row legend.",
        "",
        "## Case Mapping",
        "",
        "- Paper Case 1 uses rollout/video Case 2.",
        "- Paper Case 2 uses rollout/video Case 3.",
        "",
        "## Video-Aligned Endpoints",
        "",
        f"- Case 1 stops when `{END_X_EGO_KEY}` reaches about {END_X_BY_PAPER_CASE[1]} m.",
        f"- Case 2 stops when `{END_X_EGO_KEY}` reaches about {END_X_BY_PAPER_CASE[2]} m.",
        "",
        "## Tunable Interfaces",
        "",
        "- `KEYFRAMES['ratios']`: positions of the six snapshots within the visible window.",
        "- `KEYFRAMES['default_dt']`: sampling time used when no time vector exists in the rollout.",
        "- `LAYOUT['local_x_before']` and `LAYOUT['local_x_after']`: local panel view around the ego vehicle.",
        "",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_style()
    cases = load_cases()

    for paper_case_id, cfg in PAPER_CASES.items():
        rollout_case = get_case(cases, cfg["rollout_case_id"])
        draw_case_keyframes(paper_case_id, rollout_case)

    save_standalone_legend()
    write_readme()
    print(f"Saved keyframe figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
