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
    # 2 columns x 3 rows, similar to the reference paper style.
    "figsize": (7.2, 4.8),
    "axis_label_size": 8,
    "tick_size": 7,
    "legend_size": 7,
    "timestamp_size": 9,
    "panel_label_size": 9,
}

COLORS = {
    "road": "#303A46",
    "lane": "#E8EEF6",
    "lane_minor": "#C9D1DB",
    "obstacle": "#7E8794",
    "obstacle_edge": "#F8FAFC",
    "human": "#2F80ED",
    "machine": "#F2994A",
    "proposed": "#00A6B4",
    "without_armpc": "#7B61FF",
    "ego_vehicle": "#D62728",
}

ROAD = {
    "outer_line_width": 1.00,
    "inner_line_width": 0.65,
    "inner_dash": (0, (7, 6)),
    "lane_alpha": 0.95,
    "background": True,
}

VEHICLES = {
    "ego_length": 4.6,
    "ego_width": 1.8,
    "use_background_data_size": True,
    "edge_width": 0.45,
    "draw_heading_triangle": True,
    # In each key frame, draw only the current Proposed ego vehicle in red.
    # If you want to check another controlled ego trajectory, change this key.
    "ego_snapshot_key": "controller_ego",
    "ego_alpha": 0.96,
    "obstacle_alpha": 0.78,
}

TRAJECTORIES = {
    "human_pred_ego": {
        "label": "human",
        "color": COLORS["human"],
        "linestyle": "--",
        "linewidth": 0.80,
        "zorder": 8,
    },
    "machine_ego": {
        "label": "machine",
        "color": COLORS["machine"],
        "linestyle": "-.",
        "linewidth": 0.80,
        "zorder": 9,
    },
    "controller_ego": {
        "label": "Proposed",
        "color": COLORS["proposed"],
        "linestyle": "-",
        "linewidth": 1.05,
        "zorder": 11,
    },
    "ego": {
        "label": "Proposed without ARMPC",
        "color": COLORS["without_armpc"],
        "linestyle": ":",
        "linewidth": 0.90,
        "zorder": 10,
    },
}

LAYOUT = {
    # Local window around the current ego position in each panel.
    # This makes the figure look like video snapshots instead of a long trajectory plot.
    "local_x_before": 18.0,
    "local_x_after": 42.0,
    "y_margin": 1.2,
    "match_browser_y_direction": True,
    "show_axes": False,
    "show_legend": True,
    "legend_ncol": 4,
    "subplots_adjust": {
        "top": 0.88,
        "bottom": 0.06,
        "left": 0.04,
        "right": 0.99,
        "hspace": 0.10,
        "wspace": 0.03,
    },
}

# =========================
# Minimal Case-2 ratio fix
# =========================
# IMPORTANT: Do not change the original visual style.
# Case 1 is drawn with the original code path.
# Only Case 2 is forced to use the same road-window height as Case 1.
CASE2_MATCH_CASE1_RATIO = True

PANEL_EXPORT = {
    # Export individual panel files in addition to the combined figure.
    # Panel content uses the same drawing functions and colors as the combined figure.
    "enabled": True,
    "figsize": (4.8, 1.15),
    "formats": ("png", "pdf", "svg"),
    "clear_old_files": True,
}

PAPER_CASES = {
    1: {"rollout_case_id": 2, "title": "Case 1"},
    2: {"rollout_case_id": 3, "title": "Case 2"},
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
    # Avoid the exact first/last frame so that vehicles are not clipped by the panel boundary.
    "ratios": [0.05, 0.22, 0.39, 0.56, 0.73, 0.90],
    # If the rollout has no explicit time vector, estimate time from frame index.
    # Change this to your data sampling time if needed.
    # highD tracks are sampled at 25 Hz, i.e., dt = 0.04 s.
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


def road_ylim(case: dict) -> tuple[float, float]:
    """Original y-limit logic used by the initial script."""
    marks = lane_markings(case)
    if marks:
        return (float(min(marks) - LAYOUT["y_margin"]), float(max(marks) + LAYOUT["y_margin"]))

    all_points = np.vstack([xy(case[k]) for k in TRAJECTORIES.keys()])
    return (
        float(np.min(all_points[:, 1]) - LAYOUT["y_margin"]),
        float(np.max(all_points[:, 1]) + LAYOUT["y_margin"]),
    )


def make_same_span_ylim(case: dict, reference_y_span: float) -> tuple[float, float]:
    """Keep the current case lane center, but force Case 2 to Case 1's y-span."""
    marks = lane_markings(case)
    if marks:
        y_center = 0.5 * (float(min(marks)) + float(max(marks)))
    else:
        ylim0 = road_ylim(case)
        y_center = 0.5 * (ylim0[0] + ylim0[1])
    return (y_center - 0.5 * reference_y_span, y_center + 0.5 * reference_y_span)


def draw_road(ax, case: dict, xlim: tuple[float, float], fixed_ylim: tuple[float, float] | None = None) -> None:
    if ROAD["background"]:
        ax.set_facecolor(COLORS["road"])

    marks = lane_markings(case)
    ylim = fixed_ylim if fixed_ylim is not None else road_ylim(case)

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
    ax.set_aspect("equal", adjustable="box")

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

    ego_key = VEHICLES["ego_snapshot_key"]
    add_vehicle(
        ax,
        case[ego_key],
        idx,
        VEHICLES["ego_length"],
        VEHICLES["ego_width"],
        COLORS["ego_vehicle"],
        VEHICLES["ego_alpha"],
        TRAJECTORIES.get(ego_key, {"zorder": 11})["zorder"] + 5,
    )


def annotate_panel(ax, case_id: int, panel_id: int, idx: int, time_vec: np.ndarray, win: slice) -> None:
    start = 0 if win.start is None else int(win.start)
    t = float(time_vec[idx])
    if KEYFRAMES["relative_time"]:
        t -= float(time_vec[start])

    ax.text(
        0.82,
        0.82,
        f"T={t:.1f}s",
        transform=ax.transAxes,
        color="red",
        fontsize=STYLE["timestamp_size"],
        ha="left",
        va="center",
        zorder=30,
    )
    ax.text(
        0.96,
        0.14,
        f"({chr(ord('a') + panel_id)})",
        transform=ax.transAxes,
        color="black",
        fontsize=STYLE["panel_label_size"],
        ha="right",
        va="center",
        bbox=dict(facecolor="#F2994A", edgecolor="none", alpha=0.95, pad=2.2),
        zorder=31,
    )


def legend_handles_labels():
    handles = []
    labels = []

    for key, cfg in TRAJECTORIES.items():
        handles.append(plt.Line2D([0], [0], color=cfg["color"], linestyle=cfg["linestyle"], linewidth=cfg["linewidth"]))
        labels.append(cfg["label"])

    handles.extend(
        [
            Rectangle((0, 0), 1, 1, facecolor=COLORS["obstacle"], edgecolor=COLORS["obstacle_edge"], alpha=VEHICLES["obstacle_alpha"]),
            Rectangle((0, 0), 1, 1, facecolor=COLORS["ego_vehicle"], edgecolor=COLORS["obstacle_edge"], alpha=VEHICLES["ego_alpha"]),
        ]
    )
    labels.extend(["surrounding vehicles", "ego vehicle"])
    return handles, labels


def clear_panel_folder(panel_dir: Path, paper_case_id: int) -> None:
    if not PANEL_EXPORT["clear_old_files"]:
        return
    for ext in PANEL_EXPORT["formats"]:
        for p in panel_dir.glob(f"Case{paper_case_id}_Panel_*.{ext}"):
            p.unlink(missing_ok=True)


def save_case_panels(
    paper_case_id: int,
    rollout_case: dict,
    indices: list[int],
    time_vec: np.ndarray,
    win: slice,
    case1_y_span: float | None = None,
) -> None:
    if not PANEL_EXPORT["enabled"]:
        return

    panel_dir = OUT_DIR / f"Case{paper_case_id}_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)
    clear_panel_folder(panel_dir, paper_case_id)

    for panel_id, idx in enumerate(indices):
        fig, ax = plt.subplots(figsize=PANEL_EXPORT["figsize"])
        ego_points = xy(rollout_case[VEHICLES["ego_snapshot_key"]])
        cx = float(ego_points[idx, 0])
        xlim = (cx - float(LAYOUT["local_x_before"]), cx + float(LAYOUT["local_x_after"]))

        fixed_ylim = None
        if paper_case_id == 2 and CASE2_MATCH_CASE1_RATIO and case1_y_span is not None:
            fixed_ylim = make_same_span_ylim(rollout_case, case1_y_span)

        draw_road(ax, rollout_case, xlim, fixed_ylim=fixed_ylim)
        draw_surrounding_traces(ax, rollout_case, idx, xlim)
        draw_local_trajectories(ax, rollout_case, idx, xlim)
        draw_current_vehicles(ax, rollout_case, idx)
        annotate_panel(ax, paper_case_id, panel_id, idx, time_vec, win)

        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        stem = f"Case{paper_case_id}_Panel_{chr(ord('a') + panel_id)}"
        for ext in PANEL_EXPORT["formats"]:
            fig.savefig(
                panel_dir / f"{stem}.{ext}",
                dpi=STYLE["save_dpi"] if ext == "png" else None,
                pad_inches=0.0,
            )
        plt.close(fig)

    print(f"Saved individual panels: {panel_dir}")


def draw_case_keyframes(paper_case_id: int, rollout_case: dict, case1_y_span: float | None = None) -> None:
    win = critical_slice(rollout_case)
    win = apply_video_aligned_end(rollout_case, win, paper_case_id)
    indices = keyframe_indices(rollout_case, win)
    time_vec = get_time_vector(rollout_case)

    save_case_panels(paper_case_id, rollout_case, indices, time_vec, win, case1_y_span=case1_y_span)

    fig, axes = plt.subplots(3, 2, figsize=STYLE["figsize"])
    axes = axes.reshape(-1)

    for panel_id, (ax, idx) in enumerate(zip(axes, indices)):
        ego_points = xy(rollout_case[VEHICLES["ego_snapshot_key"]])
        cx = float(ego_points[idx, 0])
        xlim = (cx - float(LAYOUT["local_x_before"]), cx + float(LAYOUT["local_x_after"]))

        fixed_ylim = None
        if paper_case_id == 2 and CASE2_MATCH_CASE1_RATIO and case1_y_span is not None:
            fixed_ylim = make_same_span_ylim(rollout_case, case1_y_span)

        draw_road(ax, rollout_case, xlim, fixed_ylim=fixed_ylim)
        draw_surrounding_traces(ax, rollout_case, idx, xlim)
        draw_local_trajectories(ax, rollout_case, idx, xlim)
        draw_current_vehicles(ax, rollout_case, idx)
        annotate_panel(ax, paper_case_id, panel_id, idx, time_vec, win)

    if LAYOUT["show_legend"]:
        handles, labels = legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.04, 0.98),
            ncol=LAYOUT["legend_ncol"],
            frameon=False,
            handlelength=2.5,
            columnspacing=1.0,
        )

    fig.subplots_adjust(**LAYOUT["subplots_adjust"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"Case{paper_case_id}_Trajectory_Keyframes"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=STYLE["save_dpi"] if ext == "png" else None)
    plt.close(fig)

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

    # Reference span is taken from the ORIGINAL Case 1 y-limit.
    # Case 1 itself is not modified; this value is only used to prevent Case 2 from being flattened.
    case1_rollout = get_case(cases, PAPER_CASES[1]["rollout_case_id"])
    y0, y1 = road_ylim(case1_rollout)
    case1_y_span = abs(y1 - y0)
    print(f"Case 1 original y-span used as Case 2 reference: {case1_y_span:.3f} m")

    for paper_case_id, cfg in PAPER_CASES.items():
        rollout_case = get_case(cases, cfg["rollout_case_id"])
        draw_case_keyframes(paper_case_id, rollout_case, case1_y_span=case1_y_span)

    write_readme()
    print(f"Saved keyframe figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
