"""Generate GIF animations for checking Case 1 & 2 trajectory data.

This file is based on plot_case1_2_trajectory.py and reads the SAME rollout
data file:

outputs/shared_authority_validation/shared_authority_rollouts.js

Purpose
-------
The static trajectory figure overlays multiple time instants, so surrounding
vehicle boxes can visually look like a collision. This GIF uses one common
timestamp for ego and all surrounding vehicles in each frame, which is better
for checking whether the underlying rollout data actually collides.

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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_JS = ROOT / "outputs" / "shared_authority_validation" / "shared_authority_rollouts.js"
OUT_DIR = ROOT / "results" / "Case1_2_Trajectory_Figures"
GIF_DIR = OUT_DIR / "gif_check"


STYLE = {
    "font_family": ["Times New Roman", "DejaVu Serif"],
    "font_size": 8,
    "figure_dpi": 140,
    "save_dpi": 300,
    "figsize": (7.2, 3.2),
    "axis_label_size": 8,
    "tick_size": 7,
    "legend_size": 7,
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
    "outer_line_width": 1.10,
    "inner_line_width": 0.72,
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

    # For GIF checking, draw current red vehicle box for these ego trajectories.
    # Default only draws Proposed, because this is usually the final controlled ego.
    # If you want to check all four ego trajectories as vehicle boxes, change to:
    # ["human_pred_ego", "machine_ego", "controller_ego", "ego"]
    "gif_ego_vehicle_keys": ["controller_ego"],
}

TRAJECTORIES = {
    "human_pred_ego": {
        "label": "human",
        "color": COLORS["human"],
        "linestyle": "--",
        "linewidth": 0.95,
        "zorder": 8,
    },
    "machine_ego": {
        "label": "machine",
        "color": COLORS["machine"],
        "linestyle": "-.",
        "linewidth": 0.95,
        "zorder": 9,
    },
    "controller_ego": {
        "label": "Proposed",
        "color": COLORS["proposed"],
        "linestyle": "-",
        "linewidth": 1.25,
        "zorder": 11,
    },
    "ego": {
        "label": "Proposed without ARMPC",
        "color": COLORS["without_armpc"],
        "linestyle": ":",
        "linewidth": 1.05,
        "zorder": 10,
    },
}

LAYOUT = {
    "xlim": None,
    "ylim": None,
    "x_margin": 4.0,
    "y_margin": 1.2,
    "legend_ncol": 4,
    "subplots_adjust": {"top": 0.72, "bottom": 0.20, "left": 0.08, "right": 0.98},
    "match_browser_y_direction": True,
}

PAPER_CASES = {
    1: {"rollout_case_id": 2, "title": "Case 1"},
    2: {"rollout_case_id": 3, "title": "Case 2"},
}

# Video-aligned end positions from your observation.
# The endpoint is selected by the Proposed ego trajectory reaching this x.
# The same final frame is then used for all ego trajectories and all surrounding vehicles.
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
    "limit_from_ego_only": True,
}

ANIMATION = {
    "frame_step": 2,
    "fps": 8,

    # True: each frame only displays trajectory history from window start to current timestamp.
    # False: every frame displays the full trajectory window, only vehicles move.
    "draw_history_until_current": True,

    # True: draw gray surrounding vehicle boxes at the same timestamp as the red ego box.
    "draw_current_surrounding_vehicles": True,

    # True: also draw gray surrounding vehicle traces within the visible window.
    "draw_surrounding_vehicle_traces": True,

    # True: save a few PNG frames for quick checking besides GIF.
    "save_check_frames": True,
    "check_frame_ratios": [0.0, 0.25, 0.50, 0.75, 1.0],
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


def wxy(vehicle: dict, win: slice) -> np.ndarray:
    return xy(vehicle)[win]


def lane_markings(case: dict) -> list[float]:
    road = case.get("road", {})
    return road.get("lane_markings", []) or road.get("upper_lane_markings", []) or []


def yaw_series(vehicle: dict) -> np.ndarray:
    if "yaw" in vehicle:
        return np.asarray(vehicle["yaw"], dtype=float)
    points = xy(vehicle)
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    return np.unwrap(np.arctan2(dy, dx))


def plot_limits(case: dict, win: slice) -> tuple[tuple[float, float], tuple[float, float]]:
    point_sets = [wxy(case[k], win) for k in TRAJECTORIES.keys()]
    if not WINDOW["limit_from_ego_only"]:
        point_sets.extend(wxy(n, win) for n in case.get("neighbors", []))

    stack = np.vstack(point_sets)

    if LAYOUT["xlim"] is None:
        xlim = (
            float(np.min(stack[:, 0]) - LAYOUT["x_margin"]),
            float(np.max(stack[:, 0]) + LAYOUT["x_margin"]),
        )
    else:
        xlim = LAYOUT["xlim"]

    if LAYOUT["ylim"] is None:
        marks = lane_markings(case)
        if marks:
            ylim = (float(min(marks) - LAYOUT["y_margin"]), float(max(marks) + LAYOUT["y_margin"]))
        else:
            ylim = (
                float(np.min(stack[:, 1]) - LAYOUT["y_margin"]),
                float(np.max(stack[:, 1]) + LAYOUT["y_margin"]),
            )
    else:
        ylim = LAYOUT["ylim"]

    return xlim, ylim


def draw_road(ax, case: dict, win: slice) -> None:
    xlim, ylim = plot_limits(case, win)

    if ROAD["background"]:
        ax.set_facecolor(COLORS["road"])

    marks = lane_markings(case)
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
    ax.set_xlabel("Longitudinal position (m)")
    ax.set_ylabel("Lateral position (m)")

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


def draw_static_context(ax, case: dict, win: slice) -> None:
    if ANIMATION["draw_surrounding_vehicle_traces"]:
        for neighbor in case.get("neighbors", []):
            points = wxy(neighbor, win)
            ax.plot(points[:, 0], points[:, 1], color=COLORS["obstacle"], linewidth=0.65, alpha=0.45, zorder=2)


def draw_trajectory_history(ax, case: dict, win: slice, current_idx: int) -> None:
    start = 0 if win.start is None else int(win.start)
    stop = len(case["controller_ego"]["xy"]) if win.stop is None else int(win.stop)

    if ANIMATION["draw_history_until_current"]:
        hist_win = slice(start, min(current_idx + 1, stop))
    else:
        hist_win = win

    for key, cfg in TRAJECTORIES.items():
        points = wxy(case[key], hist_win)
        ax.plot(
            points[:, 0],
            points[:, 1],
            color=cfg["color"],
            linestyle=cfg["linestyle"],
            linewidth=cfg["linewidth"],
            label=cfg["label"],
            zorder=cfg["zorder"],
        )


def draw_current_vehicles(ax, case: dict, idx: int) -> None:
    if ANIMATION["draw_current_surrounding_vehicles"]:
        for neighbor in case.get("neighbors", []):
            length = float(neighbor.get("length", VEHICLES["ego_length"])) if VEHICLES["use_background_data_size"] else VEHICLES["ego_length"]
            width = float(neighbor.get("width", VEHICLES["ego_width"])) if VEHICLES["use_background_data_size"] else VEHICLES["ego_width"]
            add_vehicle(ax, neighbor, idx, length, width, COLORS["obstacle"], 0.82, 4)

    for key in VEHICLES["gif_ego_vehicle_keys"]:
        if key not in case:
            continue
        add_vehicle(
            ax,
            case[key],
            idx,
            VEHICLES["ego_length"],
            VEHICLES["ego_width"],
            COLORS["ego_vehicle"],
            0.96,
            TRAJECTORIES.get(key, {"zorder": 11})["zorder"] + 5,
        )


def place_legend(fig, ax) -> None:
    handles = []
    labels = []

    for key, cfg in TRAJECTORIES.items():
        line = plt.Line2D([0], [0], color=cfg["color"], linestyle=cfg["linestyle"], linewidth=cfg["linewidth"])
        handles.append(line)
        labels.append(cfg["label"])

    handles.extend(
        [
            Rectangle((0, 0), 1, 1, facecolor=COLORS["obstacle"], edgecolor=COLORS["obstacle_edge"], alpha=0.82),
            Rectangle((0, 0), 1, 1, facecolor=COLORS["ego_vehicle"], edgecolor=COLORS["obstacle_edge"], alpha=0.96),
        ]
    )
    labels.extend(["current surrounding vehicles", "current ego vehicle"])

    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.76),
        ncol=LAYOUT["legend_ncol"],
        frameon=False,
        handlelength=2.6,
        columnspacing=1.1,
    )


def save_check_frames(paper_case_id: int, rollout_case: dict, win: slice, frame_indices: list[int]) -> None:
    if not ANIMATION["save_check_frames"]:
        return

    out_subdir = GIF_DIR / f"Case{paper_case_id}_frames"
    out_subdir.mkdir(parents=True, exist_ok=True)

    chosen = []
    for ratio in ANIMATION["check_frame_ratios"]:
        pos = int(round(ratio * (len(frame_indices) - 1)))
        chosen.append(frame_indices[max(0, min(pos, len(frame_indices) - 1))])
    chosen = sorted(set(chosen))

    for idx in chosen:
        fig, ax = plt.subplots(figsize=STYLE["figsize"])
        draw_road(ax, rollout_case, win)
        draw_static_context(ax, rollout_case, win)
        draw_trajectory_history(ax, rollout_case, win, idx)
        draw_current_vehicles(ax, rollout_case, idx)
        current_x = float(xy(rollout_case[END_X_EGO_KEY])[idx, 0])
        ax.set_title(f"Case {paper_case_id}, frame={idx}, ego x={current_x:.1f} m", loc="left", fontsize=8)
        place_legend(fig, ax)
        fig.subplots_adjust(**LAYOUT["subplots_adjust"])
        fig.savefig(out_subdir / f"Case{paper_case_id}_frame_{idx:04d}.png", dpi=STYLE["save_dpi"], bbox_inches="tight")
        plt.close(fig)


def draw_case_animation(paper_case_id: int, rollout_case: dict) -> Path:
    win = critical_slice(rollout_case)
    win = apply_video_aligned_end(rollout_case, win, paper_case_id)

    start = 0 if win.start is None else int(win.start)
    stop = len(rollout_case["controller_ego"]["xy"]) if win.stop is None else int(win.stop)

    frame_step = max(1, int(ANIMATION["frame_step"]))
    frame_indices = list(range(start, stop, frame_step))
    if frame_indices[-1] != stop - 1:
        frame_indices.append(stop - 1)

    GIF_DIR.mkdir(parents=True, exist_ok=True)

    save_check_frames(paper_case_id, rollout_case, win, frame_indices)

    gif_path = GIF_DIR / f"Case{paper_case_id}_Trajectory_Check.gif"

    fig, ax = plt.subplots(figsize=STYLE["figsize"])

    # Some Matplotlib versions do not support `with PillowWriter(...) as writer`.
    # Use the explicit setup/grab_frame/finish workflow for better compatibility.
    writer = PillowWriter(fps=int(ANIMATION["fps"]))
    writer.setup(fig, str(gif_path), dpi=STYLE["save_dpi"])
    try:
        for idx in frame_indices:
            ax.clear()
            draw_road(ax, rollout_case, win)
            draw_static_context(ax, rollout_case, win)
            draw_trajectory_history(ax, rollout_case, win, idx)
            draw_current_vehicles(ax, rollout_case, idx)

            current_x = float(xy(rollout_case[END_X_EGO_KEY])[idx, 0])
            ax.set_title(f"Case {paper_case_id}, frame={idx}, ego x={current_x:.1f} m", loc="left", fontsize=8)

            # Draw legend every frame because ax.clear() removes previous artists.
            place_legend(fig, ax)
            fig.subplots_adjust(**LAYOUT["subplots_adjust"])
            writer.grab_frame()
    finally:
        writer.finish()
        plt.close(fig)

    return gif_path


def write_readme(paths: list[Path]) -> None:
    lines = [
        "# GIF Check for Case 1 & 2 Trajectories",
        "",
        f"Source rollout file: `{ROLLOUT_JS}`",
        "",
        "These GIFs use one common timestamp per frame for ego and surrounding vehicles.",
        "They are used to check whether the apparent overlap in static figures is a real data collision or only a multi-time-overlay artifact.",
        "",
        "## Outputs",
        "",
    ]
    for p in paths:
        lines.append(f"- `{p}`")
    lines.extend(
        [
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
        ]
    )
    (GIF_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_style()
    cases = load_cases()

    paths = []
    for paper_case_id, cfg in PAPER_CASES.items():
        rollout_case = get_case(cases, cfg["rollout_case_id"])
        gif_path = draw_case_animation(paper_case_id, rollout_case)
        paths.append(gif_path)
        print(f"Saved GIF: {gif_path}")

    write_readme(paths)
    print(f"Saved GIF check outputs to: {GIF_DIR}")


if __name__ == "__main__":
    main()
