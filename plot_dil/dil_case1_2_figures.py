"""Generate DIL paper-style figures for paper Cases 1 and 2.

The plotting style follows the existing highD paper figures:

1. Case-specific six-keyframe trajectory figures.
2. A 2x3 comparison figure:
   row 1 = paper Case 1, row 2 = paper Case 2;
   columns = authority/trust, speed response with steering color, phase plane.

Data sources:
- DIL logs under driver_in_loop/experiments.
- Machine trajectory and surrounding vehicles from the same highD-injected
  rollout cases used by the DIL backend.

The script intentionally does not overwrite the original highD result figures.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = ROOT / "driver_in_loop" / "python_backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from scenarios.dil_extension import extend_case_for_dil  # noqa: E402
from scenarios.rollout_loader import RolloutScenarioRepository  # noqa: E402


ROLLOUT_JS = ROOT / "outputs" / "shared_authority_validation" / "shared_authority_rollouts.js"
EXPERIMENT_ROOT = ROOT / "driver_in_loop" / "experiments"
OUT_DIR = ROOT / "results" / "DIL_Experiment_Figures" / "Case1_2"


@dataclass(frozen=True)
class PaperCaseConfig:
    paper_case_id: int
    rollout_case_id: int
    experiment_dir: str
    event_start_s: float
    plot_duration_s: float
    x_before: float
    x_after: float


PAPER_CASES = {
    1: PaperCaseConfig(1, 2, "paper1_case2", 5.0, 3.5, 18.0, 42.0),
    2: PaperCaseConfig(2, 3, "paper2_case3", 5.0, 3.8, 18.0, 54.0),
}

MODES = {
    "human_only": {"label": "Human", "color": "#1F77B4", "linestyle": "--", "marker": "o"},
    "machine": {"label": "Machine", "color": "#FF7F0E", "linestyle": "-.", "marker": "s"},
    "ra_rldm": {"label": "RA-RLDM", "color": "#7B61FF", "linestyle": "-.", "marker": "v"},
    "ta_rldm": {"label": "TA-RLDM", "color": "#2CA02C", "linestyle": ":", "marker": "D"},
    "ta_rldm_armpc": {"label": "TA-RL-ARMPC", "color": "#D62728", "linestyle": "-", "marker": "^"},
}

STYLE = {
    "font_family": ["Times New Roman", "DejaVu Serif"],
    "font_size": 8.4,
    "label_size": 8.6,
    "tick_size": 7.4,
    "legend_size": 7.0,
    "line_width": 1.05,
    "proposed_line_width": 1.28,
    "dpi": 600,
    "trajectory_figsize": (11.4, 5.0),
    "summary_figsize": (7.2, 4.35),
}

ROAD = {
    "road_color": "#5E5D5D",
    "lane_color": "#E8EEF6",
    "lane_minor_color": "#C9D1DB",
    "lane_dash": (0, (7, 6)),
    "outer_lw": 0.70,
    "inner_lw": 0.42,
}

VEHICLE = {
    "ego_length": 4.6,
    "ego_width": 1.8,
    "obstacle_color": "#8F98A3",
    "obstacle_edge": "#F8FAFC",
    "obstacle_alpha": 0.78,
    "ego_alpha": 0.96,
    "edge_lw": 0.45,
}

KEYFRAME_RATIOS = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]


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


def save_fig(fig: plt.Figure, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            path_no_ext.with_suffix(f".{ext}"),
            dpi=STYLE["dpi"] if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.018,
        )


def load_extended_case(cfg: PaperCaseConfig) -> dict[str, Any]:
    repo = RolloutScenarioRepository(ROLLOUT_JS)
    case = repo.get_case(cfg.rollout_case_id)
    return extend_case_for_dil(case, start_s=5.0, end_s=6.0, source_frame_rate=repo.frame_rate)


def read_latest_complete_log(case_dir: str, mode: str, min_duration_s: float) -> tuple[pd.DataFrame | None, Path | None]:
    base = EXPERIMENT_ROOT / case_dir / mode
    if not base.exists():
        return None, None
    logs = sorted(base.glob("*/log.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    best: tuple[pd.DataFrame | None, Path | None] = (None, None)
    for path in logs:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "time_s" not in df or len(df) < 10:
            continue
        if float(df["time_s"].max()) >= min_duration_s:
            return df, path
        if best[0] is None:
            best = (df, path)
    return best


def load_logs(cfg: PaperCaseConfig) -> tuple[dict[str, pd.DataFrame | None], dict[str, str]]:
    min_duration = cfg.event_start_s + cfg.plot_duration_s - 0.15
    logs: dict[str, pd.DataFrame | None] = {}
    sources: dict[str, str] = {}
    for mode in ("human_only", "ra_rldm", "ta_rldm", "ta_rldm_armpc"):
        df, path = read_latest_complete_log(cfg.experiment_dir, mode, min_duration)
        logs[mode] = df
        sources[mode] = str(path) if path else "missing"
    return logs, sources


def sample_df(df: pd.DataFrame, col: str, t: np.ndarray, fallback: float = 0.0) -> np.ndarray:
    if df is None or col not in df:
        return np.full_like(t, fallback, dtype=float)
    src_t = np.asarray(df["time_s"], dtype=float)
    values = np.asarray(df[col], dtype=float)
    order = np.argsort(src_t)
    return np.interp(t, src_t[order], values[order], left=values[order][0], right=values[order][-1])


def xy_from_df(df: pd.DataFrame | None, t: np.ndarray) -> np.ndarray | None:
    if df is None or "ego_x" not in df or "ego_y" not in df:
        return None
    return np.column_stack([sample_df(df, "ego_x", t), sample_df(df, "ego_y", t)])


def rollout_time(case: dict[str, Any]) -> np.ndarray:
    n = len(case["controller_ego"]["xy"])
    return np.arange(n, dtype=float) / 25.0


def sample_vehicle(case: dict[str, Any], key: str, t: np.ndarray, field: str = "xy") -> np.ndarray:
    src_t = rollout_time(case)
    vehicle = case[key]
    arr = np.asarray(vehicle[field], dtype=float)
    if arr.ndim == 2:
        return np.column_stack([np.interp(t, src_t, arr[:, 0]), np.interp(t, src_t, arr[:, 1])])
    return np.interp(t, src_t, arr)


def yaw_from_xy(points: np.ndarray) -> np.ndarray:
    dx = np.gradient(points[:, 0])
    dy = np.gradient(points[:, 1])
    return np.unwrap(np.arctan2(dy, dx))


def lane_marks(case: dict[str, Any]) -> list[float]:
    road = case.get("road", {})
    return sorted(road.get("lane_markings", []) or road.get("upper_lane_markings", []) or [])


def selected_lane_marks(case: dict[str, Any], y_ref: float, n_lanes: int = 3) -> list[float]:
    marks = lane_marks(case)
    if len(marks) < 4:
        return marks
    centers = np.asarray([(marks[i] + marks[i + 1]) / 2 for i in range(len(marks) - 1)])
    center_lane = int(np.argmin(np.abs(centers - y_ref)))
    start = max(0, min(center_lane - n_lanes // 2, len(marks) - 1 - n_lanes))
    return marks[start : start + n_lanes + 1]


def draw_road(ax: plt.Axes, case: dict[str, Any], xlim: tuple[float, float], y_ref: float) -> None:
    ax.set_facecolor(ROAD["road_color"])
    marks = selected_lane_marks(case, y_ref)
    if marks:
        ylim = (min(marks) - 0.85, max(marks) + 0.85)
        for i, y in enumerate(marks):
            outer = i == 0 or i == len(marks) - 1
            ax.plot(
                xlim,
                [y, y],
                color=ROAD["lane_color"] if outer else ROAD["lane_minor_color"],
                linestyle="-" if outer else ROAD["lane_dash"],
                linewidth=ROAD["outer_lw"] if outer else ROAD["inner_lw"],
                alpha=0.84,
                zorder=1,
            )
    else:
        ylim = (y_ref - 5.5, y_ref + 5.5)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_vehicle(
    ax: plt.Axes,
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
    color: str,
    alpha: float,
    zorder: int,
) -> None:
    # Keep vehicle bodies as visually rectangular boxes. The trajectory panels
    # intentionally use an auto-scaled road view for IEEE layout; rotating
    # rectangles in data coordinates under non-equal x/y scaling makes them
    # look like parallelograms. Therefore the body is road-aligned, while the
    # front triangle still indicates the driving direction.
    rect = Rectangle(
        (float(x) - length / 2, float(y) - width / 2),
        length,
        width,
        facecolor=color,
        edgecolor=VEHICLE["obstacle_edge"],
        linewidth=VEHICLE["edge_lw"],
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(rect)
    tri = Polygon(
        np.asarray(
            [
                [float(x) + length / 2, float(y)],
                [float(x) + length * 0.28, float(y) + width * 0.28],
                [float(x) + length * 0.28, float(y) - width * 0.28],
            ]
        ),
        closed=True,
        facecolor=VEHICLE["obstacle_edge"],
        edgecolor="none",
        alpha=min(1.0, alpha + 0.1),
        zorder=zorder + 1,
    )
    ax.add_patch(tri)


def build_method_series(
    cfg: PaperCaseConfig,
    case: dict[str, Any],
    logs: dict[str, pd.DataFrame | None],
    t_abs: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    series: dict[str, dict[str, np.ndarray]] = {}

    human_xy = xy_from_df(logs.get("human_only"), t_abs)
    if human_xy is None:
        human_xy = sample_vehicle(case, "human_pred_ego", t_abs, "xy")
    series["human_only"] = {
        "xy": human_xy,
        "speed": sample_df(logs.get("human_only"), "ego_speed", t_abs, fallback=np.nan),
        "steer": sample_df(logs.get("human_only"), "ego_steer", t_abs, fallback=np.nan),
        "beta": sample_df(logs.get("human_only"), "sideslip_angle_beta", t_abs, fallback=np.nan),
        "yaw_rate": sample_df(logs.get("human_only"), "yaw_rate", t_abs, fallback=np.nan),
    }
    if np.isnan(series["human_only"]["speed"]).all():
        series["human_only"]["speed"] = sample_vehicle(case, "human_pred_ego", t_abs, "speed")
        series["human_only"]["steer"] = sample_vehicle(case, "human_pred_ego", t_abs, "steer")

    series["machine"] = {
        "xy": sample_vehicle(case, "machine_ego", t_abs, "xy"),
        "speed": sample_vehicle(case, "machine_ego", t_abs, "speed"),
        "steer": sample_vehicle(case, "machine_ego", t_abs, "steer"),
    }
    for mode in ("ra_rldm", "ta_rldm", "ta_rldm_armpc"):
        df = logs.get(mode)
        xy_points = xy_from_df(df, t_abs)
        fallback_key = {"ra_rldm": "ra_rldm_ego", "ta_rldm": "reference_ego", "ta_rldm_armpc": "controller_ego"}[mode]
        if xy_points is None:
            xy_points = sample_vehicle(case, fallback_key, t_abs, "xy")
        series[mode] = {
            "xy": xy_points,
            "speed": sample_df(df, "ego_speed", t_abs, fallback=np.nan),
            "steer": sample_df(df, "ego_steer", t_abs, fallback=np.nan),
            "beta": sample_df(df, "sideslip_angle_beta", t_abs, fallback=np.nan),
            "yaw_rate": sample_df(df, "yaw_rate", t_abs, fallback=np.nan),
        }
        if np.isnan(series[mode]["speed"]).all():
            series[mode]["speed"] = sample_vehicle(case, fallback_key, t_abs, "speed")
            series[mode]["steer"] = sample_vehicle(case, fallback_key, t_abs, "steer")
    return series


def method_yaw(series: dict[str, dict[str, np.ndarray]], key: str, idx: int) -> float:
    return float(yaw_from_xy(series[key]["xy"])[idx])


def draw_keyframe(
    ax: plt.Axes,
    cfg: PaperCaseConfig,
    case: dict[str, Any],
    series: dict[str, dict[str, np.ndarray]],
    t_abs: np.ndarray,
    idx: int,
    panel_idx: int,
    show_panel_label: bool = True,
) -> None:
    ego_xy = series["ta_rldm_armpc"]["xy"]
    y_ref = float(np.median([series[k]["xy"][idx, 1] for k in series]))

    current_x = [float(series[k]["xy"][idx, 0]) for k in series]
    src_t = rollout_time(case)
    for neighbor in case.get("neighbors", []):
        pts = np.asarray(neighbor["xy"], dtype=float)
        nx = float(np.interp(t_abs[idx], src_t, pts[:, 0]))
        if min(current_x) - 35.0 <= nx <= max(current_x) + 55.0:
            current_x.append(nx)
    span_min = min(current_x) - 8.0
    span_max = max(current_x) + 18.0
    min_span = cfg.x_before + cfg.x_after
    if span_max - span_min < min_span:
        center = 0.5 * (span_min + span_max)
        span_min = center - 0.5 * min_span
        span_max = center + 0.5 * min_span
    xlim = (span_min, span_max)
    draw_road(ax, case, xlim, y_ref)

    local_start = max(0, idx - 18)
    local_stop = min(len(t_abs), idx + 25)
    for neighbor in case.get("neighbors", []):
        pts = np.asarray(neighbor["xy"], dtype=float)
        src_t = rollout_time(case)
        px = np.interp(t_abs[local_start:local_stop], src_t, pts[:, 0])
        py = np.interp(t_abs[local_start:local_stop], src_t, pts[:, 1])
        ax.plot(px, py, color=VEHICLE["obstacle_color"], linewidth=0.55, alpha=0.36, zorder=2)
        x = float(np.interp(t_abs[idx], src_t, pts[:, 0]))
        y = float(np.interp(t_abs[idx], src_t, pts[:, 1]))
        prev_x = float(np.interp(max(t_abs[idx] - 0.04, 0), src_t, pts[:, 0]))
        prev_y = float(np.interp(max(t_abs[idx] - 0.04, 0), src_t, pts[:, 1]))
        yaw = math.atan2(y - prev_y, x - prev_x)
        draw_vehicle(
            ax,
            x,
            y,
            yaw,
            float(neighbor.get("length", VEHICLE["ego_length"])),
            float(neighbor.get("width", VEHICLE["ego_width"])),
            VEHICLE["obstacle_color"],
            VEHICLE["obstacle_alpha"],
            4,
        )

    for key, st in series.items():
        info = MODES[key]
        pts = st["xy"][local_start:local_stop]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=info["color"],
            linestyle=info["linestyle"],
            linewidth=STYLE["proposed_line_width"] if key == "ta_rldm_armpc" else STYLE["line_width"],
            marker=info["marker"],
            markevery=8,
            markersize=2.0,
            markerfacecolor=info["color"],
            markeredgecolor="white",
            markeredgewidth=0.25,
            zorder=8,
        )
    for key, st in series.items():
        info = MODES[key]
        draw_vehicle(
            ax,
            float(st["xy"][idx, 0]),
            float(st["xy"][idx, 1]),
            method_yaw(series, key, idx),
            VEHICLE["ego_length"],
            VEHICLE["ego_width"],
            info["color"],
            VEHICLE["ego_alpha"],
            12,
        )

    t_rel = float(t_abs[idx] - cfg.event_start_s)
    ax.text(
        0.90,
        0.17,
        f"T={t_rel:.1f}s",
        transform=ax.transAxes,
        color="#222222",
        fontsize=7.4,
        ha="center",
        va="center",
        bbox={"boxstyle": "square,pad=0.22", "facecolor": "white", "edgecolor": "white", "alpha": 0.96},
        zorder=30,
    )
    if show_panel_label:
        ax.text(0.5, -0.25, f"({chr(ord('a') + panel_idx)})", transform=ax.transAxes, ha="center", va="top", fontsize=10)


def plot_single_trajectory_panel(
    cfg: PaperCaseConfig,
    case: dict[str, Any],
    series: dict[str, dict[str, np.ndarray]],
    t_abs: np.ndarray,
    idx: int,
    panel_idx: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(3.55, 1.05))
    draw_keyframe(ax, cfg, case, series, t_abs, idx, panel_idx, show_panel_label=False)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.02)
    panel_name = f"Case{cfg.paper_case_id}_Panel_{chr(ord('a') + panel_idx)}"
    save_fig(fig, OUT_DIR / "trajectory" / "panels" / panel_name)
    plt.close(fig)


def trajectory_legend() -> tuple[list[Any], list[str]]:
    handles = []
    labels = []
    for key in ("human_only", "machine", "ra_rldm", "ta_rldm", "ta_rldm_armpc"):
        info = MODES[key]
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=info["color"],
                linestyle=info["linestyle"],
                linewidth=STYLE["proposed_line_width"] if key == "ta_rldm_armpc" else STYLE["line_width"],
                marker=info["marker"],
                markersize=2.5,
                markerfacecolor=info["color"],
                markeredgecolor="white",
                markeredgewidth=0.25,
            )
        )
        labels.append(info["label"])
    handles.append(Rectangle((0, 0), 1, 1, facecolor=VEHICLE["obstacle_color"], edgecolor=VEHICLE["obstacle_edge"], alpha=0.78))
    labels.append("Surrounding Vehicles")
    return handles, labels


def plot_trajectory_case(cfg: PaperCaseConfig, case: dict[str, Any], logs: dict[str, pd.DataFrame | None]) -> None:
    t_rel = np.asarray([r * cfg.plot_duration_s for r in KEYFRAME_RATIOS], dtype=float)
    t_abs = cfg.event_start_s + np.linspace(0.0, cfg.plot_duration_s, 200)
    series = build_method_series(cfg, case, logs, t_abs)
    key_idx = [int(round(r * (len(t_abs) - 1))) for r in KEYFRAME_RATIOS]

    fig, axes = plt.subplots(2, 3, figsize=STYLE["trajectory_figsize"])
    for pidx, (ax, idx) in enumerate(zip(axes.reshape(-1), key_idx)):
        draw_keyframe(ax, cfg, case, series, t_abs, idx, pidx)
        plot_single_trajectory_panel(cfg, case, series, t_abs, idx, pidx)

    handles, labels = trajectory_legend()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=6,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.2,
        fontsize=STYLE["legend_size"],
    )
    fig.subplots_adjust(top=0.90, bottom=0.24, left=0.04, right=0.99, hspace=0.70, wspace=0.04)
    save_fig(fig, OUT_DIR / "trajectory" / f"DIL_Case{cfg.paper_case_id}_Trajectory_Keyframes")
    plt.close(fig)


def get_authority_source_log(logs: dict[str, pd.DataFrame | None]) -> pd.DataFrame | None:
    for mode in ("ta_rldm_armpc", "ta_rldm", "ra_rldm", "human_only"):
        df = logs.get(mode)
        if df is not None and "authority_ref" in df:
            return df
    return None


def interp_mode_value(series: dict[str, dict[str, np.ndarray]], key: str, field: str, t: np.ndarray, t_abs: np.ndarray) -> np.ndarray:
    return np.interp(t, t_abs, np.asarray(series[key][field], dtype=float))


def plot_authority_trust(ax: plt.Axes, cfg: PaperCaseConfig, logs: dict[str, pd.DataFrame | None], t_abs: np.ndarray) -> None:
    df = get_authority_source_log(logs)
    t_rel = t_abs - cfg.event_start_s
    if df is None:
        authority_ref = np.full_like(t_abs, 0.5)
        authority_ra = np.full_like(t_abs, 0.05)
        authority_ta = np.full_like(t_abs, 0.5)
        trust_hm = np.full_like(t_abs, 0.7)
        trust_mh = np.full_like(t_abs, 0.7)
    else:
        authority_ref = sample_df(df, "authority_ref", t_abs, 0.5)
        authority_ra = sample_df(df, "authority_ra_rldm", t_abs, 0.05)
        authority_ta = sample_df(df, "authority_ta_rldm", t_abs, 0.5)
        trust_hm = sample_df(df, "trust_human_to_machine", t_abs, 0.7)
        trust_mh = sample_df(df, "trust_machine_to_human", t_abs, 0.7)

    ax2 = ax.twinx()
    lns = []
    lns += ax.plot(t_rel, authority_ra, color=MODES["ra_rldm"]["color"], linestyle="-.", linewidth=1.05, label="RA-RLDM")
    lns += ax.plot(t_rel, authority_ta, color=MODES["ta_rldm_armpc"]["color"], linestyle="-", linewidth=1.20, label="TA-RLDM")
    lns += ax.plot(t_rel, authority_ref, color="#8B5CF6", linestyle=":", linewidth=1.05, label="Reference Authority")
    lns += ax2.plot(t_rel, trust_hm, color=MODES["machine"]["color"], linestyle="-.", linewidth=1.05, label=r"$T_h$")
    lns += ax2.plot(t_rel, trust_mh, color=MODES["human_only"]["color"], linestyle="--", linewidth=1.05, label=r"$T_m$")
    ax.set_xlim(0, cfg.plot_duration_s)
    ax.set_ylim(-0.03, 1.03)
    ax2.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Authority Level")
    ax2.set_ylabel("Trust Level")
    ax.grid(True, color="#D1D5DB", linewidth=0.35, alpha=0.7)
    ax.legend(lns, [h.get_label() for h in lns], loc="upper center", bbox_to_anchor=(0.5, 1.34), ncol=3, frameon=False, fontsize=6.4, handlelength=1.4, columnspacing=0.55)


def plot_speed_steer_color(ax: plt.Axes, cfg: PaperCaseConfig, series: dict[str, dict[str, np.ndarray]], t_abs: np.ndarray) -> None:
    t_rel = t_abs - cfg.event_start_s
    all_steer = np.concatenate([series[k]["steer"] for k in series if "steer" in series[k]])
    max_abs = max(float(np.nanmax(np.abs(all_steer))), 1e-4)
    norm = Normalize(vmin=-max_abs, vmax=max_abs)
    mappable = None
    for key in ("human_only", "machine", "ra_rldm", "ta_rldm", "ta_rldm_armpc"):
        info = MODES[key]
        speed = series[key]["speed"]
        steer = series[key]["steer"]
        ax.plot(t_rel, speed, color=info["color"], linestyle=info["linestyle"], linewidth=1.25 if key == "ta_rldm_armpc" else 1.05, label=info["label"])
        mark_idx = np.arange(0, len(t_rel), 20)
        mappable = ax.scatter(t_rel[mark_idx], speed[mark_idx], c=steer[mark_idx], cmap="coolwarm", norm=norm, marker=info["marker"], s=10, edgecolors="white", linewidths=0.25, zorder=4)
    ax.set_xlim(0, cfg.plot_duration_s)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, color="#D1D5DB", linewidth=0.35, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.34), ncol=3, frameon=False, fontsize=6.4, handlelength=1.5, columnspacing=0.6)
    cbar = plt.colorbar(mappable, ax=ax, pad=0.02, fraction=0.052)
    cbar.set_label("Steering Angle (rad)", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.6, direction="in")


def reconstruct_beta_yaw(points: np.ndarray, t_abs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dt = np.gradient(t_abs)
    dt = np.where(np.abs(dt) < 1e-6, 1 / 60.0, dt)
    yaw = yaw_from_xy(points)
    vx = np.gradient(points[:, 0]) / dt
    vy = np.gradient(points[:, 1]) / dt
    course = np.arctan2(vy, vx)
    beta = (course - yaw + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = np.gradient(yaw) / dt
    return beta, yaw_rate


def plot_phase(ax: plt.Axes, cfg: PaperCaseConfig, series: dict[str, dict[str, np.ndarray]], t_abs: np.ndarray) -> None:
    phase_values = []
    for key in ("human_only", "machine", "ra_rldm", "ta_rldm", "ta_rldm_armpc"):
        info = MODES[key]
        beta = series[key].get("beta")
        yaw_rate = series[key].get("yaw_rate")
        if beta is None or yaw_rate is None or np.isnan(beta).all() or np.isnan(yaw_rate).all():
            beta, yaw_rate = reconstruct_beta_yaw(series[key]["xy"], t_abs)
        beta = np.clip(np.nan_to_num(beta, nan=0.0), -0.08, 0.08)
        yaw_rate = np.clip(np.nan_to_num(yaw_rate, nan=0.0), -0.45, 0.45)
        phase_values.append(yaw_rate)
        ax.plot(beta, yaw_rate, color=info["color"], linestyle=info["linestyle"], linewidth=1.25 if key == "ta_rldm_armpc" else 1.05, label=info["label"])
    ax.axhline(0, color="#9CA3AF", linewidth=0.5)
    ax.axvline(0, color="#9CA3AF", linewidth=0.5)
    speed = series["ta_rldm_armpc"]["speed"]
    v = float(np.nanmean(speed)) if len(speed) else 30.0
    yaw_lim = 0.45 * 0.85 * 9.81 / max(v, 1.0)
    ax.text(0.02, 0.04, rf"Robust boundary: $|\beta|\leq0.05$ rad, $|r|\leq{yaw_lim:.2f}$ rad/s", transform=ax.transAxes, fontsize=6.4, color="#374151")
    ax.set_xlabel(r"Sideslip Angle $\beta$ (rad)")
    ax.set_ylabel("Yaw Rate (rad/s)")
    ax.set_xlim(-0.06, 0.06)
    y_abs = float(np.nanpercentile(np.abs(np.concatenate(phase_values)), 98)) if phase_values else 0.2
    y_lim = min(0.45, max(0.18, y_abs * 1.15))
    ax.set_ylim(-y_lim, y_lim)
    ax.grid(True, color="#D1D5DB", linewidth=0.35, alpha=0.7)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.34), ncol=3, frameon=False, fontsize=6.4, handlelength=1.5, columnspacing=0.6)
    inset = inset_axes(ax, width="29%", height="36%", loc="upper right", borderpad=0.75)
    beta_lim = 0.08
    yaw_nom = 0.60 * 0.85 * 9.81 / max(v, 1.0)
    inset.add_patch(Rectangle((-beta_lim, -yaw_nom), 2 * beta_lim, 2 * yaw_nom, fill=False, linestyle="--", linewidth=0.7, edgecolor="#111827"))
    inset.add_patch(Rectangle((-0.05, -yaw_lim), 0.10, 2 * yaw_lim, fill=False, linestyle=":", linewidth=0.75, edgecolor="#6B7280"))
    inset.axhline(0, color="#9CA3AF", linewidth=0.4)
    inset.axvline(0, color="#9CA3AF", linewidth=0.4)
    for key in ("ta_rldm_armpc",):
        beta = series[key].get("beta")
        yaw_rate = series[key].get("yaw_rate")
        if beta is None or yaw_rate is None or np.isnan(beta).all() or np.isnan(yaw_rate).all():
            beta, yaw_rate = reconstruct_beta_yaw(series[key]["xy"], t_abs)
        inset.plot(beta, yaw_rate, color=MODES[key]["color"], linewidth=0.65)
    inset.set_xlim(-beta_lim * 1.08, beta_lim * 1.08)
    inset.set_ylim(-yaw_nom * 1.08, yaw_nom * 1.08)
    inset.tick_params(labelsize=5.5, direction="in", pad=1)
    inset.set_title(f"Stability domain, v={v:.1f} m/s", fontsize=5.8, pad=1)


def plot_summary(cases: dict[int, dict[str, Any]], logs_all: dict[int, dict[str, pd.DataFrame | None]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=STYLE["summary_figsize"])
    for row, paper_case_id in enumerate((1, 2)):
        cfg = PAPER_CASES[paper_case_id]
        case = cases[paper_case_id]
        logs = logs_all[paper_case_id]
        t_abs = cfg.event_start_s + np.linspace(0.0, cfg.plot_duration_s, 220)
        series = build_method_series(cfg, case, logs, t_abs)
        plot_authority_trust(axes[row, 0], cfg, logs, t_abs)
        plot_speed_steer_color(axes[row, 1], cfg, series, t_abs)
        plot_phase(axes[row, 2], cfg, series, t_abs)
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for ax, label in zip(axes.reshape(-1), labels):
        ax.text(0.5, -0.32, label, transform=ax.transAxes, ha="center", va="top", fontsize=10)
        ax.tick_params(axis="both", labelsize=7.2, direction="in")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.90, bottom=0.16, wspace=0.62, hspace=0.72)
    save_fig(fig, OUT_DIR / "DIL_Case1_2_Authority_Control_Stability")
    plt.close(fig)
    plot_summary_panels(cases, logs_all)


def plot_summary_panels(cases: dict[int, dict[str, Any]], logs_all: dict[int, dict[str, pd.DataFrame | None]]) -> None:
    panel_specs = [
        (1, "authority", "a"),
        (1, "speed", "b"),
        (1, "phase", "c"),
        (2, "authority", "d"),
        (2, "speed", "e"),
        (2, "phase", "f"),
    ]
    for paper_case_id, panel_type, letter in panel_specs:
        cfg = PAPER_CASES[paper_case_id]
        case = cases[paper_case_id]
        logs = logs_all[paper_case_id]
        t_abs = cfg.event_start_s + np.linspace(0.0, cfg.plot_duration_s, 220)
        series = build_method_series(cfg, case, logs, t_abs)
        fig, ax = plt.subplots(1, 1, figsize=(2.35, 1.55))
        if panel_type == "authority":
            plot_authority_trust(ax, cfg, logs, t_abs)
        elif panel_type == "speed":
            plot_speed_steer_color(ax, cfg, series, t_abs)
        else:
            plot_phase(ax, cfg, series, t_abs)
        ax.tick_params(axis="both", labelsize=7.2, direction="in")
        fig.subplots_adjust(left=0.17, right=0.93, top=0.78, bottom=0.20)
        save_fig(fig, OUT_DIR / "summary" / "panels" / f"Summary_Panel_{letter}")
        plt.close(fig)


def write_report(sources: dict[int, dict[str, str]]) -> None:
    def rel(path_text: str) -> str:
        if path_text == "missing":
            return path_text
        try:
            return str(Path(path_text).resolve().relative_to(ROOT))
        except Exception:
            return path_text

    lines = [
        "# DIL Case 1 & 2 Figure Generation Report",
        "",
        "Generated figures follow the paper-style Case 1 and Case 2 requirements.",
        "",
        "## Data Sources",
    ]
    for case_id, src in sources.items():
        lines.append(f"### Paper Case {case_id}")
        for mode, path in src.items():
            lines.append(f"- {mode}: `{rel(path)}`")
    lines.extend(
        [
            "",
            "## Outputs",
            f"- `{(OUT_DIR / 'trajectory' / 'DIL_Case1_Trajectory_Keyframes.png').relative_to(ROOT)}`",
            f"- `{(OUT_DIR / 'trajectory' / 'DIL_Case2_Trajectory_Keyframes.png').relative_to(ROOT)}`",
            f"- `{(OUT_DIR / 'DIL_Case1_2_Authority_Control_Stability.png').relative_to(ROOT)}`",
            f"- `{(OUT_DIR / 'trajectory' / 'panels').relative_to(ROOT)}`",
            f"- `{(OUT_DIR / 'summary' / 'panels').relative_to(ROOT)}`",
            "",
            "## Notes",
            "- Machine trajectory and surrounding vehicles are read from the same highD-injected rollout cases used by the DIL backend.",
            "- If a complete `human_only` DIL log is missing for a case, the script falls back to the rollout human intention trajectory for the Human curve.",
            "- Authority plots include Reference Authority, RA-RLDM, TA-RLDM, and bidirectional trust `T_h` and `T_m`.",
        ]
    )
    (OUT_DIR / "DIL_figure_generation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    setup_style()
    cases: dict[int, dict[str, Any]] = {}
    logs_all: dict[int, dict[str, pd.DataFrame | None]] = {}
    sources: dict[int, dict[str, str]] = {}
    for paper_case_id, cfg in PAPER_CASES.items():
        case = load_extended_case(cfg)
        logs, src = load_logs(cfg)
        cases[paper_case_id] = case
        logs_all[paper_case_id] = logs
        sources[paper_case_id] = src
        plot_trajectory_case(cfg, case, logs)
    plot_summary(cases, logs_all)
    write_report(sources)
    print(f"Saved DIL figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
