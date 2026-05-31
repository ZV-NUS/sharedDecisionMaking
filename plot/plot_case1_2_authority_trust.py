"""Plot authority and bidirectional trust sequences for paper Cases 1-4.

Source data:
    outputs/shared_authority_validation/shared_authority_rollouts.js

Case mapping:
    Paper Case 1 <- rollout validation case 2
    Paper Case 2 <- rollout validation case 3
    Paper Case 3 <- rollout validation case 6
    Paper Case 4 <- rollout validation case 7

The figures are saved independently for IEEE-style manuscript use.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_case1_2_trajectory import apply_video_aligned_end, critical_slice, get_time_vector


ROOT = Path(__file__).resolve().parents[1]
ROLLOUT_JS = ROOT / "outputs" / "shared_authority_validation" / "shared_authority_rollouts.js"
OUT_DIR = ROOT / "results" / "Case1_2_Trajectory_Figures" / "authority_trust"

PAPER_CASES = {
    1: 2,
    2: 3,
    3: 6,
    4: 7,
}

STYLE = {
    # Compact size for an IEEE Transactions 1 x 3 subfigure layout.
    # A full double-column figure is typically about 7.0 in wide, so one
    # subfigure should stay near 2.2-2.4 in.
    "figsize": (2.45, 1.78),
    "dpi": 600,
    "font_family": ["Times New Roman", "DejaVu Serif"],
    "font_size": 7.7,
    "label_size": 7.8,
    "tick_size": 7.0,
    "legend_size": 6.3,
    "line_width": 0.82,
    "grid_width": 0.35,
}

COLORS = {
    "authority_ref": "#8B5CF6",
    "authority_ra": "#7B61FF",
    "authority_ta": "#D62728",
    "trust_mh": "#1F77B4",
    "trust_hm": "#FF7F0E",
    "urgency": "#6B7280",
}


def load_rollouts() -> dict:
    text = ROLLOUT_JS.read_text(encoding="utf-8").strip()
    prefix = "window.SHARED_AUTHORITY_ROLLOUTS = "
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if text.endswith(";"):
        text = text[:-1]
    return json.loads(text)


def find_case(data: dict, rollout_case_id: int) -> dict:
    for case in data["cases"]:
        if int(case["record"]["case_id"]) == int(rollout_case_id):
            return case
    raise ValueError(f"Cannot find rollout case_id={rollout_case_id}")


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


def arr(case: dict, key: str, win: slice) -> np.ndarray:
    values = np.asarray(case["signals"][key], dtype=float)
    return values[win]


def plot_case(paper_case_id: int, case: dict) -> None:
    # Use exactly the same visible maneuver window and relative-time convention
    # as plot_case1_2_trajectory.py, so the authority/trust time axis matches
    # the keyframe trajectory figure.
    win = apply_video_aligned_end(case, critical_slice(case), paper_case_id)
    time_vec = get_time_vector(case)
    start = 0 if win.start is None else int(win.start)
    t = time_vec[win] - float(time_vec[start])

    authority_ref = arr(case, "authority_ref", win)
    authority_ra = arr(case, "authority_ra", win)
    authority_ta = arr(case, "authority_rl", win)
    trust_mh = arr(case, "trust_machine_to_human", win)
    trust_hm = arr(case, "trust_human_to_machine", win)

    fig, ax = plt.subplots(figsize=STYLE["figsize"])
    ax_trust = ax.twinx()

    # Authority sequences.
    authority_lines = []
    trust_lines = []
    authority_lines += ax.plot(
        t,
        authority_ra,
        color=COLORS["authority_ra"],
        linestyle=(0, (4, 1, 1, 1)),
        linewidth=STYLE["line_width"],
        label="RA-RLDM",
    )
    authority_lines += ax.plot(
        t,
        authority_ta,
        color=COLORS["authority_ta"],
        linestyle="-",
        linewidth=STYLE["line_width"] + 0.15,
        label="TA-RLDM",
    )
    authority_lines += ax.plot(
        t,
        authority_ref,
        color=COLORS["authority_ref"],
        linestyle=":",
        linewidth=STYLE["line_width"],
        label="Reference Authority",
    )

    # Bidirectional trust.
    trust_lines += ax_trust.plot(
        t,
        trust_mh,
        color=COLORS["trust_mh"],
        linestyle="--",
        linewidth=STYLE["line_width"],
        label=r"$T_m$",
    )
    trust_lines += ax_trust.plot(
        t,
        trust_hm,
        color=COLORS["trust_hm"],
        linestyle="-.",
        linewidth=STYLE["line_width"],
        label=r"$T_h$",
    )

    ax.set_xlim(float(t[0]), float(t[-1]))
    ax.set_ylim(-0.03, 1.03)
    ax_trust.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Time (s)", fontsize=STYLE["label_size"])
    ax.set_ylabel("Authority Level", fontsize=STYLE["label_size"])
    ax_trust.set_ylabel("Trust Level", fontsize=STYLE["label_size"])
    ax.tick_params(axis="both", labelsize=STYLE["tick_size"], direction="in")
    ax_trust.tick_params(axis="y", labelsize=STYLE["tick_size"], direction="in")
    ax.grid(True, color="#D1D5DB", linewidth=STYLE["grid_width"], alpha=0.65)
    handles = authority_lines + trust_lines
    labels = [h.get_label() for h in handles]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.31),
        ncol=3,
        frameon=False,
        fontsize=STYLE["legend_size"],
        handlelength=1.35,
        columnspacing=0.55,
        labelspacing=0.28,
    )

    fig.subplots_adjust(left=0.17, right=0.83, bottom=0.22, top=0.74)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"Case{paper_case_id}_Authority_Trust_Sequences"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"{stem}.{ext}",
            dpi=STYLE["dpi"] if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.018,
        )
    plt.close(fig)


def main() -> None:
    setup_style()
    data = load_rollouts()
    for paper_case_id, rollout_case_id in PAPER_CASES.items():
        plot_case(paper_case_id, find_case(data, rollout_case_id))
    print(f"Saved authority/trust figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
