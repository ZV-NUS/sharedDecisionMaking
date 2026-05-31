from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect generated highD samples.")
    parser.add_argument("--samples", default="data/processed/highd_base_samples_full_01.npz")
    parser.add_argument("--out-dir", default="data/processed/inspection")
    parser.add_argument("--num-examples", type=int, default=3)
    args = parser.parse_args()

    sample_path = ROOT / args.samples
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(sample_path, allow_pickle=True)
    labels = data["decision_label"]
    decision_names = data["decision_names"]
    future_speed = data["future_speed"]
    future_acc = data["future_acceleration"]
    future_steer = data["future_steer"]
    future_traj = data["future_trajectory"]
    ego_hist = data["ego_history"]
    neighbor_mask = data["neighbor_mask"]
    risk_hist = data["risk_history"]
    future_decision_sequence = data["future_decision_sequence"] if "future_decision_sequence" in data.files else None

    summary = {
        "sample_file": str(sample_path),
        "num_samples": int(labels.shape[0]),
        "shapes": {k: list(data[k].shape) for k in [
            "ego_history",
            "neighbor_history",
            "neighbor_mask",
            "risk_history",
            "future_trajectory",
            "future_speed",
            "future_acceleration",
            "future_steer",
            *(['future_decision_sequence'] if "future_decision_sequence" in data.files else []),
        ]},
        "decision_counts": {
            str(decision_names[i]): int(np.sum(labels == i))
            for i in range(len(decision_names))
        },
        "future_decision_frame_counts": {
            str(decision_names[i]): int(np.sum(future_decision_sequence == i))
            for i in range(len(decision_names))
        } if future_decision_sequence is not None else None,
        "return_samples": int(np.sum(data["return_flag"])) if "return_flag" in data.files else None,
        "multi_lane_change_samples": int(np.sum(data["lane_change_count"] >= 2)) if "lane_change_count" in data.files else None,
        "time_to_first_lane_change_frames": _stats(data["time_to_first_lane_change"][data["time_to_first_lane_change"] >= 0])
        if "time_to_first_lane_change" in data.files and np.any(data["time_to_first_lane_change"] >= 0)
        else None,
        "future_speed_mps": _stats(future_speed),
        "future_acceleration_mps2": _stats(future_acc),
        "future_steer_rad": _stats(future_steer),
        "future_steer_deg": _stats(np.rad2deg(future_steer)),
        "future_rel_x_m": _stats(future_traj[:, :, 0]),
        "future_rel_y_m": _stats(future_traj[:, :, 1]),
        "ego_history_speed_mps": _stats(ego_hist[:, :, 6]),
        "risk_front": _stats(risk_hist[:, :, 3]),
        "neighbor_presence_ratio": float(np.mean(neighbor_mask)),
    }

    with (out_dir / "inspection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _plot_histograms(out_dir, labels, decision_names, future_speed, future_acc, future_steer)
    _plot_decision_examples(out_dir, data, args.num_examples)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved inspection plots to: {out_dir}")


def _stats(values: np.ndarray) -> dict[str, float]:
    flat = values.reshape(-1)
    return {
        "min": float(np.min(flat)),
        "p01": float(np.percentile(flat, 1)),
        "mean": float(np.mean(flat)),
        "p50": float(np.percentile(flat, 50)),
        "p99": float(np.percentile(flat, 99)),
        "max": float(np.max(flat)),
    }


def _plot_histograms(
    out_dir: Path,
    labels: np.ndarray,
    decision_names: np.ndarray,
    future_speed: np.ndarray,
    future_acc: np.ndarray,
    future_steer: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    counts = [np.sum(labels == i) for i in range(len(decision_names))]
    axes[0, 0].bar(decision_names, counts, color=["#386cb0", "#7fc97f", "#f0027f"])
    axes[0, 0].set_title("Decision label counts")
    axes[0, 0].set_ylabel("samples")

    axes[0, 1].hist(future_speed.reshape(-1), bins=80, color="#386cb0")
    axes[0, 1].set_title("Future speed")
    axes[0, 1].set_xlabel("m/s")

    axes[1, 0].hist(future_acc.reshape(-1), bins=80, color="#7fc97f")
    axes[1, 0].set_title("Future acceleration")
    axes[1, 0].set_xlabel("m/s^2")

    axes[1, 1].hist(np.rad2deg(future_steer.reshape(-1)), bins=80, color="#f0027f")
    axes[1, 1].set_title("Pseudo steering")
    axes[1, 1].set_xlabel("deg")

    fig.tight_layout()
    fig.savefig(out_dir / "basic_distributions.png", dpi=160)
    plt.close(fig)


def _plot_decision_examples(out_dir: Path, data: np.lib.npyio.NpzFile, num_examples: int) -> None:
    labels = data["decision_label"]
    decision_names = data["decision_names"]
    future_traj = data["future_trajectory"]
    ego_hist = data["ego_history"]
    future_steer = data["future_steer"]
    future_speed = data["future_speed"]
    recording_id = data["recording_id"]
    vehicle_id = data["vehicle_id"]
    frame_id = data["frame_id"]

    for label_id, name in enumerate(decision_names):
        idxs = np.where(labels == label_id)[0][:num_examples]
        if len(idxs) == 0:
            continue
        fig, axes = plt.subplots(len(idxs), 3, figsize=(13, 3.4 * len(idxs)))
        if len(idxs) == 1:
            axes = np.expand_dims(axes, 0)
        for row, idx in enumerate(idxs):
            hist_xy = ego_hist[idx, :, :2]
            fut_xy = future_traj[idx]
            axes[row, 0].plot(hist_xy[:, 0], hist_xy[:, 1], label="history")
            axes[row, 0].plot(fut_xy[:, 0], fut_xy[:, 1], label="future")
            axes[row, 0].scatter([0], [0], s=20, color="black", label="anchor")
            axes[row, 0].set_title(
                f"{name} rec={recording_id[idx]} veh={vehicle_id[idx]} frame={frame_id[idx]}"
            )
            axes[row, 0].set_xlabel("relative x / m")
            axes[row, 0].set_ylabel("relative y / m")
            axes[row, 0].legend(fontsize=8)

            axes[row, 1].plot(np.rad2deg(future_steer[idx]))
            axes[row, 1].set_title("future pseudo steering")
            axes[row, 1].set_xlabel("future frame")
            axes[row, 1].set_ylabel("deg")

            axes[row, 2].plot(future_speed[idx])
            axes[row, 2].set_title("future speed")
            axes[row, 2].set_xlabel("future frame")
            axes[row, 2].set_ylabel("m/s")

        fig.tight_layout()
        fig.savefig(out_dir / f"examples_{name}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
