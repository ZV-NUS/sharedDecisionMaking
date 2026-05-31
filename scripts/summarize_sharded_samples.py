from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize sharded highD sample npz files.")
    parser.add_argument("--sample-dir", default="data/processed/highd_base_samples_all_frame_label")
    parser.add_argument("--out-dir", default="data/processed/inspection_all_frame_label")
    parser.add_argument("--max-values-per-shard", type=int, default=50000)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sample_dir = root / args.sample_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(sample_dir.glob("*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No npz shards found under {sample_dir}")

    rng = np.random.default_rng(2026)
    total_samples = 0
    decision_counts = np.zeros(3, dtype=np.int64)
    future_decision_counts = np.zeros(3, dtype=np.int64)
    return_samples = 0
    multi_lane_change_samples = 0
    time_to_first = []
    value_samples = {
        "future_speed": [],
        "future_acceleration": [],
        "future_steer_deg": [],
        "future_rel_x": [],
        "future_rel_y": [],
        "risk_front": [],
    }
    recording_summaries = []

    for shard_path in shard_paths:
        data = np.load(shard_path, allow_pickle=True)
        labels = data["decision_label"]
        n = int(labels.shape[0])
        total_samples += n
        decision_counts += np.bincount(labels, minlength=3)
        future_decision_counts += np.bincount(data["future_decision_sequence"].reshape(-1), minlength=3)
        return_count = int(np.sum(data["return_flag"]))
        multi_count = int(np.sum(data["lane_change_count"] >= 2))
        return_samples += return_count
        multi_lane_change_samples += multi_count
        valid_t = data["time_to_first_lane_change"]
        valid_t = valid_t[valid_t >= 0]
        if valid_t.size:
            time_to_first.append(_sample_flat(valid_t, args.max_values_per_shard, rng))

        value_samples["future_speed"].append(_sample_flat(data["future_speed"], args.max_values_per_shard, rng))
        value_samples["future_acceleration"].append(_sample_flat(data["future_acceleration"], args.max_values_per_shard, rng))
        value_samples["future_steer_deg"].append(_sample_flat(np.rad2deg(data["future_steer"]), args.max_values_per_shard, rng))
        value_samples["future_rel_x"].append(_sample_flat(data["future_trajectory"][:, :, 0], args.max_values_per_shard, rng))
        value_samples["future_rel_y"].append(_sample_flat(data["future_trajectory"][:, :, 1], args.max_values_per_shard, rng))
        value_samples["risk_front"].append(_sample_flat(data["risk_history"][:, :, 3], args.max_values_per_shard, rng))

        recording_summaries.append(
            {
                "shard": str(shard_path),
                "num_samples": n,
                "decision_counts": _counts_dict(labels),
                "future_decision_frame_counts": _counts_dict(data["future_decision_sequence"].reshape(-1)),
                "return_samples": return_count,
                "multi_lane_change_samples": multi_count,
            }
        )

    sampled_values = {key: np.concatenate(chunks) for key, chunks in value_samples.items() if chunks}
    sampled_time = np.concatenate(time_to_first) if time_to_first else np.array([], dtype=np.float32)
    summary = {
        "global": {
            "num_shards": len(shard_paths),
            "num_samples": int(total_samples),
            "decision_counts": _counts_array(decision_counts),
            "future_decision_frame_counts": _counts_array(future_decision_counts),
            "return_samples": int(return_samples),
            "multi_lane_change_samples": int(multi_lane_change_samples),
            "sampled_time_to_first_lane_change_frames": _stats(sampled_time) if sampled_time.size else None,
            "sampled_future_speed_mps": _stats(sampled_values["future_speed"]),
            "sampled_future_acceleration_mps2": _stats(sampled_values["future_acceleration"]),
            "sampled_future_steer_deg": _stats(sampled_values["future_steer_deg"]),
            "sampled_future_rel_x_m": _stats(sampled_values["future_rel_x"]),
            "sampled_future_rel_y_m": _stats(sampled_values["future_rel_y"]),
            "sampled_risk_front": _stats(sampled_values["risk_front"]),
        },
        "shards": [str(p) for p in shard_paths],
        "recordings": recording_summaries,
    }

    summary_path = out_dir / "all_frame_label_summary.json"
    manifest_path = sample_dir / "highd_base_samples_all_frame_label.full_manifest.json"
    for path in [summary_path, manifest_path]:
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    _plot_distributions(out_dir, decision_counts, future_decision_counts, sampled_values)
    print(json.dumps(summary["global"], indent=2, ensure_ascii=False))
    print(f"Saved summary: {summary_path}")
    print(f"Saved full manifest: {manifest_path}")


def _sample_flat(values: np.ndarray, max_values: int, rng: np.random.Generator) -> np.ndarray:
    flat = values.reshape(-1)
    if flat.size <= max_values:
        return flat.astype(np.float32)
    idx = rng.choice(flat.size, size=max_values, replace=False)
    return flat[idx].astype(np.float32)


def _counts_dict(labels: np.ndarray) -> dict[str, int]:
    return _counts_array(np.bincount(labels.astype(np.int64), minlength=3))


def _counts_array(counts: np.ndarray) -> dict[str, int]:
    return {"L": int(counts[0]), "S": int(counts[1]), "R": int(counts[2])}


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def _plot_distributions(
    out_dir: Path,
    decision_counts: np.ndarray,
    future_decision_counts: np.ndarray,
    values: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    names = ["L", "S", "R"]
    axes[0, 0].bar(names, decision_counts, color=["#386cb0", "#7fc97f", "#f0027f"])
    axes[0, 0].set_title("Window decision counts")
    axes[0, 1].bar(names, future_decision_counts, color=["#386cb0", "#7fc97f", "#f0027f"])
    axes[0, 1].set_title("Frame decision counts")
    axes[0, 2].hist(values["future_speed"], bins=80, color="#386cb0")
    axes[0, 2].set_title("Future speed / mps")
    axes[1, 0].hist(values["future_acceleration"], bins=80, color="#7fc97f")
    axes[1, 0].set_title("Future acceleration / mps2")
    axes[1, 1].hist(values["future_steer_deg"], bins=80, color="#f0027f")
    axes[1, 1].set_title("Pseudo steering / deg")
    axes[1, 2].hist(values["future_rel_y"], bins=80, color="#bf5b17")
    axes[1, 2].set_title("Future lateral displacement / m")
    fig.tight_layout()
    fig.savefig(out_dir / "all_basic_distributions.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
