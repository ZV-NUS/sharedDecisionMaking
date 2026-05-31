from pathlib import Path
import argparse

import numpy as np


KEYS = [
    "ego_history",
    "neighbor_history",
    "neighbor_mask",
    "risk_history",
    "future_trajectory",
    "future_speed",
    "future_acceleration",
    "future_steer",
    "future_decision_sequence",
    "lane_change_count",
    "return_flag",
    "time_to_first_lane_change",
    "decision_label",
    "recording_id",
    "vehicle_id",
    "frame_id",
    "driving_direction",
    "vehicle_length",
    "vehicle_width",
    "current_lane_id",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack a small uncompressed subset for fast model smoke tests.")
    parser.add_argument("--index", default="data/processed/intent_splits/train_index.npz")
    parser.add_argument("--output", default="data/processed/debug_intent/train_debug_4096.npz")
    parser.add_argument("--num-samples", type=int, default=4096)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    index = np.load(root / args.index, allow_pickle=True)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    remaining = int(args.num_samples)
    chunks = {key: [] for key in KEYS}
    shard_paths = index["shards"].astype(str)
    counts = index["counts"]
    offsets = index["offsets"]
    all_indices = index["indices"]

    for shard_id, shard_path in enumerate(shard_paths):
        if remaining <= 0:
            break
        start, end = int(offsets[shard_id]), int(offsets[shard_id + 1])
        local_indices = all_indices[start:end]
        if local_indices.size == 0:
            continue
        take = min(remaining, int(local_indices.size))
        selected = local_indices[:take]
        with np.load(shard_path, allow_pickle=True) as shard:
            for key in KEYS:
                chunks[key].append(shard[key][selected])
        remaining -= take

    arrays = {key: np.concatenate(values, axis=0) for key, values in chunks.items() if values}
    np.savez(output, **arrays)

    debug_index = output.with_name(output.stem + "_index.npz")
    np.savez_compressed(
        debug_index,
        shards=np.array([str(output)]),
        recordings=np.array(["debug"]),
        counts=np.array([arrays["decision_label"].shape[0]], dtype=np.int64),
        offsets=np.array([0, arrays["decision_label"].shape[0]], dtype=np.int64),
        indices=np.arange(arrays["decision_label"].shape[0], dtype=np.int64),
    )
    counts = np.bincount(arrays["decision_label"].astype(np.int64), minlength=3)
    print(f"Saved {output}")
    print(f"Saved {debug_index}")
    print({"num_samples": int(arrays["decision_label"].shape[0]), "L": int(counts[0]), "S": int(counts[1]), "R": int(counts[2])})


if __name__ == "__main__":
    main()
