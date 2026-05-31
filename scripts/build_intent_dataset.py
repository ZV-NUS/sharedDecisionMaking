from pathlib import Path
import argparse
import json
import sys

import numpy as np
import yaml


DECISION_NAMES = np.array(["L", "S", "R"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test index splits for highD intent prediction.")
    parser.add_argument("--config", default="configs/build_intent_dataset.yaml")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    with (root / args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sample_dir = root / config["sample_dir"]
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))

    shard_paths = _discover_shards(sample_dir)
    split_recordings = {
        "train": _normalize_recordings(config["split"]["train_recordings"]),
        "val": _normalize_recordings(config["split"]["val_recordings"]),
        "test": _normalize_recordings(config["split"]["test_recordings"]),
    }

    global_summary = {
        "sample_dir": str(sample_dir),
        "output_dir": str(output_dir),
        "quality_filter": config["quality_filter"],
        "balanced_train": config["balanced_train"],
        "splits": {},
    }

    raw_split_indices: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    for split_name, rec_ids in split_recordings.items():
        for rec_id in rec_ids:
            shard_path = shard_paths.get(rec_id)
            if shard_path is None:
                print(f"Warning: missing shard for recording {rec_id}", file=sys.stderr)
                continue
            data = np.load(shard_path, allow_pickle=True)
            labels = data["decision_label"]
            quality_mask = _quality_mask(data, config["quality_filter"])
            valid_idx = np.where(quality_mask)[0].astype(np.int64)
            raw_split_indices[split_name].append(
                {
                    "recording": rec_id,
                    "shard": str(shard_path),
                    "indices": valid_idx,
                    "num_raw_samples": int(labels.shape[0]),
                    "num_valid_samples": int(valid_idx.shape[0]),
                    "decision_counts_valid": _counts(labels[valid_idx]),
                    "return_samples_valid": int(np.sum(data["return_flag"][valid_idx])),
                    "multi_lane_change_samples_valid": int(np.sum(data["lane_change_count"][valid_idx] >= 2)),
                }
            )

    for split_name, entries in raw_split_indices.items():
        saved_entries = entries
        if split_name == "train" and config["balanced_train"].get("enabled", True):
            saved_entries = _balance_train_entries(entries, rng, float(config["balanced_train"]["straight_to_lane_change_ratio"]))
        split_path = output_dir / f"{split_name}_index.npz"
        _save_index(split_path, saved_entries)
        global_summary["splits"][split_name] = _summarize_entries(saved_entries)
        global_summary["splits"][split_name]["index_file"] = str(split_path)

    raw_train_path = output_dir / "train_raw_index.npz"
    _save_index(raw_train_path, raw_split_indices["train"])
    global_summary["splits"]["train_raw"] = _summarize_entries(raw_split_indices["train"])
    global_summary["splits"]["train_raw"]["index_file"] = str(raw_train_path)

    summary_path = output_dir / "intent_split_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(global_summary["splits"], indent=2, ensure_ascii=False))
    print(f"Saved summary: {summary_path}")


def _discover_shards(sample_dir: Path) -> dict[str, Path]:
    shards = {}
    for path in sorted(sample_dir.glob("*.npz")):
        rec_id = path.stem.split("_")[-1]
        if rec_id.isdigit():
            shards[rec_id.zfill(2)] = path
    return shards


def _normalize_recordings(values: list) -> list[str]:
    return [str(v).zfill(2) for v in values]


def _quality_mask(data: np.lib.npyio.NpzFile, cfg: dict) -> np.ndarray:
    min_speed = float(cfg["min_speed_mps"])
    max_abs_steer = float(cfg["max_abs_steer_rad"])
    hist_speed_ok = np.min(data["ego_history"][:, :, 6], axis=1) >= min_speed
    fut_speed_ok = np.min(data["future_speed"], axis=1) >= min_speed
    steer_ok = np.max(np.abs(data["future_steer"]), axis=1) <= max_abs_steer
    return hist_speed_ok & fut_speed_ok & steer_ok


def _balance_train_entries(entries: list[dict], rng: np.random.Generator, straight_ratio: float) -> list[dict]:
    labels_all = []
    entry_ids = []
    local_indices = []
    for entry_id, entry in enumerate(entries):
        data = np.load(entry["shard"], allow_pickle=True)
        idx = entry["indices"]
        labels = data["decision_label"][idx]
        labels_all.append(labels)
        entry_ids.append(np.full(idx.shape[0], entry_id, dtype=np.int32))
        local_indices.append(idx)
    labels_all = np.concatenate(labels_all)
    entry_ids = np.concatenate(entry_ids)
    local_indices = np.concatenate(local_indices)

    lr_mask = labels_all != 1
    s_mask = labels_all == 1
    lr_count = int(np.sum(lr_mask))
    target_s = min(int(np.sum(s_mask)), int(round(lr_count * straight_ratio)))
    selected_s_positions = rng.choice(np.where(s_mask)[0], size=target_s, replace=False) if target_s > 0 else np.array([], dtype=np.int64)
    selected_positions = np.concatenate([np.where(lr_mask)[0], selected_s_positions])
    rng.shuffle(selected_positions)

    balanced_entries = []
    for entry_id, entry in enumerate(entries):
        keep = selected_positions[entry_ids[selected_positions] == entry_id]
        selected_idx = local_indices[keep]
        selected_idx.sort()
        data = np.load(entry["shard"], allow_pickle=True)
        labels = data["decision_label"][selected_idx]
        balanced = dict(entry)
        balanced["indices"] = selected_idx.astype(np.int64)
        balanced["num_valid_samples"] = int(selected_idx.shape[0])
        balanced["decision_counts_valid"] = _counts(labels)
        balanced["return_samples_valid"] = int(np.sum(data["return_flag"][selected_idx]))
        balanced["multi_lane_change_samples_valid"] = int(np.sum(data["lane_change_count"][selected_idx] >= 2))
        balanced_entries.append(balanced)
    return balanced_entries


def _save_index(path: Path, entries: list[dict]) -> None:
    shards = np.array([entry["shard"] for entry in entries])
    recordings = np.array([entry["recording"] for entry in entries])
    counts = np.array([entry["indices"].shape[0] for entry in entries], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    indices = np.concatenate([entry["indices"] for entry in entries]).astype(np.int64) if entries else np.array([], dtype=np.int64)
    np.savez_compressed(path, shards=shards, recordings=recordings, counts=counts, offsets=offsets, indices=indices)


def _summarize_entries(entries: list[dict]) -> dict:
    counts = np.zeros(3, dtype=np.int64)
    raw = 0
    valid = 0
    returns = 0
    multi = 0
    for entry in entries:
        raw += int(entry["num_raw_samples"])
        valid += int(entry["num_valid_samples"])
        returns += int(entry["return_samples_valid"])
        multi += int(entry["multi_lane_change_samples_valid"])
        counts += np.array([
            entry["decision_counts_valid"]["L"],
            entry["decision_counts_valid"]["S"],
            entry["decision_counts_valid"]["R"],
        ], dtype=np.int64)
    return {
        "num_recordings": len(entries),
        "num_raw_samples": raw,
        "num_samples": valid,
        "decision_counts": {"L": int(counts[0]), "S": int(counts[1]), "R": int(counts[2])},
        "return_samples": returns,
        "multi_lane_change_samples": multi,
    }


def _counts(labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels.astype(np.int64), minlength=3)
    return {"L": int(counts[0]), "S": int(counts[1]), "R": int(counts[2])}


if __name__ == "__main__":
    main()
