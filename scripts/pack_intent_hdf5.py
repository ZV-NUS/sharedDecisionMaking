from pathlib import Path
import argparse
import json
import shutil
import tempfile
import zipfile

import h5py
import numpy as np


FLOAT_KEYS = [
    "ego_history",
    "neighbor_history",
    "neighbor_mask",
    "risk_history",
    "future_trajectory",
    "future_speed",
    "future_acceleration",
    "future_steer",
    "vehicle_length",
    "vehicle_width",
]

INT_KEYS = [
    "future_decision_sequence",
    "lane_change_count",
    "return_flag",
    "time_to_first_lane_change",
    "decision_label",
    "recording_id",
    "vehicle_id",
    "frame_id",
    "driving_direction",
    "current_lane_id",
]

STORE_DTYPES = {
    "future_decision_sequence": "i1",
    "lane_change_count": "i2",
    "return_flag": "i1",
    "time_to_first_lane_change": "i2",
    "decision_label": "i1",
    "recording_id": "i2",
    "vehicle_id": "i4",
    "frame_id": "i4",
    "driving_direction": "i1",
    "current_lane_id": "i2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack highD intent index splits into uncompressed HDF5 files.")
    parser.add_argument("--split-dir", default="data/processed/intent_splits")
    parser.add_argument("--output-dir", default="data/processed/intent_hdf5")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--temp-dir", default="data/processed/tmp_npz_extract")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    split_dir = root / args.split_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for split in args.splits:
        index_path = split_dir / f"{split}_index.npz"
        output_path = output_dir / f"{split}.h5"
        summaries[split] = pack_split(index_path, output_path, args.block_size, root / args.temp_dir)

    summary_path = output_dir / "hdf5_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f"Saved summary: {summary_path}")


def pack_split(index_path: Path, output_path: Path, block_size: int, temp_dir: Path) -> dict:
    index = np.load(index_path, allow_pickle=True)
    total = int(index["offsets"][-1])
    if total <= 0:
        raise ValueError(f"No samples in {index_path}")

    first_shard_id = int(np.flatnonzero(index["counts"] > 0)[0])
    first_idx = int(index["indices"][index["offsets"][first_shard_id]])
    with np.load(str(index["shards"][first_shard_id]), allow_pickle=True) as first:
        specs = _dataset_specs(first, first_idx, total)

    if output_path.exists():
        output_path.unlink()

    decision_counts = np.zeros(3, dtype=np.int64)
    return_samples = 0
    multi_lane_change_samples = 0
    written = 0
    with h5py.File(output_path, "w") as h5:
        h5.attrs["source_index"] = str(index_path)
        h5.attrs["num_samples"] = total
        h5.attrs["format"] = "highd_intent_hdf5_v1"
        for key, spec in specs.items():
            h5.create_dataset(
                key,
                shape=spec["shape"],
                dtype=spec["dtype"],
                chunks=spec["chunks"],
            )

        temp_dir.mkdir(parents=True, exist_ok=True)
        for shard_id, shard_path in enumerate(index["shards"].astype(str)):
            start = int(index["offsets"][shard_id])
            end = int(index["offsets"][shard_id + 1])
            if end <= start:
                continue
            selected = index["indices"][start:end].astype(np.int64)
            out_start = written
            out_end = written + selected.shape[0]
            for key in specs:
                values = _load_member_as_mmap(Path(shard_path), key, temp_dir)
                _write_selected(h5[key], values, selected, out_start, block_size, key)
                if hasattr(values, "_mmap") and values._mmap is not None:
                    values._mmap.close()
                extracted = temp_dir / f"{Path(shard_path).stem}_{key}.npy"
                extracted.unlink(missing_ok=True)

            labels = np.asarray(h5["decision_label"][out_start:out_end], dtype=np.int64)
            decision_counts += np.bincount(labels, minlength=3)
            return_samples += int(np.sum(h5["return_flag"][out_start:out_end]))
            multi_lane_change_samples += int(np.sum(h5["lane_change_count"][out_start:out_end] >= 2))
            written = out_end
            print(f"{output_path.name}: packed shard {shard_id + 1}/{len(index['shards'])}, samples={written}/{total}")

    size_gb = output_path.stat().st_size / (1024 ** 3)
    return {
        "output": str(output_path),
        "num_samples": total,
        "size_gb": round(size_gb, 2),
        "decision_counts": {"L": int(decision_counts[0]), "S": int(decision_counts[1]), "R": int(decision_counts[2])},
        "return_samples": int(return_samples),
        "multi_lane_change_samples": int(multi_lane_change_samples),
    }


def _load_member_as_mmap(npz_path: Path, key: str, temp_dir: Path) -> np.ndarray:
    member = f"{key}.npy"
    target = temp_dir / f"{npz_path.stem}_{key}.npy"
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(npz_path, "r") as zf:
        with zf.open(member, "r") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024 * 16)
    return np.load(target, mmap_mode="r")


def _write_selected(dataset: h5py.Dataset, values: np.ndarray, selected: np.ndarray, out_start: int, block_size: int, key: str) -> None:
    for block_start in range(0, selected.shape[0], block_size):
        block_indices = selected[block_start : block_start + block_size]
        out0 = out_start + block_start
        out1 = out0 + block_indices.shape[0]
        if _is_contiguous(block_indices):
            block = values[int(block_indices[0]) : int(block_indices[-1]) + 1]
        else:
            block = values[block_indices]
        if key in STORE_DTYPES:
            block = block.astype(STORE_DTYPES[key], copy=False)
        else:
            block = block.astype(np.float32, copy=False)
        dataset[out0:out1] = block


def _is_contiguous(indices: np.ndarray) -> bool:
    return indices.size == 0 or (int(indices[-1]) - int(indices[0]) + 1 == indices.size and np.all(np.diff(indices) == 1))


def _dataset_specs(first: np.lib.npyio.NpzFile, first_idx: int, total: int) -> dict:
    specs = {}
    for key in FLOAT_KEYS + INT_KEYS:
        sample = first[key][first_idx]
        shape = (total,) + tuple(np.asarray(sample).shape)
        dtype = np.dtype(STORE_DTYPES.get(key, "f4"))
        chunk0 = min(1024, total)
        chunks = (chunk0,) + tuple(np.asarray(sample).shape)
        specs[key] = {"shape": shape, "dtype": dtype, "chunks": chunks}
    return specs


if __name__ == "__main__":
    main()
