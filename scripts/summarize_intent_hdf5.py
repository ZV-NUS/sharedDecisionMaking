from pathlib import Path
import argparse
import json

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize packed highD intent HDF5 files.")
    parser.add_argument("--hdf5-dir", default="data/processed/intent_hdf5")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    hdf5_dir = root / args.hdf5_dir
    summary = {}
    for split in args.splits:
        path = hdf5_dir / f"{split}.h5"
        with h5py.File(path, "r") as h5:
            labels = h5["decision_label"][:].astype(np.int64)
            counts = np.bincount(labels, minlength=3)
            summary[split] = {
                "path": str(path),
                "size_gb": round(path.stat().st_size / (1024 ** 3), 2),
                "num_samples": int(h5.attrs["num_samples"]),
                "decision_counts": {"L": int(counts[0]), "S": int(counts[1]), "R": int(counts[2])},
                "return_samples": int(h5["return_flag"][:].sum()),
                "multi_lane_change_samples": int((h5["lane_change_count"][:] >= 2).sum()),
            }

    summary_path = hdf5_dir / "hdf5_full_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
