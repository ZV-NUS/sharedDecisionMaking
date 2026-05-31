from pathlib import Path
import argparse
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.intent_dataset import HighDIntentDataset, create_intent_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test highD intent Dataset/DataLoader.")
    parser.add_argument("--index", default="data/processed/intent_splits/train_index.npz")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    index_path = ROOT / args.index
    dataset = HighDIntentDataset(index_path, include_meta=True)
    print(f"dataset length: {len(dataset):,}")
    one = dataset[0]
    print("single sample:")
    _print_tree(one)
    dataset.close()

    loader = create_intent_dataloader(
        index_path,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        include_meta=False,
    )
    start = time.time()
    for batch_id, batch in enumerate(loader):
        print(f"\nbatch {batch_id}:")
        _print_tree(batch)
        if batch_id + 1 >= args.batches:
            break
    elapsed = time.time() - start
    print(f"\nread {args.batches} batches in {elapsed:.2f}s")


def _print_tree(obj, prefix: str = "") -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _print_tree(value, f"{prefix}{key}.")
        return
    shape = tuple(obj.shape) if hasattr(obj, "shape") else ""
    dtype = getattr(obj, "dtype", type(obj).__name__)
    print(f"  {prefix[:-1]}: shape={shape} dtype={dtype}")


if __name__ == "__main__":
    main()
