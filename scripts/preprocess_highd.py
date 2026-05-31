from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.highd_preprocess import load_config, preprocess_highd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reusable highD samples for intent prediction and RL scene setup.")
    parser.add_argument("--config", default="configs/preprocess_highd.yaml", help="Path to preprocessing config.")
    parser.add_argument("--recordings", nargs="*", default=None, help="Optional recording ids, e.g. 01 02 15.")
    parser.add_argument("--output-name", default=None, help="Override output npz filename.")
    parser.add_argument("--max-samples-per-recording", type=int, default=None, help="Override debug cap per recording.")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    if args.recordings is not None:
        config["recordings"] = args.recordings
    if args.output_name is not None:
        config["output_name"] = args.output_name
    if args.max_samples_per_recording is not None:
        config["sampling"]["max_samples_per_recording"] = args.max_samples_per_recording

    preprocess_highd(config, ROOT)


if __name__ == "__main__":
    main()
