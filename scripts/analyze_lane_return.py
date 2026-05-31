from pathlib import Path
import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Find A-B-A lane-change candidates in highD.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_dir = root / args.data_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    examples = []
    for meta_path in sorted(data_dir.glob("*_tracksMeta.csv")):
        rec = meta_path.name.split("_")[0]
        tracks_path = data_dir / f"{rec}_tracks.csv"
        if not tracks_path.exists():
            continue

        meta = pd.read_csv(meta_path)
        multi = meta[meta["numLaneChanges"] >= 2]
        if multi.empty:
            rows.append(_row(rec, 0, 0, 0, 0))
            continue

        tracks = pd.read_csv(
            tracks_path,
            usecols=["id", "frame", "laneId", "xVelocity", "precedingId", "dhw", "thw", "ttc"],
        )
        aba = 0
        car_aba = 0
        truck_aba = 0

        for _, meta_row in multi.iterrows():
            vehicle_id = int(meta_row["id"])
            vt = tracks[tracks["id"] == vehicle_id].sort_values("frame")
            compressed_lanes, compressed_frames = _compress_lane_sequence(vt)
            aba_hit = _find_aba(compressed_lanes)
            if aba_hit is None:
                continue

            aba += 1
            if meta_row["class"] == "Car":
                car_aba += 1
            if meta_row["class"] == "Truck":
                truck_aba += 1

            if len(examples) < args.max_examples:
                i = aba_hit
                first_change = compressed_frames[i + 1]
                return_change = compressed_frames[i + 2]
                before = vt[vt["frame"] < first_change].tail(25)
                middle = vt[(vt["frame"] >= first_change) & (vt["frame"] < return_change)]
                after = vt[vt["frame"] >= return_change].head(25)
                examples.append(
                    {
                        "recording": rec,
                        "vehicle_id": vehicle_id,
                        "class": meta_row["class"],
                        "drivingDirection": int(meta_row["drivingDirection"]),
                        "numLaneChanges": int(meta_row["numLaneChanges"]),
                        "lane_pattern": "->".join(map(str, compressed_lanes)),
                        "change_frames": ";".join(map(str, compressed_frames)),
                        "first_change_frame": int(first_change),
                        "return_change_frame": int(return_change),
                        "middle_duration_s": float((return_change - first_change) / 25.0),
                        "mean_abs_speed_before": _mean_abs_speed(before),
                        "mean_abs_speed_middle": _mean_abs_speed(middle),
                        "mean_abs_speed_after": _mean_abs_speed(after),
                        "min_dhw_before": _positive_min(before, "dhw"),
                        "min_thw_before": _positive_min(before, "thw"),
                        "min_ttc_before": _positive_min(before, "ttc"),
                    }
                )

        rows.append(_row(rec, len(multi), aba, car_aba, truck_aba))

    summary = pd.DataFrame(rows)
    examples_df = pd.DataFrame(examples)
    summary_path = out_dir / "aba_lane_change_candidates.csv"
    examples_path = out_dir / "aba_lane_change_examples.csv"
    summary.to_csv(summary_path, index=False)
    examples_df.to_csv(examples_path, index=False)

    print("Total:")
    print(summary.sum(numeric_only=True).to_string())
    print("\nTop recordings:")
    print(summary.sort_values("aba_candidates", ascending=False).head(10).to_string(index=False))
    print("\nExamples:")
    print(examples_df.head(10).to_string(index=False))
    print(f"\nSaved {summary_path}")
    print(f"Saved {examples_path}")


def _row(rec: str, ge2: int, aba: int, car_aba: int, truck_aba: int) -> dict:
    return {
        "recording": rec,
        "vehicles_ge2_lane_changes": int(ge2),
        "aba_candidates": int(aba),
        "car_aba": int(car_aba),
        "truck_aba": int(truck_aba),
    }


def _compress_lane_sequence(vehicle_track: pd.DataFrame) -> tuple[list[int], list[int]]:
    lanes = vehicle_track["laneId"].to_numpy()
    frames = vehicle_track["frame"].to_numpy()
    if len(lanes) == 0:
        return [], []
    compressed_lanes = [int(lanes[0])]
    compressed_frames = [int(frames[0])]
    for lane, frame in zip(lanes[1:], frames[1:]):
        lane = int(lane)
        if lane != compressed_lanes[-1]:
            compressed_lanes.append(lane)
            compressed_frames.append(int(frame))
    return compressed_lanes, compressed_frames


def _find_aba(lanes: list[int]) -> int | None:
    for i in range(len(lanes) - 2):
        if lanes[i] == lanes[i + 2] and abs(lanes[i + 1] - lanes[i]) == 1:
            return i
    return None


def _mean_abs_speed(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    return float(df["xVelocity"].abs().mean())


def _positive_min(df: pd.DataFrame, column: str) -> float | None:
    if df.empty:
        return None
    values = df[column]
    values = values[values > 0]
    if values.empty:
        return None
    return float(values.min())


if __name__ == "__main__":
    main()
