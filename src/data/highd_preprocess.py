from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - fallback for minimal environments.
    savgol_filter = None


DECISION_TO_ID = {"L": 0, "S": 1, "R": 2}
NEIGHBOR_FEATURES = ("rel_x", "rel_y", "rel_vx", "rel_vy", "length", "width")
EGO_FEATURES = ("rel_x", "rel_y", "vx", "vy", "ax", "ay", "speed", "lane_id")
RISK_FEATURES = ("dhw", "thw", "ttc", "front_risk")


@dataclass(frozen=True)
class RecordingPaths:
    recording_id: str
    tracks: Path
    tracks_meta: Path
    recording_meta: Path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_highd(config: dict[str, Any], root: Path) -> None:
    raw_dir = root / config["raw_data_dir"]
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = _discover_recordings(raw_dir, config.get("recordings") or [])
    if not paths:
        raise FileNotFoundError(f"No highD recordings found in {raw_dir}")

    if config.get("save_per_recording", False):
        _preprocess_highd_sharded(config, output_dir, paths)
        return

    all_samples: dict[str, list[np.ndarray | int | float]] = {
        "ego_history": [],
        "neighbor_history": [],
        "neighbor_mask": [],
        "risk_history": [],
        "future_trajectory": [],
        "future_speed": [],
        "future_acceleration": [],
        "future_steer": [],
        "future_decision_sequence": [],
        "lane_change_count": [],
        "return_flag": [],
        "time_to_first_lane_change": [],
        "decision_label": [],
        "recording_id": [],
        "vehicle_id": [],
        "frame_id": [],
        "driving_direction": [],
        "vehicle_length": [],
        "vehicle_width": [],
        "current_lane_id": [],
    }
    summaries: list[dict[str, Any]] = []

    for rec_paths in paths:
        samples, summary = _process_recording(rec_paths, config)
        summaries.append(summary)
        for key, values in samples.items():
            all_samples[key].extend(values)

    arrays = _to_arrays(all_samples)
    output_path = output_dir / config["output_name"]
    np.savez_compressed(
        output_path,
        **arrays,
        decision_names=np.array(["L", "S", "R"]),
        ego_feature_names=np.array(EGO_FEATURES),
        neighbor_feature_names=np.array(NEIGHBOR_FEATURES),
        neighbor_slot_names=np.array(config["neighbors"]["slots"]),
        risk_feature_names=np.array(RISK_FEATURES),
    )

    summary = _build_global_summary(arrays, summaries, config)
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved samples: {output_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary["global"], indent=2, ensure_ascii=False))


def _preprocess_highd_sharded(
    config: dict[str, Any],
    output_dir: Path,
    paths: list[RecordingPaths],
) -> None:
    stem = Path(config["output_name"]).stem
    shard_dir = output_dir / stem
    shard_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    shard_paths: list[str] = []

    for rec_paths in paths:
        samples, rec_summary = _process_recording(rec_paths, config)
        arrays = _to_arrays(samples)
        shard_path = shard_dir / f"{stem}_{rec_paths.recording_id}.npz"
        np.savez_compressed(
            shard_path,
            **arrays,
            decision_names=np.array(["L", "S", "R"]),
            ego_feature_names=np.array(EGO_FEATURES),
            neighbor_feature_names=np.array(NEIGHBOR_FEATURES),
            neighbor_slot_names=np.array(config["neighbors"]["slots"]),
            risk_feature_names=np.array(RISK_FEATURES),
        )
        rec_summary["shard"] = str(shard_path)
        rec_summary["global"] = _build_global_summary(arrays, [rec_summary], config)["global"]
        summaries.append(rec_summary)
        shard_paths.append(str(shard_path))
        print(f"Saved shard: {shard_path}")

    manifest = _build_sharded_manifest(summaries, shard_paths, config)
    manifest_path = shard_dir / f"{stem}.manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved manifest: {manifest_path}")
    print(json.dumps(manifest["global"], indent=2, ensure_ascii=False))


def _build_sharded_manifest(
    summaries: list[dict[str, Any]],
    shard_paths: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    decision_counts = {name: 0 for name in DECISION_TO_ID}
    total_samples = 0
    return_samples = 0
    multi_lane_change_samples = 0
    for summary in summaries:
        total_samples += int(summary["samples"])
        for name in DECISION_TO_ID:
            decision_counts[name] += int(summary["decision_counts"].get(name, 0))
        rec_global = summary.get("global", {})
        return_samples += int(rec_global.get("return_samples", 0))
        multi_lane_change_samples += int(rec_global.get("multi_lane_change_samples", 0))
    return {
        "global": {
            "num_recordings": len(summaries),
            "num_samples": total_samples,
            "decision_counts": decision_counts,
            "return_samples": return_samples,
            "multi_lane_change_samples": multi_lane_change_samples,
        },
        "shards": shard_paths,
        "recordings": summaries,
        "config": config,
    }


def _discover_recordings(raw_dir: Path, requested: list[str]) -> list[RecordingPaths]:
    ids = [str(x).zfill(2) for x in requested]
    if not ids:
        ids = sorted({p.name.split("_")[0] for p in raw_dir.glob("*_tracks.csv")})

    paths: list[RecordingPaths] = []
    for rec_id in ids:
        rec_paths = RecordingPaths(
            recording_id=rec_id,
            tracks=raw_dir / f"{rec_id}_tracks.csv",
            tracks_meta=raw_dir / f"{rec_id}_tracksMeta.csv",
            recording_meta=raw_dir / f"{rec_id}_recordingMeta.csv",
        )
        if rec_paths.tracks.exists() and rec_paths.tracks_meta.exists() and rec_paths.recording_meta.exists():
            paths.append(rec_paths)
        else:
            print(f"Skip recording {rec_id}: missing tracks/tracksMeta/recordingMeta file.")
    return paths


def _process_recording(rec_paths: RecordingPaths, config: dict[str, Any]) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    tracks = pd.read_csv(rec_paths.tracks)
    tracks_meta = pd.read_csv(rec_paths.tracks_meta)
    recording_meta = pd.read_csv(rec_paths.recording_meta).iloc[0].to_dict()
    frame_rate = int(recording_meta.get("frameRate", config["sampling"]["frame_rate"]))

    hist_len = int(round(config["sampling"]["history_seconds"] * frame_rate))
    fut_len = int(round(config["sampling"]["future_seconds"] * frame_rate))
    stride = int(config["sampling"]["stride_frames"])

    filtered_meta = _filter_vehicle_meta(tracks_meta, config)
    tracks = tracks[tracks["id"].isin(filtered_meta["id"])].copy()
    tracks.sort_values(["id", "frame"], inplace=True)

    meta_by_id = filtered_meta.set_index("id").to_dict("index")
    rows_by_frame_id = {
        (int(row.frame), int(row.id)): row
        for row in tracks.itertuples(index=False)
    }
    samples = _empty_sample_lists()
    decision_counts = {name: 0 for name in DECISION_TO_ID}

    for vehicle_id, vehicle_track in tqdm(
        tracks.groupby("id", sort=False),
        desc=f"recording {rec_paths.recording_id}",
        leave=False,
    ):
        vehicle_track = vehicle_track.sort_values("frame").reset_index(drop=True)
        if len(vehicle_track) < hist_len + fut_len + 1:
            continue

        meta = meta_by_id[int(vehicle_id)]
        long_sign, lat_sign = _coordinate_signs(int(meta["drivingDirection"]), config)
        steer = _estimate_steering(vehicle_track, meta, frame_rate, config)
        speed = _speed(vehicle_track["xVelocity"].to_numpy(), vehicle_track["yVelocity"].to_numpy())
        accel = _signed_acceleration(vehicle_track, speed, frame_rate)
        vehicle_arrays = _precompute_vehicle_arrays(
            vehicle_track,
            rows_by_frame_id,
            config,
            long_sign,
            lat_sign,
            speed,
        )

        stop = len(vehicle_track) - fut_len
        for anchor in range(hist_len - 1, stop, stride):
            hist_slice = slice(anchor - hist_len + 1, anchor + 1)
            fut_slice = slice(anchor + 1, anchor + fut_len + 1)
            current_lane = int(vehicle_arrays["lane_id"][anchor])
            future_lanes = vehicle_arrays["lane_id"][fut_slice]
            decision_sequence = _future_decision_sequence(
                current_lane,
                future_lanes,
                int(meta["drivingDirection"]),
            )
            decision = _decision_label_from_sequence(decision_sequence)
            decision_counts[decision] += 1

            ego_history = vehicle_arrays["ego_base"][hist_slice].copy()
            ego_history[:, 0] -= vehicle_arrays["x_norm"][anchor]
            ego_history[:, 1] -= vehicle_arrays["y_norm"][anchor]
            neighbor_history = vehicle_arrays["neighbor_history"][hist_slice]
            neighbor_mask = vehicle_arrays["neighbor_mask"][hist_slice]
            risk_history = vehicle_arrays["risk_history"][hist_slice]
            future_rel = np.column_stack(
                [
                    vehicle_arrays["x_norm"][fut_slice] - vehicle_arrays["x_norm"][anchor],
                    vehicle_arrays["y_norm"][fut_slice] - vehicle_arrays["y_norm"][anchor],
                ]
            )

            samples["ego_history"].append(ego_history)
            samples["neighbor_history"].append(neighbor_history.copy())
            samples["neighbor_mask"].append(neighbor_mask.copy())
            samples["risk_history"].append(risk_history.copy())
            samples["future_trajectory"].append(future_rel.astype(np.float32))
            samples["future_speed"].append(speed[anchor + 1 : anchor + fut_len + 1].astype(np.float32))
            samples["future_acceleration"].append(accel[anchor + 1 : anchor + fut_len + 1].astype(np.float32))
            samples["future_steer"].append(steer[anchor + 1 : anchor + fut_len + 1].astype(np.float32))
            samples["future_decision_sequence"].append(decision_sequence.astype(np.int32))
            samples["lane_change_count"].append(_lane_change_count(current_lane, future_lanes))
            samples["return_flag"].append(_return_flag(current_lane, future_lanes))
            samples["time_to_first_lane_change"].append(_time_to_first_lane_change(decision_sequence))
            samples["decision_label"].append(DECISION_TO_ID[decision])
            samples["recording_id"].append(int(rec_paths.recording_id))
            samples["vehicle_id"].append(int(vehicle_id))
            samples["frame_id"].append(int(vehicle_arrays["frame"][anchor]))
            samples["driving_direction"].append(int(meta["drivingDirection"]))
            samples["vehicle_length"].append(float(meta[config["vehicle_filter"]["length_column"]]))
            samples["vehicle_width"].append(float(meta[config["vehicle_filter"]["width_column"]]))
            samples["current_lane_id"].append(current_lane)

            max_samples = config["sampling"].get("max_samples_per_recording")
            if max_samples is not None and len(samples["decision_label"]) >= int(max_samples):
                break

        max_samples = config["sampling"].get("max_samples_per_recording")
        if max_samples is not None and len(samples["decision_label"]) >= int(max_samples):
            break

    summary = {
        "recording_id": rec_paths.recording_id,
        "frame_rate": frame_rate,
        "raw_vehicles": int(len(tracks_meta)),
        "filtered_vehicles": int(len(filtered_meta)),
        "samples": int(len(samples["decision_label"])),
        "decision_counts": decision_counts,
    }
    return samples, summary


def _empty_sample_lists() -> dict[str, list[Any]]:
    return {
        "ego_history": [],
        "neighbor_history": [],
        "neighbor_mask": [],
        "risk_history": [],
        "future_trajectory": [],
        "future_speed": [],
        "future_acceleration": [],
        "future_steer": [],
        "future_decision_sequence": [],
        "lane_change_count": [],
        "return_flag": [],
        "time_to_first_lane_change": [],
        "decision_label": [],
        "recording_id": [],
        "vehicle_id": [],
        "frame_id": [],
        "driving_direction": [],
        "vehicle_length": [],
        "vehicle_width": [],
        "current_lane_id": [],
    }


def _filter_vehicle_meta(tracks_meta: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    vehicle_filter = config["vehicle_filter"]
    length_col = vehicle_filter["length_column"]
    width_col = vehicle_filter["width_column"]
    keep = tracks_meta["class"].isin(vehicle_filter["keep_classes"])
    keep &= tracks_meta[length_col].between(vehicle_filter["min_length_m"], vehicle_filter["max_length_m"])
    keep &= tracks_meta[width_col].between(vehicle_filter["min_width_m"], vehicle_filter["max_width_m"])
    return tracks_meta[keep].copy()


def _precompute_vehicle_arrays(
    track: pd.DataFrame,
    rows_by_frame_id: dict[tuple[int, int], Any],
    config: dict[str, Any],
    long_sign: float,
    lat_sign: float,
    speed: np.ndarray,
) -> dict[str, np.ndarray]:
    x_norm = track["x"].to_numpy(dtype=np.float32) * long_sign
    y_norm = track["y"].to_numpy(dtype=np.float32) * lat_sign
    vx = track["xVelocity"].to_numpy(dtype=np.float32) * long_sign
    vy = track["yVelocity"].to_numpy(dtype=np.float32) * lat_sign
    ax = track["xAcceleration"].to_numpy(dtype=np.float32) * long_sign
    ay = track["yAcceleration"].to_numpy(dtype=np.float32) * lat_sign
    lane_id = track["laneId"].to_numpy(dtype=np.int32)

    ego_base = np.column_stack(
        [x_norm, y_norm, vx, vy, ax, ay, speed.astype(np.float32), lane_id.astype(np.float32)]
    ).astype(np.float32)
    neighbor_history, neighbor_mask = _build_neighbor_history_fast(track, rows_by_frame_id, config, long_sign, lat_sign)
    risk_history = _build_risk_history(track, config)
    return {
        "frame": track["frame"].to_numpy(dtype=np.int32),
        "lane_id": lane_id,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "ego_base": ego_base,
        "neighbor_history": neighbor_history,
        "neighbor_mask": neighbor_mask,
        "risk_history": risk_history,
    }


def _coordinate_signs(driving_direction: int, config: dict[str, Any]) -> tuple[float, float]:
    if not config.get("coordinate", {}).get("normalize_driving_direction", True):
        return 1.0, 1.0
    if driving_direction == 1:
        return -1.0, -1.0
    return 1.0, 1.0


def _estimate_steering(track: pd.DataFrame, meta: dict[str, Any], frame_rate: int, config: dict[str, Any]) -> np.ndarray:
    long_sign, lat_sign = _coordinate_signs(int(meta["drivingDirection"]), config)
    vx = _smooth(track["xVelocity"].to_numpy(dtype=np.float64) * long_sign, config)
    vy = _smooth(track["yVelocity"].to_numpy(dtype=np.float64) * lat_sign, config)
    heading = np.unwrap(np.arctan2(vy, vx))
    yaw_rate = np.gradient(_smooth(heading, config), 1.0 / frame_rate)
    speed = np.maximum(_speed(vx, vy), config["steering"]["min_speed_mps"])
    length_col = config["vehicle_filter"]["length_column"]
    wheelbase = float(meta[length_col]) * float(config["steering"]["wheelbase_ratio"])
    steer = np.arctan((wheelbase * yaw_rate) / speed)
    max_abs = float(config["steering"]["max_abs_steer_rad"])
    return np.clip(_smooth(steer, config), -max_abs, max_abs).astype(np.float32)


def _smooth(values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    window = int(config["steering"]["smooth_window"])
    polyorder = int(config["steering"]["smooth_polyorder"])
    if window <= 2 or len(values) < window:
        return values
    if window % 2 == 0:
        window += 1
    if savgol_filter is not None and window > polyorder:
        return savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _speed(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    return np.sqrt(vx * vx + vy * vy)


def _signed_acceleration(track: pd.DataFrame, speed: np.ndarray, frame_rate: int) -> np.ndarray:
    if {"xAcceleration", "yAcceleration"}.issubset(track.columns):
        vx = track["xVelocity"].to_numpy(dtype=np.float64)
        vy = track["yVelocity"].to_numpy(dtype=np.float64)
        ax = track["xAcceleration"].to_numpy(dtype=np.float64)
        ay = track["yAcceleration"].to_numpy(dtype=np.float64)
        denom = np.maximum(speed, 1e-3)
        return ((vx * ax + vy * ay) / denom).astype(np.float32)
    return np.gradient(speed, 1.0 / frame_rate).astype(np.float32)


def _future_decision_sequence(current_lane: int, future_lanes: np.ndarray, driving_direction: int) -> np.ndarray:
    labels = np.full(len(future_lanes), DECISION_TO_ID["S"], dtype=np.int32)
    previous_lane = current_lane
    for idx, lane in enumerate(future_lanes.astype(int)):
        delta = int(lane) - int(previous_lane)
        if delta != 0:
            labels[idx] = DECISION_TO_ID[_direction_from_lane_delta(delta, driving_direction)]
        previous_lane = int(lane)
    return labels


def _decision_label_from_sequence(decision_sequence: np.ndarray) -> str:
    non_keep = decision_sequence[decision_sequence != DECISION_TO_ID["S"]]
    if len(non_keep) == 0:
        return "S"
    first = int(non_keep[0])
    for name, idx in DECISION_TO_ID.items():
        if idx == first:
            return name
    return "S"


def _direction_from_lane_delta(delta: int, driving_direction: int) -> str:
    if driving_direction == 1:
        return "L" if delta < 0 else "R"
    return "L" if delta > 0 else "R"


def _lane_change_count(current_lane: int, future_lanes: np.ndarray) -> int:
    previous_lane = current_lane
    count = 0
    for lane in future_lanes.astype(int):
        if int(lane) != previous_lane:
            count += 1
            previous_lane = int(lane)
    return count


def _return_flag(current_lane: int, future_lanes: np.ndarray) -> int:
    left_origin = False
    for lane in future_lanes.astype(int):
        if int(lane) != current_lane:
            left_origin = True
        elif left_origin:
            return 1
    return 0


def _time_to_first_lane_change(decision_sequence: np.ndarray) -> int:
    changed = np.where(decision_sequence != DECISION_TO_ID["S"])[0]
    if len(changed) == 0:
        return -1
    return int(changed[0] + 1)


def _build_ego_history(hist: pd.DataFrame, current: pd.Series, long_sign: float, lat_sign: float) -> np.ndarray:
    xy = hist[["x", "y"]].to_numpy(dtype=np.float32)
    current_xy = current[["x", "y"]].to_numpy(dtype=np.float32)
    rel_xy = xy - current_xy
    rel_xy[:, 0] *= long_sign
    rel_xy[:, 1] *= lat_sign
    vx = hist["xVelocity"].to_numpy(dtype=np.float32) * long_sign
    vy = hist["yVelocity"].to_numpy(dtype=np.float32) * lat_sign
    ax = hist["xAcceleration"].to_numpy(dtype=np.float32) * long_sign
    ay = hist["yAcceleration"].to_numpy(dtype=np.float32) * lat_sign
    speed = _speed(vx, vy).astype(np.float32)
    lane = hist["laneId"].to_numpy(dtype=np.float32)
    return np.column_stack([rel_xy, vx, vy, ax, ay, speed, lane]).astype(np.float32)


def _build_neighbor_history(
    hist: pd.DataFrame,
    rows_by_frame_id: dict[tuple[int, int], Any],
    config: dict[str, Any],
    long_sign: float,
    lat_sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    slots = config["neighbors"]["slots"]
    neighbor_history = np.zeros((len(hist), len(slots), len(NEIGHBOR_FEATURES)), dtype=np.float32)
    neighbor_mask = np.zeros((len(hist), len(slots)), dtype=np.float32)
    for t_idx, (_, ego_row) in enumerate(hist.iterrows()):
        for s_idx, slot in enumerate(slots):
            neighbor_id = int(ego_row[slot])
            if neighbor_id == 0:
                continue
            key = (int(ego_row["frame"]), neighbor_id)
            nb = rows_by_frame_id.get(key)
            if nb is None:
                continue
            neighbor_history[t_idx, s_idx] = np.array(
                [
                    float(nb.x - ego_row["x"]),
                    float(nb.y - ego_row["y"]),
                    float(nb.xVelocity - ego_row["xVelocity"]),
                    float(nb.yVelocity - ego_row["yVelocity"]),
                    float(nb.width),
                    float(nb.height),
                ],
                dtype=np.float32,
            )
            neighbor_history[t_idx, s_idx, 0] *= long_sign
            neighbor_history[t_idx, s_idx, 1] *= lat_sign
            neighbor_history[t_idx, s_idx, 2] *= long_sign
            neighbor_history[t_idx, s_idx, 3] *= lat_sign
            neighbor_mask[t_idx, s_idx] = 1.0
    return neighbor_history, neighbor_mask


def _build_neighbor_history_fast(
    hist: pd.DataFrame,
    rows_by_frame_id: dict[tuple[int, int], Any],
    config: dict[str, Any],
    long_sign: float,
    lat_sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    slots = config["neighbors"]["slots"]
    n = len(hist)
    neighbor_history = np.zeros((n, len(slots), len(NEIGHBOR_FEATURES)), dtype=np.float32)
    neighbor_mask = np.zeros((n, len(slots)), dtype=np.float32)
    slot_values = {slot: hist[slot].to_numpy(dtype=np.int32) for slot in slots}
    frames = hist["frame"].to_numpy(dtype=np.int32)
    ego_x = hist["x"].to_numpy(dtype=np.float32)
    ego_y = hist["y"].to_numpy(dtype=np.float32)
    ego_vx = hist["xVelocity"].to_numpy(dtype=np.float32)
    ego_vy = hist["yVelocity"].to_numpy(dtype=np.float32)
    for t_idx, frame in enumerate(frames):
        for s_idx, slot in enumerate(slots):
            neighbor_id = int(slot_values[slot][t_idx])
            if neighbor_id == 0:
                continue
            nb = rows_by_frame_id.get((int(frame), neighbor_id))
            if nb is None:
                continue
            neighbor_history[t_idx, s_idx] = np.array(
                [
                    float((nb.x - ego_x[t_idx]) * long_sign),
                    float((nb.y - ego_y[t_idx]) * lat_sign),
                    float((nb.xVelocity - ego_vx[t_idx]) * long_sign),
                    float((nb.yVelocity - ego_vy[t_idx]) * lat_sign),
                    float(nb.width),
                    float(nb.height),
                ],
                dtype=np.float32,
            )
            neighbor_mask[t_idx, s_idx] = 1.0
    return neighbor_history, neighbor_mask


def _build_risk_history(hist: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    risk_cfg = config["risk"]
    dhw = _sanitize_positive(hist["dhw"].to_numpy(dtype=np.float32), cap=float(risk_cfg["sigma_distance_m"]) * 3)
    thw = _sanitize_positive(hist["thw"].to_numpy(dtype=np.float32), cap=float(risk_cfg["ttc_cap_s"]))
    ttc = _sanitize_positive(hist["ttc"].to_numpy(dtype=np.float32), cap=float(risk_cfg["ttc_cap_s"]))
    delta_v = np.abs(hist["xVelocity"].to_numpy(dtype=np.float32) - hist["precedingXVelocity"].to_numpy(dtype=np.float32))
    sigma = float(risk_cfg["sigma_distance_m"])
    front_risk = np.exp(-(dhw * dhw) / (sigma * sigma)) * (1.0 + float(risk_cfg["alpha_delta_v"]) * delta_v)
    front_risk = np.where(hist["precedingId"].to_numpy() > 0, front_risk, 0.0)
    return np.column_stack([dhw, thw, ttc, front_risk]).astype(np.float32)


def _sanitize_positive(values: np.ndarray, cap: float) -> np.ndarray:
    out = values.copy()
    out[out < 0] = cap
    out[out == 0] = cap
    return np.minimum(out, cap).astype(np.float32)


def _to_arrays(samples: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, values in samples.items():
        if not values:
            arrays[key] = np.array([])
            continue
        if key in {
            "decision_label",
            "future_decision_sequence",
            "lane_change_count",
            "return_flag",
            "time_to_first_lane_change",
            "recording_id",
            "vehicle_id",
            "frame_id",
            "driving_direction",
            "current_lane_id",
        }:
            arrays[key] = np.asarray(values, dtype=np.int32)
        elif key in {"vehicle_length", "vehicle_width"}:
            arrays[key] = np.asarray(values, dtype=np.float32)
        else:
            arrays[key] = np.stack(values).astype(np.float32)
    return arrays


def _build_global_summary(
    arrays: dict[str, np.ndarray],
    recording_summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    labels = arrays["decision_label"]
    counts = {
        name: int(np.sum(labels == idx)) if labels.size else 0
        for name, idx in DECISION_TO_ID.items()
    }
    return {
        "global": {
            "num_recordings": len(recording_summaries),
            "num_samples": int(labels.shape[0]) if labels.size else 0,
            "decision_counts": counts,
            "ego_history_shape": list(arrays["ego_history"].shape),
            "neighbor_history_shape": list(arrays["neighbor_history"].shape),
            "future_trajectory_shape": list(arrays["future_trajectory"].shape),
            "future_steer_shape": list(arrays["future_steer"].shape),
            "future_decision_sequence_shape": list(arrays["future_decision_sequence"].shape),
            "return_samples": int(np.sum(arrays["return_flag"])) if arrays["return_flag"].size else 0,
            "multi_lane_change_samples": int(np.sum(arrays["lane_change_count"] >= 2)) if arrays["lane_change_count"].size else 0,
        },
        "recordings": recording_summaries,
        "config": config,
    }
