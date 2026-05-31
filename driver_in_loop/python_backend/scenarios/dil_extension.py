from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


METHOD_KEYS = (
    "ego",
    "human_pred_ego",
    "machine_ego",
    "reference_ego",
    "ra_rldm_ego",
    "controller_ego",
)


def extend_case_for_dil(
    case: dict[str, Any],
    *,
    start_s: float,
    end_s: float,
    source_frame_rate: float,
) -> dict[str, Any]:
    """Extend a paper validation case for driver-in-the-loop interaction.

    The original paper cases are short conflict windows. For DIL validation we
    keep the highD-derived conflict window unchanged, then prepend/append a
    constant-velocity segment from the boundary states. This produces a longer
    observation/recovery horizon without changing the central validation event.
    """

    start_frames = max(0, int(round(float(start_s) * float(source_frame_rate))))
    end_frames = max(0, int(round(float(end_s) * float(source_frame_rate))))
    if start_frames == 0 and end_frames == 0:
        return deepcopy(case)

    out = deepcopy(case)
    n = len(np.asarray(case["controller_ego"]["xy"], dtype=float))

    for key in METHOD_KEYS:
        if key in out:
            out[key] = _extend_vehicle(out[key], start_frames, end_frames, source_frame_rate)

    extended_neighbors = []
    for vehicle in out.get("neighbors", []):
        extended_neighbors.append(_extend_vehicle(vehicle, start_frames, end_frames, source_frame_rate))
    out["neighbors"] = extended_neighbors

    signals = out.get("signals", {})
    for key, values in list(signals.items()):
        arr = np.asarray(values)
        if arr.ndim == 1 and len(arr) == n:
            signals[key] = _extend_scalar(arr, start_frames, end_frames).tolist()
    out["signals"] = signals

    record = out.setdefault("record", {})
    record["dil_extension_start_s"] = float(start_s)
    record["dil_extension_end_s"] = float(end_s)
    record["dil_original_frames"] = int(n)
    record["dil_extended_frames"] = int(n + start_frames + end_frames)
    return out


def _extend_vehicle(vehicle: dict[str, Any], start_frames: int, end_frames: int, source_frame_rate: float) -> dict[str, Any]:
    out = deepcopy(vehicle)
    if "xy" in out:
        out["xy"] = _extend_xy(np.asarray(out["xy"], dtype=float), start_frames, end_frames).tolist()

    xy_len = len(out.get("xy", []))
    for key in ("yaw", "speed", "acceleration", "steer"):
        if key not in out:
            continue
        arr = np.asarray(out[key], dtype=float)
        if arr.ndim == 1 and len(arr) > 0:
            out[key] = _extend_scalar(arr, start_frames, end_frames).tolist()

    # If a surrounding vehicle has no explicit speed, keep the original fields
    # and let the runtime infer yaw/speed from the extended xy trajectory.
    if xy_len == 0:
        return out
    return out


def _extend_xy(xy: np.ndarray, start_frames: int, end_frames: int) -> np.ndarray:
    if xy.ndim != 2 or len(xy) == 0:
        return xy
    if len(xy) >= 2:
        start_step = xy[1] - xy[0]
        end_step = xy[-1] - xy[-2]
    else:
        start_step = np.array([0.0, 0.0])
        end_step = np.array([0.0, 0.0])

    prefix = np.array([xy[0] - start_step * i for i in range(start_frames, 0, -1)], dtype=float)
    suffix = np.array([xy[-1] + end_step * i for i in range(1, end_frames + 1)], dtype=float)
    return np.vstack([prefix, xy, suffix])


def _extend_scalar(values: np.ndarray, start_frames: int, end_frames: int) -> np.ndarray:
    if values.ndim != 1 or len(values) == 0:
        return values
    prefix = np.full(start_frames, values[0], dtype=values.dtype)
    suffix = np.full(end_frames, values[-1], dtype=values.dtype)
    return np.concatenate([prefix, values, suffix])
