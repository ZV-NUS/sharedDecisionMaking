from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.highd_injected_env import HighDInjectedTrafficEnv, HighDInjectedTrafficEnvConfig
from src.policies.machine_intent_policy import MachineIntentPolicy, MachineIntentPolicyConfig


DECISION_NAMES = {0: "L", 1: "S", 2: "R"}
INPUT_KEYS = ("ego_history", "neighbor_history", "neighbor_mask", "risk_history")
TARGET_KEYS = ("future_trajectory", "future_speed", "future_acceleration", "future_steer", "future_decision_sequence", "decision_label")
META_KEYS = ("recording_id", "vehicle_id", "frame_id", "current_lane_id", "return_flag", "lane_change_count")
SLOT_NAMES = ("front", "rear", "left-front", "left-rear", "right-front", "right-rear")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize rule/APF machine policy rollouts in highD-injected scenes.")
    parser.add_argument("--config", default="configs/machine_policy_visual_validation.yaml")
    args = parser.parse_args()

    config = _load_config(ROOT / args.config)
    random.seed(int(config["selection"]["seed"]))
    np.random.seed(int(config["selection"]["seed"]))
    out_dir = ROOT / config["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_fig in out_dir.glob("case_*.png"):
        old_fig.unlink()

    policy = MachineIntentPolicy(MachineIntentPolicyConfig(**config["policy"]))
    env = HighDInjectedTrafficEnv(HighDInjectedTrafficEnvConfig(**config["environment"]))
    h5_path = ROOT / config["data"]["h5_path"]
    case_indices = _select_cases(h5_path, config["selection"])

    summary = []
    with h5py.File(h5_path, "r") as h5:
        for case_no, idx in enumerate(case_indices, start=1):
            sample = _read_sample(h5, idx)
            inputs = {key: torch.as_tensor(sample[key], dtype=torch.float32).unsqueeze(0) for key in INPUT_KEYS}
            outputs = policy.predict(inputs)
            rollout = env.rollout(sample, outputs)
            record = _case_summary(sample, rollout, idx, case_no)
            summary.append(record)
            fig_path = out_dir / f"case_{case_no:02d}_idx_{idx}.png"
            _plot_case(sample, rollout, record, fig_path)

    summary_path = out_dir / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"output_dir": str(out_dir), "summary": str(summary_path), "cases": summary}, indent=2))


def _select_cases(h5_path: Path, cfg: dict) -> list[int]:
    with h5py.File(h5_path, "r") as h5:
        n = int(h5["decision_label"].shape[0])
        scan_limit = min(n, int(cfg.get("scan_limit", n)))
        risk = h5["risk_history"][:scan_limit, -1, 3]
        labels = h5["decision_label"][:scan_limit].astype(np.int64)
        candidate = np.where(risk >= float(cfg["min_front_risk"]))[0]
        if bool(cfg.get("include_lane_change_cases", True)):
            lane_change = np.where(labels != 1)[0]
            candidate = np.unique(np.concatenate([candidate, lane_change[: max(10, len(candidate) // 3)]]))
        if candidate.size == 0:
            candidate = np.arange(min(scan_limit, int(cfg["num_cases"])))
        candidate = _deduplicate_vehicle_windows(h5, candidate)
        order = np.argsort(-risk[candidate])
        selected = candidate[order[: int(cfg["num_cases"])]]
        return [int(x) for x in selected]


def _deduplicate_vehicle_windows(h5: h5py.File, candidate: np.ndarray) -> np.ndarray:
    if "recording_id" not in h5 or "vehicle_id" not in h5:
        return candidate
    rec = h5["recording_id"][candidate].astype(np.int64)
    veh = h5["vehicle_id"][candidate].astype(np.int64)
    frame = h5["frame_id"][candidate].astype(np.int64) if "frame_id" in h5 else np.arange(candidate.size)
    risk = h5["risk_history"][candidate, -1, 3]
    best_by_vehicle: dict[tuple[int, int], tuple[float, int, int]] = {}
    for local_i, idx in enumerate(candidate):
        key = (int(rec[local_i]), int(veh[local_i]))
        score = float(risk[local_i])
        current = best_by_vehicle.get(key)
        if current is None or score > current[0]:
            best_by_vehicle[key] = (score, int(frame[local_i]), int(idx))
    diverse = [item[2] for item in best_by_vehicle.values()]
    return np.asarray(diverse, dtype=np.int64)


def _read_sample(h5: h5py.File, idx: int) -> dict[str, np.ndarray | int | float]:
    sample: dict[str, np.ndarray | int | float] = {}
    for key in INPUT_KEYS + TARGET_KEYS:
        sample[key] = np.asarray(h5[key][idx])
    for key in META_KEYS:
        if key in h5:
            value = np.asarray(h5[key][idx])
            sample[key] = value.item() if value.ndim == 0 else value
    return sample


def _case_summary(sample: dict, rollout: dict, idx: int, case_no: int) -> dict:
    true_label = int(np.asarray(sample["decision_label"]).item())
    pred_label = int(rollout["decision"])
    return {
        "case": case_no,
        "sample_index": int(idx),
        "recording_id": int(sample.get("recording_id", -1)),
        "vehicle_id": int(sample.get("vehicle_id", -1)),
        "frame_id": int(sample.get("frame_id", -1)),
        "true_decision": DECISION_NAMES[true_label],
        "machine_decision": DECISION_NAMES[pred_label],
        "machine_event_idx": int(rollout["event_idx"]),
        "front_risk": float(rollout["front_risk"]),
        "ttc": float(rollout["ttc"]),
        "thw": float(rollout["thw"]),
        "dhw": float(rollout["dhw"]),
        "collision": bool(rollout["collision"]),
        "min_clearance_m": float(rollout["min_clearance_m"]),
        "min_clearance_slot": SLOT_NAMES[int(rollout["min_clearance_slot"])] if int(rollout["min_clearance_slot"]) >= 0 else "none",
        "min_clearance_frame": int(rollout["min_clearance_frame"]),
        "max_abs_steer_rad": float(rollout["max_abs_steer_rad"]),
        "max_abs_accel_mps2": float(rollout["max_abs_accel_mps2"]),
        "mean_abs_steer_rad": float(rollout["mean_abs_steer_rad"]),
        "mean_abs_accel_mps2": float(rollout["mean_abs_accel_mps2"]),
    }


def _plot_case(sample: dict, rollout: dict, record: dict, fig_path: Path) -> None:
    future_len = rollout["ego_xy"].shape[0]
    t = np.arange(future_len) / 25.0
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0])
    ax_traj = fig.add_subplot(gs[0, :])
    ax_speed = fig.add_subplot(gs[1, 0])
    ax_ctrl = fig.add_subplot(gs[1, 1])

    ego_xy = rollout["ego_xy"]
    gt_xy = rollout["ground_truth_xy"]
    ax_traj.plot(ego_xy[:, 0], ego_xy[:, 1], color="#1f77b4", linewidth=2.4, label="machine ego rollout")
    ax_traj.plot(gt_xy[:, 0], gt_xy[:, 1], color="#7f7f7f", linewidth=1.8, linestyle="--", label="highD human future")
    ax_traj.scatter([0.0], [0.0], color="#1f77b4", s=40)
    ax_traj.axhline(0.0, color="#333333", linewidth=1.0)
    for lane_y in (-3.5, 3.5, -7.0, 7.0):
        ax_traj.axhline(lane_y, color="#d0d0d0", linewidth=0.8, linestyle=":")

    neighbor_xy = rollout["neighbor_xy"]
    neighbor_mask = rollout["neighbor_mask"]
    for slot, valid in enumerate(neighbor_mask):
        if not bool(valid):
            continue
        ax_traj.plot(neighbor_xy[slot, :, 0], neighbor_xy[slot, :, 1], linewidth=1.3, alpha=0.8, label=SLOT_NAMES[slot])
        ax_traj.scatter(neighbor_xy[slot, 0, 0], neighbor_xy[slot, 0, 1], s=26)

    event_idx = int(rollout["event_idx"])
    if record["machine_decision"] != "S":
        ax_traj.scatter(ego_xy[event_idx, 0], ego_xy[event_idx, 1], marker="x", color="#d62728", s=80, label="machine event")

    title = (
        f"case {record['case']} idx={record['sample_index']} "
        f"machine={record['machine_decision']} true={record['true_decision']} "
        f"risk={record['front_risk']:.2f} ttc={record['ttc']:.1f}s "
        f"collision={record['collision']}"
    )
    ax_traj.set_title(title)
    ax_traj.set_xlabel("x in ego traffic coordinates (m)")
    ax_traj.set_ylabel("y, left positive (m)")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.legend(loc="upper left", ncol=3, fontsize=8)

    ax_speed.plot(t, rollout["speed"], color="#1f77b4", label="machine speed")
    ax_speed.plot(t, sample["future_speed"], color="#7f7f7f", linestyle="--", label="highD speed")
    ax_speed.set_xlabel("time (s)")
    ax_speed.set_ylabel("speed (m/s)")
    ax_speed.grid(True, alpha=0.25)
    ax_speed.legend(fontsize=8)

    ax_ctrl.plot(t, rollout["steer"], color="#2ca02c", label="machine steer")
    ax_ctrl.plot(t, rollout["acceleration"], color="#d62728", label="machine accel")
    ax_ctrl.axhline(0.0, color="#333333", linewidth=0.8)
    ax_ctrl.set_xlabel("time (s)")
    ax_ctrl.set_ylabel("rad / m/s^2")
    ax_ctrl.grid(True, alpha=0.25)
    ax_ctrl.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
