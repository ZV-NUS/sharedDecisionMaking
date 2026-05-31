from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.visualize_machine_policy_rollouts import _case_summary, _read_sample, _select_cases
from src.envs.highd_injected_env import HighDInjectedTrafficEnv, HighDInjectedTrafficEnvConfig
from src.policies.machine_intent_policy import MachineIntentPolicy, MachineIntentPolicyConfig


INPUT_KEYS = ("ego_history", "neighbor_history", "neighbor_mask", "risk_history")
SLOT_NAMES = ("front", "rear", "left-front", "left-rear", "right-front", "right-rear")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export highD-injected machine policy rollouts for realtime browser playback.")
    parser.add_argument("--config", default="configs/machine_policy_visual_validation.yaml")
    args = parser.parse_args()

    config = _load_config(ROOT / args.config)
    out_dir = ROOT / config["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = MachineIntentPolicy(MachineIntentPolicyConfig(**config["policy"]))
    env = HighDInjectedTrafficEnv(HighDInjectedTrafficEnvConfig(**config["environment"]))
    h5_path = ROOT / config["data"]["h5_path"]
    case_indices = _select_cases(h5_path, config["selection"])

    cases = []
    with h5py.File(h5_path, "r") as h5:
        for case_no, idx in enumerate(case_indices, start=1):
            sample = _read_sample(h5, idx)
            inputs = {key: torch.as_tensor(sample[key], dtype=torch.float32).unsqueeze(0) for key in INPUT_KEYS}
            outputs = policy.predict(inputs)
            rollout = env.rollout(sample, outputs)
            record = _case_summary(sample, rollout, idx, case_no)
            cases.append(_pack_case(sample, rollout, record))

    payload = {
        "frame_rate": float(config["environment"]["frame_rate"]),
        "lane_width_m": float(config["environment"]["lane_width_m"]),
        "slot_names": SLOT_NAMES,
        "cases": cases,
    }
    js_path = out_dir / "realtime_rollouts.js"
    js_path.write_text("window.MACHINE_POLICY_ROLLOUTS = " + json.dumps(payload, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(json.dumps({"realtime_data": str(js_path), "cases": len(cases)}, indent=2))


def _pack_case(sample: dict, rollout: dict, record: dict) -> dict:
    neighbor_mask = np.asarray(rollout["neighbor_mask"], dtype=bool)
    neighbors = []
    neighbor_xy = np.asarray(rollout["neighbor_xy"], dtype=np.float32)
    neighbor_state = np.asarray(sample["neighbor_history"], dtype=np.float32)[-1]
    for slot, valid in enumerate(neighbor_mask):
        if not bool(valid):
            continue
        length = float(neighbor_state[slot, 4]) if float(neighbor_state[slot, 4]) > 0 else 4.6
        width = float(neighbor_state[slot, 5]) if float(neighbor_state[slot, 5]) > 0 else 1.8
        neighbors.append(
            {
                "slot": int(slot),
                "name": SLOT_NAMES[slot],
                "length": length,
                "width": width,
                "xy": _round_array(neighbor_xy[slot]),
            }
        )

    return {
        "record": record,
        "ego": {
            "length": 4.6,
            "width": 1.8,
            "xy": _round_array(np.asarray(rollout["ego_xy"], dtype=np.float32)),
            "yaw": _round_array(np.asarray(rollout["ego_yaw"], dtype=np.float32)),
            "speed": _round_array(np.asarray(rollout["speed"], dtype=np.float32)),
            "acceleration": _round_array(np.asarray(rollout["acceleration"], dtype=np.float32)),
            "steer": _round_array(np.asarray(rollout["steer"], dtype=np.float32)),
        },
        "human_future": _round_array(np.asarray(rollout["ground_truth_xy"], dtype=np.float32)),
        "neighbors": neighbors,
    }


def _round_array(array: np.ndarray) -> list:
    return np.round(array, 4).tolist()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
