from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_human_intent_transformer import MetricsMeter
from src.data.intent_dataset import HighDIntentDataset
from src.policies.machine_intent_policy import MachineIntentPolicy, MachineIntentPolicyConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic rule/APF machine intent policy on highD samples.")
    parser.add_argument("--config", default="configs/machine_intent_policy.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    args = parser.parse_args()

    config = _load_config(ROOT / args.config)
    _set_seed(int(config["evaluation"]["seed"]))
    device = _resolve_device(config["evaluation"]["device"])
    policy = MachineIntentPolicy(MachineIntentPolicyConfig(**config["policy"]))
    output_dir = ROOT / config["evaluation"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for split in args.splits:
        loader = _make_loader(config, split)
        metrics = evaluate_split(policy, loader, config, device, max_batches=config["evaluation"].get(f"max_{split}_batches"))
        all_metrics[split] = metrics
        with (output_dir / f"{split}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(split.upper())
        print(json.dumps(metrics, indent=2))

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)


def evaluate_split(
    policy: MachineIntentPolicy,
    loader: DataLoader,
    config: dict,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    eval_cfg = config["evaluation"]
    meter = MetricsMeter(
        {
            "future_eval_mode": eval_cfg["future_eval_mode"],
            "future_time_eval_mode": eval_cfg["future_time_eval_mode"],
            "future_time_index_scale": eval_cfg.get("future_time_index_scale", 1.0),
            "future_time_index_offset": eval_cfg.get("future_time_index_offset", 0),
        }
    )
    max_batches = int(max_batches) if max_batches is not None else None
    iterator = tqdm(loader, total=max_batches, desc="machine", leave=False)
    safety = MachineSafetyMeter()
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch["inputs"].items()}
        targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
        outputs = policy.predict(inputs)
        zero_loss = torch.zeros((), device=device)
        meter.update(outputs, targets, zero_loss, {})
        safety.update(inputs, outputs)
        iterator.set_postfix(event_f1=f"{meter.compute()['future_event_macro_f1']:.3f}")
    metrics = meter.compute()
    metrics.update(safety.compute())
    return metrics


class MachineSafetyMeter:
    def __init__(self) -> None:
        self.count = 0
        self.lane_change_count = 0
        self.high_front_risk_count = 0
        self.high_front_risk_lane_change_count = 0
        self.unsafe_lane_change_count = 0
        self.abs_steer_sum = 0.0
        self.abs_accel_sum = 0.0
        self.max_abs_steer = 0.0

    def update(self, inputs: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        decision = outputs["future_event_logits"].argmax(dim=-1)
        risk_last = inputs["risk_history"][:, -1]
        neighbors = inputs["neighbor_history"][:, -1]
        mask = inputs["neighbor_mask"][:, -1] > 0.5
        speed = torch.clamp(inputs["ego_history"][:, -1, 6], min=1.0)

        lane_change = decision != 1
        high_risk = (risk_last[:, 3] > 0.36) | (risk_last[:, 1] < 1.2) | (risk_last[:, 2] < 5.0)
        unsafe = _unsafe_lane_change(decision, neighbors, mask, speed)

        self.count += int(decision.numel())
        self.lane_change_count += int(lane_change.sum().detach().cpu())
        self.high_front_risk_count += int(high_risk.sum().detach().cpu())
        self.high_front_risk_lane_change_count += int((high_risk & lane_change).sum().detach().cpu())
        self.unsafe_lane_change_count += int((unsafe & lane_change).sum().detach().cpu())
        self.abs_steer_sum += float(outputs["future_steer"].abs().sum().detach().cpu())
        if "future_acceleration" in outputs:
            self.abs_accel_sum += float(outputs["future_acceleration"].abs().sum().detach().cpu())
        self.max_abs_steer = max(self.max_abs_steer, float(outputs["future_steer"].abs().max().detach().cpu()))

    def compute(self) -> dict:
        denom_frames = max(self.count * 125, 1)
        return {
            "machine_lane_change_rate": float(self.lane_change_count / max(self.count, 1)),
            "machine_high_risk_lane_change_rate": float(self.high_front_risk_lane_change_count / max(self.high_front_risk_count, 1)),
            "machine_unsafe_lane_change_rate": float(self.unsafe_lane_change_count / max(self.lane_change_count, 1)),
            "machine_mean_abs_steer_rad": float(self.abs_steer_sum / denom_frames),
            "machine_mean_abs_accel_mps2": float(self.abs_accel_sum / denom_frames),
            "machine_max_abs_steer_rad": float(self.max_abs_steer),
        }


def _unsafe_lane_change(
    decision: torch.Tensor,
    neighbors: torch.Tensor,
    mask: torch.Tensor,
    speed: torch.Tensor,
) -> torch.Tensor:
    unsafe = torch.zeros_like(decision, dtype=torch.bool)
    for label, front_slot, rear_slot in [(0, 2, 3), (2, 4, 5)]:
        active = decision == label
        front = neighbors[:, front_slot]
        rear = neighbors[:, rear_slot]
        front_valid = mask[:, front_slot]
        rear_valid = mask[:, rear_slot]
        front_gap = torch.where(front_valid, torch.clamp(front[:, 0], min=0.0), torch.full_like(speed, 100.0))
        rear_gap = torch.where(rear_valid, torch.clamp(-rear[:, 0], min=0.0), torch.full_like(speed, 100.0))
        rear_rel_v = torch.where(rear_valid, rear[:, 2], torch.zeros_like(speed))
        unsafe_side = (front_gap < 18.0 + 0.35 * speed) | (rear_gap < 15.0 + 0.25 * torch.clamp(rear_rel_v, min=0.0))
        unsafe = unsafe | (active & unsafe_side)
    return unsafe


def _make_loader(config: dict, split: str) -> DataLoader:
    dataset = HighDIntentDataset(
        ROOT / config["data"][f"{split}_index"],
        include_meta=False,
        preload_shards=bool(config["data"].get("preload_shards", False)),
    )
    batch_size = int(config["data"]["batch_size"])
    max_batches = config["evaluation"].get(f"max_{split}_batches")
    sampler = None
    if bool(config["data"].get("eval_random_subset", True)) and max_batches is not None:
        generator = torch.Generator()
        generator.manual_seed(int(config["evaluation"]["seed"]) + {"train": 0, "val": 1, "test": 2}[split])
        num_samples = min(len(dataset), int(max_batches) * batch_size)
        if bool(config["data"].get("eval_balanced", False)):
            sample_indices = _balanced_eval_indices(
                ROOT / config["data"][f"{split}_index"],
                num_samples=num_samples,
                straight_to_lane_change_ratio=float(config["data"].get("eval_straight_to_lane_change_ratio", 1.0)),
                seed=int(config["evaluation"]["seed"]) + {"train": 0, "val": 1, "test": 2}[split],
            )
            sampler = SubsetRandomSampler(sample_indices, generator=generator)
        else:
            sampler = RandomSampler(dataset, replacement=False, num_samples=num_samples, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else False,
        sampler=sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=False,
        drop_last=False,
    )


def _balanced_eval_indices(index_path: Path, num_samples: int, straight_to_lane_change_ratio: float, seed: int) -> list[int]:
    labels = _load_decision_labels(index_path)
    rng = np.random.default_rng(seed)
    lane_change_pos = np.where(labels != 1)[0]
    straight_pos = np.where(labels == 1)[0]
    target_straight = min(straight_pos.shape[0], int(round(lane_change_pos.shape[0] * straight_to_lane_change_ratio)))
    selected = np.concatenate(
        [
            lane_change_pos,
            rng.choice(straight_pos, size=target_straight, replace=False) if target_straight > 0 else np.array([], dtype=np.int64),
        ]
    )
    if selected.shape[0] > num_samples:
        selected = rng.choice(selected, size=num_samples, replace=False)
    rng.shuffle(selected)
    return selected.astype(np.int64).tolist()


def _load_decision_labels(index_path: Path) -> np.ndarray:
    if index_path.suffix.lower() in {".h5", ".hdf5"}:
        with h5py.File(index_path, "r") as h5:
            return np.asarray(h5["decision_label"][:], dtype=np.int64)
    raise ValueError(f"Unsupported index path: {index_path}")


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    main()
