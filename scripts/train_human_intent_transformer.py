from pathlib import Path
import argparse
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import h5py
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.intent_dataset import HighDIntentDataset, create_intent_dataloader
from src.models.human_intent_transformer import HumanIntentTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer baseline for highD human intent prediction.")
    parser.add_argument("--config", default="configs/human_intent_transformer.yaml")
    parser.add_argument("--eval-only", action="store_true", help="Load the best checkpoint and run the test split only.")
    parser.add_argument("--time-mode", default=None, help="Override future_time_eval_mode for evaluation.")
    parser.add_argument("--time-offset", type=int, default=None, help="Override future_time_index_offset for evaluation.")
    parser.add_argument("--time-scale", type=float, default=None, help="Override future_time_index_scale for evaluation.")
    args = parser.parse_args()

    config = _load_config(ROOT / args.config)
    if args.time_mode is not None:
        config.setdefault("loss", {})["future_time_eval_mode"] = args.time_mode
    if args.time_offset is not None:
        config.setdefault("loss", {})["future_time_index_offset"] = args.time_offset
    if args.time_scale is not None:
        config.setdefault("loss", {})["future_time_index_scale"] = args.time_scale
    _set_seed(int(config["training"]["seed"]))
    device = _resolve_device(config["training"]["device"])
    print(f"device: {device}")

    train_loader = _make_loader(config, "train", shuffle=bool(config["data"].get("shuffle_train", False)), drop_last=True)
    val_loader = _make_loader(config, "val", shuffle=False, drop_last=False)
    test_loader = _make_loader(config, "test", shuffle=False, drop_last=False)

    model = HumanIntentTransformer(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    checkpoint_dir = ROOT / config["training"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        checkpoint = torch.load(checkpoint_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = run_epoch(
            model,
            test_loader,
            config,
            device,
            optimizer=None,
            max_batches=config["training"].get("max_test_batches"),
            train=False,
        )
        with (checkpoint_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        print("TEST")
        print(json.dumps(test_metrics, indent=2))
        return

    best_macro_f1 = -1.0
    history = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            config,
            device,
            optimizer=optimizer,
            max_batches=config["training"].get("max_train_batches"),
            train=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            config,
            device,
            optimizer=None,
            max_batches=config["training"].get("max_val_batches"),
            train=False,
        )
        elapsed = time.time() - t0
        record = {"epoch": epoch, "elapsed_s": elapsed, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, indent=2))

        checkpoint_score = (
            0.35 * val_metrics["decision_macro_f1"]
            + 0.45 * val_metrics["future_event_macro_f1"]
            + 0.20 * val_metrics["lane_change_event_precision"]
            - 0.20 * val_metrics["lane_change_event_time_mae_frames"] / float(config["model"]["future_len"])
        )
        if checkpoint_score > best_macro_f1:
            best_macro_f1 = checkpoint_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "checkpoint_score": checkpoint_score,
                },
                checkpoint_dir / "best.pt",
            )

        with (checkpoint_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(
        model,
        test_loader,
        config,
        device,
        optimizer=None,
        max_batches=config["training"].get("max_test_batches"),
        train=False,
    )
    with (checkpoint_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print("TEST")
    print(json.dumps(test_metrics, indent=2))


def _make_loader(config: dict, split: str, shuffle: bool, drop_last: bool) -> DataLoader:
    index_key = f"{split}_index"
    dataset = HighDIntentDataset(
        ROOT / config["data"][index_key],
        include_meta=False,
        preload_shards=bool(config["data"].get("preload_shards", False)),
    )
    max_batches = config["training"].get(f"max_{split}_batches")
    batch_size = int(config["data"]["batch_size"])
    sampler = None
    if split == "train" and bool(config["data"].get("train_balanced", False)) and max_batches is not None:
        generator = torch.Generator()
        generator.manual_seed(int(config["training"]["seed"]))
        num_samples = min(len(dataset), int(max_batches) * batch_size)
        sample_indices = _balanced_eval_indices(
            ROOT / config["data"][index_key],
            num_samples=num_samples,
            straight_to_lane_change_ratio=float(config["data"].get("train_straight_to_lane_change_ratio", 1.0)),
            seed=int(config["training"]["seed"]),
        )
        sampler = SubsetRandomSampler(sample_indices, generator=generator)
    elif split in {"val", "test"} and bool(config["data"].get("eval_random_subset", True)) and max_batches is not None:
        generator = torch.Generator()
        generator.manual_seed(int(config["training"]["seed"]) + (1 if split == "val" else 2))
        num_samples = min(len(dataset), int(max_batches) * batch_size)
        if bool(config["data"].get("eval_balanced", False)):
            sample_indices = _balanced_eval_indices(
                ROOT / config["data"][index_key],
                num_samples=num_samples,
                straight_to_lane_change_ratio=float(config["data"].get("eval_straight_to_lane_change_ratio", 1.0)),
                seed=int(config["training"]["seed"]) + (1 if split == "val" else 2),
            )
            sampler = SubsetRandomSampler(sample_indices, generator=generator)
        else:
            sampler = RandomSampler(dataset, replacement=False, num_samples=num_samples, generator=generator)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=False,
        drop_last=drop_last,
    )


def run_epoch(
    model: nn.Module,
    loader,
    config: dict,
    device: torch.device,
    optimizer=None,
    max_batches=None,
    train: bool = True,
) -> dict:
    model.train(train)
    meter = MetricsMeter(config.get("loss", {}))
    max_batches = int(max_batches) if max_batches is not None else None
    iterator = tqdm(loader, total=max_batches, desc="train" if train else "val", leave=False)
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch["inputs"].items()}
        targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}
        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            loss, loss_parts = compute_loss(outputs, targets, config["loss"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["grad_clip_norm"]))
                optimizer.step()
        meter.update(outputs, targets, loss.detach(), loss_parts)
        iterator.set_postfix(loss=f"{meter.loss_avg:.4f}", f1=f"{meter.decision_macro_f1:.3f}")
    return meter.compute()


def compute_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], cfg: dict) -> tuple[torch.Tensor, dict]:
    decision_weight = _tensor_weights(cfg.get("decision_class_weights"), outputs["decision_logits"].device)
    decision_loss = _cross_entropy_or_focal(
        outputs["decision_logits"],
        targets["decision_label"],
        weight=decision_weight,
        gamma=float(cfg.get("focal_gamma", 0.0)),
    )
    future_event_label, future_event_time, future_event_mask = _future_event_targets(targets["future_decision_sequence"])
    future_event_weight = _tensor_weights(cfg.get("future_event_class_weights"), outputs["future_event_logits"].device)
    future_event_loss = _cross_entropy_or_focal(
        outputs["future_event_logits"],
        future_event_label,
        weight=future_event_weight,
        gamma=float(cfg.get("future_event_focal_gamma", cfg.get("focal_gamma", 0.0))),
    )
    time_error = F.smooth_l1_loss(outputs["future_event_time"], future_event_time, reduction="none")
    future_event_time_loss = (time_error * future_event_mask.float()).sum() / torch.clamp(future_event_mask.float().sum(), min=1.0)
    future_event_time_bin_loss = _masked_time_bin_loss(
        outputs["future_event_time_logits"],
        torch.round(future_event_time * float(outputs["future_event_time_logits"].shape[1] - 1)).long(),
        future_event_mask,
    )
    future_event_time_class_loss = _masked_class_time_loss(
        outputs["future_event_time_by_class"],
        future_event_label,
        future_event_time,
        future_event_mask,
        future_len=outputs["future_event_time_logits"].shape[1],
    )
    future_event_time_bin_class_loss = _masked_class_time_bin_loss(
        outputs["future_event_time_bin_by_class"],
        future_event_label,
        torch.round(future_event_time * float(outputs["future_event_time_logits"].shape[1] - 1)).long(),
        future_event_mask,
    )

    future_weight = _tensor_weights(cfg.get("future_decision_class_weights"), outputs["future_decision_logits"].device)
    frame_weight = _future_sequence_frame_weights(
        targets["future_decision_sequence"],
        window=int(cfg.get("sequence_event_window", 0)),
        event_window_weight=float(cfg.get("sequence_event_window_weight", 1.0)),
    )
    future_decision_loss = _cross_entropy_or_focal(
        outputs["future_decision_logits"].reshape(-1, outputs["future_decision_logits"].shape[-1]),
        targets["future_decision_sequence"].reshape(-1),
        weight=future_weight,
        gamma=float(cfg.get("future_focal_gamma", cfg.get("focal_gamma", 0.0))),
        sample_weight=frame_weight.reshape(-1),
    )
    speed_loss = F.mse_loss(outputs["future_speed"], targets["future_speed"])
    steer_scale = float(cfg["steer_scale"])
    steer_loss = F.mse_loss(outputs["future_steer"] * steer_scale, targets["future_steer"] * steer_scale)
    loss = (
        float(cfg["decision_weight"]) * decision_loss
        + float(cfg["future_decision_weight"]) * future_decision_loss
        + float(cfg.get("future_event_weight", 0.0)) * future_event_loss
        + float(cfg.get("future_event_time_weight", 0.0)) * future_event_time_loss
        + float(cfg.get("future_event_time_bin_weight", 0.0)) * future_event_time_bin_loss
        + float(cfg.get("future_event_time_class_weight", 0.0)) * future_event_time_class_loss
        + float(cfg.get("future_event_time_bin_class_weight", 0.0)) * future_event_time_bin_class_loss
        + float(cfg["speed_weight"]) * speed_loss
        + float(cfg["steer_weight"]) * steer_loss
    )
    parts = {
        "decision_loss": float(decision_loss.detach().cpu()),
        "future_decision_loss": float(future_decision_loss.detach().cpu()),
        "future_event_loss": float(future_event_loss.detach().cpu()),
        "future_event_time_loss": float(future_event_time_loss.detach().cpu()),
        "future_event_time_bin_loss": float(future_event_time_bin_loss.detach().cpu()),
        "future_event_time_class_loss": float(future_event_time_class_loss.detach().cpu()),
        "future_event_time_bin_class_loss": float(future_event_time_bin_class_loss.detach().cpu()),
        "speed_loss": float(speed_loss.detach().cpu()),
        "steer_loss": float(steer_loss.detach().cpu()),
    }
    return loss, parts


def _cross_entropy_or_focal(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 0.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    if gamma <= 0:
        loss = ce
    else:
        pt = torch.exp(-ce)
        loss = (1.0 - pt) ** gamma * ce
    if sample_weight is None:
        return loss.mean()
    sample_weight = sample_weight.to(device=loss.device, dtype=loss.dtype)
    return (loss * sample_weight).sum() / torch.clamp(sample_weight.sum(), min=1.0)


def _future_event_targets(seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    event, first_idx = _first_lane_change_event(seq)
    event = event.to(seq.device)
    first_idx = first_idx.to(seq.device)
    has_event = event != 1
    denom = max(int(seq.shape[1]) - 1, 1)
    event_time = torch.clamp(first_idx.float(), min=0.0) / float(denom)
    return event.long(), event_time, has_event


def _masked_time_bin_loss(logits: torch.Tensor, target_bin: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sigma = 5.0
    per_sample = _soft_time_distribution_loss(logits, target_bin, sigma)
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    return (per_sample * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def _masked_class_time_loss(
    time_by_class: torch.Tensor,
    event_label: torch.Tensor,
    event_time: torch.Tensor,
    mask: torch.Tensor,
    future_len: int,
) -> torch.Tensor:
    pred = time_by_class.gather(1, event_label.view(-1, 1)).squeeze(1)
    denom = float(max(future_len - 1, 1))
    pred_frame = pred * denom
    target_frame = event_time * denom
    per_sample = F.smooth_l1_loss(pred_frame, target_frame, reduction="none") / denom
    mask = mask.to(device=time_by_class.device, dtype=time_by_class.dtype)
    return (per_sample * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def _masked_class_time_bin_loss(
    logits_by_class: torch.Tensor,
    event_label: torch.Tensor,
    target_bin: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    logits = logits_by_class.gather(
        1,
        event_label.view(-1, 1, 1).expand(-1, 1, logits_by_class.shape[-1]),
    ).squeeze(1)
    sigma = 4.0
    per_sample = _soft_time_distribution_loss(logits, target_bin, sigma)
    mask = mask.to(device=logits_by_class.device, dtype=logits_by_class.dtype)
    return (per_sample * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def _soft_time_distribution_loss(logits: torch.Tensor, target_bin: torch.Tensor, sigma: float) -> torch.Tensor:
    frame_ids = torch.arange(logits.shape[1], device=logits.device, dtype=torch.float32).unsqueeze(0)
    center = target_bin.to(device=logits.device, dtype=torch.float32).unsqueeze(1)
    target = torch.exp(-0.5 * ((frame_ids - center) / max(float(sigma), 1e-6)) ** 2)
    target = target / torch.clamp(target.sum(dim=1, keepdim=True), min=1e-8)
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1)


def _future_sequence_frame_weights(seq: torch.Tensor, window: int, event_window_weight: float) -> torch.Tensor:
    weights = torch.ones_like(seq, dtype=torch.float32)
    non_straight = seq != 1
    weights[non_straight] = max(event_window_weight, 1.0)
    if window <= 0:
        return weights
    _, first_idx = _first_lane_change_event(seq)
    first_idx = first_idx.to(seq.device)
    frame_ids = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
    has_event = first_idx >= 0
    near_event = torch.abs(frame_ids - first_idx.unsqueeze(1)) <= window
    weights[near_event & has_event.unsqueeze(1)] = max(event_window_weight, 1.0)
    return weights


def _tensor_weights(values, device: torch.device) -> torch.Tensor | None:
    if values is None:
        return None
    return torch.as_tensor(values, dtype=torch.float32, device=device)


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
    index = np.load(index_path, allow_pickle=True)
    labels = []
    for shard_path, start, end in zip(index["shards"], index["offsets"][:-1], index["offsets"][1:]):
        data = np.load(str(shard_path), allow_pickle=True)
        labels.append(np.asarray(data["decision_label"][index["indices"][start:end]], dtype=np.int64))
    return np.concatenate(labels) if labels else np.array([], dtype=np.int64)


class MetricsMeter:
    def __init__(self, loss_config: dict | None = None) -> None:
        self.loss_config = loss_config or {}
        self.loss_sum = 0.0
        self.count = 0
        self.confusion = torch.zeros(3, 3, dtype=torch.long)
        self.future_correct = 0
        self.future_count = 0
        self.future_confusion = torch.zeros(3, 3, dtype=torch.long)
        self.event_same_direction = 0
        self.event_true_count = 0
        self.event_pred_count = 0
        self.event_time_abs_error = 0.0
        self.event_time_match_count = 0
        self.speed_sse = 0.0
        self.steer_sse = 0.0
        self.reg_count = 0
        self.loss_parts_sum = {}

    @property
    def loss_avg(self) -> float:
        return self.loss_sum / max(self.count, 1)

    @property
    def decision_macro_f1(self) -> float:
        return self._macro_f1().item()

    def update(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], loss: torch.Tensor, parts: dict) -> None:
        batch_size = int(targets["decision_label"].shape[0])
        self.loss_sum += float(loss.cpu()) * batch_size
        self.count += batch_size
        pred = outputs["decision_logits"].argmax(dim=-1).detach().cpu()
        true = targets["decision_label"].detach().cpu()
        for t, p in zip(true, pred):
            self.confusion[int(t), int(p)] += 1

        future_true = targets["future_decision_sequence"]
        future_pred = _future_prediction_sequence(outputs, self.loss_config, future_true.shape[1])
        self.future_correct += int((future_pred == future_true).sum().detach().cpu())
        self.future_count += int(future_true.numel())
        self._update_event_metrics(future_pred.detach().cpu(), future_true.detach().cpu())

        speed_err = outputs["future_speed"] - targets["future_speed"]
        steer_err = outputs["future_steer"] - targets["future_steer"]
        self.speed_sse += float(torch.sum(speed_err * speed_err).detach().cpu())
        self.steer_sse += float(torch.sum(steer_err * steer_err).detach().cpu())
        self.reg_count += int(speed_err.numel())
        for key, value in parts.items():
            self.loss_parts_sum[key] = self.loss_parts_sum.get(key, 0.0) + value * batch_size

    def compute(self) -> dict:
        precision, recall, f1 = self._prf()
        result = {
            "loss": self.loss_avg,
            "decision_accuracy": float(torch.trace(self.confusion).item() / max(int(self.confusion.sum().item()), 1)),
            "decision_macro_f1": float(f1.mean().item()),
            "decision_precision_LSR": [float(x) for x in precision],
            "decision_recall_LSR": [float(x) for x in recall],
            "decision_f1_LSR": [float(x) for x in f1],
            "future_decision_accuracy": float(self.future_correct / max(self.future_count, 1)),
            "future_event_accuracy": float(torch.trace(self.future_confusion).item() / max(int(self.future_confusion.sum().item()), 1)),
            "future_event_macro_f1": float(self._macro_f1_from_confusion(self.future_confusion).item()),
            "future_event_recall_LSR": [float(x) for x in self._prf_from_confusion(self.future_confusion)[1]],
            "future_event_confusion_matrix": self.future_confusion.tolist(),
            "lane_change_event_recall": float(self.event_same_direction / max(self.event_true_count, 1)),
            "lane_change_event_precision": float(self.event_same_direction / max(self.event_pred_count, 1)),
            "lane_change_event_time_mae_frames": float(self.event_time_abs_error / max(self.event_time_match_count, 1)),
            "speed_rmse": float((self.speed_sse / max(self.reg_count, 1)) ** 0.5),
            "steer_rmse_rad": float((self.steer_sse / max(self.reg_count, 1)) ** 0.5),
            "confusion_matrix": self.confusion.tolist(),
        }
        for key, value in self.loss_parts_sum.items():
            result[key] = value / max(self.count, 1)
        return result

    def _prf(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._prf_from_confusion(self.confusion)

    @staticmethod
    def _prf_from_confusion(confusion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cm = confusion.float()
        tp = torch.diag(cm)
        precision = tp / torch.clamp(cm.sum(dim=0), min=1.0)
        recall = tp / torch.clamp(cm.sum(dim=1), min=1.0)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)
        return precision, recall, f1

    def _macro_f1(self) -> torch.Tensor:
        return self._macro_f1_from_confusion(self.confusion)

    @staticmethod
    def _macro_f1_from_confusion(confusion: torch.Tensor) -> torch.Tensor:
        return MetricsMeter._prf_from_confusion(confusion)[2].mean()

    def _update_event_metrics(self, future_pred: torch.Tensor, future_true: torch.Tensor) -> None:
        true_event, true_time = _first_lane_change_event(future_true)
        pred_event, pred_time = _first_lane_change_event(future_pred)
        for t, p, tt, pt in zip(true_event, pred_event, true_time, pred_time):
            self.future_confusion[int(t), int(p)] += 1
            if int(t) != 1:
                self.event_true_count += 1
            if int(p) != 1:
                self.event_pred_count += 1
            if int(t) != 1 and int(t) == int(p):
                self.event_same_direction += 1
                self.event_time_abs_error += abs(float(tt) - float(pt))
                self.event_time_match_count += 1


def _first_lane_change_event(seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    non_straight = seq != 1
    has_event = non_straight.any(dim=1)
    first_idx = torch.argmax(non_straight.to(torch.long), dim=1)
    event = seq[torch.arange(seq.shape[0]), first_idx].clone()
    event[~has_event] = 1
    first_idx[~has_event] = -1
    return event.cpu(), first_idx.cpu()


def _future_prediction_sequence(outputs: dict[str, torch.Tensor], cfg: dict, future_len: int) -> torch.Tensor:
    if cfg.get("future_eval_mode") != "event_head_sequence" or "future_event_logits" not in outputs:
        return outputs["future_decision_logits"].argmax(dim=-1)
    event = outputs["future_event_logits"].argmax(dim=-1)
    time_mode = cfg.get("future_time_eval_mode")
    if time_mode == "class_time_bin" and "future_event_time_bin_by_class" in outputs:
        logits = outputs["future_event_time_bin_by_class"].gather(
            1,
            event.view(-1, 1, 1).expand(-1, 1, future_len),
        ).squeeze(1)
        event_idx = logits.argmax(dim=-1)
    elif time_mode == "class_time_expectation" and "future_event_time_bin_by_class" in outputs:
        logits = outputs["future_event_time_bin_by_class"].gather(
            1,
            event.view(-1, 1, 1).expand(-1, 1, future_len),
        ).squeeze(1)
        probs = F.softmax(logits, dim=-1)
        frame_ids = torch.arange(future_len, device=probs.device, dtype=probs.dtype)
        event_idx = torch.round((probs * frame_ids.unsqueeze(0)).sum(dim=1)).long()
    elif time_mode == "class_hybrid" and "future_event_time_by_class" in outputs and "future_event_time_bin_by_class" in outputs:
        event_time = outputs["future_event_time_by_class"].gather(1, event.view(-1, 1)).squeeze(1)
        event_time = torch.clamp(event_time, 0.0, 1.0)
        scalar_idx = event_time * float(max(future_len - 1, 1))
        logits = outputs["future_event_time_bin_by_class"].gather(
            1,
            event.view(-1, 1, 1).expand(-1, 1, future_len),
        ).squeeze(1)
        probs = F.softmax(logits, dim=-1)
        frame_ids = torch.arange(future_len, device=probs.device, dtype=probs.dtype)
        expected_idx = (probs * frame_ids.unsqueeze(0)).sum(dim=1)
        alpha = float(cfg.get("future_time_hybrid_scalar_weight", 0.75))
        event_idx = torch.round(alpha * scalar_idx + (1.0 - alpha) * expected_idx).long()
    elif time_mode == "class_scalar" and "future_event_time_by_class" in outputs:
        event_time = outputs["future_event_time_by_class"].gather(1, event.view(-1, 1)).squeeze(1)
        event_time = torch.clamp(event_time, 0.0, 1.0)
        event_idx = torch.round(event_time * float(max(future_len - 1, 1))).long()
    elif time_mode == "time_bin" and "future_event_time_logits" in outputs:
        event_idx = outputs["future_event_time_logits"].argmax(dim=-1)
    elif time_mode == "time_expectation" and "future_event_time_logits" in outputs:
        probs = F.softmax(outputs["future_event_time_logits"], dim=-1)
        frame_ids = torch.arange(future_len, device=probs.device, dtype=probs.dtype)
        event_idx = torch.round((probs * frame_ids.unsqueeze(0)).sum(dim=1)).long()
    elif time_mode == "hybrid" and "future_event_time_logits" in outputs:
        event_time = torch.clamp(outputs["future_event_time"], 0.0, 1.0)
        scalar_idx = event_time * float(max(future_len - 1, 1))
        probs = F.softmax(outputs["future_event_time_logits"], dim=-1)
        frame_ids = torch.arange(future_len, device=probs.device, dtype=probs.dtype)
        expected_idx = (probs * frame_ids.unsqueeze(0)).sum(dim=1)
        alpha = float(cfg.get("future_time_hybrid_scalar_weight", 0.75))
        event_idx = torch.round(alpha * scalar_idx + (1.0 - alpha) * expected_idx).long()
    else:
        event_time = torch.clamp(outputs["future_event_time"], 0.0, 1.0)
        event_idx = torch.round(event_time * float(max(future_len - 1, 1))).long()
    event_idx = torch.round(event_idx.float() * float(cfg.get("future_time_index_scale", 1.0))).long()
    event_idx = torch.clamp(event_idx + int(cfg.get("future_time_index_offset", 0)), min=0, max=future_len - 1)
    pred = torch.ones((event.shape[0], future_len), dtype=torch.long, device=event.device)
    frame_ids = torch.arange(future_len, device=event.device).unsqueeze(0)
    event_mask = event != 1
    pred[frame_ids.expand_as(pred) >= event_idx.unsqueeze(1)] = event.unsqueeze(1).expand_as(pred)[
        frame_ids.expand_as(pred) >= event_idx.unsqueeze(1)
    ]
    pred[~event_mask] = 1
    return pred


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
