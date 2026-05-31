from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.intent_dataset import HighDIntentDataset
from src.models.human_intent_transformer import HumanIntentTransformer
from src.policies.machine_intent_policy import MachineIntentPolicy, MachineIntentPolicyConfig
from src.trust import (
    BidirectionalTrustConfig,
    BidirectionalTrustEstimator,
    IT2TSKAuthorityConfig,
    IntervalType2TSKAuthority,
    blend_human_machine_intent,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Work3 bidirectional trust and IT2-TSK authority reference generation.")
    parser.add_argument("--config", default="configs/work3_authority.yaml")
    args = parser.parse_args()

    config = _load_config(ROOT / args.config)
    _set_seed(int(config["evaluation"]["seed"]))
    device = _resolve_device(config["evaluation"]["device"])
    out_dir = ROOT / config["evaluation"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    human = _load_human_model(config, device)
    machine = MachineIntentPolicy(MachineIntentPolicyConfig(**config["machine_policy"]))
    trust = BidirectionalTrustEstimator(BidirectionalTrustConfig(**config["trust"]))
    authority = IntervalType2TSKAuthority(IT2TSKAuthorityConfig(**config["authority"]))
    loader = _make_loader(config)

    meter = Work3Meter()
    example_pool = []
    max_batches = int(config["evaluation"]["max_batches"]) if config["evaluation"].get("max_batches") is not None else None
    human.eval()
    with torch.no_grad():
        iterator = tqdm(loader, total=max_batches, desc="work3", leave=False)
        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
            human_outputs = human(inputs)
            machine_outputs = machine.predict(inputs)
            trust_outputs = trust.estimate(inputs, human_outputs, machine_outputs)
            authority_outputs = authority.infer(
                trust_outputs["trust_machine_to_human"],
                trust_outputs["trust_human_to_machine"],
                environment_urgency=trust_outputs["environment_urgency"],
            )
            shared_outputs = blend_human_machine_intent(
                human_outputs,
                machine_outputs,
                authority_outputs["authority_ref"],
            )
            shared_rollout = trust.rollout_intent(inputs, shared_outputs)
            shared_risk = trust._traffic_risk(inputs, shared_rollout)
            meter.update(trust_outputs, authority_outputs, shared_outputs, shared_risk)
            example_pool.extend(_collect_examples(batch, trust_outputs, authority_outputs, shared_risk))
            iterator.set_postfix(authority=f"{meter.authority_mean:.3f}", risk=f"{meter.shared_risk_mean:.3f}")

    metrics = meter.compute()
    examples = _select_informative_examples(example_pool, limit=6)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (out_dir / "examples.json").open("w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2)
    _plot_examples(examples, out_dir / "authority_examples.png")
    print(json.dumps({"metrics": metrics, "examples": str(out_dir / "examples.json"), "plot": str(out_dir / "authority_examples.png")}, indent=2))


class Work3Meter:
    def __init__(self) -> None:
        self.count = 0
        self.authority_sum = 0.0
        self.authority_machine_dominant = 0
        self.authority_human_dominant = 0
        self.trust_mh_sum = 0.0
        self.trust_hm_sum = 0.0
        self.human_risk_sum = 0.0
        self.intent_diff_sum = 0.0
        self.env_urgency_sum = 0.0
        self.shared_risk_sum = 0.0
        self.shared_risk_max = 0.0
        self.authority_delta_sum = 0.0
        self.authority_delta_count = 0

    @property
    def authority_mean(self) -> float:
        return self.authority_sum / max(self.count, 1)

    @property
    def shared_risk_mean(self) -> float:
        return self.shared_risk_sum / max(self.count, 1)

    def update(
        self,
        trust_outputs: dict[str, torch.Tensor],
        authority_outputs: dict[str, torch.Tensor],
        shared_outputs: dict[str, torch.Tensor],
        shared_risk: torch.Tensor,
    ) -> None:
        authority = authority_outputs["authority_ref"]
        numel = int(authority.numel())
        self.count += numel
        self.authority_sum += float(authority.sum().detach().cpu())
        self.authority_machine_dominant += int((authority < 0.35).sum().detach().cpu())
        self.authority_human_dominant += int((authority > 0.65).sum().detach().cpu())
        self.trust_mh_sum += float(trust_outputs["trust_machine_to_human"].sum().detach().cpu())
        self.trust_hm_sum += float(trust_outputs["trust_human_to_machine"].sum().detach().cpu())
        self.human_risk_sum += float(trust_outputs["human_risk"].sum().detach().cpu())
        self.env_urgency_sum += float(trust_outputs["environment_urgency"].sum().detach().cpu())
        self.intent_diff_sum += float(trust_outputs["intent_disagreement"].sum().detach().cpu())
        self.shared_risk_sum += float(shared_risk.sum().detach().cpu())
        self.shared_risk_max = max(self.shared_risk_max, float(shared_risk.max().detach().cpu()))
        delta = torch.abs(authority[:, 1:] - authority[:, :-1])
        self.authority_delta_sum += float(delta.sum().detach().cpu())
        self.authority_delta_count += int(delta.numel())

    def compute(self) -> dict:
        return {
            "authority_human_mean": float(self.authority_mean),
            "authority_machine_mean": float(1.0 - self.authority_mean),
            "authority_human_dominant_rate": float(self.authority_human_dominant / max(self.count, 1)),
            "authority_machine_dominant_rate": float(self.authority_machine_dominant / max(self.count, 1)),
            "trust_machine_to_human_mean": float(self.trust_mh_sum / max(self.count, 1)),
            "trust_human_to_machine_mean": float(self.trust_hm_sum / max(self.count, 1)),
            "human_rollout_risk_mean": float(self.human_risk_sum / max(self.count, 1)),
            "environment_urgency_mean": float(self.env_urgency_sum / max(self.count, 1)),
            "intent_disagreement_mean": float(self.intent_diff_sum / max(self.count, 1)),
            "shared_rollout_risk_mean": float(self.shared_risk_mean),
            "shared_rollout_risk_max": float(self.shared_risk_max),
            "authority_mean_abs_delta": float(self.authority_delta_sum / max(self.authority_delta_count, 1)),
        }


def _collect_examples(
    batch: dict,
    trust_outputs: dict[str, torch.Tensor],
    authority_outputs: dict[str, torch.Tensor],
    shared_risk: torch.Tensor,
    limit: int | None = None,
) -> list[dict]:
    items = []
    n = int(shared_risk.shape[0]) if limit is None else min(limit, int(shared_risk.shape[0]))
    for i in range(n):
        meta = batch.get("meta", {})
        item = {
            "sample": int(i),
            "authority_human_mean": float(authority_outputs["authority_ref"][i].mean().detach().cpu()),
            "trust_machine_to_human_mean": float(trust_outputs["trust_machine_to_human"][i].mean().detach().cpu()),
            "trust_human_to_machine_mean": float(trust_outputs["trust_human_to_machine"][i].mean().detach().cpu()),
            "human_risk_mean": float(trust_outputs["human_risk"][i].mean().detach().cpu()),
            "shared_risk_mean": float(shared_risk[i].mean().detach().cpu()),
            "environment_urgency_mean": float(trust_outputs["environment_urgency"][i].mean().detach().cpu()),
            "intent_disagreement_mean": float(trust_outputs["intent_disagreement"][i].mean().detach().cpu()),
            "front_gap_min": float(trust_outputs["front_gap"][i].min().detach().cpu()),
            "front_ttc_min": float(trust_outputs["front_ttc"][i].min().detach().cpu()),
            "authority": [float(x) for x in authority_outputs["authority_ref"][i].detach().cpu().numpy()[::5]],
            "trust_machine_to_human": [float(x) for x in trust_outputs["trust_machine_to_human"][i].detach().cpu().numpy()[::5]],
            "trust_human_to_machine": [float(x) for x in trust_outputs["trust_human_to_machine"][i].detach().cpu().numpy()[::5]],
            "environment_urgency": [float(x) for x in trust_outputs["environment_urgency"][i].detach().cpu().numpy()[::5]],
            "front_gap": [float(x) for x in trust_outputs["front_gap"][i].detach().cpu().numpy()[::5]],
        }
        for key in ("recording_id", "vehicle_id", "frame_id"):
            if key in meta:
                value = meta[key][i] if hasattr(meta[key], "__len__") else meta[key]
                item[key] = int(value)
        items.append(item)
    return items


def _select_informative_examples(examples: list[dict], limit: int) -> list[dict]:
    def score(item: dict) -> float:
        return (
            2.4 * float(item["environment_urgency_mean"])
            + 2.0 * float(item["human_risk_mean"])
            + 1.4 * float(item["intent_disagreement_mean"])
            + 1.2 * float(item["shared_risk_mean"])
            + 0.20 * max(0.0, 25.0 - float(item["front_gap_min"])) / 25.0
            + 0.20 * max(0.0, 5.0 - float(item["front_ttc_min"])) / 5.0
        )

    ranked = sorted(examples, key=score, reverse=True)
    seen = set()
    selected = []
    for item in ranked:
        key = (item.get("recording_id", -1), item.get("vehicle_id", -1))
        if key in seen and len(selected) < limit - 1:
            continue
        selected.append(item)
        seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def _plot_examples(examples: list[dict], path: Path) -> None:
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 2.6 * len(examples)), sharex=True)
    if len(examples) == 1:
        axes = [axes]
    for ax, example in zip(axes, examples):
        x = np.arange(len(example["authority"])) * 5 / 25.0
        ax.plot(x, example["authority"], label="human authority", linewidth=2.0)
        ax.plot(x, example["trust_machine_to_human"], label="T_m->h", linewidth=1.5)
        ax.plot(x, example["trust_human_to_machine"], label="T_h->m", linewidth=1.5)
        ax.plot(x, example["environment_urgency"], label="env urgency", linewidth=1.5)
        title = (
            f"authority={example['authority_human_mean']:.2f} "
            f"urg={example['environment_urgency_mean']:.2f} "
            f"human_risk={example['human_risk_mean']:.2f} shared_risk={example['shared_risk_mean']:.2f} "
            f"gap_min={example['front_gap_min']:.1f}m"
        )
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("prediction time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _load_human_model(config: dict, device: torch.device) -> HumanIntentTransformer:
    human_cfg = _load_config(ROOT / config["human_model"]["config"])
    model = HumanIntentTransformer(**human_cfg["model"]).to(device)
    checkpoint = torch.load(ROOT / config["human_model"]["checkpoint"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model


def _make_loader(config: dict) -> DataLoader:
    split = config["evaluation"]["split"]
    dataset = HighDIntentDataset(ROOT / config["data"][f"{split}_index"], include_meta=True)
    return DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=False,
        drop_last=False,
    )


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
