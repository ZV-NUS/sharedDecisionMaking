from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import torch
import yaml
import h5py
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_work3_authority import _load_human_model
from src.data.intent_dataset import HighDIntentDataset
from src.policies.machine_intent_policy import MachineIntentPolicy, MachineIntentPolicyConfig
from src.rl import AuthorityObservationConfig, AuthorityRLEnv, AuthorityRewardConfig, TransformerSACAuthority, TransformerSACConfig
from src.rl.authority_observation import build_authority_observation
from src.trust import BidirectionalTrustConfig, BidirectionalTrustEstimator, IT2TSKAuthorityConfig, IntervalType2TSKAuthority


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Transformer-SAC shared authority optimizer.")
    parser.add_argument("--config", default="configs/rl_authority.yaml")
    args = parser.parse_args()

    config = _load_config(ROOT / args.config)
    _set_seed(int(config["training"]["seed"]))
    device = _resolve_device(config["training"]["device"])
    ckpt_dir = ROOT / config["training"]["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    human = _load_human_model(config, device)
    human.eval()
    machine = MachineIntentPolicy(MachineIntentPolicyConfig(**config["machine_policy"]))
    trust = BidirectionalTrustEstimator(BidirectionalTrustConfig(**config["trust"]))
    authority = IntervalType2TSKAuthority(IT2TSKAuthorityConfig(**config["authority"]))
    env = AuthorityRLEnv(trust, AuthorityRewardConfig(**config["reward"]))
    agent = TransformerSACAuthority(TransformerSACConfig(**config["rl"])).to(device)
    agent.training_config = config["training"]
    obs_cfg = AuthorityObservationConfig()

    hard_indices = None
    if bool(config["data"].get("hard_case_sampler", False)):
        hard_indices = _build_hard_case_indices(config, human, machine, trust, authority, env, device)
        print(json.dumps({"hard_case_count": len(hard_indices), "hard_case_preview": hard_indices[:10]}, indent=2))
    train_loader = _make_loader(config, "train", subset_indices=hard_indices)
    val_loader = _make_loader(config, "val")
    history = []
    best_reward = -float("inf")
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics = run_epoch(
            human, machine, trust, authority, env, agent, obs_cfg, train_loader, device, True, int(config["training"]["max_train_batches"])
        )
        val_metrics = run_epoch(
            human, machine, trust, authority, env, agent, obs_cfg, val_loader, device, False, int(config["training"]["max_val_batches"])
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, indent=2))
        if val_metrics["reward_mean"] > best_reward:
            best_reward = val_metrics["reward_mean"]
            torch.save({"agent_state": agent.state_dict(), "config": config, "epoch": epoch, "val": val_metrics}, ckpt_dir / "best.pt")
        with (ckpt_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)


def run_epoch(
    human: torch.nn.Module,
    machine: MachineIntentPolicy,
    trust: BidirectionalTrustEstimator,
    authority: IntervalType2TSKAuthority,
    env: AuthorityRLEnv,
    agent: TransformerSACAuthority,
    obs_cfg: AuthorityObservationConfig,
    loader: DataLoader,
    device: torch.device,
    train: bool,
    max_batches: int,
) -> dict:
    meter = RLMeter()
    iterator = tqdm(loader, total=max_batches, desc="rl-train" if train else "rl-val", leave=False)
    for batch_idx, batch in enumerate(iterator):
        if batch_idx >= max_batches:
            break
        inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
        with torch.no_grad():
            human_outputs = human(inputs)
            machine_outputs = machine.predict(inputs)
            trust_outputs = trust.estimate(inputs, human_outputs, machine_outputs)
            authority_outputs = authority.infer(
                trust_outputs["trust_machine_to_human"],
                trust_outputs["trust_human_to_machine"],
                environment_urgency=trust_outputs["environment_urgency"],
            )
            obs = build_authority_observation(trust_outputs, authority_outputs, human_outputs, machine_outputs, obs_cfg)

        if train:
            action, _, _ = agent.sample_action(obs)
        else:
            with torch.no_grad():
                _, _, mu = agent.sample_action(obs)
                action = torch.clamp(agent.config.max_delta * torch.tanh(mu), -agent.config.max_delta, agent.config.max_delta)

        with torch.no_grad():
            step = env.step(
                inputs,
                human_outputs,
                machine_outputs,
                trust_outputs,
                authority_outputs["authority_ref"],
                action,
            )
        update_metrics = {}
        if train:
            if bool(agent.training_config.get("oracle_supervision", False)):
                with torch.no_grad():
                    oracle_action, oracle_advantage = _oracle_delta_search(
                        env,
                        inputs,
                        human_outputs,
                        machine_outputs,
                        trust_outputs,
                        authority_outputs["authority_ref"],
                        float(agent.config.max_delta),
                    )
                update_metrics = agent.supervised_update(obs, oracle_action, torch.clamp(oracle_advantage, min=0.0))
                update_metrics["oracle_advantage"] = float(oracle_advantage.mean().detach().cpu())
                update_metrics["oracle_positive_rate"] = float((oracle_advantage > 1e-4).to(torch.float32).mean().detach().cpu())
            else:
                update_metrics = agent.update(obs, action.detach(), step["reward"])
        meter.update(step, action, authority_outputs["authority_ref"], update_metrics)
        iterator.set_postfix(reward=f"{meter.reward_mean:.3f}", risk=f"{meter.risk_mean:.3f}")
    return meter.compute()


def _oracle_delta_search(
    env: AuthorityRLEnv,
    inputs: dict[str, torch.Tensor],
    human_outputs: dict[str, torch.Tensor],
    machine_outputs: dict[str, torch.Tensor],
    trust_outputs: dict[str, torch.Tensor],
    authority_ref: torch.Tensor,
    max_delta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, future_len = authority_ref.shape
    device = authority_ref.device
    base_delta = torch.zeros_like(authority_ref)
    base_step = env.step(inputs, human_outputs, machine_outputs, trust_outputs, authority_ref, base_delta)
    best_reward = base_step["reward"]
    best_delta = base_delta
    candidates = torch.tensor([-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0], device=device) * max_delta
    urgency = torch.clamp(trust_outputs["environment_urgency"], 0.0, 1.0)
    human_risk = torch.clamp(trust_outputs["human_risk"], 0.0, 1.0)
    profile = torch.clamp(0.35 + 0.65 * torch.maximum(urgency, human_risk), 0.35, 1.0)
    for scalar in candidates:
        candidate_delta = scalar * profile
        candidate_step = env.step(inputs, human_outputs, machine_outputs, trust_outputs, authority_ref, candidate_delta)
        better = candidate_step["reward"] > best_reward
        best_reward = torch.where(better, candidate_step["reward"], best_reward)
        best_delta = torch.where(better.view(batch, 1), candidate_delta, best_delta)
    return best_delta.view(batch, future_len), best_reward - base_step["reward"]


class RLMeter:
    def __init__(self) -> None:
        self.count = 0
        self.reward = 0.0
        self.risk = 0.0
        self.max_risk = 0.0
        self.delta_abs = 0.0
        self.authority = 0.0
        self.parts: dict[str, float] = {}

    @property
    def reward_mean(self) -> float:
        return self.reward / max(self.count, 1)

    @property
    def risk_mean(self) -> float:
        return self.risk / max(self.count, 1)

    def update(self, step: dict[str, torch.Tensor], action: torch.Tensor, authority_ref: torch.Tensor, update_metrics: dict[str, float]) -> None:
        batch = int(step["reward"].shape[0])
        self.count += batch
        self.reward += float(step["reward"].sum().detach().cpu())
        self.risk += float(step["shared_risk"].mean(dim=1).sum().detach().cpu())
        self.max_risk = max(self.max_risk, float(step["shared_risk"].max().detach().cpu()))
        self.delta_abs += float(action.abs().mean(dim=1).sum().detach().cpu())
        self.authority += float(step["authority_rl"].mean(dim=1).sum().detach().cpu())
        for key in ("safety_cost", "efficiency_cost", "comfort_cost", "smooth_cost", "reference_cost", "trust_cost"):
            self.parts[key] = self.parts.get(key, 0.0) + float(step[key].sum().detach().cpu())
        for key, value in update_metrics.items():
            self.parts[key] = self.parts.get(key, 0.0) + float(value) * batch

    def compute(self) -> dict:
        result = {
            "reward_mean": float(self.reward_mean),
            "shared_risk_mean": float(self.risk_mean),
            "shared_risk_max": float(self.max_risk),
            "mean_abs_delta_authority": float(self.delta_abs / max(self.count, 1)),
            "authority_rl_mean": float(self.authority / max(self.count, 1)),
        }
        result.update({key: float(value / max(self.count, 1)) for key, value in self.parts.items()})
        return result


def _make_loader(config: dict, split: str, subset_indices: list[int] | None = None) -> DataLoader:
    dataset = HighDIntentDataset(ROOT / config["data"][f"{split}_index"], include_meta=False)
    sampler = None
    if subset_indices is not None:
        sampler = SubsetRandomSampler(subset_indices)
    elif split == "train" and bool(config["data"].get("risk_sampler", False)):
        sampler = _make_risk_sampler(
            ROOT / config["data"][f"{split}_index"],
            pool_size=int(config["data"].get("risk_sampler_pool", 12000)),
            threshold=float(config["data"].get("risk_sampler_threshold", 0.22)),
            num_samples=int(config["training"]["max_train_batches"]) * int(config["data"]["batch_size"]),
        )
    return DataLoader(
        dataset,
        batch_size=int(config["data"]["batch_size"]),
        shuffle=(split == "train" and sampler is None),
        sampler=sampler,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=False,
        drop_last=False,
    )


def _build_hard_case_indices(
    config: dict,
    human: torch.nn.Module,
    machine: MachineIntentPolicy,
    trust: BidirectionalTrustEstimator,
    authority: IntervalType2TSKAuthority,
    env: AuthorityRLEnv,
    device: torch.device,
) -> list[int]:
    pool = int(config["data"].get("hard_case_pool", 4096))
    top_k = int(config["data"].get("hard_case_top_k", 1536))
    batch_size = int(config["data"]["batch_size"])
    dataset = HighDIntentDataset(ROOT / config["data"]["train_index"], include_meta=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    scores = []
    seen = 0
    human.eval()
    with torch.no_grad():
        iterator = tqdm(loader, total=max(1, (min(pool, len(dataset)) + batch_size - 1) // batch_size), desc="hard-cases", leave=False)
        for batch in iterator:
            if seen >= pool:
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
            zero_delta = torch.zeros_like(authority_outputs["authority_ref"])
            ref_step = env.step(
                inputs,
                human_outputs,
                machine_outputs,
                trust_outputs,
                authority_outputs["authority_ref"],
                zero_delta,
            )
            ref_risk = ref_step["shared_risk"].mean(dim=1)
            max_risk = ref_step["shared_risk"].amax(dim=1)
            urgency = trust_outputs["environment_urgency"].mean(dim=1)
            disagreement = trust_outputs["intent_disagreement"].mean(dim=1)
            human_risk = trust_outputs["human_risk"].mean(dim=1)
            score = 2.8 * ref_risk + 1.8 * max_risk + 0.8 * human_risk + 0.7 * urgency + 0.5 * disagreement
            for local_i, value in enumerate(score.detach().cpu().tolist()):
                global_i = seen + local_i
                if global_i >= pool:
                    break
                scores.append((float(value), global_i))
            seen += int(score.shape[0])
            iterator.set_postfix(max_score=f"{max(v for v, _ in scores):.3f}")
    scores.sort(key=lambda item: item[0], reverse=True)
    selected = [idx for _, idx in scores[: min(top_k, len(scores))]]
    if not selected:
        selected = list(range(min(pool, len(dataset))))
    return selected


def _make_risk_sampler(index_path: Path, pool_size: int, threshold: float, num_samples: int) -> WeightedRandomSampler:
    with h5py.File(index_path, "r") as h5:
        n = int(h5["decision_label"].shape[0])
        m = min(n, pool_size)
        front_risk = torch.as_tensor(h5["risk_history"][:m, -1, 3], dtype=torch.float32)
        thw = torch.as_tensor(h5["risk_history"][:m, -1, 1], dtype=torch.float32)
        labels = torch.as_tensor(h5["decision_label"][:m], dtype=torch.long)
    high_risk = front_risk > threshold
    close_headway = thw < 1.2
    lane_change = labels != 1
    weights = torch.ones((m,), dtype=torch.float32)
    weights += 5.0 * high_risk.float()
    weights += 3.0 * close_headway.float()
    weights += 2.0 * lane_change.float()
    return WeightedRandomSampler(weights=weights, num_samples=min(num_samples, m), replacement=True)


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
