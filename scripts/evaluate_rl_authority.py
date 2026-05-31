from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_work3_authority import _load_human_model, _resolve_device
from scripts.train_rl_authority import RLMeter
from src.data.intent_dataset import HighDIntentDataset
from src.policies.machine_intent_policy import MachineIntentPolicy, MachineIntentPolicyConfig
from src.rl import AuthorityObservationConfig, AuthorityRLEnv, AuthorityRewardConfig, TransformerSACAuthority, TransformerSACConfig
from src.rl.authority_observation import build_authority_observation
from src.trust import BidirectionalTrustConfig, BidirectionalTrustEstimator, IT2TSKAuthorityConfig, IntervalType2TSKAuthority


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Transformer-SAC RL authority optimizer.")
    parser.add_argument("--config", default="configs/rl_authority.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-batches", type=int, default=8)
    args = parser.parse_args()

    config = _load_config(ROOT / args.config)
    device = _resolve_device(config["training"]["device"])
    human = _load_human_model(config, device)
    human.eval()
    machine = MachineIntentPolicy(MachineIntentPolicyConfig(**config["machine_policy"]))
    trust = BidirectionalTrustEstimator(BidirectionalTrustConfig(**config["trust"]))
    authority = IntervalType2TSKAuthority(IT2TSKAuthorityConfig(**config["authority"]))
    env = AuthorityRLEnv(trust, AuthorityRewardConfig(**config["reward"]))
    agent = TransformerSACAuthority(TransformerSACConfig(**config["rl"])).to(device)
    checkpoint = torch.load(ROOT / config["training"]["checkpoint_dir"] / "best.pt", map_location=device)
    agent.load_state_dict(checkpoint["agent_state"])
    agent.eval()
    loader = _make_loader(config, args.split)
    obs_cfg = AuthorityObservationConfig()

    ref_meter = RLMeter()
    rl_meter = RLMeter()
    with torch.no_grad():
        iterator = tqdm(loader, total=args.max_batches, desc="rl-eval", leave=False)
        for batch_idx, batch in enumerate(iterator):
            if batch_idx >= args.max_batches:
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
            obs = build_authority_observation(trust_outputs, authority_outputs, human_outputs, machine_outputs, obs_cfg)
            zero_delta = torch.zeros_like(authority_outputs["authority_ref"])
            ref_step = env.step(inputs, human_outputs, machine_outputs, trust_outputs, authority_outputs["authority_ref"], zero_delta)
            _, _, mu = agent.sample_action(obs)
            rl_delta = torch.clamp(agent.config.max_delta * torch.tanh(mu), -agent.config.max_delta, agent.config.max_delta)
            rl_step = env.step(inputs, human_outputs, machine_outputs, trust_outputs, authority_outputs["authority_ref"], rl_delta)
            ref_meter.update(ref_step, zero_delta, authority_outputs["authority_ref"], {})
            rl_meter.update(rl_step, rl_delta, authority_outputs["authority_ref"], {})

    result = {"reference_authority": ref_meter.compute(), "rl_authority": rl_meter.compute()}
    out_dir = ROOT / config["training"]["checkpoint_dir"]
    with (out_dir / f"{args.split}_eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


def _make_loader(config: dict, split: str) -> DataLoader:
    dataset = HighDIntentDataset(ROOT / config["data"][f"{split}_index"], include_meta=False)
    return DataLoader(dataset, batch_size=int(config["data"]["batch_size"]), shuffle=False, num_workers=0)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    main()
