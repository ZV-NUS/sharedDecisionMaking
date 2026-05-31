from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_work3_authority import _load_human_model, _make_loader, _resolve_device
from src.policies.machine_intent_policy import MachineIntentPolicy, MachineIntentPolicyConfig
from src.trust import BidirectionalTrustConfig, BidirectionalTrustEstimator, IT2TSKAuthorityConfig, IntervalType2TSKAuthority
from src.trust.authority_optimizer import AuthorityObjectiveWeights, BayesianCEMConfig, SurrogateAssistedRuleOptimizer
from src.trust.shared_intent import blend_human_machine_intent


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline optimize Work3 IT2-TSK fuzzy authority rules.")
    parser.add_argument("--config", default="configs/work3_authority_optimize.yaml")
    args = parser.parse_args()

    opt_config = _load_config(ROOT / args.config)
    base_config = _load_config(ROOT / opt_config["base_config"])
    base_config["evaluation"]["split"] = opt_config["optimization"]["split"]
    base_config["evaluation"]["max_batches"] = opt_config["optimization"]["max_batches"]
    _set_seed(int(opt_config["optimization"]["seed"]))
    device = _resolve_device(base_config["evaluation"]["device"])
    out_dir = ROOT / opt_config["optimization"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    human = _load_human_model(base_config, device)
    human.eval()
    machine = MachineIntentPolicy(MachineIntentPolicyConfig(**base_config["machine_policy"]))
    trust = BidirectionalTrustEstimator(BidirectionalTrustConfig(**base_config["trust"]))
    authority = IntervalType2TSKAuthority(IT2TSKAuthorityConfig(**base_config["authority"]))
    batches = _precompute_batches(base_config, human, machine, trust, device)

    weights = AuthorityObjectiveWeights(
        safety=float(opt_config["objective_weights"]["safety"]),
        efficiency=float(opt_config["objective_weights"]["efficiency"]),
        comfort=float(opt_config["objective_weights"]["comfort"]),
        smooth=float(opt_config["objective_weights"]["smooth"]),
    )
    optimizer = SurrogateAssistedRuleOptimizer(
        BayesianCEMConfig(
            iterations=int(opt_config["optimization"]["iterations"]),
            candidates=int(opt_config["optimization"]["candidates"]),
            elite_fraction=float(opt_config["optimization"]["elite_fraction"]),
            init_std=float(opt_config["optimization"]["init_std"]),
            min_std=float(opt_config["optimization"]["min_std"]),
            seed=int(opt_config["optimization"]["seed"]),
        )
    )
    target_speed = float(opt_config["objective_weights"]["target_speed_mps"])

    def evaluate(vector: torch.Tensor) -> dict[str, float]:
        candidate = authority.with_parameters(vector)
        return _evaluate_candidate(candidate, trust, batches, target_speed)

    result = optimizer.optimize(authority.parameter_vector(), evaluate, weights=weights)
    serializable = {
        "best_score": float(result["best_score"]),
        "best_metrics": result["best_metrics"],
        "best_vector": [float(x) for x in result["best_vector"].detach().cpu().numpy()],
        "history": result["history"],
        "method": "surrogate-assisted CEM scaffold for MOBO-EHVI rule optimization",
    }
    with (out_dir / "optimized_rules.json").open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(json.dumps(serializable, indent=2))


def _precompute_batches(
    config: dict,
    human: torch.nn.Module,
    machine: MachineIntentPolicy,
    trust: BidirectionalTrustEstimator,
    device: torch.device,
) -> list[dict]:
    loader = _make_loader(config)
    max_batches = int(config["evaluation"]["max_batches"])
    records = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
            human_outputs = human(inputs)
            machine_outputs = machine.predict(inputs)
            trust_outputs = trust.estimate(inputs, human_outputs, machine_outputs)
            records.append(
                {
                    "inputs": inputs,
                    "human_outputs": human_outputs,
                    "machine_outputs": machine_outputs,
                    "trust_outputs": trust_outputs,
                }
            )
    return records


def _evaluate_candidate(
    authority: IntervalType2TSKAuthority,
    trust: BidirectionalTrustEstimator,
    batches: list[dict],
    target_speed: float,
) -> dict[str, float]:
    safety = 0.0
    efficiency = 0.0
    comfort = 0.0
    smooth = 0.0
    count = 0
    for batch in batches:
        authority_outputs = authority.infer(
            batch["trust_outputs"]["trust_machine_to_human"],
            batch["trust_outputs"]["trust_human_to_machine"],
            environment_urgency=batch["trust_outputs"]["environment_urgency"],
        )
        shared = blend_human_machine_intent(
            batch["human_outputs"],
            batch["machine_outputs"],
            authority_outputs["authority_ref"],
        )
        rollout = trust.rollout_intent(batch["inputs"], shared)
        risk = trust._traffic_risk(batch["inputs"], rollout)
        accel = shared["future_acceleration"]
        steer_rate = torch.zeros_like(shared["future_steer"])
        steer_rate[:, 1:] = shared["future_steer"][:, 1:] - shared["future_steer"][:, :-1]
        authority_delta = authority_outputs["authority_ref"][:, 1:] - authority_outputs["authority_ref"][:, :-1]
        safety += float(risk.mean().detach().cpu())
        efficiency += float(torch.mean(((target_speed - shared["future_speed"]) / max(target_speed, 1e-6)) ** 2).detach().cpu())
        comfort += float(torch.mean(0.25 * accel**2 + 10.0 * steer_rate**2).detach().cpu())
        smooth += float(torch.mean(authority_delta**2).detach().cpu())
        count += 1
    denom = max(count, 1)
    return {
        "safety": safety / denom,
        "efficiency": efficiency / denom,
        "comfort": comfort / denom,
        "smooth": smooth / denom,
    }


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
