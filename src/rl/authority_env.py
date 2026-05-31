from __future__ import annotations

from dataclasses import dataclass

import torch

from src.trust.bidirectional_trust import BidirectionalTrustEstimator
from src.trust.shared_intent import blend_human_machine_intent


@dataclass(frozen=True)
class AuthorityRewardConfig:
    target_speed_mps: float = 31.0
    safety_weight: float = 10.0
    efficiency_weight: float = 1.0
    comfort_weight: float = 0.15
    smooth_weight: float = 0.60
    reference_weight: float = 0.20
    trust_weight: float = 0.20
    collision_risk_threshold: float = 0.95
    max_delta: float = 0.35


class AuthorityRLEnv:
    """One-step highD planning environment for RL authority optimization."""

    def __init__(
        self,
        trust_estimator: BidirectionalTrustEstimator,
        reward_config: AuthorityRewardConfig | None = None,
    ) -> None:
        self.trust_estimator = trust_estimator
        self.reward_config = reward_config or AuthorityRewardConfig()

    def step(
        self,
        inputs: dict[str, torch.Tensor],
        human_outputs: dict[str, torch.Tensor],
        machine_outputs: dict[str, torch.Tensor],
        trust_outputs: dict[str, torch.Tensor],
        authority_ref: torch.Tensor,
        delta_authority: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.reward_config
        delta = torch.clamp(delta_authority, -cfg.max_delta, cfg.max_delta)
        authority_rl = torch.clamp(authority_ref + delta, 0.0, 1.0)
        shared_outputs = blend_human_machine_intent(human_outputs, machine_outputs, authority_rl)
        rollout = self.trust_estimator.rollout_intent(inputs, shared_outputs)
        risk = self.trust_estimator._traffic_risk(inputs, rollout)

        safety_cost = risk.mean(dim=1) + 3.0 * (risk.amax(dim=1) > cfg.collision_risk_threshold).to(dtype=risk.dtype)
        efficiency_cost = torch.mean(((cfg.target_speed_mps - shared_outputs["future_speed"]) / cfg.target_speed_mps) ** 2, dim=1)
        accel = shared_outputs["future_acceleration"]
        steer = shared_outputs["future_steer"]
        steer_rate = torch.zeros_like(steer)
        if steer.shape[1] > 1:
            steer_rate[:, 1:] = steer[:, 1:] - steer[:, :-1]
        comfort_cost = torch.mean(0.05 * accel**2 + 12.0 * steer_rate**2, dim=1)
        smooth_cost = torch.mean((authority_rl[:, 1:] - authority_rl[:, :-1]) ** 2, dim=1)
        ref_cost = torch.mean((authority_rl - authority_ref) ** 2, dim=1)
        trust_target = trust_outputs["trust_machine_to_human"] / torch.clamp(
            trust_outputs["trust_machine_to_human"] + trust_outputs["trust_human_to_machine"],
            min=1e-6,
        )
        trust_cost = torch.mean((authority_rl - trust_target) ** 2, dim=1)

        reward = -(
            cfg.safety_weight * safety_cost
            + cfg.efficiency_weight * efficiency_cost
            + cfg.comfort_weight * comfort_cost
            + cfg.smooth_weight * smooth_cost
            + cfg.reference_weight * ref_cost
            + cfg.trust_weight * trust_cost
        )
        return {
            "reward": reward,
            "authority_rl": authority_rl,
            "shared_risk": risk,
            "safety_cost": safety_cost,
            "efficiency_cost": efficiency_cost,
            "comfort_cost": comfort_cost,
            "smooth_cost": smooth_cost,
            "reference_cost": ref_cost,
            "trust_cost": trust_cost,
        }
