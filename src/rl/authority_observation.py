from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AuthorityObservationConfig:
    max_speed_mps: float = 35.0
    max_abs_steer_rad: float = 0.45
    max_abs_accel_mps2: float = 4.0
    max_front_gap_m: float = 80.0
    max_ttc_s: float = 20.0


def build_authority_observation(
    trust_outputs: dict[str, torch.Tensor],
    authority_outputs: dict[str, torch.Tensor],
    human_outputs: dict[str, torch.Tensor],
    machine_outputs: dict[str, torch.Tensor],
    config: AuthorityObservationConfig | None = None,
) -> torch.Tensor:
    cfg = config or AuthorityObservationConfig()
    human_speed = human_outputs["future_speed"]
    machine_speed = machine_outputs["future_speed"]
    human_steer = human_outputs["future_steer"]
    machine_steer = machine_outputs["future_steer"]
    human_accel = _future_accel(human_outputs)
    machine_accel = _future_accel(machine_outputs)
    decision_diff = (
        trust_outputs["human_decision_sequence"] != trust_outputs["machine_decision_sequence"]
    ).to(dtype=human_speed.dtype)

    features = [
        authority_outputs["authority_ref"],
        trust_outputs["trust_machine_to_human"],
        trust_outputs["trust_human_to_machine"],
        trust_outputs["human_risk"],
        trust_outputs["environment_urgency"],
        trust_outputs["intent_disagreement"],
        torch.clamp(trust_outputs["front_gap"] / cfg.max_front_gap_m, 0.0, 1.5),
        torch.clamp(trust_outputs["front_relative_speed"] / cfg.max_speed_mps, -1.0, 1.0),
        torch.clamp(trust_outputs["front_ttc"] / cfg.max_ttc_s, 0.0, 1.0),
        torch.clamp(human_speed / cfg.max_speed_mps, 0.0, 1.5),
        torch.clamp(human_steer / cfg.max_abs_steer_rad, -1.5, 1.5),
        torch.clamp(human_accel / cfg.max_abs_accel_mps2, -1.5, 1.5),
        torch.clamp(machine_speed / cfg.max_speed_mps, 0.0, 1.5),
        torch.clamp(machine_steer / cfg.max_abs_steer_rad, -1.5, 1.5),
        torch.clamp(machine_accel / cfg.max_abs_accel_mps2, -1.5, 1.5),
        decision_diff,
    ]
    return torch.stack(features, dim=-1)


def _future_accel(outputs: dict[str, torch.Tensor], frame_rate: float = 25.0) -> torch.Tensor:
    if "future_acceleration" in outputs:
        return outputs["future_acceleration"]
    speed = outputs["future_speed"]
    accel = torch.zeros_like(speed)
    if speed.shape[1] > 1:
        accel[:, 1:] = (speed[:, 1:] - speed[:, :-1]) * frame_rate
        accel[:, 0] = accel[:, 1]
    return accel
