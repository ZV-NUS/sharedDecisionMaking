from __future__ import annotations

import torch


def blend_human_machine_intent(
    human_outputs: dict[str, torch.Tensor],
    machine_outputs: dict[str, torch.Tensor],
    authority_human: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Blend Work1/Work2 continuous intents by reference human authority.

    The returned dictionary keeps the same control fields expected by the
    kinematic environment and later shared-control/RL modules.
    """

    lam = torch.clamp(authority_human, 0.0, 1.0)
    while lam.ndim < human_outputs["future_speed"].ndim:
        lam = lam.unsqueeze(-1)
    speed = lam.squeeze(-1) * human_outputs["future_speed"] + (1.0 - lam.squeeze(-1)) * machine_outputs["future_speed"]
    steer = lam.squeeze(-1) * human_outputs["future_steer"] + (1.0 - lam.squeeze(-1)) * machine_outputs["future_steer"]
    result = {
        "future_speed": speed,
        "future_steer": steer,
        "future_acceleration": _blend_acceleration(human_outputs, machine_outputs, lam.squeeze(-1)),
        "authority_human": authority_human,
    }
    if "future_event_logits" in machine_outputs:
        result["future_event_logits"] = machine_outputs["future_event_logits"]
    if "future_event_time_by_class" in machine_outputs:
        result["future_event_time_by_class"] = machine_outputs["future_event_time_by_class"]
    return result


def _blend_acceleration(
    human_outputs: dict[str, torch.Tensor],
    machine_outputs: dict[str, torch.Tensor],
    authority_human: torch.Tensor,
) -> torch.Tensor:
    human_accel = human_outputs.get("future_acceleration")
    if human_accel is None:
        human_accel = _derive_acceleration(human_outputs["future_speed"])
    machine_accel = machine_outputs.get("future_acceleration")
    if machine_accel is None:
        machine_accel = _derive_acceleration(machine_outputs["future_speed"])
    return authority_human * human_accel + (1.0 - authority_human) * machine_accel


def _derive_acceleration(speed: torch.Tensor, frame_rate: float = 25.0) -> torch.Tensor:
    accel = torch.zeros_like(speed)
    if speed.shape[1] > 1:
        accel[:, 1:] = (speed[:, 1:] - speed[:, :-1]) * frame_rate
        accel[:, 0] = accel[:, 1]
    return accel
