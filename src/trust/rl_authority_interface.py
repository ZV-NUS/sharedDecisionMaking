from __future__ import annotations

import torch


def build_authority_rl_observation(
    trust_outputs: dict[str, torch.Tensor],
    authority_outputs: dict[str, torch.Tensor],
    human_outputs: dict[str, torch.Tensor],
    machine_outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Package Work3 outputs as Work4 RL authority-optimization inputs.

    Work3 provides reference/prior authority only. The RL module should use
    these signals with traffic context and intent disagreement to produce the
    final executable authority sequence.
    """

    human_authority_ref = authority_outputs["authority_ref"]
    return {
        "human_authority_ref": human_authority_ref,
        "machine_authority_ref": 1.0 - human_authority_ref,
        "trust_machine_to_human": trust_outputs["trust_machine_to_human"],
        "trust_human_to_machine": trust_outputs["trust_human_to_machine"],
        "human_risk": trust_outputs["human_risk"],
        "environment_urgency": trust_outputs["environment_urgency"],
        "front_gap": trust_outputs["front_gap"],
        "front_relative_speed": trust_outputs["front_relative_speed"],
        "front_ttc": trust_outputs["front_ttc"],
        "intent_disagreement": trust_outputs["intent_disagreement"],
        "human_speed_intent": human_outputs["future_speed"],
        "human_steer_intent": human_outputs["future_steer"],
        "machine_speed_intent": machine_outputs["future_speed"],
        "machine_steer_intent": machine_outputs["future_steer"],
    }


def blend_with_rl_authority(
    human_outputs: dict[str, torch.Tensor],
    machine_outputs: dict[str, torch.Tensor],
    human_authority_rl: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Blend human/machine intents using final RL-optimized authority."""

    lam = torch.clamp(human_authority_rl, 0.0, 1.0)
    speed = lam * human_outputs["future_speed"] + (1.0 - lam) * machine_outputs["future_speed"]
    steer = lam * human_outputs["future_steer"] + (1.0 - lam) * machine_outputs["future_steer"]
    return {
        "future_speed": speed,
        "future_steer": steer,
        "human_authority_rl": lam,
        "machine_authority_rl": 1.0 - lam,
    }
