from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


DECISION_S = 1


@dataclass(frozen=True)
class BidirectionalTrustConfig:
    frame_rate: float = 25.0
    future_len: int = 125
    wheelbase_m: float = 2.7
    ego_length_m: float = 4.6
    ego_width_m: float = 1.8
    collision_margin_m: float = 0.5
    safe_clearance_m: float = 4.0
    safe_ttc_s: float = 4.0
    safe_dhw_m: float = 15.0
    decay_eta: float = 0.035
    decision_weight: float = 0.45
    steer_weight: float = 0.25
    speed_weight: float = 0.20
    accel_weight: float = 0.10
    max_abs_steer_rad: float = 0.45
    max_speed_mps: float = 35.0
    max_abs_accel_mps2: float = 4.0
    human_risk_bias: float = 2.4
    human_risk_gain: float = 4.0
    disagreement_bias: float = 2.3
    disagreement_gain: float = 3.2


class BidirectionalTrustEstimator:
    """Compute bidirectional trust sequences from human and machine intents.

    The estimator keeps the interface aligned with Work1/Work2 model outputs.
    Machine-to-human trust is based on traffic risk if the predicted human
    intent is executed by a kinematic model. Human-to-machine trust is based on
    time-weighted disagreement between human and machine decision/control
    intent sequences.
    """

    def __init__(self, config: BidirectionalTrustConfig | None = None) -> None:
        self.config = config or BidirectionalTrustConfig()

    def estimate(
        self,
        inputs: dict[str, torch.Tensor],
        human_outputs: dict[str, torch.Tensor],
        machine_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        human_seq = _future_decision_sequence(human_outputs, self.config.future_len)
        machine_seq = _future_decision_sequence(machine_outputs, self.config.future_len)
        human_accel = _future_acceleration(human_outputs, self.config.frame_rate)
        machine_accel = _future_acceleration(machine_outputs, self.config.frame_rate)

        decision_diff = (human_seq != machine_seq).to(dtype=human_outputs["future_speed"].dtype)
        steer_diff = torch.abs(human_outputs["future_steer"] - machine_outputs["future_steer"]) / self.config.max_abs_steer_rad
        speed_diff = torch.abs(human_outputs["future_speed"] - machine_outputs["future_speed"]) / self.config.max_speed_mps
        accel_diff = torch.abs(human_accel - machine_accel) / self.config.max_abs_accel_mps2
        intent_diff_inst = (
            self.config.decision_weight * decision_diff
            + self.config.steer_weight * torch.clamp(steer_diff, max=1.5)
            + self.config.speed_weight * torch.clamp(speed_diff, max=1.5)
            + self.config.accel_weight * torch.clamp(accel_diff, max=1.5)
        )
        intent_diff = _causal_exponential_average(intent_diff_inst, self.config.decay_eta)
        trust_human_to_machine = torch.sigmoid(self.config.disagreement_bias - self.config.disagreement_gain * intent_diff)

        human_rollout = self.rollout_intent(inputs, human_outputs)
        human_risk = self._traffic_risk(inputs, human_rollout)
        env_context = self.environment_context(inputs)
        trust_machine_to_human = torch.sigmoid(self.config.human_risk_bias - self.config.human_risk_gain * human_risk)

        return {
            "trust_machine_to_human": trust_machine_to_human,
            "trust_human_to_machine": trust_human_to_machine,
            "human_risk": human_risk,
            "environment_urgency": env_context["urgency"],
            "front_gap": env_context["front_gap"],
            "front_relative_speed": env_context["front_relative_speed"],
            "front_ttc": env_context["front_ttc"],
            "intent_disagreement": intent_diff,
            "human_rollout_xy": human_rollout["xy"],
            "human_rollout_yaw": human_rollout["yaw"],
            "human_decision_sequence": human_seq,
            "machine_decision_sequence": machine_seq,
        }

    def environment_context(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cfg = self.config
        risk_last = inputs["risk_history"][:, -1]
        neighbors = inputs["neighbor_history"][:, -1]
        mask = inputs["neighbor_mask"][:, -1] > 0.5
        speed = torch.clamp(inputs["ego_history"][:, -1, 6], min=1.0)
        horizon = int(cfg.future_len)
        frame_t = (torch.arange(horizon, device=speed.device, dtype=speed.dtype) + 1.0) / cfg.frame_rate

        front_valid = mask[:, 0]
        raw_gap = torch.where(front_valid, torch.clamp(neighbors[:, 0, 0], min=0.0), risk_last[:, 0])
        rel_v = torch.where(front_valid, neighbors[:, 0, 2], torch.zeros_like(speed))
        closing_speed = torch.clamp(-rel_v, min=0.0)
        front_gap = torch.clamp(raw_gap.view(-1, 1) - closing_speed.view(-1, 1) * frame_t.view(1, -1), min=0.0)
        front_ttc = torch.where(
            closing_speed.view(-1, 1) > 0.05,
            front_gap / torch.clamp(closing_speed.view(-1, 1), min=0.05),
            torch.full_like(front_gap, 20.0),
        )
        desired_gap = cfg.safe_dhw_m + 0.7 * speed.view(-1, 1)
        gap_urgency = torch.clamp((desired_gap - front_gap) / torch.clamp(desired_gap, min=1.0), min=0.0, max=1.0)
        ttc_urgency = torch.clamp((cfg.safe_ttc_s - front_ttc) / cfg.safe_ttc_s, min=0.0, max=1.0)
        slow_front_urgency = torch.clamp(closing_speed.view(-1, 1) / 8.0, min=0.0, max=1.0)
        front_risk = torch.clamp(risk_last[:, 3].view(-1, 1), min=0.0, max=1.0)
        urgency = torch.clamp(
            0.38 * gap_urgency + 0.24 * ttc_urgency + 0.22 * slow_front_urgency + 0.16 * front_risk,
            min=0.0,
            max=1.0,
        )
        return {
            "urgency": urgency,
            "front_gap": front_gap,
            "front_relative_speed": rel_v.view(-1, 1).expand_as(front_gap),
            "front_ttc": torch.clamp(front_ttc, min=0.0, max=20.0),
        }

    def rollout_intent(self, inputs: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cfg = self.config
        dt = 1.0 / cfg.frame_rate
        speed = outputs["future_speed"]
        steer = outputs["future_steer"]
        batch, horizon = speed.shape
        xy = torch.zeros((batch, horizon, 2), dtype=speed.dtype, device=speed.device)
        yaw = torch.zeros((batch, horizon), dtype=speed.dtype, device=speed.device)
        x = torch.zeros((batch,), dtype=speed.dtype, device=speed.device)
        y = torch.zeros((batch,), dtype=speed.dtype, device=speed.device)
        psi = torch.zeros((batch,), dtype=speed.dtype, device=speed.device)
        for i in range(horizon):
            delta = torch.clamp(steer[:, i], min=-0.6, max=0.6)
            psi = psi + speed[:, i] / max(cfg.wheelbase_m, 1e-6) * torch.tan(delta) * dt
            x = x + speed[:, i] * torch.cos(psi) * dt
            y = y + speed[:, i] * torch.sin(psi) * dt
            xy[:, i, 0] = x
            xy[:, i, 1] = y
            yaw[:, i] = psi
        return {"xy": xy, "yaw": yaw}

    def _traffic_risk(self, inputs: dict[str, torch.Tensor], rollout: dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.config
        neighbors = inputs["neighbor_history"][:, -1]
        neighbor_mask = inputs["neighbor_mask"][:, -1] > 0.5
        ego_last = inputs["ego_history"][:, -1]
        ego_xy = rollout["xy"]
        batch, horizon, _ = ego_xy.shape
        frame_t = (torch.arange(horizon, device=ego_xy.device, dtype=ego_xy.dtype) + 1.0) / cfg.frame_rate
        ego_vx = ego_last[:, 2].view(batch, 1)
        ego_vy = ego_last[:, 3].view(batch, 1)
        min_clearance = torch.full((batch, horizon), 999.0, device=ego_xy.device, dtype=ego_xy.dtype)
        front_dhw = torch.full((batch, horizon), 999.0, device=ego_xy.device, dtype=ego_xy.dtype)
        front_ttc = torch.full((batch, horizon), 20.0, device=ego_xy.device, dtype=ego_xy.dtype)

        for slot in range(neighbors.shape[1]):
            valid = neighbor_mask[:, slot]
            if not bool(valid.any()):
                continue
            nb = neighbors[:, slot]
            nb_x = nb[:, 0:1] + (ego_vx + nb[:, 2:3]) * frame_t.view(1, -1)
            nb_y = nb[:, 1:2] + (ego_vy + nb[:, 3:4]) * frame_t.view(1, -1)
            dx = nb_x - ego_xy[:, :, 0]
            dy = nb_y - ego_xy[:, :, 1]
            nb_length = torch.where(nb[:, 4] > 0, nb[:, 4], torch.full_like(nb[:, 4], cfg.ego_length_m)).view(batch, 1)
            nb_width = torch.where(nb[:, 5] > 0, nb[:, 5], torch.full_like(nb[:, 5], cfg.ego_width_m)).view(batch, 1)
            long_limit = 0.5 * (cfg.ego_length_m + nb_length) + cfg.collision_margin_m
            lat_limit = 0.5 * (cfg.ego_width_m + nb_width) + cfg.collision_margin_m
            signed_clearance = torch.maximum(torch.abs(dx) - long_limit, torch.abs(dy) - lat_limit)
            signed_clearance = torch.where(valid.view(batch, 1), signed_clearance, min_clearance)
            min_clearance = torch.minimum(min_clearance, signed_clearance)

            same_lane_front = valid.view(batch, 1) & (torch.abs(dy) < cfg.ego_width_m) & (dx > 0)
            rel_vx = torch.clamp(-nb[:, 2:3], min=0.0)
            ttc = torch.where(rel_vx > 0.05, dx / torch.clamp(rel_vx, min=0.05), torch.full_like(dx, 20.0))
            front_dhw = torch.where(same_lane_front & (dx < front_dhw), dx, front_dhw)
            front_ttc = torch.where(same_lane_front & (ttc < front_ttc), torch.clamp(ttc, min=0.0, max=20.0), front_ttc)

        collision_risk = torch.clamp((cfg.safe_clearance_m - min_clearance) / cfg.safe_clearance_m, min=0.0, max=2.0)
        dhw_risk = torch.clamp((cfg.safe_dhw_m - front_dhw) / cfg.safe_dhw_m, min=0.0, max=1.0)
        ttc_risk = torch.clamp((cfg.safe_ttc_s - front_ttc) / cfg.safe_ttc_s, min=0.0, max=1.0)
        return torch.clamp(0.62 * collision_risk + 0.22 * ttc_risk + 0.16 * dhw_risk, min=0.0, max=2.0)


def _future_decision_sequence(outputs: dict[str, torch.Tensor], future_len: int) -> torch.Tensor:
    if "future_decision_logits" in outputs:
        return outputs["future_decision_logits"].argmax(dim=-1)
    event = outputs["future_event_logits"].argmax(dim=-1)
    if "future_event_time_by_class" in outputs:
        event_time = outputs["future_event_time_by_class"].gather(1, event.view(-1, 1)).squeeze(1)
    else:
        event_time = outputs.get("future_event_time", torch.ones_like(event, dtype=torch.float32))
    event_idx = torch.clamp(torch.round(event_time * float(max(future_len - 1, 1))).long(), min=0, max=future_len - 1)
    seq = torch.full((event.shape[0], future_len), DECISION_S, dtype=torch.long, device=event.device)
    frame_ids = torch.arange(future_len, device=event.device).view(1, -1)
    active = event != DECISION_S
    seq[(frame_ids >= event_idx.view(-1, 1)) & active.view(-1, 1)] = event.view(-1, 1).expand(-1, future_len)[
        (frame_ids >= event_idx.view(-1, 1)) & active.view(-1, 1)
    ]
    return seq


def _future_acceleration(outputs: dict[str, torch.Tensor], frame_rate: float) -> torch.Tensor:
    if "future_acceleration" in outputs:
        return outputs["future_acceleration"]
    speed = outputs["future_speed"]
    accel = torch.zeros_like(speed)
    accel[:, 1:] = (speed[:, 1:] - speed[:, :-1]) * frame_rate
    accel[:, 0] = accel[:, 1] if speed.shape[1] > 1 else 0.0
    return accel


def _causal_exponential_average(values: torch.Tensor, eta: float) -> torch.Tensor:
    horizon = values.shape[1]
    out = torch.zeros_like(values)
    frame_ids = torch.arange(horizon, device=values.device, dtype=values.dtype)
    for i in range(horizon):
        dist = i - frame_ids[: i + 1]
        weights = torch.exp(-float(eta) * dist)
        weights = weights / torch.clamp(weights.sum(), min=1e-6)
        out[:, i] = (values[:, : i + 1] * weights.view(1, -1)).sum(dim=1)
    return out
