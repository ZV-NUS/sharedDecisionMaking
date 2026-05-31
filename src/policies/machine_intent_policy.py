from __future__ import annotations

from dataclasses import dataclass

import torch


DECISION_L = 0
DECISION_S = 1
DECISION_R = 2


@dataclass(frozen=True)
class MachineIntentPolicyConfig:
    future_len: int = 125
    frame_rate: float = 25.0
    lane_width_m: float = 3.5
    wheelbase_m: float = 2.7
    max_abs_steer_rad: float = 0.45
    max_accel_mps2: float = 1.5
    comfortable_decel_mps2: float = 2.5
    max_decel_mps2: float = 5.0
    close_rear_max_decel_mps2: float = 1.0
    target_speed_mps: float = 31.0
    min_speed_mps: float = 1.0
    desired_time_headway_s: float = 1.4
    min_front_gap_m: float = 18.0
    min_rear_gap_m: float = 15.0
    side_safety_buffer_m: float = 5.0
    critical_ttc_s: float = 5.0
    critical_thw_s: float = 1.2
    front_risk_threshold: float = 0.36
    overtake_front_gap_m: float = 25.0
    min_overtake_closing_speed_mps: float = 1.2
    min_overtake_speed_mps: float = 18.0
    potential_sigma_m: float = 28.0
    lane_change_duration_s: float = 3.2
    min_event_time_s: float = 0.4
    decision_logit_margin: float = 3.0


class MachineIntentPolicy:
    """Rule/APF machine decision and control-intent policy.

    This module is deterministic by design. It uses highD ego, neighbor and
    risk states to produce the same output schema as the learned human intent
    model, so later trust and authority modules can compare human-vs-machine
    decisions, event timing, speed intent and steering intent directly.
    """

    def __init__(self, config: MachineIntentPolicyConfig | None = None) -> None:
        self.config = config or MachineIntentPolicyConfig()

    def predict(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cfg = self.config
        ego_last = inputs["ego_history"][:, -1]
        risk_last = inputs["risk_history"][:, -1]
        neighbors = inputs["neighbor_history"][:, -1]
        mask = inputs["neighbor_mask"][:, -1] > 0.5

        speed = torch.clamp(ego_last[:, 6], min=cfg.min_speed_mps)
        front_gap = risk_last[:, 0]
        thw = risk_last[:, 1]
        ttc = risk_last[:, 2]
        front_risk = risk_last[:, 3]

        front_valid = mask[:, 0]
        front_rel_v = torch.where(front_valid, neighbors[:, 0, 2], torch.zeros_like(speed))
        closing_speed = torch.clamp(-front_rel_v, min=0.0)
        slow_front = closing_speed > cfg.min_overtake_closing_speed_mps
        close_front = front_gap < torch.minimum(
            cfg.overtake_front_gap_m * torch.ones_like(speed),
            cfg.min_front_gap_m + cfg.desired_time_headway_s * speed,
        )
        critical_front = front_valid & close_front & ((ttc < cfg.critical_ttc_s) | (thw < cfg.critical_thw_s))
        overtake_needed = front_valid & slow_front & close_front & (speed > cfg.min_overtake_speed_mps)
        front_urgent = critical_front | overtake_needed | (front_valid & close_front & slow_front & (front_risk > cfg.front_risk_threshold))

        left = self._side_state(neighbors, mask, front_slot=2, rear_slot=3, speed=speed)
        right = self._side_state(neighbors, mask, front_slot=4, rear_slot=5, speed=speed)
        left_safe = self._side_safe(left, speed)
        right_safe = self._side_safe(right, speed)

        left_potential = self._side_potential(left)
        right_potential = self._side_potential(right)
        keep_potential = self._keep_lane_potential(front_gap, ttc, front_risk, speed)

        left_better = left_safe & (left_potential + 0.08 < torch.minimum(keep_potential, right_potential))
        right_better = right_safe & (right_potential + 0.08 < torch.minimum(keep_potential, left_potential))
        choose_left_slots = front_urgent & left_better
        choose_right_slots = front_urgent & (~choose_left_slots) & right_better
        decision = torch.full_like(speed, DECISION_S, dtype=torch.long)
        decision[choose_left_slots] = DECISION_L
        decision[choose_right_slots] = DECISION_R

        event_time = self._event_time(front_risk, ttc, front_gap, speed, decision)
        event_idx = torch.clamp(
            torch.round(event_time * float(max(cfg.future_len - 1, 1))).long(),
            min=0,
            max=cfg.future_len - 1,
        )
        front_rel_v = torch.where(mask[:, 0], neighbors[:, 0, 2], torch.zeros_like(speed))
        rear_gap = torch.where(mask[:, 1], torch.clamp(-neighbors[:, 1, 0], min=0.0), torch.full_like(speed, cfg.potential_sigma_m * 3))
        rear_rel_v = torch.where(mask[:, 1], neighbors[:, 1, 2], torch.zeros_like(speed))
        front_valid = mask[:, 0]
        future_speed, future_accel = self._future_speed(speed, front_gap, ttc, front_rel_v, front_valid, rear_gap, rear_rel_v, decision)
        future_steer = self._future_steer(decision, event_idx, speed)
        future_decision_logits = self._future_decision_logits(decision, event_idx, inputs["ego_history"].device)

        event_logits = torch.full((speed.shape[0], 3), -cfg.decision_logit_margin, device=speed.device)
        event_logits.scatter_(1, decision.view(-1, 1), cfg.decision_logit_margin)
        time_by_class = torch.ones((speed.shape[0], 3), device=speed.device)
        time_by_class[:, DECISION_L] = event_time
        time_by_class[:, DECISION_R] = event_time
        time_bins = self._time_bin_logits(event_time, decision, speed.device)

        return {
            "decision_logits": event_logits,
            "future_event_logits": event_logits,
            "future_decision_logits": future_decision_logits,
            "future_event_time": event_time,
            "future_event_time_logits": time_bins.gather(
                1,
                decision.view(-1, 1, 1).expand(-1, 1, cfg.future_len),
            ).squeeze(1),
            "future_event_time_by_class": time_by_class,
            "future_event_time_bin_by_class": time_bins,
            "future_speed": future_speed,
            "future_acceleration": future_accel,
            "future_steer": future_steer,
        }

    def _side_state(
        self,
        neighbors: torch.Tensor,
        mask: torch.Tensor,
        front_slot: int,
        rear_slot: int,
        speed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cfg = self.config
        front = neighbors[:, front_slot]
        rear = neighbors[:, rear_slot]
        front_valid = mask[:, front_slot]
        rear_valid = mask[:, rear_slot]
        front_gap = torch.where(front_valid, torch.clamp(front[:, 0], min=0.0), torch.full_like(speed, cfg.potential_sigma_m * 3))
        rear_gap = torch.where(rear_valid, torch.clamp(-rear[:, 0], min=0.0), torch.full_like(speed, cfg.potential_sigma_m * 3))
        front_rel_v = torch.where(front_valid, front[:, 2], torch.zeros_like(speed))
        rear_rel_v = torch.where(rear_valid, rear[:, 2], torch.zeros_like(speed))
        return {
            "front_gap": front_gap,
            "rear_gap": rear_gap,
            "front_rel_v": front_rel_v,
            "rear_rel_v": rear_rel_v,
            "front_valid": front_valid,
            "rear_valid": rear_valid,
        }

    def _side_safe(self, state: dict[str, torch.Tensor], speed: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        front_required = cfg.min_front_gap_m + 0.35 * speed
        rear_required = cfg.min_rear_gap_m + 0.25 * torch.clamp(state["rear_rel_v"], min=0.0)
        horizon = cfg.lane_change_duration_s + cfg.min_event_time_s
        predicted_front_gap = state["front_gap"] + torch.clamp(state["front_rel_v"], max=0.0) * horizon
        predicted_rear_gap = state["rear_gap"] - torch.clamp(state["rear_rel_v"], min=0.0) * horizon
        buffered_rear_required = rear_required + cfg.side_safety_buffer_m
        return (
            (state["front_gap"] > front_required)
            & (predicted_front_gap > front_required)
            & (state["rear_gap"] > buffered_rear_required)
            & (predicted_rear_gap > buffered_rear_required)
        )

    def _front_following_accel(
        self,
        speed: torch.Tensor,
        front_gap: torch.Tensor,
        ttc: torch.Tensor,
        front_rel_v: torch.Tensor,
        rear_gap: torch.Tensor,
        rear_rel_v: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        desired_gap = cfg.min_front_gap_m + cfg.desired_time_headway_s * speed
        gap_error = torch.clamp(desired_gap - front_gap, min=0.0)
        closing_speed = torch.clamp(-front_rel_v, min=0.0)
        available_gap = torch.clamp(front_gap - cfg.min_front_gap_m, min=1.0)
        kinematic_decel = closing_speed * closing_speed / (2.0 * available_gap)
        feedback_decel = 0.05 * gap_error + 0.60 * closing_speed
        urgent_ttc_decel = torch.where(ttc < cfg.critical_ttc_s, cfg.max_decel_mps2 * torch.ones_like(speed), torch.zeros_like(speed))
        requested_decel = torch.maximum(torch.maximum(feedback_decel, kinematic_decel), urgent_ttc_decel)
        requested_decel = torch.where(
            ttc < cfg.critical_ttc_s,
            requested_decel,
            torch.minimum(requested_decel, cfg.comfortable_decel_mps2 * torch.ones_like(speed)),
        )
        rear_closing = torch.clamp(rear_rel_v, min=0.0)
        close_rear = rear_gap < cfg.min_rear_gap_m + 0.35 * speed + 1.2 * rear_closing
        allowed_decel = torch.where(
            close_rear,
            cfg.close_rear_max_decel_mps2 * torch.ones_like(speed),
            cfg.max_decel_mps2 * torch.ones_like(speed),
        )
        return -torch.minimum(torch.clamp(requested_decel, min=0.0), allowed_decel)

    def _side_potential(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.config
        front_cost = torch.exp(-state["front_gap"] / cfg.potential_sigma_m)
        rear_cost = 0.85 * torch.exp(-state["rear_gap"] / cfg.potential_sigma_m)
        closing_front = torch.clamp(-state["front_rel_v"], min=0.0) / 12.0
        closing_rear = torch.clamp(state["rear_rel_v"], min=0.0) / 12.0
        return front_cost * (1.0 + closing_front) + rear_cost * (1.0 + closing_rear)

    def _keep_lane_potential(
        self,
        front_gap: torch.Tensor,
        ttc: torch.Tensor,
        front_risk: torch.Tensor,
        speed: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        distance_cost = torch.exp(-front_gap / cfg.potential_sigma_m)
        ttc_cost = torch.clamp((cfg.critical_ttc_s - ttc) / cfg.critical_ttc_s, min=0.0)
        speed_cost = torch.clamp((speed - cfg.target_speed_mps) / cfg.target_speed_mps, min=0.0)
        return distance_cost + 0.8 * ttc_cost + 0.8 * front_risk + 0.3 * speed_cost

    def _event_time(
        self,
        front_risk: torch.Tensor,
        ttc: torch.Tensor,
        front_gap: torch.Tensor,
        speed: torch.Tensor,
        decision: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        urgency = (
            0.45 * torch.clamp(front_risk / max(cfg.front_risk_threshold, 1e-6), 0.0, 2.0)
            + 0.35 * torch.clamp((cfg.critical_ttc_s - ttc) / cfg.critical_ttc_s, min=0.0, max=1.0)
            + 0.20 * torch.clamp((cfg.min_front_gap_m + speed - front_gap) / (cfg.min_front_gap_m + speed), min=0.0, max=1.0)
        )
        event_s = cfg.min_event_time_s + (1.4 - torch.clamp(urgency, 0.0, 1.0)) * 0.9
        event_s = torch.clamp(event_s, min=cfg.min_event_time_s, max=2.0)
        event_time = event_s / max((cfg.future_len - 1) / cfg.frame_rate, 1e-6)
        return torch.where(decision == DECISION_S, torch.ones_like(event_time), event_time)

    def _future_speed(
        self,
        speed: torch.Tensor,
        front_gap: torch.Tensor,
        ttc: torch.Tensor,
        front_rel_v: torch.Tensor,
        front_valid: torch.Tensor,
        rear_gap: torch.Tensor,
        rear_rel_v: torch.Tensor,
        decision: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        dt = 1.0 / cfg.frame_rate
        front_need_decel = front_valid & (
            (front_gap < cfg.min_front_gap_m + cfg.desired_time_headway_s * speed) | (ttc < cfg.critical_ttc_s)
        )
        rear_pressure = rear_gap < cfg.min_rear_gap_m + 1.25 * speed + torch.clamp(rear_rel_v, min=0.0)
        accel = torch.full_like(speed, cfg.max_accel_mps2 * 0.35)
        accel = torch.where((~front_need_decel) & rear_pressure, cfg.max_accel_mps2 * torch.ones_like(accel), accel)
        accel = torch.where(speed > cfg.target_speed_mps, -0.6 * torch.ones_like(accel), accel)
        following_accel = self._front_following_accel(speed, front_gap, ttc, front_rel_v, rear_gap, rear_rel_v)
        accel = torch.where(front_need_decel, following_accel, accel)
        overtake_target_speed = cfg.target_speed_mps + 2.0
        overtake_accel = torch.where(
            speed < overtake_target_speed,
            cfg.max_accel_mps2 * torch.ones_like(accel),
            -0.2 * torch.ones_like(accel),
        )
        accel = torch.where(decision != DECISION_S, overtake_accel, accel)

        speeds = []
        accels = []
        current = speed
        for _ in range(cfg.future_len):
            current = torch.clamp(current + accel * dt, min=0.0, max=cfg.target_speed_mps + 2.0)
            speeds.append(current)
            accels.append(accel)
        return torch.stack(speeds, dim=1), torch.stack(accels, dim=1)

    def _future_steer(self, decision: torch.Tensor, event_idx: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        batch = decision.shape[0]
        frame_ids = torch.arange(cfg.future_len, device=decision.device, dtype=torch.float32).view(1, -1)
        start = event_idx.to(dtype=torch.float32).view(-1, 1)
        duration = max(int(round(cfg.lane_change_duration_s * cfg.frame_rate)), 1)
        tau = torch.clamp((frame_ids - start) / float(duration), min=0.0, max=1.0)
        lateral_progress = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        lateral_direction = torch.zeros(batch, device=decision.device)
        lateral_direction[decision == DECISION_L] = -1.0
        lateral_direction[decision == DECISION_R] = 1.0
        lateral = lateral_direction.view(-1, 1) * cfg.lane_width_m * lateral_progress
        dt = 1.0 / cfg.frame_rate
        dy = torch.zeros_like(lateral)
        dy[:, 1:] = (lateral[:, 1:] - lateral[:, :-1]) / dt
        dy[:, 0] = dy[:, 1]
        ddy = torch.zeros_like(dy)
        ddy[:, 1:] = (dy[:, 1:] - dy[:, :-1]) / dt
        ddy[:, 0] = ddy[:, 1]
        v = torch.clamp(speed.view(-1, 1), min=cfg.min_speed_mps)
        curvature = ddy / torch.clamp(v * v, min=1.0)
        steer = torch.atan(cfg.wheelbase_m * curvature)
        return torch.clamp(steer, min=-cfg.max_abs_steer_rad, max=cfg.max_abs_steer_rad)

    def _future_decision_logits(self, decision: torch.Tensor, event_idx: torch.Tensor, device: torch.device) -> torch.Tensor:
        cfg = self.config
        logits = torch.full((decision.shape[0], cfg.future_len, 3), -cfg.decision_logit_margin, device=device)
        logits[:, :, DECISION_S] = cfg.decision_logit_margin
        frame_ids = torch.arange(cfg.future_len, device=device).view(1, -1)
        active = (decision != DECISION_S).view(-1, 1) & (frame_ids >= event_idx.view(-1, 1))
        for label in (DECISION_L, DECISION_R):
            mask = active & (decision.view(-1, 1) == label)
            logits[:, :, label] = torch.where(mask, torch.full_like(logits[:, :, label], cfg.decision_logit_margin), logits[:, :, label])
            logits[:, :, DECISION_S] = torch.where(mask, torch.full_like(logits[:, :, DECISION_S], -cfg.decision_logit_margin), logits[:, :, DECISION_S])
        return logits

    def _time_bin_logits(self, event_time: torch.Tensor, decision: torch.Tensor, device: torch.device) -> torch.Tensor:
        cfg = self.config
        frame_ids = torch.arange(cfg.future_len, device=device, dtype=torch.float32).view(1, 1, -1)
        centers = event_time.view(-1, 1, 1) * float(max(cfg.future_len - 1, 1))
        logits = -0.5 * ((frame_ids - centers) / 5.0) ** 2
        logits = logits.expand(-1, 3, -1).clone()
        straight_mask = decision == DECISION_S
        if bool(straight_mask.any()):
            straight_logits = -0.5 * ((frame_ids - float(cfg.future_len - 1)) / 8.0) ** 2
            logits[straight_mask, :, :] = straight_logits.expand(int(straight_mask.sum().item()), 3, -1)
        return logits
