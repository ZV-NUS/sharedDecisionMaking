from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveRobustMPCLiteConfig:
    frame_rate: float = 25.0
    future_len: int = 125
    horizon: int = 12
    wheelbase_m: float = 2.7
    lane_width_m: float = 3.5
    lane_margin_m: float = 0.5
    ego_length_m: float = 4.6
    ego_width_m: float = 1.8
    mass_kg: float = 1500.0
    yaw_inertia_kgm2: float = 2800.0
    lf_m: float = 1.2
    lr_m: float = 1.5
    cornering_stiffness_front: float = 55000.0
    cornering_stiffness_rear: float = 60000.0
    tire_mu: float = 0.85
    max_abs_steer_rad: float = 0.45
    max_abs_accel_mps2: float = 4.0
    max_steer_rate_radps: float = 0.55
    max_jerk_mps3: float = 10.0
    beta_limit_rad: float = 0.12
    yaw_rate_limit_rps: float = 0.65
    target_speed_mps: float = 31.0
    safe_distance_m: float = 8.0
    robust_distance_gain_m: float = 5.0
    robust_stability_gain: float = 0.25
    lane_center_weight: float = 3.2
    lane_boundary_weight: float = 5000.0
    lane_change_threshold_m: float = 1.4
    lane_change_duration_s: float = 3.2
    lane_change_start_deadband_m: float = 0.35
    collision_longitudinal_margin_m: float = 1.0
    collision_lateral_margin_m: float = 0.4


class AdaptiveRobustMPCLite:
    """Sampling-based adaptive robust MPC for the shared-control layer.

    It is intentionally lightweight: at every frame the controller evaluates a
    small grid of steering/acceleration corrections around the RL-fused
    reference control, rolls out a dynamic bicycle model for a short horizon,
    and executes only the first control.
    """

    def __init__(self, config: AdaptiveRobustMPCLiteConfig | None = None) -> None:
        self.config = config or AdaptiveRobustMPCLiteConfig()

    def rollout(
        self,
        reference: dict[str, np.ndarray],
        neighbor_xy: np.ndarray,
        neighbor_mask: np.ndarray,
        neighbor_state: np.ndarray,
        trust_machine_to_human: np.ndarray,
        trust_human_to_machine: np.ndarray,
        environment_urgency: np.ndarray,
        authority_rl: np.ndarray,
    ) -> dict[str, np.ndarray | float | int | bool]:
        cfg = self.config
        n = min(cfg.future_len, len(reference["speed"]))
        dt = 1.0 / cfg.frame_rate
        ref_xy = np.asarray(reference["xy"], dtype=np.float32)[:n]
        ref_yaw = np.asarray(reference["yaw"], dtype=np.float32)[:n]
        ref_speed = np.asarray(reference["speed"], dtype=np.float32)[:n]
        ref_steer = np.asarray(reference["steer"], dtype=np.float32)[:n]
        ref_accel = np.asarray(reference["acceleration"], dtype=np.float32)[:n]
        lane_plan = self._lane_plan(ref_xy)
        ref_xy = ref_xy.copy()
        ref_xy[:, 1] = self._lane_reference_y(n, lane_plan)
        ref_yaw = self._reference_yaw(ref_xy).astype(np.float32)
        ref_steer = self._reference_steer(ref_xy).astype(np.float32)
        if abs(lane_plan["target_center"]) < 1e-6:
            ref_yaw[:] = 0.0
            ref_steer[:] = 0.0

        xy = np.zeros((n, 2), dtype=np.float32)
        yaw = np.zeros((n,), dtype=np.float32)
        speed = np.zeros((n,), dtype=np.float32)
        accel = np.zeros((n,), dtype=np.float32)
        steer = np.zeros((n,), dtype=np.float32)
        beta = np.zeros((n,), dtype=np.float32)
        yaw_rate = np.zeros((n,), dtype=np.float32)
        stability_margin = np.zeros((n,), dtype=np.float32)
        safety_margin = np.zeros((n,), dtype=np.float32)
        weights = np.zeros((n, 5), dtype=np.float32)

        state = np.array([0.0, 0.0, 0.0, max(1.0, float(ref_speed[0])), 0.0, 0.0], dtype=np.float64)
        prev_u = np.array([float(ref_steer[0]), float(ref_accel[0])], dtype=np.float64)
        for k in range(n):
            w = self._adaptive_weights(
                state,
                float(environment_urgency[min(k, len(environment_urgency) - 1)]),
                float(trust_machine_to_human[min(k, len(trust_machine_to_human) - 1)]),
                float(trust_human_to_machine[min(k, len(trust_human_to_machine) - 1)]),
                float(authority_rl[min(k, len(authority_rl) - 1)]),
            )
            weights[k] = np.array([w["track"], w["stable"], w["safe"], w["comfort"], w["efficiency"]], dtype=np.float32)
            best_u = self._select_control(
                k,
                state,
                prev_u,
                ref_xy,
                ref_yaw,
                ref_speed,
                ref_steer,
                ref_accel,
                neighbor_xy,
                neighbor_mask,
                neighbor_state,
                environment_urgency,
                trust_machine_to_human,
                lane_plan,
                authority_rl,
                w,
            )
            state = self._step_dynamics(state, best_u[0], best_u[1], dt)
            prev_u = best_u
            xy[k] = state[:2]
            yaw[k] = state[2]
            speed[k] = max(0.0, state[3])
            steer[k] = best_u[0]
            accel[k] = best_u[1]
            beta[k] = np.arctan2(state[4], max(state[3], 1e-3))
            yaw_rate[k] = state[5]
            stability_margin[k] = self._stability_margin(beta[k], yaw_rate[k], float(environment_urgency[min(k, len(environment_urgency) - 1)]))
            safety_margin[k] = self._min_clearance_at(k, xy[k], neighbor_xy, neighbor_mask, neighbor_state)

        collision, min_clearance, min_slot, min_frame = self._check_collision(xy, neighbor_xy, neighbor_mask, neighbor_state)
        return {
            "xy": xy,
            "yaw": yaw,
            "speed": speed,
            "acceleration": accel,
            "steer": steer,
            "beta": beta,
            "yaw_rate": yaw_rate,
            "stability_margin": stability_margin,
            "safety_margin": safety_margin,
            "weights": weights,
            "collision": bool(collision),
            "min_clearance_m": float(min_clearance),
            "min_clearance_slot": int(min_slot),
            "min_clearance_frame": int(min_frame),
            "mean_abs_steer_rad": float(np.mean(np.abs(steer))),
            "mean_abs_accel_mps2": float(np.mean(np.abs(accel))),
            "max_abs_beta_rad": float(np.max(np.abs(beta))),
            "max_abs_yaw_rate_rps": float(np.max(np.abs(yaw_rate))),
        }

    def _select_control(
        self,
        start: int,
        state: np.ndarray,
        prev_u: np.ndarray,
        ref_xy: np.ndarray,
        ref_yaw: np.ndarray,
        ref_speed: np.ndarray,
        ref_steer: np.ndarray,
        ref_accel: np.ndarray,
        neighbor_xy: np.ndarray,
        neighbor_mask: np.ndarray,
        neighbor_state: np.ndarray,
        environment_urgency: np.ndarray,
        trust_machine_to_human: np.ndarray,
        lane_plan: dict[str, float],
        authority_rl: np.ndarray,
        weights: dict[str, float],
    ) -> np.ndarray:
        cfg = self.config
        dt = 1.0 / cfg.frame_rate
        k = min(start, len(ref_speed) - 1)
        e_y = float(ref_xy[k, 1] - state[1])
        e_psi = _wrap_angle(float(ref_yaw[k] - state[2]))
        e_v = float(ref_speed[k] - state[3])
        lane_change = abs(lane_plan["target_center"]) > 1e-6
        if lane_change:
            feedback_steer = 0.032 * e_y + 0.22 * e_psi
            steer_offsets = np.array([-0.035, -0.017, 0.0, 0.017, 0.035], dtype=np.float64)
        else:
            feedback_steer = 0.0
            steer_offsets = np.array([0.0], dtype=np.float64)
        feedback_accel = 0.35 * e_v
        base_steer = float(ref_steer[k]) + feedback_steer
        base_accel = float(ref_accel[k]) + feedback_accel
        accel_offsets = np.array([-3.0, -2.0, -1.0, 0.0, 0.6], dtype=np.float64)
        best_cost = float("inf")
        best = np.array([float(ref_steer[k]), float(ref_accel[k])], dtype=np.float64)
        for ds in steer_offsets:
            for da in accel_offsets:
                u0 = self._clip_control(
                    np.array([base_steer + ds, base_accel + da], dtype=np.float64),
                    prev_u,
                    dt,
                )
                cost = self._candidate_cost(
                    start,
                    state,
                    u0,
                    prev_u,
                    ref_xy,
                    ref_yaw,
                    ref_speed,
                    ref_steer,
                    ref_accel,
                    neighbor_xy,
                    neighbor_mask,
                    neighbor_state,
                    environment_urgency,
                    trust_machine_to_human,
                    authority_rl,
                    lane_plan,
                    weights,
                )
                if cost < best_cost:
                    best_cost = cost
                    best = u0
        return best

    def _candidate_cost(
        self,
        start: int,
        state: np.ndarray,
        u0: np.ndarray,
        prev_u: np.ndarray,
        ref_xy: np.ndarray,
        ref_yaw: np.ndarray,
        ref_speed: np.ndarray,
        ref_steer: np.ndarray,
        ref_accel: np.ndarray,
        neighbor_xy: np.ndarray,
        neighbor_mask: np.ndarray,
        neighbor_state: np.ndarray,
        environment_urgency: np.ndarray,
        trust_machine_to_human: np.ndarray,
        authority_rl: np.ndarray,
        lane_plan: dict[str, float],
        weights: dict[str, float],
    ) -> float:
        cfg = self.config
        dt = 1.0 / cfg.frame_rate
        sim = state.copy()
        prev = prev_u.copy()
        total = 0.0
        horizon = min(cfg.horizon, len(ref_speed) - start)
        for j in range(horizon):
            idx = start + j
            blend = j / max(horizon - 1, 1)
            ref_u = np.array([float(ref_steer[idx]), float(ref_accel[idx])], dtype=np.float64)
            u = self._clip_control((1.0 - blend) * u0 + blend * ref_u, prev, dt)
            sim = self._step_dynamics(sim, u[0], u[1], dt)
            beta = np.arctan2(sim[4], max(sim[3], 1e-3))
            margin = self._min_clearance_at(idx, sim[:2], neighbor_xy, neighbor_mask, neighbor_state)
            robust_risk = float(environment_urgency[min(idx, len(environment_urgency) - 1)])
            human_uncertainty = 1.0 - float(trust_machine_to_human[min(idx, len(trust_machine_to_human) - 1)])
            safe_distance = cfg.safe_distance_m + cfg.robust_distance_gain_m * (0.6 * robust_risk + 0.4 * human_uncertainty)
            track = (sim[1] - ref_xy[idx, 1]) ** 2 + 0.25 * _wrap_angle(sim[2] - ref_yaw[idx]) ** 2 + 0.08 * (sim[3] - ref_speed[idx]) ** 2
            beta_ratio = abs(beta) / max(cfg.beta_limit_rad, 1e-6)
            yaw_ratio = abs(sim[5]) / max(cfg.yaw_rate_limit_rps, 1e-6)
            stable = beta_ratio**2 + 0.35 * yaw_ratio**2 + 30.0 * max(0.0, beta_ratio - 1.0) ** 2 + 12.0 * max(0.0, yaw_ratio - 1.0) ** 2
            safe = max(0.0, safe_distance - margin) ** 2
            lane = self._lane_cost(sim[1], ref_xy[idx, 1], lane_plan, robust_risk)
            comfort = 0.15 * u[1] ** 2 + 4.0 * ((u[0] - prev[0]) / dt) ** 2 + 0.05 * ((u[1] - prev[1]) / dt) ** 2
            efficiency = ((cfg.target_speed_mps - sim[3]) / max(cfg.target_speed_mps, 1.0)) ** 2
            ref_dev = 0.2 * (u[0] - ref_steer[idx]) ** 2 + 0.03 * (u[1] - ref_accel[idx]) ** 2
            # High human authority makes the controller track the fused intent more,
            # while low trust still gives the robust safety term room to dominate.
            authority_track_gain = 0.5 + 0.5 * float(authority_rl[min(idx, len(authority_rl) - 1)])
            total += (
                weights["track"] * authority_track_gain * track
                + weights["stable"] * stable
                + weights["safe"] * safe
                + weights["lane"] * lane
                + weights["comfort"] * comfort
                + weights["efficiency"] * efficiency
                + weights["robust"] * ref_dev
            )
            prev = u
        return float(total / max(horizon, 1))

    def _adaptive_weights(
        self,
        state: np.ndarray,
        environment_urgency: float,
        trust_machine_to_human: float,
        trust_human_to_machine: float,
        authority_rl: float,
    ) -> dict[str, float]:
        beta = np.arctan2(state[4], max(state[3], 1e-3))
        stability_margin = self._stability_margin(beta, state[5], environment_urgency)
        risk = float(np.clip(environment_urgency, 0.0, 1.0))
        human_uncertainty = float(np.clip(1.0 - trust_machine_to_human, 0.0, 1.0))
        trust_balance = float(np.clip(0.5 * (trust_machine_to_human + trust_human_to_machine), 0.0, 1.0))
        stable_pressure = 1.0 - stability_margin
        return {
            "track": 1.3 * (1.0 + 0.5 * authority_rl * trust_balance),
            "stable": 1.4 * (1.0 + 6.0 * stable_pressure + 1.5 * risk),
            "safe": 0.8 * (1.0 + 5.0 * risk + 2.0 * human_uncertainty),
            "comfort": 0.20 * max(0.35, 1.0 - 0.55 * risk),
            "efficiency": 0.18 * max(0.2, 1.0 - 0.7 * risk),
            "robust": 0.25 * (1.0 + 2.5 * human_uncertainty + 1.5 * risk),
            "lane": self.config.lane_center_weight * (1.0 + 1.2 * risk),
        }

    def _lane_plan(self, ref_xy: np.ndarray) -> dict[str, float]:
        cfg = self.config
        current_center = 0.0
        tail = ref_xy[-max(5, min(20, len(ref_xy))) :, 1]
        ref_final_y = float(np.median(tail))
        if abs(ref_final_y - current_center) < cfg.lane_change_threshold_m:
            target_center = current_center
        else:
            target_lane = int(np.clip(np.round(ref_final_y / cfg.lane_width_m), -1, 1))
            target_center = float(target_lane * cfg.lane_width_m)
        if abs(target_center) < 1e-6:
            event_start = 0
        else:
            sign = np.sign(target_center)
            crossed = np.flatnonzero(sign * ref_xy[:, 1] > cfg.lane_change_start_deadband_m)
            event_start = int(crossed[0]) if len(crossed) else 0
        lower_center = min(current_center, target_center)
        upper_center = max(current_center, target_center)
        lower = lower_center - 0.5 * cfg.lane_width_m + cfg.lane_margin_m
        upper = upper_center + 0.5 * cfg.lane_width_m - cfg.lane_margin_m
        return {
            "current_center": current_center,
            "target_center": target_center,
            "event_start": float(event_start),
            "lower": float(lower),
            "upper": float(upper),
        }

    def _lane_reference_y(self, n: int, lane_plan: dict[str, float]) -> np.ndarray:
        target_center = float(lane_plan["target_center"])
        if abs(target_center) < 1e-6:
            return np.zeros((n,), dtype=np.float32)
        cfg = self.config
        start = int(max(0, min(n - 1, round(float(lane_plan["event_start"])))))
        duration = max(1, int(round(cfg.lane_change_duration_s * cfg.frame_rate)))
        t = np.arange(n, dtype=np.float32)
        tau = np.clip((t - start) / float(duration), 0.0, 1.0)
        smooth = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        return (target_center * smooth).astype(np.float32)

    def _reference_yaw(self, ref_xy: np.ndarray) -> np.ndarray:
        dx = np.gradient(ref_xy[:, 0]).astype(np.float64)
        dy = np.gradient(ref_xy[:, 1]).astype(np.float64)
        dx = np.where(np.abs(dx) < 1e-3, np.sign(dx + 1e-9) * 1e-3, dx)
        return np.arctan2(dy, dx)

    def _reference_steer(self, ref_xy: np.ndarray) -> np.ndarray:
        cfg = self.config
        x = ref_xy[:, 0].astype(np.float64)
        y = ref_xy[:, 1].astype(np.float64)
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = np.maximum((dx * dx + dy * dy) ** 1.5, 1e-3)
        curvature = (dx * ddy - dy * ddx) / denom
        steer = np.arctan(cfg.wheelbase_m * curvature)
        return np.clip(steer, -0.16, 0.16)

    def _lane_cost(self, y: float, ref_y: float, lane_plan: dict[str, float], risk: float) -> float:
        cfg = self.config
        target_center = lane_plan["target_center"]
        lower = lane_plan["lower"]
        upper = lane_plan["upper"]
        boundary_violation = max(0.0, lower - y) ** 2 + max(0.0, y - upper) ** 2
        if abs(target_center) < 1e-6:
            center_target = target_center
            center_gain = 12.0
        else:
            center_target = ref_y
            center_gain = 6.0
        center_error = (y - center_target) ** 2
        corridor_width = max(upper - lower, 1e-6)
        soft_boundary = cfg.lane_boundary_weight * boundary_violation / corridor_width
        return soft_boundary + center_gain * (0.45 + 0.85 * (1.0 - risk)) * center_error

    def _step_dynamics(self, state: np.ndarray, steer: float, accel: float, dt: float) -> np.ndarray:
        cfg = self.config
        x, y, psi, vx, vy, r = state
        vx_safe = max(float(vx), 1.0)
        steer = float(np.clip(steer, -cfg.max_abs_steer_rad, cfg.max_abs_steer_rad))
        accel = float(np.clip(accel, -cfg.max_abs_accel_mps2, cfg.max_abs_accel_mps2))
        alpha_f = steer - np.arctan2(vy + cfg.lf_m * r, vx_safe)
        alpha_r = -np.arctan2(vy - cfg.lr_m * r, vx_safe)
        normal_f = cfg.mass_kg * 9.81 * cfg.lr_m / (cfg.lf_m + cfg.lr_m)
        normal_r = cfg.mass_kg * 9.81 * cfg.lf_m / (cfg.lf_m + cfg.lr_m)
        fyf = float(np.clip(cfg.cornering_stiffness_front * alpha_f, -cfg.tire_mu * normal_f, cfg.tire_mu * normal_f))
        fyr = float(np.clip(cfg.cornering_stiffness_rear * alpha_r, -cfg.tire_mu * normal_r, cfg.tire_mu * normal_r))
        x_dot = vx * np.cos(psi) - vy * np.sin(psi)
        y_dot = vx * np.sin(psi) + vy * np.cos(psi)
        psi_dot = r
        vx_dot = accel + vy * r
        vy_dot = (fyf + fyr) / cfg.mass_kg - vx * r - 0.35 * vy
        r_dot = (cfg.lf_m * fyf - cfg.lr_m * fyr) / cfg.yaw_inertia_kgm2 - 0.22 * r
        next_vx = float(np.clip(vx + vx_dot * dt, 0.0, 45.0))
        next_vy = float(np.clip(vy + vy_dot * dt, -4.0, 4.0))
        next_r = float(np.clip(r + r_dot * dt, -cfg.yaw_rate_limit_rps, cfg.yaw_rate_limit_rps))
        beta_bound = np.tan(cfg.beta_limit_rad) * max(next_vx, 1.0)
        next_vy = float(np.clip(next_vy, -beta_bound, beta_bound))
        next_state = np.array(
            [
                x + x_dot * dt,
                y + y_dot * dt,
                _wrap_angle(psi + psi_dot * dt),
                next_vx,
                next_vy,
                next_r,
            ],
            dtype=np.float64,
        )
        return next_state

    def _clip_control(self, u: np.ndarray, prev_u: np.ndarray, dt: float) -> np.ndarray:
        cfg = self.config
        steer = float(np.clip(u[0], -cfg.max_abs_steer_rad, cfg.max_abs_steer_rad))
        accel = float(np.clip(u[1], -cfg.max_abs_accel_mps2, cfg.max_abs_accel_mps2))
        steer = float(np.clip(steer, prev_u[0] - cfg.max_steer_rate_radps * dt, prev_u[0] + cfg.max_steer_rate_radps * dt))
        accel = float(np.clip(accel, prev_u[1] - cfg.max_jerk_mps3 * dt, prev_u[1] + cfg.max_jerk_mps3 * dt))
        return np.array([steer, accel], dtype=np.float64)

    def _stability_margin(self, beta: float, yaw_rate: float, risk: float) -> float:
        cfg = self.config
        beta_lim = cfg.beta_limit_rad * (1.0 - cfg.robust_stability_gain * float(np.clip(risk, 0.0, 1.0)))
        score = max(abs(beta) / max(beta_lim, 1e-6), abs(yaw_rate) / max(cfg.yaw_rate_limit_rps, 1e-6))
        return float(np.clip(1.0 - score, 0.0, 1.0))

    def _min_clearance_at(
        self,
        frame: int,
        ego_xy: np.ndarray,
        neighbor_xy: np.ndarray,
        neighbor_mask: np.ndarray,
        neighbor_state: np.ndarray,
    ) -> float:
        cfg = self.config
        min_clearance = 999.0
        i = min(max(frame, 0), neighbor_xy.shape[1] - 1)
        for slot in range(neighbor_state.shape[0]):
            if not bool(neighbor_mask[slot]):
                continue
            nb_length = float(neighbor_state[slot, 4]) if float(neighbor_state[slot, 4]) > 0 else cfg.ego_length_m
            nb_width = float(neighbor_state[slot, 5]) if float(neighbor_state[slot, 5]) > 0 else cfg.ego_width_m
            dx = abs(float(neighbor_xy[slot, i, 0]) - float(ego_xy[0]))
            dy = abs(float(neighbor_xy[slot, i, 1]) - float(ego_xy[1]))
            long_limit = 0.5 * (cfg.ego_length_m + nb_length) + cfg.collision_longitudinal_margin_m
            lat_limit = 0.5 * (cfg.ego_width_m + nb_width) + cfg.collision_lateral_margin_m
            min_clearance = min(min_clearance, max(dx - long_limit, dy - lat_limit))
        return float(min_clearance)

    def _check_collision(
        self,
        ego_xy: np.ndarray,
        neighbor_xy: np.ndarray,
        neighbor_mask: np.ndarray,
        neighbor_state: np.ndarray,
    ) -> tuple[bool, float, int, int]:
        cfg = self.config
        collision = False
        min_clearance = float("inf")
        min_slot = -1
        min_frame = -1
        for slot in range(neighbor_state.shape[0]):
            if not bool(neighbor_mask[slot]):
                continue
            nb_length = float(neighbor_state[slot, 4]) if float(neighbor_state[slot, 4]) > 0 else cfg.ego_length_m
            nb_width = float(neighbor_state[slot, 5]) if float(neighbor_state[slot, 5]) > 0 else cfg.ego_width_m
            dx = np.abs(neighbor_xy[slot, :, 0] - ego_xy[:, 0])
            dy = np.abs(neighbor_xy[slot, :, 1] - ego_xy[:, 1])
            long_limit = 0.5 * (cfg.ego_length_m + nb_length) + cfg.collision_longitudinal_margin_m
            lat_limit = 0.5 * (cfg.ego_width_m + nb_width) + cfg.collision_lateral_margin_m
            signed = np.maximum(dx - long_limit, dy - lat_limit)
            frame = int(np.argmin(signed))
            if float(signed[frame]) < min_clearance:
                min_clearance = float(signed[frame])
                min_slot = slot
                min_frame = frame
            collision = collision or bool(np.any((dx < long_limit) & (dy < lat_limit)))
        if min_clearance == float("inf"):
            min_clearance = 999.0
        return collision, min_clearance, min_slot, min_frame


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)
