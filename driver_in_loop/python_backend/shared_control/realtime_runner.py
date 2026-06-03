from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from input_adapters.base import DriverCommand
from network.udp_state import command_to_dict


@dataclass
class RealtimeRunnerConfig:
    frame_rate: float = 25.0
    source_frame_rate: float = 25.0
    wheelbase_m: float = 2.7
    ego_length_m: float = 4.6
    ego_width_m: float = 1.8
    max_abs_steer_rad: float = 0.20
    max_abs_accel_mps2: float = 4.0
    max_steer_rate_radps: float = 0.35
    max_jerk_mps3: float = 8.0
    lane_width_m: float = 3.5
    collision_longitudinal_margin_m: float = 0.8
    collision_lateral_margin_m: float = 0.35
    safety_front_gap_m: float = 9.0
    safety_ttc_s: float = 3.0
    driver_authority_gain: float = 1.0
    driver_accel_gain: float = 1.8
    mpc_smoothing_gain: float = 0.55
    replay_policy_key: str | None = None
    replay_policy_mode: str = "state"
    paper_case_id: int | None = None
    dil_mode: str = "ta_rldm_armpc"
    policy_tracking_lookahead_s: float = 1.2
    intention_horizon_s: float = 4.0
    intention_stride: int = 4


class RealtimeSharedControlRunner:
    """Real-time DIL runner based on exported highD validation rollouts.

    Surrounding vehicles are replayed from highD-derived trajectories. The ego
    vehicle is updated online by blending the live driver command with the
    machine command through the precomputed shared-authority sequence. A
    lightweight safety/comfort filter emulates the MPC-lite layer for real-time
    driver-in-the-loop prototyping.
    """

    def __init__(self, case: dict[str, Any], config: RealtimeRunnerConfig | None = None) -> None:
        self.case = case
        self.config = config or RealtimeRunnerConfig(frame_rate=float(case.get("frame_rate", 25.0)))
        self.frame_rate = float(self.config.frame_rate)
        self.source_frame_rate = float(self.config.source_frame_rate)
        self.dt = 1.0 / self.frame_rate
        self.n = len(case["controller_ego"]["xy"])
        self.duration_s = max(0.0, (self.n - 1) / max(self.source_frame_rate, 1e-6))
        self.reset()

    def reset(self) -> None:
        start = self.case["controller_ego"]
        xy0 = np.asarray(start["xy"], dtype=float)[0]
        self.x = float(xy0[0])
        self.y = float(xy0[1])
        self.yaw = float(np.asarray(start["yaw"], dtype=float)[0]) if "yaw" in start else 0.0
        self.speed = float(np.asarray(start["speed"], dtype=float)[0])
        self.acceleration = 0.0
        self.steer = 0.0
        self.step_index = 0
        self.elapsed_s = 0.0
        self.prev_accel = 0.0
        self.prev_steer = 0.0
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_yaw_for_dynamics = self.yaw
        self.beta = 0.0
        self.yaw_rate = 0.0
        self.lateral_velocity = 0.0
        self.lateral_acceleration = 0.0
        self.driver_intent_strength = 0.0
        self.effective_human_authority = 0.0
        self.collision = False

    def step(self, command: DriverCommand) -> dict[str, Any]:
        if command.reset or self.elapsed_s > self.duration_s:
            self.reset()

        sample_index = min(self.elapsed_s * self.source_frame_rate, self.n - 1)
        i = int(math.floor(sample_index))
        machine = self._method_at("machine_ego", sample_index)
        authority_ref = self._signal_at("authority_ref", sample_index, fallback=0.5)
        authority_rl = self._signal_at("authority_rl", sample_index, fallback=authority_ref)
        authority_ra = self._signal_at("authority_ra", sample_index, fallback=authority_ref)
        trust_hm = self._signal_at("trust_human_to_machine", sample_index, fallback=0.7)
        trust_mh = self._signal_at("trust_machine_to_human", sample_index, fallback=0.7)
        urgency = self._signal_at("environment_urgency", sample_index, fallback=0.0)

        if self.config.replay_policy_key:
            if self.config.replay_policy_mode == "state":
                self.driver_intent_strength = 0.0
                self.effective_human_authority = 0.0
                self._replay_policy_state(self.config.replay_policy_key, sample_index)
                front_distance, ttc = self._front_risk(sample_index)
            else:
                policy = self._method_at(self.config.replay_policy_key, sample_index)
                front_distance, ttc = self._front_risk(sample_index)
                accel_ref = self._safety_filter_accel(policy["acceleration"], front_distance, ttc)
                steer_cmd, accel_cmd = self._smooth_control(policy["steer"], accel_ref)
                self._integrate(steer_cmd, accel_cmd)
                front_distance, ttc = self._front_risk(sample_index)
            self.collision = self.collision or self._collision_check(sample_index)
        else:
            mode = self.config.dil_mode
            active_authority = self._active_authority(mode, authority_ref, authority_rl, authority_ra)
            lam = float(np.clip(active_authority * self.config.driver_authority_gain, 0.0, 1.0))
            driver_accel = float(command.acceleration_mps2) * self.config.driver_accel_gain
            if mode == "human_only":
                steer_ref = command.delta_rad
                accel_ref = driver_accel
                lam = 1.0
                self.driver_intent_strength = self._driver_intent_strength(command)
                self.effective_human_authority = 1.0
            else:
                method_steer, method_accel = self._shared_method_tracking_control(mode, sample_index)
                intent_strength = self._driver_intent_strength(command)
                conflict = self._control_conflict(command.delta_rad, method_steer)
                lam_eff = lam * intent_strength * (1.0 - 0.75 * conflict)
                self.driver_intent_strength = intent_strength
                self.effective_human_authority = lam_eff
                steer_ref = lam_eff * command.delta_rad + (1.0 - lam_eff) * method_steer
                accel_ref = lam_eff * driver_accel + (1.0 - lam_eff) * method_accel

            front_distance, ttc = self._front_risk(sample_index)
            if mode in ("ta_rldm_armpc", "ra_rldm"):
                accel_ref = self._safety_filter_accel(accel_ref, front_distance, ttc)
                steer_cmd, accel_cmd = self._smooth_control(steer_ref, accel_ref)
            else:
                steer_cmd, accel_cmd = self._direct_physical_control(steer_ref, accel_ref)
            self._integrate(steer_cmd, accel_cmd)
            self.collision = self.collision or self._collision_check(sample_index)

        state = self._state_payload(
            i=i,
            sample_index=sample_index,
            command=command,
            machine=machine,
            authority_ref=authority_ref,
            authority_rl=authority_rl,
            authority_ra=authority_ra,
            trust_hm=trust_hm,
            trust_mh=trust_mh,
            urgency=urgency,
            front_distance=front_distance,
            ttc=ttc,
        )
        self.step_index += 1
        self.elapsed_s += self.dt
        return state

    def _active_authority(self, mode: str, authority_ref: float, authority_rl: float, authority_ra: float) -> float:
        if mode == "human_only":
            return 1.0
        if mode == "ra_rldm":
            return authority_ra
        if mode in ("ta_rldm", "ta_rldm_armpc"):
            return authority_rl
        return authority_ref

    def _driver_intent_strength(self, command: DriverCommand) -> float:
        steer_intent = min(1.0, abs(float(command.delta_rad)) / max(self.config.max_abs_steer_rad, 1e-6))
        pedal_intent = min(1.0, max(abs(float(command.throttle)), abs(float(command.brake))))
        return float(np.clip(max(steer_intent, pedal_intent), 0.0, 1.0))

    def _control_conflict(self, driver_steer: float, method_steer: float) -> float:
        if abs(driver_steer) < 1e-3 or abs(method_steer) < 1e-3:
            return 0.0
        if driver_steer * method_steer >= 0.0:
            return 0.0
        driver_level = min(1.0, abs(driver_steer) / max(self.config.max_abs_steer_rad, 1e-6))
        method_level = min(1.0, abs(method_steer) / max(self.config.max_abs_steer_rad, 1e-6))
        return float(np.clip(max(driver_level, method_level), 0.0, 1.0))

    def _method_key_for_mode(self, mode: str) -> str:
        if mode == "ta_rldm":
            return "reference_ego"
        if mode == "ta_rldm_armpc":
            return "controller_ego"
        if mode == "ra_rldm":
            return "ra_rldm_ego"
        return "human_pred_ego"

    def _shared_method_tracking_control(self, mode: str, i: float) -> tuple[float, float]:
        key = self._method_key_for_mode(mode)
        if key not in self.case:
            return self._method_at("machine_ego", i)["steer"], self._method_at("machine_ego", i)["acceleration"]

        lookahead = max(1.0, self.config.policy_tracking_lookahead_s * self.source_frame_rate)
        target_i = min(i + lookahead, self.n - 1)
        vehicle = self.case[key]
        target_xy = self._sample_xy(vehicle["xy"], target_i)
        target_yaw = self._sample_angle(vehicle.get("yaw", [0.0] * self.n), target_i)
        target_speed = self._sample_scalar(vehicle.get("speed", [self.speed] * self.n), target_i)
        target_accel = self._sample_scalar(vehicle.get("acceleration", [0.0] * self.n), i)
        target_steer = self._sample_scalar(vehicle.get("steer", [0.0] * self.n), i)

        lateral_error = float(target_xy[1] - self.y)
        yaw_error = _wrap_angle(target_yaw - self.yaw)
        speed_error = float(target_speed - self.speed)

        steer_ref = target_steer + 0.055 * lateral_error + 0.42 * yaw_error
        accel_ref = target_accel + 0.75 * speed_error
        return steer_ref, accel_ref

    def _method_at(self, key: str, i: float) -> dict[str, float]:
        vehicle = self.case[key]
        speed = self._sample_scalar(vehicle.get("speed", [0.0] * self.n), i)
        accel = self._sample_scalar(vehicle.get("acceleration", [0.0] * self.n), i)
        steer = self._sample_scalar(vehicle.get("steer", [0.0] * self.n), i)
        return {"speed": speed, "acceleration": accel, "steer": steer}

    def _replay_policy_state(self, key: str, i: float) -> None:
        vehicle = self.case[key]
        xy = self._sample_xy(vehicle["xy"], i)
        self.x = float(xy[0])
        self.y = float(xy[1])
        self.yaw = self._sample_angle(vehicle.get("yaw", [0.0] * self.n), i)
        self.speed = self._sample_scalar(vehicle.get("speed", [0.0] * self.n), i)
        self.acceleration = self._sample_scalar(vehicle.get("acceleration", [0.0] * self.n), i)
        self.steer = self._sample_scalar(vehicle.get("steer", [0.0] * self.n), i)
        self.prev_accel = self.acceleration
        self.prev_steer = self.steer

    def _signal_at(self, key: str, i: float, fallback: float) -> float:
        signals = self.case.get("signals", {})
        values = signals.get(key)
        if values is None or len(values) == 0:
            return float(fallback)
        return self._sample_scalar(values, i)

    def _sample_scalar(self, values: Any, i: float) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return 0.0
        lo = int(np.clip(math.floor(i), 0, arr.size - 1))
        hi = int(np.clip(lo + 1, 0, arr.size - 1))
        alpha = float(np.clip(i - lo, 0.0, 1.0))
        return float((1.0 - alpha) * arr[lo] + alpha * arr[hi])

    def _sample_xy(self, values: Any, i: float) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        lo = int(np.clip(math.floor(i), 0, len(arr) - 1))
        hi = int(np.clip(lo + 1, 0, len(arr) - 1))
        alpha = float(np.clip(i - lo, 0.0, 1.0))
        return (1.0 - alpha) * arr[lo] + alpha * arr[hi]

    def _sample_angle(self, values: Any, i: float) -> float:
        arr = np.unwrap(np.asarray(values, dtype=float))
        return _wrap_angle(self._sample_scalar(arr, i))

    def _front_risk(self, i: float) -> tuple[float, float]:
        front = None
        for vehicle in self.case.get("neighbors", []):
            if int(vehicle.get("slot", -1)) == 0:
                front = vehicle
                break
        if front is None:
            return 999.0, 999.0
        p = self._sample_xy(front["xy"], i)
        dx = float(p[0] - self.x)
        dy = abs(float(p[1] - self.y))
        if dy > self.config.lane_width_m * 0.55 or dx <= 0.0:
            return 999.0, 999.0
        gap = dx - 0.5 * (float(front.get("length", 4.6)) + self.config.ego_length_m)
        pp = self._sample_xy(front["xy"], max(0.0, i - 1.0))
        front_v = float((p[0] - pp[0]) * self.source_frame_rate) if i > 0 else self.speed
        closing = max(0.0, self.speed - front_v)
        ttc = gap / closing if closing > 0.1 else 999.0
        return float(gap), float(ttc)

    def _safety_filter_accel(self, accel_ref: float, front_distance: float, ttc: float) -> float:
        safe_gap = self.config.safety_front_gap_m + 0.35 * self.speed
        if front_distance < safe_gap:
            accel_ref = min(accel_ref, -1.8 - 0.08 * (safe_gap - front_distance))
        if ttc < self.config.safety_ttc_s:
            accel_ref = min(accel_ref, -3.2)
        return accel_ref

    def _smooth_control(self, steer_ref: float, accel_ref: float) -> tuple[float, float]:
        cfg = self.config
        steer_ref = float(np.clip(steer_ref, -cfg.max_abs_steer_rad, cfg.max_abs_steer_rad))
        accel_ref = float(np.clip(accel_ref, -cfg.max_abs_accel_mps2, cfg.max_abs_accel_mps2))

        max_dsteer = cfg.max_steer_rate_radps * self.dt
        steer = self.prev_steer + float(np.clip(steer_ref - self.prev_steer, -max_dsteer, max_dsteer))

        max_daccel = cfg.max_jerk_mps3 * self.dt
        accel = self.prev_accel + float(np.clip(accel_ref - self.prev_accel, -max_daccel, max_daccel))
        steer = cfg.mpc_smoothing_gain * steer + (1.0 - cfg.mpc_smoothing_gain) * self.prev_steer
        accel = cfg.mpc_smoothing_gain * accel + (1.0 - cfg.mpc_smoothing_gain) * self.prev_accel
        return steer, accel

    def _direct_physical_control(self, steer_ref: float, accel_ref: float) -> tuple[float, float]:
        cfg = self.config
        steer_ref = float(np.clip(steer_ref, -cfg.max_abs_steer_rad, cfg.max_abs_steer_rad))
        accel_ref = float(np.clip(accel_ref, -cfg.max_abs_accel_mps2, cfg.max_abs_accel_mps2))
        max_dsteer = cfg.max_steer_rate_radps * self.dt * 1.8
        max_daccel = cfg.max_jerk_mps3 * self.dt * 1.5
        steer = self.prev_steer + float(np.clip(steer_ref - self.prev_steer, -max_dsteer, max_dsteer))
        accel = self.prev_accel + float(np.clip(accel_ref - self.prev_accel, -max_daccel, max_daccel))
        return steer, accel

    def _integrate(self, steer: float, acceleration: float) -> None:
        cfg = self.config
        prev_x, prev_y, prev_yaw = self.x, self.y, self.yaw
        self.speed = max(0.0, self.speed + acceleration * self.dt)
        yaw_rate = self.speed / max(cfg.wheelbase_m, 1e-6) * math.tan(steer)
        self.yaw = _wrap_angle(self.yaw + yaw_rate * self.dt)
        self.x += self.speed * math.cos(self.yaw) * self.dt
        self.y += self.speed * math.sin(self.yaw) * self.dt
        self.steer = steer
        self.acceleration = acceleration
        self.prev_steer = steer
        self.prev_accel = acceleration
        self._update_dynamics(prev_x, prev_y, prev_yaw)

    def _update_dynamics(self, prev_x: float, prev_y: float, prev_yaw: float) -> None:
        vx = (self.x - prev_x) / max(self.dt, 1e-6)
        vy = (self.y - prev_y) / max(self.dt, 1e-6)
        velocity_yaw = math.atan2(vy, vx) if abs(vx) + abs(vy) > 1e-6 else self.yaw
        self.beta = _wrap_angle(velocity_yaw - self.yaw)
        self.yaw_rate = _wrap_angle(self.yaw - prev_yaw) / max(self.dt, 1e-6)
        self.lateral_velocity = self.speed * math.sin(self.beta)
        self.lateral_acceleration = self.speed * self.yaw_rate

    def _collision_check(self, i: float) -> bool:
        for vehicle in self.case.get("neighbors", []):
            p = self._sample_xy(vehicle["xy"], i)
            dx = abs(float(p[0] - self.x))
            dy = abs(float(p[1] - self.y))
            long_lim = 0.5 * (float(vehicle.get("length", 4.6)) + self.config.ego_length_m) + self.config.collision_longitudinal_margin_m
            lat_lim = 0.5 * (float(vehicle.get("width", 1.8)) + self.config.ego_width_m) + self.config.collision_lateral_margin_m
            if dx < long_lim and dy < lat_lim:
                return True
        return False

    def _state_payload(
        self,
        *,
        i: int,
        sample_index: float,
        command: DriverCommand,
        machine: dict[str, float],
        authority_ref: float,
        authority_rl: float,
        authority_ra: float,
        trust_hm: float,
        trust_mh: float,
        urgency: float,
        front_distance: float,
        ttc: float,
    ) -> dict[str, Any]:
        return {
            "type": "sim_state",
            "session_id": "",
            "paper_case_id": int(self.config.paper_case_id) if self.config.paper_case_id is not None else 0,
            "case_id": int(self.case["record"]["case_id"]),
            "case_name": self.case["record"].get("case_name", ""),
            "frame_index": int(i),
            "time_s": round(self.elapsed_s, 4),
            "dt": self.dt,
            "ego": {
                "x": self.x,
                "y": self.y,
                "yaw": self.yaw,
                "speed": self.speed,
                "acceleration": self.acceleration,
                "steer": self.steer,
                "length": self.config.ego_length_m,
                "width": self.config.ego_width_m,
            },
            "vehicles": self._vehicles_at(sample_index),
            "intention": self._intention_payload(sample_index),
            "driver_input": command_to_dict(command),
            "machine": machine,
            "authority": {
                "reference": authority_ref,
                "rl": authority_rl,
                "ra": authority_ra,
                "active": self._active_authority(self.config.dil_mode, authority_ref, authority_rl, authority_ra),
                "driver_intent_strength": self.driver_intent_strength,
                "effective_human": self.effective_human_authority,
            },
            "trust": {"human_to_machine": trust_hm, "machine_to_human": trust_mh},
            "dynamics": {
                "sideslip_angle_beta": self.beta,
                "yaw_rate": self.yaw_rate,
                "lateral_velocity": self.lateral_velocity,
                "lateral_acceleration": self.lateral_acceleration,
            },
            "method_states": self._method_states_at(sample_index),
            "mode": self.config.dil_mode,
            "risk": {
                "environment_urgency": urgency,
                "front_distance_m": front_distance,
                "ttc_s": ttc,
            },
            "safety": {"collision": self.collision},
            "road": self.case.get("road", {}),
        }

    def _method_states_at(self, i: float) -> dict[str, dict[str, float]]:
        out = {}
        for key in ("human_pred_ego", "machine_ego", "reference_ego", "ra_rldm_ego", "controller_ego"):
            if key not in self.case:
                continue
            vehicle = self.case[key]
            xy = self._sample_xy(vehicle["xy"], i)
            out[key] = {
                "x": float(xy[0]),
                "y": float(xy[1]),
                "yaw": self._sample_angle(vehicle.get("yaw", [0.0] * self.n), i),
                "speed": self._sample_scalar(vehicle.get("speed", [0.0] * self.n), i),
                "acceleration": self._sample_scalar(vehicle.get("acceleration", [0.0] * self.n), i),
                "steer": self._sample_scalar(vehicle.get("steer", [0.0] * self.n), i),
                "beta": self._sample_scalar(vehicle.get("beta", [0.0] * self.n), i),
                "yaw_rate": self._sample_scalar(vehicle.get("yaw_rate", [0.0] * self.n), i),
            }
        return out

    def _intention_payload(self, i: float) -> dict[str, Any]:
        if self.config.dil_mode == "human_only":
            return {"machine": [], "human": []}
        return {
            "machine": self._trajectory_points("machine_ego", i),
            "human": self._trajectory_points("human_pred_ego", i),
        }

    def _trajectory_points(self, key: str, i: float) -> list[dict[str, float]]:
        if key not in self.case:
            return []
        horizon_frames = int(round(self.config.intention_horizon_s * self.source_frame_rate))
        stride = max(1, int(self.config.intention_stride))
        end = min(self.n - 1, int(math.ceil(i + horizon_frames)))
        points = []
        for idx in range(int(math.floor(i)), end + 1, stride):
            xy = self._sample_xy(self.case[key]["xy"], float(idx))
            points.append({"x": float(xy[0]), "y": float(xy[1])})
        return points

    def _vehicles_at(self, i: float) -> list[dict[str, Any]]:
        vehicles = []
        for item in self.case.get("neighbors", []):
            p = self._sample_xy(item["xy"], i)
            prev = self._sample_xy(item["xy"], max(0.0, i - 1.0))
            yaw = math.atan2(float(p[1] - prev[1]), float(p[0] - prev[0])) if i > 0 else 0.0
            vehicles.append(
                {
                    "id": str(item.get("name", item.get("slot", len(vehicles)))),
                    "slot": int(item.get("slot", -1)),
                    "name": str(item.get("name", "")),
                    "x": float(p[0]),
                    "y": float(p[1]),
                    "yaw": yaw,
                    "length": float(item.get("length", 4.6)),
                    "width": float(item.get("width", 1.8)),
                }
            )
        return vehicles


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
