from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class HighDInjectedTrafficEnvConfig:
    frame_rate: float = 25.0
    future_len: int = 125
    lane_width_m: float = 3.5
    wheelbase_m: float = 2.7
    ego_length_m: float = 4.6
    ego_width_m: float = 1.8
    collision_longitudinal_margin_m: float = 1.0
    collision_lateral_margin_m: float = 0.4


class HighDInjectedTrafficEnv:
    """HighD sample injection environment for deterministic machine policy checks.

    The environment uses one highD sample as the initial traffic scene. Ego is
    rolled forward by a kinematic bicycle model using machine speed/steering
    intent. Surrounding vehicles are injected from highD neighbor slots and
    propagated by constant velocity from their current relative states.
    """

    def __init__(self, config: HighDInjectedTrafficEnvConfig | None = None) -> None:
        self.config = config or HighDInjectedTrafficEnvConfig()

    def rollout(
        self,
        sample: dict[str, np.ndarray],
        machine_outputs: dict[str, torch.Tensor],
    ) -> dict[str, np.ndarray | float | int | bool]:
        cfg = self.config
        ego_history = np.asarray(sample["ego_history"], dtype=np.float32)
        neighbors = np.asarray(sample["neighbor_history"], dtype=np.float32)[-1]
        neighbor_mask = np.asarray(sample["neighbor_mask"], dtype=np.float32)[-1] > 0.5
        risk = np.asarray(sample["risk_history"], dtype=np.float32)[-1]

        speed = _first_batch_item(machine_outputs["future_speed"]).astype(np.float32)
        steer = _first_batch_item(machine_outputs["future_steer"]).astype(np.float32)
        accel = _first_batch_item(machine_outputs.get("future_acceleration"))
        if accel is None:
            accel = np.gradient(speed, 1.0 / cfg.frame_rate).astype(np.float32)
        decision = int(torch.argmax(machine_outputs["future_event_logits"][0]).detach().cpu())
        event_time = float(machine_outputs["future_event_time_by_class"][0, decision].detach().cpu())
        event_idx = int(np.clip(round(event_time * float(cfg.future_len - 1)), 0, cfg.future_len - 1))

        ego_xy, ego_yaw = self._roll_ego(speed, steer)
        neighbor_xy = self._roll_neighbors(ego_history, neighbors, neighbor_mask, sample)
        collision, min_clearance, min_slot, min_frame = self._check_collision(ego_xy, neighbor_xy, neighbors, neighbor_mask)
        ground_truth = np.asarray(sample.get("future_trajectory", np.zeros((cfg.future_len, 2))), dtype=np.float32)

        return {
            "decision": decision,
            "event_idx": event_idx,
            "front_risk": float(risk[3]),
            "ttc": float(risk[2]),
            "thw": float(risk[1]),
            "dhw": float(risk[0]),
            "ego_xy": ego_xy,
            "ego_yaw": ego_yaw,
            "neighbor_xy": neighbor_xy,
            "neighbor_mask": neighbor_mask,
            "ground_truth_xy": ground_truth,
            "speed": speed,
            "acceleration": accel,
            "steer": steer,
            "collision": bool(collision),
            "min_clearance_m": float(min_clearance),
            "min_clearance_slot": int(min_slot),
            "min_clearance_frame": int(min_frame),
            "max_abs_steer_rad": float(np.max(np.abs(steer))),
            "max_abs_accel_mps2": float(np.max(np.abs(accel))),
            "mean_abs_steer_rad": float(np.mean(np.abs(steer))),
            "mean_abs_accel_mps2": float(np.mean(np.abs(accel))),
        }

    def _roll_ego(self, speed: np.ndarray, steer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        dt = 1.0 / cfg.frame_rate
        xy = np.zeros((cfg.future_len, 2), dtype=np.float32)
        yaw = np.zeros((cfg.future_len,), dtype=np.float32)
        x = 0.0
        y = 0.0
        psi = 0.0
        for i in range(cfg.future_len):
            v = float(speed[i])
            delta = float(np.clip(steer[i], -0.6, 0.6))
            psi += v / max(cfg.wheelbase_m, 1e-6) * np.tan(delta) * dt
            x += v * np.cos(psi) * dt
            y += v * np.sin(psi) * dt
            xy[i] = (x, y)
            yaw[i] = psi
        return xy, yaw

    def _roll_neighbors(
        self,
        ego_history: np.ndarray,
        neighbors: np.ndarray,
        neighbor_mask: np.ndarray,
        sample: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        cfg = self.config
        dt = 1.0 / cfg.frame_rate
        frame_t = (np.arange(cfg.future_len, dtype=np.float32) + 1.0) * dt
        ego_vx = float(ego_history[-1, 2])
        ego_vy = float(ego_history[-1, 3])
        neighbor_ax = None
        if sample is not None and "_neighbor_future_ax" in sample:
            neighbor_ax = np.asarray(sample["_neighbor_future_ax"], dtype=np.float32)
        neighbor_xy = np.zeros((neighbors.shape[0], cfg.future_len, 2), dtype=np.float32)
        for slot in range(neighbors.shape[0]):
            if not bool(neighbor_mask[slot]):
                continue
            rel_x, rel_y, rel_vx, rel_vy = neighbors[slot, :4]
            vx = ego_vx + float(rel_vx)
            vy = ego_vy + float(rel_vy)
            if neighbor_ax is not None and slot < neighbor_ax.shape[0]:
                ax = neighbor_ax[slot, : cfg.future_len]
                vx_profile = vx + np.cumsum(ax, dtype=np.float32) * dt
                neighbor_xy[slot, :, 0] = float(rel_x) + np.cumsum(vx_profile * dt, dtype=np.float32)
            else:
                neighbor_xy[slot, :, 0] = float(rel_x) + vx * frame_t
            neighbor_xy[slot, :, 1] = float(rel_y) + vy * frame_t
        return neighbor_xy

    def _check_collision(
        self,
        ego_xy: np.ndarray,
        neighbor_xy: np.ndarray,
        neighbors: np.ndarray,
        neighbor_mask: np.ndarray,
    ) -> tuple[bool, float, int, int]:
        cfg = self.config
        min_clearance = float("inf")
        min_slot = -1
        min_frame = -1
        collision = False
        for slot in range(neighbors.shape[0]):
            if not bool(neighbor_mask[slot]):
                continue
            nb_length = float(neighbors[slot, 4]) if float(neighbors[slot, 4]) > 0 else cfg.ego_length_m
            nb_width = float(neighbors[slot, 5]) if float(neighbors[slot, 5]) > 0 else cfg.ego_width_m
            dx = np.abs(neighbor_xy[slot, :, 0] - ego_xy[:, 0])
            dy = np.abs(neighbor_xy[slot, :, 1] - ego_xy[:, 1])
            long_limit = 0.5 * (cfg.ego_length_m + nb_length) + cfg.collision_longitudinal_margin_m
            lat_limit = 0.5 * (cfg.ego_width_m + nb_width) + cfg.collision_lateral_margin_m
            long_clearance = dx - long_limit
            lat_clearance = dy - lat_limit
            signed_clearance = np.maximum(long_clearance, lat_clearance)
            local_frame = int(np.argmin(signed_clearance))
            local_clearance = float(signed_clearance[local_frame])
            if local_clearance < min_clearance:
                min_clearance = local_clearance
                min_slot = slot
                min_frame = local_frame
            collision = collision or bool(np.any((dx < long_limit) & (dy < lat_limit)))
        if min_clearance == float("inf"):
            min_clearance = 999.0
        return collision, min_clearance, min_slot, min_frame


def _first_batch_item(value: torch.Tensor | None) -> np.ndarray | None:
    if value is None:
        return None
    return value[0].detach().cpu().numpy()
