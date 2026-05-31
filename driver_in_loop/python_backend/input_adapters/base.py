from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass
class DriverCommand:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    delta_rad: float = 0.0
    acceleration_mps2: float = 0.0
    reset: bool = False
    quit: bool = False
    source: str = "unknown"
    timestamp: float = 0.0


class DriverInputAdapter:
    """Base interface for keyboard and Logitech G29 driver input."""

    def __init__(
        self,
        max_steer_rad: float = 0.45,
        max_throttle_accel_mps2: float = 2.5,
        max_brake_decel_mps2: float = 4.5,
    ) -> None:
        self.max_steer_rad = float(max_steer_rad)
        self.max_throttle_accel_mps2 = float(max_throttle_accel_mps2)
        self.max_brake_decel_mps2 = float(max_brake_decel_mps2)

    def read(self) -> DriverCommand:
        raise NotImplementedError

    def _command(
        self,
        steer: float,
        throttle: float,
        brake: float,
        *,
        reset: bool = False,
        quit: bool = False,
        source: str,
    ) -> DriverCommand:
        steer = max(-1.0, min(1.0, float(steer)))
        throttle = max(0.0, min(1.0, float(throttle)))
        brake = max(0.0, min(1.0, float(brake)))
        delta = self.max_steer_rad * steer
        accel = self.max_throttle_accel_mps2 * throttle - self.max_brake_decel_mps2 * brake
        return DriverCommand(
            steer=steer,
            throttle=throttle,
            brake=brake,
            delta_rad=delta,
            acceleration_mps2=accel,
            reset=reset,
            quit=quit,
            source=source,
            timestamp=time.time(),
        )
