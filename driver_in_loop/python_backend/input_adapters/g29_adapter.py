from __future__ import annotations

from .base import DriverCommand, DriverInputAdapter


class G29InputAdapter(DriverInputAdapter):
    """Logitech G29 continuous driver-intention adapter.

    The wheel is treated exactly like the keyboard substitute used in DIL:
    steering, throttle, and brake are continuous human control intentions. The
    downstream shared-control runner decides how much of this intention is
    accepted according to trust, risk, authority, and the selected DIL mode.
    """

    def __init__(
        self,
        *args,
        joystick_index: int = 0,
        steer_axis: int = 0,
        throttle_axis: int = 2,
        brake_axis: int = 3,
        invert_steer: bool = False,
        invert_pedals: bool = True,
        deadzone: float = 0.03,
        smoothing: float = 0.38,
        steer_shape_exponent: float = 1.45,
        pedal_shape_exponent: float = 0.65,
        brake_shape_exponent: float = 0.30,
        brake_gain: float = 2.00,
        brake_rise_smoothing: float = 0.15,
        reset_button: int = 6,
        quit_button: int = 7,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.joystick_index = int(joystick_index)
        self.steer_axis = int(steer_axis)
        self.throttle_axis = int(throttle_axis)
        self.brake_axis = int(brake_axis)
        self.invert_steer = bool(invert_steer)
        self.invert_pedals = bool(invert_pedals)
        self.deadzone = max(0.0, min(0.5, float(deadzone)))
        self.smoothing = max(0.0, min(0.95, float(smoothing)))
        self.steer_shape_exponent = max(1.0, float(steer_shape_exponent))
        self.pedal_shape_exponent = max(0.25, min(2.0, float(pedal_shape_exponent)))
        self.brake_shape_exponent = max(0.25, min(2.0, float(brake_shape_exponent)))
        self.brake_gain = max(0.1, min(3.0, float(brake_gain)))
        self.brake_rise_smoothing = max(0.0, min(0.95, float(brake_rise_smoothing)))
        self.reset_button = int(reset_button)
        self.quit_button = int(quit_button)
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.throttle_rest_raw = 1.0
        self.brake_rest_raw = 1.0
        self.throttle_min_raw = 1.0
        self.brake_min_raw = 1.0
        try:
            import pygame  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local hardware
            raise RuntimeError(
                "G29InputAdapter requires pygame and a connected Logitech G29. "
                "Use --input keyboard until the wheel is available."
            ) from exc

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() <= self.joystick_index:
            raise RuntimeError("No Logitech G29-compatible joystick was detected.")
        self._pygame = pygame
        self._joy = pygame.joystick.Joystick(self.joystick_index)
        self._joy.init()
        pygame.event.pump()
        self.throttle_rest_raw = self._safe_axis(self.throttle_axis)
        self.brake_rest_raw = self._safe_axis(self.brake_axis)
        self.throttle_min_raw = self.throttle_rest_raw
        self.brake_min_raw = self.brake_rest_raw

    def read(self) -> DriverCommand:  # pragma: no cover - depends on local hardware
        pg = self._pygame
        pg.event.pump()
        steer_raw = self._safe_axis(self.steer_axis)
        throttle_raw = self._safe_axis(self.throttle_axis)
        brake_raw = self._safe_axis(self.brake_axis)

        steer = -steer_raw if self.invert_steer else steer_raw
        steer = self._apply_deadzone(steer)
        steer = self._shape_steer(steer)
        if self.invert_pedals:
            self.throttle_min_raw = min(self.throttle_min_raw, throttle_raw)
            self.brake_min_raw = min(self.brake_min_raw, brake_raw)
        else:
            self.throttle_min_raw = max(self.throttle_min_raw, throttle_raw)
            self.brake_min_raw = max(self.brake_min_raw, brake_raw)
        throttle = self._pedal_to_unit(throttle_raw, self.throttle_rest_raw, self.throttle_min_raw)
        brake = self._pedal_to_unit(
            brake_raw,
            self.brake_rest_raw,
            self.brake_min_raw,
            shape_exponent=self.brake_shape_exponent,
            gain=self.brake_gain,
        )

        alpha = 1.0 - self.smoothing
        self.steer = self.smoothing * self.steer + alpha * steer
        self.throttle = self.smoothing * self.throttle + alpha * throttle
        brake_smoothing = self.brake_rise_smoothing if brake > self.brake else self.smoothing
        brake_alpha = 1.0 - brake_smoothing
        self.brake = brake_smoothing * self.brake + brake_alpha * brake

        reset = self._safe_button(self.reset_button)
        quit_requested = self._safe_button(self.quit_button)
        if reset:
            self.steer = 0.0
            self.throttle = 0.0
            self.brake = 0.0

        return self._command(
            self.steer,
            self.throttle,
            self.brake,
            reset=reset,
            quit=quit_requested,
            source="g29",
        )

    def close(self) -> None:  # pragma: no cover - depends on local hardware
        self._joy.quit()
        self._pygame.joystick.quit()

    def _safe_axis(self, axis: int) -> float:
        if axis < 0 or axis >= self._joy.get_numaxes():
            return 0.0
        return float(self._joy.get_axis(axis))

    def _safe_button(self, button: int) -> bool:
        if button < 0 or button >= self._joy.get_numbuttons():
            return False
        return bool(self._joy.get_button(button))

    def _apply_deadzone(self, value: float) -> float:
        value = max(-1.0, min(1.0, float(value)))
        if abs(value) <= self.deadzone:
            return 0.0
        sign = 1.0 if value >= 0.0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _shape_steer(self, value: float) -> float:
        value = max(-1.0, min(1.0, float(value)))
        if abs(value) <= 0.0:
            return 0.0
        return (1.0 if value >= 0.0 else -1.0) * (abs(value) ** self.steer_shape_exponent)

    def _pedal_to_unit(
        self,
        raw: float,
        rest_raw: float,
        travel_raw: float,
        *,
        shape_exponent: float | None = None,
        gain: float = 1.0,
    ) -> float:
        raw = max(-1.0, min(1.0, float(raw)))
        rest_raw = max(-1.0, min(1.0, float(rest_raw)))
        travel_raw = max(-1.0, min(1.0, float(travel_raw)))
        if self.invert_pedals:
            denom = max(0.12, rest_raw - travel_raw)
            value = (rest_raw - raw) / denom
        else:
            denom = max(0.12, travel_raw - rest_raw)
            value = (raw - rest_raw) / denom
        value = max(0.0, min(1.0, value))
        if value <= self.deadzone:
            return 0.0
        normalized = max(0.0, min(1.0, (value - self.deadzone) / (1.0 - self.deadzone)))
        exponent = self.pedal_shape_exponent if shape_exponent is None else float(shape_exponent)
        return max(0.0, min(1.0, gain * (normalized ** exponent)))
