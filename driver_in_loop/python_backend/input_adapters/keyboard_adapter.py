from __future__ import annotations

import os

from .base import DriverCommand, DriverInputAdapter


class KeyboardInputAdapter(DriverInputAdapter):
    """Non-blocking Windows-console keyboard adapter.

    This adapter is a temporary substitute for Logitech G29. It outputs the
    same normalized driver command as the wheel adapter, so the downstream
    shared-control stack does not need to know which device is used.
    """

    def __init__(self, *args, steer_step: float = 0.12, pedal_step: float = 0.10, decay: float = 0.88, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.steer_step = float(steer_step)
        self.pedal_step = float(pedal_step)
        self.decay = float(decay)
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self._msvcrt = None
        if os.name == "nt":
            import msvcrt

            self._msvcrt = msvcrt

    def read(self) -> DriverCommand:
        reset = False
        quit_requested = False
        touched_steer = False
        touched_pedal = False

        if self._msvcrt is None:
            return self._command(self.steer, self.throttle, self.brake, source="keyboard")

        while self._msvcrt.kbhit():
            key = self._msvcrt.getch()
            if key in (b"\x00", b"\xe0"):
                key = self._msvcrt.getch()
                if key == b"K":  # left arrow
                    self.steer -= self.steer_step
                    touched_steer = True
                elif key == b"M":  # right arrow
                    self.steer += self.steer_step
                    touched_steer = True
                elif key == b"H":  # up arrow
                    self.throttle += self.pedal_step
                    self.brake *= 0.55
                    touched_pedal = True
                elif key == b"P":  # down arrow
                    self.brake += self.pedal_step
                    self.throttle *= 0.55
                    touched_pedal = True
                continue

            lower = key.lower()
            if lower == b"q":
                quit_requested = True
            elif lower == b"r":
                reset = True
            elif key == b" ":
                self.brake = 1.0
                self.throttle = 0.0
                touched_pedal = True
            elif lower == b"a":
                self.steer -= self.steer_step
                touched_steer = True
            elif lower == b"d":
                self.steer += self.steer_step
                touched_steer = True
            elif lower == b"w":
                self.throttle += self.pedal_step
                self.brake *= 0.55
                touched_pedal = True
            elif lower == b"s":
                self.brake += self.pedal_step
                self.throttle *= 0.55
                touched_pedal = True

        if not touched_steer:
            self.steer *= self.decay
        if not touched_pedal:
            self.throttle *= 0.96
            self.brake *= 0.92

        self.steer = max(-1.0, min(1.0, self.steer))
        self.throttle = max(0.0, min(1.0, self.throttle))
        self.brake = max(0.0, min(1.0, self.brake))

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
            source="keyboard",
        )
