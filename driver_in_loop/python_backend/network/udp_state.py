from __future__ import annotations

import json
import socket
from dataclasses import asdict
from typing import Any

from input_adapters.base import DriverCommand


class UdpStatePublisher:
    """Send real-time simulation states to Unity as UTF-8 JSON datagrams."""

    def __init__(self, host: str = "127.0.0.1", port: int = 50710, enabled: bool = True) -> None:
        self.target = (host, int(port))
        self.enabled = bool(enabled)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        # Keep a conservative datagram size. Unity receives a state snapshot
        # every frame, so dropping oversized debug fields is better than
        # fragmenting UDP packets.
        if len(data) > 60_000:
            payload = dict(payload)
            payload["debug"] = {"warning": "payload truncated before UDP send"}
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.sock.sendto(data, self.target)

    def close(self) -> None:
        self.sock.close()


class UdpDriverInputReceiver:
    """Optional Unity-to-Python driver command receiver.

    This is useful when keyboard or wheel input is handled inside Unity. The
    Python backend can otherwise use KeyboardInputAdapter directly.
    """

    def __init__(self, port: int = 50711, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        if self.enabled:
            self.sock.bind(("127.0.0.1", int(port)))
        self.latest = DriverCommand(source="udp")
        self.ready = False

    def read(self) -> DriverCommand:
        if not self.enabled:
            return self.latest
        while True:
            try:
                raw, _ = self.sock.recvfrom(8192)
            except BlockingIOError:
                break
            msg = json.loads(raw.decode("utf-8"))
            self.ready = self.ready or bool(msg.get("ready", False))
            self.latest = DriverCommand(
                steer=float(msg.get("steer", 0.0)),
                throttle=float(msg.get("throttle", 0.0)),
                brake=float(msg.get("brake", 0.0)),
                delta_rad=float(msg.get("delta_rad", 0.0)),
                acceleration_mps2=float(msg.get("acceleration_mps2", 0.0)),
                reset=bool(msg.get("reset", False)),
                quit=bool(msg.get("quit", False)),
                source=str(msg.get("source", "unity")),
            )
        return self.latest

    def is_ready(self) -> bool:
        if self.enabled:
            self.read()
        return self.ready

    def close(self) -> None:
        self.sock.close()


def command_to_dict(command: DriverCommand) -> dict[str, Any]:
    return asdict(command)
