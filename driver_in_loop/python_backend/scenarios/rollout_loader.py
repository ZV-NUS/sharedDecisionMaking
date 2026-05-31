from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_rollouts(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8").strip()
    prefix = "window.SHARED_AUTHORITY_ROLLOUTS = "
    if text.startswith(prefix):
        text = text[len(prefix) :]
    if text.endswith(";"):
        text = text[:-1]
    return json.loads(text)


class RolloutScenarioRepository:
    """Load highD-derived validation cases exported by the existing pipeline."""

    def __init__(self, rollout_js: Path) -> None:
        self.rollout_js = rollout_js
        self.data = load_rollouts(rollout_js)

    @property
    def frame_rate(self) -> float:
        return float(self.data.get("frame_rate", 25.0))

    @property
    def lane_width_m(self) -> float:
        return float(self.data.get("lane_width_m", 3.5))

    def case_ids(self) -> list[int]:
        return [int(case["record"]["case_id"]) for case in self.data["cases"]]

    def get_case(self, case_id: int) -> dict[str, Any]:
        for case in self.data["cases"]:
            if int(case["record"]["case_id"]) == int(case_id):
                return case
        raise KeyError(f"Cannot find rollout case_id={case_id}; available={self.case_ids()}")
