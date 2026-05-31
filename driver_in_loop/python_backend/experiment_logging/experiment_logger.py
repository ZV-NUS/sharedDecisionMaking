from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Append driver-in-the-loop states to a CSV file."""

    FIELDNAMES = [
        "time_s",
        "case_id",
        "ego_x",
        "ego_y",
        "ego_yaw",
        "ego_speed",
        "ego_acceleration",
        "ego_steer",
        "driver_steer",
        "driver_throttle",
        "driver_brake",
        "driver_delta_rad",
        "driver_acceleration_mps2",
        "machine_steer",
        "machine_acceleration",
        "authority_ref",
        "authority_ra_rldm",
        "authority_ta_rldm",
        "authority_active",
        "authority_rl",
        "driver_intent_strength",
        "effective_human_authority",
        "trust_human_to_machine",
        "trust_machine_to_human",
        "environment_urgency",
        "front_distance_m",
        "ttc_s",
        "collision",
        "sideslip_angle_beta",
        "yaw_rate",
        "lateral_velocity",
        "lateral_acceleration",
        "human_x",
        "human_y",
        "machine_x",
        "machine_y",
        "reference_x",
        "reference_y",
        "ra_rldm_x",
        "ra_rldm_y",
        "ta_rldm_armpc_x",
        "ta_rldm_armpc_y",
        "human_speed",
        "machine_speed",
        "ra_rldm_speed",
        "ta_rldm_armpc_speed",
        "human_steer",
        "machine_method_steer",
        "ra_rldm_steer",
        "ta_rldm_armpc_steer",
        "mode",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDNAMES)
        self.writer.writeheader()

    def write(self, state: dict[str, Any]) -> None:
        ego = state["ego"]
        driver = state["driver_input"]
        machine = state["machine"]
        authority = state["authority"]
        dynamics = state.get("dynamics", {})
        methods = state.get("method_states", {})
        human = methods.get("human_pred_ego", {})
        machine_state = methods.get("machine_ego", {})
        reference = methods.get("reference_ego", {})
        ra = methods.get("ra_rldm_ego", {})
        proposed = methods.get("controller_ego", {})
        self.writer.writerow(
            {
                "time_s": state["time_s"],
                "case_id": state["case_id"],
                "ego_x": ego["x"],
                "ego_y": ego["y"],
                "ego_yaw": ego["yaw"],
                "ego_speed": ego["speed"],
                "ego_acceleration": ego["acceleration"],
                "ego_steer": ego["steer"],
                "driver_steer": driver["steer"],
                "driver_throttle": driver["throttle"],
                "driver_brake": driver["brake"],
                "driver_delta_rad": driver["delta_rad"],
                "driver_acceleration_mps2": driver["acceleration_mps2"],
                "machine_steer": machine["steer"],
                "machine_acceleration": machine["acceleration"],
                "authority_ref": authority["reference"],
                "authority_ra_rldm": authority.get("ra", ""),
                "authority_ta_rldm": authority.get("rl", ""),
                "authority_active": authority.get("active", ""),
                "authority_rl": authority.get("rl", ""),
                "driver_intent_strength": authority.get("driver_intent_strength", ""),
                "effective_human_authority": authority.get("effective_human", ""),
                "trust_human_to_machine": state["trust"]["human_to_machine"],
                "trust_machine_to_human": state["trust"]["machine_to_human"],
                "environment_urgency": state["risk"]["environment_urgency"],
                "front_distance_m": state["risk"]["front_distance_m"],
                "ttc_s": state["risk"]["ttc_s"],
                "collision": state["safety"]["collision"],
                "sideslip_angle_beta": dynamics.get("sideslip_angle_beta", ""),
                "yaw_rate": dynamics.get("yaw_rate", ""),
                "lateral_velocity": dynamics.get("lateral_velocity", ""),
                "lateral_acceleration": dynamics.get("lateral_acceleration", ""),
                "human_x": human.get("x", ""),
                "human_y": human.get("y", ""),
                "machine_x": machine_state.get("x", ""),
                "machine_y": machine_state.get("y", ""),
                "reference_x": reference.get("x", ""),
                "reference_y": reference.get("y", ""),
                "ra_rldm_x": ra.get("x", ""),
                "ra_rldm_y": ra.get("y", ""),
                "ta_rldm_armpc_x": proposed.get("x", ""),
                "ta_rldm_armpc_y": proposed.get("y", ""),
                "human_speed": human.get("speed", ""),
                "machine_speed": machine_state.get("speed", ""),
                "ra_rldm_speed": ra.get("speed", ""),
                "ta_rldm_armpc_speed": proposed.get("speed", ""),
                "human_steer": human.get("steer", ""),
                "machine_method_steer": machine_state.get("steer", ""),
                "ra_rldm_steer": ra.get("steer", ""),
                "ta_rldm_armpc_steer": proposed.get("steer", ""),
                "mode": state.get("mode", ""),
            }
        )

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        path = self.path.parent / "metadata.json"
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        self.file.flush()
        self.file.close()
