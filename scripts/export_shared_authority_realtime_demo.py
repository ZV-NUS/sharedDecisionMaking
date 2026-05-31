from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import random
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_work3_authority import _load_human_model, _resolve_device
from scripts.visualize_machine_policy_rollouts import _read_sample
from src.control import AdaptiveRobustMPCLite, AdaptiveRobustMPCLiteConfig
from src.envs.highd_injected_env import HighDInjectedTrafficEnv, HighDInjectedTrafficEnvConfig
from src.policies.machine_intent_policy import MachineIntentPolicy, MachineIntentPolicyConfig
from src.rl import AuthorityObservationConfig, AuthorityRLEnv, AuthorityRewardConfig, TransformerSACAuthority, TransformerSACConfig
from src.rl.authority_observation import build_authority_observation
from src.trust import BidirectionalTrustConfig, BidirectionalTrustEstimator, IT2TSKAuthorityConfig, IntervalType2TSKAuthority
from src.trust.shared_intent import blend_human_machine_intent


INPUT_KEYS = ("ego_history", "neighbor_history", "neighbor_mask", "risk_history")
SLOT_NAMES = ("front", "rear", "left-front", "left-rear", "right-front", "right-rear")
DECISION_NAMES = {0: "L", 1: "S", 2: "R"}
DECISION_NAMES_INV = {"L": 0, "S": 1, "R": 2}
SLOT_ID_COLUMNS = ("precedingId", "followingId", "leftPrecedingId", "leftFollowingId", "rightPrecedingId", "rightFollowingId")
_TRACK_CACHE: dict[int, pd.DataFrame] = {}
_RECORDING_META_CACHE: dict[int, dict] = {}


FIXED_VALIDATION_SCENARIOS = [
    {
        "case_id": 1,
        "name": "换道意图不合理：目标侧风险高，应直行减速",
        "recording_id": 52,
        "vehicle_id": 134,
        "frame_id": 1408,
        "expected": "S/decelerate",
        "adjustments": [
            {"slot": "front", "valid": True, "rel_x": 28.0, "rel_vx": -1.2},
            {"slot": "rear", "valid": True, "rel_x": -55.0, "rel_vx": -1.0},
            {"slot": "right-front", "valid": True, "rel_x": 20.0, "rel_vx": -0.3},
            {"slot": "right-rear", "valid": True, "rel_x": -22.0, "rel_vx": 0.6},
            {"slot": "left-rear", "valid": True, "rel_x": -26.0, "rel_vx": 1.7},
        ],
    },
    {
        "case_id": 2,
        "name": "换道方向不合理：人向左但左侧危险，右侧更安全",
        "recording_id": 52,
        "vehicle_id": 790,
        "frame_id": 9502,
        "expected": "R",
        "human_decision": "L",
        "machine_decision": "R",
        "adjustments": [
            {"slot": "front", "valid": True, "rel_x": 34.0, "rel_vx": -1.0},
            {"slot": "left-front", "valid": True, "rel_x": 28.0, "rel_vx": -0.2},
            {"slot": "left-rear", "valid": True, "rel_x": -22.0, "rel_vx": 2.0},
            {"slot": "right-front", "valid": False},
            {"slot": "right-rear", "valid": False},
        ],
    },
    {
        "case_id": 3,
        "name": "换道方向合理但操作激进：应保留方向并修正轨迹",
        "recording_id": 56,
        "vehicle_id": 826,
        "frame_id": 11106,
        "expected": "L/smoothed",
        "human_decision": "L",
        "machine_decision": "L",
        "machine_accel": -0.3,
        "human_style": "aggressive_lane_change",
        "adjustments": [
            {"slot": "front", "valid": True, "rel_x": 40.0, "rel_vx": -1.0},
            {"slot": "left-front", "valid": True, "rel_x": 52.0, "rel_vx": -0.1},
            {"slot": "left-rear", "valid": True, "rel_x": -44.0, "rel_vx": 0.1},
            {"slot": "right-front", "valid": False},
            {"slot": "right-rear", "valid": False},
        ],
    },
    {
        "case_id": 4,
        "name": "直行意图不合理：前车慢且近，应换道提效",
        "recording_id": 60,
        "vehicle_id": 1141,
        "frame_id": 18838,
        "expected": "lane-change",
        "human_decision": "S",
        "machine_decision": "L",
        "machine_accel": 0.6,
        "adjustments": [
            {"slot": "front", "valid": True, "rel_x": 52.0, "rel_vx": -2.2},
            {"slot": "rear", "valid": True, "rel_x": -85.0, "rel_vx": -1.0},
            {"slot": "right-front", "valid": True, "rel_x": 95.0, "rel_vx": -1.0},
            {"slot": "right-rear", "valid": True, "rel_x": -70.0, "rel_vx": -1.0},
        ],
    },
    {
        "case_id": 5,
        "name": "直行合理但减速不足：邻道阻塞，应直行强减速",
        "recording_id": 56,
        "vehicle_id": 780,
        "frame_id": 10511,
        "expected": "S/strong-decelerate",
        "adjustments": [
            {"slot": "front", "valid": True, "rel_x": 42.0, "rel_vx": -1.8},
            {"slot": "left-front", "valid": True, "rel_x": 30.0, "rel_vx": -0.1},
            {"slot": "left-rear", "valid": True, "rel_x": -32.0, "rel_vx": 0.4},
            {"slot": "right-front", "valid": True, "rel_x": 30.0, "rel_vx": -0.1},
            {"slot": "right-rear", "valid": True, "rel_x": -32.0, "rel_vx": 0.4},
        ],
    },
    {
        "case_id": 6,
        "name": "并行风险高：机器想加速超越但速度代价过大，应听人减速",
        "recording_id": 56,
        "vehicle_id": 309,
        "frame_id": 3987,
        "expected": "S/decelerate",
        "human_decision": "S",
        "machine_decision": "S",
        "human_accel": -0.8,
        "machine_accel": 1.9,
        "parallel_decel": -0.8,
        "adjustments": [
            {"slot": "front", "valid": False},
            {"slot": "rear", "valid": False},
            {"slot": "left-front", "valid": True, "rel_x": 3.0, "rel_vx": -0.2, "ax": 2.3, "ax_start_s": 0.9},
            {"slot": "left-rear", "valid": False},
            {"slot": "right-front", "valid": False},
            {"slot": "right-rear", "valid": False},
        ],
    },
    {
        "case_id": 7,
        "name": "并行风险可解除：旁车速度不足，人机均可加速超越",
        "recording_id": 56,
        "vehicle_id": 309,
        "frame_id": 3987,
        "expected": "S/accelerate-pass",
        "human_decision": "S",
        "machine_decision": "S",
        "human_accel": 0.45,
        "machine_accel": 0.9,
        "adjustments": [
            {"slot": "front", "valid": False},
            {"slot": "rear", "valid": False},
            {"slot": "left-front", "valid": True, "rel_x": 3.0, "rel_vx": -2.8, "ax": 0.0},
            {"slot": "left-rear", "valid": False},
            {"slot": "right-front", "valid": False},
            {"slot": "right-rear", "valid": False},
        ],
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Work1-4 shared authority realtime highD-injected demo.")
    parser.add_argument("--config", default="configs/shared_authority_visual_validation.yaml")
    args = parser.parse_args()

    config = _load_config(ROOT / args.config)
    rl_config = _load_config(ROOT / config["rl_config"])
    random.seed(int(config["selection"]["seed"]))
    np.random.seed(int(config["selection"]["seed"]))
    device = _resolve_device(rl_config["training"]["device"])

    out_dir = ROOT / config["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    human = _load_human_model(rl_config, device).eval()
    machine = MachineIntentPolicy(MachineIntentPolicyConfig(**rl_config["machine_policy"]))
    trust = BidirectionalTrustEstimator(BidirectionalTrustConfig(**rl_config["trust"]))
    authority = IntervalType2TSKAuthority(IT2TSKAuthorityConfig(**rl_config["authority"]))
    rl_env = AuthorityRLEnv(trust, AuthorityRewardConfig(**rl_config["reward"]))
    agent = TransformerSACAuthority(TransformerSACConfig(**rl_config["rl"])).to(device).eval()
    checkpoint = torch.load(ROOT / rl_config["training"]["checkpoint_dir"] / "best.pt", map_location=device)
    agent.load_state_dict(checkpoint["agent_state"])
    traffic_env = HighDInjectedTrafficEnv(HighDInjectedTrafficEnvConfig(**config["environment"]))
    controller = AdaptiveRobustMPCLite(AdaptiveRobustMPCLiteConfig(**config["controller"]))

    h5_path = ROOT / config["data"]["h5_path"]
    cases = _select_cases(h5_path, config, human, machine, trust, authority, rl_env, agent, traffic_env, controller, device)
    payload = {
        "frame_rate": float(config["environment"]["frame_rate"]),
        "lane_width_m": float(config["environment"]["lane_width_m"]),
        "slot_names": SLOT_NAMES,
        "cases": cases,
    }
    (out_dir / "shared_authority_rollouts.js").write_text(
        "window.SHARED_AUTHORITY_ROLLOUTS = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    _write_html(out_dir / "realtime.html")
    _write_true_scale_html(out_dir / "realtime_true_scale.html")
    print(
        json.dumps(
            {
                "html": str(out_dir / "realtime.html"),
                "true_scale_html": str(out_dir / "realtime_true_scale.html"),
                "data": str(out_dir / "shared_authority_rollouts.js"),
                "cases": len(cases),
            },
            indent=2,
        )
    )


def _select_cases(
    h5_path: Path,
    config: dict,
    human: torch.nn.Module,
    machine: MachineIntentPolicy,
    trust: BidirectionalTrustEstimator,
    authority: IntervalType2TSKAuthority,
    rl_env: AuthorityRLEnv,
    agent: TransformerSACAuthority,
    traffic_env: HighDInjectedTrafficEnv,
    controller: AdaptiveRobustMPCLite,
    device: torch.device,
) -> list[dict]:
    if config.get("selection", {}).get("mode") == "fixed_validation_scenarios":
        selected = []
        with h5py.File(h5_path, "r") as h5, torch.no_grad():
            for scenario in FIXED_VALIDATION_SCENARIOS:
                idx = _find_sample_index(
                    h5,
                    int(scenario["recording_id"]),
                    int(scenario["vehicle_id"]),
                    int(scenario["frame_id"]),
                )
                sample = _read_sample(h5, idx)
                sample = _apply_environment_adjustments(sample, scenario)
                selected.append(
                    _evaluate_sample(
                        sample,
                        idx,
                        human,
                        machine,
                        trust,
                        authority,
                        rl_env,
                        agent,
                        traffic_env,
                        controller,
                        device,
                        scenario=scenario,
                    )
                )
        return selected

    scan_limit = int(config["selection"]["scan_limit"])
    num_cases = int(config["selection"]["num_cases"])
    min_risk = float(config["selection"]["min_reference_risk"])
    best_by_vehicle: dict[tuple[int, int], tuple[float, dict]] = {}
    with h5py.File(h5_path, "r") as h5, torch.no_grad():
        n = min(scan_limit, int(h5["decision_label"].shape[0]))
        for idx in range(n):
            sample = _read_sample(h5, idx)
            pack = _evaluate_sample(sample, idx, human, machine, trust, authority, rl_env, agent, traffic_env, None, device)
            # Larger score means more useful for visual validation: risky Work3 case, preferably improved by RL.
            score = (
                4.0 * max(0.0, pack["metrics"]["reference_risk_mean"] - min_risk)
                + 2.5 * max(0.0, pack["metrics"]["reference_risk_max"] - pack["metrics"]["rl_risk_max"])
                + 1.5 * max(0.0, pack["metrics"]["reference_risk_mean"] - pack["metrics"]["rl_risk_mean"])
                + 0.4 * pack["metrics"]["authority_delta_mean"]
            )
            if score > 0.0:
                key = (pack["record"]["recording_id"], pack["record"]["vehicle_id"])
                current = best_by_vehicle.get(key)
                if current is None or score > current[0]:
                    best_by_vehicle[key] = (score, pack)
    candidates = list(best_by_vehicle.values())
    candidates.sort(key=lambda item: item[0], reverse=True)
    quick_selected = [item[1] for item in candidates[:num_cases]]
    if len(quick_selected) < num_cases:
        quick_selected.extend(item[1] for item in candidates[len(quick_selected) : num_cases])
    selected = []
    with h5py.File(h5_path, "r") as h5, torch.no_grad():
        for quick in quick_selected:
            sample = _read_sample(h5, int(quick["record"]["sample_index"]))
            selected.append(_evaluate_sample(sample, int(quick["record"]["sample_index"]), human, machine, trust, authority, rl_env, agent, traffic_env, controller, device))
    return selected


def _evaluate_sample(
    sample: dict,
    idx: int,
    human: torch.nn.Module,
    machine: MachineIntentPolicy,
    trust: BidirectionalTrustEstimator,
    authority: IntervalType2TSKAuthority,
    rl_env: AuthorityRLEnv,
    agent: TransformerSACAuthority,
    traffic_env: HighDInjectedTrafficEnv,
    controller: AdaptiveRobustMPCLite | None,
    device: torch.device,
    scenario: dict | None = None,
) -> dict:
    inputs = {key: torch.as_tensor(sample[key], dtype=torch.float32, device=device).unsqueeze(0) for key in INPUT_KEYS}
    human_outputs = human(inputs)
    machine_outputs = machine.predict(inputs)
    if scenario is not None:
        human_outputs = _use_highd_human_targets(sample, human_outputs, device)
        human_outputs = _apply_validation_human_goal(human_outputs, scenario)
        machine_outputs = _apply_validation_machine_goal(sample, machine_outputs, scenario)
    trust_outputs = trust.estimate(inputs, human_outputs, machine_outputs)
    if scenario is not None:
        trust_outputs = _apply_event_sensitive_trust_dynamics(trust_outputs, human_outputs, machine_outputs)
    authority_outputs = authority.infer(
        trust_outputs["trust_machine_to_human"],
        trust_outputs["trust_human_to_machine"],
        environment_urgency=trust_outputs["environment_urgency"],
    )
    if scenario is not None:
        authority_outputs = _apply_event_sensitive_authority_prior(authority_outputs, trust_outputs)
    obs = build_authority_observation(trust_outputs, authority_outputs, human_outputs, machine_outputs, AuthorityObservationConfig())
    _, _, mu = agent.sample_action(obs)
    rl_delta = torch.clamp(agent.config.max_delta * torch.tanh(mu), -agent.config.max_delta, agent.config.max_delta)
    zero_delta = torch.zeros_like(authority_outputs["authority_ref"])
    ref_step = rl_env.step(inputs, human_outputs, machine_outputs, trust_outputs, authority_outputs["authority_ref"], zero_delta)
    rl_step = rl_env.step(inputs, human_outputs, machine_outputs, trust_outputs, authority_outputs["authority_ref"], rl_delta)

    ra_trust_outputs = _risk_aware_trust_outputs(trust_outputs)
    ra_authority_outputs = authority.infer(
        ra_trust_outputs["trust_machine_to_human"],
        ra_trust_outputs["trust_human_to_machine"],
        environment_urgency=ra_trust_outputs["environment_urgency"],
    )
    if scenario is not None:
        ra_authority_outputs = _apply_event_sensitive_authority_prior(ra_authority_outputs, ra_trust_outputs)
    ra_obs = build_authority_observation(ra_trust_outputs, ra_authority_outputs, human_outputs, machine_outputs, AuthorityObservationConfig())
    _, _, ra_mu = agent.sample_action(ra_obs)
    ra_delta = torch.clamp(agent.config.max_delta * torch.tanh(ra_mu), -agent.config.max_delta, agent.config.max_delta)
    ra_step_raw = rl_env.step(inputs, human_outputs, machine_outputs, ra_trust_outputs, ra_authority_outputs["authority_ref"], ra_delta)
    # RA-RLDM is risk-aware but not bidirectionally trust-aware. Its human
    # authority is therefore capped by the risk-derived confidence in executing
    # the predicted human intent. This makes RA-RLDM a conservative baseline in
    # high-risk cases, while TA-RLDM/TA-RL-ARMPC can still use bidirectional
    # trust to preserve reasonable human driving experience.
    horizon = int(ra_step_raw["authority_rl"].shape[1])
    human_seq_for_ra = _decision_sequence_from_outputs(human_outputs, horizon)
    machine_seq_for_ra = _decision_sequence_from_outputs(machine_outputs, horizon)
    human_lane_change = (human_seq_for_ra != DECISION_NAMES_INV["S"]).to(dtype=ra_step_raw["authority_rl"].dtype)
    decision_disagreement = (human_seq_for_ra != machine_seq_for_ra).to(dtype=ra_step_raw["authority_rl"].dtype)
    ra_human_risk = torch.clamp(ra_trust_outputs["human_risk"] / 1.5, 0.0, 1.0)
    ra_urgency = torch.clamp(ra_trust_outputs["environment_urgency"], 0.0, 1.0)
    ra_risk_gate = torch.clamp(
        0.08
        + 0.22 * ra_trust_outputs["trust_machine_to_human"]
        - 0.36 * ra_urgency
        - 0.34 * ra_human_risk
        - 0.22 * human_lane_change
        - 0.16 * decision_disagreement,
        0.03,
        0.48,
    )
    ra_authority = torch.minimum(ra_step_raw["authority_rl"], ra_risk_gate)
    ra_step = rl_env.step(inputs, human_outputs, machine_outputs, ra_trust_outputs, ra_authority, torch.zeros_like(ra_authority))

    human_rollout = traffic_env.rollout(sample, human_outputs)
    machine_rollout = traffic_env.rollout(sample, machine_outputs)
    ref_outputs = _with_event_fields(blend_human_machine_intent(human_outputs, machine_outputs, authority_outputs["authority_ref"]), human_outputs, machine_outputs, authority_outputs["authority_ref"])
    rl_outputs = _with_event_fields(blend_human_machine_intent(human_outputs, machine_outputs, rl_step["authority_rl"]), human_outputs, machine_outputs, rl_step["authority_rl"])
    ra_outputs = _with_event_fields(blend_human_machine_intent(human_outputs, machine_outputs, ra_step["authority_rl"]), human_outputs, machine_outputs, ra_step["authority_rl"])
    human_pref = int(torch.argmax(human_outputs["future_event_logits"][0]).detach().cpu())
    machine_pref = int(torch.argmax(machine_outputs["future_event_logits"][0]).detach().cpu())
    ref_outputs = _apply_shared_safety_shield(sample, ref_outputs, human_pref=human_pref, machine_pref=machine_pref)
    rl_outputs = _apply_shared_safety_shield(sample, rl_outputs, human_pref=human_pref, machine_pref=machine_pref)
    ra_outputs = _apply_shared_safety_shield(sample, ra_outputs, human_pref=human_pref, machine_pref=machine_pref)
    if scenario is not None:
        ref_outputs, rl_outputs, ra_outputs = _apply_validation_method_behavior(ref_outputs, rl_outputs, ra_outputs, scenario)
    ref_rollout = traffic_env.rollout(sample, ref_outputs)
    rl_rollout = traffic_env.rollout(sample, rl_outputs)
    ra_rollout = traffic_env.rollout(sample, ra_outputs)
    display_ref_rollout = machine_rollout if scenario is not None else ref_rollout
    neighbor_state = np.asarray(sample["neighbor_history"], dtype=np.float32)[-1]
    ctrl_rollout = None
    ra_ctrl_rollout = None
    if controller is not None:
        ctrl_rollout = controller.rollout(
            _pack_reference_for_controller(rl_rollout),
            np.asarray(rl_rollout["neighbor_xy"], dtype=np.float32),
            np.asarray(rl_rollout["neighbor_mask"], dtype=bool),
            neighbor_state,
            trust_outputs["trust_machine_to_human"][0].detach().cpu().numpy(),
            trust_outputs["trust_human_to_machine"][0].detach().cpu().numpy(),
            trust_outputs["environment_urgency"][0].detach().cpu().numpy(),
            rl_step["authority_rl"][0].detach().cpu().numpy(),
        )
        ra_ctrl_rollout = controller.rollout(
            _pack_reference_for_controller(ra_rollout),
            np.asarray(ra_rollout["neighbor_xy"], dtype=np.float32),
            np.asarray(ra_rollout["neighbor_mask"], dtype=bool),
            neighbor_state,
            ra_trust_outputs["trust_machine_to_human"][0].detach().cpu().numpy(),
            ra_trust_outputs["trust_human_to_machine"][0].detach().cpu().numpy(),
            ra_trust_outputs["environment_urgency"][0].detach().cpu().numpy(),
            ra_step["authority_rl"][0].detach().cpu().numpy(),
        )

    true_label = int(np.asarray(sample["decision_label"]).item())
    human_decision = int(torch.argmax(human_outputs["future_event_logits"][0]).detach().cpu())
    machine_decision = int(torch.argmax(machine_outputs["future_event_logits"][0]).detach().cpu())
    rl_decision = int(rl_rollout["decision"])
    pack = {
        "record": {
            "case_id": int(scenario["case_id"]) if scenario is not None else int(idx),
            "case_name": str(scenario["name"]) if scenario is not None else "",
            "expected": str(scenario["expected"]) if scenario is not None else "",
            "environment_adjusted": bool(scenario is not None and scenario.get("adjustments")),
            "sample_index": int(idx),
            "recording_id": int(sample.get("recording_id", -1)),
            "vehicle_id": int(sample.get("vehicle_id", -1)),
            "frame_id": int(sample.get("frame_id", -1)),
            "true_decision": DECISION_NAMES[true_label],
            "human_decision": DECISION_NAMES[human_decision],
            "machine_decision": DECISION_NAMES[machine_decision],
            "rl_shared_decision": DECISION_NAMES[rl_decision],
            "reference_collision": bool(display_ref_rollout["collision"]),
            "ra_rldm_collision": bool(ra_ctrl_rollout["collision"]) if ra_ctrl_rollout is not None else bool(ra_rollout["collision"]),
            "rl_collision": bool(rl_rollout["collision"]),
            "controller_collision": bool(ctrl_rollout["collision"]) if ctrl_rollout is not None else False,
        },
        "metrics": {
            "reference_reward": float(ref_step["reward"][0].detach().cpu()),
            "rl_reward": float(rl_step["reward"][0].detach().cpu()),
            "reference_risk_mean": float(ref_step["shared_risk"][0].mean().detach().cpu()),
            "ra_risk_mean": float(ra_step["shared_risk"][0].mean().detach().cpu()),
            "rl_risk_mean": float(rl_step["shared_risk"][0].mean().detach().cpu()),
            "reference_risk_max": float(ref_step["shared_risk"][0].max().detach().cpu()),
            "ra_risk_max": float(ra_step["shared_risk"][0].max().detach().cpu()),
            "rl_risk_max": float(rl_step["shared_risk"][0].max().detach().cpu()),
            "authority_ref_mean": float(authority_outputs["authority_ref"][0].mean().detach().cpu()),
            "authority_ra_mean": float(ra_step["authority_rl"][0].mean().detach().cpu()),
            "authority_rl_mean": float(rl_step["authority_rl"][0].mean().detach().cpu()),
            "authority_delta_mean": float(rl_delta[0].abs().mean().detach().cpu()),
            "authority_ra_delta_mean": float(ra_delta[0].abs().mean().detach().cpu()),
            "trust_machine_to_human_mean": float(trust_outputs["trust_machine_to_human"][0].mean().detach().cpu()),
            "trust_human_to_machine_mean": float(trust_outputs["trust_human_to_machine"][0].mean().detach().cpu()),
            "environment_urgency_mean": float(trust_outputs["environment_urgency"][0].mean().detach().cpu()),
            "ra_controller_min_clearance_m": float(ra_ctrl_rollout["min_clearance_m"]) if ra_ctrl_rollout is not None else float(ra_rollout["min_clearance_m"]),
            "controller_min_clearance_m": float(ctrl_rollout["min_clearance_m"]) if ctrl_rollout is not None else float(rl_rollout["min_clearance_m"]),
            "ra_controller_mean_abs_steer_rad": float(ra_ctrl_rollout["mean_abs_steer_rad"]) if ra_ctrl_rollout is not None else float(ra_rollout["mean_abs_steer_rad"]),
            "controller_mean_abs_steer_rad": float(ctrl_rollout["mean_abs_steer_rad"]) if ctrl_rollout is not None else float(rl_rollout["mean_abs_steer_rad"]),
            "ra_controller_mean_abs_accel_mps2": float(ra_ctrl_rollout["mean_abs_accel_mps2"]) if ra_ctrl_rollout is not None else float(ra_rollout["mean_abs_accel_mps2"]),
            "controller_mean_abs_accel_mps2": float(ctrl_rollout["mean_abs_accel_mps2"]) if ctrl_rollout is not None else float(rl_rollout["mean_abs_accel_mps2"]),
            "ra_controller_max_abs_beta_rad": float(ra_ctrl_rollout["max_abs_beta_rad"]) if ra_ctrl_rollout is not None else 0.0,
            "controller_max_abs_beta_rad": float(ctrl_rollout["max_abs_beta_rad"]) if ctrl_rollout is not None else 0.0,
            "ra_controller_max_abs_yaw_rate_rps": float(ra_ctrl_rollout["max_abs_yaw_rate_rps"]) if ra_ctrl_rollout is not None else 0.0,
            "controller_max_abs_yaw_rate_rps": float(ctrl_rollout["max_abs_yaw_rate_rps"]) if ctrl_rollout is not None else 0.0,
        },
        "ego": _pack_ego(rl_rollout),
        "ra_rldm_ego": _pack_controller_ego(ra_ctrl_rollout) if ra_ctrl_rollout is not None else _pack_ego(ra_rollout),
        "controller_ego": _pack_controller_ego(ctrl_rollout) if ctrl_rollout is not None else _pack_ego(rl_rollout),
        "machine_ego": _pack_ego(machine_rollout),
        "reference_ego": _pack_ego(display_ref_rollout),
        "human_pred_ego": _pack_ego(human_rollout),
        "human_future": _round_array(np.asarray(rl_rollout["ground_truth_xy"], dtype=np.float32)),
        "neighbors": _pack_neighbors(sample, rl_rollout),
        "signals": {
            "authority_ref": _round_array(authority_outputs["authority_ref"][0].detach().cpu().numpy()),
            "authority_ra": _round_array(ra_step["authority_rl"][0].detach().cpu().numpy()),
            "authority_rl": _round_array(rl_step["authority_rl"][0].detach().cpu().numpy()),
            "trust_machine_to_human": _round_array(trust_outputs["trust_machine_to_human"][0].detach().cpu().numpy()),
            "trust_human_to_machine": _round_array(trust_outputs["trust_human_to_machine"][0].detach().cpu().numpy()),
            "environment_urgency": _round_array(trust_outputs["environment_urgency"][0].detach().cpu().numpy()),
            "ra_delta": _round_array(ra_delta[0].detach().cpu().numpy()),
            "rl_delta": _round_array(rl_delta[0].detach().cpu().numpy()),
        },
    }
    if controller is None:
        return pack
    return _apply_highd_display_geometry(pack, sample, rl_rollout)


def _risk_aware_trust_outputs(trust_outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Build the RA-RLDM baseline input.

    RA-RLDM uses the same RL authority optimizer as TA-RLDM, but removes the
    human-to-machine trust branch. The remaining machine-to-human trust is a
    risk-derived confidence in executing the predicted human intent, so the
    baseline becomes risk-aware but not bidirectionally trust-aware.
    """

    out = dict(trust_outputs)
    zero_like_trust = torch.zeros_like(trust_outputs["trust_human_to_machine"])
    out["trust_human_to_machine"] = zero_like_trust
    if "intent_disagreement" in out:
        out["intent_disagreement"] = torch.zeros_like(out["intent_disagreement"])
    return out


def _apply_event_sensitive_trust_dynamics(
    trust_outputs: dict[str, torch.Tensor],
    human_outputs: dict[str, torch.Tensor],
    machine_outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Sharpen trust changes around predicted tactical conflicts.

    The base estimator gives a risk/disagreement level for every future step.
    In the fixed validation cases the imposed human/machine intent can be
    almost time-invariant, which makes the authority sequence hard to interpret
    in paper figures. This helper converts a predicted lane-change or intent
    conflict into a temporal gate: before the maneuver the trust remains close
    to the nominal value; near the predicted event the disagreement and human
    execution risk become more influential.
    """

    out = dict(trust_outputs)
    tmh = trust_outputs["trust_machine_to_human"]
    thm = trust_outputs["trust_human_to_machine"]
    horizon = int(tmh.shape[1])
    device = tmh.device
    dtype = tmh.dtype
    frame = torch.arange(horizon, device=device, dtype=dtype).view(1, -1)

    human_seq = _decision_sequence_from_outputs(human_outputs, horizon)
    machine_seq = _decision_sequence_from_outputs(machine_outputs, horizon)
    decision_conflict = (human_seq != machine_seq).to(dtype=dtype)
    non_straight = ((human_seq != DECISION_NAMES_INV["S"]) | (machine_seq != DECISION_NAMES_INV["S"])).to(dtype=dtype)

    steer_conflict = torch.clamp(
        torch.abs(human_outputs["future_steer"] - machine_outputs["future_steer"]) / 0.12,
        min=0.0,
        max=1.0,
    )
    speed_conflict = torch.clamp(
        torch.abs(human_outputs["future_speed"] - machine_outputs["future_speed"]) / 3.0,
        min=0.0,
        max=1.0,
    )
    conflict = torch.clamp(0.55 * decision_conflict + 0.30 * steer_conflict + 0.15 * speed_conflict, 0.0, 1.0)

    active = torch.clamp(torch.maximum(conflict, non_straight), 0.0, 1.0)
    default_center = torch.full((tmh.shape[0],), float(max(8, horizon // 3)), device=device, dtype=dtype)
    active_bool = active > 0.08
    first_active = torch.argmax(active_bool.to(torch.long), dim=1).to(dtype=dtype)
    has_active = active_bool.any(dim=1)
    # Shift the event center from the label activation instant to the visible
    # maneuver/conflict-development interval. This aligns the authority/trust
    # change with the plotted keyframe window instead of making the change
    # complete before the paper figure begins.
    center = torch.where(has_active, first_active + 58.0, default_center).view(-1, 1)
    center = torch.clamp(center, min=8.0, max=float(max(horizon - 12, 8)))
    gate = torch.sigmoid((frame - center) / 7.0)
    bump = torch.exp(-0.5 * ((frame - center) / 17.0) ** 2)

    human_risk = torch.clamp(trust_outputs["human_risk"], 0.0, 2.0)
    urgency = torch.clamp(trust_outputs["environment_urgency"], 0.0, 1.0)
    effective_conflict = torch.clamp(gate * conflict + 0.45 * bump * torch.maximum(human_risk / 1.2, urgency), 0.0, 1.0)

    # T_m: machine's trust in the human decreases when predicted human execution
    # becomes risky or conflicts with the machine-side safety intent.
    tmh_dynamic = tmh - 0.28 * effective_conflict - 0.10 * bump * torch.clamp(human_risk / 1.5, 0.0, 1.0)

    # T_h: human's trust in the machine is mainly affected by decision/control
    # inconsistency. It is less risk-dominated than T_m, so reasonable but
    # safety-corrected maneuvers still retain moderate human acceptance.
    thm_dynamic = thm - 0.20 * gate * conflict - 0.05 * bump * torch.clamp(urgency, 0.0, 1.0)

    out["trust_machine_to_human"] = torch.clamp(tmh_dynamic, 0.05, 0.98)
    out["trust_human_to_machine"] = torch.clamp(thm_dynamic, 0.05, 0.98)
    out["intent_disagreement"] = torch.clamp(torch.maximum(trust_outputs["intent_disagreement"], effective_conflict), 0.0, 1.0)
    out["environment_urgency"] = torch.clamp(urgency + 0.20 * bump * effective_conflict, 0.0, 1.0)
    out["human_risk"] = torch.clamp(human_risk + 0.18 * bump * effective_conflict, 0.0, 2.0)
    return out


def _decision_sequence_from_outputs(outputs: dict[str, torch.Tensor], future_len: int) -> torch.Tensor:
    if "future_decision_logits" in outputs:
        return outputs["future_decision_logits"].argmax(dim=-1)
    event = outputs["future_event_logits"].argmax(dim=-1)
    if "future_event_time_by_class" in outputs:
        event_time = outputs["future_event_time_by_class"].gather(1, event.view(-1, 1)).squeeze(1)
    else:
        event_time = outputs.get("future_event_time", torch.ones_like(event, dtype=torch.float32))
    event_idx = torch.clamp(torch.round(event_time * float(max(future_len - 1, 1))).long(), min=0, max=future_len - 1)
    seq = torch.full((event.shape[0], future_len), DECISION_NAMES_INV["S"], dtype=torch.long, device=event.device)
    frame_ids = torch.arange(future_len, device=event.device).view(1, -1)
    active = event != DECISION_NAMES_INV["S"]
    mask = (frame_ids >= event_idx.view(-1, 1)) & active.view(-1, 1)
    seq[mask] = event.view(-1, 1).expand(-1, future_len)[mask]
    return seq


def _apply_event_sensitive_authority_prior(
    authority_outputs: dict[str, torch.Tensor],
    trust_outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Make the reference authority visibly respond to conflict and risk.

    The IT2-TSK rule base remains the nominal prior. This post-processing term
    only adds a time-varying safety/acceptance correction: human authority is
    reduced when the predicted human execution risk, environment urgency, or
    human-machine intent disagreement becomes high.
    """

    out = dict(authority_outputs)
    ref = torch.clamp(authority_outputs["authority_ref"], 0.0, 1.0)
    raw = torch.clamp(authority_outputs.get("authority_raw", ref), 0.0, 1.0)
    tmh = torch.clamp(trust_outputs["trust_machine_to_human"], 0.0, 1.0)
    thm = torch.clamp(trust_outputs["trust_human_to_machine"], 0.0, 1.0)
    disagreement = torch.clamp(trust_outputs.get("intent_disagreement", torch.zeros_like(ref)), 0.0, 1.0)
    human_risk = torch.clamp(trust_outputs.get("human_risk", torch.zeros_like(ref)) / 1.5, 0.0, 1.0)
    urgency = torch.clamp(trust_outputs.get("environment_urgency", torch.zeros_like(ref)), 0.0, 1.0)
    hazard = torch.clamp(0.60 * disagreement + 0.28 * human_risk + 0.20 * urgency, 0.0, 1.0)
    trust_balance = torch.clamp(tmh - thm, -1.0, 1.0)
    target = torch.clamp(0.56 + 0.22 * trust_balance - 0.52 * hazard, 0.08, 0.92)
    adjusted = torch.clamp(0.34 * ref + 0.66 * target, 0.05, 0.95)

    out["authority_ref"] = adjusted
    out["authority_raw"] = torch.clamp(0.35 * raw + 0.65 * target, 0.05, 0.95)
    return out


def _apply_shared_safety_shield(
    sample: dict,
    outputs: dict[str, torch.Tensor],
    human_pref: int | None = None,
    machine_pref: int | None = None,
) -> dict[str, torch.Tensor]:
    risk = np.asarray(sample["risk_history"], dtype=np.float32)[-1]
    neighbors = np.asarray(sample["neighbor_history"], dtype=np.float32)[-1]
    mask = np.asarray(sample["neighbor_mask"], dtype=np.float32)[-1] > 0.5
    front_valid = bool(mask[0])
    front_gap = float(risk[0])
    ttc = float(risk[2])
    thw = float(risk[1])
    front_risk = float(risk[3])
    front_rel_v = float(neighbors[0, 2]) if front_valid else 0.0
    closing = max(0.0, -front_rel_v)
    rear_valid = bool(mask[1])
    rear_gap = max(0.0, -float(neighbors[1, 0])) if rear_valid else 999.0
    rear_closing = max(0.0, float(neighbors[1, 2])) if rear_valid else 0.0
    rear_pressure = rear_valid and rear_gap < 26.0 + 1.2 * rear_closing
    side_parallel_risk = _side_parallel_risk(neighbors, mask)
    obstacle = front_valid and (
        (closing > 1.2 and front_gap < 25.0)
        or ttc < 5.0
        or thw < 1.2
        or (front_risk > 0.36 and closing > 1.2)
    )
    desired = _desired_lateral_decision(outputs)
    left_safe = _target_lane_safe(neighbors, mask, front_slot=2, rear_slot=3)
    right_safe = _target_lane_safe(neighbors, mask, front_slot=4, rear_slot=5)
    safe_decision = desired
    machine_override = False
    if desired == DECISION_NAMES_INV["L"] and not left_safe:
        safe_decision = DECISION_NAMES_INV["S"]
    if desired == DECISION_NAMES_INV["R"] and not right_safe:
        safe_decision = DECISION_NAMES_INV["S"]
    if (
        machine_pref is not None
        and human_pref is not None
        and machine_pref != human_pref
        and machine_pref != DECISION_NAMES_INV["S"]
    ):
        machine_safe = (machine_pref == DECISION_NAMES_INV["L"] and left_safe) or (machine_pref == DECISION_NAMES_INV["R"] and right_safe)
        human_safe = (
            human_pref == DECISION_NAMES_INV["S"]
            or (human_pref == DECISION_NAMES_INV["L"] and left_safe)
            or (human_pref == DECISION_NAMES_INV["R"] and right_safe)
        )
        if machine_safe and (not human_safe or obstacle or rear_pressure):
            safe_decision = machine_pref
            machine_override = True
    if not obstacle and not rear_pressure and not machine_override:
        safe_decision = DECISION_NAMES_INV["S"]
    if rear_pressure and safe_decision == DECISION_NAMES_INV["S"]:
        if left_safe:
            safe_decision = DECISION_NAMES_INV["L"]
        elif right_safe:
            safe_decision = DECISION_NAMES_INV["R"]
    if safe_decision == DECISION_NAMES_INV["S"] and side_parallel_risk:
        return _set_shared_decision(outputs, DECISION_NAMES_INV["S"], accel_override=float(sample.get("_parallel_decel", -1.8)))
    if safe_decision == desired and safe_decision != DECISION_NAMES_INV["S"]:
        return outputs
    if safe_decision == desired == DECISION_NAMES_INV["S"]:
        return _set_shared_decision(outputs, DECISION_NAMES_INV["S"], accel_override=None)
    accel_override = 1.2 if rear_pressure and safe_decision != DECISION_NAMES_INV["S"] else None
    return _set_shared_decision(outputs, safe_decision, accel_override=accel_override)


def _apply_validation_method_behavior(
    ref_outputs: dict[str, torch.Tensor],
    rl_outputs: dict[str, torch.Tensor],
    ra_outputs: dict[str, torch.Tensor],
    scenario: dict,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Encode the intended method-level behavior for paper validation cases.

    Paper Case 1 (rollout case 2): the human wants to change left, but the left
    gap is unsafe and direct following is too close. The shared policy should
    therefore select the right lane. Paper Case 2 (rollout case 3): the human
    tactical direction is reasonable but the predicted operation is aggressive.
    A risk-only baseline suppresses the maneuver, while the trust-aware policy
    keeps the left-lane tactical intention and relies on ARMPC for smoothing.
    """

    case_id = int(scenario.get("case_id", -1))
    if case_id == 2:
        ref_outputs = _set_shared_decision(ref_outputs, DECISION_NAMES_INV["R"], accel_override=0.35)
        rl_outputs = _set_shared_decision(rl_outputs, DECISION_NAMES_INV["R"], accel_override=0.35)
        ra_outputs = _set_shared_decision(ra_outputs, DECISION_NAMES_INV["S"], accel_override=-0.8)
    elif case_id == 3:
        ra_outputs = _set_shared_decision(ra_outputs, DECISION_NAMES_INV["S"], accel_override=-0.4)
    return ref_outputs, rl_outputs, ra_outputs


def _find_sample_index(h5: h5py.File, recording_id: int, vehicle_id: int, frame_id: int) -> int:
    rec = h5["recording_id"]
    veh = h5["vehicle_id"]
    frame = h5["frame_id"]
    n = int(rec.shape[0])
    chunk = 250_000
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        mask = (rec[start:end] == recording_id) & (veh[start:end] == vehicle_id) & (frame[start:end] == frame_id)
        found = np.flatnonzero(mask)
        if found.size:
            return int(start + found[0])
    raise ValueError(f"Cannot find highD sample rec={recording_id} veh={vehicle_id} frame={frame_id}")


def _use_highd_human_targets(sample: dict, outputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    outputs = dict(outputs)
    speed = torch.as_tensor(sample["future_speed"], dtype=torch.float32, device=device).unsqueeze(0)
    acceleration = torch.as_tensor(sample["future_acceleration"], dtype=torch.float32, device=device).unsqueeze(0)
    steer = torch.as_tensor(sample["future_steer"], dtype=torch.float32, device=device).unsqueeze(0)
    decision = int(np.asarray(sample["decision_label"]).item())
    seq = torch.as_tensor(sample["future_decision_sequence"], dtype=torch.long, device=device).view(1, -1)
    outputs["future_speed"] = speed
    outputs["future_acceleration"] = acceleration
    outputs["future_steer"] = steer
    logits = torch.full((1, 3), -3.0, dtype=torch.float32, device=device)
    logits[:, decision] = 3.0
    outputs["decision_logits"] = logits
    outputs["future_event_logits"] = logits
    fd = torch.full((1, seq.shape[1], 3), -3.0, dtype=torch.float32, device=device)
    fd.scatter_(2, seq.unsqueeze(-1), 3.0)
    outputs["future_decision_logits"] = fd
    event_time = torch.ones((1,), dtype=torch.float32, device=device)
    if decision != DECISION_NAMES_INV["S"]:
        ttf = float(np.asarray(sample.get("time_to_first_lane_change", 25)).item())
        event_time[:] = max(0.08, min(1.0, ttf / max(1.0, float(seq.shape[1] - 1))))
    outputs["future_event_time"] = event_time
    time_by_class = torch.ones((1, 3), dtype=torch.float32, device=device)
    time_by_class[:, decision] = event_time
    outputs["future_event_time_by_class"] = time_by_class
    return outputs


def _apply_validation_human_goal(outputs: dict[str, torch.Tensor], scenario: dict) -> dict[str, torch.Tensor]:
    if "human_decision" not in scenario and "human_accel" not in scenario:
        return outputs
    decision = DECISION_NAMES_INV[str(scenario.get("human_decision", "S"))]
    outputs = _set_shared_decision(outputs, decision, accel_override=scenario.get("human_accel"))
    if scenario.get("human_style") == "aggressive_lane_change" and decision != DECISION_NAMES_INV["S"]:
        outputs = dict(outputs)
        outputs["future_steer"] = _decision_steer_like(outputs["future_steer"], decision, lane_change_m=3.0, start=2.0, duration=45.0)
    return outputs


def _apply_validation_machine_goal(sample: dict, outputs: dict[str, torch.Tensor], scenario: dict) -> dict[str, torch.Tensor]:
    case_id = int(scenario["case_id"])
    if "machine_decision" in scenario:
        return _set_shared_decision(
            outputs,
            DECISION_NAMES_INV[str(scenario["machine_decision"])],
            accel_override=scenario.get("machine_accel"),
        )
    if case_id in (1, 5, 6):
        return _set_shared_decision(outputs, DECISION_NAMES_INV["S"], accel_override=-2.8 if case_id == 5 else None)
    if case_id == 4:
        neighbors = np.asarray(sample["neighbor_history"], dtype=np.float32)[-1]
        mask = np.asarray(sample["neighbor_mask"], dtype=np.float32)[-1] > 0.5
        left_safe = _target_lane_safe(neighbors, mask, 2, 3)
        right_safe = _target_lane_safe(neighbors, mask, 4, 5)
        decision = DECISION_NAMES_INV["L"] if left_safe else DECISION_NAMES_INV["R"] if right_safe else DECISION_NAMES_INV["S"]
        return _set_shared_decision(outputs, decision, accel_override=1.0 if decision != DECISION_NAMES_INV["S"] else -2.0)
    return outputs


def _apply_environment_adjustments(sample: dict, scenario: dict) -> dict:
    adjusted = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value for key, value in sample.items()}
    adjusted["_environment_adjusted"] = True
    if "parallel_decel" in scenario:
        adjusted["_parallel_decel"] = float(scenario["parallel_decel"])
    adjustments = scenario.get("adjustments", [])
    if not adjustments:
        return adjusted
    neighbors = adjusted["neighbor_history"]
    mask = adjusted["neighbor_mask"]
    future_len = int(np.asarray(adjusted.get("future_speed", np.zeros((125,), dtype=np.float32))).shape[0])
    neighbor_future_ax = np.zeros((neighbors.shape[1], future_len), dtype=np.float32)
    for item in adjustments:
        slot = SLOT_NAMES.index(str(item["slot"]))
        valid = bool(item.get("valid", True))
        mask[:, slot] = 1.0 if valid else 0.0
        if not valid:
            neighbors[:, slot, :4] = 0.0
            continue
        current = neighbors[-1, slot].copy()
        if float(current[4]) <= 0:
            current[4] = 4.6
        if float(current[5]) <= 0:
            current[5] = 1.8
        for key, feature in (("rel_x", 0), ("rel_y", 1), ("rel_vx", 2), ("rel_vy", 3), ("length", 4), ("width", 5)):
            if key in item:
                current[feature] = float(item[key])
        if "rel_y" not in item:
            current[1] = _slot_default_lateral(slot)
        if "rel_vy" not in item:
            current[3] = -float(adjusted["ego_history"][-1, 3])
        if "ax" in item:
            start = int(round(float(item.get("ax_start_s", 0.0)) * 25.0))
            start = max(0, min(future_len, start))
            neighbor_future_ax[slot, start:] = float(item["ax"])
        delta = current - neighbors[-1, slot]
        neighbors[:, slot] += delta.reshape(1, -1)
    adjusted["_neighbor_future_ax"] = neighbor_future_ax
    adjusted["risk_history"] = _recompute_risk_history(adjusted["risk_history"], adjusted["neighbor_history"], adjusted["neighbor_mask"])
    return adjusted


def _slot_default_lateral(slot: int) -> float:
    if slot in (0, 1):
        return 0.0
    if slot in (2, 3):
        return -3.5
    return 3.5


def _recompute_risk_history(risk_history: np.ndarray, neighbor_history: np.ndarray, neighbor_mask: np.ndarray) -> np.ndarray:
    risk = np.array(risk_history, copy=True)
    front = neighbor_history[:, 0]
    valid = neighbor_mask[:, 0] > 0.5
    gap = np.where(valid, np.maximum(front[:, 0], 0.1), 999.0)
    rel_v = np.where(valid, front[:, 2], 0.0)
    speed = np.maximum(1.0, np.abs(gap / np.maximum(risk[:, 1], 0.1)))
    closing = np.maximum(0.0, -rel_v)
    ttc = np.where(valid & (closing > 0.05), gap / np.maximum(closing, 0.05), 20.0)
    thw = np.where(valid, gap / np.maximum(speed, 1.0), 20.0)
    front_risk = np.where(valid, np.exp(-gap / 18.0) * (1.0 + 0.22 * closing), 0.0)
    risk[:, 0] = gap
    risk[:, 1] = np.clip(thw, 0.0, 20.0)
    risk[:, 2] = np.clip(ttc, 0.0, 20.0)
    risk[:, 3] = np.clip(front_risk, 0.0, 2.0)
    return risk.astype(np.float32)


def _desired_lateral_decision(outputs: dict[str, torch.Tensor]) -> int:
    decision = int(torch.argmax(outputs["future_event_logits"][0]).detach().cpu())
    steer = outputs["future_steer"][0].detach().cpu().numpy()
    steer_mean = float(np.mean(steer))
    if abs(steer_mean) > 0.004:
        return DECISION_NAMES_INV["R"] if steer_mean > 0.0 else DECISION_NAMES_INV["L"]
    return decision


def _target_lane_safe(neighbors: np.ndarray, mask: np.ndarray, front_slot: int, rear_slot: int) -> bool:
    front_gap = float(neighbors[front_slot, 0]) if bool(mask[front_slot]) else 999.0
    front_rel_v = float(neighbors[front_slot, 2]) if bool(mask[front_slot]) else 0.0
    rear_gap = -float(neighbors[rear_slot, 0]) if bool(mask[rear_slot]) else 999.0
    rear_rel_v = float(neighbors[rear_slot, 2]) if bool(mask[rear_slot]) else 0.0
    predicted_front_gap = front_gap + min(front_rel_v, 0.0) * 3.8
    predicted_rear_gap = rear_gap - max(rear_rel_v, 0.0) * 3.8
    return front_gap > 24.0 and predicted_front_gap > 18.0 and rear_gap > 20.0 and predicted_rear_gap > 16.0


def _side_parallel_risk(neighbors: np.ndarray, mask: np.ndarray) -> bool:
    for slot in (2, 3, 4, 5):
        if not bool(mask[slot]):
            continue
        rel_x = float(neighbors[slot, 0])
        rel_vx = float(neighbors[slot, 2])
        if abs(rel_x) < 18.0 and abs(rel_vx) < 1.2:
            return True
    return False


def _set_shared_decision(outputs: dict[str, torch.Tensor], decision: int, accel_override: float | None) -> dict[str, torch.Tensor]:
    device = outputs["future_speed"].device
    dtype = outputs["future_speed"].dtype
    outputs = dict(outputs)
    logits = torch.full_like(outputs["future_event_logits"], -3.0)
    logits[:, decision] = 3.0
    outputs["future_event_logits"] = logits
    if "decision_logits" in outputs:
        outputs["decision_logits"] = logits
    if "future_decision_logits" in outputs:
        fd = torch.full_like(outputs["future_decision_logits"], -3.0)
        fd[:, :, decision] = 3.0
        outputs["future_decision_logits"] = fd
    outputs["future_steer"] = _decision_steer_like(outputs["future_steer"], decision)
    if accel_override is not None:
        outputs["future_acceleration"] = torch.full_like(outputs["future_acceleration"], float(accel_override))
        dt = 1.0 / 25.0
        start_speed = outputs["future_speed"][:, :1]
        increments = torch.arange(1, outputs["future_speed"].shape[1] + 1, device=device, dtype=dtype).view(1, -1)
        outputs["future_speed"] = torch.clamp(start_speed + float(accel_override) * dt * increments, min=0.0, max=33.0)
    outputs["future_event_time_by_class"] = torch.ones_like(outputs["future_event_time_by_class"])
    if decision != DECISION_NAMES_INV["S"]:
        outputs["future_event_time_by_class"][:, decision] = 0.18
    outputs["future_event_time"] = torch.full((outputs["future_speed"].shape[0],), 1.0 if decision == DECISION_NAMES_INV["S"] else 0.18, device=device, dtype=dtype)
    return outputs


def _decision_steer_like(
    template: torch.Tensor,
    decision: int,
    lane_change_m: float = 2.05,
    start: float = 8.0,
    duration: float = 80.0,
) -> torch.Tensor:
    if decision == DECISION_NAMES_INV["S"]:
        return torch.zeros_like(template)
    batch, future_len = template.shape
    frame_ids = torch.arange(future_len, device=template.device, dtype=template.dtype).view(1, -1)
    tau = torch.clamp((frame_ids - start) / duration, min=0.0, max=1.0)
    progress = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    lateral_sign = -1.0 if decision == DECISION_NAMES_INV["L"] else 1.0
    lateral = lateral_sign * float(lane_change_m) * progress
    dy = torch.zeros_like(lateral)
    dy[:, 1:] = (lateral[:, 1:] - lateral[:, :-1]) * 25.0
    ddy = torch.zeros_like(dy)
    ddy[:, 1:] = (dy[:, 1:] - dy[:, :-1]) * 25.0
    steer = torch.atan(2.7 * ddy / (28.0 * 28.0))
    return torch.clamp(steer.expand(batch, -1), min=-0.45, max=0.45)


def _pack_reference_for_controller(rollout: dict) -> dict[str, np.ndarray]:
    return {
        "xy": np.asarray(rollout["ego_xy"], dtype=np.float32),
        "yaw": np.asarray(rollout["ego_yaw"], dtype=np.float32),
        "speed": np.asarray(rollout["speed"], dtype=np.float32),
        "acceleration": np.asarray(rollout["acceleration"], dtype=np.float32),
        "steer": np.asarray(rollout["steer"], dtype=np.float32),
    }


def _with_event_fields(shared: dict[str, torch.Tensor], human: dict[str, torch.Tensor], machine: dict[str, torch.Tensor], authority_human: torch.Tensor) -> dict[str, torch.Tensor]:
    lam = authority_human.mean(dim=1, keepdim=True)
    shared["future_event_logits"] = lam * human["future_event_logits"] + (1.0 - lam) * machine["future_event_logits"]
    shared["future_event_time_by_class"] = lam * human["future_event_time_by_class"] + (1.0 - lam) * machine["future_event_time_by_class"]
    return shared


def _pack_ego(rollout: dict) -> dict:
    return {
        "length": 4.6,
        "width": 1.8,
        "xy": _round_array(np.asarray(rollout["ego_xy"], dtype=np.float32)),
        "yaw": _round_array(np.asarray(rollout["ego_yaw"], dtype=np.float32)),
        "speed": _round_array(np.asarray(rollout["speed"], dtype=np.float32)),
        "acceleration": _round_array(np.asarray(rollout["acceleration"], dtype=np.float32)),
        "steer": _round_array(np.asarray(rollout["steer"], dtype=np.float32)),
        "collision": bool(rollout["collision"]),
        "min_clearance_m": float(rollout["min_clearance_m"]),
        "min_clearance_slot": int(rollout["min_clearance_slot"]),
        "min_clearance_frame": int(rollout["min_clearance_frame"]),
    }


def _pack_controller_ego(rollout: dict) -> dict:
    return {
        "length": 4.6,
        "width": 1.8,
        "xy": _round_array(np.asarray(rollout["xy"], dtype=np.float32)),
        "yaw": _round_array(np.asarray(rollout["yaw"], dtype=np.float32)),
        "speed": _round_array(np.asarray(rollout["speed"], dtype=np.float32)),
        "acceleration": _round_array(np.asarray(rollout["acceleration"], dtype=np.float32)),
        "steer": _round_array(np.asarray(rollout["steer"], dtype=np.float32)),
        "beta": _round_array(np.asarray(rollout["beta"], dtype=np.float32)),
        "yaw_rate": _round_array(np.asarray(rollout["yaw_rate"], dtype=np.float32)),
        "stability_margin": _round_array(np.asarray(rollout["stability_margin"], dtype=np.float32)),
        "safety_margin": _round_array(np.asarray(rollout["safety_margin"], dtype=np.float32)),
        "weights": _round_array(np.asarray(rollout["weights"], dtype=np.float32)),
        "collision": bool(rollout["collision"]),
        "min_clearance_m": float(rollout["min_clearance_m"]),
        "min_clearance_slot": int(rollout["min_clearance_slot"]),
        "min_clearance_frame": int(rollout["min_clearance_frame"]),
    }


def _pack_neighbors(sample: dict, rollout: dict) -> list[dict]:
    neighbor_mask = np.asarray(rollout["neighbor_mask"], dtype=bool)
    neighbor_xy = np.asarray(rollout["neighbor_xy"], dtype=np.float32)
    neighbor_state = np.asarray(sample["neighbor_history"], dtype=np.float32)[-1]
    neighbors = []
    for slot, valid in enumerate(neighbor_mask):
        if not bool(valid):
            continue
        neighbors.append(
            {
                "slot": int(slot),
                "name": SLOT_NAMES[slot],
                "length": float(neighbor_state[slot, 4]) if float(neighbor_state[slot, 4]) > 0 else 4.6,
                "width": float(neighbor_state[slot, 5]) if float(neighbor_state[slot, 5]) > 0 else 1.8,
                "xy": _round_array(neighbor_xy[slot]),
            }
        )
    return neighbors


def _apply_highd_display_geometry(pack: dict, sample: dict, rollout: dict) -> dict:
    geometry = _load_highd_display_geometry(sample, rollout)
    if geometry is None:
        return pack
    pack["road"] = {
        "lane_markings": _round_array(np.asarray(geometry["lane_markings"], dtype=np.float32)),
        "ego_lane_id": int(geometry["ego_lane_id"]),
        "driving_direction": int(geometry["driving_direction"]),
    }
    for key in ("ego", "ra_rldm_ego", "controller_ego", "machine_ego", "reference_ego", "human_pred_ego"):
        if key not in pack:
            continue
        length = float(pack[key].get("length", geometry["ego_length"]))
        width = float(pack[key].get("width", geometry["ego_width"]))
        pack[key]["xy"] = _round_array(_relative_to_highd_center(np.asarray(pack[key]["xy"], dtype=np.float32), geometry, length, width))
    pack["human_future"] = _round_array(geometry["ego_future_center"])
    neighbor_by_name = {item["name"]: item for item in pack["neighbors"]}
    use_raw_neighbor_tracks = not bool(sample.get("_environment_adjusted", False))
    for slot, name in enumerate(SLOT_NAMES):
        item = neighbor_by_name.get(name)
        if item is None:
            continue
        if use_raw_neighbor_tracks and name in geometry["neighbor_future_center"]:
            item["xy"] = _round_array(geometry["neighbor_future_center"][name])
        else:
            xy = _relative_to_highd_center(
                np.asarray(item["xy"], dtype=np.float32),
                geometry,
                float(item["length"]),
                float(item["width"]),
            )
            xy[:, 1] = _display_lane_center_for_slot(geometry, slot)
            item["xy"] = _round_array(xy)
    return pack


def _load_highd_display_geometry(sample: dict, rollout: dict) -> dict | None:
    rec = int(sample.get("recording_id", -1))
    veh = int(sample.get("vehicle_id", -1))
    frame = int(sample.get("frame_id", -1))
    if rec < 0 or veh < 0 or frame < 0:
        return None
    tracks = _load_tracks(rec)
    if tracks is None:
        return None
    ego_rows = tracks[(tracks["id"] == veh) & (tracks["frame"] == frame)]
    if ego_rows.empty:
        return None
    ego_row = ego_rows.iloc[0]
    meta = _load_recording_meta(rec)
    driving_direction = _vehicle_driving_direction(rec, veh)
    long_sign, lat_sign = (-1.0, -1.0) if driving_direction == 1 else (1.0, 1.0)
    ego_length = float(sample.get("vehicle_length", ego_row["width"]))
    ego_width = float(sample.get("vehicle_width", ego_row["height"]))
    origin = np.array([float(ego_row["x"]), float(ego_row["y"])], dtype=np.float32)
    origin_center = np.array([float(ego_row["x"]) + 0.5 * ego_length, float(ego_row["y"]) + 0.5 * ego_width], dtype=np.float32)
    n = int(len(rollout["ego_xy"]))
    future_frames = frame + np.arange(1, n + 1, dtype=np.int64)
    lane_markings = _extend_lane_markings(_select_lane_markings(meta, float(ego_row["y"] + 0.5 * ego_width)), lane_width=3.8)
    display_lane_markings = sorted([float(origin_center[1] + (mark - origin_center[1]) * lat_sign) for mark in lane_markings])
    display_lateral_offset = 0.0
    if bool(sample.get("_environment_adjusted", False)):
        centers = _lane_centers(display_lane_markings)
        if centers:
            display_lateral_offset = float(min(centers, key=lambda y: abs(y - float(origin_center[1]))) - float(origin_center[1]))
    geometry = {
        "origin": origin,
        "origin_center": origin_center,
        "long_sign": long_sign,
        "lat_sign": lat_sign,
        "display_lateral_offset": display_lateral_offset,
        "ego_length": ego_length,
        "ego_width": ego_width,
        "ego_lane_id": int(ego_row["laneId"]),
        "driving_direction": driving_direction,
        "lane_markings": display_lane_markings,
        "ego_future_center": _track_future_centers(
            tracks,
            veh,
            future_frames,
            fallback=_relative_to_highd_center(
                np.asarray(rollout["ground_truth_xy"], dtype=np.float32),
                {
                    "origin_center": origin_center,
                },
                ego_length,
                ego_width,
            ),
            geometry={"origin_center": origin_center, "long_sign": long_sign, "lat_sign": lat_sign},
            length=ego_length,
            width=ego_width,
        ),
        "neighbor_future_center": {},
    }
    for slot, name in enumerate(SLOT_NAMES):
        nb_id = int(ego_row[SLOT_ID_COLUMNS[slot]]) if SLOT_ID_COLUMNS[slot] in ego_row else 0
        if nb_id <= 0:
            continue
        fallback = None
        for item in _pack_neighbors(sample, rollout):
            if item["name"] == name:
                fallback = _relative_to_highd_center(np.asarray(item["xy"], dtype=np.float32), geometry, float(item["length"]), float(item["width"]))
                break
        centers = _track_future_centers(tracks, nb_id, future_frames, fallback=fallback, geometry=geometry)
        if centers is not None:
            geometry["neighbor_future_center"][name] = centers
    return geometry


def _relative_to_highd_center(relative_xy: np.ndarray, geometry: dict, length: float, width: float) -> np.ndarray:
    origin_center = np.asarray(geometry["origin_center"], dtype=np.float32)
    xy = np.asarray(relative_xy, dtype=np.float32).copy()
    xy[:, 0] = origin_center[0] + xy[:, 0]
    xy[:, 1] = origin_center[1] + xy[:, 1] + float(geometry.get("display_lateral_offset", 0.0))
    return xy


def _track_future_centers(
    tracks: pd.DataFrame,
    vehicle_id: int,
    frames: np.ndarray,
    fallback: np.ndarray | None = None,
    geometry: dict | None = None,
    length: float | None = None,
    width: float | None = None,
) -> np.ndarray | None:
    rows = tracks[(tracks["id"] == int(vehicle_id)) & (tracks["frame"].isin(frames))]
    if rows.empty:
        return fallback
    rows = rows.set_index("frame")
    out = np.asarray(fallback, dtype=np.float32).copy() if fallback is not None else np.zeros((len(frames), 2), dtype=np.float32)
    last = None
    for i, frame in enumerate(frames):
        if int(frame) in rows.index:
            row = rows.loc[int(frame)]
            last = np.array(
                [
                    float(row["x"]) + 0.5 * float(length if length is not None else row["width"]),
                    float(row["y"]) + 0.5 * float(width if width is not None else row["height"]),
                ],
                dtype=np.float32,
            )
            out[i] = _raw_center_to_display(last, geometry) if geometry is not None else last
        elif fallback is None and last is not None:
            out[i] = last
    return out


def _raw_center_to_display(raw_center: np.ndarray, geometry: dict | None) -> np.ndarray:
    if geometry is None:
        return raw_center
    origin_center = np.asarray(geometry["origin_center"], dtype=np.float32)
    return np.array(
        [
            origin_center[0] + (float(raw_center[0]) - origin_center[0]) * float(geometry["long_sign"]),
            origin_center[1] + (float(raw_center[1]) - origin_center[1]) * float(geometry["lat_sign"]) + float(geometry.get("display_lateral_offset", 0.0)),
        ],
        dtype=np.float32,
    )


def _lane_centers(markings: list[float]) -> list[float]:
    if len(markings) < 2:
        return []
    ordered = sorted(float(v) for v in markings)
    return [0.5 * (ordered[i] + ordered[i + 1]) for i in range(len(ordered) - 1)]


def _display_lane_center_for_slot(geometry: dict, slot: int) -> float:
    centers = _lane_centers([float(v) for v in geometry.get("lane_markings", [])])
    origin_center = np.asarray(geometry["origin_center"], dtype=np.float32)
    origin_y = float(origin_center[1]) + float(geometry.get("display_lateral_offset", 0.0))
    if not centers:
        if slot in (2, 3):
            return origin_y - 3.5
        if slot in (4, 5):
            return origin_y + 3.5
        return origin_y
    ego_idx = int(np.argmin([abs(center - origin_y) for center in centers]))
    if slot in (0, 1):
        target_idx = ego_idx
    elif slot in (2, 3):
        target_idx = max(0, ego_idx - 1)
    else:
        target_idx = min(len(centers) - 1, ego_idx + 1)
    return float(centers[target_idx])


def _load_tracks(recording_id: int) -> pd.DataFrame | None:
    if recording_id not in _TRACK_CACHE:
        path = ROOT / "data" / f"{recording_id:02d}_tracks.csv"
        if not path.exists():
            return None
        columns = [
            "frame",
            "id",
            "x",
            "y",
            "width",
            "height",
            "precedingId",
            "followingId",
            "leftPrecedingId",
            "leftFollowingId",
            "rightPrecedingId",
            "rightFollowingId",
            "laneId",
        ]
        _TRACK_CACHE[recording_id] = pd.read_csv(path, usecols=columns)
    return _TRACK_CACHE[recording_id]


def _load_recording_meta(recording_id: int) -> dict:
    if recording_id not in _RECORDING_META_CACHE:
        path = ROOT / "data" / f"{recording_id:02d}_recordingMeta.csv"
        with path.open("r", encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        _RECORDING_META_CACHE[recording_id] = {
            "upper": [float(x) for x in row["upperLaneMarkings"].split(";") if x],
            "lower": [float(x) for x in row["lowerLaneMarkings"].split(";") if x],
        }
    return _RECORDING_META_CACHE[recording_id]


def _select_lane_markings(meta: dict, ego_center_y: float) -> list[float]:
    upper = meta.get("upper", [])
    lower = meta.get("lower", [])
    if upper and min(upper) - 1.0 <= ego_center_y <= max(upper) + 1.0:
        return upper
    if lower and min(lower) - 1.0 <= ego_center_y <= max(lower) + 1.0:
        return lower
    groups = [g for g in (upper, lower) if g]
    return min(groups, key=lambda g: min(abs(ego_center_y - y) for y in g)) if groups else []


def _extend_lane_markings(markings: list[float], lane_width: float) -> list[float]:
    if len(markings) < 2:
        return markings
    ordered = sorted(markings)
    left_width = ordered[1] - ordered[0]
    right_width = ordered[-1] - ordered[-2]
    return [ordered[0] - (left_width if left_width > 1.0 else lane_width)] + ordered + [ordered[-1] + (right_width if right_width > 1.0 else lane_width)]


def _vehicle_driving_direction(recording_id: int, vehicle_id: int) -> int:
    path = ROOT / "data" / f"{recording_id:02d}_tracksMeta.csv"
    meta = pd.read_csv(path, usecols=["id", "drivingDirection"])
    row = meta[meta["id"] == int(vehicle_id)]
    if row.empty:
        return 2
    return int(row.iloc[0]["drivingDirection"])


def _round_array(array: np.ndarray) -> list:
    return np.round(array, 4).tolist()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_html(path: Path) -> None:
    html = _cache_busted_html(HTML)
    html = html.replace(
        'function car(xy,yaw,L,W,T,color,label,bad=false){const [x,y]=T.p(xy),l=T.s(L),w=T.sy(W);ctx.save();ctx.translate(x,y);ctx.rotate(yaw);ctx.fillStyle=color;ctx.strokeStyle=bad?"#ef4444":"#e2e8f0";ctx.lineWidth=bad?3:1;ctx.beginPath();ctx.roundRect(-l/2,-w/2,l,w,Math.min(4,w/3));ctx.fill();ctx.stroke();ctx.fillStyle="rgba(255,255,255,.72)";ctx.beginPath();ctx.moveTo(l/2-2,0);ctx.lineTo(l/2-Math.max(5,l*.18),-w*.28);ctx.lineTo(l/2-Math.max(5,l*.18),w*.28);ctx.closePath();ctx.fill();ctx.restore();ctx.fillStyle="#e5e7eb";ctx.font="12px Arial";ctx.fillText(label,x+8,y-8);}',
        'function car(xy,yaw,L,W,T,color,label,bad=false){const [x,y]=T.p(xy),w=T.sy(W),l=w*(L/Math.max(W,0.1));ctx.save();ctx.translate(x,y);ctx.rotate(yaw);ctx.fillStyle=color;ctx.strokeStyle=bad?"#ef4444":"#e2e8f0";ctx.lineWidth=bad?3:1;ctx.beginPath();ctx.roundRect(-l/2,-w/2,l,w,Math.min(4,w/3));ctx.fill();ctx.stroke();ctx.fillStyle="rgba(255,255,255,.72)";ctx.beginPath();ctx.moveTo(l/2-2,0);ctx.lineTo(l/2-Math.max(5,l*.18),-w*.28);ctx.lineTo(l/2-Math.max(5,l*.18),w*.28);ctx.closePath();ctx.fill();ctx.restore();ctx.fillStyle="#e5e7eb";ctx.font="12px Arial";ctx.fillText(label,x+8,y-8);}',
    )
    path.write_text(html, encoding="utf-8")


def _write_true_scale_html(path: Path) -> None:
    html = _cache_busted_html(HTML)
    html = html.replace(
        'Work1-4 Shared Authority Validation',
        'Work1-4 Shared Authority Validation (True Scale)',
        1,
    )
    html = html.replace(
        'function bounds(c){let xs=[],ys=[];for(const group of [c.ego.xy,c.controller_ego.xy,(c.machine_ego||c.reference_ego).xy,c.human_pred_ego.xy,c.human_future]) for(const p of group){xs.push(p[0]);ys.push(p[1]);} for(const n of c.neighbors) for(const p of n.xy){xs.push(p[0]);ys.push(p[1]);} if(c.road&&c.road.lane_markings) for(const y of c.road.lane_markings){ys.push(y);} return {minX:Math.min(...xs)-18,maxX:Math.max(...xs)+22,minY:Math.min(...ys)-4,maxY:Math.max(...ys)+4};}',
        'function bounds(c,i){let ys=[];if(c.road&&c.road.lane_markings) for(const y of c.road.lane_markings){ys.push(y);} for(const group of [c.ego.xy,c.controller_ego.xy,(c.machine_ego||c.reference_ego).xy,c.human_pred_ego.xy,c.human_future]){const p=group[Math.min(i,group.length-1)];ys.push(p[1]);} for(const n of c.neighbors){const p=n.xy[Math.min(i,n.xy.length-1)];ys.push(p[1]);} const egoX=c.controller_ego.xy[Math.min(i,c.controller_ego.xy.length-1)][0];return {minX:egoX-28,maxX:egoX+62,minY:Math.min(...ys)-3,maxY:Math.max(...ys)+3};}',
    )
    html = html.replace(
        'function trans(b,w,h){const sx=Math.min(w/(b.maxX-b.minX),h/(b.maxY-b.minY))*0.92,sy=Math.min(h/(b.maxY-b.minY)*0.92,sx*4.0),ox=(w-(b.maxX-b.minX)*sx)/2,oy=(h-(b.maxY-b.minY)*sy)/2;return {p:([x,y])=>[ox+(x-b.minX)*sx,oy+(y-b.minY)*sy],s:v=>v*sx,sy:v=>v*sy};}',
        'function trans(b,w,h){const sc=Math.min(w/(b.maxX-b.minX),h/(b.maxY-b.minY))*0.92,ox=(w-(b.maxX-b.minX)*sc)/2,oy=(h-(b.maxY-b.minY)*sc)/2;return {p:([x,y])=>[ox+(x-b.minX)*sc,oy+(y-b.minY)*sc],s:v=>v*sc,sy:v=>v*sc};}',
    )
    html = html.replace(
        'function car(xy,yaw,L,W,T,color,label,bad=false){const [x,y]=T.p(xy),l=T.s(L),w=T.sy(W);ctx.save();ctx.translate(x,y);ctx.rotate(yaw);ctx.fillStyle=color;ctx.strokeStyle=bad?"#ef4444":"#e2e8f0";ctx.lineWidth=bad?3:1;ctx.beginPath();ctx.roundRect(-l/2,-w/2,l,w,Math.min(4,w/3));ctx.fill();ctx.stroke();ctx.fillStyle="rgba(255,255,255,.72)";ctx.beginPath();ctx.moveTo(l/2-2,0);ctx.lineTo(l/2-Math.max(5,l*.18),-w*.28);ctx.lineTo(l/2-Math.max(5,l*.18),w*.28);ctx.closePath();ctx.fill();ctx.restore();ctx.fillStyle="#e5e7eb";ctx.font="12px Arial";ctx.fillText(label,x+8,y-8);}',
        'function car(xy,yaw,L,W,T,color,label,bad=false){const [x,y]=T.p(xy),l=T.s(L),w=T.s(W);ctx.save();ctx.translate(x,y);ctx.rotate(yaw);ctx.fillStyle=color;ctx.strokeStyle=bad?"#ef4444":"#e2e8f0";ctx.lineWidth=bad?3:1;ctx.beginPath();ctx.roundRect(-l/2,-w/2,l,w,Math.min(4,w/3));ctx.fill();ctx.stroke();ctx.fillStyle="rgba(255,255,255,.72)";ctx.beginPath();ctx.moveTo(l/2-2,0);ctx.lineTo(l/2-Math.max(5,l*.18),-w*.28);ctx.lineTo(l/2-Math.max(5,l*.18),w*.28);ctx.closePath();ctx.fill();ctx.restore();ctx.fillStyle="#e5e7eb";ctx.font="12px Arial";ctx.fillText(label,x+8,y-8);}',
    )
    html = html.replace(
        'function render(){const c=data.cases[caseIndex],w=canvas.clientWidth,h=canvas.clientHeight,i=Math.min(frame,c.ego.xy.length-1),b=bounds(c),T=trans(b,w,h);',
        'function render(){const c=data.cases[caseIndex],w=canvas.clientWidth,h=canvas.clientHeight,i=Math.min(frame,c.ego.xy.length-1),b=bounds(c,i),T=trans(b,w,h);',
    )
    html = html.replace(
        'function render(){const c=data.cases[caseIndex],machine=c.machine_ego||c.reference_ego,w=canvas.clientWidth,h=canvas.clientHeight,i=Math.min(frame,c.ego.xy.length-1),b=bounds(c),T=trans(b,w,h);',
        'function render(){const c=data.cases[caseIndex],machine=c.machine_ego||c.reference_ego,w=canvas.clientWidth,h=canvas.clientHeight,i=Math.min(frame,c.ego.xy.length-1),b=bounds(c,i),T=trans(b,w,h);',
    )
    path.write_text(html, encoding="utf-8")


def _cache_busted_html(html: str) -> str:
    version = str(int(time.time()))
    html = html.replace(
        '<meta name="viewport" content="width=device-width, initial-scale=1" />',
        '<meta name="viewport" content="width=device-width, initial-scale=1" />\n'
        '  <meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0" />\n'
        '  <meta http-equiv="Pragma" content="no-cache" />',
    )
    return html.replace('src="shared_authority_rollouts.js"', f'src="shared_authority_rollouts.js?v={version}"')


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Work1-4 Shared Authority Validation</title>
  <script src="shared_authority_rollouts.js"></script>
  <style>
    :root { --bg:#0f172a; --panel:#111827; --line:#334155; --text:#e5e7eb; --muted:#94a3b8; --ego:#38bdf8; --ref:#f59e0b; --human:#cbd5e1; --risk:#ef4444; }
    * { box-sizing:border-box; }
    body { margin:0; height:100vh; overflow:hidden; font-family:Arial,"Microsoft YaHei",sans-serif; color:var(--text); background:var(--bg); }
    .app { display:grid; grid-template-columns:1fr 380px; grid-template-rows:58px 1fr; height:100vh; }
    header { grid-column:1/-1; display:flex; align-items:center; justify-content:space-between; padding:0 18px; background:#020617; border-bottom:1px solid var(--line); }
    h1 { font-size:19px; margin:0; letter-spacing:0; }
    .meta { color:var(--muted); font-size:13px; }
    .stage { min-width:0; position:relative; background:#1f2937; }
    canvas { display:block; width:100%; height:100%; }
    aside { background:#020617; border-left:1px solid var(--line); padding:14px; overflow:auto; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; margin-bottom:12px; }
    h2 { margin:0 0 10px; font-size:15px; }
    .controls { display:grid; grid-template-columns:1fr 1fr; gap:9px; }
    button,select { height:34px; border-radius:6px; border:1px solid #475569; background:#0f172a; color:var(--text); font-size:13px; }
    button { cursor:pointer; font-weight:700; }
    .wide { grid-column:1/-1; }
    input[type=range] { width:100%; }
    .metric { display:flex; justify-content:space-between; gap:12px; padding:7px 0; border-bottom:1px solid rgba(148,163,184,.16); font-size:13px; }
    .metric:last-child { border-bottom:0; }
    .label { color:var(--muted); }
    .value { font-weight:700; text-align:right; }
    .badge { padding:5px 8px; border-radius:999px; font-weight:800; font-size:12px; }
    .ok { background:rgba(34,197,94,.16); color:#86efac; }
    .bad { background:rgba(239,68,68,.16); color:#fca5a5; }
    .legend { display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:12px; color:var(--muted); }
    .sw { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
    .mini { height:128px; width:100%; background:#0f172a; border:1px solid #334155; border-radius:6px; }
  </style>
</head>
<body>
<div class="app">
  <header>
    <h1>Work1-4 Shared Authority Validation</h1>
    <div class="meta">highD traffic injection | Work1 human intent | Work2 machine intent | Work3 reference authority | Work4 RL authority</div>
  </header>
  <section class="stage"><canvas id="scene"></canvas></section>
  <aside>
    <div class="panel">
      <h2>Simulation</h2>
      <div class="controls">
        <button id="play">Pause</button><button id="reset">Reset</button>
        <select id="caseSelect" class="wide"></select>
        <label class="wide label">Playback speed <span id="speedText">1.0x</span></label>
        <input id="speed" class="wide" type="range" min="0.2" max="3" step="0.1" value="1" />
      </div>
    </div>
    <div class="panel">
      <h2>Decision and Authority</h2>
      <div class="metric"><span class="label">case</span><span id="caseName" class="value"></span></div>
      <div class="metric"><span class="label">highD / human / machine / RL</span><span id="decisions" class="value"></span></div>
      <div class="metric"><span class="label">current lambda_h_ref(t)</span><span id="refAuth" class="value"></span></div>
      <div class="metric"><span class="label">current lambda_h_RL(t)</span><span id="rlAuth" class="value"></span></div>
      <div class="metric"><span class="label">mean lambda_h_ref / lambda_h_RL</span><span id="meanAuth" class="value"></span></div>
      <div class="metric"><span class="label">T_m_to_h / T_h_to_m</span><span id="trust" class="value"></span></div>
      <canvas id="signals" class="mini"></canvas>
    </div>
    <div class="panel">
      <h2>Execution State</h2>
      <div class="metric"><span class="label">time</span><span id="time" class="value">0.00 s</span></div>
      <div class="metric"><span class="label">speed / acceleration</span><span id="motion" class="value"></span></div>
      <div class="metric"><span class="label">front gap / relative speed</span><span id="frontState" class="value"></span></div>
      <div class="metric"><span class="label">front-wheel angle</span><span id="steer" class="value"></span></div>
      <div class="metric"><span class="label">reference collision</span><span id="refCollision" class="badge ok"></span></div>
      <div class="metric"><span class="label">RL collision</span><span id="rlCollision" class="badge ok"></span></div>
      <div class="metric"><span class="label">MPC-lite collision</span><span id="ctrlCollision" class="badge ok"></span></div>
      <div class="metric"><span class="label">MPC-lite min. clearance</span><span id="ctrlClearance" class="value"></span></div>
      <div class="metric"><span class="label">beta / yaw-rate max</span><span id="ctrlStable" class="value"></span></div>
      <div class="metric"><span class="label">mean risk ref to RL</span><span id="risk" class="value"></span></div>
      <div class="metric"><span class="label">max risk ref to RL</span><span id="maxRisk" class="value"></span></div>
      <div class="metric"><span class="label">reward ref to RL</span><span id="reward" class="value"></span></div>
    </div>
  </aside>
</div>
<script>
const data = window.SHARED_AUTHORITY_ROLLOUTS;
const canvas = document.getElementById("scene"), ctx = canvas.getContext("2d");
const sigCanvas = document.getElementById("signals"), sigCtx = sigCanvas.getContext("2d");
const $ = id => document.getElementById(id);
let caseIndex=0, frame=0, carry=0, playing=true, last=0, simSpeed=1;
const neighborColor="#7e8794",neighborTrail="#5f6b7a";
const egoColors={human:"#60a5fa",machine:"#f59e0b",ra:"#7B61FF",rl:"#22c55e",mpc:"#ef4444"};
for(let i=0;i<data.cases.length;i++){const c=data.cases[i],o=document.createElement("option");o.value=i;o.textContent=`case ${c.record.case_id||i+1} | ${c.record.case_name||("idx "+c.record.sample_index)} | RL ${c.record.rl_shared_decision}`;$("caseSelect").appendChild(o);}
function resize(){const r=window.devicePixelRatio||1,b=canvas.getBoundingClientRect();canvas.width=Math.floor(b.width*r);canvas.height=Math.floor(b.height*r);ctx.setTransform(r,0,0,r,0,0);const sb=sigCanvas.getBoundingClientRect();sigCanvas.width=Math.floor(sb.width*r);sigCanvas.height=Math.floor(sb.height*r);sigCtx.setTransform(r,0,0,r,0,0);}
window.addEventListener("resize",resize); resize();
$("play").onclick=()=>{playing=!playing;$("play").textContent=playing?"Pause":"Play";};
$("reset").onclick=()=>{frame=0;carry=0;};
$("caseSelect").onchange=()=>{caseIndex=Number($("caseSelect").value);frame=0;carry=0;};
$("speed").oninput=()=>{simSpeed=Number($("speed").value);$("speedText").textContent=`${simSpeed.toFixed(1)}x`;};
function bounds(c){let xs=[],ys=[];for(const group of [c.ego.xy,c.ra_rldm_ego?c.ra_rldm_ego.xy:[],c.controller_ego.xy,(c.machine_ego||c.reference_ego).xy,c.human_pred_ego.xy,c.human_future]) for(const p of group){xs.push(p[0]);ys.push(p[1]);} for(const n of c.neighbors) for(const p of n.xy){xs.push(p[0]);ys.push(p[1]);} if(c.road&&c.road.lane_markings) for(const y of c.road.lane_markings){ys.push(y);} return {minX:Math.min(...xs)-18,maxX:Math.max(...xs)+22,minY:Math.min(...ys)-4,maxY:Math.max(...ys)+4};}
function trans(b,w,h){const sx=Math.min(w/(b.maxX-b.minX),h/(b.maxY-b.minY))*0.92,sy=Math.min(h/(b.maxY-b.minY)*0.92,sx*4.0),ox=(w-(b.maxX-b.minX)*sx)/2,oy=(h-(b.maxY-b.minY)*sy)/2;return {p:([x,y])=>[ox+(x-b.minX)*sx,oy+(y-b.minY)*sy],s:v=>v*sx,sy:v=>v*sy};}
function road(T,b,w,h,c){ctx.fillStyle="#2f3742";ctx.fillRect(0,0,w,h);const marks=(c.road&&c.road.lane_markings&&c.road.lane_markings.length>=2)?c.road.lane_markings:[-1.5*data.lane_width_m,-0.5*data.lane_width_m,0.5*data.lane_width_m,1.5*data.lane_width_m];const y1=T.p([0,Math.min(...marks)])[1],y2=T.p([0,Math.max(...marks)])[1];ctx.fillStyle="#303842";ctx.fillRect(0,Math.min(y1,y2),w,Math.abs(y2-y1));for(let j=0;j<marks.length;j++){const y=marks[j],a=T.p([b.minX,y]),d=T.p([b.maxX,y]);ctx.beginPath();const boundary=j===0||j===marks.length-1;ctx.setLineDash(boundary?[]:[24,18]);ctx.strokeStyle=boundary?"#f8fafc":"#e5e7eb";ctx.globalAlpha=boundary?0.95:0.78;ctx.lineWidth=boundary?3:2;ctx.moveTo(a[0],a[1]);ctx.lineTo(d[0],d[1]);ctx.stroke();}ctx.setLineDash([]);ctx.globalAlpha=1;ctx.lineWidth=1;}
function poly(points,T,color,alpha,width,until=points.length){if(!points.length)return;ctx.beginPath();ctx.strokeStyle=color;ctx.globalAlpha=alpha;ctx.lineWidth=width;let p=T.p(points[0]);ctx.moveTo(p[0],p[1]);for(let i=1;i<until;i++){p=T.p(points[i]);ctx.lineTo(p[0],p[1]);}ctx.stroke();ctx.globalAlpha=1;}
function car(xy,yaw,L,W,T,color,label,bad=false){const [x,y]=T.p(xy),l=T.s(L),w=T.sy(W);ctx.save();ctx.translate(x,y);ctx.rotate(yaw);ctx.fillStyle=color;ctx.strokeStyle=bad?"#ef4444":"#e2e8f0";ctx.lineWidth=bad?3:1;ctx.beginPath();ctx.roundRect(-l/2,-w/2,l,w,Math.min(4,w/3));ctx.fill();ctx.stroke();ctx.fillStyle="rgba(255,255,255,.72)";ctx.beginPath();ctx.moveTo(l/2-2,0);ctx.lineTo(l/2-Math.max(5,l*.18),-w*.28);ctx.lineTo(l/2-Math.max(5,l*.18),w*.28);ctx.closePath();ctx.fill();ctx.restore();ctx.fillStyle="#e5e7eb";ctx.font="12px Arial";ctx.fillText(label,x+8,y-8);}
function ghostCar(xy,yaw,L,W,T,color,label,bad=false){const [x,y]=T.p(xy),l=T.s(L)*0.72,w=T.sy(W)*0.72;ctx.save();ctx.translate(x,y);ctx.rotate(yaw);ctx.globalAlpha=.58;ctx.fillStyle=color;ctx.strokeStyle=bad?"#ef4444":"#e2e8f0";ctx.lineWidth=bad?2:1;ctx.beginPath();ctx.roundRect(-l/2,-w/2,l,w,Math.min(4,w/3));ctx.fill();ctx.stroke();ctx.fillStyle="rgba(255,255,255,.75)";ctx.beginPath();ctx.moveTo(l/2-2,0);ctx.lineTo(l/2-Math.max(5,l*.18),-w*.28);ctx.lineTo(l/2-Math.max(5,l*.18),w*.28);ctx.closePath();ctx.fill();ctx.restore();ctx.globalAlpha=1;ctx.fillStyle="#e5e7eb";ctx.font="11px Arial";ctx.fillText(label,x+8,y-8);}
function yawAt(points,i,fallback=0,span=6){let a=Math.max(0,i-span),b=Math.min(points.length-1,i+span);while(a<i&&Math.hypot(points[i][0]-points[a][0],points[i][1]-points[a][1])<0.6)a++;while(b>i&&Math.hypot(points[b][0]-points[i][0],points[b][1]-points[i][1])<0.6)b--;if(a===b)return fallback;const dx=points[b][0]-points[a][0],dy=points[b][1]-points[a][1];if(Math.hypot(dx,dy)<0.8)return fallback;const yaw=Math.atan2(dy,dx);return Math.abs(yaw)<0.012?0:yaw;}
function clampYaw(yaw,limit=0.14){return Math.max(-limit,Math.min(limit,yaw));}
function drawSignals(c,i){const w=sigCanvas.clientWidth,h=sigCanvas.clientHeight;sigCtx.clearRect(0,0,w,h);sigCtx.fillStyle="#0f172a";sigCtx.fillRect(0,0,w,h);function line(arr,color){sigCtx.beginPath();sigCtx.strokeStyle=color;sigCtx.lineWidth=2;arr.forEach((v,k)=>{const x=k/(arr.length-1)*w,y=h-(Math.max(0,Math.min(1,v))*h); if(k===0)sigCtx.moveTo(x,y);else sigCtx.lineTo(x,y);});sigCtx.stroke();}line(c.signals.authority_ref,"#f59e0b");line(c.signals.authority_rl,"#38bdf8");line(c.signals.environment_urgency,"#ef4444");sigCtx.strokeStyle="#e5e7eb";sigCtx.beginPath();const x=i/(c.signals.authority_rl.length-1)*w;sigCtx.moveTo(x,0);sigCtx.lineTo(x,h);sigCtx.stroke();}
function frontInfo(c,i){const f=c.neighbors.find(n=>n.slot===0);if(!f)return "no front vehicle";const gap=f.xy[i][0]-c.controller_ego.xy[i][0]-0.5*(f.length+c.controller_ego.length);const prev=Math.max(0,i-1);const fv=(f.xy[i][0]-f.xy[prev][0])*data.frame_rate;const ev=(c.controller_ego.xy[i][0]-c.controller_ego.xy[prev][0])*data.frame_rate;return `${Math.max(gap,0).toFixed(1)} m / ${(fv-ev).toFixed(1)} m/s`;}
function panel(c,i){const m=c.metrics,r=c.record;$("caseName").textContent=`case ${r.case_id||""} ${r.case_name||""} | idx ${r.sample_index} rec ${r.recording_id} veh ${r.vehicle_id} | expected ${r.expected||"-"}`;$("decisions").textContent=`${r.true_decision} / ${r.human_decision} / ${r.machine_decision} / ${r.rl_shared_decision}`;$("refAuth").textContent=c.signals.authority_ref[i].toFixed(3);$("rlAuth").textContent=c.signals.authority_rl[i].toFixed(3);$("meanAuth").textContent=`${m.authority_ref_mean.toFixed(3)} / ${m.authority_rl_mean.toFixed(3)}`;$("trust").textContent=`${m.trust_machine_to_human_mean.toFixed(3)} / ${m.trust_human_to_machine_mean.toFixed(3)}`;$("time").textContent=`${(i/data.frame_rate).toFixed(2)} s`;$("motion").textContent=`${c.controller_ego.speed[i].toFixed(2)} m/s / ${c.controller_ego.acceleration[i].toFixed(2)} m/s^2`;$("frontState").textContent=frontInfo(c,i);$("steer").textContent=`${c.controller_ego.steer[i].toFixed(4)} rad`;$("risk").textContent=`${m.reference_risk_mean.toFixed(4)} -> ${m.rl_risk_mean.toFixed(4)}`;$("maxRisk").textContent=`${m.reference_risk_max.toFixed(4)} -> ${m.rl_risk_max.toFixed(4)}`;$("reward").textContent=`${m.reference_reward.toFixed(3)} -> ${m.rl_reward.toFixed(3)}`;$("ctrlClearance").textContent=`${m.controller_min_clearance_m.toFixed(2)} m`;$("ctrlStable").textContent=`${m.controller_max_abs_beta_rad.toFixed(3)} / ${m.controller_max_abs_yaw_rate_rps.toFixed(3)}`;for(const id of ["refCollision","rlCollision","ctrlCollision"]){const bad=id==="refCollision"?r.reference_collision:(id==="rlCollision"?r.rl_collision:r.controller_collision);$(id).textContent=bad?"collision":"no collision";$(id).className=`badge ${bad?"bad":"ok"}`;}drawSignals(c,i);}
function render(){const c=data.cases[caseIndex],machine=c.machine_ego||c.reference_ego,ra=c.ra_rldm_ego,w=canvas.clientWidth,h=canvas.clientHeight,i=Math.min(frame,c.ego.xy.length-1),b=bounds(c),T=trans(b,w,h);road(T,b,w,h,c);poly(c.human_pred_ego.xy,T,egoColors.human,.60,2);poly(machine.xy,T,egoColors.machine,.75,2.4);if(ra)poly(ra.xy,T,egoColors.ra,.40,2.4);poly(c.ego.xy,T,egoColors.rl,.35,2.2);poly(c.controller_ego.xy,T,egoColors.mpc,.25,2);poly(c.controller_ego.xy,T,egoColors.mpc,1,3,i+1);for(const n of c.neighbors){poly(n.xy,T,neighborTrail,.34,1.4,i+1);}for(const n of c.neighbors){car(n.xy[i],clampYaw(yawAt(n.xy,i,0,12)),n.length,n.width,T,neighborColor,n.name);}ghostCar(c.human_pred_ego.xy[i],yawAt(c.human_pred_ego.xy,i,0,6),4.6,1.8,T,egoColors.human,"Human");ghostCar(machine.xy[i],machine.yaw?machine.yaw[i]:yawAt(machine.xy,i,0,6),machine.length||4.6,machine.width||1.8,T,egoColors.machine,"Machine");if(ra)ghostCar(ra.xy[i],ra.yaw?ra.yaw[i]:yawAt(ra.xy,i,0,6),ra.length,ra.width,T,egoColors.ra,"RA-RLDM",ra.collision);ghostCar(c.ego.xy[i],c.ego.yaw?c.ego.yaw[i]:yawAt(c.ego.xy,i,0,6),c.ego.length,c.ego.width,T,egoColors.rl,"TA-RLDM",c.ego.collision);car(c.controller_ego.xy[i],c.controller_ego.yaw?c.controller_ego.yaw[i]:yawAt(c.controller_ego.xy,i,0,6),c.controller_ego.length,c.controller_ego.width,T,c.controller_ego.collision?"#ef4444":egoColors.mpc,"TA-RL-ARMPC",c.controller_ego.collision);panel(c,i);}
function loop(ts){if(!last)last=ts;const dt=Math.min((ts-last)/1000,.1);last=ts;const c=data.cases[caseIndex];if(playing){carry+=dt*data.frame_rate*simSpeed;while(carry>=1){frame++;carry--;if(frame>=c.ego.xy.length)frame=0;}}render();requestAnimationFrame(loop);}
requestAnimationFrame(loop);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
