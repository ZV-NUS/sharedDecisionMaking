from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
EGO_COLORS = {
    "human": (96, 165, 250),
    "ref": (56, 189, 248),
    "rl": (6, 182, 212),
    "mpc": (34, 211, 238),
}
NEIGHBOR_COLOR = (139, 149, 163)
NEIGHBOR_TRAIL = (100, 116, 139)
ROAD_BG = (47, 55, 66)
ROAD_FILL = (48, 56, 66)
TEXT = (229, 231, 235)
MUTED = (148, 163, 184)
WHITE = (248, 250, 252)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a summary MP4 for shared-authority validation cases.")
    parser.add_argument(
        "--input",
        default="outputs/shared_authority_validation/shared_authority_rollouts.js",
        help="Rollout JS exported by export_shared_authority_realtime_demo.py.",
    )
    parser.add_argument(
        "--output",
        default="outputs/shared_authority_validation/videos/work1_4_validation_cases.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--stride", type=int, default=2, help="Use every Nth rollout frame.")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=912)
    args = parser.parse_args()

    payload = _load_payload(ROOT / args.input)
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    font = _load_font(24)
    font_small = _load_font(18)
    font_big = _load_font(34)

    with imageio.get_writer(out_path, fps=args.fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for case in payload["cases"]:
            title = _title_frame(case, args.width, args.height, font_big, font, font_small)
            for _ in range(max(1, args.fps)):
                writer.append_data(np.asarray(title))
            frames = len(case["controller_ego"]["xy"])
            for frame_idx in range(0, frames, max(1, args.stride)):
                writer.append_data(np.asarray(_draw_frame(payload, case, frame_idx, args.width, args.height, font, font_small)))

    print(json.dumps({"video": str(out_path), "cases": len(payload["cases"]), "fps": args.fps}, ensure_ascii=False, indent=2))


def _load_payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"window\.SHARED_AUTHORITY_ROLLOUTS\s*=\s*(.*);\s*$", text, re.S)
    if not match:
        raise ValueError(f"Cannot parse rollout payload from {path}")
    return json.loads(match.group(1))


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _bounds(case: dict) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []
    for group in (
        case["ego"]["xy"],
        case["controller_ego"]["xy"],
        case["reference_ego"]["xy"],
        case["human_pred_ego"]["xy"],
    ):
        for x, y in group:
            xs.append(float(x))
            ys.append(float(y))
    for neighbor in case["neighbors"]:
        for x, y in neighbor["xy"]:
            xs.append(float(x))
            ys.append(float(y))
    for y in case.get("road", {}).get("lane_markings", []):
        ys.append(float(y))
    return {
        "min_x": min(xs) - 18.0,
        "max_x": max(xs) + 22.0,
        "min_y": min(ys) - 4.0,
        "max_y": max(ys) + 4.0,
    }


def _transform(bounds: dict[str, float], width: int, height: int):
    scene_w = width
    scene_h = int(height * 0.78)
    sx = min(scene_w / (bounds["max_x"] - bounds["min_x"]), scene_h / (bounds["max_y"] - bounds["min_y"])) * 0.92
    sy = min(scene_h / (bounds["max_y"] - bounds["min_y"]) * 0.92, sx * 4.0)
    ox = (scene_w - (bounds["max_x"] - bounds["min_x"]) * sx) / 2.0
    oy = (scene_h - (bounds["max_y"] - bounds["min_y"]) * sy) / 2.0

    def p(xy):
        x, y = xy
        return ox + (float(x) - bounds["min_x"]) * sx, oy + (float(y) - bounds["min_y"]) * sy

    return p, sx, sy, scene_h


def _draw_frame(payload: dict, case: dict, frame_idx: int, width: int, height: int, font, font_small) -> Image.Image:
    image = Image.new("RGB", (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(image)
    bounds = _bounds(case)
    transform, sx, sy, scene_h = _transform(bounds, width, height)
    _draw_road(draw, case, transform, bounds, width, scene_h)
    _poly(draw, case["human_pred_ego"]["xy"], transform, EGO_COLORS["human"], width=3)
    _poly(draw, case["reference_ego"]["xy"], transform, EGO_COLORS["ref"], width=4)
    _poly(draw, case["ego"]["xy"], transform, EGO_COLORS["rl"], width=3)
    _poly(draw, case["controller_ego"]["xy"][: frame_idx + 1], transform, EGO_COLORS["mpc"], width=5)
    for neighbor in case["neighbors"]:
        _poly(draw, neighbor["xy"][: frame_idx + 1], transform, NEIGHBOR_TRAIL, width=2)
    for neighbor in case["neighbors"]:
        _car(draw, neighbor["xy"][frame_idx], _yaw_at(neighbor["xy"], frame_idx, limit=0.14), neighbor["length"], neighbor["width"], transform, sx, sy, NEIGHBOR_COLOR, neighbor["name"], font_small)
    _car(draw, case["human_pred_ego"]["xy"][frame_idx], _yaw_at(case["human_pred_ego"]["xy"], frame_idx), 4.6, 1.8, transform, sx, sy, EGO_COLORS["human"], "human", font_small)
    _car(draw, case["reference_ego"]["xy"][frame_idx], _yaw_at(case["reference_ego"]["xy"], frame_idx), 4.6, 1.8, transform, sx, sy, EGO_COLORS["ref"], "ref", font_small)
    _car(draw, case["ego"]["xy"][frame_idx], _yaw_at(case["ego"]["xy"], frame_idx), 4.6, 1.8, transform, sx, sy, EGO_COLORS["rl"], "RL-ref", font_small)
    _car(draw, case["controller_ego"]["xy"][frame_idx], _yaw_at(case["controller_ego"]["xy"], frame_idx), 4.6, 1.8, transform, sx, sy, EGO_COLORS["mpc"], "MPC", font_small)
    _hud(draw, payload, case, frame_idx, width, height, scene_h, font, font_small)
    return image


def _draw_road(draw: ImageDraw.ImageDraw, case: dict, transform, bounds: dict[str, float], width: int, scene_h: int) -> None:
    draw.rectangle((0, 0, width, scene_h), fill=ROAD_BG)
    marks = case.get("road", {}).get("lane_markings") or [-5.25, -1.75, 1.75, 5.25]
    y_values = [transform((0.0, y))[1] for y in marks]
    draw.rectangle((0, min(y_values), width, max(y_values)), fill=ROAD_FILL)
    for idx, mark in enumerate(marks):
        y = transform((bounds["min_x"], mark))[1]
        boundary = idx == 0 or idx == len(marks) - 1
        color = WHITE if boundary else (220, 226, 232)
        line_width = 4 if boundary else 3
        if boundary:
            draw.line((0, y, width, y), fill=color, width=line_width)
        else:
            dash = 36
            gap = 24
            x = 0
            while x < width:
                draw.line((x, y, min(width, x + dash), y), fill=color, width=line_width)
                x += dash + gap


def _poly(draw: ImageDraw.ImageDraw, points: list, transform, color: tuple[int, int, int], width: int) -> None:
    if len(points) < 2:
        return
    draw.line([transform(p) for p in points], fill=color, width=width, joint="curve")


def _car(draw: ImageDraw.ImageDraw, xy, yaw: float, length: float, width: float, transform, sx: float, sy: float, color, label: str, font) -> None:
    x, y = transform(xy)
    car_w = sy * float(width)
    car_l = car_w * float(length) / max(float(width), 0.1)
    corners = np.array(
        [
            [-car_l / 2, -car_w / 2],
            [car_l / 2, -car_w / 2],
            [car_l / 2, car_w / 2],
            [-car_l / 2, car_w / 2],
        ],
        dtype=np.float32,
    )
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]], dtype=np.float32)
    poly = corners @ rot.T + np.array([x, y], dtype=np.float32)
    draw.polygon([tuple(p) for p in poly], fill=color, outline=(226, 232, 240))
    nose = np.array([[car_l / 2 - 4, 0], [car_l / 2 - max(8, car_l * 0.20), -car_w * 0.28], [car_l / 2 - max(8, car_l * 0.20), car_w * 0.28]], dtype=np.float32)
    nose_poly = nose @ rot.T + np.array([x, y], dtype=np.float32)
    draw.polygon([tuple(p) for p in nose_poly], fill=(235, 241, 248))
    draw.text((x + 8, y - 24), label, fill=TEXT, font=font)


def _yaw_at(points: list, idx: int, span: int = 6, limit: float | None = None) -> float:
    a = max(0, idx - span)
    b = min(len(points) - 1, idx + span)
    dx = float(points[b][0]) - float(points[a][0])
    dy = float(points[b][1]) - float(points[a][1])
    if (dx * dx + dy * dy) ** 0.5 < 0.3:
        yaw = 0.0
    else:
        yaw = float(np.arctan2(dy, dx))
    if limit is not None:
        yaw = float(np.clip(yaw, -limit, limit))
    return 0.0 if abs(yaw) < 0.012 else yaw


def _hud(draw: ImageDraw.ImageDraw, payload: dict, case: dict, frame_idx: int, width: int, height: int, scene_h: int, font, font_small) -> None:
    record = case["record"]
    metrics = case["metrics"]
    y0 = scene_h + 18
    draw.rectangle((0, scene_h, width, height), fill=(2, 6, 23))
    title = f"case {record.get('case_id')} | {record.get('case_name')} | expected: {record.get('expected')}"
    draw.text((28, y0), title, fill=TEXT, font=font)
    decisions = f"Decision highD / human / machine / RL: {record['true_decision']} / {record['human_decision']} / {record['machine_decision']} / {record['rl_shared_decision']}"
    draw.text((28, y0 + 42), decisions, fill=TEXT, font=font_small)
    t = frame_idx / float(payload["frame_rate"])
    mpc = case["controller_ego"]
    line2 = (
        f"t={t:4.2f}s | v={mpc['speed'][frame_idx]:.2f} m/s | a={mpc['acceleration'][frame_idx]:.2f} m/s2 | "
        f"MPC clearance={metrics['controller_min_clearance_m']:.2f} m | collision: ref/RL/MPC="
        f"{record['reference_collision']}/{record['rl_collision']}/{record['controller_collision']}"
    )
    draw.text((28, y0 + 76), line2, fill=MUTED, font=font_small)
    legend_x = width - 520
    for i, (name, color) in enumerate((("human", EGO_COLORS["human"]), ("ref", EGO_COLORS["ref"]), ("RL-ref", EGO_COLORS["rl"]), ("MPC", EGO_COLORS["mpc"]), ("traffic", NEIGHBOR_COLOR))):
        yy = y0 + 8 + i * 28
        draw.rectangle((legend_x, yy, legend_x + 28, yy + 14), fill=color)
        draw.text((legend_x + 38, yy - 5), name, fill=TEXT, font=font_small)


def _title_frame(case: dict, width: int, height: int, font_big, font, font_small) -> Image.Image:
    image = Image.new("RGB", (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(image)
    record = case["record"]
    draw.text((80, 250), f"Validation case {record.get('case_id')}", fill=TEXT, font=font_big)
    draw.text((80, 315), str(record.get("case_name")), fill=TEXT, font=font)
    draw.text((80, 370), f"Expected behavior: {record.get('expected')}", fill=MUTED, font=font)
    draw.text(
        (80, 430),
        f"Decision highD / human / machine / RL: {record['true_decision']} / {record['human_decision']} / {record['machine_decision']} / {record['rl_shared_decision']}",
        fill=TEXT,
        font=font_small,
    )
    draw.text((80, 475), "Shared authority validation: safety first, efficiency second, human intent when safe.", fill=MUTED, font=font_small)
    return image


if __name__ == "__main__":
    main()
