from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Logitech G29 pygame axes and buttons.")
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import pygame  # type: ignore
    except ModuleNotFoundError:
        print("pygame is not installed. Install it with:")
        print("  conda run -n tase_highd pip install pygame")
        raise SystemExit(2)

    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    if count <= args.joystick_index:
        print(f"No joystick index {args.joystick_index}. Detected joystick count: {count}.")
        print("Connect Logitech G29, open Logitech G HUB, then run this script again.")
        raise SystemExit(3)

    joy = pygame.joystick.Joystick(args.joystick_index)
    joy.init()
    print(f"name={joy.get_name()}")
    print(f"axes={joy.get_numaxes()} buttons={joy.get_numbuttons()} hats={joy.get_numhats()}")
    print("Move wheel/pedals. Press Ctrl+C to stop.")

    dt = 1.0 / max(args.rate_hz, 1e-6)
    end = time.perf_counter() + args.duration_s
    try:
        while time.perf_counter() < end:
            pygame.event.pump()
            axes = [round(float(joy.get_axis(i)), 3) for i in range(joy.get_numaxes())]
            buttons = [i for i in range(joy.get_numbuttons()) if joy.get_button(i)]
            hats = [joy.get_hat(i) for i in range(joy.get_numhats())]
            print(f"axes={axes} buttons={buttons} hats={hats}")
            time.sleep(dt)
    finally:
        joy.quit()
        pygame.joystick.quit()


if __name__ == "__main__":
    main()
