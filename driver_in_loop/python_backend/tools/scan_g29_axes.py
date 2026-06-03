from __future__ import annotations

import argparse
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan G29/G923 axes and report changed axes.")
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import pygame  # type: ignore
    except ModuleNotFoundError:
        print("pygame is not installed. Run: conda run -n tase_highd python -m pip install pygame")
        raise SystemExit(2)

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() <= args.joystick_index:
        print(f"No joystick index {args.joystick_index}. Detected joystick count: {pygame.joystick.get_count()}.")
        raise SystemExit(3)

    joy = pygame.joystick.Joystick(args.joystick_index)
    joy.init()
    pygame.event.pump()
    n_axes = joy.get_numaxes()
    baseline = [float(joy.get_axis(i)) for i in range(n_axes)]
    min_values = baseline[:]
    max_values = baseline[:]

    print(f"name={joy.get_name()}")
    print(f"axes={n_axes} buttons={joy.get_numbuttons()} hats={joy.get_numhats()}")
    print("Now turn the wheel, press throttle, and press brake within the next seconds.")
    print("The script will report which axis changed.")

    dt = 1.0 / max(args.rate_hz, 1e-6)
    end = time.perf_counter() + args.duration_s
    try:
        while time.perf_counter() < end:
            pygame.event.pump()
            current = [float(joy.get_axis(i)) for i in range(n_axes)]
            for i, value in enumerate(current):
                min_values[i] = min(min_values[i], value)
                max_values[i] = max(max_values[i], value)
            print("axes=" + str([round(v, 3) for v in current]))
            time.sleep(dt)
    finally:
        joy.quit()
        pygame.joystick.quit()

    print("")
    print("Axis change summary:")
    for i in range(n_axes):
        span = max_values[i] - min_values[i]
        print(
            f"  axis {i}: baseline={baseline[i]: .3f}, "
            f"min={min_values[i]: .3f}, max={max_values[i]: .3f}, span={span: .3f}"
        )
    print("")
    print("Use the axes with the largest pedal spans, for example:")
    print("  --g29-throttle-axis 1 --g29-brake-axis 2")


if __name__ == "__main__":
    main()
