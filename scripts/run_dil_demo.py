from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "driver_in_loop" / "python_backend" / "run_driver_in_loop.py"),
        "--paper-case-id",
        "1",
        "--dil-mode",
        "ta_rldm_armpc",
        "--input",
        "policy",
        "--policy-key",
        "controller_ego",
        "--policy-mode",
        "state",
        "--headless",
        "--no-udp",
        "--duration-s",
        "1.0",
    ]
    return subprocess.call(cmd, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())

