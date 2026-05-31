from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], label: str) -> None:
    print(f"\n==== {label} ====")
    print(" ".join(cmd))
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise SystemExit(f"Failed: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run release smoke tests.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip figure-generation tests.")
    parser.add_argument("--skip-dil-matrix", action="store_true", help="Skip the DIL 2-case x 4-mode matrix.")
    args = parser.parse_args()

    py = sys.executable
    run([py, "scripts/check_environment.py"], "environment check")
    run([py, "-m", "compileall", "-q", "src", "scripts", "plot", "plot_dil", "driver_in_loop/python_backend"], "python compile check")

    if not args.skip_plots:
        run([py, "scripts/plot_highd_figures.py"], "highD 4-case figure generation")

    if not args.skip_dil_matrix:
        for paper_case in (1, 2):
            for mode in ("human_only", "ra_rldm", "ta_rldm", "ta_rldm_armpc"):
                run(
                    [
                        py,
                        "driver_in_loop/python_backend/run_driver_in_loop.py",
                        "--paper-case-id",
                        str(paper_case),
                        "--dil-mode",
                        mode,
                        "--input",
                        "keyboard",
                        "--no-udp",
                        "--duration-s",
                        "0.6",
                    ],
                    f"DIL keyboard case {paper_case} mode {mode}",
                )

    if not args.skip_plots:
        run([py, "scripts/plot_dil_figures.py"], "DIL Case 1-2 figure generation")

    print("\nRelease smoke tests passed.")


if __name__ == "__main__":
    main()

