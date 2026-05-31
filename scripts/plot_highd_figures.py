from __future__ import annotations

from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    script = PROJECT_ROOT / "plot" / "plot_case1_2_trajectory.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

