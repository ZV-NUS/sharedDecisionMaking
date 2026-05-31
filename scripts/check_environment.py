from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "matplotlib",
    "yaml",
    "h5py",
    "torch",
]


REQUIRED_PATHS = [
    "src",
    "scripts",
    "plot",
    "plot_dil",
    "driver_in_loop/python_backend",
    "driver_in_loop/unity_client/Assets",
    "driver_in_loop/unity_client/Packages",
    "driver_in_loop/unity_client/ProjectSettings",
    "outputs/shared_authority_validation/shared_authority_rollouts.js",
]


def check_modules() -> list[str]:
    missing: list[str] = []
    for name in REQUIRED_MODULES:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    return missing


def check_paths() -> list[str]:
    missing: list[str] = []
    for rel in REQUIRED_PATHS:
        if not (PROJECT_ROOT / rel).exists():
            missing.append(rel)
    return missing


def main() -> int:
    print(f"Project root: {PROJECT_ROOT}")
    missing_modules = check_modules()
    missing_paths = check_paths()

    if missing_modules:
        print("Missing Python modules:")
        for name in missing_modules:
            print(f"  - {name}")
    else:
        print("Python module check: OK")

    if missing_paths:
        print("Missing project paths:")
        for rel in missing_paths:
            print(f"  - {rel}")
    else:
        print("Project path check: OK")

    if missing_modules or missing_paths:
        return 1
    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

