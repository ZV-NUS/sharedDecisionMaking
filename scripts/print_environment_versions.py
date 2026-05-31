from __future__ import annotations

import importlib
import platform
import subprocess
import sys


MODULES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("PyYAML", "yaml"),
    ("h5py", "h5py"),
    ("torch", "torch"),
    ("tensorboard", "tensorboard"),
    ("pygame", "pygame"),
    ("tqdm", "tqdm"),
]


def command_version(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - diagnostic helper
        return f"unavailable ({exc})"
    text = (completed.stdout or completed.stderr).strip()
    return text.splitlines()[0] if text else "unavailable"


def module_version(import_name: str) -> str:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:
        return f"missing ({exc})"
    return str(getattr(module, "__version__", "installed"))


def main() -> int:
    print("System")
    print(f"  OS: {platform.platform()}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Git: {command_version(['git', '--version'])}")
    print(f"  Conda: {command_version(['conda', '--version'])}")
    print("")
    print("Python packages")
    for display_name, import_name in MODULES:
        print(f"  {display_name}: {module_version(import_name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
