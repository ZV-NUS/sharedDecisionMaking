# TASE02 Shared Driving Validation Release

This repository is the clean release version for reproducing the highD-based and driver-in-the-loop validation workflows of the human-machine shared driving study.

It includes:

- Work1-4 integrated highD validation code;
- trust-aware and risk-aware shared authority decision code;
- adaptive robust MPC-lite control code;
- Unity-based driver-in-the-loop visualization;
- keyboard and Logitech G29 input adapters;
- IEEE Transactions-style figure generation scripts.

## Quick Start

One-click setup on Windows:

```powershell
.\setup_env.bat
```

The command creates or updates the `tase_highd` Conda environment, checks the
installation, and runs the release smoke-test matrix. To skip smoke tests:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -SkipSmokeTest
```

Manual Python environment setup:

```powershell
conda env create -f environment.yml
conda activate tase_highd
python scripts/check_environment.py
```

Run a headless DIL smoke test:

```powershell
python scripts/run_dil_demo.py
```

Run the full release smoke test matrix:

```powershell
python scripts/run_release_smoke_tests.py
```

Generate highD paper trajectory figures from the included rollout file:

```powershell
python scripts/plot_highd_figures.py
```

Generate DIL paper figures after DIL experiment logs are available:

```powershell
python scripts/plot_dil_figures.py
```

## Documentation

- `ENVIRONMENT_SETUP.md`: one-click environment setup and software version guide.
- `INSTALL.md`: Python and dependency installation.
- `RUN_HIGHD_EXPERIMENTS.md`: highD validation and visualization workflow.
- `RUN_DIL_EXPERIMENTS.md`: keyboard/G29 DIL workflow.
- `UNITY_SETUP.md`: Unity project setup and operation.
- `G29_SETUP.md`: Logitech G29 setup.
- `FIGURE_GENERATION.md`: paper figure generation.
- `DATA_STRUCTURE.md`: data layout and highD data policy.
- `RELEASE_REQUIREMENTS.md`: release scope and validation checklist.

## Data Notice

The full raw highD dataset is not included. Users must obtain highD separately and configure its local path through `configs/local.yaml` or the `HIGHD_ROOT` environment variable.

The included `outputs/shared_authority_validation/shared_authority_rollouts.js` is the lightweight rollout file used by the demo visualization, Unity DIL backend, and paper plotting scripts.
