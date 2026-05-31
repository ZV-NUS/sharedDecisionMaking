# Installation

For a full collaborator-oriented setup guide, including software versions,
Unity, keyboard/G29 DIL configuration, and one-click commands, read
`ENVIRONMENT_SETUP.md`.

## Python

Tested with Python 3.10 through Conda.

Recommended one-click setup on Windows:

```powershell
.\setup_env.bat
```

This command:

1. creates `tase_highd` from `environment.yml` if it does not exist;
2. updates the environment if it already exists;
3. runs `scripts/check_environment.py`;
4. runs `scripts/run_release_smoke_tests.py`.

To install without running smoke tests:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -SkipSmokeTest
```

If Conda is unavailable and a Python 3.10+ environment is already active:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -UsePipOnly
```

Manual setup:

```powershell
conda env create -f environment.yml
conda activate tase_highd
python scripts/check_environment.py
```

If PowerShell cannot activate Conda:

```powershell
conda init powershell
```

Restart PowerShell and run `conda activate tase_highd` again.

## Optional G29 Dependency

The Logitech G29 adapter uses `pygame`.

```powershell
conda run -n tase_highd pip install pygame
```

If no G29 is connected, `inspect_g29.py` may report no joystick. That is expected. A `ModuleNotFoundError: pygame` means the package is not installed.

## Unity

Open the Unity project at:

```text
driver_in_loop/unity_client
```

The project was developed with Unity 6.x. Unity 2022.3 LTS may require minor project migration and should be tested before formal experiments.
