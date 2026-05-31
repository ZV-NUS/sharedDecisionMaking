# Installation

## Python

Tested with Python 3.10 through Conda.

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

