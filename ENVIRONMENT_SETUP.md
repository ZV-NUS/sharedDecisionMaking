# Environment Setup Guide

This document is written for collaborators who want to reproduce the highD
validation, Unity driver-in-the-loop (DIL) experiment, and paper figure
generation workflows from a fresh Windows machine.

## 1. Tested Software Versions

The release was prepared and smoke-tested with the following environment.

| Component | Tested version | Notes |
|---|---:|---|
| Windows | Windows 10/11 | PowerShell is recommended. |
| Git | 2.53.0.windows.1 | Any recent Git for Windows should work. |
| Conda | 4.10.3 | Miniconda or Anaconda is acceptable. |
| Python | 3.10.x | The Conda environment pins `python=3.10`. |
| Unity | 6000.3.16f1 | Project version recorded in `driver_in_loop/unity_client/ProjectSettings/ProjectVersion.txt`. |
| Unity package `com.unity.ugui` | 2.0.0 | Required by the DIL HUD. |
| Unity package `com.unity.ide.visualstudio` | 2.0.22 | Optional for editing C# scripts. |
| Logitech G29 | Optional | Keyboard input can replace G29 for testing. |

Core Python package versions on the prepared workstation:

| Package | Version observed |
|---|---:|
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| scipy | 1.15.3 |
| scikit-learn | 1.7.2 |
| matplotlib | 3.10.9 |
| PyYAML | 6.0.3 |
| h5py | 3.16.0 |
| torch | 2.12.0+cpu |
| tqdm | 4.67.3 |
| pygame | 2.6.1; required for G29 only. |

The exact package resolver may install newer patch versions. If the smoke tests
pass, the environment is acceptable.

## 2. Install Required Software

### 2.1 Git

Install Git for Windows:

```text
https://git-scm.com/download/win
```

Verify:

```powershell
git --version
```

### 2.2 Miniconda or Anaconda

Install Miniconda or Anaconda. Then open a new PowerShell window and verify:

```powershell
conda --version
```

If `conda activate` does not work in PowerShell:

```powershell
conda init powershell
```

Close PowerShell and open it again.

### 2.3 Unity for Driver-in-the-Loop Experiments

Install Unity Hub and a compatible Unity Editor.

Recommended:

```text
Unity 6000.3.16f1 or Unity 6.x LTS
```

Open this Unity project:

```text
driver_in_loop/unity_client
```

If Unity asks to migrate the project, make a backup first. For formal
experiments, use the Unity version listed above whenever possible.

### 2.4 Logitech G29

G29 is optional. The same DIL backend supports:

- keyboard input;
- Logitech G29 input through `pygame`.

Keyboard mode is sufficient for software validation before the steering wheel is
available.

## 3. Clone the Repository

```powershell
cd E:\gitProjects
git clone https://github.com/ZV-NUS/sharedDecisionMaking.git
cd sharedDecisionMaking
```

You can use any local path. The code uses paths relative to the repository root.
Do not hard-code the original development path.

## 4. One-Click Python Environment Setup

Run:

```powershell
.\setup_env.bat
```

This command automatically:

1. creates the `tase_highd` Conda environment if it does not exist;
2. updates the environment from `environment.yml` if it already exists;
3. refreshes Python dependencies with `pip install -r requirements.txt`,
   including optional G29 dependency `pygame`;
4. checks required Python modules and project paths;
5. runs the release smoke-test matrix.

The full setup can take several minutes, especially when installing PyTorch.
If Conda package metadata cannot be downloaded because of a temporary network
problem, the script will continue to the pip dependency refresh step and then
run the environment checks. If both Conda and pip cannot access the network,
connect to a stable network or configure a local package mirror.

If you only want to install dependencies and skip smoke tests:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -SkipSmokeTest
```

If the environment name `tase_highd` is already used for another project:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -EnvName tase_shared_driving
```

If Conda is unavailable but Python 3.10+ is already active:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1 -UsePipOnly
```

## 5. Verify the Environment

Activate the environment:

```powershell
conda activate tase_highd
```

Print installed versions:

```powershell
python scripts/print_environment_versions.py
```

Run the basic check:

```powershell
python scripts/check_environment.py
```

Run the release smoke tests:

```powershell
python scripts/run_release_smoke_tests.py
```

Expected result:

```text
Environment check passed.
Release smoke tests passed.
```

## 6. Run highD-Based Validation

The release includes a lightweight rollout file for reproducing the paper-style
validation figures without redistributing the raw highD dataset.

Generate highD scenario figures:

```powershell
python scripts/plot_highd_figures.py
```

Run the full highD workflow described for collaborators:

```powershell
type RUN_HIGHD_EXPERIMENTS.md
```

If you want to process the raw highD dataset, obtain the dataset separately and
configure the path through either:

```powershell
copy configs\local.example.yaml configs\local.yaml
```

or:

```powershell
$env:HIGHD_ROOT="D:\path\to\highD"
```

The raw highD dataset is not included in this repository.

## 7. Run Driver-in-the-Loop Backend Without Unity

Use this first to make sure the DIL Python backend works:

```powershell
python scripts/run_dil_demo.py
```

Run all keyboard-mode DIL smoke tests:

```powershell
python scripts/run_release_smoke_tests.py
```

The DIL smoke tests cover:

- paper Case 1 and Case 2;
- `human_only`;
- `ra_rldm`;
- `ta_rldm`;
- `ta_rldm_armpc`.

## 8. Run Unity Driver-in-the-Loop Visualization

### 8.1 Start Python Backend

Keyboard mode:

```powershell
conda activate tase_highd
python driver_in_loop\python_backend\run_driver_in_loop.py --paper-case-id 1 --input unity --mode ta_rldm_armpc
```

For Case 2:

```powershell
python driver_in_loop\python_backend\run_driver_in_loop.py --paper-case-id 2 --input unity --mode ta_rldm_armpc
```

Available modes:

```text
human_only
ra_rldm
ta_rldm
ta_rldm_armpc
```

### 8.2 Start Unity

1. Open Unity Hub.
2. Open project `driver_in_loop/unity_client`.
3. Open the generated DIL scene.
4. Press Play.
5. Use arrow keys as continuous driver steering/throttle/brake input.

The Python backend sends scenario state to Unity through UDP. Unity sends driver
input back to Python through UDP.

### 8.3 G29 Mode

Install and verify `pygame`:

```powershell
python -m pip install pygame
python driver_in_loop\python_backend\tools\inspect_g29.py
```

Run:

```powershell
python driver_in_loop\python_backend\run_driver_in_loop.py --paper-case-id 1 --input g29 --mode ta_rldm_armpc
```

If no G29 is connected, `inspect_g29.py` should report no joystick. That is
expected.

## 9. Generate DIL Paper Figures

After DIL experiment logs are available:

```powershell
python scripts/plot_dil_figures.py
```

Outputs are written under:

```text
results/
```

Generated figures are ignored by Git by default.

## 10. Common Problems

### `conda activate` is not recognized

Run:

```powershell
conda init powershell
```

Restart PowerShell.

### `pygame` is missing

Run:

```powershell
conda activate tase_highd
python -m pip install pygame
```

### UDP port is already in use

Find the process:

```powershell
Get-NetUDPEndpoint -LocalPort 50711 | Select-Object LocalAddress,LocalPort,OwningProcess
```

Stop it:

```powershell
Stop-Process <PID> -Force
```

### Unity scene opens but vehicles do not move

Check:

1. Python backend is running.
2. Unity is in Play mode.
3. UDP ports match the values in the Unity inspector and Python command.
4. The Game window is focused when using keyboard input.

### GitHub clone is slow

The Unity client contains project assets. Use a stable network connection. Raw
highD data and generated result videos are intentionally not included.

## 11. Minimal Command List

For a collaborator who only wants to verify the release:

```powershell
git clone https://github.com/ZV-NUS/sharedDecisionMaking.git
cd sharedDecisionMaking
.\setup_env.bat
conda activate tase_highd
python scripts/plot_highd_figures.py
python scripts/run_dil_demo.py
```

For Unity DIL:

```powershell
conda activate tase_highd
python driver_in_loop\python_backend\run_driver_in_loop.py --paper-case-id 1 --input unity --mode ta_rldm_armpc
```

Then open `driver_in_loop/unity_client` in Unity and press Play.
