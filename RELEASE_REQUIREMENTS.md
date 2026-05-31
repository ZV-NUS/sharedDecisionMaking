# GitHub Release Requirements for TASE02

## 1. Working Directory Boundary

The original development directory:

```text
E:\NUS科研2025\code\TASE02
```

is treated as a **read-only source reference** during GitHub release preparation.

No further code, configuration, documentation, plotting script, Unity file, or experiment file should be modified in the original directory.

The clean GitHub release repository must be prepared under:

```text
E:\gitProjects\project_1
```

All GitHub-release-specific work must be performed only inside `E:\gitProjects\project_1`, including:

- copying necessary source code from the original project;
- removing unused or temporary files;
- refactoring hard-coded local paths;
- writing release documentation;
- creating environment and configuration files;
- preparing demo data and reproducible cases;
- testing highD validation workflows;
- testing driver-in-the-loop workflows;
- testing Unity communication;
- testing keyboard and Logitech G29 input modes;
- generating paper figures;
- preparing `.gitignore`;
- initializing and uploading the Git repository.

## 2. Release Objective

Build a clean GitHub-ready repository that allows collaborators to:

1. install the required Python and Unity environments;
2. run the highD-based system validation;
3. run driver-in-the-loop validation with keyboard input;
4. run driver-in-the-loop validation with Logitech G29 input;
5. generate IEEE Transactions-style experimental figures;
6. reproduce the main paper cases without relying on the original local path structure.

The release repository should contain only the code, lightweight data, configuration files, Unity project files, and documentation required for reproduction.

## 3. Repository Cleaning Requirement

The current development repository contains historical scripts, temporary outputs, logs, videos, intermediate figures, local debugging files, and local absolute paths. These should not be uploaded directly.

Recommended workflow:

1. Create and use the clean release folder:

```text
E:\gitProjects\project_1
```

2. Copy only necessary code and reproducible experiment assets into this folder.
3. Remove or ignore unnecessary large files and local-only outputs.
4. Test all workflows inside `project_1`.
5. Initialize Git and upload this clean folder to GitHub.

## 4. Code and Content to Keep

### 4.1 Core highD System Validation Code

Keep code required for:

- highD scenario loading or processed-case loading;
- Work1 driver decision and control-intent prediction;
- Work2 automation-side tactical decision and intent generation;
- Work3 trust-aware shared authority decision;
- Work4 authority-aware adaptive robust MPC-lite control;
- Work1-4 integrated validation;
- HTML or lightweight visualization for highD validation;
- paper case rollout generation;
- paper figure generation.

### 4.2 Driver-in-the-Loop Code

Keep the complete DIL implementation:

- Python backend;
- UDP communication modules;
- keyboard input adapter;
- Logitech G29 input adapter;
- shared-control runtime;
- experiment logging;
- DIL case loader;
- DIL plotting scripts;
- Unity client project.

The DIL experiment should support:

```text
human_only
ra_rldm
ta_rldm
ta_rldm_armpc
```

### 4.3 Unity Project

Keep only Unity project files required to open and run the DIL scene:

```text
driver_in_loop/unity_client/Assets/
driver_in_loop/unity_client/Packages/
driver_in_loop/unity_client/ProjectSettings/
```

If a free vehicle model asset is used, include only the required imported asset files if licensing permits. Otherwise, document how to import it from Unity Asset Store.

### 4.4 Plotting Code

Keep plotting scripts required for:

- highD paper trajectory figures;
- highD authority, trust, speed, steering, and phase-plane figures;
- DIL paper trajectory figures;
- DIL authority, trust, speed, steering, and phase-plane figures;
- single-panel figure export for LaTeX assembly.

Each paper panel should be exportable as:

```text
.png
.pdf
.svg
```

### 4.5 Lightweight Demo Data

Prefer including a small processed demo dataset, not the full raw highD dataset.

The demo data should be sufficient to:

- run one highD validation case;
- run paper Case 1 and Case 2 if licensing allows;
- generate at least one trajectory figure and one authority/control figure.

If highD licensing prevents distribution, include only:

- scenario metadata;
- synthetic or anonymized demo case;
- instructions for users to place their own highD data.

## 5. Content Not to Upload

Exclude the following:

```text
raw highD full dataset
large HDF5/NPZ/PKL datasets
model checkpoints
training logs
debug logs
generated videos
temporary results
old output figures
large result folders
Unity Library/
Unity Temp/
Unity Logs/
Unity Obj/
Unity Build/
Unity UserSettings/
__pycache__/
.pytest_cache/
.ipynb_checkpoints/
local path configuration files
personal environment files
```

These should be covered by `.gitignore`.

## 6. Path Portability Requirement

The released code must not depend on local absolute paths such as:

```text
E:\NUS科研2025\code\TASE02
E:\NUS科研2025\02_our_TASE
E:\gitProjects\project_1
```

All paths should be resolved by one of the following methods.

### 6.1 Project-Root Relative Paths

Use project-root relative paths in code:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
RESULTS_DIR = PROJECT_ROOT / "results"
```

### 6.2 Configuration File

Use a default configuration file with relative paths:

```text
configs/default.yaml
```

Example:

```yaml
paths:
  demo_data: data_demo
  outputs: outputs
  results: results
  unity_client: driver_in_loop/unity_client
```

### 6.3 Local User Configuration

Provide:

```text
configs/local.example.yaml
```

Users copy it to:

```text
configs/local.yaml
```

`configs/local.yaml` should be ignored by Git.

Example:

```yaml
paths:
  highd_raw_root: "D:/datasets/highD"
  highd_processed_root: "D:/datasets/highD_processed"
```

### 6.4 Environment Variables

Support environment variables for external datasets:

```powershell
$env:HIGHD_ROOT="D:\datasets\highD"
```

Python should read:

```python
import os
from pathlib import Path

highd_root = Path(os.environ.get("HIGHD_ROOT", "data_demo/highd"))
```

## 7. Recommended Release Repository Structure

Use a clean ASCII-safe structure:

```text
project_1/
|-- README.md
|-- INSTALL.md
|-- RUN_HIGHD_EXPERIMENTS.md
|-- RUN_DIL_EXPERIMENTS.md
|-- UNITY_SETUP.md
|-- G29_SETUP.md
|-- FIGURE_GENERATION.md
|-- DATA_STRUCTURE.md
|-- RELEASE_REQUIREMENTS.md
|-- environment.yml
|-- requirements.txt
|-- .gitignore
|-- configs/
|   |-- default.yaml
|   `-- local.example.yaml
|-- scripts/
|   |-- check_environment.py
|   |-- run_highd_demo.py
|   |-- run_dil_demo.py
|   |-- plot_highd_figures.py
|   `-- plot_dil_figures.py
|-- src/
|   |-- work1_driver_prediction/
|   |-- work2_machine_intent/
|   |-- work3_authority_rl/
|   |-- work4_armpc/
|   |-- shared_validation/
|   `-- utils/
|-- driver_in_loop/
|   |-- python_backend/
|   |-- unity_client/
|   `-- docs/
|-- plot/
|-- plot_dil/
|-- data_demo/
|-- outputs/
`-- results/
```

`outputs/` and `results/` may exist with `.gitkeep`, but generated files should usually be ignored.

## 8. Environment Documentation Requirements

### 8.1 Python Environment

Document:

- operating system tested;
- Python version;
- Conda environment name;
- `environment.yml` usage;
- `requirements.txt` usage if needed;
- PyTorch CPU/GPU requirement;
- required plotting libraries;
- required communication/input libraries;
- optional `pygame` dependency for G29.

Required example commands:

```powershell
conda env create -f environment.yml
conda activate tase_highd
python scripts/check_environment.py
```

If `conda activate` fails in PowerShell, document:

```powershell
conda init powershell
```

Then restart PowerShell.

### 8.2 Unity Environment

Document:

- Unity Hub requirement;
- tested Unity Editor version;
- whether Unity 2022.3 LTS is supported;
- required Unity packages;
- required vehicle model assets;
- scene path;
- required GameObjects;
- required UDP ports.

### 8.3 Hardware Environment

Document two DIL input modes:

1. keyboard mode;
2. Logitech G29 mode.

For G29:

- install Logitech G HUB;
- connect G29;
- install `pygame`;
- run G29 inspection script;
- record steering, throttle, and brake axes.

Example:

```powershell
conda run -n tase_highd pip install pygame
conda run -n tase_highd python driver_in_loop/python_backend/tools/inspect_g29.py
```

## 9. highD Experiment Workflow Requirements

The release should provide commands for:

1. checking the environment;
2. loading or preparing demo highD cases;
3. running integrated Work1-4 validation;
4. running HTML or local visualization;
5. generating paper figures.

Example command style:

```powershell
python scripts/run_highd_demo.py --case 1
python scripts/plot_highd_figures.py --cases 1 2 3 4
```

The documentation should clarify:

- which case corresponds to which paper scenario;
- which data files are required;
- which results are generated;
- where figures are saved.

## 10. DIL Keyboard Workflow Requirements

The keyboard-based DIL workflow should allow collaborators without G29 to test the system.

Required steps:

1. open Unity project in Unity Hub;
2. open the DIL highway scene;
3. press Play;
4. start the Python backend;
5. select paper case;
6. select DIL mode;
7. use keyboard as continuous driver input;
8. save experiment log;
9. generate DIL paper figures.

Example:

```powershell
python driver_in_loop/python_backend/run_driver_in_loop.py ^
  --paper-case-id 1 ^
  --dil-mode ta_rldm_armpc ^
  --input keyboard
```

Keyboard control should be interpreted as continuous driver control intention:

- left/right keys: steering wheel intention;
- up/down keys: acceleration/braking intention.

In shared modes, keyboard input is not the final vehicle command by itself. It is fused with the machine intent through the authority decision module.

## 11. DIL G29 Workflow Requirements

The G29 workflow should be consistent with the keyboard workflow, except that driver input comes from the G29 steering wheel and pedals.

Required steps:

1. connect G29;
2. open Logitech G HUB;
3. check G29 axes;
4. open Unity DIL scene;
5. start Python backend in G29 mode;
6. run each DIL mode;
7. save logs;
8. generate figures.

Example:

```powershell
python driver_in_loop/python_backend/run_driver_in_loop.py ^
  --paper-case-id 1 ^
  --dil-mode ta_rldm_armpc ^
  --input g29
```

The G29 adapter should expose:

- steering angle;
- throttle intention;
- brake intention;
- optional reset/quit buttons.

## 12. Unity Operation Requirements

`UNITY_SETUP.md` should include:

1. how to open the Unity project;
2. which scene to open;
3. how to check `DIL_Client`;
4. how to check UDP ports;
5. how to check camera mode;
6. how to import vehicle models;
7. how to start Play mode;
8. how to verify Python-Unity communication;
9. how to troubleshoot common issues.

Required port documentation:

```text
Python -> Unity state port: 50710
Unity -> Python driver input port: 50711
```

If a port is occupied, document how to find and kill the process:

```powershell
Get-NetUDPEndpoint -LocalPort 50711 | Select-Object LocalAddress,LocalPort,OwningProcess
Stop-Process <PID> -Force
```

## 13. Figure Generation Requirements

The release should support generation of highD and DIL paper figures.

### 13.1 highD Paper Figures

- trajectory comparison figures;
- authority and bidirectional trust figures;
- steering and speed response figures;
- steering-speed-time or heat-map figures if retained;
- sideslip angle and yaw-rate phase-plane figures;
- single-panel exports for LaTeX.

### 13.2 DIL Paper Figures

For paper Case 1 and Case 2:

- human trajectory;
- machine trajectory;
- RA-RLDM trajectory;
- TA-RLDM trajectory;
- TA-RL-ARMPC trajectory;
- surrounding vehicle states;
- reference authority;
- RA-RLDM authority;
- TA-RLDM authority;
- bidirectional trust;
- speed;
- steering angle;
- sideslip angle;
- yaw rate.

Each subfigure should be exported separately:

```text
Case1_Panel_a.png/pdf/svg
Case1_Panel_b.png/pdf/svg
...
Case2_Panel_f.png/pdf/svg
```

This is required because final paper figures are assembled in LaTeX.

## 14. Required Documentation

The release should include the following documents.

### 14.1 README.md

Purpose:

- introduce the repository;
- explain highD validation and DIL validation;
- provide quick start commands.

### 14.2 INSTALL.md

Purpose:

- Python setup;
- Conda setup;
- Unity setup;
- optional G29 setup.

### 14.3 RUN_HIGHD_EXPERIMENTS.md

Purpose:

- how to run highD validation;
- how to prepare data;
- how to run each case;
- how to generate figures.

### 14.4 RUN_DIL_EXPERIMENTS.md

Purpose:

- how to run DIL keyboard experiments;
- how to run DIL G29 experiments;
- how to select case and mode;
- how to save logs.

### 14.5 UNITY_SETUP.md

Purpose:

- how to open and configure Unity;
- how to import vehicle assets;
- how to run the DIL scene.

### 14.6 G29_SETUP.md

Purpose:

- how to install Logitech G HUB;
- how to install pygame;
- how to inspect G29 axes;
- how to map axes to steering, throttle, and brake.

### 14.7 FIGURE_GENERATION.md

Purpose:

- how to generate all paper figures;
- where figures are saved;
- how to use single panels in LaTeX.

### 14.8 DATA_STRUCTURE.md

Purpose:

- explain required data structure;
- explain demo data;
- explain highD raw and processed data placement;
- explain why full highD raw data is not uploaded.

## 15. Validation Before Git Upload

Before uploading to GitHub, the clean repository should be tested from inside:

```text
E:\gitProjects\project_1
```

Required checks:

1. environment creation works;
2. environment check script passes;
3. highD demo case runs;
4. highD figures are generated;
5. Unity project opens;
6. DIL keyboard mode runs;
7. DIL G29 inspection script runs or fails gracefully if no G29 is connected;
8. DIL logs are saved;
9. DIL figures are generated;
10. no code depends on local absolute paths.

## 16. GitHub Upload Workflow

After validation:

```powershell
cd E:\gitProjects\project_1
git init
git add .
git commit -m "Initial clean release for highD and DIL validation"
git branch -M main
git remote add origin <github-repo-url>
git push -u origin main
```

If large demo data is required, consider using GitHub Releases instead of committing it directly.

## 17. Open Questions Before Building the Release Folder

The following should be confirmed during cleanup:

1. Which processed highD demo cases can legally be shared?
2. Which Unity vehicle model files can be redistributed?
3. Whether Unity 6.3 LTS is required or Unity 2022.3 LTS is also supported.
4. Whether trained Work1 checkpoints are required for reproduction or whether inference can run from saved scenario outputs.
5. Whether collaborators need to retrain models or only reproduce validation and figures.
6. Whether HTML visualization should be included as a formal workflow or only as debugging support.
7. Whether all paper figures should be generated by one command or by separate highD/DIL scripts.

