# highD Validation Workflow

## Included Lightweight Workflow

The repository includes the rollout file:

```text
outputs/shared_authority_validation/shared_authority_rollouts.js
```

This file supports:

- local HTML visualization;
- paper trajectory figure generation;
- Unity DIL backend case injection.

Generate the main highD trajectory paper figure:

```powershell
python scripts/plot_highd_figures.py
```

The output is saved under:

```text
results/Case1_2_Trajectory_Figures/
```

The current highD figure workflow generates four paper cases from the included rollout file.

## HTML Visualization

Start a local static server:

```powershell
python -m http.server 8766 --directory outputs/shared_authority_validation
```

Open:

```text
http://127.0.0.1:8766/realtime.html
```

## Full highD Dataset

The full highD dataset is not distributed in this repository. If full preprocessing or retraining is needed, obtain highD separately and configure:

```powershell
$env:HIGHD_ROOT="D:\datasets\highD"
```

or copy `configs/local.example.yaml` to `configs/local.yaml` and set your local paths.
