# Figure Generation

## highD Figures

```powershell
python scripts/plot_highd_figures.py
```

Outputs:

```text
results/Case1_2_Trajectory_Figures/
```

## DIL Figures

First run DIL experiments for the required paper cases and modes. Then:

```powershell
python scripts/plot_dil_figures.py
```

Outputs:

```text
results/DIL_Experiment_Figures/Case1_2/
```

The DIL plotting script exports both combined figures and single panels for LaTeX assembly.

Single trajectory panels are saved as:

```text
results/DIL_Experiment_Figures/Case1_2/trajectory/panels/
```

Single authority/control/stability panels are saved as:

```text
results/DIL_Experiment_Figures/Case1_2/summary/panels/
```

