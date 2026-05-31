# Driver-in-the-Loop Experiment Workflow

## DIL Modes

The backend supports:

```text
human_only
ra_rldm
ta_rldm
ta_rldm_armpc
```

## Keyboard Mode

Keyboard input represents continuous driver intention:

- left/right: steering intention;
- up/down: acceleration/braking intention.

Run with Unity communication:

```powershell
python driver_in_loop/python_backend/run_driver_in_loop.py --paper-case-id 1 --dil-mode ta_rldm_armpc --input unity
```

Run a headless smoke test without Unity:

```powershell
python scripts/run_dil_demo.py
```

Run the full keyboard matrix for paper Case 1 and Case 2 across all modes:

```powershell
python scripts/run_release_smoke_tests.py
```

## G29 Mode

After configuring G29 axes:

```powershell
python driver_in_loop/python_backend/run_driver_in_loop.py --paper-case-id 1 --dil-mode ta_rldm_armpc --input g29
```

## Experiment Logs

DIL logs are saved under:

```text
driver_in_loop/experiments/
```

This folder is ignored by Git because it contains generated experiment data.
