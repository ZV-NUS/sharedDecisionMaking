# Driver-in-the-Loop Experiment

This folder contains a real-time driver-in-the-loop experiment scaffold that is separate from the offline highD validation pipeline.

The current implementation supports:

- highD-derived surrounding-vehicle replay from `outputs/shared_authority_validation/shared_authority_rollouts.js`;
- keyboard driver input as a temporary replacement for Logitech G29;
- a reserved G29 adapter interface;
- real-time shared-control simulation in Python;
- UDP JSON communication with a Unity visualization client;
- CSV logging for experiment analysis.

## Quick Start

From the repository root:

```powershell
conda run -n tase_highd python driver_in_loop\python_backend\run_driver_in_loop.py --paper-case-id 3 --input unity
```

Useful keys in keyboard mode:

- Left/Right arrows: steering command
- Up arrow: throttle
- Down arrow: brake
- Space: brake pulse
- R: reset
- Q: quit

When `--input unity` is used, press these keys in the Unity Game window. When
`--input keyboard` is used, press them in the PowerShell/Python terminal.

If Unity is not connected, the Python backend still runs and logs data. Unity can later subscribe to the UDP state stream.

## Architecture

```text
Keyboard or G29 input
        |
        v
DriverInputAdapter
        |
        v
RealtimeSharedControlRunner
  - highD surrounding traffic replay
  - human/machine control coupling
  - authority and trust replay interface
  - MPC-lite compatible vehicle update
        |
        v
UDP state stream -> Unity front-view visualization
        |
        v
CSV experiment log
```

## Paper Scenario Mapping

The driver-in-the-loop experiment reuses the same highD-injected traffic cases
used by the paper figures. Surrounding vehicles are replayed from these cases,
while the ego vehicle is controlled online by the human input interface and the
shared-control algorithm.

| Paper case | highD-injected rollout case | Scenario role |
| --- | --- | --- |
| Case 1 | case 2 | Unsafe left lane; shared policy selects right lane |
| Case 2 | case 3 | Reasonable left intention; aggressive human operation is smoothed |
| Case 3 | case 6 | Parallel-driving risk is high; shared policy follows human deceleration |
| Case 4 | case 7 | Parallel-driving risk can be resolved; acceleration is allowed |

Use `--paper-case-id 1`, `--paper-case-id 2`, `--paper-case-id 3`, or
`--paper-case-id 4` to select these scenarios.

## Folder Structure

```text
driver_in_loop/
  python_backend/
    input_adapters/
    logging/
    network/
    scenarios/
    shared_control/
    run_driver_in_loop.py
  unity_client/
    Assets/Scripts/
    README.md
  experiments/
```

## Notes

The present version is designed for software validation before a physical G29 wheel is available. The `G29InputAdapter` is intentionally provided as a replaceable module with the same output format as the keyboard adapter.
