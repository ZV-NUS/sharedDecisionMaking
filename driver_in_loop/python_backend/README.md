# Python Backend

This backend runs the driver-in-the-loop shared-control experiment.

## Run Without Unity

```powershell
conda run -n tase_highd python driver_in_loop\python_backend\run_driver_in_loop.py --paper-case-id 3 --headless --no-udp --duration-s 5
```

## Run With Keyboard Input

```powershell
conda run -n tase_highd python driver_in_loop\python_backend\run_driver_in_loop.py --paper-case-id 3 --input keyboard
```

Keyboard controls:

- Left/Right or A/D: steering
- Up/W: throttle
- Down/S: brake
- Space: full brake pulse
- R: reset
- Q: quit

## Run With Unity Visualization

1. Open the Unity project in `driver_in_loop/unity_client`.
2. Press Play in Unity.
3. If you want to control the ego vehicle from the Unity Game window, start Python with Unity input:

```powershell
conda run -n tase_highd python driver_in_loop\python_backend\run_driver_in_loop.py --paper-case-id 3 --input unity
```

Python sends state snapshots to Unity by UDP:

- Python -> Unity state: `127.0.0.1:50710`
- Unity -> Python input, optional: `127.0.0.1:50711`

If you use `--input keyboard` instead, the keyboard focus must be on the
PowerShell/Python terminal rather than the Unity Game window.

## Run With Logitech G29

The G29 adapter uses pygame/SDL and outputs the same continuous driver-intention
command as the keyboard substitute. The shared-control stack then decides the
actual vehicle command according to the selected DIL mode.

```powershell
conda run -n tase_highd python driver_in_loop\python_backend\tools\inspect_g29.py
```

Use the inspection script first to confirm the steering, throttle, and brake
axis indices. Then run:

```powershell
conda run -n tase_highd python driver_in_loop\python_backend\run_driver_in_loop.py --paper-case-id 1 --input g29 --dil-mode ta_rldm_armpc
```

Common calibration options:

- `--g29-steer-axis 0`
- `--g29-throttle-axis 2`
- `--g29-brake-axis 3`
- `--g29-invert-steer`
- `--g29-no-invert-pedals`
- `--g29-deadzone 0.03`
- `--g29-smoothing 0.35`

## Scenario Mapping

The traffic environment is fixed to the four paper validation scenarios. The
surrounding vehicles are replayed from the highD-injected rollout data, and the
ego vehicle is controlled by the live driver input.

| Option | Paper case | Rollout case |
| --- | --- | --- |
| `--paper-case-id 1` | Case 1 | case 2 |
| `--paper-case-id 2` | Case 2 | case 3 |
| `--paper-case-id 3` | Case 3 | case 6 |
| `--paper-case-id 4` | Case 4 | case 7 |
