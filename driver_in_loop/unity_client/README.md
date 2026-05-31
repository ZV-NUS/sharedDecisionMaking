# Unity Client for Driver-in-the-Loop Visualization

Unity Hub is installed at `E:\Unity\Unity Hub\Unity Hub.exe`. The Unity Editor was not detected yet. Install an LTS editor in Unity Hub before opening this folder as a Unity project.

Recommended editor:

- Unity 2022.3 LTS
- Windows Build Support

## Project Setup

1. Open Unity Hub.
2. Install Unity Editor 2022.3 LTS if it is not installed yet.
3. Open this folder as an existing Unity project:

   `driver_in_loop/unity_client`

4. Wait until Unity finishes importing scripts.
5. In the Unity top menu, click:

   `TASE -> Create Driver-in-the-Loop Scene`

   This creates `Assets/Scenes/DILHighway.unity`, the camera, road materials,
   UDP client, HUD, and scene controller automatically.

6. Press Play.
7. Start the Python backend. To control the ego vehicle from the Unity Game
   window, use Unity input mode:

```powershell
conda run -n tase_highd python driver_in_loop\python_backend\run_driver_in_loop.py --paper-case-id 3 --input unity
```

The Unity view will receive ego and surrounding-vehicle states through UDP port `50710`.
Unity sends keyboard driver input back to Python through UDP port `50711`.

The visualization uses the same four highD-injected traffic cases as the paper
figures. Surrounding vehicles are replayed from the selected case; only the ego
vehicle is updated from live driver input and the shared-control algorithm.

## Coordinate Convention

Python sends highD-style positions:

- `x`: longitudinal road coordinate
- `y`: lateral road coordinate
- `yaw`: heading angle in radians

Unity displays:

- Unity `X` = highD `x`
- Unity `Z` = negative highD `y`
- Unity `Y` = vertical height

## Optional Unity Keyboard Input

The scene includes `UnityDriverInputSender`, so the Game window can directly
send driver input to Python. Use:

```powershell
conda run -n tase_highd python driver_in_loop\python_backend\run_driver_in_loop.py --paper-case-id 3 --input unity
```

Controls in the Unity Game window:

- Left/Right or A/D: steering
- Up/W: throttle
- Down/S: brake
- Space: full brake
- R: reset
- Q: quit Python backend
