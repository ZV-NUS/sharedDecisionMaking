# Logitech G29 Force Feedback Setup

This project includes an optional Unity-side force-feedback controller for Logitech wheels. It is disabled by default so that the Unity client can compile without the Logitech SDK.

## What Was Added

- `driver_in_loop/unity_client/Assets/Scripts/LogitechForceFeedbackController.cs`
- The component reads the real-time DIL state from `DilUdpClient`.
- Force-feedback intensity is computed from:
  - shared authority,
  - environment risk,
  - human-to-machine trust,
  - steering conflict between driver input and shared-control output.

## Enable Force Feedback

1. Install Logitech G HUB and make sure the G29/G923 wheel is recognized by Windows.
2. Import the Logitech Steering Wheel SDK into the Unity project.
3. Place the SDK C# wrapper and native DLLs under Unity `Assets/Plugins/Logitech/`.
4. In Unity, open:

   `Edit -> Project Settings -> Player -> Other Settings -> Scripting Define Symbols`

5. Add:

   `LOGITECH_STEERING_WHEEL_SDK`

6. Select `DIL_Client` in the scene.
7. In `Logitech Force Feedback Controller`, set:

   `Enable Force Feedback = true`

8. Press Play in Unity and start the Python backend, for example:

   ```powershell
   python driver_in_loop\python_backend\run_driver_in_loop.py --paper-case-id 1 --input g29 --dil-mode ta_rldm_armpc
   ```

## Force Feedback Meaning

- Higher environment risk increases centering and damping.
- Lower human-to-machine trust increases intervention feedback.
- Steering conflict creates a directional guidance force toward the shared-control command.
- Pure human mode should use lighter feedback; shared-control modes can use stronger feedback.

## Tuning Parameters

Tune these in the Unity Inspector:

- `Base Spring Percent`
- `Base Damper Percent`
- `Max Spring Percent`
- `Max Damper Percent`
- `Max Guidance Force Percent`
- `Trust Sensitivity`
- `Risk Sensitivity`
- `Steering Conflict Scale`

If the wheel feels too heavy, lower `Max Spring Percent`, `Max Damper Percent`, and `Max Guidance Force Percent`.
