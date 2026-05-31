# Unity Setup

## Open Project

Use Unity Hub to open:

```text
driver_in_loop/unity_client
```

Open the DIL highway scene or use the `TASE` menu to create the scene if the scene is not present.

## Ports

Default UDP ports:

```text
Python -> Unity state port: 50710
Unity -> Python driver input port: 50711
```

If a port is occupied:

```powershell
Get-NetUDPEndpoint -LocalPort 50711 | Select-Object LocalAddress,LocalPort,OwningProcess
Stop-Process <PID> -Force
```

## Typical Run Order

1. Open Unity project.
2. Open the DIL scene.
3. Press Play.
4. Start the Python backend.
5. Confirm that the HUD receives states and the vehicle moves.

The backend uses a handshake mechanism, so starting Python before Unity should also work, but starting Unity first is easier for debugging.

