# Unity Vehicle Model Replacement Guide

This driver-in-the-loop scene supports Unity Asset Store vehicle prefabs.

## Recommended Asset Type

Use a lightweight passenger-car prefab:

- format: Unity prefab, FBX, or GLB imported into Unity
- type: sedan / passenger car / traffic car
- polygon level: low or medium poly
- orientation: preferably forward along Unity `+Z`; the scene converts it to the simulation `+X` direction by default
- scale: arbitrary is acceptable; the script rescales to highD-like car size

Suggested Asset Store search terms:

- `low poly traffic cars`
- `realistic car pack`
- `sedan car free`
- `city traffic vehicles`

## Unity Setup

1. Import the vehicle asset into the Unity project.
2. Create two prefabs if needed:
   - `Assets/Prefabs/Vehicles/EgoVehicle.prefab`
   - `Assets/Prefabs/Vehicles/SurroundingVehicle.prefab`
3. Select `DIL_Client` in the Unity Hierarchy.
4. In `Highway Scene Controller`, assign:
   - `Ego Vehicle Prefab`
   - `Surrounding Vehicle Prefab`
5. Keep `Use Imported Vehicle Prefabs` enabled.
6. If the vehicle faces the wrong direction, adjust:
   - `Imported Vehicle Euler Offset`
   - default is `(0, 90, 0)`, which maps common Unity `+Z` vehicle models to this simulator's `+X` driving direction.

## Fallback

If no prefab is assigned, the simulator automatically uses the built-in procedural sedan model.

If the vehicle appearance does not change after code updates, this is expected unless a real Asset Store vehicle prefab
has been assigned to the two prefab slots above. The script does not download or create CARLA/PreScan-quality assets by
itself; it only provides the runtime replacement interface.

## Notes

- The algorithm and UDP interface do not depend on the vehicle mesh.
- The model is automatically scaled to the highD car dimensions sent by Python.
- Colliders are removed from imported models because collision checking is handled by the Python shared-control backend.
