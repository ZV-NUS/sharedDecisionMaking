# Logitech G29 Setup

## Install Dependencies

Install Logitech G HUB and connect the wheel.

Install `pygame` in the Python environment:

```powershell
conda run -n tase_highd pip install pygame
```

## Inspect Axes

```powershell
conda run -n tase_highd python driver_in_loop/python_backend/tools/inspect_g29.py
```

Record the axis indices for:

- steering;
- throttle;
- brake.

Then pass them to the backend if the defaults are not correct:

```powershell
python driver_in_loop/python_backend/run_driver_in_loop.py --input g29 --g29-steer-axis 0 --g29-throttle-axis 2 --g29-brake-axis 3
```

