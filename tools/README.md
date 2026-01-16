# Tools

This folder contains utility scripts and standalone tools for mask fitting and measurement. These are not required for the main web application but are useful for development, testing, and alternative use cases.

## Scripts

| Script | Description |
|--------|-------------|
| `fitmask.py` | Standalone mask fitting with live camera |
| `live_nose_width.py` | Live camera measurement with DECA |
| `python_app.py` | Desktop application with PyGame UI |
| `dataset_fitmask.py` | Batch process dataset with fixed vertices |
| `dataset_fitmask_threshold.py` | Batch process with Y-threshold method |
| `select_y_threshold.py` | Interactive tool to select Y-threshold value |
| `validate_y_threshold.py` | Validate Y-threshold selection |
| `batch_fitmask.py` | Batch processing utility |
| `live_cam_raw_vertex.py` | Raw vertex visualization for debugging |
| `nose_width_v1609.py` | Legacy nose width measurement script |

## Usage

Run from the project root directory:

```bash
# From project root
python tools/live_nose_width.py
python tools/fitmask.py
python tools/dataset_fitmask.py
```

## Note

These scripts use `deca_measurement.py` from the parent directory. Make sure to run them from the project root.
