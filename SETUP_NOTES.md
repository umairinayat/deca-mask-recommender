# Live Nose Width Measurement Setup

## Changes Made

### 1. Fixed Import Structure

The `live_nose_width.py` script was failing with `ModuleNotFoundError: No module named 'decalib'`.

**Root Cause**: The `deca_measurement.py` file was at the project root but used relative imports (`.models`, `.utils`) designed for being inside a package.

**Solution**:

#### `deca_measurement.py` (at project root)
Changed relative imports to absolute imports:
```python
# Before (relative - broken)
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .utils import util
...

# After (absolute - working)
from decalib.models.encoders import ResnetEncoder
from decalib.models.FLAME import FLAME, FLAMETex
from decalib.utils import util
...
```

#### `live_nose_width.py`
Updated import to use local file:
```python
# Before
from decalib.deca_measurement import DECAMeasurement

# After
from deca_measurement import DECAMeasurement
```

---

## Project Structure

```
project_1/
├── live_nose_width.py      # Main script - live camera nose measurement
├── deca_measurement.py     # DECAMeasurement class (measurement-only DECA)
├── DECA/                   # DECA library (DO NOT MODIFY)
│   ├── decalib/            # Core library
│   ├── data/               # FLAME model files
│   └── configs/            # Configuration files
└── ...
```

---

## Key Files

| File | Purpose |
|------|---------|
| `live_nose_width.py` | Live camera feed with nose width measurement |
| `deca_measurement.py` | Lightweight DECA class without rendering (measurement only) |

---

## How It Works

1. **Camera captures frame** → face_alignment detects face
2. **Face cropped & resized** → 224x224 tensor
3. **DECA encodes** → extracts shape/expression/pose parameters
4. **DECA decodes** → generates 3D FLAME mesh (5023 vertices)
5. **Measure distance** between vertex 3092 and 2057 (nose width)

---

## Production Requirements

### Required:
- `live_nose_width.py`
- `deca_measurement.py`
- `DECA/decalib/` (core library)
- `DECA/data/` (FLAME models)
- `DECA/configs/` (configuration)

### Dependencies:
- PyTorch
- face_alignment
- OpenCV
- NumPy

---

## Note
> **IMPORTANT**: Never create or modify files inside the `DECA/` folder. 
> All custom code should be placed in the project root (`project_1/`).
