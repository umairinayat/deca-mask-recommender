# CPAP Mask Measurement System

3D facial measurement system for CPAP mask sizing using DECA (Detailed Expression Capture and Animation).

## Quick Start

### 1. Activate DECA Environment
```bash
conda activate deca-env
cd DECA
```

### 2. Run Measurement Capture
```bash
python cpap_measurement.py
```
- Press **SPACE** to capture measurements
- Press **ESC** to quit
- Results saved to `../results/<session_index>/`

### 3. Validate Results
```bash
# Validate latest session
python validator.py

# Validate specific session
python validator.py 1
```

## What It Measures

### 3 Critical CPAP Measurements:

1. **Nose Width (Alar Base)**
   - FLAME vertices: 3632 ↔ 3325
   - Primary sizing for nasal & pillow masks
   - Brands: ResMed AirFit N20/P10, F&P Brevida, Philips DreamWear

2. **Cheekbone Width (Zygion-Zygion)**
   - FLAME vertices: 4478 ↔ 2051
   - Primary sizing for full-face masks
   - Brands: ResMed AirFit F20/F30, F&P Simplus/Vitera

3. **Nose-to-Chin Distance (Subnasale → Menton)**
   - FLAME vertices: 175 ↔ 152
   - Secondary check for full-face cushion size

## Output Format

### JSON Structure
```json
{
  "timestamp": "2025-11-23T00:45:30.123456",
  "measurement_number": 1,
  "measurements": {
    "nose_width": 0.045123,
    "cheekbone_width": 0.290956,
    "nose_to_chin": 0.182337
  },
  "processing_time_seconds": 2.34,
  "vertex_indices": {
    "nose_left": 3632,
    "nose_right": 3325,
    "cheek_left": 4478,
    "cheek_right": 2051,
    "nose_base": 175,
    "chin": 152
  }
}
```

### Validator Output
- **Statistics**: Mean, std, CV% for each measurement
- **Graphs**: 3 trend plots showing measurement consistency
- **Assessment**: Consistency rating (Excellent <2%, Good <5%)

## Project Structure

```
facial_project/
├── DECA/                           # DECA folder (all DECA-related files)
│   ├── cpap_measurement.py        # Main CPAP measurement capture
│   ├── validator.py                # Visualization & statistics
│   ├── temp.py                     # Quick consistency test
│   ├── live_cam_demo.py            # General DECA demo
│   ├── deca_context.md             # DECA setup documentation
│   ├── data/                       # DECA models
│   │   ├── deca_model.pkl
│   │   ├── generic_model.pkl
│   │   └── ...
│   └── decalib/                    # DECA library code
├── results/                        # Measurement sessions (outside DECA)
│   ├── 1/                          # Session 1
│   │   ├── measurement_*.json
│   │   └── session_1_visualization.png
│   └── 2/                          # Session 2
├── context.md                      # Project overview
└── README.md                       # This file
```

## Key Features

✅ **Distance-Independent**: FLAME units consistent across camera distances  
✅ **3D Accuracy**: True anatomical measurements from 3D reconstruction  
✅ **Pose-Robust**: Works with slight head rotation  
✅ **Session Management**: Automatic indexing and organization  
✅ **Validation Tools**: Statistical analysis and visualization

## Next Steps

1. **Calibration**: Convert FLAME units to millimeters using reference measurements
2. **Testing**: Validate distance independence by measuring at different distances
3. **Integration**: Map measurements to brand-specific sizing charts (S/M/L)

## Documentation

- `context.md` - Project overview and problem statement
- `DECA/deca_context.md` - Detailed DECA setup and technical documentation
