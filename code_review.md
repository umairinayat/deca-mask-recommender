# CPAP Mask Measurement System - Code Review

## Project Overview

This project implements a **CPAP mask sizing system** that uses DECA (Detailed Expression Capture and Animation) for 3D face reconstruction. The goal is to accurately measure facial features (nose width, cheekbone width, nose-to-chin distance) to recommend the correct mask size (S/M/L) for various CPAP mask brands.

### Why DECA/FLAME?

Previously, an **OpenCV + ArUco marker** approach was attempted but proved unreliable:
- 2D facial landmarks don't maintain consistent real-world proportions across camera distances
- MediaPipe landmarks showed 15-20mm measurement errors
- Perspective distortion caused face width to vary from 135mm → 114mm at different distances

**DECA Solution:**
- True 3D face reconstruction from single images
- FLAME-based model with 5,023 vertices and anatomical accuracy
- Distance-independent measurements in consistent 3D coordinate system
- Pose-invariant (works with head rotation)

---

## Repository Structure

```
project_1/
├── README.md                    # Quick start guide
├── code_review.md               # This file
├── CPAP Template PDFs/          # Reference sizing guides from mask manufacturers
│   └── CPAP Template PDFs/
│       ├── airfit/              # ResMed AirFit sizing templates
│       ├── dreamwear-*.pdf      # Philips DreamWear guides
│       └── evora-*.pdf          # Fisher & Paykel guides
│
└── DECA/                        # Main working directory
    ├── cpap_measurement.py      # ⭐ MAIN: Live camera measurement capture
    ├── validator.py             # ⭐ MAIN: Statistical analysis & visualization
    ├── live_cam_demo.py         # General DECA live demo (less specific)
    ├── frontalize_face.py       # Face frontalization utility
    │
    ├── decalib/                 # ⭐ Core DECA library (modified)
    │   ├── deca.py              # Main DECA class
    │   ├── models/              # Neural network models (FLAME, encoders, decoders)
    │   ├── datasets/            # Data loading utilities
    │   └── utils/               # Helpers (config, renderer, etc.)
    │
    ├── data/                    # ⭐ Model files (required)
    │   ├── deca_model.pkl       # Pre-trained DECA model (434MB)
    │   ├── generic_model.pkl    # FLAME 2020 model (53MB)
    │   └── *.npy, *.png, etc.   # Supporting data files
    │
    ├── demos/                   # Original DECA demo scripts
    ├── configs/                 # Training configurations (not needed for inference)
    ├── tests/                   # Test files
    ├── TestSamples/             # Sample images for testing
    ├── FLAME2020/               # Additional FLAME models (male/female)
    │
    ├── requirements.txt         # Original requirements (outdated)
    ├── requirements_fixed.txt   # Fixed versions (still issues)
    ├── requirements_arm64.txt   # ARM64 Mac requirements
    │
    ├── deca_context.md          # Detailed DECA documentation
    ├── SUGGESTED_IMPROVEMENTS.md # Future enhancement ideas
    │
    └── deca_original_repo/      # ❌ REDUNDANT: Original DECA repo copy
```

---

## Key Files Explained

### Active/Essential Files

| File | Purpose | Status |
|------|---------|--------|
| `cpap_measurement.py` | Live camera capture for 3 CPAP measurements | ✅ Main tool |
| `validator.py` | Visualize & analyze measurement consistency | ✅ Main tool |
| `live_cam_demo.py` | General-purpose DECA live demo | ✅ Working |
| `frontalize_face.py` | Convert angled faces to frontal view | ✅ Working |
| `decalib/` | Core DECA library | ✅ Required |
| `data/` | Model files (.pkl, .npy) | ✅ Required |

### Redundant/Can Be Removed

| File/Folder | Reason |
|-------------|--------|
| `deca_original_repo/` | Complete duplicate of DECA - kept for reference but not needed |
| `logs/` | Training logs from original repo - not needed |
| `Detailed_Expression_Capture_and_Animation.ipynb` | Tutorial notebook - optional |
| `main_train.py` | Training script - not needed for inference |
| `Dockerfile`, `docker-compose.yml` | Docker setup - not needed locally |
| `fetch_data.sh`, `install_*.sh`, `launch.sh` | Linux scripts - not needed on Windows |
| `requirements_fixed.txt` | Outdated/incorrect versions |
| `requirements_arm64.txt` | ARM64-specific - not needed on Windows |

---

## The 3 CPAP Measurements

The system captures these anatomical measurements using FLAME vertex indices:

### 1. Nose Width (Alar Base)
- **Vertices**: 3632 ↔ 3325
- **Use**: Primary sizing for nasal/pillow masks
- **Brands**: ResMed AirFit N20/P10, F&P Brevida, Philips DreamWear

### 2. Cheekbone Width (Zygion-Zygion)
- **Vertices**: 4478 ↔ 2051
- **Use**: Primary sizing for full-face masks
- **Brands**: ResMed AirFit F20/F30, F&P Simplus/Vitera

### 3. Nose-to-Chin Distance (Subnasale → Menton)
- **Vertices**: 175 ↔ 152
- **Use**: Secondary check for full-face cushion height

---

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CPAP Measurement Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌─────────────┐    ┌──────────┐    ┌─────────┐ │
│  │ Camera  │ ─► │ Face Detect │ ─► │  DECA    │ ─► │ Extract │ │
│  │ Frame   │    │ (FAN)       │    │ Encode/  │    │ Vertices│ │
│  │ 640x480 │    │ 68 landmarks│    │ Decode   │    │ (5023)  │ │
│  └─────────┘    └─────────────┘    └──────────┘    └────┬────┘ │
│                                                          │      │
│  ┌───────────────────────────────────────────────────────▼────┐ │
│  │                    Measurement Extraction                   │ │
│  │  • vertices[3632] - vertices[3325] → Nose Width            │ │
│  │  • vertices[4478] - vertices[2051] → Cheekbone Width       │ │
│  │  • vertices[175] - vertices[152] → Nose-to-Chin            │ │
│  └───────────────────────────────────────────────────────┬────┘ │
│                                                          │      │
│  ┌───────────────────────────────────────────────────────▼────┐ │
│  │                      Output (JSON)                          │ │
│  │  results/<session>/measurement_<timestamp>.json             │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Dependencies Overview

### Core Requirements

| Package | Purpose | Notes |
|---------|---------|-------|
| `torch` | Deep learning framework | Need ≥1.8.0, CUDA optional |
| `pytorch3d` | 3D rendering | Tricky to install on Windows |
| `face-alignment` | Face detection (FAN) | Auto-downloads models |
| `numpy` | Array operations | ≤1.23 for chumpy compatibility |
| `opencv-python` | Camera/image handling | Standard |
| `chumpy` | FLAME model loading | Requires older numpy |
| `kornia` | Image transformations | ≥0.4.0 |

### Installation Challenges

1. **PyTorch3D**: Notoriously difficult on Windows
   - No official Windows wheels
   - Requires Visual Studio Build Tools
   - Alternatively: use `--rasterizer_type=pytorch3d` flag

2. **NumPy Version**: Must use ≤1.23.x
   - `chumpy` uses deprecated `numpy.bool` which was removed in numpy 1.24+

3. **CUDA**: Optional but recommended for speed
   - CPU works but ~2-3 seconds per frame
   - CUDA reduces to ~0.1-0.2 seconds

---

## Current Status

### ✅ Working
- DECA model initialization and inference
- Face detection and preprocessing
- 3D vertex extraction
- Measurement calculation (in FLAME units)
- Session-based result saving
- Statistical validation and visualization

### 🔄 Needs Work
- **FLAME to millimeter conversion**: Currently outputs FLAME units, needs calibration
- **Size recommendation**: No mapping to S/M/L yet
- **GPU acceleration**: Configured for CPU, CUDA available but not tested

---

## Recommended Cleanup

### Files/Folders to Delete

```bash
# Redundant - complete duplicate of DECA
DECA/deca_original_repo/

# Training logs - not needed
DECA/logs/

# Linux scripts - not needed on Windows
DECA/*.sh

# Docker files - not using Docker
DECA/Dockerfile
DECA/docker-compose.yml

# Outdated requirements
DECA/requirements_fixed.txt
DECA/requirements_arm64.txt

# Training configs/code - not needed for inference
DECA/main_train.py
DECA/configs/

# Optional tutorial
DECA/Detailed_Expression_Capture_and_Animation.ipynb
```

### Keep These

```bash
# Core functionality
DECA/cpap_measurement.py
DECA/validator.py
DECA/live_cam_demo.py
DECA/frontalize_face.py
DECA/decalib/
DECA/data/
DECA/demos/

# Documentation
DECA/README.md
DECA/deca_context.md
DECA/SUGGESTED_IMPROVEMENTS.md

# Testing
DECA/tests/
DECA/TestSamples/  # At least keep some examples
```

---

## Quick Start (After Setup)

```bash
# Activate environment
conda activate deca-env

# Navigate to DECA folder
cd DECA

# Run live camera measurement
python cpap_measurement.py
# Press SPACE to capture, ESC to quit

# Validate results
python validator.py
```

---

## Next Steps

1. **Set up clean Python environment** (see setup instructions below)
2. **Test live camera demo** to verify DECA works
3. **Calibrate measurements** - convert FLAME units to millimeters
4. **Add size recommendations** - map measurements to S/M/L per brand
5. **Clean up redundant files** to reduce repo size

---

## Summary

This is a well-structured project that uses DECA/FLAME for accurate 3D facial measurements. The core functionality is working, with the main files being:

- `cpap_measurement.py` - Main measurement capture tool
- `validator.py` - Results validation/visualization
- `decalib/` - Core DECA library

The main redundancy is the `deca_original_repo/` folder (complete duplicate) and various unused training/Docker files. After cleanup, the project should be much leaner while maintaining all essential functionality.

