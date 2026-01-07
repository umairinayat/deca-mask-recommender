# DECA Setup Guide for Windows

This guide will help you set up the DECA environment for the CPAP mask measurement system on Windows.

## Prerequisites

- **Conda** (Anaconda or Miniconda) - [Download here](https://docs.conda.io/en/latest/miniconda.html)
- **Visual Studio Build Tools** (for pytorch3d) - [Download here](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- **Git** (optional)

## Quick Setup (Recommended)

### Step 1: Open Anaconda Prompt

**Important**: Use **Anaconda Prompt** (not PowerShell) to avoid script execution issues.
- Search for "Anaconda Prompt" in Windows Start Menu
- Or run `cmd.exe` and then `conda activate base`

### Step 2: Navigate to Project

```cmd
cd D:\Job\project_1\project_1\project_1\DECA
```

### Step 3: Create Conda Environment

```cmd
conda create -n deca-env python=3.10 -y
conda activate deca-env
```

### Step 4: Install PyTorch

**For CPU only:**
```cmd
pip install torch torchvision torchaudio
```

**For CUDA 11.8 (if you have NVIDIA GPU):**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 5: Install Dependencies

```cmd
pip install -r requirements_windows.txt
```

### Step 6: Install PyTorch3D

PyTorch3D is tricky on Windows. Try these methods in order:

**Method A: Pre-built wheel (easiest)**
```cmd
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html
```

**Method B: From GitHub (if Method A fails)**
```cmd
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

**Method C: Build from source (last resort)**
```cmd
# Requires Visual Studio Build Tools installed
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install -e .
cd ..
```

**Method D: Skip pytorch3d (fallback)**
If all else fails, DECA can use its built-in rasterizer (slower but works):
- The code will use CUDA rasterizer if pytorch3d is unavailable
- Make sure you have `ninja` installed: `pip install ninja`

### Step 7: Verify Installation

```cmd
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import face_alignment; print('face_alignment: OK')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

Test DECA:
```cmd
python -c "from decalib.deca import DECA; print('DECA import: OK')"
```

### Step 8: Run Live Demo

```cmd
python live_cam_demo.py
```

Or for CPAP measurements:
```cmd
python cpap_measurement.py
```

---

## Troubleshooting

### Problem: "conda is not recognized"
**Solution**: Use Anaconda Prompt instead of PowerShell/CMD, or add conda to PATH:
```cmd
set PATH=%PATH%;C:\Users\<username>\anaconda3\Scripts;C:\Users\<username>\anaconda3
```

### Problem: PowerShell execution policy error
**Solution**: Use CMD or Anaconda Prompt. If you must use PowerShell:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Problem: "numpy.bool" error
**Solution**: NumPy version is too high. Downgrade:
```cmd
pip install "numpy>=1.22.0,<1.24.0"
```

### Problem: pytorch3d installation fails
**Solutions**:
1. Make sure Visual Studio Build Tools is installed
2. Try a different Python version (3.9 or 3.10 work best)
3. Use the fallback rasterizer (code handles this automatically)

### Problem: "No module named 'chumpy'"
**Solution**:
```cmd
pip install chumpy
```

### Problem: Face not detected
**Solutions**:
- Ensure good lighting
- Face the camera directly
- First run downloads face detection models (~200MB) - wait for it

### Problem: CUDA out of memory
**Solution**: Use CPU instead:
```python
# In the code, device is auto-detected
# To force CPU, modify the device line:
self.device = 'cpu'
```

---

## Required Model Files

Make sure these files exist in `DECA/data/`:

| File | Size | Source |
|------|------|--------|
| `deca_model.pkl` | ~434MB | Pre-trained DECA model |
| `generic_model.pkl` | ~53MB | FLAME 2020 model |
| `head_template.obj` | Small | Head mesh template |
| `mean_texture.jpg` | Small | Average face texture |
| `*.npy` files | Various | Supporting data |

If missing, download from:
- DECA model: [Google Drive link in original repo](https://github.com/YadiraF/DECA)
- FLAME model: [FLAME website](https://flame.is.tue.mpg.de/) (requires registration)

---

## Verifying Setup Works

Run this test script:

```cmd
cd D:\Job\project_1\project_1\project_1\DECA
python -c "
import torch
import cv2
import numpy as np

print('=== Environment Check ===')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'OpenCV: {cv2.__version__}')
print(f'NumPy: {np.__version__}')

try:
    import pytorch3d
    print(f'PyTorch3D: {pytorch3d.__version__}')
except ImportError:
    print('PyTorch3D: Not installed (will use fallback)')

try:
    import face_alignment
    print('face_alignment: OK')
except ImportError:
    print('face_alignment: MISSING')

try:
    from decalib.deca import DECA
    print('DECA import: OK')
except Exception as e:
    print(f'DECA import: FAILED - {e}')

print('========================')
"
```

---

## GPU vs CPU Performance

| Device | Processing Time | Notes |
|--------|-----------------|-------|
| CPU | ~2-3 seconds/frame | Works on any machine |
| CUDA (GPU) | ~0.1-0.2 seconds/frame | Requires NVIDIA GPU |

For live camera demo, GPU is recommended for smoother experience.

---

## Commands Summary

```cmd
# Activate environment
conda activate deca-env

# Run CPAP measurement (main tool)
python cpap_measurement.py

# Run general live demo
python live_cam_demo.py

# Validate measurement results
python validator.py

# Test on single image
python demos/demo_reconstruct.py -i TestSamples/examples --saveObj True
```

