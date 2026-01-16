# DECA Mask Recommender

AI-powered CPAP mask fitting system using DECA face reconstruction. This web application analyzes facial measurements through a webcam to recommend the optimal CPAP mask size for users.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)

## Features

- 🎥 **Web-based face scanning** - Records 3-second video from webcam
- 📏 **Accurate measurements** - Uses DECA/FLAME 3D face reconstruction
- 🎭 **Multiple mask recommendations** - Supports N10, N20, N30, N30i, F10, F20, F30, F30i, F40 masks
- 📊 **Statistical analysis** - Averages measurements across multiple frames for accuracy
- 💻 **Easy to use** - Simple browser-based interface

## Measurements

The system uses fixed FLAME mesh vertices for precise measurements:

| Measurement | Vertices | Purpose |
|-------------|----------|---------|
| Nose Width | V3092 → V2057 | Nasal masks (N10, N20, N30, etc.) |
| Face Height F10 | V3553 → V3487 | Quattro Air F10 mask sizing |
| Face Height F20 | V3704 → V3487 | AirFit F20 mask sizing |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/umairinayat/deca-mask-recommender.git
cd deca-mask-recommender
```

### 2. Create a virtual environment

```bash
# Create environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# For GPU support (CUDA 11.8), install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install pytorch3d (required for 3D operations)
# Windows:
pip install git+https://github.com/facebookresearch/pytorch3d.git

# Or download pre-built wheel from:
# https://github.com/facebookresearch/pytorch3d/releases
```

### 4. Download DECA model files

Download the required model files and place them in the `DECA/data/` folder:

1. **DECA model**: Download from [DECA repository](https://github.com/yfeng95/DECA)
   - `deca_model.tar` → Extract to `DECA/data/`
   
2. **FLAME model**: Download from [FLAME website](https://flame.is.tue.mpg.de/)
   - `generic_model.pkl` → Place in `DECA/data/`
   - `FLAME_masks.pkl` → Place in `DECA/data/FLAME2020/FLAME_masks/`

## Usage

### Run the Web Application

```bash
python web_app.py
```

Then open your browser to: **http://localhost:5000**

### How to use:

1. Allow camera access when prompted
2. Position your face within the guide frame
3. Click "Start Scan" button
4. Hold still for 3 seconds while recording
5. View your measurements and mask recommendations

## Supported Masks

### Nasal Masks (based on nose width)
- AirFit N10 / Swift FX Nano
- AirFit N20
- AirFit N30
- AirFit N30i

### Full Face Masks
- Quattro Air F10 (based on face height F10)
- AirFit F20 (based on face height F20)
- AirFit F30 (based on nose width)
- AirFit F30i (based on nose width)
- AirFit F40 (based on nose width)

## Project Structure

```
deca-mask-recommender/
├── web_app.py              # Main Flask web application
├── deca_measurement.py     # DECA wrapper for measurements
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web frontend
├── tools/                  # Utility scripts (see tools/README.md)
│   ├── fitmask.py          # Standalone mask fitting
│   ├── live_nose_width.py  # Live camera measurement
│   ├── python_app.py       # Desktop application
│   ├── dataset_fitmask.py  # Dataset batch processing
│   └── ...                 # Other utilities
├── DECA/                   # DECA library
│   ├── decalib/            # Core DECA code
│   ├── configs/            # Configuration files
│   └── data/               # Model files (download separately)
└── dataset/                # Test dataset (optional)
```

## Additional Tools

The `tools/` folder contains utility scripts for development and testing:

- **Live measurement tools** - Camera-based measurement scripts
- **Dataset processing** - Batch processing for multiple images
- **Y-threshold tools** - Threshold selection and validation

See `tools/README.md` for details.

## Requirements

- Python 3.9 or 3.10
- CUDA GPU (recommended) or CPU
- Webcam
- Modern web browser (Chrome, Firefox, Edge)

## License

This project uses the DECA model which is for non-commercial research purposes only. See [DECA License](DECA/LICENSE) for details.

## Acknowledgments

- [DECA](https://github.com/yfeng95/DECA) - Detailed Expression Capture and Animation
- [FLAME](https://flame.is.tue.mpg.de/) - Faces Learned with an Articulated Model and Expressions
- [MediaPipe](https://google.github.io/mediapipe/) - Face mesh detection for web interface
