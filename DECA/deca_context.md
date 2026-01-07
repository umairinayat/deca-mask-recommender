# DECA 3D Face Reconstruction Setup & Integration

## Project Context

This project develops a **CPAP mask sizing system** that captures 3 critical facial measurements to determine mask size (S/M/L) across all major brands (ResMed, Fisher & Paykel, Philips). We use DECA (Detailed Expression Capture and Animation) for **true 3D face reconstruction** to achieve distance-independent, anatomically accurate measurements.

## The 3 Critical CPAP Measurements

Based on industry sizing standards, mask size is determined by **3 key facial dimensions**:

### 1. Nose Width (Alar Base)
- **FLAME Vertices**: 3632 ↔ 3325
- **Measurement**: Distance between left and right nostrils
- **Primary for**: Nasal masks and pillow masks
- **Brands**: ResMed AirFit N20/P10, Fisher & Paykel Brevida, Philips DreamWear
- **Why critical**: Determines seal fit at nostril interface

### 2. Cheekbone Width (Zygion-to-Zygion)
- **FLAME Vertices**: 4478 ↔ 2051
- **Measurement**: Distance between left and right cheekbones (widest part of face)
- **Primary for**: Full-face masks
- **Brands**: ResMed AirFit F20/F30, Fisher & Paykel Simplus/Vitera, Philips DreamWear Full
- **Why critical**: Determines cushion width and seal coverage

### 3. Nose-to-Chin Distance (Subnasale → Menton)
- **FLAME Vertices**: 175 ↔ 152
- **Measurement**: Vertical distance from nose base to chin tip
- **Secondary for**: Full-face mask height verification
- **Why critical**: Confirms cushion height and helps distinguish between nasal vs pillow suitability

**Note**: Mask type (nasal/pillow/full-face) is a user preference based on lifestyle, not determined by face measurements. These 3 measurements only determine the SIZE (S/M/L) within the chosen mask type.

## Problem with Previous Approach

### MediaPipe + ArUco Issues:
- **Inconsistent face width**: 135mm → 114mm when moving closer to camera
- **Landmark drift**: 2D landmarks don't scale proportionally with distance
- **Perspective distortion**: 2D projection fails with pose variations
- **15-20mm measurement errors**: Unacceptable for mask sizing precision

### Root Cause:
MediaPipe landmarks (normalized coordinates) don't maintain consistent real-world proportions across camera distances.

## DECA Solution

### What is DECA?
- **3D face reconstruction** from single images
- **FLAME-based model**: 5,023 vertex mesh with anatomical accuracy
- **Distance-independent**: Measurements in consistent 3D coordinate system
- **Pose-invariant**: Works with head rotation and tilting

## DECA Setup Process

### 1. Environment Setup
```bash
# Create conda environment
conda create -n deca-env python=3.8 -y
conda activate deca-env

# Install dependencies (ARM64 compatible)
pip install -r DECA/requirements_arm64.txt
pip install pytorch3d  # Built from source for ARM64
pip install chumpy numpy==1.22.0  # Version compatibility fixes
```

### 2. Model Downloads
- **DECA Model**: `deca_model.pkl` (434MB) - Pre-trained reconstruction model
- **FLAME Model**: `generic_model.pkl` (53MB) - Requires FLAME registration at https://flame.is.tue.mpg.de/
- **Face Detection**: Auto-downloaded on first run (s3fd, 2DFAN4 models)

### 3. Directory Structure
```
DECA/
├── data/
│   ├── deca_model.pkl          # Main DECA model (434MB)
│   ├── generic_model.pkl       # FLAME 2020 model (53MB)
│   ├── head_template.obj       # Template mesh
│   ├── mean_texture.jpg        # Average face texture
│   ├── fixed_displacement_256.npy  # Displacement data
│   └── ...other data files
├── decalib/                    # DECA library code
└── demos/                      # Example scripts
```

### 4. Key Dependencies
- **PyTorch**: 2.4.1 (with CUDA support if available)
- **PyTorch3D**: 0.7.8 (built from source for ARM64)
- **NumPy**: 1.22.0 (compatibility with chumpy)
- **face-alignment**: For face detection and landmark extraction
- **chumpy**: Required for FLAME model loading

## DECA Input/Output Specification

### Input Requirements
```python
# Image tensor: [batch_size, 3, 224, 224]
# - RGB format (not BGR)
# - Normalized to [0, 1] range
# - Face should be cropped and centered
# - 224x224 resolution (DECA standard)
```

### DECA Processing Pipeline
1. **Face Detection**: Locate face in image
2. **Face Cropping**: Extract and align face region
3. **Encoding**: Image → FLAME parameters
4. **Decoding**: FLAME parameters → 3D mesh

### Output Structure

#### 1. Code Dictionary (Encoded Parameters)
```python
codedict = {
    'shape': [1, 100],      # Identity parameters (who you are)
    'exp': [1, 50],         # Expression parameters (facial expression)
    'pose': [1, 6],         # Head pose (rotation, translation)
    'tex': [1, 50],         # Texture parameters
    'cam': [1, 3],          # Camera parameters
    'light': [1, 9, 3],     # Lighting parameters
    'detail': [1, 128]      # Fine detail parameters
}
```

#### 2. Output Dictionary (3D Reconstruction)
```python
opdict = {
    'verts': [1, 5023, 3],           # 3D vertices (main mesh)
    'landmarks3d': [1, 68, 4],       # 3D facial landmarks
    'landmarks3d_world': [1, 68, 3], # World coordinate landmarks
    'rendered_images': [1, 3, 224, 224], # Rendered face view
    'normals': [1, 5023, 3],         # Vertex normals
    'uv_texture': [1, 3, 256, 256]   # UV texture map
}
```

## Using DECA for CPAP Measurements

### Key Output: `verts` - 3D Vertices
- **Shape**: [1, 5023, 3] - 5,023 vertices with (x, y, z) coordinates
- **Coordinate System**: FLAME canonical space
- **Units**: FLAME units (require calibration to mm)

### Extracting the 3 CPAP Measurements
```python
vertices = opdict['verts'][0].cpu().numpy()  # Shape: (5023, 3)

# 1. Nose Width (Alar Base) - Primary for nasal/pillow masks
nose_left = vertices[3632]   # Left nostril
nose_right = vertices[3325]  # Right nostril
nose_width = np.linalg.norm(nose_left - nose_right)

# 2. Cheekbone Width (Zygion-Zygion) - Primary for full-face masks
cheek_left = vertices[4478]  # Left cheekbone
cheek_right = vertices[2051] # Right cheekbone
cheekbone_width = np.linalg.norm(cheek_left - cheek_right)

# 3. Nose-to-Chin Distance - Secondary check
nose_base = vertices[175]    # Subnasale (nose base)
chin = vertices[152]         # Menton (chin point)
nose_to_chin = np.linalg.norm(nose_base - chin)
```

### Advantages for CPAP Mask Sizing
1. **Distance Independence**: FLAME measurements stay consistent regardless of camera distance
2. **Pose Invariance**: Works with head rotation/tilting
3. **Exact Vertex Correspondence**: Known anatomical landmarks (not arbitrary 2D points)
4. **3D Accuracy**: True 3D geometry vs 2D projection
5. **Anatomical Consistency**: FLAME model based on 33,000+ real face scans
6. **Reproducibility**: Same person = same FLAME measurements every time

## Integration with Existing System

### Calibration Strategy
1. **DECA Measurement**: Get face width in FLAME units
2. **ArUco Calibration**: Use existing ArUco system for scale reference
3. **Conversion Factor**: Calculate FLAME units → millimeters
4. **Validation**: Test consistency across distances

### Current Workflow
```python
# 1. Capture image
image = capture_frame()

# 2. DECA reconstruction  
codedict = deca.encode(image_tensor)
opdict, visdict = deca.decode(codedict)

# 3. Extract 3 CPAP measurements
vertices = opdict['verts'][0].cpu().numpy()

nose_width = np.linalg.norm(vertices[3632] - vertices[3325])
cheekbone_width = np.linalg.norm(vertices[4478] - vertices[2051])
nose_to_chin = np.linalg.norm(vertices[175] - vertices[152])

# 4. Save measurements (in FLAME units for now)
save_to_json({
    'nose_width': nose_width,
    'cheekbone_width': cheekbone_width,
    'nose_to_chin': nose_to_chin
})

# Future: Convert to millimeters and map to S/M/L
# nose_width_mm = nose_width * flame_to_mm_factor
# mask_size = determine_size(nose_width_mm, mask_type='nasal')
```

## Current Status

### ✅ Completed
- DECA environment setup and model installation (Python 3.8, PyTorch3D, FLAME 2020)
- Successful 3D face reconstruction from images
- Identified exact FLAME vertices for 3 CPAP measurements
- Built `cpap_measurement.py` - Live camera capture system with session management
- Built `validator.py` - Statistical analysis and visualization tool
- Built `temp.py` - Quick consistency testing for face width measurements

### 📊 Current Implementation
**Measurement System (`cpap_measurement.py`)**:
- Live camera feed with DECA 3D reconstruction
- Captures 3 measurements per SPACE key press
- Auto-saves to `results/<session_index>/measurement_*.json`
- Session management with auto-incrementing indices

**Validation System (`validator.py`)**:
- Loads measurements from session directory
- Calculates statistics: mean, std, CV% for each measurement
- Generates 3 graphs showing measurement trends
- Consistency assessment (Excellent <2%, Good <5%)

**Output Format**:
```json
{
  "timestamp": "2025-11-23T00:45:30.123456",
  "measurement_number": 1,
  "measurements": {
    "nose_width": 0.045123,
    "cheekbone_width": 0.290956,
    "nose_to_chin": 0.182337
  },
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

### 🔄 Next Steps
1. **Validate consistency** - Test CV% across different distances and poses
2. **FLAME-to-millimeter calibration** - Establish conversion factor using reference measurements
3. **Build sizing database** - Create lookup tables mapping measurements to S/M/L for major brands
4. **Distance independence testing** - Verify measurements stay consistent at varying camera distances
5. **Production integration** - Web interface for end-user face scanning

## Technical Notes

### Performance Considerations
- **CPU vs GPU**: Currently configured for CPU (ARM64 compatibility)
- **Processing Time**: ~2-3 seconds per frame on CPU
- **Memory Usage**: ~2GB RAM for model loading

### Limitations
- **Face Detection Dependency**: Requires clear, well-lit face images
- **Single Face**: Processes one face per image
- **Frontal Bias**: Best results with near-frontal face poses

### Troubleshooting
- **NumPy Version**: Must use 1.22.0 for chumpy compatibility
- **PyTorch3D**: Built from source for ARM64 Macs
- **Model Path**: Ensure `deca_model.pkl` and `generic_model.pkl` in `/data/`

## File Structure
```
facial_project/
├── DECA/                           # DECA folder (all DECA-related files)
│   ├── cpap_measurement.py        # Main CPAP measurement capture system
│   ├── validator.py                # Measurement validation & visualization
│   ├── temp.py                     # Quick face width consistency test
│   ├── live_cam_demo.py            # DECA demo (general purpose)
│   ├── deca_context.md             # This file - DECA setup documentation
│   ├── data/                       # DECA models and data
│   │   ├── deca_model.pkl         # Pre-trained DECA model
│   │   ├── generic_model.pkl      # FLAME 2020 model
│   │   └── ...other data files
│   └── decalib/                    # DECA library code
├── results/                        # Measurement sessions (outside DECA)
│   ├── 1/                          # Session 1
│   │   ├── measurement_*.json     # Individual measurements
│   │   └── session_1_visualization.png
│   └── 2/                          # Session 2
├── context.md                      # Project overview
└── README.md                       # Quick start guide
```

## Usage

### Capture Measurements
```bash
conda activate deca-env
cd DECA
python cpap_measurement.py
# Press SPACE to capture, ESC to quit
```

### Validate Results
```bash
cd DECA
python validator.py          # Latest session
python validator.py 1        # Specific session
```

### Quick Consistency Test
```bash
cd DECA
python temp.py              # Test face width CV%
```

This setup provides the foundation for accurate, distance-independent CPAP mask sizing using state-of-the-art 3D face reconstruction technology.
