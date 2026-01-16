"""
Flask Web App for Nose Width Measurement - Using DECA
======================================================
Receives JPEG frames from frontend, processes with DECA to get 3D vertices,
measures nose width, and returns averaged results.

Usage:
    python python_app.py
    Open browser to http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import torch
import sys
import os

# Add DECA to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DECA'))

from deca_measurement import DECAMeasurement
from decalib.utils.config import cfg as deca_cfg
import face_alignment

app = Flask(__name__)

# =============================================================================
# FLAME VERTEX INDICES for measurements (same as live_nose_width.py)
# =============================================================================
# Nose width vertices
VERTEX_MIN_X = 3092  # Left alar
VERTEX_MAX_X = 2057  # Right alar

# Additional vertices for nose height
NASION = 3560        # Top of nose bridge (between eyes)
SUBNASALE = 3551     # Bottom center of nose
CHIN_CENTER = 3414   # Chin

# =============================================================================
# FLAME to mm conversion
# FLAME model uses METERS, so multiply by 1000 to get mm
# =============================================================================
FLAME_TO_MM = 1000

# =============================================================================
# MASK SIZE THRESHOLDS
# =============================================================================
MASK_CONFIGS = {
    "N10": {
        "name": "AirFit N10",
        "type": "Nasal Cradle",
        "measurement": "nose_width",
        "sizes": {
            "Small": {"min": 0, "max": 36.95},
            "Standard": {"min": 36.95, "max": 40.33},
            "Wide": {"min": 40.33, "max": 60}
        }
    },
    "N20": {
        "name": "AirFit N20",
        "type": "Nasal Mask",
        "measurement": "nose_height",
        "sizes": {
            "Small": {"min": 0, "max": 37.17},
            "Medium": {"min": 37.17, "max": 45.38},
            "Large": {"min": 45.38, "max": 60}
        }
    },
    "N30": {
        "name": "AirFit N30",
        "type": "Nasal Cradle",
        "measurement": "nose_width",
        "sizes": {
            "Small": {"min": 0, "max": 35.93},
            "Medium": {"min": 35.93, "max": 46.31},
            "Large": {"min": 46.31, "max": 60}
        }
    },
    "N30i": {
        "name": "AirFit N30i",
        "type": "Nasal Cradle",
        "measurement": "nose_width",
        "sizes": {
            "Small": {"min": 0, "max": 35.53},
            "Medium": {"min": 35.53, "max": 46.15},
            "Wide": {"min": 46.15, "max": 60}
        }
    },
    "F40": {
        "name": "AirFit F40 / Quattro Air",
        "type": "Full Face",
        "measurement": "face_height",
        "sizes": {
            "XS": {"min": 0, "max": 79.38},
            "S": {"min": 79.38, "max": 89.38},
            "M": {"min": 89.38, "max": 100.00},
            "L": {"min": 100.00, "max": 115.00}
        }
    }
}


# =============================================================================
# DECA Processor Class
# =============================================================================
class DECAProcessor:
    """DECA-based face processor for nose width measurement"""
    
    def __init__(self):
        print("="*60)
        print("Initializing DECA Processor...")
        print("="*60)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")
        
        # Initialize DECA
        deca_cfg.model.use_tex = False
        self.deca = DECAMeasurement(config=deca_cfg, device=self.device)
        
        # Initialize face detector
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=self.device
        )
        
        print("DECA Processor initialized successfully!")
        print("="*60)
    
    def detect_and_crop_face(self, frame):
        """Detect face and return cropped/resized image"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = self.fa.get_landmarks(frame_rgb)
        if landmarks is None or len(landmarks) == 0:
            return None, None
        
        lmk = landmarks[0]
        h, w = frame.shape[:2]
        
        x_min = max(0, int(np.min(lmk[:, 0])) - 30)
        x_max = min(w, int(np.max(lmk[:, 0])) + 30)
        y_min = max(0, int(np.min(lmk[:, 1])) - 50)
        y_max = min(h, int(np.max(lmk[:, 1])) + 30)
        
        size = max(x_max - x_min, y_max - y_min)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        x_min = max(0, center_x - size // 2)
        x_max = min(w, center_x + size // 2)
        y_min = max(0, center_y - size // 2)
        y_max = min(h, center_y + size // 2)
        
        face_crop = frame_rgb[y_min:y_max, x_min:x_max]
        face_resized = cv2.resize(face_crop, (224, 224))
        
        return face_resized, lmk
    
    def process_frame(self, frame):
        """
        Process a single frame and return measurements.
        Uses EXACT same logic as live_nose_width.py:
        - Get 3D FLAME vertices from DECA
        - Calculate Euclidean distance between vertex pairs
        - FLAME units are in METERS, so multiply by 1000 to get mm
        
        Returns measurements dict or None if face not detected.
        """
        face_img, landmarks = self.detect_and_crop_face(frame)
        if face_img is None:
            return None
        
        # Convert to tensor
        face_tensor = torch.tensor(face_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        face_tensor = face_tensor.to(self.device)
        
        # Get 3D vertices from DECA (same as live_nose_width.py)
        with torch.no_grad():
            codedict = self.deca.encode(face_tensor)
            opdict = self.deca.decode(codedict)
        
        verts = opdict['verts'][0].cpu().numpy()
        
        # =================================================================
        # NOSE WIDTH - Exact same logic as live_nose_width.py
        # Distance between vertices 3092 (MIN X) and 2057 (MAX X)
        # FLAME units are in METERS, multiply by 1000 to get mm
        # =================================================================
        v_min = verts[VERTEX_MIN_X]  # 3092
        v_max = verts[VERTEX_MAX_X]  # 2057
        nose_width_flame = np.linalg.norm(v_min - v_max)
        nose_width_mm = nose_width_flame * 1000  # meters to mm
        
        # =================================================================
        # NOSE HEIGHT - Nasion (3560) to Subnasale (3551)
        # =================================================================
        nasion = verts[NASION]
        subnasale = verts[SUBNASALE]
        nose_height_flame = np.linalg.norm(nasion - subnasale)
        nose_height_mm = nose_height_flame * 1000
        
        # =================================================================
        # FACE HEIGHT - Chin to inner canthus center
        # =================================================================
        chin = verts[CHIN_CENTER]
        left_inner_canthus = verts[2764]
        right_inner_canthus = verts[1163]
        inner_canthus_center = (left_inner_canthus + right_inner_canthus) / 2
        face_height_flame = np.linalg.norm(chin - inner_canthus_center)
        face_height_mm = face_height_flame * 1000
        
        # =================================================================
        # FACE WIDTH - Jaw width
        # =================================================================
        left_jaw = verts[3424]
        right_jaw = verts[3652]
        face_width_flame = np.linalg.norm(left_jaw - right_jaw)
        face_width_mm = face_width_flame * 1000
        
        return {
            'nose_width': float(nose_width_mm),
            'nose_height': float(nose_height_mm),
            'face_height': float(face_height_mm),
            'face_width': float(face_width_mm),
            'nose_width_raw': float(nose_width_flame),  # For debugging
        }


# Global DECA processor instance
deca_processor = None


def get_deca_processor():
    """Lazy initialization of DECA processor"""
    global deca_processor
    if deca_processor is None:
        deca_processor = DECAProcessor()
    return deca_processor


def calculate_stats(values):
    """Calculate statistics for measurement series"""
    if len(values) == 0:
        return {'mean': 0, 'std': 0, 'cv': 0}
    if len(values) == 1:
        return {'mean': values[0], 'std': 0, 'cv': 0}
    
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    return {
        'count': len(arr),
        'mean': float(mean),
        'std': float(std),
        'cv': float((std / mean) * 100) if mean > 0 else 0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr))
    }


def get_mask_recommendations(measurements):
    """Get mask recommendations based on measurements"""
    recommendations = []
    
    for mask_id, config in MASK_CONFIGS.items():
        meas_key = config['measurement']
        meas_val = measurements.get(meas_key, 0)
        
        for size_name, bounds in config['sizes'].items():
            if bounds['min'] <= meas_val < bounds['max']:
                # Calculate confidence
                range_size = bounds['max'] - bounds['min']
                center = (bounds['min'] + bounds['max']) / 2
                distance = abs(meas_val - center)
                
                if distance < range_size * 0.25:
                    confidence = 'high'
                elif distance < range_size * 0.4:
                    confidence = 'medium'
                else:
                    confidence = 'low'
                
                recommendations.append({
                    'id': mask_id,
                    'name': config['name'],
                    'type': config['type'],
                    'recommended_size': size_name,
                    'confidence': confidence
                })
                break
    
    # Sort by confidence
    order = {'high': 0, 'medium': 1, 'low': 2}
    recommendations.sort(key=lambda x: order.get(x['confidence'], 3))
    
    return recommendations


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frames', methods=['POST'])
def process_frames():
    """
    Process uploaded JPEG frames using DECA.
    Extracts frames, measures nose width, returns averaged results.
    """
    data = request.json
    frames_data = data.get('frames', [])
    
    if not frames_data:
        return jsonify({'error': 'No frames provided'}), 400
    
    print(f"\n{'='*60}")
    print(f"[INFO] Received {len(frames_data)} frames from client")
    print(f"{'='*60}")
    
    # Get DECA processor
    processor = get_deca_processor()
    
    # Process each frame
    all_measurements = []
    
    for i, frame_data in enumerate(frames_data):
        print(f"\n[{i+1}/{len(frames_data)}] Processing frame...")
        
        try:
            # Decode base64 image
            img_data = base64.b64decode(frame_data.split(',')[1])
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                print(f"  [ERROR] Failed to decode frame")
                continue
            
            # Process with DECA
            measurements = processor.process_frame(frame)
            
            if measurements:
                all_measurements.append(measurements)
                print(f"  [OK] Nose Width: {measurements['nose_width']:.1f} mm, "
                      f"Nose Height: {measurements['nose_height']:.1f} mm")
            else:
                print(f"  [WARN] No face detected in frame")
                
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
    
    if not all_measurements:
        return jsonify({'error': 'No valid measurements. Face may not be detected.'}), 400
    
    # Calculate averages
    keys = ['nose_width', 'nose_height', 'face_height', 'face_width']
    averages = {}
    stats = {}
    
    for key in keys:
        values = [m[key] for m in all_measurements if m.get(key)]
        if values:
            key_stats = calculate_stats(values)
            averages[key] = round(key_stats['mean'], 2)
            stats[key] = {
                'mean': round(key_stats['mean'], 2),
                'std': round(key_stats.get('std', 0), 2),
                'cv': round(key_stats.get('cv', 0), 2)
            }
    
    # Get recommendations
    recommendations = get_mask_recommendations(averages)
    
    # Print summary
    print(f"\n{'='*60}")
    print("MEASUREMENT SUMMARY (DECA)")
    print(f"{'='*60}")
    print(f"Frames Processed: {len(all_measurements)}")
    print(f"Nose Width:  {averages.get('nose_width', 'N/A')} mm")
    print(f"Nose Height: {averages.get('nose_height', 'N/A')} mm")
    print(f"Face Height: {averages.get('face_height', 'N/A')} mm")
    print(f"Face Width:  {averages.get('face_width', 'N/A')} mm")
    print(f"{'='*60}\n")
    
    return jsonify({
        'success': True,
        'frames_processed': len(all_measurements),
        'averages': averages,
        'statistics': stats,
        'recommendations': recommendations
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("MASKFIT AI - Web App (DECA Backend)")
    print("="*60)
    print("\nMeasurements using DECA + FLAME vertices:")
    print(f"  - Nose Width: vertices {VERTEX_MIN_X} to {VERTEX_MAX_X}")
    print(f"  - Nose Height: vertices {NASION} to {SUBNASALE}")
    print(f"  - FLAME units × 1000 = mm (FLAME is in meters)")
    print("\nOpen browser to: http://localhost:5000")
    print("="*60 + "\n")
    
    # Pre-initialize DECA
    get_deca_processor()
    
    app.run(debug=False, host='0.0.0.0', port=5000)
