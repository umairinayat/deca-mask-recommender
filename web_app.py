"""
Web App for Mask Fitting - DECA-Based
======================================
Client records 3-second video, server extracts frames,
processes with DECA using fixed FLAME vertices, and returns
mask recommendations.

Measurements:
  - Nose Width: V3092 → V2057
  - Face Height F10: V3553 → V3487
  - Face Height F20: V3704 → V3487

Usage:
    python web_app.py
    Open browser to http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import torch
import sys
import os
import base64

# Add DECA to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DECA'))

from deca_measurement import DECAMeasurement
from decalib.utils.config import cfg as deca_cfg
import face_alignment

app = Flask(__name__)


# =============================================================================
# FLAME VERTEX INDICES (from live_nose_width.py)
# =============================================================================

# Nose width vertices
VERTEX_MIN_X = 3092  # Left alar
VERTEX_MAX_X = 2057  # Right alar

# Face height vertices for Quattro Air F10
QUATTRO_AIR_UP = 3553
QUATTRO_AIR_DOWN = 3487

# Face height vertices for AirFit F20
AIRFIT_F20_UP = 3704
AIRFIT_F20_DOWN = 3487


# =============================================================================
# MASK CONFIGURATIONS (from dataset_fitmask.py)
# =============================================================================

MASK_CONFIGS = {
    "N10": {
        "name": "AirFit N10 / Swift FX Nano",
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
        "measurement": "nose_width",
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
            "Medium-Small": {"min": 46.31, "max": 60}
        }
    },
    "N30i": {
        "name": "AirFit N30i",
        "type": "Nasal Cradle",
        "measurement": "nose_width",
        "sizes": {
            "Small": {"min": 0, "max": 35.53},
            "Medium": {"min": 35.53, "max": 46.15},
            "Small Wide": {"min": 46.15, "max": 48.76},
            "Wide": {"min": 48.76, "max": 60}
        }
    },
    "F30": {
        "name": "AirFit F30",
        "type": "Nose",
        "measurement": "nose_width",
        "sizes": {
            "Small": {"min": 0, "max": 33.85},
            "Medium": {"min": 33.85, "max": 47.38},
        }
    },
    "F10": {
        "name": "Quattro Air F10",
        "type": "Full Face",
        "measurement": "face_height_f10",
        "sizes": {
            "X-Small": {"min": 0, "max": 79.38},
            "Small": {"min": 79.38, "max": 89.38},
            "Medium": {"min": 89.38, "max": 100.00},
            "Large": {"min": 100.00, "max": 115.00}
        }
    },
    "F20": {
        "name": "AirFit F20",
        "type": "Full Face",
        "measurement": "face_height_f20",
        "sizes": {
            "Small": {"min": 0, "max": 87.28},
            "Medium": {"min": 87.28, "max": 99.29},
            "Large": {"min": 99.29, "max": 111.60}
        }
    },
    "F40": {
        "name": "AirFit F40",
        "type": "Full Face",
        "measurement": "nose_width",
        "sizes": {
            "Medium": {"min": 0, "max": 44.52},
            "Small Wide": {"min": 44.52, "max": 48.39},
            "Large": {"min": 48.39, "max": 60}
        }
    },
    "F30i": {
        "name": "AirFit F30i",
        "type": "Full Face",
        "measurement": "nose_width",
        "sizes": {
            "Small": {"min": 0, "max": 40.34},
            "Medium": {"min": 40.34, "max": 45.37},
            "Small Wide": {"min": 45.37, "max": 49.83},
            "Wide": {"min": 49.83, "max": 58.56}
        }
    },
}


# =============================================================================
# DECA PROCESSOR (Global - initialized once)
# =============================================================================

class DECAProcessor:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        print("=" * 60)
        print("Initializing DECA Processor...")
        print("=" * 60)
        
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
        
        print("DECA initialized successfully!")
        print("=" * 60)
    
    def detect_and_crop_face(self, frame):
        """Detect face and crop for DECA input."""
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
        
        crop_info = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
        
        face_crop = frame_rgb[y_min:y_max, x_min:x_max]
        face_resized = cv2.resize(face_crop, (224, 224))
        
        return face_resized, crop_info
    
    def process_frame(self, frame):
        """Process a single frame through DECA and return vertices."""
        face_img, crop_info = self.detect_and_crop_face(frame)
        
        if face_img is None:
            return None
        
        face_tensor = torch.tensor(face_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        face_tensor = face_tensor.to(self.device)
        
        with torch.no_grad():
            codedict = self.deca.encode(face_tensor)
            opdict = self.deca.decode(codedict)
        
        verts = opdict['verts'][0].cpu().numpy()
        
        return verts
    
    def measure(self, verts):
        """Calculate all measurements from FLAME vertices."""
        # Nose width (in FLAME units → mm)
        v_min = verts[VERTEX_MIN_X]
        v_max = verts[VERTEX_MAX_X]
        nose_width_mm = np.linalg.norm(v_min - v_max) * 1000
        
        # Face height F10 (Quattro Air)
        f10_up = verts[QUATTRO_AIR_UP]
        f10_down = verts[QUATTRO_AIR_DOWN]
        face_height_f10 = np.linalg.norm(f10_up - f10_down) * 1000
        
        # Face height F20 (AirFit F20)
        f20_up = verts[AIRFIT_F20_UP]
        f20_down = verts[AIRFIT_F20_DOWN]
        face_height_f20 = np.linalg.norm(f20_up - f20_down) * 1000
        
        return {
            'nose_width': float(nose_width_mm),
            'face_height_f10': float(face_height_f10),
            'face_height_f20': float(face_height_f20),
        }


def get_mask_size(measurement_mm, sizes):
    """Determine mask size based on measurement."""
    for size_name, bounds in sizes.items():
        if bounds["min"] <= measurement_mm < bounds["max"]:
            return size_name
    return "Unknown"


def get_mask_recommendations(measurements):
    """Get mask recommendations based on measurements."""
    recommendations = []
    
    for mask_id, config in MASK_CONFIGS.items():
        meas_key = config['measurement']
        meas_val = measurements.get(meas_key, 0)
        
        for size_name, bounds in config['sizes'].items():
            if bounds['min'] <= meas_val < bounds['max']:
                # Calculate confidence based on distance from center
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
    
    # Sort by confidence (high first)
    order = {'high': 0, 'medium': 1, 'low': 2}
    recommendations.sort(key=lambda x: order.get(x['confidence'], 3))
    
    return recommendations


def calculate_stats(values):
    """Calculate statistics for a measurement series."""
    if len(values) < 2:
        return {'mean': values[0] if values else 0, 'std': 0, 'cv': 0}
    
    arr = np.array(values)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    
    return {
        'count': len(arr),
        'mean': float(mean),
        'std': float(std),
        'cv': float((std / mean) * 100) if mean > 0 else 0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frames', methods=['POST'])
def process_frames():
    """
    Process uploaded JPEG frames using DECA.
    Client sends base64-encoded JPEG frames.
    Server processes each with DECA, averages measurements,
    and returns mask recommendations.
    """
    try:
        data = request.json
        frames_data = data.get('frames', [])
        
        if not frames_data:
            return jsonify({'error': 'No frames provided'}), 400
        
        print(f"\n{'='*60}")
        print(f"[INFO] Received {len(frames_data)} frames from client")
        print(f"{'='*60}")
        
        # Get DECA processor (singleton)
        try:
            processor = DECAProcessor.get_instance()
        except Exception as e:
            print(f"[ERROR] Failed to get DECA processor: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'DECA initialization failed: {str(e)}'}), 500
        
        # Process each frame
        all_measurements = []
        
        for i, frame_data in enumerate(frames_data):
            print(f"\n[{i+1}/{len(frames_data)}] Processing frame...", end=" ")
            
            try:
                # Decode base64 image
                img_data = base64.b64decode(frame_data.split(',')[1])
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if frame is None:
                    print("[ERROR] Failed to decode")
                    continue
                
                # Process with DECA
                verts = processor.process_frame(frame)
                
                if verts is None:
                    print("[NO FACE]")
                    continue
                
                # Get measurements
                measurements = processor.measure(verts)
                all_measurements.append(measurements)
                
                print(f"[OK] Nose: {measurements['nose_width']:.1f}mm | "
                      f"F10: {measurements['face_height_f10']:.1f}mm | "
                      f"F20: {measurements['face_height_f20']:.1f}mm")
                
            except Exception as e:
                print(f"[ERROR] Frame processing failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not all_measurements:
            return jsonify({'error': 'No valid measurements. Face may not be detected.'}), 400
        
        # Calculate averages
        keys = ['nose_width', 'face_height_f10', 'face_height_f20']
        averages = {}
        stats = {}
        
        for key in keys:
            values = [m[key] for m in all_measurements if m.get(key)]
            if values:
                key_stats = calculate_stats(values)
                averages[key] = round(key_stats['mean'], 2)
                stats[key] = {
                    'mean': round(key_stats['mean'], 2),
                    'std': round(key_stats['std'], 2),
                    'cv': round(key_stats['cv'], 2)
                }
        
        # Get recommendations
        recommendations = get_mask_recommendations(averages)
        
        # Summary
        print(f"\n{'='*60}")
        print("MEASUREMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Frames Processed: {len(all_measurements)}")
        print(f"Nose Width:      {averages.get('nose_width', 'N/A')} mm (CV: {stats.get('nose_width', {}).get('cv', 'N/A')}%)")
        print(f"Face Height F10: {averages.get('face_height_f10', 'N/A')} mm (CV: {stats.get('face_height_f10', {}).get('cv', 'N/A')}%)")
        print(f"Face Height F20: {averages.get('face_height_f20', 'N/A')} mm (CV: {stats.get('face_height_f20', {}).get('cv', 'N/A')}%)")
        print(f"\nMask Recommendations:")
        for rec in recommendations:
            print(f"  {rec['id']} ({rec['name']}): {rec['recommended_size']} [{rec['confidence']}]")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'frames_processed': len(all_measurements),
            'averages': averages,
            'statistics': stats,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Request processing failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("MASKFIT AI - DECA Web App")
    print("=" * 60)
    print("\nMeasurements (Fixed FLAME Vertices):")
    print(f"  - Nose Width: V{VERTEX_MIN_X} → V{VERTEX_MAX_X}")
    print(f"  - Face Height F10: V{QUATTRO_AIR_UP} → V{QUATTRO_AIR_DOWN}")
    print(f"  - Face Height F20: V{AIRFIT_F20_UP} → V{AIRFIT_F20_DOWN}")
    print("\nMask Recommendations:")
    print("  - Nasal: N10, N20, N30, N30i")
    print("  - Full Face: F10, F20, F30, F30i, F40")
    print("\nOpen browser to: http://localhost:5000")
    print("=" * 60 + "\n")
    
    # Pre-initialize DECA
    DECAProcessor.get_instance()
    
    app.run(debug=False, host='0.0.0.0', port=5000)
