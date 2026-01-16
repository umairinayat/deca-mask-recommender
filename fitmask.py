#!/usr/bin/env python3
"""
MASK FIT MEASUREMENT
====================
Live face measurement for CPAP mask fitting recommendations.

Measurements:
1. Nose Width: vertices 3092 to 2057 (for nasal masks)
2. Face Height (Quattro Air F10): vertices 3553 to 3487
3. Face Height (AirFit F20): vertices 3704 to 3487

Press SPACE to capture and measure
Press ESC to exit
"""
import cv2
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deca_measurement import DECAMeasurement
from decalib.utils.config import cfg as deca_cfg
import face_alignment


# =============================================================================
# FLAME VERTEX DEFINITIONS
# =============================================================================

# Nose width vertices
VERTEX_MIN_X = 3092  # Left alar
VERTEX_MAX_X = 2057  # Right alar

# Face height vertices for masks (from masks_min_max.txt)
# Quattro Air F10: up(3553) down(3487)
QUATTRO_AIR_UP = 3553
QUATTRO_AIR_DOWN = 3487

# AirFit F20: up(3572) down(3487)
AIRFIT_F20_UP = 3572
AIRFIT_F20_DOWN = 3487


# =============================================================================
# MASK CONFIGURATIONS
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
    "F10": {
        "name": "AirFit F10",
        "type": "Full Face",
        "measurement": "face_height_f10",
        "sizes": {
            "Extra Small": {"min": 0, "max": 79.38},
            "Small": {"min": 79.38, "max": 89.38},
            "Medium": {"min": 89.38, "max": 100.00},
            "Large": {"min": 100.00, "max": 115.00}
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
# MASK FIT CLASS
# =============================================================================

class MaskFit:
    def __init__(self):
        print("="*60)
        print("MASK FIT MEASUREMENT")
        print("="*60)
        print(f"Nose Width: V{VERTEX_MIN_X} to V{VERTEX_MAX_X}")
        print(f"Face Height (F10/F40): V{QUATTRO_AIR_UP} to V{QUATTRO_AIR_DOWN}")
        print(f"Face Height (F20): V{AIRFIT_F20_UP} to V{AIRFIT_F20_DOWN}")
        print("="*60)
        
        # Initialize DECA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")
        
        deca_cfg.model.use_tex = False
        self.deca = DECAMeasurement(config=deca_cfg, device=self.device)
        
        # Initialize face detector
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=self.device
        )
        
        self.processing = False
        self.last_result = None
        self.last_measurements = None
        
        print("="*60)
        print("Ready! Press SPACE to measure, ESC to exit")
        print("="*60)
    
    def detect_and_crop_face(self, frame):
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
        face_img, crop_info = self.detect_and_crop_face(frame)
        if face_img is None:
            return None, None, None
        
        face_tensor = torch.tensor(face_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        face_tensor = face_tensor.to(self.device)
        
        with torch.no_grad():
            codedict = self.deca.encode(face_tensor)
            opdict = self.deca.decode(codedict)
        
        verts = opdict['verts'][0].cpu().numpy()
        trans_verts = opdict['trans_verts'][0].cpu().numpy()
        
        return verts, trans_verts, crop_info
    
    def project_to_image(self, trans_verts, crop_info, idx):
        v = trans_verts[idx]
        x_224 = (v[0] + 1) * 112
        y_224 = (v[1] + 1) * 112
        
        crop_w = crop_info['x_max'] - crop_info['x_min']
        crop_h = crop_info['y_max'] - crop_info['y_min']
        
        x_orig = int(x_224 * crop_w / 224 + crop_info['x_min'])
        y_orig = int(y_224 * crop_h / 224 + crop_info['y_min'])
        
        return x_orig, y_orig
    
    def get_mask_size(self, measurement_mm, sizes):
        """Determine mask size based on measurement"""
        for size_name, bounds in sizes.items():
            if bounds["min"] <= measurement_mm < bounds["max"]:
                return size_name
        return "Unknown"
    
    def get_all_recommendations(self, measurements):
        """Get recommendations for all masks based on measurements"""
        recommendations = []
        
        for mask_id, config in MASK_CONFIGS.items():
            meas_key = config['measurement']
            meas_val = measurements.get(meas_key, 0)
            
            size = self.get_mask_size(meas_val, config['sizes'])
            if size != "Unknown":
                recommendations.append({
                    'id': mask_id,
                    'name': config['name'],
                    'type': config['type'],
                    'measurement_mm': meas_val,
                    'recommended_size': size
                })
        
        return recommendations
    
    def measure(self, verts):
        """Calculate all measurements from vertices"""
        # Nose width
        v_min = verts[VERTEX_MIN_X]
        v_max = verts[VERTEX_MAX_X]
        nose_width = np.linalg.norm(v_min - v_max)
        nose_width_mm = nose_width * 1000
        
        # Face height for F10/F40 (Quattro Air)
        quattro_up = verts[QUATTRO_AIR_UP]
        quattro_down = verts[QUATTRO_AIR_DOWN]
        face_height_f10 = np.linalg.norm(quattro_up - quattro_down) * 1000
        
        # Face height for F20
        f20_up = verts[AIRFIT_F20_UP]
        f20_down = verts[AIRFIT_F20_DOWN]
        face_height_f20 = np.linalg.norm(f20_up - f20_down) * 1000
        
        return {
            'nose_width': nose_width_mm,
            'face_height_f10': face_height_f10,
            'face_height_f20': face_height_f20,
        }
    
    def visualize(self, frame, verts, trans_verts, crop_info):
        vis = frame.copy()
        
        # Get measurements
        measurements = self.measure(verts)
        
        # Project nose points to image
        x1, y1 = self.project_to_image(trans_verts, crop_info, VERTEX_MIN_X)
        x2, y2 = self.project_to_image(trans_verts, crop_info, VERTEX_MAX_X)
        
        # Project F10/F40 points
        qx1, qy1 = self.project_to_image(trans_verts, crop_info, QUATTRO_AIR_UP)
        qx2, qy2 = self.project_to_image(trans_verts, crop_info, QUATTRO_AIR_DOWN)
        
        # Project F20 points
        fx1, fy1 = self.project_to_image(trans_verts, crop_info, AIRFIT_F20_UP)
        fx2, fy2 = self.project_to_image(trans_verts, crop_info, AIRFIT_F20_DOWN)
        
        # Draw nose width line (RED)
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(vis, (x1, y1), 4, (0, 0, 255), -1)
        cv2.circle(vis, (x2, y2), 4, (0, 0, 255), -1)
        
        # Draw F10/F40 face height line (GREEN)
        cv2.line(vis, (qx1, qy1), (qx2, qy2), (0, 255, 0), 2)
        cv2.circle(vis, (qx1, qy1), 4, (0, 255, 0), -1)
        cv2.circle(vis, (qx2, qy2), 4, (0, 255, 0), -1)
        
        # Draw F20 face height line (BLUE)
        cv2.line(vis, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
        cv2.circle(vis, (fx1, fy1), 4, (255, 0, 0), -1)
        cv2.circle(vis, (fx2, fy2), 4, (255, 0, 0), -1)
        
        # Display measurements on image
        y_offset = 30
        cv2.putText(vis, f"Nose Width: {measurements['nose_width']:.1f} mm", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25
        cv2.putText(vis, f"F10/F40 Height: {measurements['face_height_f10']:.1f} mm", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(vis, f"F20 Height: {measurements['face_height_f20']:.1f} mm", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return vis, measurements
    
    def print_recommendations(self, measurements):
        """Print all mask recommendations"""
        recommendations = self.get_all_recommendations(measurements)
        
        print(f"\n{'='*60}")
        print("MEASUREMENTS")
        print(f"{'='*60}")
        print(f"Nose Width:        {measurements['nose_width']:.1f} mm")
        print(f"Face Height (F10): {measurements['face_height_f10']:.1f} mm")
        print(f"Face Height (F20): {measurements['face_height_f20']:.1f} mm")
        
        print(f"\n{'='*60}")
        print("MASK RECOMMENDATIONS")
        print(f"{'='*60}")
        
        # Group by type
        nasal_masks = [r for r in recommendations if 'Nasal' in r['type'] or r['type'] == 'Nose']
        full_face_masks = [r for r in recommendations if r['type'] == 'Full Face']
        
        if nasal_masks:
            print("\n--- NASAL MASKS ---")
            for rec in nasal_masks:
                print(f"  {rec['name']}")
                print(f"    -> Size: {rec['recommended_size']}")
        
        if full_face_masks:
            print("\n--- FULL FACE MASKS ---")
            for rec in full_face_masks:
                print(f"  {rec['name']}")
                print(f"    -> Size: {rec['recommended_size']}")
        
        print(f"\n{'='*60}\n")
    
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            
            # Show processing indicator or ready status
            if self.processing:
                cv2.putText(display, "PROCESSING...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(display, "SPACE: measure | ESC: exit", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show last measurement summary on camera view
            if self.last_measurements is not None:
                m = self.last_measurements
                cv2.putText(display, f"Nose: {m['nose_width']:.1f}mm", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow('Camera', display)
            
            # Show result visualization if available
            if self.last_result is not None:
                cv2.imshow('Measurements', self.last_result)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and not self.processing:
                self.processing = True
                
                # Update display to show processing
                display = frame.copy()
                cv2.putText(display, "PROCESSING...", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Camera', display)
                cv2.waitKey(1)
                
                # Process
                verts, trans_verts, crop_info = self.process_frame(frame)
                
                if verts is not None:
                    self.last_result, self.last_measurements = self.visualize(frame, verts, trans_verts, crop_info)
                    self.print_recommendations(self.last_measurements)
                else:
                    print("No face detected")
                
                self.processing = False
            
            elif key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = MaskFit()
    app.run()
