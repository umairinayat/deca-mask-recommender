#!/usr/bin/env python3
"""
BATCH MASK FIT PROCESSING
=========================
Process test images and output results with measurement labels.

Usage:
    python batch_fitmask.py
"""
import cv2
import numpy as np
import torch
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deca_measurement import DECAMeasurement
from decalib.utils.config import cfg as deca_cfg
import face_alignment


# =============================================================================
# FLAME VERTEX DEFINITIONS
# =============================================================================

VERTEX_MIN_X = 3092  # Left alar nose
VERTEX_MAX_X = 2057  # Right alar nose

# Face height vertices
QUATTRO_AIR_UP = 3553
QUATTRO_AIR_DOWN = 3487

AIRFIT_F20_UP = 3572
AIRFIT_F20_DOWN = 3487


# =============================================================================
# MASK CONFIGURATIONS (Updated from fitmask.py)
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
            "X-Small": {"min": 0, "max": 79.38},
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
# BATCH PROCESSOR
# =============================================================================

class BatchMaskFit:
    def __init__(self):
        print("="*60)
        print("BATCH MASK FIT PROCESSING")
        print("="*60)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {self.device}")
        
        deca_cfg.model.use_tex = False
        self.deca = DECAMeasurement(config=deca_cfg, device=self.device)
        
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=self.device
        )
        
        print("Initialized successfully!")
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
        for size_name, bounds in sizes.items():
            if bounds["min"] <= measurement_mm < bounds["max"]:
                return size_name
        return "Unknown"
    
    def measure(self, verts):
        v_min = verts[VERTEX_MIN_X]
        v_max = verts[VERTEX_MAX_X]
        nose_width_mm = np.linalg.norm(v_min - v_max) * 1000
        
        quattro_up = verts[QUATTRO_AIR_UP]
        quattro_down = verts[QUATTRO_AIR_DOWN]
        face_height_f10 = np.linalg.norm(quattro_up - quattro_down) * 1000
        
        f20_up = verts[AIRFIT_F20_UP]
        f20_down = verts[AIRFIT_F20_DOWN]
        face_height_f20 = np.linalg.norm(f20_up - f20_down) * 1000
        
        return {
            'nose_width': nose_width_mm,
            'face_height_f10': face_height_f10,
            'face_height_f20': face_height_f20,
        }
    
    def get_all_recommendations(self, measurements):
        recommendations = {}
        
        for mask_id, config in MASK_CONFIGS.items():
            meas_key = config['measurement']
            meas_val = measurements.get(meas_key, 0)
            
            size = self.get_mask_size(meas_val, config['sizes'])
            recommendations[mask_id] = {
                'name': config['name'],
                'type': config['type'],
                'size': size
            }
        
        return recommendations
    
    def create_output_image(self, frame, verts, trans_verts, crop_info, filename):
        """Create output image with measurements and recommendations"""
        # Scale image if too small
        min_height = 800
        h, w = frame.shape[:2]
        if h < min_height:
            scale = min_height / h
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            # Adjust crop_info for scaling
            crop_info = {
                'x_min': int(crop_info['x_min'] * scale),
                'x_max': int(crop_info['x_max'] * scale),
                'y_min': int(crop_info['y_min'] * scale),
                'y_max': int(crop_info['y_max'] * scale)
            }
        
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Get measurements
        measurements = self.measure(verts)
        recommendations = self.get_all_recommendations(measurements)
        
        # Project nose points
        x1, y1 = self.project_to_image(trans_verts, crop_info, VERTEX_MIN_X)
        x2, y2 = self.project_to_image(trans_verts, crop_info, VERTEX_MAX_X)
        
        # Project F10 points
        qx1, qy1 = self.project_to_image(trans_verts, crop_info, QUATTRO_AIR_UP)
        qx2, qy2 = self.project_to_image(trans_verts, crop_info, QUATTRO_AIR_DOWN)
        
        # Project F20 points
        fx1, fy1 = self.project_to_image(trans_verts, crop_info, AIRFIT_F20_UP)
        fx2, fy2 = self.project_to_image(trans_verts, crop_info, AIRFIT_F20_DOWN)
        
        # Draw measurement lines
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.circle(vis, (x1, y1), 5, (0, 0, 255), -1)
        cv2.circle(vis, (x2, y2), 5, (0, 0, 255), -1)
        
        cv2.line(vis, (qx1, qy1), (qx2, qy2), (0, 255, 0), 3)
        cv2.circle(vis, (qx1, qy1), 5, (0, 255, 0), -1)
        cv2.circle(vis, (qx2, qy2), 5, (0, 255, 0), -1)
        
        cv2.line(vis, (fx1, fy1), (fx2, fy2), (255, 0, 0), 3)
        cv2.circle(vis, (fx1, fy1), 5, (255, 0, 0), -1)
        cv2.circle(vis, (fx2, fy2), 5, (255, 0, 0), -1)
        
        # Create info panel at top
        panel_height = 200
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background
        
        # Title
        cv2.putText(panel, f"Image: {filename}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Measurements
        y_pos = 60
        cv2.putText(panel, f"MEASUREMENTS:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 25
        cv2.putText(panel, f"Nose Width: {measurements['nose_width']:.1f} mm", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(panel, f"Face Height (F10): {measurements['face_height_f10']:.1f} mm", (300, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(panel, f"Face Height (F20): {measurements['face_height_f20']:.1f} mm", (600, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Recommendations
        y_pos += 35
        cv2.putText(panel, f"MASK RECOMMENDATIONS:", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Nasal masks
        y_pos += 25
        nasal_recs = []
        for mask_id in ['N10', 'N20', 'N30', 'N30i', 'F30']:
            rec = recommendations.get(mask_id)
            if rec and rec['size'] != 'Unknown':
                nasal_recs.append(f"{mask_id}:{rec['size']}")
        cv2.putText(panel, f"Nasal: {' | '.join(nasal_recs)}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Full face masks
        y_pos += 25
        ff_recs = []
        for mask_id in ['F10', 'F20', 'F40', 'F30i']:
            rec = recommendations.get(mask_id)
            if rec and rec['size'] != 'Unknown':
                ff_recs.append(f"{mask_id}:{rec['size']}")
        cv2.putText(panel, f"Full Face: {' | '.join(ff_recs)}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Combine panel and image
        output = np.vstack([panel, vis])
        
        return output, measurements, recommendations
    
    def process_images(self, input_dir, output_dir):
        """Process all images in input directory"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        image_files = sorted(list(input_path.glob('*.jpg')) + list(input_path.glob('*.png')))
        
        print(f"\nFound {len(image_files)} images to process")
        print("="*60)
        
        results = []
        
        for img_file in image_files:
            print(f"\nProcessing: {img_file.name}")
            
            frame = cv2.imread(str(img_file))
            if frame is None:
                print(f"  [ERROR] Failed to load image")
                continue
            
            verts, trans_verts, crop_info = self.process_frame(frame)
            
            if verts is None:
                print(f"  [WARNING] No face detected, skipping")
                continue
            
            output_img, measurements, recommendations = self.create_output_image(
                frame, verts, trans_verts, crop_info, img_file.name
            )
            
            # Save output
            output_file = output_path / f"result_{img_file.name}"
            cv2.imwrite(str(output_file), output_img)
            print(f"  [OK] Saved: {output_file.name}")
            
            # Print summary
            print(f"  Nose Width: {measurements['nose_width']:.1f} mm")
            print(f"  Face Height (F10): {measurements['face_height_f10']:.1f} mm")
            print(f"  Face Height (F20): {measurements['face_height_f20']:.1f} mm")
            
            results.append({
                'file': img_file.name,
                'measurements': measurements,
                'recommendations': recommendations
            })
        
        # Print summary table
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'Image':<12} {'Nose(mm)':<10} {'F10(mm)':<10} {'F20(mm)':<10}")
        print("-"*60)
        for r in results:
            m = r['measurements']
            print(f"{r['file']:<12} {m['nose_width']:<10.1f} {m['face_height_f10']:<10.1f} {m['face_height_f20']:<10.1f}")
        print("="*60)
        
        return results


if __name__ == "__main__":
    input_dir = "images_for_test"
    output_dir = "results_fitmask"
    
    processor = BatchMaskFit()
    results = processor.process_images(input_dir, output_dir)
    
    print(f"\nProcessing complete! Results saved to: {output_dir}/")
