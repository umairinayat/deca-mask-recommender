#!/usr/bin/env python3
"""
DATASET MASK FIT PROCESSING WITH Y-THRESHOLD METHOD
=====================================================
Process dataset with multiple persons, each having multiple selfie images.
Uses Y-threshold method to dynamically find nose width vertices instead of fixed vertices.

Key Difference from dataset_fitmask.py:
- dataset_fitmask.py: Uses fixed vertices (3092, 2057) for nose width
- This script: Uses Y-threshold to filter nose vertices, then finds MIN/MAX X dynamically

Also compares both methods to show calibration differences.

Dataset Structure:
    dataset/
        1/  (Person 1)
            Selfie_1.jpg, Selfie_2.jpg, ... Selfie_13.jpg
        2/  (Person 2)
            ...
        10/ (Person 10)
            ...

Usage:
    python dataset_fitmask_threshold.py
"""
import cv2
import numpy as np
import pickle
import torch
import sys
import os
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DECA'))

from deca_measurement import DECAMeasurement
from decalib.utils.config import cfg as deca_cfg
import face_alignment


# =============================================================================
# FIXED VERTEX DEFINITIONS (for comparison)
# =============================================================================

VERTEX_MIN_X_FIXED = 3092
VERTEX_MAX_X_FIXED = 2057

QUATTRO_AIR_UP = 3553
QUATTRO_AIR_DOWN = 3487

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
# DATASET PROCESSOR WITH Y-THRESHOLD
# =============================================================================

class DatasetMaskFitThreshold:
    def __init__(self):
        print("="*60)
        print("DATASET MASK FIT (Y-THRESHOLD METHOD)")
        print("="*60)
        
        # Load Y-threshold from file
        threshold_file = os.path.join(os.path.dirname(__file__), 'y_threshold.txt')
        if os.path.exists(threshold_file):
            with open(threshold_file, 'r') as f:
                line = f.readline()
                self.percentile = float(line.split('=')[1])
            print(f"Loaded Y threshold: {self.percentile:.1f}%")
        else:
            print("WARNING: y_threshold.txt not found, using default 76.0%")
            self.percentile = 76.0
        
        # Load FLAME nose mask
        masks_path = r'd:\Job\project_1\MICA\data\FLAME2020\FLAME_masks\FLAME_masks.pkl'
        with open(masks_path, 'rb') as f:
            masks = pickle.load(f, encoding='latin1')
        
        self.nose_vertices = sorted(list(masks['nose']))
        print(f"Loaded {len(self.nose_vertices)} nose vertices")
        
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
    
    def fix_orientation(self, frame):
        """Fix horizontal images by rotating to vertical"""
        h, w = frame.shape[:2]
        if w > h:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame
    
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
    
    def find_threshold_vertices(self, verts, trans_verts):
        """
        Find MIN/MAX X vertices using Y-threshold method.
        Returns the dynamically found vertex indices and the filtered vertices.
        """
        # Use PROJECTED coordinates for threshold (matches image Y)
        nose_trans_y = trans_verts[self.nose_vertices, 1]
        min_trans_y = np.min(nose_trans_y)  # Top of nose (bridge)
        max_trans_y = np.max(nose_trans_y)  # Bottom of nose (alar)
        
        # Calculate threshold in projected coordinates
        threshold_trans_y = min_trans_y + (max_trans_y - min_trans_y) * (self.percentile / 100.0)
        
        # Filter vertices BELOW the threshold (at alar level)
        # Higher trans_y = lower in image = toward alar
        below_threshold = []
        for idx in self.nose_vertices:
            v_trans_y = trans_verts[idx, 1]
            if v_trans_y >= threshold_trans_y:
                below_threshold.append(idx)
        
        if len(below_threshold) == 0:
            return None, None, [], threshold_trans_y
        
        # Find MIN/MAX X from filtered vertices
        below_coords = verts[below_threshold]
        min_x_local = np.argmin(below_coords[:, 0])
        max_x_local = np.argmax(below_coords[:, 0])
        
        min_x_vertex = below_threshold[min_x_local]
        max_x_vertex = below_threshold[max_x_local]
        
        return min_x_vertex, max_x_vertex, below_threshold, threshold_trans_y
    
    def measure(self, verts, trans_verts):
        """
        Measure using BOTH methods:
        1. Fixed vertices (like dataset_fitmask.py)
        2. Y-threshold method (dynamic vertices)
        """
        # Fixed vertex method
        v_min_fixed = verts[VERTEX_MIN_X_FIXED]
        v_max_fixed = verts[VERTEX_MAX_X_FIXED]
        nose_width_fixed = np.linalg.norm(v_min_fixed - v_max_fixed) * 1000
        
        # Y-threshold method
        min_v, max_v, below_threshold, threshold_y = self.find_threshold_vertices(verts, trans_verts)
        
        if min_v is not None and max_v is not None:
            v_min_thresh = verts[min_v]
            v_max_thresh = verts[max_v]
            nose_width_threshold = np.linalg.norm(v_min_thresh - v_max_thresh) * 1000
        else:
            nose_width_threshold = nose_width_fixed  # Fallback
            min_v = VERTEX_MIN_X_FIXED
            max_v = VERTEX_MAX_X_FIXED
        
        # Face heights (same for both methods)
        quattro_up = verts[QUATTRO_AIR_UP]
        quattro_down = verts[QUATTRO_AIR_DOWN]
        face_height_f10 = np.linalg.norm(quattro_up - quattro_down) * 1000
        
        f20_up = verts[AIRFIT_F20_UP]
        f20_down = verts[AIRFIT_F20_DOWN]
        face_height_f20 = np.linalg.norm(f20_up - f20_down) * 1000
        
        # Calculate difference
        diff = nose_width_threshold - nose_width_fixed
        diff_percent = (diff / nose_width_fixed * 100) if nose_width_fixed > 0 else 0
        
        return {
            'nose_width': float(nose_width_threshold),  # Primary: Y-threshold method
            'nose_width_fixed': float(nose_width_fixed),  # Comparison: Fixed method
            'nose_width_diff': float(diff),
            'nose_width_diff_percent': float(diff_percent),
            'vertex_min_x': int(min_v) if min_v is not None else None,
            'vertex_max_x': int(max_v) if max_v is not None else None,
            'vertices_below_threshold': int(len(below_threshold)) if min_v else 0,
            'face_height_f10': float(face_height_f10),
            'face_height_f20': float(face_height_f20),
        }
    
    def calculate_statistics(self, measurements_list):
        """Calculate mean, std, CV for each measurement type"""
        stats = {}
        
        keys = ['nose_width', 'nose_width_fixed', 'nose_width_diff', 
                'face_height_f10', 'face_height_f20']
        
        for key in keys:
            values = [m[key] for m in measurements_list if m.get(key) is not None]
            
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0
                cv = (std / abs(mean) * 100) if mean != 0 else 0
                
                stats[key] = {
                    'mean': float(mean),
                    'std': float(std),
                    'cv': float(cv),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return stats
    
    def get_recommendations_from_stats(self, stats):
        """Get mask recommendations based on mean measurements"""
        recommendations = {}
        measurements = {
            'nose_width': stats.get('nose_width', {}).get('mean', 0),
            'face_height_f10': stats.get('face_height_f10', {}).get('mean', 0),
            'face_height_f20': stats.get('face_height_f20', {}).get('mean', 0),
        }
        
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
    
    def create_output_image(self, frame, verts, trans_verts, crop_info, filename, person_stats=None):
        """Create output image with measurements and recommendations"""
        # Scale image if too small
        min_height = 600
        h, w = frame.shape[:2]
        scale = 1.0
        if h < min_height:
            scale = min_height / h
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            crop_info = {
                'x_min': int(crop_info['x_min'] * scale),
                'x_max': int(crop_info['x_max'] * scale),
                'y_min': int(crop_info['y_min'] * scale),
                'y_max': int(crop_info['y_max'] * scale)
            }
        
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Get measurements
        measurements = self.measure(verts, trans_verts)
        
        # Get threshold vertices
        min_v, max_v, below_threshold, threshold_y = self.find_threshold_vertices(verts, trans_verts)
        
        # Project Y-threshold points (Cyan - primary)
        if min_v is not None and max_v is not None:
            x1, y1 = self.project_to_image(trans_verts, crop_info, min_v)
            x2, y2 = self.project_to_image(trans_verts, crop_info, max_v)
            cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Cyan
            cv2.circle(vis, (x1, y1), 5, (255, 255, 0), -1)
            cv2.circle(vis, (x2, y2), 5, (255, 255, 0), -1)
            cv2.putText(vis, f"V{min_v}", (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            cv2.putText(vis, f"V{max_v}", (x2+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        # Project fixed points (Red - for comparison)
        fx1, fy1 = self.project_to_image(trans_verts, crop_info, VERTEX_MIN_X_FIXED)
        fx2, fy2 = self.project_to_image(trans_verts, crop_info, VERTEX_MAX_X_FIXED)
        cv2.line(vis, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)  # Red
        cv2.circle(vis, (fx1, fy1), 4, (0, 0, 255), -1)
        cv2.circle(vis, (fx2, fy2), 4, (0, 0, 255), -1)
        
        # Project face height points (Green - F10)
        qx1, qy1 = self.project_to_image(trans_verts, crop_info, QUATTRO_AIR_UP)
        qx2, qy2 = self.project_to_image(trans_verts, crop_info, QUATTRO_AIR_DOWN)
        cv2.line(vis, (qx1, qy1), (qx2, qy2), (0, 255, 0), 2)
        cv2.circle(vis, (qx1, qy1), 4, (0, 255, 0), -1)
        cv2.circle(vis, (qx2, qy2), 4, (0, 255, 0), -1)
        
        # Draw threshold line
        if threshold_y is not None:
            threshold_img_y = int((threshold_y + 1) * 112)
            crop_h = crop_info['y_max'] - crop_info['y_min']
            threshold_img_y = int(threshold_img_y * crop_h / 224 + crop_info['y_min'])
            cv2.line(vis, (0, threshold_img_y), (w, threshold_img_y), (255, 0, 255), 1)  # Magenta
        
        # Create info panel
        panel_height = 150
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        # Title
        cv2.putText(panel, f"{filename}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Measurements - Y-Threshold method (primary)
        y_pos = 50
        cv2.putText(panel, f"Y-Thresh Nose: {measurements['nose_width']:.1f}mm (V{measurements['vertex_min_x']}-V{measurements['vertex_max_x']})", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        # Measurements - Fixed method (comparison)
        y_pos += 20
        cv2.putText(panel, f"Fixed Nose: {measurements['nose_width_fixed']:.1f}mm (V{VERTEX_MIN_X_FIXED}-V{VERTEX_MAX_X_FIXED})", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        
        # Difference
        y_pos += 20
        diff_color = (0, 255, 0) if abs(measurements['nose_width_diff']) < 2 else (0, 165, 255)
        cv2.putText(panel, f"Diff: {measurements['nose_width_diff']:+.2f}mm ({measurements['nose_width_diff_percent']:+.1f}%)", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, diff_color, 1)
        
        # Face heights
        cv2.putText(panel, f"F10: {measurements['face_height_f10']:.1f}mm", (300, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(panel, f"F20: {measurements['face_height_f20']:.1f}mm", (430, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        
        # If we have person stats, show CV
        if person_stats:
            y_pos += 25
            nose_cv = person_stats.get('nose_width', {}).get('cv', 0)
            fixed_cv = person_stats.get('nose_width_fixed', {}).get('cv', 0)
            cv2.putText(panel, f"CV: Threshold:{nose_cv:.1f}% | Fixed:{fixed_cv:.1f}%", 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # Mask recommendations
        y_pos += 25
        recommendations = self.get_recommendations_from_stats(person_stats) if person_stats else {}
        rec_str = ""
        for mask_id in ['N20', 'F10', 'F20', 'F40']:
            rec = recommendations.get(mask_id, {})
            if rec.get('size') and rec['size'] != 'Unknown':
                rec_str += f"{mask_id}:{rec['size']} | "
        if rec_str:
            cv2.putText(panel, f"Masks: {rec_str[:-3]}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        output = np.vstack([panel, vis])
        
        return output, measurements
    
    def process_dataset(self, dataset_dir, output_dir):
        """Process entire dataset with Y-threshold method and comparison"""
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all person folders (1-10)
        person_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()],
                               key=lambda x: int(x.name) if x.name.isdigit() else 0)
        
        print(f"\nFound {len(person_folders)} person folders")
        print(f"Y-Threshold: {self.percentile:.1f}%")
        print("="*60)
        
        all_results = {}
        
        for person_folder in person_folders:
            person_id = person_folder.name
            print(f"\n{'='*60}")
            print(f"Processing Person {person_id}")
            print(f"{'='*60}")
            
            # Create output folder for this person
            person_output = output_path / person_id
            person_output.mkdir(exist_ok=True)
            
            # Find all Selfie images (both .jpg and .jpeg)
            selfie_files = sorted(
                list(person_folder.glob('Selfie_*.jpg')) + 
                list(person_folder.glob('Selfie_*.jpeg'))
            )
            print(f"Found {len(selfie_files)} selfie images")
            
            person_measurements = []
            
            for selfie_file in selfie_files:
                print(f"  Processing: {selfie_file.name}", end=" ")
                
                frame = cv2.imread(str(selfie_file))
                if frame is None:
                    print("[ERROR] Failed to load")
                    continue
                
                # Fix orientation if horizontal
                frame = self.fix_orientation(frame)
                
                verts, trans_verts, crop_info = self.process_frame(frame)
                
                if verts is None:
                    print("[NO FACE]")
                    continue
                
                measurements = self.measure(verts, trans_verts)
                person_measurements.append({
                    'file': selfie_file.name,
                    **measurements
                })
                
                print(f"[OK] Thresh: {measurements['nose_width']:.1f}mm | Fixed: {measurements['nose_width_fixed']:.1f}mm | Diff: {measurements['nose_width_diff']:+.1f}mm")
            
            if not person_measurements:
                print(f"  No valid measurements for Person {person_id}")
                continue
            
            # Calculate statistics for this person
            stats = self.calculate_statistics(person_measurements)
            
            # Print statistics
            print(f"\n  --- Person {person_id} Statistics ---")
            print(f"  Y-Threshold Method:")
            s = stats.get('nose_width', {})
            print(f"    Nose Width: Mean={s.get('mean', 0):.1f}mm, Std={s.get('std', 0):.2f}, CV={s.get('cv', 0):.2f}%")
            
            print(f"  Fixed Vertex Method:")
            s = stats.get('nose_width_fixed', {})
            print(f"    Nose Width: Mean={s.get('mean', 0):.1f}mm, Std={s.get('std', 0):.2f}, CV={s.get('cv', 0):.2f}%")
            
            print(f"  Difference (Threshold - Fixed):")
            s = stats.get('nose_width_diff', {})
            print(f"    Mean Diff: {s.get('mean', 0):+.2f}mm")
            
            print(f"  Face Heights:")
            for key in ['face_height_f10', 'face_height_f20']:
                s = stats.get(key, {})
                print(f"    {key}: Mean={s.get('mean', 0):.1f}mm, CV={s.get('cv', 0):.2f}%")
            
            # Get recommendations
            recommendations = self.get_recommendations_from_stats(stats)
            print(f"\n  --- Mask Recommendations (Y-Threshold) ---")
            for mask_id, rec in recommendations.items():
                if rec['size'] != 'Unknown':
                    print(f"  {mask_id} ({rec['name']}): {rec['size']}")
            
            # Save output images with stats overlay
            for selfie_file in selfie_files:
                frame = cv2.imread(str(selfie_file))
                if frame is None:
                    continue
                
                frame = self.fix_orientation(frame)
                verts, trans_verts, crop_info = self.process_frame(frame)
                
                if verts is None:
                    continue
                
                output_img, _ = self.create_output_image(
                    frame, verts, trans_verts, crop_info,
                    f"Person {person_id}: {selfie_file.name}",
                    stats
                )
                
                output_file = person_output / f"result_{selfie_file.name}"
                cv2.imwrite(str(output_file), output_img)
            
            # Store results
            all_results[person_id] = {
                'measurements': person_measurements,
                'statistics': stats,
                'recommendations': {k: v['size'] for k, v in recommendations.items()},
                'threshold_percentile': self.percentile
            }
        
        # Save JSON results
        results_file = output_path / "results_summary.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Print final summary with comparison
        print("\n" + "="*100)
        print("FINAL SUMMARY - METHOD COMPARISON")
        print(f"Y-Threshold: {self.percentile:.1f}%")
        print("="*100)
        print(f"{'Person':<8} {'#Imgs':<6} {'Thresh Mean':<12} {'Thresh CV%':<10} {'Fixed Mean':<12} {'Fixed CV%':<10} {'Diff':<10}")
        print("-"*100)
        
        for person_id, data in all_results.items():
            stats = data['statistics']
            n = stats.get('nose_width', {}).get('count', 0)
            thresh_mean = stats.get('nose_width', {}).get('mean', 0)
            thresh_cv = stats.get('nose_width', {}).get('cv', 0)
            fixed_mean = stats.get('nose_width_fixed', {}).get('mean', 0)
            fixed_cv = stats.get('nose_width_fixed', {}).get('cv', 0)
            diff_mean = stats.get('nose_width_diff', {}).get('mean', 0)
            
            print(f"{person_id:<8} {n:<6} {thresh_mean:<12.1f} {thresh_cv:<10.2f} {fixed_mean:<12.1f} {fixed_cv:<10.2f} {diff_mean:+.2f}mm")
        
        print("="*100)
        
        return all_results


if __name__ == "__main__":
    dataset_dir = "dataset"
    output_dir = "dataset_output_threshold"
    
    processor = DatasetMaskFitThreshold()
    results = processor.process_dataset(dataset_dir, output_dir)
    
    print(f"\nProcessing complete! Results saved to: {output_dir}/")
