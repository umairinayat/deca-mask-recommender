#!/usr/bin/env python3
"""
DATASET MASK FIT PROCESSING WITH CROSS-VALIDATION
==================================================
Process dataset with multiple persons, each having multiple selfie images.
Calculate statistics (mean, std, CV) for measurement consistency.
Save output images with labels and recommendations.

Dataset Structure:
    dataset/
        1/  (Person 1)
            Selfie_1.jpg, Selfie_2.jpg, ... Selfie_13.jpg
        2/  (Person 2)
            ...
        10/ (Person 10)
            ...

Usage:
    python dataset_fitmask.py
"""
import cv2
import numpy as np
import torch
import sys
import os
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deca_measurement import DECAMeasurement
from decalib.utils.config import cfg as deca_cfg
import face_alignment


# =============================================================================
# FLAME VERTEX DEFINITIONS
# =============================================================================

VERTEX_MIN_X = 3092
VERTEX_MAX_X = 2057

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
# DATASET PROCESSOR
# =============================================================================

class DatasetMaskFit:
    def __init__(self):
        print("="*60)
        print("DATASET MASK FIT PROCESSING")
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
    
    def fix_orientation(self, frame):
        """Fix horizontal images by rotating to vertical"""
        h, w = frame.shape[:2]
        if w > h:
            # Horizontal image - rotate 90 degrees
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
    
    def calculate_statistics(self, measurements_list):
        """Calculate mean, std, CV for each measurement type"""
        stats = {}
        
        for key in ['nose_width', 'face_height_f10', 'face_height_f20']:
            values = [m[key] for m in measurements_list if m.get(key) is not None]
            
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values, ddof=1) if len(values) > 1 else 0
                cv = (std / mean * 100) if mean > 0 else 0
                
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
        measurements = self.measure(verts)
        
        # Project points
        x1, y1 = self.project_to_image(trans_verts, crop_info, VERTEX_MIN_X)
        x2, y2 = self.project_to_image(trans_verts, crop_info, VERTEX_MAX_X)
        qx1, qy1 = self.project_to_image(trans_verts, crop_info, QUATTRO_AIR_UP)
        qx2, qy2 = self.project_to_image(trans_verts, crop_info, QUATTRO_AIR_DOWN)
        fx1, fy1 = self.project_to_image(trans_verts, crop_info, AIRFIT_F20_UP)
        fx2, fy2 = self.project_to_image(trans_verts, crop_info, AIRFIT_F20_DOWN)
        
        # Draw measurement lines
        cv2.line(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(vis, (x1, y1), 4, (0, 0, 255), -1)
        cv2.circle(vis, (x2, y2), 4, (0, 0, 255), -1)
        
        cv2.line(vis, (qx1, qy1), (qx2, qy2), (0, 255, 0), 2)
        cv2.circle(vis, (qx1, qy1), 4, (0, 255, 0), -1)
        cv2.circle(vis, (qx2, qy2), 4, (0, 255, 0), -1)
        
        cv2.line(vis, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
        cv2.circle(vis, (fx1, fy1), 4, (255, 0, 0), -1)
        cv2.circle(vis, (fx2, fy2), 4, (255, 0, 0), -1)
        
        # Create info panel
        panel_height = 120
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)
        
        # Title
        cv2.putText(panel, f"{filename}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Measurements
        y_pos = 50
        cv2.putText(panel, f"Nose: {measurements['nose_width']:.1f}mm", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(panel, f"F10: {measurements['face_height_f10']:.1f}mm", (180, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(panel, f"F20: {measurements['face_height_f20']:.1f}mm", (340, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # If we have person stats, show CV
        if person_stats:
            y_pos += 25
            nose_cv = person_stats.get('nose_width', {}).get('cv', 0)
            f10_cv = person_stats.get('face_height_f10', {}).get('cv', 0)
            f20_cv = person_stats.get('face_height_f20', {}).get('cv', 0)
            cv2.putText(panel, f"CV: Nose:{nose_cv:.1f}% | F10:{f10_cv:.1f}% | F20:{f20_cv:.1f}%", 
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
        """Process entire dataset with cross-validation statistics"""
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find all person folders (1-10)
        person_folders = sorted([f for f in dataset_path.iterdir() if f.is_dir()],
                               key=lambda x: int(x.name) if x.name.isdigit() else 0)
        
        print(f"\nFound {len(person_folders)} person folders")
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
                
                measurements = self.measure(verts)
                person_measurements.append({
                    'file': selfie_file.name,
                    **measurements
                })
                
                print(f"[OK] Nose: {measurements['nose_width']:.1f}mm")
            
            if not person_measurements:
                print(f"  No valid measurements for Person {person_id}")
                continue
            
            # Calculate statistics for this person
            stats = self.calculate_statistics(person_measurements)
            
            # Print statistics
            print(f"\n  --- Person {person_id} Statistics ---")
            for key in ['nose_width', 'face_height_f10', 'face_height_f20']:
                s = stats.get(key, {})
                print(f"  {key}: Mean={s.get('mean', 0):.1f}mm, Std={s.get('std', 0):.2f}, CV={s.get('cv', 0):.2f}%")
            
            # Get recommendations
            recommendations = self.get_recommendations_from_stats(stats)
            print(f"\n  --- Mask Recommendations ---")
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
                'recommendations': {k: v['size'] for k, v in recommendations.items()}
            }
        
        # Save JSON results
        results_file = output_path / "results_summary.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Print final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        print(f"{'Person':<8} {'#Images':<8} {'Nose Mean':<12} {'Nose CV%':<10} {'F10 Mean':<12} {'F20 Mean':<12}")
        print("-"*80)
        
        for person_id, data in all_results.items():
            stats = data['statistics']
            n = stats.get('nose_width', {}).get('count', 0)
            nose_mean = stats.get('nose_width', {}).get('mean', 0)
            nose_cv = stats.get('nose_width', {}).get('cv', 0)
            f10_mean = stats.get('face_height_f10', {}).get('mean', 0)
            f20_mean = stats.get('face_height_f20', {}).get('mean', 0)
            
            print(f"{person_id:<8} {n:<8} {nose_mean:<12.1f} {nose_cv:<10.2f} {f10_mean:<12.1f} {f20_mean:<12.1f}")
        
        print("="*80)
        
        return all_results


if __name__ == "__main__":
    dataset_dir = "dataset"
    output_dir = "dataset_output"
    
    processor = DatasetMaskFit()
    results = processor.process_dataset(dataset_dir, output_dir)
    
    print(f"\nProcessing complete! Results saved to: {output_dir}/")
