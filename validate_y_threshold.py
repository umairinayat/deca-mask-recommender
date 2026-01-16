#!/usr/bin/env python3
"""
PHASE 2: VALIDATE Y-THRESHOLD
==============================
Visualizes a horizontal bar at the selected Y percentile.
Shows MIN/MAX X only from vertices BELOW that threshold.

Instructions:
1. Run select_y_threshold.py first to create y_threshold.txt
2. Run this script
3. Press SPACE to capture
4. See the threshold line and filtered min/max
"""
import cv2
import numpy as np
import pickle
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DECA'))

from deca_measurement import DECAMeasurement
from decalib.utils.config import cfg as deca_cfg
import face_alignment


class YThresholdValidator:
    def __init__(self):
        print("="*60)
        print("Y-THRESHOLD VALIDATOR")
        print("="*60)
        
        # Load threshold
        threshold_file = os.path.join(os.path.dirname(__file__), 'y_threshold.txt')
        if os.path.exists(threshold_file):
            with open(threshold_file, 'r') as f:
                line = f.readline()
                self.percentile = float(line.split('=')[1])
            print(f"Loaded Y threshold: {self.percentile:.1f}%")
        else:
            print("WARNING: y_threshold.txt not found, using default 30%")
            self.percentile = 30.0
        
        # Load FLAME masks
        masks_path = r'd:\Job\project_1\MICA\data\FLAME2020\FLAME_masks\FLAME_masks.pkl'
        with open(masks_path, 'rb') as f:
            masks = pickle.load(f, encoding='latin1')
        
        self.nose_vertices = sorted(list(masks['nose']))
        print(f"Loaded {len(self.nose_vertices)} nose vertices")
        
        # Initialize DECA
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        deca_cfg.model.use_tex = False
        self.deca = DECAMeasurement(config=deca_cfg, device=self.device)
        
        # Initialize face detector
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=self.device
        )
        
        print("="*60)
        print("Press SPACE to capture, ESC to exit")
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
    
    def flame_y_to_image_y(self, flame_y, trans_verts, crop_info):
        """Convert FLAME Y coordinate to image Y coordinate"""
        y_224 = (flame_y + 1) * 112
        crop_h = crop_info['y_max'] - crop_info['y_min']
        y_crop = y_224 * crop_h / 224
        img_y = int(y_crop + crop_info['y_min'])
        return img_y
    
    def visualize(self, frame, verts, trans_verts, crop_info):
        vis = frame.copy()
        
        # Use PROJECTED coordinates (trans_verts) for threshold - this matches image Y
        # trans_verts Y has been flipped, so higher Y = visually lower in image
        nose_trans_y = trans_verts[self.nose_vertices, 1]
        min_trans_y = np.min(nose_trans_y)  # Top of nose (bridge) in projected coords
        max_trans_y = np.max(nose_trans_y)  # Bottom of nose (alar) in projected coords
        
        # Calculate threshold in projected coordinates
        # percentile = 0% means line at bridge, 100% means line at alar
        threshold_trans_y = min_trans_y + (max_trans_y - min_trans_y) * (self.percentile / 100.0)
        
        # Convert to image Y for drawing the line
        threshold_img_y = int((threshold_trans_y + 1) * 112)
        crop_h = crop_info['y_max'] - crop_info['y_min']
        threshold_img_y = int(threshold_img_y * crop_h / 224 + crop_info['y_min'])
        
        # Filter vertices: we want vertices BELOW the threshold line in the image
        # In projected coords, HIGHER trans_y = LOWER in image (toward alar)
        below_threshold = []
        above_threshold = []
        
        for idx in self.nose_vertices:
            v_trans_y = trans_verts[idx, 1]
            if v_trans_y >= threshold_trans_y:  # Higher projected Y = below the line visually
                below_threshold.append(idx)
            else:
                above_threshold.append(idx)
        
        print(f"\n{'='*60}")
        print(f"Y THRESHOLD VALIDATION")
        print(f"{'='*60}")
        print(f"Threshold: {self.percentile:.1f}%")
        print(f"Vertices BELOW threshold (alar level): {len(below_threshold)}")
        print(f"Vertices ABOVE threshold (bridge): {len(above_threshold)}")
        
        # Draw vertices above threshold (gray - ignored)
        for idx in above_threshold:
            try:
                x, y = self.project_to_image(trans_verts, crop_info, idx)
                if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                    cv2.circle(vis, (x, y), 1, (100, 100, 100), -1)  # Gray
            except:
                pass
        
        # Draw vertices below threshold (green - used for measurement)
        for idx in below_threshold:
            try:
                x, y = self.project_to_image(trans_verts, crop_info, idx)
                if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                    cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)  # Green
            except:
                pass
        
        # Find min/max X from BELOW threshold only
        if len(below_threshold) > 0:
            below_coords = verts[below_threshold]
            min_x_local = np.argmin(below_coords[:, 0])
            max_x_local = np.argmax(below_coords[:, 0])
            
            min_x_vertex = below_threshold[min_x_local]
            max_x_vertex = below_threshold[max_x_local]
            
            print(f"\nFiltered MIN X: V{min_x_vertex} at X={verts[min_x_vertex, 0]:.4f}")
            print(f"Filtered MAX X: V{max_x_vertex} at X={verts[max_x_vertex, 0]:.4f}")
            
            nose_width = np.linalg.norm(verts[min_x_vertex] - verts[max_x_vertex])
            print(f"Nose Width: {nose_width:.4f} FLAME units")
            
            # Draw MIN X (Red)
            x1, y1 = self.project_to_image(trans_verts, crop_info, min_x_vertex)
            cv2.circle(vis, (x1, y1), 4, (0, 0, 255), -1)
            cv2.putText(vis, f"{min_x_vertex}", (x1+5, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            # Draw MAX X (Blue)
            x2, y2 = self.project_to_image(trans_verts, crop_info, max_x_vertex)
            cv2.circle(vis, (x2, y2), 4, (255, 0, 0), -1)
            cv2.putText(vis, f"{max_x_vertex}", (x2+5, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
            
            # Draw line
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        # Draw horizontal threshold line (cyan)
        cv2.line(vis, (0, threshold_img_y), (vis.shape[1], threshold_img_y), (255, 255, 0), 2)
        
        # Info overlay
        cv2.putText(vis, f"Threshold: {self.percentile:.1f}%", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(vis, f"Below (green): {len(below_threshold)}", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis, f"Above (gray): {len(above_threshold)}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        return vis
    
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        result = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            display = frame.copy()
            cv2.putText(display, "SPACE: capture | ESC: exit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow('Camera', display)
            if result is not None:
                cv2.imshow('Threshold Validation', result)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                print("\n[*] Processing...")
                verts, trans_verts, crop_info = self.process_frame(frame)
                if verts is not None:
                    result = self.visualize(frame, verts, trans_verts, crop_info)
                    print("[OK] Done!")
                else:
                    print("[ERROR] No face detected")
            
            elif key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    validator = YThresholdValidator()
    validator.run()
