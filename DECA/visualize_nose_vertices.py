#!/usr/bin/env python3
"""
FLAME NOSE VERTEX VISUALIZER  
============================
Visualizes the official 379 nose vertices from FLAME_masks.pkl
Uses DECAMeasurement (no pytorch3d required)

Press SPACE to capture and visualize
Press ESC to exit
"""
import cv2
import numpy as np
import pickle
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from decalib.deca_measurement import DECAMeasurement
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import util
import face_alignment


class NoseVertexVisualizer:
    def __init__(self):
        print("="*60)
        print("FLAME NOSE VERTEX VISUALIZER")
        print("="*60)
        
        # Load FLAME masks
        masks_path = r'd:\Job\project_1\MICA\data\FLAME2020\FLAME_masks\FLAME_masks.pkl'
        print(f"[1/3] Loading FLAME masks...")
        
        with open(masks_path, 'rb') as f:
            masks = pickle.load(f, encoding='latin1')
        
        self.nose_vertices = sorted(list(masks['nose']))
        self.face_vertices = sorted(list(masks['face']))
        print(f"      Nose: {len(self.nose_vertices)} vertices")
        print(f"      Face: {len(self.face_vertices)} vertices")
        
        # Initialize DECA (measurement mode - no renderer needed)
        print("[2/3] Initializing DECA (measurement mode)...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"      Device: {self.device}")
        
        deca_cfg.model.use_tex = False
        self.deca = DECAMeasurement(config=deca_cfg, device=self.device)
        
        # Initialize face detector
        print("[3/3] Initializing face detector...")
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=False,
            device=self.device
        )
        
        self.image_size = 224
        
        print("="*60)
        print("Ready! Press SPACE to capture, ESC to exit")
        print("="*60)
    
    def detect_and_crop_face(self, frame):
        """Detect face and return cropped + resized image for DECA"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        landmarks = self.fa.get_landmarks(frame_rgb)
        if landmarks is None or len(landmarks) == 0:
            return None, None
        
        lmk = landmarks[0]
        
        # Get bounding box with margin
        h, w = frame.shape[:2]
        x_min = max(0, int(np.min(lmk[:, 0])) - 30)
        x_max = min(w, int(np.max(lmk[:, 0])) + 30)
        y_min = max(0, int(np.min(lmk[:, 1])) - 50)
        y_max = min(h, int(np.max(lmk[:, 1])) + 30)
        
        # Make it square
        size = max(x_max - x_min, y_max - y_min)
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        
        x_min = max(0, center_x - size // 2)
        x_max = min(w, center_x + size // 2)
        y_min = max(0, center_y - size // 2)
        y_max = min(h, center_y + size // 2)
        
        crop_info = {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'size': size
        }
        
        # Crop and resize
        face_crop = frame_rgb[y_min:y_max, x_min:x_max]
        face_resized = cv2.resize(face_crop, (224, 224))
        
        return face_resized, crop_info
    
    def process_frame(self, frame):
        """Run DECA and get vertices"""
        face_img, crop_info = self.detect_and_crop_face(frame)
        if face_img is None:
            return None, None, None
        
        # Convert to tensor
        face_tensor = torch.tensor(face_img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        face_tensor = face_tensor.to(self.device)
        
        with torch.no_grad():
            codedict = self.deca.encode(face_tensor)
            opdict = self.deca.decode(codedict)
        
        verts = opdict['verts'][0].cpu().numpy()  # [5023, 3]
        trans_verts = opdict['trans_verts'][0].cpu().numpy()  # [5023, 3] in [-1,1]
        
        return verts, trans_verts, crop_info
    
    def project_to_image(self, trans_verts, crop_info, idx):
        """Project vertex from normalized [-1,1] coords to original image"""
        v = trans_verts[idx]
        
        # trans_verts are in [-1, 1] for 224x224
        x_224 = (v[0] + 1) * 112  # [0, 224]
        y_224 = (v[1] + 1) * 112  # [0, 224]
        
        # Scale to crop size
        crop_w = crop_info['x_max'] - crop_info['x_min']
        crop_h = crop_info['y_max'] - crop_info['y_min']
        
        x_crop = x_224 * crop_w / 224
        y_crop = y_224 * crop_h / 224
        
        # Offset to original image
        x_orig = int(x_crop + crop_info['x_min'])
        y_orig = int(y_crop + crop_info['y_min'])
        
        return x_orig, y_orig
    
    def visualize(self, frame, verts, trans_verts, crop_info):
        """Draw nose vertices on frame"""
        vis = frame.copy()
        
        # Find actual min/max X in nose region
        nose_coords = verts[self.nose_vertices]
        min_x_local_idx = np.argmin(nose_coords[:, 0])
        max_x_local_idx = np.argmax(nose_coords[:, 0])
        
        min_x_vertex = self.nose_vertices[min_x_local_idx]
        max_x_vertex = self.nose_vertices[max_x_local_idx]
        
        print(f"\n{'='*60}")
        print("NOSE VERTEX ANALYSIS (from actual data)")
        print(f"{'='*60}")
        print(f"Total nose vertices: {len(self.nose_vertices)}")
        print(f"\nActual MIN X (leftmost):  V{min_x_vertex} at X={verts[min_x_vertex, 0]:.4f}")
        print(f"Actual MAX X (rightmost): V{max_x_vertex} at X={verts[max_x_vertex, 0]:.4f}")
        
        # Calculate nose width using actual min/max
        nose_width = np.linalg.norm(verts[min_x_vertex] - verts[max_x_vertex])
        print(f"\nNose Width ({min_x_vertex}-{max_x_vertex}): {nose_width:.4f} FLAME units")
        
        # Compare with claimed vertices
        print(f"\nComparison with claimed vertices:")
        print(f"  Claimed 2750: X={verts[2750, 0]:.4f}")
        print(f"  Claimed 1610: X={verts[1610, 0]:.4f}")
        print(f"  2750 == actual left?  {min_x_vertex == 2750}")
        print(f"  1610 == actual right? {max_x_vertex == 1610}")
        
        # Draw all nose vertices (tiny green dots)
        for idx in self.nose_vertices:
            if idx == min_x_vertex or idx == max_x_vertex:
                continue
            try:
                x, y = self.project_to_image(trans_verts, crop_info, idx)
                if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                    cv2.circle(vis, (x, y), 1, (0, 200, 0), -1)  # Tiny green dot
            except:
                pass
        
        # Draw actual MIN X (RED) - small clear circle
        x1, y1 = self.project_to_image(trans_verts, crop_info, min_x_vertex)
        cv2.circle(vis, (x1, y1), 3, (0, 0, 255), -1)
        cv2.putText(vis, f"{min_x_vertex}", (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        # Draw actual MAX X (BLUE) - small clear circle
        x2, y2 = self.project_to_image(trans_verts, crop_info, max_x_vertex)
        cv2.circle(vis, (x2, y2), 3, (255, 0, 0), -1)
        cv2.putText(vis, f"{max_x_vertex}", (x2+5, y2-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
        
        # Draw thin line between min/max
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
        
        # Info overlay
        cv2.putText(vis, f"Nose: {len(self.nose_vertices)} vertices", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis, f"MIN X: V{min_x_vertex} (red)", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(vis, f"MAX X: V{max_x_vertex} (blue)", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(vis, f"Width: {nose_width:.4f}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return vis
    
    def run(self):
        """Main camera loop"""
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
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Camera', display)
            if result is not None:
                cv2.imshow('Nose Vertices', result)
            
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
    viz = NoseVertexVisualizer()
    viz.run()
