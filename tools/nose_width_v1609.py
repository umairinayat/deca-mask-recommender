#!/usr/bin/env python3
"""
NOSE WIDTH MEASUREMENT - Using V1609 as Y threshold
====================================================
Uses vertex 1609 as the Y-level reference.
Finds MIN X and MAX X from nose vertices BELOW V1609's Y level.

Window 1: Camera feed - Press SPACE to capture
Window 2: Visualization with V1609, MIN X, MAX X highlighted
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
import face_alignment


class NoseWidthV1609:
    def __init__(self):
        print("="*60)
        print("NOSE WIDTH MEASUREMENT (V1609 as Y-threshold)")
        print("="*60)
        
        # Load FLAME masks
        masks_path = r'd:\Job\project_1\MICA\data\FLAME2020\FLAME_masks\FLAME_masks.pkl'
        with open(masks_path, 'rb') as f:
            masks = pickle.load(f, encoding='latin1')
        
        self.nose_vertices = sorted(list(masks['nose']))
        print(f"Loaded {len(self.nose_vertices)} nose vertices")
        print(f"Reference vertex: 1609")
        
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
    
    def visualize(self, frame, verts, trans_verts, crop_info):
        vis = frame.copy()
        
        # Get V1609's Y coordinate (FLAME 3D Y)
        v1609_y = verts[1609, 1]
        
        # Find all nose vertices with Y <= V1609's Y (at or below alar level)
        # In FLAME: lower Y = lower on face
        vertices_below = []
        vertices_above = []
        
        for idx in self.nose_vertices:
            if verts[idx, 1] <= v1609_y:
                vertices_below.append(idx)
            else:
                vertices_above.append(idx)
        
        print(f"\n{'='*60}")
        print(f"NOSE WIDTH ANALYSIS (V1609 as threshold)")
        print(f"{'='*60}")
        print(f"V1609 Y-coordinate: {v1609_y:.4f}")
        print(f"Vertices at/below V1609 level: {len(vertices_below)}")
        print(f"Vertices above V1609 level: {len(vertices_above)}")
        
        # Find MIN X and MAX X from vertices below V1609
        if len(vertices_below) > 0:
            below_coords = verts[vertices_below]
            min_x_local = np.argmin(below_coords[:, 0])
            max_x_local = np.argmax(below_coords[:, 0])
            
            min_x_vertex = vertices_below[min_x_local]
            max_x_vertex = vertices_below[max_x_local]
            
            nose_width = np.linalg.norm(verts[min_x_vertex] - verts[max_x_vertex])
            
            print(f"\nMIN X vertex: V{min_x_vertex} at X={verts[min_x_vertex, 0]:.4f}")
            print(f"MAX X vertex: V{max_x_vertex} at X={verts[max_x_vertex, 0]:.4f}")
            print(f"Nose Width: {nose_width:.4f} FLAME units")
        else:
            min_x_vertex = None
            max_x_vertex = None
            nose_width = 0
        
        # Draw vertices ABOVE threshold (gray - ignored)
        for idx in vertices_above:
            x, y = self.project_to_image(trans_verts, crop_info, idx)
            if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                cv2.circle(vis, (x, y), 1, (100, 100, 100), -1)
        
        # Draw vertices BELOW threshold (green - used)
        for idx in vertices_below:
            if idx in [1609, min_x_vertex, max_x_vertex]:
                continue
            x, y = self.project_to_image(trans_verts, crop_info, idx)
            if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                cv2.circle(vis, (x, y), 1, (0, 200, 0), -1)
        
        # Draw V1609 - YELLOW (reference point)
        x1609, y1609 = self.project_to_image(trans_verts, crop_info, 1609)
        cv2.circle(vis, (x1609, y1609), 4, (0, 255, 255), -1)
        cv2.putText(vis, "1609", (x1609+5, y1609), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        
        # Draw horizontal line at V1609 level
        cv2.line(vis, (0, y1609), (vis.shape[1], y1609), (0, 255, 255), 1)
        
        # Draw MIN X - RED
        if min_x_vertex is not None:
            x_min, y_min = self.project_to_image(trans_verts, crop_info, min_x_vertex)
            cv2.circle(vis, (x_min, y_min), 4, (0, 0, 255), -1)
            cv2.putText(vis, f"{min_x_vertex}", (x_min+5, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        # Draw MAX X - BLUE
        if max_x_vertex is not None:
            x_max, y_max = self.project_to_image(trans_verts, crop_info, max_x_vertex)
            cv2.circle(vis, (x_max, y_max), 4, (255, 0, 0), -1)
            cv2.putText(vis, f"{max_x_vertex}", (x_max+5, y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
        
        # Draw line between MIN and MAX
        if min_x_vertex is not None and max_x_vertex is not None:
            cv2.line(vis, (x_min, y_min), (x_max, y_max), (255, 0, 255), 1)
        
        # Info overlay
        cv2.putText(vis, f"V1609 threshold (yellow)", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(vis, f"Below: {len(vertices_below)} (green)", (10, 38),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        cv2.putText(vis, f"MIN X: V{min_x_vertex} (red)", (10, 56),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(vis, f"MAX X: V{max_x_vertex} (blue)", (10, 74),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
        cv2.putText(vis, f"Width: {nose_width:.4f}", (10, 92),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
        
        return vis, min_x_vertex, max_x_vertex, nose_width
    
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
            cv2.putText(display, "SPACE: capture | S: save | ESC: exit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow('Camera', display)
            if result is not None:
                cv2.imshow('V1609 Threshold Analysis', result)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                print("\n[*] Processing...")
                verts, trans_verts, crop_info = self.process_frame(frame)
                if verts is not None:
                    result, min_v, max_v, width = self.visualize(frame, verts, trans_verts, crop_info)
                    print(f"[OK] Done! MIN={min_v}, MAX={max_v}, Width={width:.4f}")
                else:
                    print("[ERROR] No face detected")
            
            elif key == ord('s') and result is not None:
                filename = 'nose_v1609_analysis.png'
                cv2.imwrite(filename, result)
                print(f"[SAVED] {filename}")
            
            elif key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    analyzer = NoseWidthV1609()
    analyzer.run()
