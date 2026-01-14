#!/usr/bin/env python3
"""
PHASE 1: Y-THRESHOLD SELECTOR (FIXED)
=====================================
Click to mark a horizontal line.
Finds the vertex with highest Y that is still BELOW that line.
Calculates the percentile rank of that vertex.

Instructions:
1. Press SPACE to capture frame
2. LEFT-CLICK on the visualization to mark your threshold line
3. The percentile of highest vertex below that line is saved
4. Press ESC to exit
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


class YThresholdSelector:
    def __init__(self):
        print("="*60)
        print("Y-THRESHOLD SELECTOR")
        print("="*60)
        
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
        
        # State
        self.current_frame = None
        self.current_verts = None
        self.current_trans_verts = None
        self.current_crop_info = None
        self.vertex_image_coords = None  # List of (idx, x, y) for each nose vertex
        self.clicked_y = None
        self.selected_vertex = None
        self.percentile = None
        
        print("="*60)
        print("Press SPACE to capture, click to mark threshold, ESC to exit")
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
    
    def compute_vertex_image_coords(self):
        """Compute image coordinates for all nose vertices"""
        self.vertex_image_coords = []
        for idx in self.nose_vertices:
            x, y = self.project_to_image(self.current_trans_verts, self.current_crop_info, idx)
            self.vertex_image_coords.append((idx, x, y))
        
        # Sort by image Y (top to bottom in image = increasing Y)
        self.vertex_image_coords.sort(key=lambda v: v[2])
    
    def find_vertex_below_line(self, clicked_y):
        """
        Find the vertex with HIGHEST image Y that is still BELOW the clicked line.
        Below in image means Y < clicked_y (higher up in image).
        """
        vertices_below = [(idx, x, y) for idx, x, y in self.vertex_image_coords if y < clicked_y]
        
        if len(vertices_below) == 0:
            return None, 0
        
        # Get the one with highest Y (closest to the line but still above it)
        selected = max(vertices_below, key=lambda v: v[2])
        
        # Calculate percentile: what fraction of vertices are at or below this one
        # We use rank in the sorted list
        rank = self.vertex_image_coords.index(selected)
        percentile = (rank / len(self.vertex_image_coords)) * 100
        
        return selected, percentile
    
    def visualize(self, frame):
        if self.current_verts is None:
            return frame
        
        vis = frame.copy()
        
        # Draw all nose vertices
        for idx, x, y in self.vertex_image_coords:
            if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
                cv2.circle(vis, (x, y), 1, (0, 200, 0), -1)
        
        # Draw clicked threshold line if exists
        if self.clicked_y is not None:
            cv2.line(vis, (0, self.clicked_y), (vis.shape[1], self.clicked_y), (0, 255, 255), 1)
            
            if self.selected_vertex is not None:
                idx, sx, sy = self.selected_vertex
                # Highlight selected vertex
                cv2.circle(vis, (sx, sy), 4, (0, 0, 255), -1)
                cv2.putText(vis, f"V{idx}", (sx+5, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
            cv2.putText(vis, f"Percentile: {self.percentile:.1f}%", (10, self.clicked_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Instructions
        cv2.putText(vis, "Click below nose to mark threshold", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_verts is not None and self.vertex_image_coords is not None:
                self.clicked_y = y
                
                # Find vertex with highest Y below this line
                self.selected_vertex, self.percentile = self.find_vertex_below_line(y)
                
                print(f"\n{'='*50}")
                print(f"CLICKED LINE at Y = {y} pixels")
                
                if self.selected_vertex is not None:
                    idx, sx, sy = self.selected_vertex
                    print(f"Selected vertex: V{idx} at image Y = {sy}")
                    print(f"This vertex is at PERCENTILE = {self.percentile:.1f}%")
                    print(f"(Meaning {self.percentile:.1f}% of vertices are above this in image)")
                    
                    # Save to file
                    with open('y_threshold.txt', 'w') as f:
                        f.write(f"percentile={self.percentile:.1f}\n")
                        f.write(f"vertex={idx}\n")
                    print(f"Saved to y_threshold.txt")
                else:
                    print("No vertices found below the clicked line")
                
                print(f"{'='*50}")
    
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        cv2.namedWindow('Y Threshold Selector')
        cv2.setMouseCallback('Y Threshold Selector', self.mouse_callback)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.current_frame = frame
            
            display = frame.copy()
            cv2.putText(display, "SPACE: capture | Click: mark threshold | ESC: exit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow('Camera', display)
            
            if self.current_verts is not None:
                vis = self.visualize(frame)
                cv2.imshow('Y Threshold Selector', vis)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):
                print("\n[*] Processing...")
                verts, trans_verts, crop_info = self.process_frame(frame)
                if verts is not None:
                    self.current_verts = verts
                    self.current_trans_verts = trans_verts
                    self.current_crop_info = crop_info
                    self.compute_vertex_image_coords()
                    self.clicked_y = None
                    self.selected_vertex = None
                    self.percentile = None
                    print(f"[OK] Computed {len(self.vertex_image_coords)} vertex positions")
                    print("Click below the alar level to mark threshold")
                else:
                    print("[ERROR] No face detected")
            
            elif key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    selector = YThresholdSelector()
    selector.run()
