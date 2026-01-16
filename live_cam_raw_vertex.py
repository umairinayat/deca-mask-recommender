#!/usr/bin/env python3
"""
DECA Raw Vertex Measurement - NO IPD CALIBRATION
=================================================
This version outputs RAW vertex distances from the FLAME mesh.
NO conversion to millimeters - just pure vertex-to-vertex distances.

FLAME mesh has 5023 vertices in fixed positions.
All measurements are in FLAME UNITS (arbitrary scale).
"""
import sys
import os
import numpy as np
import cv2
import torch
import time


class DECARawMeasurement:
    """
    Measure face dimensions using raw FLAME vertex distances.
    NO IPD calibration - just vertex-to-vertex Euclidean distance.
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.deca = None
        self.face_detector = None
        
        # ============================================================
        # FLAME VERTEX INDICES (verified with MICA)
        # These are the actual mesh vertex positions
        # ============================================================
        
        # NOSE WIDTH - Alar vertices (outer nostril edges)
        self.VERTEX_LEFT_ALAR = 2750    # Left nostril outer edge
        self.VERTEX_RIGHT_ALAR = 1610   # Right nostril outer edge
        
        # NOSE HEIGHT - Bridge (nasion) to base (subnasale)
        self.VERTEX_NASION = 3560       # Bridge of nose (between eyes)
        self.VERTEX_SUBNASALE = 3551    # Base of nose (above upper lip)
        
        # NOSE TIP
        self.VERTEX_NOSE_TIP = 3564     # Pronasale (tip of nose)
        
        # CHIN
        self.VERTEX_CHIN = 3414         # Menton (bottom of chin)
        
        # MOUTH
        self.VERTEX_MOUTH_LEFT = 3799
        self.VERTEX_MOUTH_RIGHT = 3922
        
    def initialize_deca(self):
        """Initialize DECA model and face detector"""
        print("=" * 60)
        print("DECA RAW VERTEX MEASUREMENT")
        print("NO IPD CALIBRATION - Pure vertex distances")
        print("=" * 60)
        
        try:
            from decalib.deca_measurement import DECAMeasurement
            from decalib.utils.config import cfg as deca_cfg
            import face_alignment
            
            deca_cfg.model.use_tex = False
            
            print("[*] Loading DECA model...")
            self.deca = DECAMeasurement(config=deca_cfg, device=self.device)
            print("[OK] DECA model loaded")
            
            print("[*] Loading face detector...")
            self.face_detector = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, 
                flip_input=False, 
                device=self.device
            )
            print("[OK] Face detector loaded")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_face(self, image, landmarks, target_size=224):
        """Crop and align face for DECA processing"""
        h, w = image.shape[:2]
        
        x_min, x_max = int(np.min(landmarks[:, 0])), int(np.max(landmarks[:, 0]))
        y_min, y_max = int(np.min(landmarks[:, 1])), int(np.max(landmarks[:, 1]))
        
        padding = 50
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        face_crop = image[y_min:y_max, x_min:x_max]
        face_resized = cv2.resize(face_crop, (target_size, target_size))
        
        face_tensor = torch.tensor(face_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        face_tensor = face_tensor.to(self.device)
        
        return face_tensor, (x_min, y_min, x_max, y_max)
    
    def extract_raw_measurements(self, vertices):
        """
        Extract RAW measurements from FLAME vertices.
        
        NO IPD CALIBRATION - just Euclidean distance between vertices.
        All values are in FLAME UNITS (arbitrary scale).
        
        Args:
            vertices: (5023, 3) numpy array of vertex positions
            
        Returns:
            dict: Raw vertex distances and coordinates
        """
        assert vertices.shape[0] == 5023, f"Expected 5023 vertices, got {vertices.shape[0]}"
        
        # Get specific vertex coordinates
        left_alar = vertices[self.VERTEX_LEFT_ALAR]
        right_alar = vertices[self.VERTEX_RIGHT_ALAR]
        nasion = vertices[self.VERTEX_NASION]
        subnasale = vertices[self.VERTEX_SUBNASALE]
        nose_tip = vertices[self.VERTEX_NOSE_TIP]
        chin = vertices[self.VERTEX_CHIN]
        mouth_left = vertices[self.VERTEX_MOUTH_LEFT]
        mouth_right = vertices[self.VERTEX_MOUTH_RIGHT]
        
        # ===== RAW EUCLIDEAN DISTANCES =====
        
        # 1. NOSE WIDTH (alar to alar)
        nose_width = np.linalg.norm(left_alar - right_alar)
        
        # 2. NOSE HEIGHT (nasion to subnasale)
        nose_height = np.linalg.norm(nasion - subnasale)
        
        # 3. NOSE TIP TO BASE
        tip_to_base = np.linalg.norm(nose_tip - subnasale)
        
        # 4. NOSE PROTRUSION (Z-axis depth)
        alar_plane_z = (left_alar[2] + right_alar[2]) / 2
        nose_protrusion = nose_tip[2] - alar_plane_z
        
        # 5. NOSE TO CHIN
        nose_to_chin = np.linalg.norm(subnasale - chin)
        
        # 6. MOUTH WIDTH
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        
        # 7. FACE OVERALL DIMENSIONS
        face_width = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        face_height = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        face_depth = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
        
        return {
            # RAW DISTANCES (FLAME UNITS)
            'nose_width': round(nose_width, 4),
            'nose_height': round(nose_height, 4),
            'tip_to_base': round(tip_to_base, 4),
            'nose_protrusion': round(nose_protrusion, 4),
            'nose_to_chin': round(nose_to_chin, 4),
            'mouth_width': round(mouth_width, 4),
            'face_width': round(face_width, 4),
            'face_height': round(face_height, 4),
            'face_depth': round(face_depth, 4),
            
            # VERTEX COORDINATES (for debugging)
            'left_alar_xyz': left_alar.tolist(),
            'right_alar_xyz': right_alar.tolist(),
            'nasion_xyz': nasion.tolist(),
            'subnasale_xyz': subnasale.tolist(),
            'nose_tip_xyz': nose_tip.tolist(),
            
            # VERTEX INDICES USED
            'vertices_used': {
                'left_alar': self.VERTEX_LEFT_ALAR,
                'right_alar': self.VERTEX_RIGHT_ALAR,
                'nasion': self.VERTEX_NASION,
                'subnasale': self.VERTEX_SUBNASALE,
                'nose_tip': self.VERTEX_NOSE_TIP
            }
        }
    
    def draw_measurements(self, frame, measurements, bbox, processing_time):
        """Draw measurement information on frame"""
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        text_lines = [
            "RAW FLAME VERTEX DISTANCES (no IPD conversion)",
            f"Nose Width:      {measurements['nose_width']:.4f}  (V2750-V1610)",
            f"Nose Height:     {measurements['nose_height']:.4f}  (V3560-V3551)",
            f"Tip to Base:     {measurements['tip_to_base']:.4f}",
            f"Nose Protrusion: {measurements['nose_protrusion']:.4f}",
            f"Nose to Chin:    {measurements['nose_to_chin']:.4f}",
            f"Mouth Width:     {measurements['mouth_width']:.4f}",
            f"Face Width:      {measurements['face_width']:.4f}",
            f"Processing: {processing_time:.2f}s"
        ]
        
        text_height = 22
        bg_height = len(text_lines) * text_height + 10
        cv2.rectangle(frame, (10, 10), (480, bg_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (480, bg_height), (0, 255, 0), 2)
        
        for i, line in enumerate(text_lines):
            y_pos = 28 + i * text_height
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def run_demo(self):
        """Run live camera demo"""
        if not self.initialize_deca():
            return
        
        print("\nControls:")
        print("  SPACE - Capture and measure face")
        print("  ESC   - Quit")
        print("=" * 60)
        
        cap = cv2.VideoCapture(0)
        measurement_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]
                cv2.putText(frame, f"RAW Vertex Demo - Measurements: {measurement_count}",
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE: Measure | ESC: Quit",
                           (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('DECA Raw Vertex Demo', frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space - measure
                    print(f"\n[*] Measurement #{measurement_count + 1}")
                    start_time = time.time()
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    landmarks = self.face_detector.get_landmarks(frame_rgb)
                    
                    if landmarks is None or len(landmarks) == 0:
                        print("[WARN] No face detected")
                        continue
                    
                    landmarks = landmarks[0]
                    print(f"[OK] Face detected")
                    
                    face_tensor, bbox = self.preprocess_face(frame_rgb, landmarks)
                    
                    with torch.no_grad():
                        codedict = self.deca.encode(face_tensor)
                        opdict = self.deca.decode(codedict)
                        
                        # Get 5023 VERTICES
                        vertices = opdict['verts'][0].cpu().numpy()
                        
                        # Calculate RAW measurements (NO IPD CONVERSION)
                        measurements = self.extract_raw_measurements(vertices)
                        
                        processing_time = time.time() - start_time
                        
                        print(f"\n{'='*60}")
                        print(f"RAW FLAME VERTEX DISTANCES (NO IPD CALIBRATION)")
                        print(f"{'='*60}")
                        print(f"Nose Width:      {measurements['nose_width']:.4f}  (V{self.VERTEX_LEFT_ALAR}-V{self.VERTEX_RIGHT_ALAR})")
                        print(f"Nose Height:     {measurements['nose_height']:.4f}  (V{self.VERTEX_NASION}-V{self.VERTEX_SUBNASALE})")
                        print(f"Tip to Base:     {measurements['tip_to_base']:.4f}")
                        print(f"Nose Protrusion: {measurements['nose_protrusion']:.4f}")
                        print(f"Nose to Chin:    {measurements['nose_to_chin']:.4f}")
                        print(f"Mouth Width:     {measurements['mouth_width']:.4f}")
                        print(f"Face Width:      {measurements['face_width']:.4f}")
                        print(f"Face Height:     {measurements['face_height']:.4f}")
                        print(f"{'='*60}")
                        print(f"\nVertex Coordinates:")
                        print(f"  Left Alar (V2750):  {measurements['left_alar_xyz']}")
                        print(f"  Right Alar (V1610): {measurements['right_alar_xyz']}")
                        print(f"  Nasion (V3560):     {measurements['nasion_xyz']}")
                        print(f"  Subnasale (V3551):  {measurements['subnasale_xyz']}")
                        print(f"  Nose Tip (V3564):   {measurements['nose_tip_xyz']}")
                        print(f"Processing time: {processing_time:.2f}s")
                        print(f"{'='*60}")
                        
                        measurement_count += 1
                        
                        self.draw_measurements(frame, measurements, bbox, processing_time)
                        cv2.imshow('DECA Raw Vertex Demo', frame)
                        cv2.waitKey(2000)
                
                elif key == 27:  # ESC
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\nDemo complete! Total measurements: {measurement_count}")


def main():
    demo = DECARawMeasurement()
    demo.run_demo()


if __name__ == "__main__":
    main()
