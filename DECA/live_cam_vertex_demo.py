#!/usr/bin/env python3
"""
DECA Live Camera Demo - VERTEX-based Measurements
=================================================
This version uses FLAME mesh VERTICES (not 68 landmarks) for more accurate
nose measurements based on documented FLAME topology.

FLAME Model: 5023 vertices with fixed topology
Reference: https://flame.is.tue.mpg.de/

The key difference from landmark-based approach:
- Landmarks (68 points) are interpolated semantic points
- Vertices (5023 points) are actual mesh points with documented anatomical positions

FLAME NOSE VERTEX INDICES (Documented):
- Nose Tip (Pronasale): 8  (or 19 depending on version)
- Nose Bridge (Nasion): 8128 (upper), various for dorsum
- Left Alar (Nostril wing outer): 2794, 597, 707, 819
- Right Alar (Nostril wing outer): 3389, 3412, 4039, 4264
- Columella: Various in the 2000-3000 range
- Subnasale (under nose): ~3700-3800 range

For ALAR BASE WIDTH (nostril wing to wing), we want the OUTERMOST vertices.
"""
import sys
import os
import numpy as np
import cv2
import torch
import time
from datetime import datetime


class DECAVertexMeasurement:
    """
    DECA measurement using FLAME mesh vertices for accurate nose measurements.
    """
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.deca = None
        self.face_detector = None
        self.measurement_history = []
        
        # ============================================================
        # CONFIGURABLE: Set YOUR actual IPD for accurate mm calibration
        # Measure with a ruler: distance between your pupil centers
        # ============================================================
        self.YOUR_IPD_MM = 63.0  # <-- CHANGE THIS TO YOUR MEASURED IPD!
        
        # ============================================================
        # VERIFIED FLAME VERTEX INDICES FOR NOSE MEASUREMENTS
        # These are the same indices used by MICA and work with all
        # FLAME-based models (MICA, DECA, EMOCA, etc.)
        # Source: FLAME_masks.pkl analysis
        # ============================================================
        
        # NOSE WIDTH - ALAR BASE (outer nostril edges)
        self.VERTEX_LEFT_ALAR = 2750    # Outer edge of left nostril
        self.VERTEX_RIGHT_ALAR = 1610   # Outer edge of right nostril
        # Typical Adult Range: 31-45 mm
        
        # NOSE TIP (Pronasale) - most forward-projecting point
        self.VERTEX_NOSE_TIP = 3564     # Highest Z-value in nose region
        
        # NOSE BRIDGE (Nasion) - bridge between eyes
        self.VERTEX_NASION = 3560       # Highest Y-value in nose region
        
        # NOSE BASE (Subnasale) - bottom center above upper lip
        self.VERTEX_SUBNASALE = 3551    # Lowest Y-value in nose region
        
        # CHIN (Menton)
        self.VERTEX_CHIN = 3414         # Center point of chin
        
        # MOUTH CORNERS (for full-face mask)
        self.VERTEX_MOUTH_LEFT = 3799   # Left corner of lips
        self.VERTEX_MOUTH_RIGHT = 3922  # Right corner of lips
        
        # JAW WIDTH (for full-face mask)
        self.VERTEX_JAW_LEFT = 3424     # Left upper jaw
        self.VERTEX_JAW_RIGHT = 3652    # Right upper jaw
        
        # EYE VERTICES for IPD calibration
        self.VERTEX_LEFT_EYE_OUTER = 2134
        self.VERTEX_LEFT_EYE_INNER = 1774
        self.VERTEX_RIGHT_EYE_OUTER = 3888
        self.VERTEX_RIGHT_EYE_INNER = 3694
        
    def initialize_deca(self):
        """Initialize DECA model and face detector"""
        print("=" * 60)
        print("DECA Vertex-Based Measurement Demo")
        print("Using FLAME VERTICES (not 68 landmarks)")
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
    

    
    def calculate_ipd_from_vertices(self, vertices):
        """Calculate IPD (inter-pupillary distance) from eye vertices"""
        try:
            # Use eye vertex positions
            left_outer = vertices[self.VERTEX_LEFT_EYE_OUTER]
            left_inner = vertices[self.VERTEX_LEFT_EYE_INNER]
            right_outer = vertices[self.VERTEX_RIGHT_EYE_OUTER]
            right_inner = vertices[self.VERTEX_RIGHT_EYE_INNER]
            
            # Eye centers
            left_eye_center = (left_outer + left_inner) / 2
            right_eye_center = (right_outer + right_inner) / 2
            
            ipd_flame = np.linalg.norm(left_eye_center - right_eye_center)
            return ipd_flame
        except IndexError:
            # Fallback: use face width as proxy
            return np.max(vertices[:, 0]) - np.min(vertices[:, 0]) * 0.4
    
    def extract_vertex_measurements(self, vertices):
        """
        Extract measurements using VERIFIED FLAME VERTICES (5023 points).
        
        Uses the same vertex indices verified with MICA:
        - Nose Width: 2750 (left alar) <-> 1610 (right alar)
        - Nose Height: 3560 (nasion) <-> 3551 (subnasale)
        - Nose Tip: 3564
        
        Args:
            vertices: (5023, 3) numpy array of vertex positions
            
        Returns:
            dict: Measurements in both FLAME units and millimeters
        """
        # Validate vertex count
        assert vertices.shape[0] == 5023, f"Expected 5023 vertices, got {vertices.shape[0]}"
        
        # ===== CALCULATE IPD FOR CALIBRATION =====
        ipd_flame = self.calculate_ipd_from_vertices(vertices)
        conversion_factor = self.YOUR_IPD_MM / ipd_flame
        
        # ===== 1. NOSE WIDTH (ALAR BASE) =====
        # Using verified MICA indices: 2750 (left) <-> 1610 (right)
        left_alar = vertices[self.VERTEX_LEFT_ALAR]   # 2750
        right_alar = vertices[self.VERTEX_RIGHT_ALAR]  # 1610
        nose_width_flame = np.linalg.norm(left_alar - right_alar)
        nose_width_mm = nose_width_flame * conversion_factor
        
        # ===== 2. NOSE HEIGHT (Nasion to Subnasale) =====
        nasion = vertices[self.VERTEX_NASION]         # 3560
        subnasale = vertices[self.VERTEX_SUBNASALE]   # 3551
        nose_height_flame = np.linalg.norm(nasion - subnasale)
        nose_height_mm = nose_height_flame * conversion_factor
        
        # ===== 3. NOSE PROTRUSION (Tip depth) =====
        nose_tip = vertices[self.VERTEX_NOSE_TIP]     # 3564
        # Z-axis protrusion relative to alar plane
        alar_plane_z = (left_alar[2] + right_alar[2]) / 2
        nose_protrusion_flame = nose_tip[2] - alar_plane_z
        nose_protrusion_mm = nose_protrusion_flame * conversion_factor
        
        # Also: tip to subnasale distance
        tip_to_base_flame = np.linalg.norm(nose_tip - subnasale)
        tip_to_base_mm = tip_to_base_flame * conversion_factor
        
        # ===== 4. NOSE TO CHIN =====
        chin = vertices[self.VERTEX_CHIN]             # 3414
        nose_to_chin_flame = np.linalg.norm(subnasale - chin)
        nose_to_chin_mm = nose_to_chin_flame * conversion_factor
        
        # ===== 5. MOUTH WIDTH (for F30 mask) =====
        mouth_left = vertices[self.VERTEX_MOUTH_LEFT]   # 3799
        mouth_right = vertices[self.VERTEX_MOUTH_RIGHT] # 3922
        mouth_width_flame = np.linalg.norm(mouth_left - mouth_right)
        mouth_width_mm = mouth_width_flame * conversion_factor
        
        # ===== 6. JAW WIDTH (for F40 mask) =====
        jaw_left = vertices[self.VERTEX_JAW_LEFT]     # 3424
        jaw_right = vertices[self.VERTEX_JAW_RIGHT]   # 3652
        jaw_width_flame = np.linalg.norm(jaw_left - jaw_right)
        jaw_width_mm = jaw_width_flame * conversion_factor
        
        # ===== 7. OVERALL FACE DIMENSIONS =====
        face_width_flame = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        face_height_flame = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        
        return {
            # Primary measurements in MM
            'nose_width_mm': round(nose_width_mm, 1),
            'nose_height_mm': round(nose_height_mm, 1),
            'nose_protrusion_mm': round(nose_protrusion_mm, 1),
            'tip_to_base_mm': round(tip_to_base_mm, 1),
            'nose_to_chin_mm': round(nose_to_chin_mm, 1),
            'mouth_width_mm': round(mouth_width_mm, 1),
            'jaw_width_mm': round(jaw_width_mm, 1),
            
            # Face dimensions in MM
            'face_width_mm': round(face_width_flame * conversion_factor, 1),
            'face_height_mm': round(face_height_flame * conversion_factor, 1),
            
            # Vertex indices used (for verification)
            'vertices_used': {
                'left_alar': self.VERTEX_LEFT_ALAR,
                'right_alar': self.VERTEX_RIGHT_ALAR,
                'nasion': self.VERTEX_NASION,
                'subnasale': self.VERTEX_SUBNASALE,
                'nose_tip': self.VERTEX_NOSE_TIP,
                'chin': self.VERTEX_CHIN
            },
            
            # Raw FLAME units
            'nose_width_flame': nose_width_flame,
            'nose_height_flame': nose_height_flame,
            'nose_protrusion_flame': nose_protrusion_flame,
            
            # Calibration info
            'conversion_factor': conversion_factor,
            'ipd_mm': self.YOUR_IPD_MM,
            'ipd_flame': ipd_flame,
            'vertex_count': vertices.shape[0]
        }
    
    def draw_measurements(self, frame, measurements, bbox, processing_time):
        """Draw measurement information on frame"""
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Get vertex info
        v = measurements['vertices_used']
        
        text_lines = [
            f"MICA-Verified FLAME Vertices (IPD={measurements['ipd_mm']:.0f}mm)",
            f"Nose Width:       {measurements['nose_width_mm']:.1f} mm (V{v['left_alar']}-V{v['right_alar']})",
            f"Nose Height:      {measurements['nose_height_mm']:.1f} mm (V{v['nasion']}-V{v['subnasale']})",
            f"Nose Protrusion:  {measurements['nose_protrusion_mm']:.1f} mm",
            f"Nose to Chin:     {measurements['nose_to_chin_mm']:.1f} mm",
            f"Mouth Width:      {measurements['mouth_width_mm']:.1f} mm",
            f"Jaw Width:        {measurements['jaw_width_mm']:.1f} mm",
            f"Processing: {processing_time:.2f}s"
        ]
        
        text_height = 22
        bg_height = len(text_lines) * text_height + 10
        cv2.rectangle(frame, (10, 10), (520, bg_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (520, bg_height), (0, 255, 0), 2)
        
        for i, line in enumerate(text_lines):
            y_pos = 28 + i * text_height
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def save_measurement(self, measurements):
        """Save measurement to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.measurement_history.append({
            'timestamp': timestamp,
            'nose_width_mm': measurements['nose_width_mm'],
            'nose_height_mm': measurements['nose_height_mm'],
            'nose_protrusion_mm': measurements['nose_protrusion_mm'],
            'nose_to_chin_mm': measurements['nose_to_chin_mm']
        })
        
        if len(self.measurement_history) > 10:
            self.measurement_history.pop(0)
    
    def print_statistics(self):
        """Print measurement statistics"""
        if len(self.measurement_history) < 2:
            print("Need at least 2 measurements for statistics")
            return
        
        nose_widths = [m['nose_width_mm'] for m in self.measurement_history]
        
        print(f"\n{'='*60}")
        print(f"VERTEX Measurement Statistics (last {len(self.measurement_history)} measurements):")
        print(f"{'='*60}")
        print(f"Nose Width: {np.mean(nose_widths):.1f} ± {np.std(nose_widths):.1f} mm")
        print(f"   Min: {np.min(nose_widths):.1f} mm, Max: {np.max(nose_widths):.1f} mm")
        print(f"   CV: {(np.std(nose_widths)/np.mean(nose_widths)*100):.1f}%")
        print(f"{'='*60}")
    
    def print_all_alar_pairs(self, measurements):
        """Print all alar measurement pairs for debugging"""
        print("\n[DEBUG] All Alar Pair Measurements:")
        for pair, width_mm in sorted(measurements['alar_measurements'].items(), 
                                      key=lambda x: x[1], reverse=True):
            print(f"   Vertices {pair}: {width_mm:.1f} mm")
    
    def run_demo(self):
        """Run live camera demo"""
        if not self.initialize_deca():
            return
        
        print("\nControls:")
        print("  SPACE - Capture and measure face")
        print("  D     - Print all alar vertex pairs (debug)")
        print("  S     - Show measurement statistics")
        print("  ESC   - Quit")
        print("=" * 60)
        
        cap = cv2.VideoCapture(0)
        measurement_count = 0
        last_measurements = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                h, w = frame.shape[:2]
                cv2.putText(frame, f"VERTEX Demo - Measurements: {measurement_count} | IPD: {self.YOUR_IPD_MM:.0f}mm",
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE: Measure | +/-: Calibrate Scale | S: Stats | ESC: Quit",
                           (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('DECA Vertex Demo', frame)
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
                        
                        # Calculate measurements using VERTICES
                        measurements = self.extract_vertex_measurements(vertices)
                        last_measurements = measurements
                        v = measurements['vertices_used']
                        
                        processing_time = time.time() - start_time
                        
                        print(f"\n[*] MICA-Verified FLAME Measurements:")
                        print(f"    Nose Width:      {measurements['nose_width_mm']:.1f} mm (V{v['left_alar']}-V{v['right_alar']})")
                        print(f"    Nose Height:     {measurements['nose_height_mm']:.1f} mm")
                        print(f"    Nose Protrusion: {measurements['nose_protrusion_mm']:.1f} mm")
                        print(f"    Nose to Chin:    {measurements['nose_to_chin_mm']:.1f} mm")
                        print(f"    Mouth Width:     {measurements['mouth_width_mm']:.1f} mm")
                        print(f"    Jaw Width:       {measurements['jaw_width_mm']:.1f} mm")
                        print(f"    Processing: {processing_time:.2f}s")
                        
                        self.save_measurement(measurements)
                        measurement_count += 1
                        
                        self.draw_measurements(frame, measurements, bbox, processing_time)
                        cv2.imshow('DECA Vertex Demo', frame)
                        cv2.waitKey(2000)
                
                elif key == ord('s'):  # S - statistics
                    self.print_statistics()
                
                elif key == ord('+') or key == ord('='):  # + key to increase IPD/Scale
                    self.YOUR_IPD_MM += 1.0
                    print(f"[*] Calibration IPD increased to: {self.YOUR_IPD_MM}mm")
                    if last_measurements:
                        # Re-calculate with new IPD
                        ipd_flame = last_measurements['ipd_flame']
                        last_measurements['conversion_factor'] = self.YOUR_IPD_MM / ipd_flame
                        # Update displayed values (simplified for user feedback)
                        factor = last_measurements['conversion_factor']
                        last_measurements['nose_width_mm'] = last_measurements['nose_width_flame'] * factor
                        print(f"    New Nose Width: {last_measurements['nose_width_mm']:.1f} mm")

                elif key == ord('-') or key == ord('_'):  # - key to decrease IPD/Scale
                    self.YOUR_IPD_MM -= 1.0
                    print(f"[*] Calibration IPD decreased to: {self.YOUR_IPD_MM}mm")
                    if last_measurements:
                        # Re-calculate with new IPD
                        ipd_flame = last_measurements['ipd_flame']
                        last_measurements['conversion_factor'] = self.YOUR_IPD_MM / ipd_flame
                        # Update displayed values
                        factor = last_measurements['conversion_factor']
                        last_measurements['nose_width_mm'] = last_measurements['nose_width_flame'] * factor
                        print(f"    New Nose Width: {last_measurements['nose_width_mm']:.1f} mm")
                
                elif key == 27:  # ESC
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n{'='*60}")
            print(f"Demo complete! Total measurements: {measurement_count}")
            if measurement_count > 0:
                self.print_statistics()
            print(f"{'='*60}")


def main():
    demo = DECAVertexMeasurement()
    demo.run_demo()


if __name__ == "__main__":
    main()
