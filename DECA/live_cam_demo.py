#!/usr/bin/env python3
"""
DECA Live Camera Demo
Real-time 3D face reconstruction and measurement extraction
"""
import sys
import os
import numpy as np
import cv2
import torch
import time
from datetime import datetime

class DECALiveDemo:
    def __init__(self):
        self.device = 'cpu'  # Use CPU for ARM64 compatibility
        self.deca = None
        self.face_detector = None
        self.measurement_history = []
        
    def initialize_deca(self):
        """Initialize DECA model and face detector"""
        print("🔥 Initializing DECA Live Demo...")
        
        try:
            # Use DECAMeasurement instead of DECA to avoid C++ compilation
            from decalib.deca_measurement import DECAMeasurement
            from decalib.utils.config import cfg as deca_cfg
            import face_alignment
            
            # Configure DECA
            deca_cfg.model.use_tex = False  # Disable texture for speed
            # No need to set rasterizer_type - DECAMeasurement doesn't use rendering
            
            # Initialize DECA (measurement-only mode - no C++ compilation needed!)
            print("🔄 Loading DECA model (measurement mode)...")
            self.deca = DECAMeasurement(config=deca_cfg, device=self.device)
            print("✅ DECA model loaded")
            
            # Initialize face detector
            print("🔄 Loading face detector...")
            self.face_detector = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, 
                flip_input=False, 
                device=self.device
            )
            print("✅ Face detector loaded")
            
            return True
            
        except Exception as e:
            print(f" Initialization error: {e}")
            return False
    
    def preprocess_face(self, image, landmarks, target_size=224):
        """Crop and align face for DECA processing"""
        h, w = image.shape[:2]
        
        # Get face bounding box from landmarks
        x_min, x_max = int(np.min(landmarks[:, 0])), int(np.max(landmarks[:, 0]))
        y_min, y_max = int(np.min(landmarks[:, 1])), int(np.max(landmarks[:, 1]))
        
        # Add padding
        padding = 50
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        
        # Crop face region
        face_crop = image[y_min:y_max, x_min:x_max]
        
        # Resize to target size
        face_resized = cv2.resize(face_crop, (target_size, target_size))
        
        # Convert to tensor [1, 3, 224, 224]
        face_tensor = torch.tensor(face_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        face_tensor = face_tensor.to(self.device)
        
        return face_tensor, (x_min, y_min, x_max, y_max)
    
    # ============ CONFIGURABLE PARAMETERS ============
    # Set YOUR actual IPD (inter-pupillary distance) in mm for accurate measurements
    # Measure with a ruler: distance between your left and right pupil centers
    # Average adult IPD is 63mm, but ranges from 54-74mm
    YOUR_IPD_MM = 63.0  # <-- CHANGE THIS TO YOUR MEASURED IPD!
    
    # 68-Point Landmark indices for measurements
    LMK_NOSE_LEFT = 31       # Left alar (nostril wing)
    LMK_NOSE_RIGHT = 35      # Right alar (nostril wing)
    LMK_NASION = 27          # Bridge of nose (between eyes)
    LMK_PRONASALE = 30       # Nose tip
    LMK_SUBNASALE = 33       # Nose base (under nose)
    LMK_CHIN = 8             # Chin tip (menton)
    LMK_LEFT_EYE_OUTER = 36  # Left eye outer corner
    LMK_LEFT_EYE_INNER = 39  # Left eye inner corner
    LMK_RIGHT_EYE_INNER = 42 # Right eye inner corner
    LMK_RIGHT_EYE_OUTER = 45 # Right eye outer corner
    
    def extract_measurements(self, landmarks3d, vertices=None):
        """
        Extract accurate face measurements from 3D landmarks.
        
        Args:
            landmarks3d: (68, 3) numpy array of 3D landmark positions
            vertices: (5023, 3) optional, for additional vertex-based measurements
            
        Returns:
            dict with measurements in both FLAME units and millimeters
        """
        # ===== CALCULATE CONVERSION FACTOR USING IPD =====
        # Use eye centers for more accurate IPD
        left_eye_center = (landmarks3d[self.LMK_LEFT_EYE_OUTER] + landmarks3d[self.LMK_LEFT_EYE_INNER]) / 2
        right_eye_center = (landmarks3d[self.LMK_RIGHT_EYE_INNER] + landmarks3d[self.LMK_RIGHT_EYE_OUTER]) / 2
        ipd_flame = np.linalg.norm(left_eye_center - right_eye_center)
        
        # Conversion factor: mm per FLAME unit
        conversion_factor = self.YOUR_IPD_MM / ipd_flame
        
        # ===== 1. NOSE WIDTH (Alar Base) =====
        # Distance between outer edges of nostrils
        nose_left = landmarks3d[self.LMK_NOSE_LEFT]
        nose_right = landmarks3d[self.LMK_NOSE_RIGHT]
        nose_width_flame = np.linalg.norm(nose_left - nose_right)
        nose_width_mm = nose_width_flame * conversion_factor
        
        # ===== 2. DOWN NOSE DISTANCE (Nasion to Subnasale) =====
        # Vertical length of nose from bridge to base
        nasion = landmarks3d[self.LMK_NASION]
        subnasale = landmarks3d[self.LMK_SUBNASALE]
        down_nose_flame = np.linalg.norm(nasion - subnasale)
        down_nose_mm = down_nose_flame * conversion_factor
        
        # ===== 3. NOSE HEIGHT (Tip Protrusion) =====
        # How far the nose tip protrudes from face plane
        pronasale = landmarks3d[self.LMK_PRONASALE]  # Nose tip
        # Z-axis difference (depth protrusion)
        nose_height_z_flame = abs(pronasale[2] - subnasale[2])
        nose_height_z_mm = nose_height_z_flame * conversion_factor
        # Full 3D distance from tip to base
        nose_height_3d_flame = np.linalg.norm(pronasale - subnasale)
        nose_height_3d_mm = nose_height_3d_flame * conversion_factor
        
        # ===== 4. NOSE TO CHIN (Lower Face Height) =====
        chin = landmarks3d[self.LMK_CHIN]
        nose_to_chin_flame = np.linalg.norm(subnasale - chin)
        nose_to_chin_mm = nose_to_chin_flame * conversion_factor
        
        # ===== 5. FACE DIMENSIONS (from vertices if available) =====
        if vertices is not None:
            face_width_flame = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
            face_height_flame = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
            face_depth_flame = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
        else:
            face_width_flame = face_height_flame = face_depth_flame = 0.0
        
        return {
            # Millimeter measurements (calibrated)
            'nose_width_mm': round(nose_width_mm, 1),
            'down_nose_mm': round(down_nose_mm, 1),
            'nose_height_z_mm': round(nose_height_z_mm, 1),
            'nose_height_3d_mm': round(nose_height_3d_mm, 1),
            'nose_to_chin_mm': round(nose_to_chin_mm, 1),
            
            # Raw FLAME units (for debugging)
            'nose_width_flame': nose_width_flame,
            'down_nose_flame': down_nose_flame,
            'nose_height_z_flame': nose_height_z_flame,
            'nose_height_3d_flame': nose_height_3d_flame,
            'nose_to_chin_flame': nose_to_chin_flame,
            
            # Face dimensions
            'face_width_flame': face_width_flame,
            'face_height_flame': face_height_flame,
            'face_depth_flame': face_depth_flame,
            
            # Calibration info
            'conversion_factor': conversion_factor,
            'ipd_mm': self.YOUR_IPD_MM,
            'ipd_flame': ipd_flame,
            'vertex_count': vertices.shape[0] if vertices is not None else 0
        }
    
    def draw_measurements(self, frame, measurements, bbox, processing_time):
        """Draw measurement information on frame"""
        x_min, y_min, x_max, y_max = bbox
        
        # Draw face bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Measurement text - now in MILLIMETERS!
        text_lines = [
            f"Nose Measurements (mm) [IPD={measurements['ipd_mm']:.0f}mm]:",
            f"Nose Width:       {measurements['nose_width_mm']:.1f} mm",
            f"Down Nose Dist:   {measurements['down_nose_mm']:.1f} mm",
            f"Nose Height (Z):  {measurements['nose_height_z_mm']:.1f} mm",
            f"Nose Height (3D): {measurements['nose_height_3d_mm']:.1f} mm",
            f"Nose to Chin:     {measurements['nose_to_chin_mm']:.1f} mm",
            f"Processing: {processing_time:.2f}s"
        ]
        
        # Draw text background
        text_height = 25
        bg_height = len(text_lines) * text_height + 10
        cv2.rectangle(frame, (10, 10), (450, bg_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, bg_height), (0, 255, 0), 2)
        
        # Draw text
        for i, line in enumerate(text_lines):
            y_pos = 30 + i * text_height
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    
    def save_measurement(self, measurements):
        """Save measurement to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        measurement_data = {
            'timestamp': timestamp,
            'nose_width_mm': measurements['nose_width_mm'],
            'down_nose_mm': measurements['down_nose_mm'],
            'nose_height_z_mm': measurements['nose_height_z_mm'],
            'nose_height_3d_mm': measurements['nose_height_3d_mm'],
            'nose_to_chin_mm': measurements['nose_to_chin_mm']
        }
        self.measurement_history.append(measurement_data)
        
        # Keep only last 10 measurements
        if len(self.measurement_history) > 10:
            self.measurement_history.pop(0)
    
    def print_statistics(self):
        """Print measurement statistics"""
        if len(self.measurement_history) < 2:
            print("Need at least 2 measurements for statistics")
            return
        
        nose_widths = [m['nose_width_mm'] for m in self.measurement_history]
        down_noses = [m['down_nose_mm'] for m in self.measurement_history]
        nose_heights = [m['nose_height_3d_mm'] for m in self.measurement_history]
        
        print(f"\n📊 Measurement Statistics (last {len(self.measurement_history)} measurements):")
        print(f"   Nose Width:     {np.mean(nose_widths):.1f} ± {np.std(nose_widths):.1f} mm  (CV: {(np.std(nose_widths)/np.mean(nose_widths)*100):.1f}%)")
        print(f"   Down Nose:      {np.mean(down_noses):.1f} ± {np.std(down_noses):.1f} mm  (CV: {(np.std(down_noses)/np.mean(down_noses)*100):.1f}%)")
        print(f"   Nose Height:    {np.mean(nose_heights):.1f} ± {np.std(nose_heights):.1f} mm  (CV: {(np.std(nose_heights)/np.mean(nose_heights)*100):.1f}%)")
    
    def run_demo(self):
        """Run live camera demo"""
        if not self.initialize_deca():
            return
            
        print("\n📷 DECA Live Camera Demo")
        print("=" * 50)
        print("Controls:")
        print("  SPACE - Capture and measure face")
        print("  S     - Show measurement statistics")
        print("  ESC   - Quit")
        print("=" * 50)
        
        cap = cv2.VideoCapture(0)
        measurement_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to RGB for face detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display instructions
                cv2.putText(frame, f"DECA Live Demo - Measurements: {measurement_count}", 
                           (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE: Measure | S: Stats | ESC: Quit", 
                           (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('DECA Live Demo', frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space - measure face
                    print(f"\n🔍 Measurement #{measurement_count + 1}")
                    start_time = time.time()
                    
                    # Detect face landmarks
                    landmarks = self.face_detector.get_landmarks(frame_rgb)
                    
                    if landmarks is None or len(landmarks) == 0:
                        print("❌ No face detected - ensure good lighting and face visibility")
                        continue
                    
                    landmarks = landmarks[0]  # Use first detected face
                    print(f"✅ Face detected with {len(landmarks)} landmarks")
                    
                    # Preprocess face for DECA
                    face_tensor, bbox = self.preprocess_face(frame_rgb, landmarks)
                    
                    # Run DECA reconstruction
                    with torch.no_grad():
                        codedict = self.deca.encode(face_tensor)
                        opdict = self.deca.decode(codedict)  # DECAMeasurement returns only opdict
                        
                        # Extract 3D vertices AND 3D landmarks
                        vertices = opdict['verts'][0].cpu().numpy()
                        landmarks3d = opdict['landmarks3d'][0].cpu().numpy()  # 68 3D landmarks
                        
                        # Calculate measurements using landmarks (more accurate)
                        measurements = self.extract_measurements(landmarks3d, vertices)
                        
                        processing_time = time.time() - start_time
                        
                        # Display results in MILLIMETERS
                        print(f"📏 Nose Measurements (mm):")
                        print(f"   Nose Width:      {measurements['nose_width_mm']:.1f} mm")
                        print(f"   Down Nose Dist:  {measurements['down_nose_mm']:.1f} mm")
                        print(f"   Nose Height (Z): {measurements['nose_height_z_mm']:.1f} mm")
                        print(f"   Nose Height 3D:  {measurements['nose_height_3d_mm']:.1f} mm")
                        print(f"   Nose to Chin:    {measurements['nose_to_chin_mm']:.1f} mm")
                        print(f"   Processing time: {processing_time:.2f}s")
                        
                        # Save measurement
                        self.save_measurement(measurements)
                        measurement_count += 1
                        
                        # Draw measurements on frame
                        self.draw_measurements(frame, measurements, bbox, processing_time)
                        cv2.imshow('DECA Live Demo', frame)
                        cv2.waitKey(2000)  # Show result for 2 seconds
                
                elif key == ord('s'):  # S - show statistics
                    self.print_statistics()
                
                elif key == 27:  # ESC - quit
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n🎉 Demo complete! Total measurements: {measurement_count}")
            if measurement_count > 0:
                self.print_statistics()
                print("\n📝 Next steps:")
                print("   1. Calibrate FLAME units to millimeters using ArUco markers")
                print("   2. Test distance independence by measuring at different distances")
                print("   3. Compare accuracy with MediaPipe approach")

def main():
    """Main function"""
    demo = DECALiveDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
