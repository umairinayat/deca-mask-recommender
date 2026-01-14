#!/usr/bin/env python3
"""
CPAP Mask Measurement System
Captures 3 critical facial measurements for mask sizing:
1. Nose width (alar base)
2. Cheekbone width (zygion-zygion)
3. Nose-to-chin distance
"""
import sys
import os
import json
import numpy as np
import cv2
import torch
import time
from datetime import datetime
from pathlib import Path

class CPAPMeasurementSystem:
    # Constants
    FACE_PADDING = 50
    TARGET_SIZE = 224
    RESULT_DISPLAY_MS = 2000
    MAX_VERTICES = 5023
    
    # 68-Point Landmark indices for CPAP measurements (dlib/FLAME format)
    # These are more accurate than using mesh vertex indices
    LMK_NOSE_LEFT = 31      # Left alar (nostril)
    LMK_NOSE_RIGHT = 35     # Right alar (nostril)
    LMK_CHEEK_LEFT = 4      # Left zygion (cheekbone)
    LMK_CHEEK_RIGHT = 12    # Right zygion (cheekbone)
    LMK_SUBNASALE = 33      # Nose base
    LMK_CHIN = 8            # Menton (chin tip)
    
    # Eye landmarks for IPD calibration (use eye centers)
    LMK_LEFT_EYE_OUTER = 36   # Left eye outer corner
    LMK_LEFT_EYE_INNER = 39   # Left eye inner corner
    LMK_RIGHT_EYE_INNER = 42  # Right eye inner corner
    LMK_RIGHT_EYE_OUTER = 45  # Right eye outer corner
    
    # Average inter-pupillary distance for mm calibration (population average)
    AVERAGE_IPD_MM = 63.0
    
    def __init__(self):
        # Auto-detect GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.deca = None
        self.face_detector = None
        # Use script directory as base for results
        self.results_dir = Path(__file__).parent.parent / 'results'
        self.session_dir = None
        self.measurement_count = 0
        
        # Conversion factor (calculated per-face using IPD)
        self.conversion_factor = None
        
    def get_next_session_index(self):
        """Find the next available session index"""
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)
            return 1
        
        existing_sessions = [d for d in self.results_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not existing_sessions:
            return 1
        
        max_index = max([int(d.name) for d in existing_sessions])
        return max_index + 1
    
    def initialize_session(self):
        """Initialize a new measurement session"""
        session_index = self.get_next_session_index()
        self.session_dir = self.results_dir / str(session_index)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Results will be saved to: {self.session_dir}")
        return session_index
    
    def initialize_deca(self):
        """Initialize DECA model and face detector"""
        print("[*] Initializing CPAP Measurement System...")
        
        try:
            # Use measurement-only DECA (no rendering required)
            from decalib.deca_measurement import DECAMeasurement
            from decalib.utils.config import cfg as deca_cfg
            import face_alignment
            
            # Configure DECA
            deca_cfg.model.use_tex = False
            
            # Initialize DECA (measurement-only mode)
            print("[*] Loading DECA model...")
            self.deca = DECAMeasurement(config=deca_cfg, device=self.device)
            print("[OK] DECA model loaded")
            
            # Initialize face detector
            print("[*] Loading face detector...")
            self.face_detector = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                flip_input=False,
                device=self.device
            )
            print("[OK] Face detector loaded")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Initialization error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_face(self, image, landmarks, target_size=None):
        """Crop and align face for DECA processing
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            landmarks: Face landmarks as numpy array (68, 2)
            target_size: Output size (default: TARGET_SIZE)
            
        Returns:
            tuple: (face_tensor, bounding_box)
            
        Raises:
            ValueError: If face crop is invalid
        """
        if target_size is None:
            target_size = self.TARGET_SIZE
            
        h, w = image.shape[:2]
        
        # Get face bounding box
        x_min, x_max = int(np.min(landmarks[:, 0])), int(np.max(landmarks[:, 0]))
        y_min, y_max = int(np.min(landmarks[:, 1])), int(np.max(landmarks[:, 1]))
        
        # Add padding
        x_min = max(0, x_min - self.FACE_PADDING)
        x_max = min(w, x_max + self.FACE_PADDING)
        y_min = max(0, y_min - self.FACE_PADDING)
        y_max = min(h, y_max + self.FACE_PADDING)
        
        # Crop face
        face_crop = image[y_min:y_max, x_min:x_max]
        
        # Validate crop
        if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            raise ValueError("Invalid face crop: region too small or empty")
        
        # Resize
        face_resized = cv2.resize(face_crop, (target_size, target_size))
        
        # Convert to tensor (more efficient)
        face_tensor = torch.from_numpy(face_resized.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        face_tensor = face_tensor.to(self.device)
        
        return face_tensor, (x_min, y_min, x_max, y_max)
    
    def extract_cpap_measurements(self, landmarks):
        """
        Extract 3 critical CPAP measurements from DECA 68-point landmarks
        
        Args:
            landmarks: (68, 3) numpy array of 3D landmark positions
            
        Returns:
            dict with measurements in both FLAME units and millimeters
            
        Raises:
            ValueError: If landmark array is invalid
        """
        # Validate landmark array
        if landmarks.shape[0] != 68:
            raise ValueError(f"Expected 68 landmarks, got {landmarks.shape[0]}")
        
        # Calculate IPD for mm calibration using eye centers
        # This is more accurate than using just inner or outer corners
        left_eye_center = (landmarks[self.LMK_LEFT_EYE_OUTER] + landmarks[self.LMK_LEFT_EYE_INNER]) / 2
        right_eye_center = (landmarks[self.LMK_RIGHT_EYE_INNER] + landmarks[self.LMK_RIGHT_EYE_OUTER]) / 2
        ipd_flame = np.linalg.norm(left_eye_center - right_eye_center)
        self.conversion_factor = self.AVERAGE_IPD_MM / ipd_flame
        
        # 1. Nose width (alar base) - Primary for nasal/pillow masks
        nose_left = landmarks[self.LMK_NOSE_LEFT]
        nose_right = landmarks[self.LMK_NOSE_RIGHT]
        nose_width_flame = np.linalg.norm(nose_left - nose_right)
        
        # 2. Cheekbone width (zygion-zygion) - Primary for full-face masks
        cheek_left = landmarks[self.LMK_CHEEK_LEFT]
        cheek_right = landmarks[self.LMK_CHEEK_RIGHT]
        cheekbone_width_flame = np.linalg.norm(cheek_left - cheek_right)
        
        # 3. Nose-to-chin distance - Secondary check for full-face
        subnasale = landmarks[self.LMK_SUBNASALE]
        chin = landmarks[self.LMK_CHIN]
        nose_to_chin_flame = np.linalg.norm(subnasale - chin)
        
        # Convert to millimeters
        nose_width_mm = nose_width_flame * self.conversion_factor
        cheekbone_width_mm = cheekbone_width_flame * self.conversion_factor
        nose_to_chin_mm = nose_to_chin_flame * self.conversion_factor
        
        return {
            'nose_width_flame': float(nose_width_flame),
            'cheekbone_width_flame': float(cheekbone_width_flame),
            'nose_to_chin_flame': float(nose_to_chin_flame),
            'nose_width_mm': float(nose_width_mm),
            'cheekbone_width_mm': float(cheekbone_width_mm),
            'nose_to_chin_mm': float(nose_to_chin_mm),
            'conversion_factor': float(self.conversion_factor),
            'ipd_mm': float(self.AVERAGE_IPD_MM)
        }
    
    def save_measurement(self, measurements, processing_time):
        """Save measurement to JSON file"""
        self.measurement_count += 1
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"measurement_{timestamp}.json"
        filepath = self.session_dir / filename
        
        # Prepare data
        data = {
            'timestamp': datetime.now().isoformat(),
            'measurement_number': self.measurement_count,
            'measurements_mm': {
                'nose_width': measurements['nose_width_mm'],
                'cheekbone_width': measurements['cheekbone_width_mm'],
                'nose_to_chin': measurements['nose_to_chin_mm']
            },
            'measurements_flame': {
                'nose_width': measurements['nose_width_flame'],
                'cheekbone_width': measurements['cheekbone_width_flame'],
                'nose_to_chin': measurements['nose_to_chin_flame']
            },
            'calibration': {
                'ipd_mm': measurements['ipd_mm'],
                'conversion_factor': measurements['conversion_factor']
            },
            'processing_time_seconds': processing_time,
            'landmark_indices': {
                'nose_left': self.LMK_NOSE_LEFT,
                'nose_right': self.LMK_NOSE_RIGHT,
                'cheek_left': self.LMK_CHEEK_LEFT,
                'cheek_right': self.LMK_CHEEK_RIGHT,
                'subnasale': self.LMK_SUBNASALE,
                'chin': self.LMK_CHIN
            }
        }
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[OK] Saved to: {filename}")
        return filepath
    
    def draw_measurements(self, frame, measurements, bbox):
        """Draw measurement information on frame"""
        x_min, y_min, x_max, y_max = bbox
        
        # Draw face bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Measurement text (now in mm!)
        text_lines = [
            "CPAP Measurements (mm):",
            f"1. Nose Width:      {measurements['nose_width_mm']:.1f} mm",
            f"2. Cheekbone Width: {measurements['cheekbone_width_mm']:.1f} mm",
            f"3. Nose-to-Chin:    {measurements['nose_to_chin_mm']:.1f} mm",
            f"",
            f"Count: {self.measurement_count}"
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
            cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    def run(self):
        """Run the measurement system"""
        if not self.initialize_deca():
            return
        
        session_index = self.initialize_session()
        
        print("\n" + "=" * 60)
        print("CPAP Mask Measurement System")
        print("=" * 60)
        print(f"Session: {session_index}")
        print(f"Results: {self.session_dir}")
        print("\nMeasurements:")
        print("  1. Nose Width      (alar base)")
        print("  2. Cheekbone Width (zygion-zygion)")
        print("  3. Nose-to-Chin    (subnasale to menton)")
        print("\nControls:")
        print("  SPACE - Capture measurement")
        print("  ESC   - Quit")
        print("=" * 60)
        
        cap = cv2.VideoCapture(0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Display instructions
                cv2.putText(frame, f"CPAP Measurement - Session {session_index} - Count: {self.measurement_count}",
                           (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE: Measure | ESC: Quit",
                           (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('CPAP Measurement System', frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space - capture measurement
                    print(f"\n[*] Capturing Measurement #{self.measurement_count + 1}")
                    start_time = time.time()
                    
                    try:
                        # Convert to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Detect face
                        landmarks = self.face_detector.get_landmarks(frame_rgb)
                        
                        if landmarks is None or len(landmarks) == 0:
                            print("[WARN] No face detected - ensure good lighting and face visibility")
                            continue
                        
                        landmarks = landmarks[0]
                        print(f"[OK] Face detected")
                        
                        # Preprocess face
                        face_tensor, bbox = self.preprocess_face(frame_rgb, landmarks)
                        
                        # Run DECA reconstruction
                        with torch.no_grad():
                            codedict = self.deca.encode(face_tensor)
                            opdict = self.deca.decode(codedict)
                            
                            # Extract 68-point 3D landmarks
                            landmarks = opdict['landmarks3d'][0].cpu().numpy()
                            
                            # Extract CPAP measurements using landmarks
                            measurements = self.extract_cpap_measurements(landmarks)
                            
                            processing_time = time.time() - start_time
                            
                            # Display results
                            print(f"[*] CPAP Measurements:")
                            print(f"    Nose Width:      {measurements['nose_width_mm']:.1f} mm")
                            print(f"    Cheekbone Width: {measurements['cheekbone_width_mm']:.1f} mm")
                            print(f"    Nose-to-Chin:    {measurements['nose_to_chin_mm']:.1f} mm")
                            print(f"    Processing time: {processing_time:.2f}s")
                            
                            # Save measurement
                            self.save_measurement(measurements, processing_time)
                            
                            # Draw measurements on frame
                            self.draw_measurements(frame, measurements, bbox)
                            cv2.imshow('CPAP Measurement System', frame)
                            cv2.waitKey(self.RESULT_DISPLAY_MS)
                    
                    except ValueError as e:
                        print(f"[ERROR] Processing error: {e}")
                
                elif key == 27:  # ESC - quit
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n" + "=" * 60)
            print(f"[DONE] Session Complete!")
            print(f"   Total measurements: {self.measurement_count}")
            print(f"   Results saved to: {self.session_dir}")
            print(f"\n[TIP] Run validator to visualize results:")
            print(f"   python validator.py {session_index}")
            print("=" * 60)

def main():
    """Main function"""
    system = CPAPMeasurementSystem()
    system.run()

if __name__ == "__main__":
    main()
