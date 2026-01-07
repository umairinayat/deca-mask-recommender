#!/usr/bin/env python3
"""
DECA Live Camera Demo - CPU Version (No pytorch3d required)
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
    """DECA Live Demo using measurement-only mode (no rendering required)"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.deca = None
        self.face_detector = None
        self.measurement_history = []
        
    def initialize_deca(self):
        """Initialize DECA model and face detector"""
        print("=" * 60)
        print("Initializing DECA Live Demo (CPU-compatible mode)...")
        print("=" * 60)
        
        try:
            # Use measurement-only DECA (no rendering required)
            from decalib.deca_measurement import DECAMeasurement
            from decalib.utils.config import cfg as deca_cfg
            import face_alignment
            
            # Configure DECA
            deca_cfg.model.use_tex = False  # Disable texture
            
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
    
    def extract_measurements(self, vertices):
        """Extract face measurements from 3D vertices"""
        # Vertices shape: (5023, 3) - (x, y, z) coordinates
        x_coords = vertices[:, 0]  # Width (left-right)
        y_coords = vertices[:, 1]  # Height (up-down)
        z_coords = vertices[:, 2]  # Depth (front-back)
        
        # Calculate face dimensions
        face_width = np.max(x_coords) - np.min(x_coords)
        face_height = np.max(y_coords) - np.min(y_coords)
        face_depth = np.max(z_coords) - np.min(z_coords)
        
        # CPAP-specific measurements using known FLAME vertex indices
        # Nose width (alar base)
        nose_left = vertices[3632]
        nose_right = vertices[3325]
        nose_width = np.linalg.norm(nose_left - nose_right)
        
        # Cheekbone width (zygion-zygion)
        cheek_left = vertices[4478]
        cheek_right = vertices[2051]
        cheekbone_width = np.linalg.norm(cheek_left - cheek_right)
        
        # Nose-to-chin distance
        nose_base = vertices[175]
        chin = vertices[152]
        nose_to_chin = np.linalg.norm(nose_base - chin)
        
        return {
            'face_width': face_width,
            'face_height': face_height,
            'face_depth': face_depth,
            'nose_width': nose_width,
            'cheekbone_width': cheekbone_width,
            'nose_to_chin': nose_to_chin,
            'vertex_count': vertices.shape[0]
        }
    
    def draw_measurements(self, frame, measurements, bbox, processing_time):
        """Draw measurement information on frame"""
        x_min, y_min, x_max, y_max = bbox
        
        # Draw face bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Measurement text
        text_lines = [
            f"DECA 3D Measurements (FLAME units):",
            f"Face Width:      {measurements['face_width']:.6f}",
            f"Face Height:     {measurements['face_height']:.6f}",
            f"Nose Width:      {measurements['nose_width']:.6f}",
            f"Cheekbone Width: {measurements['cheekbone_width']:.6f}",
            f"Nose-to-Chin:    {measurements['nose_to_chin']:.6f}",
            f"Vertices: {measurements['vertex_count']}",
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
            'nose_width': measurements['nose_width'],
            'cheekbone_width': measurements['cheekbone_width'],
            'nose_to_chin': measurements['nose_to_chin']
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
            
        nose_widths = [m['nose_width'] for m in self.measurement_history]
        cheek_widths = [m['cheekbone_width'] for m in self.measurement_history]
        
        print(f"\n{'='*60}")
        print(f"Measurement Statistics (last {len(self.measurement_history)} measurements):")
        print(f"{'='*60}")
        print(f"Nose Width:      {np.mean(nose_widths):.6f} +/- {np.std(nose_widths):.6f}")
        print(f"  CV: {(np.std(nose_widths)/np.mean(nose_widths)*100):.2f}%")
        print(f"Cheekbone Width: {np.mean(cheek_widths):.6f} +/- {np.std(cheek_widths):.6f}")
        print(f"  CV: {(np.std(cheek_widths)/np.mean(cheek_widths)*100):.2f}%")
        print(f"{'='*60}")
    
    def run_demo(self):
        """Run live camera demo"""
        if not self.initialize_deca():
            print("[ERROR] Failed to initialize DECA. Exiting.")
            return
            
        print("\n" + "=" * 60)
        print("DECA Live Camera Demo")
        print("=" * 60)
        print("Controls:")
        print("  SPACE - Capture and measure face")
        print("  S     - Show measurement statistics")
        print("  ESC   - Quit")
        print("=" * 60 + "\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open camera")
            return
            
        measurement_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[ERROR] Could not read frame")
                    break
                
                # Convert to RGB for face detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display instructions
                h, w = frame.shape[:2]
                cv2.putText(frame, f"DECA Live Demo - Measurements: {measurement_count}", 
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE: Measure | S: Stats | ESC: Quit", 
                           (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('DECA Live Demo (CPU)', frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space - measure face
                    print(f"\n[*] Measurement #{measurement_count + 1}")
                    start_time = time.time()
                    
                    # Detect face landmarks
                    landmarks = self.face_detector.get_landmarks(frame_rgb)
                    
                    if landmarks is None or len(landmarks) == 0:
                        print("[WARN] No face detected - ensure good lighting and face visibility")
                        continue
                    
                    landmarks = landmarks[0]  # Use first detected face
                    print(f"[OK] Face detected with {len(landmarks)} landmarks")
                    
                    # Preprocess face for DECA
                    face_tensor, bbox = self.preprocess_face(frame_rgb, landmarks)
                    
                    # Run DECA reconstruction (measurement-only)
                    with torch.no_grad():
                        codedict = self.deca.encode(face_tensor)
                        opdict = self.deca.decode(codedict)
                        
                        # Extract 3D vertices
                        vertices = opdict['verts'][0].cpu().numpy()
                        
                        # Calculate measurements
                        measurements = self.extract_measurements(vertices)
                        
                        processing_time = time.time() - start_time
                        
                        # Display results
                        print(f"[*] 3D Measurements (FLAME units):")
                        print(f"    Nose Width:      {measurements['nose_width']:.6f}")
                        print(f"    Cheekbone Width: {measurements['cheekbone_width']:.6f}")
                        print(f"    Nose-to-Chin:    {measurements['nose_to_chin']:.6f}")
                        print(f"    Processing time: {processing_time:.2f}s")
                        
                        # Save measurement
                        self.save_measurement(measurements)
                        measurement_count += 1
                        
                        # Draw measurements on frame
                        self.draw_measurements(frame, measurements, bbox, processing_time)
                        cv2.imshow('DECA Live Demo (CPU)', frame)
                        cv2.waitKey(2000)  # Show result for 2 seconds
                
                elif key == ord('s'):  # S - show statistics
                    self.print_statistics()
                
                elif key == 27:  # ESC - quit
                    break
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print(f"\n{'='*60}")
            print(f"Demo complete! Total measurements: {measurement_count}")
            if measurement_count > 0:
                self.print_statistics()
            print(f"{'='*60}")


def main():
    """Main function"""
    demo = DECALiveDemo()
    demo.run_demo()


if __name__ == "__main__":
    main()

