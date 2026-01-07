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
            from decalib.deca import DECA
            from decalib.utils.config import cfg as deca_cfg
            import face_alignment
            
            # Configure DECA
            deca_cfg.model.use_tex = False  # Disable texture for speed
            deca_cfg.rasterizer_type = 'pytorch3d'
            
            # Initialize DECA
            print("🔄 Loading DECA model...")
            self.deca = DECA(config=deca_cfg, device=self.device)
            print(" DECA model loaded")
            
            # Initialize face detector
            print("🔄 Loading face detector...")
            self.face_detector = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, 
                flip_input=False, 
                device=self.device
            )
            print(" Face detector loaded")
            
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
        
        return {
            'width': face_width,
            'height': face_height,
            'depth': face_depth,
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
            f"Width:  {measurements['width']:.6f}",
            f"Height: {measurements['height']:.6f}",
            f"Depth:  {measurements['depth']:.6f}",
            f"Vertices: {measurements['vertex_count']}",
            f"Processing: {processing_time:.2f}s"
        ]
        
        # Draw text background
        text_height = 25
        bg_height = len(text_lines) * text_height + 10
        cv2.rectangle(frame, (10, 10), (400, bg_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, bg_height), (0, 255, 0), 2)
        
        # Draw text
        for i, line in enumerate(text_lines):
            y_pos = 30 + i * text_height
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    def save_measurement(self, measurements):
        """Save measurement to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        measurement_data = {
            'timestamp': timestamp,
            'width': measurements['width'],
            'height': measurements['height'],
            'depth': measurements['depth']
        }
        self.measurement_history.append(measurement_data)
        
        # Keep only last 10 measurements
        if len(self.measurement_history) > 10:
            self.measurement_history.pop(0)
    
    def print_statistics(self):
        """Print measurement statistics"""
        if len(self.measurement_history) < 2:
            return
            
        widths = [m['width'] for m in self.measurement_history]
        heights = [m['height'] for m in self.measurement_history]
        
        print(f"\n📊 Measurement Statistics (last {len(self.measurement_history)} measurements):")
        print(f"   Width:  {np.mean(widths):.6f} ± {np.std(widths):.6f} FLAME units")
        print(f"   Height: {np.mean(heights):.6f} ± {np.std(heights):.6f} FLAME units")
        print(f"   Width CV: {(np.std(widths)/np.mean(widths)*100):.2f}%")
        print(f"   Height CV: {(np.std(heights)/np.mean(heights)*100):.2f}%")
    
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
                        opdict, visdict = self.deca.decode(codedict)
                        
                        # Extract 3D vertices
                        vertices = opdict['verts'][0].cpu().numpy()
                        
                        # Calculate measurements
                        measurements = self.extract_measurements(vertices)
                        
                        processing_time = time.time() - start_time
                        
                        # Display results
                        print(f"📏 3D Measurements (FLAME units):")
                        print(f"   Width:  {measurements['width']:.6f}")
                        print(f"   Height: {measurements['height']:.6f}")
                        print(f"   Depth:  {measurements['depth']:.6f}")
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
