#!/usr/bin/env python3
"""
VERIFY NOSE VERTEX INDICES
==========================
This script verifies that vertices 2750 and 1610 are actually the 
left and right alar (nostril outer edge) vertices.

VERIFICATION METHOD:
1. Load a FLAME mesh from DECA output
2. Examine the X-coordinates of all nose region vertices
3. Find the leftmost and rightmost vertices at alar level
4. Compare with claimed indices 2750 and 1610
"""
import numpy as np
import pickle
import os

def load_flame_mesh_from_camera():
    """
    Run DECA on camera and get vertices.
    Returns: (5023, 3) array of vertices
    """
    import torch
    import cv2
    
    # Initialize DECA
    from decalib.deca_measurement import DECAMeasurement
    from decalib.utils.config import cfg as deca_cfg
    import face_alignment
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    deca_cfg.model.use_tex = False
    
    print("[*] Loading DECA model...")
    deca = DECAMeasurement(config=deca_cfg, device=device)
    
    print("[*] Loading face detector...")
    face_detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, 
        flip_input=False, 
        device=device
    )
    
    print("[*] Opening camera... Press SPACE to capture")
    cap = cv2.VideoCapture(0)
    
    vertices = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.putText(frame, "Press SPACE to capture face", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = face_detector.get_landmarks(frame_rgb)
            
            if landmarks is not None and len(landmarks) > 0:
                lmk = landmarks[0]
                
                # Crop and preprocess
                h, w = frame.shape[:2]
                x_min = max(0, int(np.min(lmk[:, 0])) - 50)
                x_max = min(w, int(np.max(lmk[:, 0])) + 50)
                y_min = max(0, int(np.min(lmk[:, 1])) - 50)
                y_max = min(h, int(np.max(lmk[:, 1])) + 50)
                
                face_crop = frame_rgb[y_min:y_max, x_min:x_max]
                face_resized = cv2.resize(face_crop, (224, 224))
                face_tensor = torch.tensor(face_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                face_tensor = face_tensor.to(device)
                
                with torch.no_grad():
                    codedict = deca.encode(face_tensor)
                    opdict = deca.decode(codedict)
                    vertices = opdict['verts'][0].cpu().numpy()
                    print(f"[OK] Got {vertices.shape[0]} vertices")
                break
            else:
                print("[WARN] No face detected, try again")
        elif key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return vertices


def verify_nose_vertices(vertices):
    """
    Verify that vertices 2750 and 1610 are the outer alar vertices.
    """
    print("\n" + "="*70)
    print("VERIFICATION OF NOSE VERTEX INDICES")
    print("="*70)
    
    # The claimed vertex indices
    CLAIMED_LEFT_ALAR = 2750
    CLAIMED_RIGHT_ALAR = 1610
    
    # Get the coordinates of claimed vertices
    left_alar = vertices[CLAIMED_LEFT_ALAR]
    right_alar = vertices[CLAIMED_RIGHT_ALAR]
    
    print(f"\nClaimed Left Alar (V{CLAIMED_LEFT_ALAR}):  X={left_alar[0]:.4f}, Y={left_alar[1]:.4f}, Z={left_alar[2]:.4f}")
    print(f"Claimed Right Alar (V{CLAIMED_RIGHT_ALAR}): X={right_alar[0]:.4f}, Y={right_alar[1]:.4f}, Z={right_alar[2]:.4f}")
    
    # VERIFICATION 1: Check X-coordinates
    # In FLAME coordinate system: X is Left(-) to Right(+)
    # So left alar should have MORE NEGATIVE X than right alar
    print(f"\n--- CHECK 1: X-Coordinate Positions ---")
    print(f"Left alar X:  {left_alar[0]:.4f}")
    print(f"Right alar X: {right_alar[0]:.4f}")
    
    if left_alar[0] < right_alar[0]:
        print("✓ PASS: Left alar has more negative X (is on the LEFT side)")
    else:
        print("✗ FAIL: Left alar should have more negative X!")
    
    # VERIFICATION 2: Check Y-coordinates are similar (both at alar level)
    print(f"\n--- CHECK 2: Y-Coordinate Level (should be similar) ---")
    print(f"Left alar Y:  {left_alar[1]:.4f}")
    print(f"Right alar Y: {right_alar[1]:.4f}")
    y_diff = abs(left_alar[1] - right_alar[1])
    print(f"Y difference: {y_diff:.4f}")
    
    if y_diff < 0.01:
        print("✓ PASS: Both vertices are at similar Y level (same height on nose)")
    else:
        print("! WARN: Y-coordinates differ significantly")
    
    # VERIFICATION 3: Check Z-coordinates are similar (depth)
    print(f"\n--- CHECK 3: Z-Coordinate Depth (should be similar) ---")
    print(f"Left alar Z:  {left_alar[2]:.4f}")
    print(f"Right alar Z: {right_alar[2]:.4f}")
    z_diff = abs(left_alar[2] - right_alar[2])
    print(f"Z difference: {z_diff:.4f}")
    
    if z_diff < 0.01:
        print("✓ PASS: Both vertices have similar Z (same forward position)")
    else:
        print("! WARN: Z-coordinates differ significantly")
    
    # VERIFICATION 4: Find the ACTUAL most extreme X vertices near alar Y-level
    print(f"\n--- CHECK 4: Finding ACTUAL extreme X vertices near alar Y-level ---")
    
    alar_y_level = (left_alar[1] + right_alar[1]) / 2
    y_tolerance = 0.01
    
    # Find all vertices near the alar Y level
    candidates = []
    for i, v in enumerate(vertices):
        if abs(v[1] - alar_y_level) < y_tolerance:
            candidates.append((i, v))
    
    print(f"Found {len(candidates)} vertices near Y={alar_y_level:.4f}")
    
    if candidates:
        # Find most negative X (leftmost) and most positive X (rightmost)
        leftmost = min(candidates, key=lambda x: x[1][0])
        rightmost = max(candidates, key=lambda x: x[1][0])
        
        print(f"\nMost LEFT vertex at alar Y-level:")
        print(f"  Vertex {leftmost[0]}: X={leftmost[1][0]:.4f}, Y={leftmost[1][1]:.4f}, Z={leftmost[1][2]:.4f}")
        
        print(f"\nMost RIGHT vertex at alar Y-level:")
        print(f"  Vertex {rightmost[0]}: X={rightmost[1][0]:.4f}, Y={rightmost[1][1]:.4f}, Z={rightmost[1][2]:.4f}")
        
        if leftmost[0] == CLAIMED_LEFT_ALAR:
            print(f"\n✓ VERIFIED: Vertex {CLAIMED_LEFT_ALAR} IS the leftmost at alar level")
        else:
            print(f"\n✗ MISMATCH: Leftmost is actually vertex {leftmost[0]}, not {CLAIMED_LEFT_ALAR}")
    
    # VERIFICATION 5: Compare with nose tip
    print(f"\n--- CHECK 5: Position relative to Nose Tip (V3564) ---")
    nose_tip = vertices[3564]
    print(f"Nose Tip (V3564): X={nose_tip[0]:.4f}, Y={nose_tip[1]:.4f}, Z={nose_tip[2]:.4f}")
    print(f"Left Alar is {abs(left_alar[0] - nose_tip[0]):.4f} away in X")
    print(f"Right Alar is {abs(right_alar[0] - nose_tip[0]):.4f} away in X")
    
    # Calculate nose width
    nose_width = np.linalg.norm(left_alar - right_alar)
    print(f"\n--- FINAL MEASUREMENT ---")
    print(f"Nose Width (V{CLAIMED_LEFT_ALAR} to V{CLAIMED_RIGHT_ALAR}): {nose_width:.4f} FLAME units")
    print(f"If scale is ~1000, this would be: {nose_width * 1000:.1f} mm")
    
    print("\n" + "="*70)


def main():
    print("="*70)
    print("FLAME VERTEX VERIFICATION TOOL")
    print("="*70)
    print("\nThis will verify that vertices 2750 and 1610 are correct nose alar vertices.\n")
    
    vertices = load_flame_mesh_from_camera()
    
    if vertices is not None:
        verify_nose_vertices(vertices)
    else:
        print("No vertices captured. Exiting.")


if __name__ == "__main__":
    main()
