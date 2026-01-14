#!/usr/bin/env python3
"""
INSPECT FLAME_masks.pkl - VERIFY NOSE VERTEX INDICES
=====================================================
This script reads the official FLAME_masks.pkl file and extracts
the vertex indices for the nose region to verify if 2750 and 1610
are correct alar vertices.
"""
import pickle
import numpy as np

def main():
    # Path to FLAME_masks.pkl
    masks_path = r"d:\Job\project_1\MICA\data\FLAME2020\FLAME_masks\FLAME_masks.pkl"
    
    print("="*70)
    print("FLAME_masks.pkl INSPECTOR")
    print("="*70)
    print(f"Loading: {masks_path}\n")
    
    with open(masks_path, 'rb') as f:
        masks = pickle.load(f, encoding='latin1')
    
    print(f"Type: {type(masks)}")
    
    if isinstance(masks, dict):
        print(f"Keys: {list(masks.keys())}\n")
        
        # Print info about each mask region
        for key in sorted(masks.keys()):
            value = masks[key]
            if hasattr(value, '__len__'):
                print(f"'{key}': {len(value)} vertices")
            else:
                print(f"'{key}': {type(value)}")
        
        # Check for nose region
        nose_keys = [k for k in masks.keys() if 'nose' in k.lower()]
        print(f"\n{'='*70}")
        print(f"NOSE REGIONS FOUND: {nose_keys}")
        print(f"{'='*70}")
        
        for nk in nose_keys:
            nose_vertices = masks[nk]
            if hasattr(nose_vertices, '__len__'):
                print(f"\n'{nk}' contains {len(nose_vertices)} vertices:")
                print(f"  All indices: {sorted(nose_vertices)}")
                
                # Check if 2750 and 1610 are in this region
                v2750_in = 2750 in nose_vertices
                v1610_in = 1610 in nose_vertices
                print(f"\n  Is 2750 in '{nk}'? {v2750_in}")
                print(f"  Is 1610 in '{nk}'? {v1610_in}")
        
        # Also check all regions for 2750 and 1610
        print(f"\n{'='*70}")
        print("CHECKING ALL REGIONS FOR VERTICES 2750 and 1610")
        print(f"{'='*70}")
        
        for key in sorted(masks.keys()):
            vertices = masks[key]
            if hasattr(vertices, '__iter__'):
                if 2750 in vertices:
                    print(f"  2750 found in: '{key}'")
                if 1610 in vertices:
                    print(f"  1610 found in: '{key}'")
    
    print(f"\n{'='*70}")
    print("INSPECTION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
