#!/usr/bin/env python3
"""
INSPECT FLAME DATA FILES
========================
This script inspects FLAME model files to find:
1. What data is stored in each .pkl file
2. If there are nose region masks
3. Landmark embedding data
"""
import pickle
import numpy as np
import os

def inspect_flame_masks(filepath):
    """Inspect FLAME_masks.pkl specifically for nose region"""
    print(f"\n{'='*70}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"PATH: {filepath}")
    print(f"{'='*70}")
    
    try:
        with open(filepath, 'rb') as f:
            masks = pickle.load(f, encoding='latin1')
        
        print(f"Type: {type(masks)}")
        
        if isinstance(masks, dict):
            print(f"\nAll face regions in FLAME_masks.pkl:")
            for key in sorted(masks.keys()):
                value = masks[key]
                if hasattr(value, '__len__'):
                    print(f"  '{key}': {len(value)} vertices")
            
            # Check for nose region
            print(f"\n{'='*70}")
            print("NOSE REGION ANALYSIS")
            print(f"{'='*70}")
            
            nose_keys = [k for k in masks.keys() if 'nose' in k.lower()]
            for nk in nose_keys:
                nose_vertices = list(masks[nk])
                print(f"\n'{nk}' contains {len(nose_vertices)} vertices:")
                print(f"  First 20: {sorted(nose_vertices)[:20]}")
                print(f"  Last 20:  {sorted(nose_vertices)[-20:]}")
                
                # Check if 2750 and 1610 are in this region
                print(f"\n  Is vertex 2750 in '{nk}'? {2750 in nose_vertices}")
                print(f"  Is vertex 1610 in '{nk}'? {1610 in nose_vertices}")
            
            # Check ALL regions for 2750 and 1610
            print(f"\n{'='*70}")
            print("WHICH REGIONS CONTAIN VERTICES 2750 and 1610?")
            print(f"{'='*70}")
            
            for key in sorted(masks.keys()):
                vertices = list(masks[key])
                if 2750 in vertices:
                    print(f"  2750 found in: '{key}'")
                if 1610 in vertices:
                    print(f"  1610 found in: '{key}'")
                    
    except Exception as e:
        print(f"Error reading file: {e}")
        import traceback
        traceback.print_exc()



def inspect_pickle_file(filepath):
    """Inspect contents of a pickle file"""
    print(f"\n{'='*70}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"PATH: {filepath}")
    print(f"{'='*70}")
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        if isinstance(data, dict):
            print(f"Type: Dictionary with {len(data)} keys")
            print(f"\nKeys found:")
            for key in sorted(data.keys()):
                value = data[key]
                if isinstance(value, np.ndarray):
                    print(f"  '{key}': numpy array, shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, (list, tuple)):
                    print(f"  '{key}': {type(value).__name__}, length={len(value)}")
                elif isinstance(value, str):
                    print(f"  '{key}': string = '{value[:50]}...'")
                elif isinstance(value, (int, float)):
                    print(f"  '{key}': {type(value).__name__} = {value}")
                else:
                    print(f"  '{key}': {type(value).__name__}")
            
            # Check for masks or face region data
            mask_keys = [k for k in data.keys() if 'mask' in k.lower() or 'nose' in k.lower() or 'face' in k.lower()]
            if mask_keys:
                print(f"\n*** FOUND MASK/FACE REGION KEYS: {mask_keys}")
                for mk in mask_keys:
                    if isinstance(data[mk], np.ndarray):
                        print(f"    {mk} values: {data[mk][:20]}...")
        else:
            print(f"Type: {type(data)}")
            if hasattr(data, '__len__'):
                print(f"Length: {len(data)}")
                
    except Exception as e:
        print(f"Error reading file: {e}")


def inspect_npy_file(filepath):
    """Inspect contents of a numpy file"""
    print(f"\n{'='*70}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"PATH: {filepath}")
    print(f"{'='*70}")
    
    try:
        data = np.load(filepath, allow_pickle=True)
        print(f"Type: numpy array")
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
        
        if data.dtype == object:
            print("\nContains objects, inspecting...")
            item = data.item() if data.ndim == 0 else data
            if isinstance(item, dict):
                print(f"  Dictionary with keys: {list(item.keys())}")
                for k, v in item.items():
                    if isinstance(v, np.ndarray):
                        print(f"    '{k}': shape={v.shape}")
                    else:
                        print(f"    '{k}': {type(v)}")
        else:
            print(f"\nFirst 10 rows:\n{data[:10]}")
            
    except Exception as e:
        print(f"Error reading file: {e}")


def main():
    print("="*70)
    print("FLAME DATA FILE INSPECTOR")
    print("="*70)
    print("Looking for vertex indices for nose region...")
    
    # IMPORTANT: Inspect FLAME_masks.pkl from MICA folder
    flame_masks_path = r"d:\Job\project_1\MICA\data\FLAME2020\FLAME_masks\FLAME_masks.pkl"
    if os.path.exists(flame_masks_path):
        inspect_flame_masks(flame_masks_path)
    else:
        print(f"\n[ERROR] FLAME_masks.pkl not found at: {flame_masks_path}")
    
    print("\n" + "="*70)
    print("INSPECTION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
