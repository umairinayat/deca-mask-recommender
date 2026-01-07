# -*- coding: utf-8 -*-
"""
Face Frontalization using DECA

This script takes an input image with a face in any pose and produces
a frontalized (forward-facing) output image. This is useful for face
alignment before feeding into face recognition/embedding models.

Usage:
    python frontalize_face.py -i path/to/image.jpg -o path/to/output.jpg
    python frontalize_face.py -i path/to/image_folder -o path/to/output_folder
"""

import os
import sys
import cv2
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.rotation_converter import batch_euler2axis, deg2rad


class FaceFrontalizer:
    """
    Face frontalization/alignment using DECA 3D reconstruction.
    
    Two modes available:
    1. 3D Render Mode: Re-render face from frontal view (may have texture gaps)
    2. 2D Align Mode: Use 3D landmarks to align original image (preserves texture)
    
    For face recognition, 2D alignment is typically preferred as it preserves
    the original image texture while normalizing pose.
    """
    
    def __init__(self, device='cpu'):
        """
        Initialize the DECA model for face frontalization.
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        # Configure DECA
        deca_cfg.model.use_tex = False  # We'll extract texture from input
        deca_cfg.rasterizer_type = 'pytorch3d'
        deca_cfg.model.extract_tex = True
        
        print(f"Loading DECA model on {device}...")
        self.deca = DECA(config=deca_cfg, device=device)
        print("DECA model loaded successfully!")
    
    def frontalize(self, image_path, output_size=224):
        """
        Frontalize a face from an input image.
        
        Args:
            image_path: Path to input image
            output_size: Size of output image (square)
            
        Returns:
            frontalized_image: numpy array (H, W, 3) BGR format
            original_crop: the cropped/aligned input face
            success: boolean indicating if face was detected
        """
        # Load and preprocess image
        testdata = datasets.TestData(image_path, iscrop=True, face_detector='fan', device=self.device)
        
        if len(testdata) == 0:
            print(f"No face detected in {image_path}")
            return None, None, False
        
        # Get the first face
        data = testdata[0]
        images = data['image'].to(self.device)[None, ...]
        
        with torch.no_grad():
            # Encode: Extract FLAME parameters from input image
            codedict = self.deca.encode(images)
            
            # Store original pose for reference
            original_pose = codedict['pose'].clone()
            
            # First decode with ORIGINAL pose to extract texture properly
            original_opdict, original_visdict = self.deca.decode(codedict)
            
            # Get the extracted UV texture from the original pose
            # uv_texture_gt has texture only for visible areas
            # We'll blend it with mean texture for missing areas
            uv_texture_gt = original_opdict['uv_texture_gt'].clone()
            
            # Use mean texture as fallback for occluded areas
            # The uv_face_eye_mask indicates the face region
            # Areas outside the mask in uv_texture_gt are filled with gray (0.7)
            # We keep the extracted texture as-is since it already has fallback
            uv_texture = uv_texture_gt
            
            # FRONTALIZATION: Create frontal pose using proper axis-angle conversion
            # (Following demo_teaser.py approach)
            euler_pose = torch.zeros((1, 3), device=self.device)  # 0 degrees on all axes
            frontal_global_pose = batch_euler2axis(deg2rad(euler_pose))
            
            # Create frontal pose: frontal global rotation + original jaw pose
            frontal_pose = codedict['pose'].clone()
            frontal_pose[:, :3] = frontal_global_pose  # Frontal head rotation
            # Keep frontal_pose[:, 3:] for jaw expression
            
            # Set camera for centered frontal view (from demo_teaser.py)
            frontal_cam = torch.tensor([[8.0, 0.0, 0.0]], device=self.device, dtype=torch.float32)
            
            # Generate frontal mesh vertices
            frontal_verts, landmarks2d, landmarks3d = self.deca.flame(
                shape_params=codedict['shape'].clone(), 
                expression_params=codedict['exp'].clone(), 
                pose_params=frontal_pose
            )
            
            # Project to image space
            from decalib.utils import util as deca_util
            frontal_trans_verts = deca_util.batch_orth_proj(frontal_verts, frontal_cam)
            frontal_trans_verts[:, :, 1:] = -frontal_trans_verts[:, :, 1:]
            
            # Render the frontalized face with extracted texture
            ops = self.deca.render(
                frontal_verts.clone(), 
                frontal_trans_verts.clone(), 
                uv_texture,
                h=output_size, 
                w=output_size
            )
            
            # Also render shape for comparison (useful when texture is incomplete)
            shape_images, _, grid, alpha_images = self.deca.render.render_shape(
                frontal_verts.clone(), 
                frontal_trans_verts.clone(), 
                h=output_size, 
                w=output_size,
                return_grid=True
            )
            
            # Combine rendered face with alpha mask for clean output
            rendered_face = ops['images'][0]
            alpha = ops['alpha_images'][0]
            
            # Create white background and composite
            background = torch.ones_like(rendered_face)
            final_image = rendered_face * alpha + background * (1 - alpha)
            
            # For face alignment purposes, we can use either:
            # 1. Textured render (may have gaps for occluded areas)
            # 2. Shape render (clean geometry, no texture)
            # We'll output the textured version but also save shape
            frontalized_image = util.tensor2image(final_image)
            
            # Also create shape-only version (more reliable for alignment)
            shape_image = util.tensor2image(shape_images[0])
            
            # Get original cropped face for comparison
            original_crop = util.tensor2image(images[0])
            
        return frontalized_image, original_crop, shape_image, True
    
    
    def frontalize_batch(self, input_path, output_path, output_size=224):
        """
        Frontalize faces from a folder of images.
        
        Args:
            input_path: Path to input folder or single image
            output_path: Path to output folder or single image
            output_size: Size of output images
        """
        # Determine if input is folder or single file
        if os.path.isdir(input_path):
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                import glob
                image_files.extend(glob.glob(os.path.join(input_path, ext)))
                image_files.extend(glob.glob(os.path.join(input_path, ext.upper())))
            
            os.makedirs(output_path, exist_ok=True)
            
            print(f"Processing {len(image_files)} images...")
            for img_path in tqdm(image_files):
                basename = os.path.basename(img_path)
                name, ext = os.path.splitext(basename)
                
                frontalized, original, shape, success = self.frontalize(img_path, output_size)
                
                if success:
                    # Save frontalized image
                    out_file = os.path.join(output_path, f"{name}_frontal{ext}")
                    cv2.imwrite(out_file, frontalized)
                    
                    # Save shape-only version (more reliable for alignment)
                    shape_file = os.path.join(output_path, f"{name}_shape{ext}")
                    cv2.imwrite(shape_file, shape)
                    
                    # Optionally save comparison
                    comparison = self._create_comparison(original, frontalized)
                    comp_file = os.path.join(output_path, f"{name}_comparison{ext}")
                    cv2.imwrite(comp_file, comparison)
        else:
            # Single image
            frontalized, original, shape, success = self.frontalize(input_path, output_size)
            
            if success:
                cv2.imwrite(output_path, frontalized)
                
                # Save comparison and shape next to output
                dirname = os.path.dirname(output_path)
                basename = os.path.basename(output_path)
                name, ext = os.path.splitext(basename)
                comparison = self._create_comparison(original, frontalized)
                comp_file = os.path.join(dirname, f"{name}_comparison{ext}")
                shape_file = os.path.join(dirname, f"{name}_shape{ext}")
                cv2.imwrite(shape_file, shape)
                cv2.imwrite(comp_file, comparison)
                
                print(f"Saved frontalized image to: {output_path}")
                print(f"Saved comparison to: {comp_file}")
            else:
                print("Failed to frontalize image - no face detected")
    
    def _create_comparison(self, original, frontalized):
        """Create side-by-side comparison image."""
        # Resize to same height
        h1, w1 = original.shape[:2]
        h2, w2 = frontalized.shape[:2]
        
        target_h = max(h1, h2)
        
        if h1 != target_h:
            scale = target_h / h1
            original = cv2.resize(original, (int(w1 * scale), target_h))
        
        if h2 != target_h:
            scale = target_h / h2
            frontalized = cv2.resize(frontalized, (int(w2 * scale), target_h))
        
        # Add labels
        original_labeled = original.copy()
        frontalized_labeled = frontalized.copy()
        
        cv2.putText(original_labeled, "Original", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frontalized_labeled, "Frontalized", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Concatenate horizontally
        comparison = np.hstack([original_labeled, frontalized_labeled])
        
        return comparison


def main():
    parser = argparse.ArgumentParser(
        description='Face Frontalization using DECA - Align faces to frontal view'
    )
    parser.add_argument(
        '-i', '--input', 
        required=True,
        help='Path to input image or folder of images'
    )
    parser.add_argument(
        '-o', '--output', 
        required=True,
        help='Path to output image or folder'
    )
    parser.add_argument(
        '--device', 
        default='cpu',
        help='Device to use: cuda or cpu (default: cpu)'
    )
    parser.add_argument(
        '--size', 
        type=int, 
        default=224,
        help='Output image size (default: 224)'
    )
    
    args = parser.parse_args()
    
    # Initialize frontalizer
    frontalizer = FaceFrontalizer(device=args.device)
    
    # Process images
    frontalizer.frontalize_batch(args.input, args.output, args.size)
    
    print("Done!")


if __name__ == '__main__':
    main()
