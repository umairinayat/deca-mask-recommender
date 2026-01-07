# -*- coding: utf-8 -*-
"""
DECA for Measurement Only - No Rendering Required

This is a modified version of DECA that works without pytorch3d or CUDA rasterizer.
It only extracts 3D vertices and landmarks, which is sufficient for facial measurements.
"""

import os
import sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .utils.tensor_cropper import transform_points
from .datasets import datasets
from .utils.config import cfg
torch.backends.cudnn.benchmark = True


class DECAMeasurement(nn.Module):
    """
    DECA model for measurement purposes only.
    Does not require pytorch3d or CUDA rasterizer.
    
    This class extracts:
    - 3D vertices (shape: [batch, 5023, 3])
    - 2D/3D landmarks
    - Shape/expression/pose parameters
    
    But does NOT do:
    - Texture extraction
    - Image rendering
    - Detail mapping
    """
    
    def __init__(self, config=None, device='cpu'):
        super(DECAMeasurement, self).__init__()
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        self._create_model(self.cfg.model)
        # Note: We DO NOT call _setup_renderer - this is the key difference
        print("DECAMeasurement initialized - Rendering disabled (measurement mode)")

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = model_cfg.n_shape + model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3  # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam, model_cfg.n_light]
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)
        self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)
        
        # decoders
        self.flame = FLAME(model_cfg).to(self.device)
        self.D_detail = Generator(
            latent_dim=self.n_detail + self.n_cond, 
            out_channels=1, 
            out_scale=model_cfg.max_z, 
            sample_mode='bilinear'
        ).to(self.device)
        
        # resume model
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            print(f'Trained model found. Loading {model_path}')
            # Use map_location to handle GPU-trained models on CPU
            checkpoint = torch.load(model_path, map_location=self.device)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
            util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
        else:
            print(f'WARNING: Model not found at {model_path}')
            
        # eval mode
        self.E_flame.eval()
        self.E_detail.eval()
        self.D_detail.eval()

    def decompose_code(self, code, num_dict):
        """Convert a flattened parameter vector to a dictionary of parameters"""
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def encode(self, images, use_detail=False):
        """
        Encode images to FLAME parameters.
        
        Args:
            images: Input images tensor [batch, 3, 224, 224]
            use_detail: Whether to extract detail codes (not needed for measurements)
            
        Returns:
            codedict: Dictionary with shape, exp, pose, cam parameters
        """
        with torch.no_grad():
            parameters = self.E_flame(images)
        
        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['images'] = images
        
        if use_detail:
            detailcode = self.E_detail(images)
            codedict['detail'] = detailcode
            
        if self.cfg.model.jaw_type == 'euler':
            posecode = codedict['pose']
            euler_jaw_pose = posecode[:, 3:].clone()
            posecode[:, 3:] = batch_euler2axis(euler_jaw_pose)
            codedict['pose'] = posecode
            codedict['euler_jaw_pose'] = euler_jaw_pose
            
        return codedict

    def decode(self, codedict):
        """
        Decode FLAME parameters to 3D vertices.
        
        Args:
            codedict: Dictionary with shape, exp, pose parameters
            
        Returns:
            opdict: Dictionary with vertices and landmarks
        """
        images = codedict['images']
        batch_size = images.shape[0]

        # Decode FLAME parameters to vertices
        verts, landmarks2d, landmarks3d = self.flame(
            shape_params=codedict['shape'],
            expression_params=codedict['exp'],
            pose_params=codedict['pose']
        )
        
        landmarks3d_world = landmarks3d.clone()

        # Project landmarks (simple orthographic projection)
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:, :, :2]
        landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]
        
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam'])
        landmarks3d[:, :, 1:] = -landmarks3d[:, :, 1:]
        
        trans_verts = util.batch_orth_proj(verts, codedict['cam'])
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        opdict = {
            'verts': verts,                        # Shape: [batch, 5023, 3] - World coordinates
            'trans_verts': trans_verts,            # Shape: [batch, 5023, 3] - Projected
            'landmarks2d': landmarks2d,            # Shape: [batch, 68, 2]
            'landmarks3d': landmarks3d,            # Shape: [batch, 68, 3]
            'landmarks3d_world': landmarks3d_world, # Shape: [batch, 68, 3]
        }

        return opdict

    def extract_measurements(self, opdict, indices):
        """
        Extract measurements between specified vertex pairs.
        
        Args:
            opdict: Output dictionary from decode()
            indices: Dictionary of vertex index pairs, e.g.:
                    {'nose_width': (3632, 3325), 'cheek_width': (4478, 2051)}
                    
        Returns:
            measurements: Dictionary of distances in FLAME units
        """
        verts = opdict['verts'][0].cpu().numpy()  # [5023, 3]
        
        measurements = {}
        for name, (idx1, idx2) in indices.items():
            point1 = verts[idx1]
            point2 = verts[idx2]
            distance = np.linalg.norm(point1 - point2)
            measurements[name] = float(distance)
            
        return measurements


# For backwards compatibility - alias
DECA = DECAMeasurement

