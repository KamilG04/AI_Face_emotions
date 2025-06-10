#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WFLW Production DataLoader - Clean Version
Compatible with TensorBoard trainer
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import json
import time

class WFLWConfig:
    """Basic configuration for WFLW DataLoader"""
    
    def __init__(self):
        # Dataset parameters
        self.num_landmarks = 98
        self.num_attributes = 6
        self.attribute_names = ['pose', 'expression', 'illumination', 'makeup', 'occlusion', 'blur']
        
        # Image parameters
        self.image_size = (224, 224)  # (height, width)
        self.image_channels = 3
        self.image_mean = [0.485, 0.456, 0.406]  # ImageNet normalization
        self.image_std = [0.229, 0.224, 0.225]
        
        # Landmark parameters
        self.landmarks_normalize_range = (-0.5, 0.5)
        self.bbox_padding = 0.1
        
        # DataLoader parameters
        self.batch_size = 32
        self.num_workers = 4
        self.pin_memory = True
        self.drop_last = True
        
        # Basic training augmentations
        self.train_augmentations = {
            'horizontal_flip_prob': 0.5,
            'rotation_limit': 15,
            'brightness_limit': 0.2,
            'contrast_limit': 0.2,
            'saturation_shift_limit': 20,
            'hue_shift_limit': 10,
            'blur_limit': 3,
            'noise_limit': 10,
            'scale_limit': 0.1,
            'shift_limit': 0.05
        }
        
        # Validation
        self.validate_landmarks = True
        self.clip_landmarks = True


class WFLWConfigEnhanced(WFLWConfig):
    """Enhanced config with stronger augmentations against overfitting"""
    
    def __init__(self):
        super().__init__()
        
        # Enhanced training augmentations
        self.train_augmentations = {
            # Geometric augmentations - increased intensity
            'horizontal_flip_prob': 0.5,
            'rotation_limit': 25,        # was 15 → 25
            'scale_limit': 0.15,         # was 0.1 → 0.15  
            'shift_limit': 0.1,          # was 0.05 → 0.1
            
            # Photometric augmentations - more variation
            'brightness_limit': 0.3,     # was 0.2 → 0.3
            'contrast_limit': 0.3,       # was 0.2 → 0.3
            'saturation_shift_limit': 30, # was 20 → 30
            'hue_shift_limit': 15,       # was 10 → 15
            
            # Noise & blur - more robustness
            'blur_limit': 5,             # was 3 → 5
            'noise_limit': 15,           # was 10 → 15
            
            # New augmentations:
            'elastic_alpha': 120,
            'elastic_sigma': 6,
            'grid_distortion_limit': 0.1,
            'optical_distortion_limit': 0.05,
            'coarse_dropout_prob': 0.1,
            'coarse_dropout_max_holes': 4,
            'coarse_dropout_max_size': 0.1,
            'gaussian_blur_prob': 0.1,
            'motion_blur_prob': 0.1,
            'channel_shuffle_prob': 0.05,
        }


class WFLWAnnotationParser:
    """Optimized annotation parser for WFLW"""
    
    def __init__(self, config: WFLWConfig):
        self.config = config
        self.cache = {}
        
    def parse_line(self, line: str) -> Optional[Dict]:
        """Parse annotation line with caching"""
        
        line_hash = hash(line)
        if line_hash in self.cache:
            return self.cache[line_hash]
        
        try:
            if '--' not in line:
                return None
            
            parts = line.strip().split('--', 1)
            if len(parts) != 2:
                return None
            
            attributes_str, image_name = parts
            elements = attributes_str.split()
            
            if len(elements) < 206:
                return None
            
            # Parse landmarks (98 points × 2 coordinates = 196)
            landmarks_flat = [float(x) for x in elements[:196]]
            landmarks = np.array(landmarks_flat, dtype=np.float32).reshape(98, 2)
            
            # Parse bounding box
            bbox = [float(x) for x in elements[196:200]]
            
            # Parse attributes
            attributes = [int(x) for x in elements[200:206]]
            
            result = {
                'landmarks': landmarks,
                'bbox': bbox,
                'attributes': np.array(attributes, dtype=np.float32),
                'image_name': image_name.strip()
            }
            
            self.cache[line_hash] = result
            return result
            
        except Exception as e:
            print(f"Parsing error: {e}")
            return None


class WFLWImageCache:
    """Optimized image cache"""
    
    def __init__(self, images_path: str):
        self.images_path = Path(images_path)
        self.path_cache = {}
        self.stats = {'hits': 0, 'misses': 0}
        self._build_cache()
    
    def _build_cache(self):
        """Build map of all available images"""
        print("Building image cache...")
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for root, dirs, files in os.walk(self.images_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    file_path = Path(root) / file
                    
                    if '--' in Path(root).name:
                        category = Path(root).name.split('--', 1)[1]
                        key = f"{category}/{file}"
                        self.path_cache[key] = str(file_path)
        
        print(f"Cache built: {len(self.path_cache)} images")
    
    def get_image_path(self, image_name: str) -> Optional[str]:
        """Get image path"""
        if image_name in self.path_cache:
            self.stats['hits'] += 1
            return self.path_cache[image_name]
        
        self.stats['misses'] += 1
        return None


class WFLWTransforms:
    """Basic transforms with Albumentations"""
    
    def __init__(self, config: WFLWConfig, training: bool = True):
        self.config = config
        self.training = training
        
        if training:
            self.transforms = A.Compose([
                A.HorizontalFlip(p=config.train_augmentations['horizontal_flip_prob']),
                A.Rotate(
                    limit=config.train_augmentations['rotation_limit'], 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value=0, 
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=config.train_augmentations['brightness_limit'],
                    contrast_limit=config.train_augmentations['contrast_limit'],
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=config.train_augmentations['hue_shift_limit'],
                    sat_shift_limit=config.train_augmentations['saturation_shift_limit'],
                    val_shift_limit=0,
                    p=0.3
                ),
                A.OneOf([
                    A.Blur(blur_limit=config.train_augmentations['blur_limit']),
                    A.GaussNoise(var_limit=(10, config.train_augmentations['noise_limit'])),
                ], p=0.2),
                A.Resize(config.image_size[0], config.image_size[1]),
                A.Normalize(mean=config.image_mean, std=config.image_std),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        else:
            self.transforms = A.Compose([
                A.Resize(config.image_size[0], config.image_size[1]),
                A.Normalize(mean=config.image_mean, std=config.image_std),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def __call__(self, image: np.ndarray, landmarks: np.ndarray, bbox: Optional[List] = None) -> Dict:
        """Apply transforms"""
        
        if bbox is not None:
            image, landmarks = self._crop_face(image, landmarks, bbox)
        
        keypoints = [(x, y) for x, y in landmarks]
        
        try:
            transformed = self.transforms(image=image, keypoints=keypoints)
            
            landmarks_transformed = np.array(transformed['keypoints'], dtype=np.float32)
            landmarks_normalized = self._normalize_landmarks(landmarks_transformed)
            
            return {
                'image': transformed['image'],
                'landmarks': torch.tensor(landmarks_normalized, dtype=torch.float32),
                'success': True
            }
            
        except Exception as e:
            print(f"Transform error: {e}")
            return self._get_fallback()
    
    def _crop_face(self, image: np.ndarray, landmarks: np.ndarray, bbox: List) -> Tuple[np.ndarray, np.ndarray]:
        """Crop image to face with padding"""
        x_min, y_min, x_max, y_max = bbox
        
        width = x_max - x_min
        height = y_max - y_min
        padding = self.config.bbox_padding
        
        x_min = max(0, int(x_min - width * padding))
        y_min = max(0, int(y_min - height * padding))
        x_max = min(image.shape[1], int(x_max + width * padding))
        y_max = min(image.shape[0], int(y_max + height * padding))
        
        cropped_image = image[y_min:y_max, x_min:x_max]
        
        adjusted_landmarks = landmarks.copy()
        adjusted_landmarks[:, 0] -= x_min
        adjusted_landmarks[:, 1] -= y_min
        
        return cropped_image, adjusted_landmarks
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks to [-0.5, 0.5] range"""
        h, w = self.config.image_size
        
        normalized = landmarks.copy()
        normalized[:, 0] /= w
        normalized[:, 1] /= h
        normalized -= 0.5
        
        if self.config.clip_landmarks:
            normalized = np.clip(normalized, -0.5, 0.5)
        
        return normalized
    
    def _get_fallback(self) -> Dict:
        """Return fallback sample on error"""
        return {
            'image': torch.zeros(3, *self.config.image_size),
            'landmarks': torch.zeros(self.config.num_landmarks, 2),
            'success': False
        }


class WFLWTransformsEnhanced(WFLWTransforms):
    """Enhanced transforms against overfitting - AGGRESSIVE AUGMENTATIONS"""
    
    def __init__(self, config: WFLWConfigEnhanced, training: bool = True):
        self.config = config
        self.training = training
        
        if training:
            print("Using ENHANCED AUGMENTATIONS - Anti-overfitting mode!")
            
            self.transforms = A.Compose([
                # Geometric augmentations
                A.HorizontalFlip(p=config.train_augmentations['horizontal_flip_prob']),
                
                A.OneOf([
                    A.Rotate(
                        limit=config.train_augmentations['rotation_limit'], 
                        border_mode=cv2.BORDER_REFLECT,
                        p=1.0
                    ),
                    A.Affine(
                        scale=(1.0 - config.train_augmentations['scale_limit'], 
                               1.0 + config.train_augmentations['scale_limit']),
                        translate_percent=config.train_augmentations['shift_limit'],
                        rotate=(-config.train_augmentations['rotation_limit'], 
                               config.train_augmentations['rotation_limit']),
                        shear=(-10, 10),
                        mode=cv2.BORDER_REFLECT,
                        p=1.0
                    )
                ], p=0.8),
                
                # Elastic & distortion
                A.OneOf([
                    A.ElasticTransform(
                        alpha=config.train_augmentations['elastic_alpha'],
                        sigma=config.train_augmentations['elastic_sigma'],
                        alpha_affine=0,
                        border_mode=cv2.BORDER_REFLECT,
                        p=1.0
                    ),
                    A.GridDistortion(
                        num_steps=5,
                        distort_limit=config.train_augmentations['grid_distortion_limit'],
                        border_mode=cv2.BORDER_REFLECT,
                        p=1.0
                    ),
                    A.OpticalDistortion(
                        distort_limit=config.train_augmentations['optical_distortion_limit'],
                        shift_limit=0.05,
                        border_mode=cv2.BORDER_REFLECT,
                        p=1.0
                    )
                ], p=0.3),
                
                # Photometric augmentations
                A.RandomBrightnessContrast(
                    brightness_limit=config.train_augmentations['brightness_limit'],
                    contrast_limit=config.train_augmentations['contrast_limit'],
                    p=0.6
                ),
                
                A.HueSaturationValue(
                    hue_shift_limit=config.train_augmentations['hue_shift_limit'],
                    sat_shift_limit=config.train_augmentations['saturation_shift_limit'],
                    val_shift_limit=20,
                    p=0.4
                ),
                
                # Noise & blur variations
                A.OneOf([
                    A.GaussianBlur(
                        blur_limit=(3, config.train_augmentations['blur_limit']),
                        p=1.0
                    ),
                    A.MotionBlur(
                        blur_limit=config.train_augmentations['blur_limit'],
                        p=1.0
                    ),
                    A.MedianBlur(blur_limit=3, p=1.0)
                ], p=config.train_augmentations['gaussian_blur_prob']),
                
                A.OneOf([
                    A.GaussNoise(
                        var_limit=(5, config.train_augmentations['noise_limit']),
                        p=1.0
                    ),
                    A.MultiplicativeNoise(
                        multiplier=(0.9, 1.1),
                        per_channel=True,
                        p=1.0
                    ),
                    A.ISONoise(
                        color_shift=(0.01, 0.05),
                        intensity=(0.1, 0.5),
                        p=1.0
                    )
                ], p=0.2),
                
                # Dropout & cutout
                A.CoarseDropout(
                    max_holes=config.train_augmentations['coarse_dropout_max_holes'],
                    max_height=int(224 * config.train_augmentations['coarse_dropout_max_size']),
                    max_width=int(224 * config.train_augmentations['coarse_dropout_max_size']),
                    fill_value=0,
                    p=config.train_augmentations['coarse_dropout_prob']
                ),
                
                # Color space transforms
                A.OneOf([
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                    A.ChannelShuffle(p=1.0),
                    A.ToGray(p=1.0),
                    A.ToSepia(p=1.0)
                ], p=config.train_augmentations['channel_shuffle_prob']),
                
                A.Resize(config.image_size[0], config.image_size[1]),
                
                # Pixel-level transforms
                A.OneOf([
                    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0),
                    A.Emboss(alpha=(0.1, 0.3), strength=(0.5, 1.5), p=1.0),
                    A.RandomToneCurve(scale=0.1, p=1.0)
                ], p=0.1),
                
                A.Normalize(mean=config.image_mean, std=config.image_std),
                ToTensorV2()
                
            ], keypoint_params=A.KeypointParams(
                format='xy', 
                remove_invisible=False
            ))
        else:
            # Test transforms - no changes
            self.transforms = A.Compose([
                A.Resize(config.image_size[0], config.image_size[1]),
                A.Normalize(mean=config.image_mean, std=config.image_std),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


class WFLWDataset(Dataset):
    """Production-ready WFLW dataset"""
    
    def __init__(self, 
                 annotations_path: str,
                 images_path: str,
                 split: str = 'train',
                 config: Optional[WFLWConfig] = None,
                 max_samples: Optional[int] = None):
        
        self.config = config or WFLWConfig()
        self.split = split
        self.training = (split == 'train')
        
        # Initialize components
        self.parser = WFLWAnnotationParser(self.config)
        self.image_cache = WFLWImageCache(images_path)
        
        # Use Enhanced transforms for training if config is enhanced
        if self.training and isinstance(self.config, WFLWConfigEnhanced):
            self.transforms = WFLWTransformsEnhanced(self.config, self.training)
            print("Using ENHANCED TRANSFORMS for training!")
        else:
            self.transforms = WFLWTransforms(self.config, self.training)
        
        # Load data
        self.data = self._load_data(annotations_path, max_samples)
        self.valid_indices = list(range(len(self.data)))
        
        print(f"{split.upper()} Dataset: {len(self.data)} samples")
    
    def _load_data(self, annotations_path: str, max_samples: Optional[int]) -> List[Dict]:
        """Load data with optimization"""
        
        annotations_dir = Path(annotations_path)
        
        if self.split == 'train':
            file_path = annotations_dir / "list_98pt_rect_attr_train_test" / "list_98pt_rect_attr_train.txt"
        elif self.split == 'test':
            file_path = annotations_dir / "list_98pt_rect_attr_train_test" / "list_98pt_rect_attr_test.txt"
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        print(f"Loading {self.split} from {file_path}")
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        if max_samples:
            lines = lines[:max_samples]
        
        data = []
        for i, line in enumerate(lines):
            if i % 2000 == 0:
                print(f"   Processed {i}/{len(lines)}")
            
            parsed = self.parser.parse_line(line)
            if parsed is None:
                continue
            
            image_path = self.image_cache.get_image_path(parsed['image_name'])
            if image_path is None:
                continue
            
            parsed['image_path'] = image_path
            data.append(parsed)
        
        success_rate = len(data) / len(lines) * 100
        print(f"Loading success: {success_rate:.1f}% ({len(data)}/{len(lines)})")
        
        return data
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Main sample loading function"""
        try:
            real_idx = self.valid_indices[idx]
            sample = self.data[real_idx]
            
            # Load image
            image = cv2.imread(sample['image_path'])
            if image is None:
                return self._get_error_sample(idx)
            
            # Convert BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            transformed = self.transforms(
                image=image,
                landmarks=sample['landmarks'],
                bbox=sample['bbox']
            )
            
            if not transformed['success']:
                return self._get_error_sample(idx)
            
            result = {
                'image': transformed['image'],
                'landmarks': transformed['landmarks'],
                'attributes': torch.tensor(sample['attributes'], dtype=torch.float32),
                'bbox': torch.tensor(sample['bbox'], dtype=torch.float32),
                'image_name': sample['image_name'],
                'sample_idx': real_idx
            }
            
            return result
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self._get_error_sample(idx)
    
    def _get_error_sample(self, idx: int) -> Dict:
        """Return fallback sample"""
        return {
            'image': torch.zeros(3, *self.config.image_size),
            'landmarks': torch.zeros(self.config.num_landmarks, 2),
            'attributes': torch.zeros(self.config.num_attributes),
            'bbox': torch.zeros(4),
            'image_name': f'error_{idx}',
            'sample_idx': idx
        }


class WFLWDataModule:
    """Production DataModule for WFLW"""
    
    def __init__(self, 
                 annotations_path: str,
                 images_path: str,
                 config: Optional[WFLWConfig] = None,
                 train_max_samples: Optional[int] = None,
                 test_max_samples: Optional[int] = None):
        
        self.annotations_path = annotations_path
        self.images_path = images_path
        self.config = config or WFLWConfig()
        self.train_max_samples = train_max_samples
        self.test_max_samples = test_max_samples
        
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        
    def setup(self):
        """Prepare datasets and loaders"""
        print("Initializing WFLW DataModule")
        
        self.train_dataset = WFLWDataset(
            annotations_path=self.annotations_path,
            images_path=self.images_path,
            split='train',
            config=self.config,
            max_samples=self.train_max_samples
        )
        
        self.test_dataset = WFLWDataset(
            annotations_path=self.annotations_path,
            images_path=self.images_path,
            split='test',
            config=self.config,
            max_samples=self.test_max_samples
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
        
        print(f"DataModule ready:")
        print(f"   Train: {len(self.train_dataset)} samples, {len(self.train_loader)} batches")
        print(f"   Test: {len(self.test_dataset)} samples, {len(self.test_loader)} batches")
    
    def get_train_loader(self) -> DataLoader:
        """Return train loader"""
        if self.train_loader is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")
        return self.train_loader
    
    def get_test_loader(self) -> DataLoader:
        """Return test loader"""
        if self.test_loader is None:
            raise RuntimeError("DataModule not initialized. Call setup() first.")
        return self.test_loader


def create_wflw_dataloader(annotations_path: str, 
                          images_path: str,
                          batch_size: int = 32,
                          image_size: Tuple[int, int] = (224, 224),
                          num_workers: int = 4,
                          train_max_samples: Optional[int] = None,
                          test_max_samples: Optional[int] = None,
                          enhanced_augmentations: bool = False) -> WFLWDataModule:
    """Helper function to quickly create WFLW DataModule"""
    
    if enhanced_augmentations:
        config = WFLWConfigEnhanced()
        print("Using ENHANCED config with strong augmentations!")
    else:
        config = WFLWConfig()
        print("Using basic config")
    
    config.batch_size = batch_size
    config.image_size = image_size
    config.num_workers = num_workers
    
    data_module = WFLWDataModule(
        annotations_path=annotations_path,
        images_path=images_path,
        config=config,
        train_max_samples=train_max_samples,
        test_max_samples=test_max_samples
    )
    
    data_module.setup()
    return data_module


if __name__ == "__main__":
    # Test paths
    annotations_path = "/home/zimmermann/Projekty/SI_WFLW/data/WFLW/annotations"
    images_path = "/home/zimmermann/Projekty/SI_WFLW/data/WFLW/images"
    
    try:
        print("Testing ENHANCED DataLoader")
        
        # Test with enhanced augmentations
        data_module = create_wflw_dataloader(
            annotations_path=annotations_path,
            images_path=images_path,
            batch_size=16,
            image_size=(224, 224),
            num_workers=2,
            train_max_samples=100,
            test_max_samples=50,
            enhanced_augmentations=True  # Enable enhanced augmentations!
        )
        
        # Test loading
        train_loader = data_module.get_train_loader()
        batch = next(iter(train_loader))
        
        print("SUCCESS!")
        print(f"   Batch shape: {batch['image'].shape}")
        print(f"   Landmarks shape: {batch['landmarks'].shape}")
        print(f"   Landmarks range: [{batch['landmarks'].min():.3f}, {batch['landmarks'].max():.3f}]")
        print(f"   Attributes shape: {batch['attributes'].shape}")
        
        print("ENHANCED Production DataLoader ready for training!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
