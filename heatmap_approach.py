#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WFLW Heatmap Network - Enhanced Version with TensorBoard Logging
Professional implementation for facial landmark detection using heatmap regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from pathlib import Path
import time
from typing import Tuple, Dict, List, Any, Optional
import datetime

class WFLWHeatmapNetworkLarge(nn.Module):
    """
    Large-scale heatmap network for WFLW dataset with regularization.
    Optimized for RTX 4070 (12GB VRAM).
    
    Args:
        num_landmarks (int): Number of facial landmarks (default: 98)
        backbone (str): Backbone architecture ('resnet18', 'resnet34', 'resnet50')
        heatmap_size (int): Output heatmap resolution (default: 64)
    """
    
    def __init__(self, num_landmarks: int = 98, backbone: str = 'resnet50', heatmap_size: int = 64):
        super(WFLWHeatmapNetworkLarge, self).__init__()
        
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        
        # Configure backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            backbone_out = 2048
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            backbone_out = 512
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            backbone_out = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classification layers (avgpool, fc)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Progressive upsampling with dropout regularization
        if backbone == 'resnet50':
            # ResNet50: 7x7x2048 -> 64x64x98
            self.deconv_layers = nn.Sequential(
                # 7x7 -> 14x14
                nn.ConvTranspose2d(backbone_out, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                
                # 14x14 -> 28x28
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                
                # 28x28 -> 56x56
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.05),
                
                # Refinement layer
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.05),
            )
        else:
            # ResNet18/34: 7x7x512 -> 64x64x98
            self.deconv_layers = nn.Sequential(
                # 7x7 -> 14x14
                nn.ConvTranspose2d(backbone_out, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                
                # 14x14 -> 28x28
                nn.ConvTranspose2d(256, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                
                # 28x28 -> 56x56
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.05),
            )
        
        # Final heatmap layer
        self.final_layer = nn.Conv2d(128, num_landmarks, 1, 1, 0)
        
        # Optional upsampling to target size
        if heatmap_size != 56:
            self.upsample = nn.Upsample(
                size=(heatmap_size, heatmap_size), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            self.upsample = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for deconvolution layers."""
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Final layer initialization
        nn.init.normal_(self.final_layer.weight, std=0.001)
        nn.init.constant_(self.final_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Heatmaps tensor of shape (B, num_landmarks, heatmap_size, heatmap_size)
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Deconvolution
        upsampled = self.deconv_layers(features)
        
        # Generate heatmaps
        heatmaps = self.final_layer(upsampled)
        
        # Upsample if needed
        if self.upsample is not None:
            heatmaps = self.upsample(heatmaps)
        
        return heatmaps


def generate_target_heatmaps_fixed(landmarks: torch.Tensor, 
                                 image_size: int = 224, 
                                 heatmap_size: int = 64, 
                                 sigma: float = 2.0) -> torch.Tensor:
    """
    Generate target heatmaps from landmark coordinates.
    
    Args:
        landmarks: Tensor of shape (B, N, 2) with normalized coordinates [-0.5, 0.5]
        image_size: Original image size
        heatmap_size: Target heatmap resolution
        sigma: Gaussian kernel standard deviation
        
    Returns:
        Target heatmaps tensor of shape (B, N, heatmap_size, heatmap_size)
    """
    batch_size, num_landmarks, _ = landmarks.shape
    device = landmarks.device
    
    # Denormalize landmarks from [-0.5, 0.5] to [0, heatmap_size]
    landmarks_heatmap = (landmarks + 0.5) * heatmap_size
    
    # Create target heatmaps on the same device
    target_heatmaps = torch.zeros(batch_size, num_landmarks, heatmap_size, heatmap_size, device=device)
    
    # Gaussian kernel
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1  # Ensure odd size
    
    x = torch.arange(0, size, dtype=torch.float32, device=device)
    y = x.unsqueeze(1)
    x0, y0 = size // 2, size // 2
    gaussian = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    
    for b in range(batch_size):
        for l in range(num_landmarks):
            x_coord, y_coord = landmarks_heatmap[b, l]
            
            # Skip if landmark is out of bounds
            if x_coord < 0 or x_coord >= heatmap_size or y_coord < 0 or y_coord >= heatmap_size:
                continue
            
            x_coord, y_coord = int(round(x_coord.item())), int(round(y_coord.item()))
            
            # Place Gaussian
            x_min = max(0, x_coord - size // 2)
            x_max = min(heatmap_size, x_coord + size // 2 + 1)
            y_min = max(0, y_coord - size // 2)
            y_max = min(heatmap_size, y_coord + size // 2 + 1)
            
            g_x_min = max(0, size // 2 - x_coord)
            g_x_max = g_x_min + (x_max - x_min)
            g_y_min = max(0, size // 2 - y_coord)
            g_y_max = g_y_min + (y_max - y_min)
            
            if g_x_max > g_x_min and g_y_max > g_y_min:
                target_heatmaps[b, l, y_min:y_max, x_min:x_max] = gaussian[g_y_min:g_y_max, g_x_min:g_x_max]
    
    return target_heatmaps


def decode_heatmaps_fixed(heatmaps: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """
    Decode heatmaps to landmark coordinates with sub-pixel accuracy.
    
    Args:
        heatmaps: Tensor of shape (B, N, H, W)
        image_size: Original image size for normalization
        
    Returns:
        Landmarks tensor of shape (B, N, 2) with normalized coordinates [-0.5, 0.5]
    """
    batch_size, num_landmarks, heatmap_height, heatmap_width = heatmaps.shape
    device = heatmaps.device
    
    # Apply softmax to get probability distributions
    heatmaps_prob = F.softmax(heatmaps.view(batch_size, num_landmarks, -1), dim=2)
    heatmaps_prob = heatmaps_prob.view(batch_size, num_landmarks, heatmap_height, heatmap_width)
    
    # Find max positions
    heatmaps_flat = heatmaps_prob.view(batch_size, num_landmarks, -1)
    max_vals, max_indices = torch.max(heatmaps_flat, dim=2)
    
    # Convert flat indices to 2D coordinates
    y_coords = (max_indices // heatmap_width).float()
    x_coords = (max_indices % heatmap_width).float()
    
    # Sub-pixel refinement using center of mass around peak
    landmarks = []
    for b in range(batch_size):
        batch_landmarks = []
        for l in range(num_landmarks):
            x_int, y_int = int(x_coords[b, l]), int(y_coords[b, l])
            
            # Extract 3x3 region around peak for sub-pixel refinement
            x_min, x_max = max(0, x_int-1), min(heatmap_width, x_int+2)
            y_min, y_max = max(0, y_int-1), min(heatmap_height, y_int+2)
            
            if x_max > x_min and y_max > y_min:
                region = heatmaps_prob[b, l, y_min:y_max, x_min:x_max]
                
                # Center of mass in region
                y_indices = torch.arange(y_min, y_max, dtype=torch.float32, device=device).unsqueeze(1)
                x_indices = torch.arange(x_min, x_max, dtype=torch.float32, device=device).unsqueeze(0)
                
                total_mass = region.sum()
                if total_mass > 0:
                    x_refined = (region * x_indices).sum() / total_mass
                    y_refined = (region * y_indices).sum() / total_mass
                else:
                    x_refined, y_refined = x_coords[b, l], y_coords[b, l]
            else:
                x_refined, y_refined = x_coords[b, l], y_coords[b, l]
            
            batch_landmarks.append([x_refined, y_refined])
        
        landmarks.append(batch_landmarks)
    
    landmarks = torch.tensor(landmarks, dtype=torch.float32, device=device)
    
    # Normalize to [-0.5, 0.5]
    landmarks[:, :, 0] /= heatmap_width   # x coordinates
    landmarks[:, :, 1] /= heatmap_height  # y coordinates
    landmarks -= 0.5
    
    return landmarks


def create_landmark_visualization(images: torch.Tensor, 
                                pred_landmarks: torch.Tensor, 
                                target_landmarks: torch.Tensor) -> torch.Tensor:
    """
    Create visualization of landmarks overlaid on images for TensorBoard.
    
    Args:
        images: Input images tensor (B, 3, H, W)
        pred_landmarks: Predicted landmarks (B, N, 2)
        target_landmarks: Ground truth landmarks (B, N, 2)
        
    Returns:
        Visualization tensor for TensorBoard
    """
    batch_size = min(4, images.size(0))
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images_denorm = torch.clamp(images * std + mean, 0, 1)
    
    # Denormalize landmarks
    pred_denorm = (pred_landmarks + 0.5) * 224
    target_denorm = (target_landmarks + 0.5) * 224
    
    vis_images = []
    
    for i in range(batch_size):
        img = images_denorm[i].clone()
        
        # Convert to numpy for OpenCV operations
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Draw predicted landmarks (red)
        pred_points = pred_denorm[i].cpu().numpy()
        for x, y in pred_points:
            cv2.circle(img_np, (int(x), int(y)), 2, (0, 0, 255), -1)
        
        # Draw target landmarks (green)
        target_points = target_denorm[i].cpu().numpy()
        for x, y in target_points:
            cv2.circle(img_np, (int(x), int(y)), 1, (0, 255, 0), -1)
        
        # Convert back to tensor
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_np).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)
        vis_images.append(img_tensor)
    
    return torch.stack(vis_images)


def create_heatmap_visualization(heatmaps: torch.Tensor, max_landmarks: int = 8) -> torch.Tensor:
    """
    Create heatmap visualization for TensorBoard.
    
    Args:
        heatmaps: Generated heatmaps (B, N, H, W)
        max_landmarks: Maximum number of landmarks to visualize
        
    Returns:
        Heatmap visualization tensor
    """
    # Take first sample and first few landmarks
    sample_heatmaps = heatmaps[0][:max_landmarks]  # (N, H, W)
    
    # Normalize heatmaps to [0, 1]
    normalized_heatmaps = []
    for hm in sample_heatmaps:
        hm_min, hm_max = hm.min(), hm.max()
        if hm_max > hm_min:
            hm_norm = (hm - hm_min) / (hm_max - hm_min)
        else:
            hm_norm = hm
        normalized_heatmaps.append(hm_norm.unsqueeze(0))  # Add channel dimension
    
    return torch.stack(normalized_heatmaps)


class WFLWHeatmapTrainerMaxRegularization:
    """
    Trainer with maximum regularization and TensorBoard logging.
    
    Args:
        model: The neural network model
        config: Training configuration
        save_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
    """
    
    def __init__(self, model: nn.Module, config: Any, save_dir: str = './checkpoints', log_dir: str = './runs'):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # TensorBoard setup
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = Path(log_dir) / f'wflw_heatmap_{timestamp}'
        self.writer = SummaryWriter(self.log_dir)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer with conservative settings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,
            weight_decay=5e-3,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=8,
            verbose=True,
            min_lr=1e-7
        )
        
        # Early stopping configuration
        self.early_stopping_patience = 20
        self.early_stopping_counter = 0
        self.early_stopping_min_delta = 0.0005
        
        # Gradient clipping
        self.max_grad_norm = 0.3
        
        # Tracking metrics
        self.best_nme = float('inf')
        self.global_step = 0
        
        # Log hyperparameters to TensorBoard
        self.writer.add_hparams(
            {
                'lr': 3e-4,
                'weight_decay': 5e-3,
                'batch_size': getattr(config, 'batch_size', 48),
                'backbone': 'resnet50',
                'heatmap_size': 64,
                'max_grad_norm': self.max_grad_norm
            },
            {}
        )
        
        print(f"Trainer Configuration:")
        print(f"  Learning Rate: {3e-4}")
        print(f"  Weight Decay: {5e-3}")
        print(f"  Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Device: {self.device}")
        print(f"  TensorBoard logs: {self.log_dir}")
    
    def train_epoch(self, train_loader) -> float:
        """
        Execute one training epoch with TensorBoard logging.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                # Generate target heatmaps
                target_heatmaps = generate_target_heatmaps_fixed(
                    landmarks, 
                    image_size=self.config.image_size[0],
                    heatmap_size=64,
                    sigma=2.0
                )
                
                # Forward pass
                pred_heatmaps = self.model(images)
                loss = self.criterion(pred_heatmaps, target_heatmaps)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             max_norm=self.max_grad_norm)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                self.global_step += 1
                
                # Log to TensorBoard every 50 steps
                if batch_idx % 50 == 0:
                    self.writer.add_scalar('Train/Loss_Step', loss.item(), self.global_step)
                    self.writer.add_scalar('Train/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.global_step)
                    
                    # Log gradient norms
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    self.writer.add_scalar('Train/Gradient_Norm', total_norm, self.global_step)
                
                # Progress logging every 100 batches
                if batch_idx % 100 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.6f}, LR: {current_lr:.6f}")
                    
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader, epoch: int) -> Tuple[float, float]:
        """
        Execute one validation epoch with TensorBoard logging.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average validation loss, average NME)
        """
        self.model.eval()
        total_loss = 0.0
        total_nme = 0.0
        num_samples = 0
        
        # For visualizations
        viz_images, viz_pred_landmarks, viz_target_landmarks = None, None, None
        viz_heatmaps = None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    images = batch['image'].to(self.device)
                    landmarks = batch['landmarks'].to(self.device)
                    
                    # Generate target heatmaps
                    target_heatmaps = generate_target_heatmaps_fixed(
                        landmarks,
                        image_size=self.config.image_size[0],
                        heatmap_size=64,
                        sigma=2.0
                    )
                    
                    # Forward pass
                    pred_heatmaps = self.model(images)
                    loss = self.criterion(pred_heatmaps, target_heatmaps)
                    
                    # Decode predictions
                    pred_landmarks = decode_heatmaps_fixed(pred_heatmaps, self.config.image_size[0])
                    
                    # Calculate NME
                    nme = self.calculate_nme(pred_landmarks, landmarks)
                    
                    total_loss += loss.item()
                    total_nme += nme * images.size(0)
                    num_samples += images.size(0)
                    
                    # Collect first batch for visualization
                    if batch_idx == 0:
                        viz_images = images[:4]
                        viz_pred_landmarks = pred_landmarks[:4]
                        viz_target_landmarks = landmarks[:4]
                        viz_heatmaps = pred_heatmaps[:1]
                        
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
        
        avg_loss = total_loss / len(val_loader)
        avg_nme = total_nme / num_samples
        
        # Log to TensorBoard
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/NME', avg_nme, epoch)
        
        # Add visualizations to TensorBoard
        if viz_images is not None:
            # Landmark visualizations
            landmark_vis = create_landmark_visualization(viz_images, viz_pred_landmarks, viz_target_landmarks)
            self.writer.add_images('Val/Landmarks', landmark_vis, epoch)
            
            # Original images
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            orig_images = torch.clamp(viz_images * std + mean, 0, 1)
            self.writer.add_images('Val/Original_Images', orig_images, epoch)
            
            # Heatmap visualizations (every 5 epochs)
            if viz_heatmaps is not None and epoch % 5 == 0:
                heatmap_vis = create_heatmap_visualization(viz_heatmaps)
                self.writer.add_images('Val/Heatmaps', heatmap_vis, epoch, dataformats='NCHW')
        
        return avg_loss, avg_nme
    
    def calculate_nme(self, predictions: torch.Tensor, targets: torch.Tensor, image_size: int = 224) -> float:
        """Calculate Normalized Mean Error (NME)."""
        # Denormalize
        pred_denorm = (predictions + 0.5) * image_size
        target_denorm = (targets + 0.5) * image_size
        
        # L2 distances
        distances = torch.norm(pred_denorm - target_denorm, dim=2)
        nme = distances.mean()
        
        return (nme / image_size).item()
    
    def check_early_stopping(self, val_nme: float) -> bool:
        """Check early stopping condition."""
        if val_nme < self.best_nme - self.early_stopping_min_delta:
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
                return True
        return False
    
    def train(self, train_loader, val_loader, num_epochs: int = 50):
        """
        Main training loop with TensorBoard logging.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
        """
        print(f"Starting training for maximum {num_epochs} epochs")
        print(f"TensorBoard command: tensorboard --logdir={self.log_dir.parent}")
        
        start_time = time.time()
        
        # Log model graph
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        self.writer.add_graph(self.model, dummy_input)
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            print(f"\nEPOCH {epoch}/{num_epochs}")
            print("=" * 60)
            
            try:
                # Training phase
                train_loss = self.train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_nme = self.validate_epoch(val_loader, epoch)
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Log epoch metrics
                self.writer.add_scalar('Train/Loss_Epoch', train_loss, epoch)
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Print epoch summary
                print(f"Training Loss:   {train_loss:.6f}")
                print(f"Validation Loss: {val_loss:.6f}")
                print(f"Validation NME:  {val_nme:.6f}")
                print(f"Learning Rate:   {current_lr:.6f}")
                print(f"Epoch Time:      {epoch_time:.1f}s")
                print(f"Early Stop Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                # Save best model
                if val_nme < self.best_nme:
                    self.best_nme = val_nme
                    self.early_stopping_counter = 0
                    print(f"New best NME: {val_nme:.6f}")
                    
                    # Log best metrics
                    self.writer.add_scalar('Best/NME', val_nme, epoch)
                    self.writer.add_scalar('Best/Loss', val_loss, epoch)
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_loss,
                        'val_nme': val_nme,
                        'config': self.config
                    }, self.save_dir / 'best_model_heatmap.pth')
                
                # Check early stopping
                if self.check_early_stopping(val_nme):
                    break
                
                # Save checkpoint every 10 epochs
                if epoch % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_loss,
                        'val_nme': val_nme
                    }, self.save_dir / f'checkpoint_epoch_{epoch}.pth')
                    print(f"Checkpoint saved: epoch_{epoch}.pth")
                
            except Exception as e:
                print(f"Error in epoch {epoch}: {str(e)}")
                continue
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.1f} hours")
        print(f"Best NME achieved: {self.best_nme:.6f}")
        print(f"TensorBoard logs saved in: {self.log_dir}")
        
        # Close TensorBoard writer
        self.writer.close()


def train_heatmap_enhanced():
    """
    Main training function with enhanced augmentations and TensorBoard logging.
    
    Raises:
        ImportError: If enhanced data loader configuration is not available
        RuntimeError: If CUDA is not available when expected
        FileNotFoundError: If dataset paths are not found
    """
    try:
        from data_loader import WFLWConfigEnhanced, WFLWDataModule
        
        print("WFLW Enhanced Training with TensorBoard Logging")
        print("Configuration: Strong augmentations + maximum regularization")
        print("Target: NME < 0.07")
        
        # Enhanced configuration
        config = WFLWConfigEnhanced()
        config.batch_size = 48
        config.image_size = (224, 224)
        config.num_workers = 8
        
        # Data module
        data_module = WFLWDataModule(
            annotations_path="/home/zimmermann/Projekty/SI_WFLW/data/WFLW/annotations",
            images_path="/home/zimmermann/Projekty/SI_WFLW/data/WFLW/images",
            config=config
        )
        
        data_module.setup()
        
        # Model
        model = WFLWHeatmapNetworkLarge(
            num_landmarks=98, 
            backbone='resnet50',
            heatmap_size=64
        )
        
        # Trainer with TensorBoard
        trainer = WFLWHeatmapTrainerMaxRegularization(
            model, 
            config, 
            save_dir='./wflw_checkpoints_enhanced',
            log_dir='./runs'
        )
        
        # Get data loaders
        train_loader = data_module.get_train_loader()
        val_loader = data_module.get_test_loader()
        
        # GPU memory test
        print("Testing GPU memory...")
        test_batch = next(iter(train_loader))
        test_images = test_batch['image'][:4].to(trainer.device)
        
        with torch.no_grad():
            test_output = model(test_images)
            print(f"GPU test passed. Output shape: {test_output.shape}")
        
        # Start training
        trainer.train(train_loader, val_loader, num_epochs=50)
        
    except ImportError as e:
        print(f"Enhanced configuration not available: {e}")
        raise ImportError(f"Failed to import enhanced data loader: {e}")
    except FileNotFoundError as e:
        print(f"Dataset path not found: {e}")
        raise FileNotFoundError(f"Dataset files not accessible: {e}")
    except RuntimeError as e:
        print(f"Runtime error during training: {e}")
        raise RuntimeError(f"Training failed: {e}")


def train_heatmap_fallback():
    """
    Fallback training function using basic augmentations with TensorBoard.
    
    Raises:
        ImportError: If basic data loader configuration is not available
        RuntimeError: If CUDA is not available when expected
    """
    try:
        from data_loader import WFLWConfig, WFLWDataModule
        
        print("WARNING: Using fallback training with basic augmentations")
        print("TensorBoard logging enabled")
        
        # Basic configuration
        config = WFLWConfig()
        config.batch_size = 58
        config.image_size = (224, 224)
        config.num_workers = 12
        
        # Data module
        data_module = WFLWDataModule(
            annotations_path="/home/zimmermann/Projekty/SI_WFLW/data/WFLW/annotations",
            images_path="/home/zimmermann/Projekty/SI_WFLW/data/WFLW/images",
            config=config
        )
        
        data_module.setup()
        
        # Model
        model = WFLWHeatmapNetworkLarge(
            num_landmarks=98, 
            backbone='resnet50',
            heatmap_size=64
        )
        
        # Trainer with TensorBoard
        trainer = WFLWHeatmapTrainerMaxRegularization(
            model, 
            config, 
            save_dir='./wflw_checkpoints_fallback',
            log_dir='./runs'
        )
        
        # Get data loaders
        train_loader = data_module.get_train_loader()
        val_loader = data_module.get_test_loader()
        
        # Start training
        trainer.train(train_loader, val_loader, num_epochs=50)
        
    except ImportError as e:
        print(f"Basic configuration not available: {e}")
        raise ImportError(f"Failed to import basic data loader: {e}")
    except RuntimeError as e:
        print(f"Runtime error during fallback training: {e}")
        raise RuntimeError(f"Fallback training failed: {e}")


def train_heatmap_large():
    """
    Main training function with TensorBoard logging.
    Attempts enhanced training first, then falls back to basic.
    """
    try:
        print("Attempting enhanced training with TensorBoard...")
        train_heatmap_enhanced()
    except ImportError as e:
        print(f"Enhanced configuration unavailable: {e}")
        print("Falling back to basic training with TensorBoard...")
        try:
            train_heatmap_fallback()
        except Exception as fallback_error:
            print(f"Both enhanced and fallback training failed: {fallback_error}")
            raise RuntimeError(f"All training configurations failed. Last error: {fallback_error}")
    except Exception as e:
        print(f"Enhanced training failed: {e}")
        print("Attempting fallback training with TensorBoard...")
        try:
            train_heatmap_fallback()
        except Exception as fallback_error:
            print(f"Both enhanced and fallback training failed: {fallback_error}")
            raise RuntimeError(f"All training configurations failed. Last error: {fallback_error}")


def load_model_checkpoint(checkpoint_path: str, model: nn.Module, device: str = 'cuda') -> Dict[str, Any]:
    """
    Load model from checkpoint with error handling.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        device: Device to load model on
        
    Returns:
        Dictionary containing checkpoint information
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best NME: {checkpoint.get('val_nme', 'Unknown')}")
        
        return checkpoint
        
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}")


def evaluate_model_with_tensorboard(model: nn.Module, 
                                  test_loader, 
                                  device: str = 'cuda', 
                                  log_dir: str = './evaluation_runs'):
    """
    Evaluate trained model on test set with TensorBoard logging.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device for evaluation
        log_dir: Directory for TensorBoard logs
        
    Returns:
        Dictionary containing evaluation metrics
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_log_dir = Path(log_dir) / f'evaluation_{timestamp}'
    writer = SummaryWriter(eval_log_dir)
    
    model.eval()
    total_nme = 0.0
    num_samples = 0
    nme_scores = []
    
    print("Evaluating model on test set...")
    print(f"TensorBoard logs: {eval_log_dir}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                images = batch['image'].to(device)
                landmarks = batch['landmarks'].to(device)
                
                # Forward pass
                pred_heatmaps = model(images)
                pred_landmarks = decode_heatmaps_fixed(pred_heatmaps, 224)
                
                # Calculate NME for each sample
                batch_nmes = []
                for i in range(images.size(0)):
                    pred_denorm = (pred_landmarks[i] + 0.5) * 224
                    target_denorm = (landmarks[i] + 0.5) * 224
                    
                    distances = torch.norm(pred_denorm - target_denorm, dim=1)
                    nme = (distances.mean() / 224).item()
                    nme_scores.append(nme)
                    batch_nmes.append(nme)
                    total_nme += nme
                    num_samples += 1
                
                # Log batch NME to TensorBoard
                writer.add_scalar('Test/Batch_NME', np.mean(batch_nmes), batch_idx)
                
                # Add visualizations for first few batches
                if batch_idx < 5:
                    vis_images = create_landmark_visualization(images[:4], pred_landmarks[:4], landmarks[:4])
                    writer.add_images(f'Test/Batch_{batch_idx}_Landmarks', vis_images, 0)
                    
                    # Add heatmap visualizations
                    heatmap_vis = create_heatmap_visualization(pred_heatmaps[:1])
                    writer.add_images(f'Test/Batch_{batch_idx}_Heatmaps', heatmap_vis, 0, dataformats='NCHW')
                
                if batch_idx % 50 == 0:
                    print(f"Processed {batch_idx}/{len(test_loader)} batches")
                    
            except Exception as e:
                print(f"Error in evaluation batch {batch_idx}: {e}")
                continue
    
    # Calculate statistics
    avg_nme = total_nme / num_samples
    std_nme = np.std(nme_scores)
    min_nme = np.min(nme_scores)
    max_nme = np.max(nme_scores)
    
    # Log final metrics to TensorBoard
    writer.add_scalar('Test/Average_NME', avg_nme, 0)
    writer.add_scalar('Test/Std_NME', std_nme, 0)
    writer.add_scalar('Test/Min_NME', min_nme, 0)
    writer.add_scalar('Test/Max_NME', max_nme, 0)
    
    # Create and log NME histogram
    writer.add_histogram('Test/NME_Distribution', torch.tensor(nme_scores), 0)
    
    # Create NME distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(nme_scores, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(avg_nme, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_nme:.4f}')
    ax.set_xlabel('NME')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of NME Scores on Test Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add to TensorBoard
    writer.add_figure('Test/NME_Distribution_Plot', fig, 0)
    plt.close(fig)
    
    writer.close()
    
    results = {
        'average_nme': avg_nme,
        'std_nme': std_nme,
        'min_nme': min_nme,
        'max_nme': max_nme,
        'num_samples': num_samples,
        'nme_scores': nme_scores
    }
    
    print(f"\nEvaluation Results:")
    print(f"  Average NME: {avg_nme:.6f} ± {std_nme:.6f}")
    print(f"  Min NME: {min_nme:.6f}")
    print(f"  Max NME: {max_nme:.6f}")
    print(f"  Samples: {num_samples}")
    print(f"  TensorBoard logs: {eval_log_dir}")
    
    return results


if __name__ == "__main__":
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("WARNING: CUDA not detected. Training will be very slow on CPU.")
    
    try:
        print("=" * 80)
        print("WFLW HEATMAP NETWORK TRAINING WITH TENSORBOARD")
        print("=" * 80)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("=" * 80)
        print("To view training progress:")
        print("  tensorboard --logdir=./runs")
        print("  Then open: http://localhost:6006")
        print("=" * 80)
        
        # Start training
        train_heatmap_large()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("View results with: tensorboard --logdir=./runs")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n" + "!" * 80)
        print("TRAINING INTERRUPTED BY USER")
        print("!" * 80)
        
    except FileNotFoundError as e:
        print("\n" + "!" * 80)
        print("DATASET NOT FOUND")
        print("!" * 80)
        print(f"Error: {e}")
        print("Please check dataset paths:")
        print("  - /home/zimmermann/Projekty/SI_WFLW/data/WFLW/annotations")
        print("  - /home/zimmermann/Projekty/SI_WFLW/data/WFLW/images")
        
    except ImportError as e:
        print("\n" + "!" * 80)
        print("MISSING DEPENDENCIES")
        print("!" * 80)
        print(f"Error: {e}")
        print("Install required packages:")
        print("  pip install torch torchvision tensorboard matplotlib opencv-python")
        
    except RuntimeError as e:
        print("\n" + "!" * 80)
        print("RUNTIME ERROR")
        print("!" * 80)
        print(f"Error: {e}")
        
        if "CUDA" in str(e):
            print("CUDA-related error. Try:")
            print("  - Reduce batch size")
            print("  - Use smaller model (resnet18/34)")
            print("  - Check GPU memory")
        elif "memory" in str(e).lower():
            print("Memory error. Try:")
            print("  - Reduce batch size")
            print("  - Reduce num_workers")
            print("  - Close other applications")
        
    except Exception as e:
        print("\n" + "!" * 80)
        print("UNEXPECTED ERROR")
        print("!" * 80)
        print(f"Error: {e}")
        print("Check:")
        print("  - Dataset integrity")
        print("  - Available disk space")
        print("  - System resources")
        
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nCleanup completed.")#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WFLW Heatmap Network - Enhanced Version with Regularization
Professional implementation for facial landmark detection using heatmap regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import logging
from typing import Tuple, Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class WFLWHeatmapNetworkLarge(nn.Module):
    """
    Large-scale heatmap network for WFLW dataset with regularization.
    Optimized for RTX 4070 (12GB VRAM).
    
    Args:
        num_landmarks (int): Number of facial landmarks (default: 98)
        backbone (str): Backbone architecture ('resnet18', 'resnet34', 'resnet50')
        heatmap_size (int): Output heatmap resolution (default: 64)
    """
    
    def __init__(self, num_landmarks: int = 98, backbone: str = 'resnet50', heatmap_size: int = 64):
        super(WFLWHeatmapNetworkLarge, self).__init__()
        
        self.num_landmarks = num_landmarks
        self.heatmap_size = heatmap_size
        
        # Configure backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            backbone_out = 2048
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
            backbone_out = 512
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
            backbone_out = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classification layers (avgpool, fc)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Progressive upsampling with dropout regularization
        if backbone == 'resnet50':
            # ResNet50: 7x7x2048 -> 64x64x98
            self.deconv_layers = nn.Sequential(
                # 7x7 -> 14x14
                nn.ConvTranspose2d(backbone_out, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                
                # 14x14 -> 28x28
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                
                # 28x28 -> 56x56
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.05),
                
                # Refinement layer
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.05),
            )
        else:
            # ResNet18/34: 7x7x512 -> 64x64x98
            self.deconv_layers = nn.Sequential(
                # 7x7 -> 14x14
                nn.ConvTranspose2d(backbone_out, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                
                # 14x14 -> 28x28
                nn.ConvTranspose2d(256, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                
                # 28x28 -> 56x56
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.05),
            )
        
        # Final heatmap layer
        self.final_layer = nn.Conv2d(128, num_landmarks, 1, 1, 0)
        
        # Optional upsampling to target size
        if heatmap_size != 56:
            self.upsample = nn.Upsample(
                size=(heatmap_size, heatmap_size), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            self.upsample = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for deconvolution layers."""
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Final layer initialization
        nn.init.normal_(self.final_layer.weight, std=0.001)
        nn.init.constant_(self.final_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Heatmaps tensor of shape (B, num_landmarks, heatmap_size, heatmap_size)
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Deconvolution
        upsampled = self.deconv_layers(features)
        
        # Generate heatmaps
        heatmaps = self.final_layer(upsampled)
        
        # Upsample if needed
        if self.upsample is not None:
            heatmaps = self.upsample(heatmaps)
        
        return heatmaps


def generate_target_heatmaps_fixed(landmarks: torch.Tensor, 
                                 image_size: int = 224, 
                                 heatmap_size: int = 64, 
                                 sigma: float = 2.0) -> torch.Tensor:
    """
    Generate target heatmaps from landmark coordinates.
    
    Args:
        landmarks: Tensor of shape (B, N, 2) with normalized coordinates [-0.5, 0.5]
        image_size: Original image size
        heatmap_size: Target heatmap resolution
        sigma: Gaussian kernel standard deviation
        
    Returns:
        Target heatmaps tensor of shape (B, N, heatmap_size, heatmap_size)
    """
    batch_size, num_landmarks, _ = landmarks.shape
    device = landmarks.device
    
    # Denormalize landmarks from [-0.5, 0.5] to [0, heatmap_size]
    landmarks_heatmap = (landmarks + 0.5) * heatmap_size
    
    # Create target heatmaps on the same device
    target_heatmaps = torch.zeros(batch_size, num_landmarks, heatmap_size, heatmap_size, device=device)
    
    # Gaussian kernel
    size = int(6 * sigma + 1)
    if size % 2 == 0:
        size += 1  # Ensure odd size
    
    x = torch.arange(0, size, dtype=torch.float32, device=device)
    y = x.unsqueeze(1)
    x0, y0 = size // 2, size // 2
    gaussian = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    
    for b in range(batch_size):
        for l in range(num_landmarks):
            x_coord, y_coord = landmarks_heatmap[b, l]
            
            # Skip if landmark is out of bounds
            if x_coord < 0 or x_coord >= heatmap_size or y_coord < 0 or y_coord >= heatmap_size:
                continue
            
            x_coord, y_coord = int(round(x_coord.item())), int(round(y_coord.item()))
            
            # Place Gaussian
            x_min = max(0, x_coord - size // 2)
            x_max = min(heatmap_size, x_coord + size // 2 + 1)
            y_min = max(0, y_coord - size // 2)
            y_max = min(heatmap_size, y_coord + size // 2 + 1)
            
            g_x_min = max(0, size // 2 - x_coord)
            g_x_max = g_x_min + (x_max - x_min)
            g_y_min = max(0, size // 2 - y_coord)
            g_y_max = g_y_min + (y_max - y_min)
            
            if g_x_max > g_x_min and g_y_max > g_y_min:
                target_heatmaps[b, l, y_min:y_max, x_min:x_max] = gaussian[g_y_min:g_y_max, g_x_min:g_x_max]
    
    return target_heatmaps


def decode_heatmaps_fixed(heatmaps: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    """
    Decode heatmaps to landmark coordinates with sub-pixel accuracy.
    
    Args:
        heatmaps: Tensor of shape (B, N, H, W)
        image_size: Original image size for normalization
        
    Returns:
        Landmarks tensor of shape (B, N, 2) with normalized coordinates [-0.5, 0.5]
    """
    batch_size, num_landmarks, heatmap_height, heatmap_width = heatmaps.shape
    device = heatmaps.device
    
    # Apply softmax to get probability distributions
    heatmaps_prob = F.softmax(heatmaps.view(batch_size, num_landmarks, -1), dim=2)
    heatmaps_prob = heatmaps_prob.view(batch_size, num_landmarks, heatmap_height, heatmap_width)
    
    # Find max positions
    heatmaps_flat = heatmaps_prob.view(batch_size, num_landmarks, -1)
    max_vals, max_indices = torch.max(heatmaps_flat, dim=2)
    
    # Convert flat indices to 2D coordinates
    y_coords = (max_indices // heatmap_width).float()
    x_coords = (max_indices % heatmap_width).float()
    
    # Sub-pixel refinement using center of mass around peak
    landmarks = []
    for b in range(batch_size):
        batch_landmarks = []
        for l in range(num_landmarks):
            x_int, y_int = int(x_coords[b, l]), int(y_coords[b, l])
            
            # Extract 3x3 region around peak for sub-pixel refinement
            x_min, x_max = max(0, x_int-1), min(heatmap_width, x_int+2)
            y_min, y_max = max(0, y_int-1), min(heatmap_height, y_int+2)
            
            if x_max > x_min and y_max > y_min:
                region = heatmaps_prob[b, l, y_min:y_max, x_min:x_max]
                
                # Center of mass in region
                y_indices = torch.arange(y_min, y_max, dtype=torch.float32, device=device).unsqueeze(1)
                x_indices = torch.arange(x_min, x_max, dtype=torch.float32, device=device).unsqueeze(0)
                
                total_mass = region.sum()
                if total_mass > 0:
                    x_refined = (region * x_indices).sum() / total_mass
                    y_refined = (region * y_indices).sum() / total_mass
                else:
                    x_refined, y_refined = x_coords[b, l], y_coords[b, l]
            else:
                x_refined, y_refined = x_coords[b, l], y_coords[b, l]
            
            batch_landmarks.append([x_refined, y_refined])
        
        landmarks.append(batch_landmarks)
    
    landmarks = torch.tensor(landmarks, dtype=torch.float32, device=device)
    
    # Normalize to [-0.5, 0.5]
    landmarks[:, :, 0] /= heatmap_width   # x coordinates
    landmarks[:, :, 1] /= heatmap_height  # y coordinates
    landmarks -= 0.5
    
    return landmarks


def visualize_predictions(images: torch.Tensor, 
                        pred_landmarks: torch.Tensor, 
                        target_landmarks: torch.Tensor, 
                        epoch: int, 
                        save_dir: str = './results'):
    """
    Generate prediction visualizations.
    
    Args:
        images: Input images tensor
        pred_landmarks: Predicted landmarks
        target_landmarks: Ground truth landmarks
        epoch: Current epoch number
        save_dir: Directory to save visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    batch_size = min(4, images.size(0))
    
    fig, axes = plt.subplots(2, batch_size, figsize=(5*batch_size, 10))
    if batch_size == 1:
        axes = axes.reshape(2, 1)
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # Denormalize landmarks
    pred_denorm = (pred_landmarks + 0.5) * 224
    target_denorm = (target_landmarks + 0.5) * 224
    
    for i in range(batch_size):
        # Image with predictions
        axes[0, i].imshow(images_denorm[i].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title(f'Sample {i+1} - Predictions (Epoch {epoch})')
        
        # Plot predicted landmarks
        pred_points = pred_denorm[i].cpu().numpy()
        for j, (x, y) in enumerate(pred_points):
            color = 'red' if j < 33 else 'blue' if j < 68 else 'green' if j < 96 else 'yellow'
            axes[0, i].plot(x, y, 'o', color=color, markersize=1.5, alpha=0.8)
        axes[0, i].axis('off')
        
        # Image with ground truth
        axes[1, i].imshow(images_denorm[i].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title(f'Sample {i+1} - Ground Truth')
        
        # Plot target landmarks
        target_points = target_denorm[i].cpu().numpy()
        for j, (x, y) in enumerate(target_points):
            color = 'red' if j < 33 else 'blue' if j < 68 else 'green' if j < 96 else 'yellow'
            axes[1, i].plot(x, y, 'o', color=color, markersize=1.5, alpha=0.8)
        axes[1, i].axis('off')
        
        # Calculate and display NME for this sample
        distances = torch.norm(pred_denorm[i] - target_denorm[i], dim=1)
        nme = distances.mean() / 224
        axes[0, i].text(5, 20, f'NME: {nme:.4f}', color='white', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_dir / f'predictions_epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {save_dir}/predictions_epoch_{epoch:03d}.png")


def visualize_heatmaps(heatmaps: torch.Tensor, 
                      landmarks: torch.Tensor, 
                      epoch: int,
                      save_dir: str = './results',
                      max_landmarks: int = 12):
    """
    Visualize generated heatmaps with enhanced styling.
    
    Args:
        heatmaps: Generated heatmaps tensor
        landmarks: Corresponding landmarks
        epoch: Current epoch number
        save_dir: Directory to save visualizations
        max_landmarks: Maximum number of landmarks to visualize
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Take first sample from batch
    sample_heatmaps = heatmaps[0][:max_landmarks].cpu().numpy()
    sample_landmarks = landmarks[0][:max_landmarks].cpu().numpy()
    
    # Denormalize landmarks
    sample_landmarks = (sample_landmarks + 0.5) * heatmaps.shape[-1]
    
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()
    
    # Set style
    plt.style.use('default')
    fig.suptitle(f'Heatmap Visualizations - Epoch {epoch}', fontsize=16, fontweight='bold')
    
    for i in range(min(max_landmarks, len(axes))):
        heatmap = sample_heatmaps[i]
        landmark_x, landmark_y = sample_landmarks[i]
        
        # Plot heatmap with enhanced colormap
        im = axes[i].imshow(heatmap, cmap='plasma')


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float], 
                        val_nmes: List[float],
                        save_dir: str = './results'):
    """
    Plot training curves for loss and NME metrics.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        val_nmes: List of validation NME values
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # NME curve
    ax2.plot(epochs, val_nmes, 'g-', label='Validation NME', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('NME')
    ax2.set_title('Validation NME')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add best NME annotation
    best_nme_idx = np.argmin(val_nmes)
    best_nme = val_nmes[best_nme_idx]
    ax2.annotate(f'Best NME: {best_nme:.4f}', 
                xy=(best_nme_idx + 1, best_nme),
                xytext=(best_nme_idx + 1, best_nme + 0.01),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved: {save_dir}/training_curves.png")


class WFLWHeatmapTrainerMaxRegularization:
    """
    Trainer with maximum regularization against overfitting.
    
    Args:
        model: The neural network model
        config: Training configuration
        save_dir: Directory to save checkpoints
    """
    
    def __init__(self, model: nn.Module, config: Any, save_dir: str = './checkpoints'):
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer with conservative settings
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=3e-4,  # Lower learning rate
            weight_decay=5e-3,  # Higher weight decay
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler with patience
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=8,
            verbose=True,
            min_lr=1e-7
        )
        
        # Early stopping configuration
        self.early_stopping_patience = 20
        self.early_stopping_counter = 0
        self.early_stopping_min_delta = 0.0005
        
        # Gradient clipping
        self.max_grad_norm = 0.3
        
        # Tracking metrics
        self.best_nme = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_nmes = []
        
        # Visualization setup
        self.viz_dir = Path('./results')
        self.viz_dir.mkdir(exist_ok=True)
        
        print(f"Trainer Configuration:")
        print(f"  Learning Rate: {3e-4}")
        print(f"  Weight Decay: {5e-3}")
        print(f"  LR Patience: 8 epochs")
        print(f"  Early Stop Patience: 20 epochs")
        print(f"  Gradient Clipping: {self.max_grad_norm}")
        print(f"  Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Device: {self.device}")
    
    def train_epoch(self, train_loader) -> float:
        """
        Execute one training epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                images = batch['image'].to(self.device)
                landmarks = batch['landmarks'].to(self.device)
                
                # Generate target heatmaps
                target_heatmaps = generate_target_heatmaps_fixed(
                    landmarks, 
                    image_size=self.config.image_size[0],
                    heatmap_size=64,
                    sigma=2.0
                )
                
                # Forward pass
                pred_heatmaps = self.model(images)
                loss = self.criterion(pred_heatmaps, target_heatmaps)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             max_norm=self.max_grad_norm)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Progress logging
                if batch_idx % 100 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.6f}, LR: {current_lr:.6f}")
                    
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {str(e)}")
                continue
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, val_loader, epoch: int, visualize: bool = True) -> Tuple[float, float]:
        """
        Execute one validation epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            visualize: Whether to generate visualizations
            
        Returns:
            Tuple of (average validation loss, average NME)
        """
        self.model.eval()
        total_loss = 0.0
        total_nme = 0.0
        num_samples = 0
        
        # Collect data for visualization
        viz_images, viz_pred_landmarks, viz_target_landmarks = None, None, None
        viz_heatmaps = None
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    images = batch['image'].to(self.device)
                    landmarks = batch['landmarks'].to(self.device)
                    
                    # Generate target heatmaps
                    target_heatmaps = generate_target_heatmaps_fixed(
                        landmarks,
                        image_size=self.config.image_size[0],
                        heatmap_size=64,
                        sigma=2.0
                    )
                    
                    # Forward pass
                    pred_heatmaps = self.model(images)
                    loss = self.criterion(pred_heatmaps, target_heatmaps)
                    
                    # Decode predictions
                    pred_landmarks = decode_heatmaps_fixed(pred_heatmaps, self.config.image_size[0])
                    
                    # Calculate NME
                    nme = self.calculate_nme(pred_landmarks, landmarks)
                    
                    total_loss += loss.item()
                    total_nme += nme * images.size(0)
                    num_samples += images.size(0)
                    
                    # Collect first batch for visualization
                    if batch_idx == 0 and visualize:
                        viz_images = images[:4].cpu()
                        viz_pred_landmarks = pred_landmarks[:4].cpu()
                        viz_target_landmarks = landmarks[:4].cpu()
                        viz_heatmaps = pred_heatmaps[:1].cpu()
                        
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                    continue
        
        avg_loss = total_loss / len(val_loader)
        avg_nme = total_nme / num_samples
        
        # Generate visualizations
        if visualize and viz_images is not None:
            visualize_predictions(viz_images, viz_pred_landmarks, viz_target_landmarks, epoch, self.viz_dir)
            
            # Visualize heatmaps every 5 epochs
            if viz_heatmaps is not None and epoch % 5 == 0:
                visualize_heatmaps(viz_heatmaps, viz_target_landmarks[:1], epoch, self.viz_dir)
        
        return avg_loss, avg_nme
    
    def calculate_nme(self, predictions: torch.Tensor, targets: torch.Tensor, image_size: int = 224) -> float:
        """
        Calculate Normalized Mean Error (NME).
        
        Args:
            predictions: Predicted landmarks
            targets: Ground truth landmarks
            image_size: Image size for normalization
            
        Returns:
            NME value
        """
        # Denormalize
        pred_denorm = (predictions + 0.5) * image_size
        target_denorm = (targets + 0.5) * image_size
        
        # L2 distances
        distances = torch.norm(pred_denorm - target_denorm, dim=2)
        nme = distances.mean()
        
        return (nme / image_size).item()
    
    def check_early_stopping(self, val_nme: float) -> bool:
        """
        Check early stopping condition.
        
        Args:
            val_nme: Current validation NME
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_nme < self.best_nme - self.early_stopping_min_delta:
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement")
                return True
        return False
    
    def train(self, train_loader, val_loader, num_epochs: int = 50):
        """
        Main training loop with early stopping and learning rate scheduling.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
        """
        print(f"Starting training for maximum {num_epochs} epochs")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"Minimum delta for improvement: {self.early_stopping_min_delta}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            try:
                # Training phase
                train_loss = self.train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_nme = self.validate_epoch(val_loader, epoch, visualize=True)
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                # Store metrics
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.val_nmes.append(val_nme)
                
                # Calculate epoch time and current learning rate
                epoch_time = time.time() - epoch_start
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Print epoch summary
                print(f"\nEPOCH {epoch} SUMMARY:")
                print(f"  Training Loss:   {train_loss:.6f}")
                print(f"  Validation Loss: {val_loss:.6f}")
                print(f"  Validation NME:  {val_nme:.6f}")
                print(f"  Learning Rate:   {current_lr:.6f}")
                print(f"  Epoch Time:      {epoch_time:.1f}s")
                print(f"  Early Stop Counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                # Save best model
                if val_nme < self.best_nme:
                    self.best_nme = val_nme
                    self.early_stopping_counter = 0
                    print(f"  New best NME: {val_nme:.6f}")
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_loss,
                        'val_nme': val_nme,
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'val_nmes': self.val_nmes,
                        'config': self.config
                    }, self.save_dir / 'best_model_heatmap.pth')
                
                # Check early stopping
                if self.check_early_stopping(val_nme):
                    break
                
                # Save checkpoint every 10 epochs
                if epoch % 10 == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_loss,
                        'val_nme': val_nme,
                        'train_losses': self.train_losses,
                        'val_losses': self.val_losses,
                        'val_nmes': self.val_nmes
                    }, self.save_dir / f'checkpoint_epoch_{epoch}.pth')
                    print(f"  Checkpoint saved: epoch_{epoch}.pth")
                
                # Plot training curves every 10 epochs
                if epoch % 10 == 0:
                    plot_training_curves(self.train_losses, self.val_losses, self.val_nmes, self.viz_dir)
                
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {str(e)}")
                continue
        
        # Final training curves
        plot_training_curves(self.train_losses, self.val_losses, self.val_nmes, self.viz_dir)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.1f} hours")
        print(f"Best NME achieved: {self.best_nme:.6f}")
        print(f"Visualizations saved in: {self.viz_dir}")
        print(f"Model checkpoints saved in: {self.save_dir}")


def train_heatmap_enhanced():
    """
    Main training function with enhanced augmentations and maximum regularization.
    
    Raises:
        ImportError: If enhanced data loader configuration is not available
        RuntimeError: If CUDA is not available when expected
        FileNotFoundError: If dataset paths are not found
    """
    try:
        from data_loader import WFLWConfigEnhanced, WFLWDataModule
        
        print("WFLW Enhanced Training Started")
        print("Configuration: Strong augmentations + maximum regularization")
        print("Target: NME < 0.07")
        
        # Enhanced configuration with strong augmentations
        config = WFLWConfigEnhanced()
        config.batch_size = 48  # Smaller batch for increased stochasticity
        config.image_size = (224, 224)
        config.num_workers = 8
        
        # Data module with enhanced augmentations
        data_module = WFLWDataModule(
            annotations_path="/home/zimmermann/Projekty/SI_WFLW/data/WFLW/annotations",
            images_path="/home/zimmermann/Projekty/SI_WFLW/data/WFLW/images",
            config=config
        )
        
        data_module.setup()
        
        # Model with dropout in deconvolution layers
        model = WFLWHeatmapNetworkLarge(
            num_landmarks=98, 
            backbone='resnet50',
            heatmap_size=64
        )
        
        # Maximum regularization trainer
        trainer = WFLWHeatmapTrainerMaxRegularization(
            model, 
            config, 
            save_dir='./wflw_checkpoints_enhanced'
        )
        
        # Get data loaders
        train_loader = data_module.get_train_loader()
        val_loader = data_module.get_test_loader()
        
        # GPU memory test
        print("Testing GPU memory with enhanced augmentations...")
        test_batch = next(iter(train_loader))
        test_images = test_batch['image'][:4].to(trainer.device)
        
        with torch.no_grad():
            test_output = model(test_images)
            print(f"GPU test passed. Output shape: {test_output.shape}")
            print("Starting enhanced training...")
        
        # Start training
        trainer.train(train_loader, val_loader, num_epochs=50)
        
    except ImportError as e:
        logger.error(f"Enhanced configuration not available: {e}")
        raise ImportError(f"Failed to import enhanced data loader: {e}")
    except FileNotFoundError as e:
        logger.error(f"Dataset path not found: {e}")
        raise FileNotFoundError(f"Dataset files not accessible: {e}")
    except RuntimeError as e:
        logger.error(f"Runtime error during training: {e}")
        raise RuntimeError(f"Training failed: {e}")


def train_heatmap_fallback():
    """
    Fallback training function using basic augmentations.
    
    Raises:
        ImportError: If basic data loader configuration is not available
        RuntimeError: If CUDA is not available when expected
    """
    try:
        from data_loader import WFLWConfig, WFLWDataModule
        
        print("WARNING: Using fallback training with basic augmentations")
        print("Recommendation: Update data_loader.py with enhanced augmentations for optimal results")
        
        # Basic configuration
        config = WFLWConfig()
        config.batch_size = 58
        config.image_size = (224, 224)
        config.num_workers = 12
        
        # Data module
        data_module = WFLWDataModule(
            annotations_path="/home/zimmermann/Projekty/SI_WFLW/data/WFLW/annotations",
            images_path="/home/zimmermann/Projekty/SI_WFLW/data/WFLW/images",
            config=config
        )
        
        data_module.setup()
        
        # Model
        model = WFLWHeatmapNetworkLarge(
            num_landmarks=98, 
            backbone='resnet50',
            heatmap_size=64
        )
        
        # Enhanced trainer with basic augmentations
        trainer = WFLWHeatmapTrainerMaxRegularization(
            model, 
            config, 
            save_dir='./wflw_checkpoints_fallback'
        )
        
        # Get data loaders
        train_loader = data_module.get_train_loader()
        val_loader = data_module.get_test_loader()
        
        # Start training
        trainer.train(train_loader, val_loader, num_epochs=50)
        
    except ImportError as e:
        logger.error(f"Basic configuration not available: {e}")
        raise ImportError(f"Failed to import basic data loader: {e}")
    except RuntimeError as e:
        logger.error(f"Runtime error during fallback training: {e}")
        raise RuntimeError(f"Fallback training failed: {e}")


def train_heatmap_large():
    """
    Main training function that attempts enhanced training first, then falls back to basic.
    
    This function provides a robust training pipeline that gracefully handles missing
    enhanced configurations by falling back to basic training.
    """
    try:
        print("Attempting enhanced training configuration...")
        train_heatmap_enhanced()
    except ImportError as e:
        print(f"Enhanced configuration unavailable: {e}")
        print("Falling back to basic training configuration...")
        try:
            train_heatmap_fallback()
        except Exception as fallback_error:
            logger.error(f"Both enhanced and fallback training failed: {fallback_error}")
            raise RuntimeError(f"All training configurations failed. Last error: {fallback_error}")
    except Exception as e:
        print(f"Enhanced training failed: {e}")
        print("Attempting fallback training...")
        try:
            train_heatmap_fallback()
        except Exception as fallback_error:
            logger.error(f"Both enhanced and fallback training failed: {fallback_error}")
            raise RuntimeError(f"All training configurations failed. Last error: {fallback_error}")


def load_model_checkpoint(checkpoint_path: str, model: nn.Module, device: str = 'cuda') -> Dict[str, Any]:
    """
    Load model from checkpoint with error handling.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        device: Device to load model on
        
    Returns:
        Dictionary containing checkpoint information
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If checkpoint loading fails
    """
    try:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Best NME: {checkpoint.get('val_nme', 'Unknown')}")
        
        return checkpoint
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise RuntimeError(f"Checkpoint loading failed: {e}")


def evaluate_model(model: nn.Module, test_loader, device: str = 'cuda', save_dir: str = './evaluation'):
    """
    Evaluate trained model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device for evaluation
        save_dir: Directory to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    model.eval()
    total_nme = 0.0
    num_samples = 0
    nme_scores = []
    
    print("Evaluating model on test set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                images = batch['image'].to(device)
                landmarks = batch['landmarks'].to(device)
                
                # Forward pass
                pred_heatmaps = model(images)
                pred_landmarks = decode_heatmaps_fixed(pred_heatmaps, 224)
                
                # Calculate NME for each sample
                for i in range(images.size(0)):
                    pred_denorm = (pred_landmarks[i] + 0.5) * 224
                    target_denorm = (landmarks[i] + 0.5) * 224
                    
                    distances = torch.norm(pred_denorm - target_denorm, dim=1)
                    nme = (distances.mean() / 224).item()
                    nme_scores.append(nme)
                    total_nme += nme
                    num_samples += 1
                
                if batch_idx % 50 == 0:
                    print(f"Processed {batch_idx}/{len(test_loader)} batches")
                    
            except Exception as e:
                logger.error(f"Error in evaluation batch {batch_idx}: {e}")
                continue
    
    # Calculate statistics
    avg_nme = total_nme / num_samples
    std_nme = np.std(nme_scores)
    min_nme = np.min(nme_scores)
    max_nme = np.max(nme_scores)
    
    # Plot NME distribution
    plt.figure(figsize=(10, 6))
    plt.hist(nme_scores, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(avg_nme, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_nme:.4f}')
    plt.xlabel('NME')
    plt.ylabel('Frequency')
    plt.title('Distribution of NME Scores on Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_dir / 'nme_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    results = {
        'average_nme': avg_nme,
        'std_nme': std_nme,
        'min_nme': min_nme,
        'max_nme': max_nme,
        'num_samples': num_samples,
        'nme_scores': nme_scores
    }
    
    print(f"\nEvaluation Results:")
    print(f"  Average NME: {avg_nme:.6f} ± {std_nme:.6f}")
    print(f"  Min NME: {min_nme:.6f}")
    print(f"  Max NME: {max_nme:.6f}")
    print(f"  Samples: {num_samples}")
    print(f"  Results saved in: {save_dir}")
    
    return results


if __name__ == "__main__":
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will run on CPU (very slow).")
        print("WARNING: CUDA not detected. Consider using GPU for faster training.")
    
    try:
        print("=" * 80)
        print("WFLW HEATMAP NETWORK TRAINING")
        print("=" * 80)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("=" * 80)
        
        # Start training
        train_heatmap_large()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n" + "!" * 80)
        print("TRAINING INTERRUPTED BY USER")
        print("!" * 80)
        logger.info("Training was manually interrupted")
        
    except FileNotFoundError as e:
        print("\n" + "!" * 80)
        print("DATASET NOT FOUND")
        print("!" * 80)
        logger.error(f"Dataset files not found: {e}")
        print("Please check that the dataset paths are correct:")
        print("  - /home/zimmermann/Projekty/SI_WFLW/data/WFLW/annotations")
        print("  - /home/zimmermann/Projekty/SI_WFLW/data/WFLW/images")
        
    except ImportError as e:
        print("\n" + "!" * 80)
        print("MISSING DEPENDENCIES")
        print("!" * 80)
        logger.error(f"Import error: {e}")
        print("Please ensure all required modules are installed:")
        print("  - torch, torchvision")
        print("  - matplotlib, seaborn")
        print("  - numpy, opencv-python")
        print("  - pathlib (standard library)")
        
    except RuntimeError as e:
        print("\n" + "!" * 80)
        print("RUNTIME ERROR")
        print("!" * 80)
        logger.error(f"Runtime error during training: {e}")
        
        if "CUDA" in str(e):
            print("CUDA-related error detected.")
            print("Possible solutions:")
            print("  - Reduce batch size")
            print("  - Use smaller model (resnet18/34)")
            print("  - Check GPU memory availability")
        elif "memory" in str(e).lower():
            print("Memory error detected.")
            print("Possible solutions:")
            print("  - Reduce batch size")
            print("  - Reduce number of workers")
            print("  - Close other GPU applications")
        
        raise
        
    except Exception as e:
        print("\n" + "!" * 80)
        print("UNEXPECTED ERROR")
        print("!" * 80)
        logger.error(f"Unexpected error during training: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        
        print("An unexpected error occurred. Check the log file 'training.log' for details.")
        print("If the problem persists, please check:")
        print("  - Dataset integrity")
        print("  - Available disk space")
        print("  - System resources (RAM, GPU memory)")
        
        raise
        
    finally:
        # Cleanup operations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nCleanup completed.")


# Test compatibility with data loader
def test_data_loader_compatibility():
    """Test if trainer works with data loader"""
    try:
        # Try importing basic config first
        try:
            from data_loader import WFLWConfig, WFLWDataModule
            print("Basic data loader imported successfully")
        except ImportError:
            print("Basic data loader not available")
            
        # Try importing enhanced config
        try:
            from data_loader import WFLWConfigEnhanced, WFLWDataModule
            print("Enhanced data loader imported successfully")
        except ImportError:
            print("Enhanced data loader not available")
            
        return True
        
    except Exception as e:
        print(f"Data loader compatibility test failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing data loader compatibility first...")
    if test_data_loader_compatibility():
        print("Data loader compatible, proceeding with training...")
    else:
        print("Data loader compatibility issues detected!")
        print("Please ensure data_loader.py is in the same directory.")
    
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("WARNING: CUDA not detected. Training will be very slow on CPU.")
    
    try:
        print("=" * 80)
        print("WFLW HEATMAP NETWORK TRAINING WITH TENSORBOARD")
        print("=" * 80)
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name()}")
            print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("=" * 80)
        print("To view training progress:")
        print("  tensorboard --logdir=./runs")
        print("  Then open: http://localhost:6006")
        print("=" * 80)
        
        # Start training
        train_heatmap_large()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("View results with: tensorboard --logdir=./runs")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n" + "!" * 80)
        print("TRAINING INTERRUPTED BY USER")
        print("!" * 80)
        
    except FileNotFoundError as e:
        print("\n" + "!" * 80)
        print("DATASET NOT FOUND")
        print("!" * 80)
        print(f"Error: {e}")
        print("Please check dataset paths:")
        print("  - /home/zimmermann/Projekty/SI_WFLW/data/WFLW/annotations")
        print("  - /home/zimmermann/Projekty/SI_WFLW/data/WFLW/images")
        
    except ImportError as e:
        print("\n" + "!" * 80)
        print("MISSING DEPENDENCIES")
        print("!" * 80)
        print(f"Error: {e}")
        print("Install required packages:")
        print("  pip install torch torchvision tensorboard matplotlib opencv-python albumentations")
        
    except RuntimeError as e:
        print("\n" + "!" * 80)
        print("RUNTIME ERROR")
        print("!" * 80)
        print(f"Error: {e}")
        
        if "CUDA" in str(e):
            print("CUDA-related error. Try:")
            print("  - Reduce batch size")
            print("  - Use smaller model (resnet18/34)")
            print("  - Check GPU memory")
        elif "memory" in str(e).lower():
            print("Memory error. Try:")
            print("  - Reduce batch size")
            print("  - Reduce num_workers")
            print("  - Close other applications")
        
    except Exception as e:
        print("\n" + "!" * 80)
        print("UNEXPECTED ERROR")
        print("!" * 80)
        print(f"Error: {e}")
        print("Check:")
        print("  - Dataset integrity")
        print("  - Available disk space")
        print("  - System resources")
        
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nCleanup completed.")
        print("\nTo analyze results:")
        print("  tensorboard --logdir=./runs")
        print("  Open: http://localhost:6006")
        print("\nTraining logs saved in:")
        print("  - ./runs/ (TensorBoard logs)")
        print("  - ./wflw_checkpoints_*/ (Model checkpoints)")
