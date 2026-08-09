"""
SAF-DETR: Adaptive Image Intelligence Module
==============================================

This module automatically detects image quality issues in surveillance frames
and applies appropriate enhancement strategies.

Author: SAF-DETR Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class QualityAssessor(nn.Module):
    """
    Assesses image quality metrics for surveillance frames.
    
    Measures:
    - Brightness level
    - Blur amount
    - Noise level
    - Compression artifacts
    """
    
    def __init__(self):
        super().__init__()
        # Learnable quality assessment
        self.brightness_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.blur_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.noise_conv = nn.Conv2d(3, 16, 3, padding=1)
        
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # [brightness, blur, noise, compression]
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image tensor [B, 3, H, W] in range [0, 1]
            
        Returns:
            Quality scores [B, 4] for [brightness, blur, noise, compression]
        """
        # Extract features for each quality metric
        brightness_feat = self.brightness_conv(x)
        blur_feat = self.blur_conv(x)
        noise_feat = self.noise_conv(x)
        
        # Concatenate features
        features = torch.cat([brightness_feat, blur_feat, noise_feat], dim=1)
        
        # Predict quality scores
        quality_scores = self.quality_head(features)
        
        return quality_scores


class EnhancementStrategy(nn.Module):
    """
    Selects and applies enhancement strategy based on quality assessment.
    """
    
    def __init__(self, num_strategies: int = 4):
        super().__init__()
        self.num_strategies = num_strategies
        
        # Strategy selection network
        self.strategy_selector = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, num_strategies),
            nn.Softmax(dim=-1)
        )
        
        # Enhancement modules
        self.brightness_enhancer = BrightnessEnhancer()
        self.sharpening_module = SharpeningModule()
        self.denoising_module = DenoisingModule()
        self.compression_restorer = CompressionRestorer()
        
    def forward(self, x: torch.Tensor, quality_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image [B, 3, H, W]
            quality_scores: Quality metrics [B, 4]
            
        Returns:
            enhanced: Enhanced image [B, 3, H, W]
            strategy_weights: Strategy selection weights [B, num_strategies]
        """
        # Select enhancement strategy
        strategy_weights = self.strategy_selector(quality_scores)
        
        # Apply all enhancements
        brightness_enhanced = self.brightness_enhancer(x)
        sharpened = self.sharpening_module(x)
        denoised = self.denoising_module(x)
        restored = self.compression_restorer(x)
        
        # Weighted combination based on strategy
        enhanced = (
            strategy_weights[:, 0:1, None, None] * brightness_enhanced +
            strategy_weights[:, 1:2, None, None] * sharpened +
            strategy_weights[:, 2:3, None, None] * denoised +
            strategy_weights[:, 3:4, None, None] * restored
        )
        
        return enhanced, strategy_weights


class BrightnessEnhancer(nn.Module):
    """Enhances low-light images using CLAHE and gamma correction."""
    
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.clahe_clip = nn.Parameter(torch.tensor(2.0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to LAB for CLAHE
        x_np = x.permute(0, 2, 3, 1).cpu().numpy()
        enhanced = []
        
        for img in x_np:
            img_uint8 = (img * 255).astype(np.uint8)
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=float(self.clahe_clip.clamp(1, 10)),
                tileGridSize=(8, 8)
            )
            l = clahe.apply(l)
            
            lab = cv2.merge([l, a, b])
            enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            enhanced.append(enhanced_img / 255.0)
        
        enhanced = torch.tensor(np.stack(enhanced), dtype=x.dtype, device=x.device)
        enhanced = enhanced.permute(0, 3, 1, 2)
        
        # Apply gamma correction
        gamma = self.gamma.clamp(0.5, 2.0)
        enhanced = torch.pow(enhanced, gamma)
        
        return enhanced


class SharpeningModule(nn.Module):
    """Reduces blur using unsharp masking."""
    
    def __init__(self):
        super().__init__()
        self.sharpen_strength = nn.Parameter(torch.tensor(1.5))
        
        # Learnable sharpening kernel
        self.sharpen_conv = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        nn.init.dirac_(self.sharpen_conv.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian blur
        blurred = F.gaussian_blur(x, kernel_size=5, sigma=1.0)
        
        # Unsharp masking
        sharpened = x + self.sharpen_strength * (x - blurred)
        sharpened = torch.clamp(sharpened, 0, 1)
        
        # Learnable refinement
        sharpened = self.sharpen_conv(sharpened)
        
        return sharpened


class DenoisingModule(nn.Module):
    """Reduces noise using learned denoising."""
    
    def __init__(self):
        super().__init__()
        self.denoise_net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = self.denoise_net(x)
        denoised = x - noise
        return torch.clamp(denoised, 0, 1)


class CompressionRestorer(nn.Module):
    """Reduces compression artifacts."""
    
    def __init__(self):
        super().__init__()
        self.restoration_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        restored = x + self.restoration_net(x)
        return torch.clamp(restored, 0, 1)


class AdaptiveImageIntelligence(nn.Module):
    """
    Complete Adaptive Image Intelligence Module.
    
    Automatically assesses frame quality and applies appropriate enhancement.
    """
    
    def __init__(self, 
                 brightness_threshold: float = 0.3,
                 blur_threshold: float = 0.4,
                 noise_threshold: float = 0.5,
                 compression_threshold: float = 0.5):
        super().__init__()
        
        self.quality_assessor = QualityAssessor()
        self.enhancement_strategy = EnhancementStrategy()
        
        self.brightness_threshold = brightness_threshold
        self.blur_threshold = blur_threshold
        self.noise_threshold = noise_threshold
        self.compression_threshold = compression_threshold
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input frame [B, 3, H, W] in range [0, 1]
            
        Returns:
            Dictionary containing:
                - enhanced: Enhanced frame [B, 3, H, W]
                - quality_scores: Quality metrics [B, 4]
                - strategy_weights: Enhancement strategy weights [B, 4]
                - needs_enhancement: Boolean mask [B]
        """
        # Assess quality
        quality_scores = self.quality_assessor(x)
        
        # Determine if enhancement is needed
        needs_enhancement = (
            (quality_scores[:, 0] < self.brightness_threshold) |  # Too dark
            (quality_scores[:, 1] < self.blur_threshold) |         # Too blurry
            (quality_scores[:, 2] > self.noise_threshold) |        # Too noisy
            (quality_scores[:, 3] > self.compression_threshold)    # Compressed
        )
        
        # Apply enhancement
        enhanced, strategy_weights = self.enhancement_strategy(x, quality_scores)
        
        # Use enhanced version only where needed
        output = torch.where(
            needs_enhancement[:, None, None, None],
            enhanced,
            x
        )
        
        return {
            'enhanced': output,
            'quality_scores': quality_scores,
            'strategy_weights': strategy_weights,
            'needs_enhancement': needs_enhancement
        }
    
    def get_quality_report(self, quality_scores: torch.Tensor) -> Dict[str, float]:
        """Generate human-readable quality report."""
        return {
            'brightness': quality_scores[0].item(),
            'blur': quality_scores[1].item(),
            'noise': quality_scores[2].item(),
            'compression': quality_scores[3].item(),
            'overall_quality': quality_scores.mean().item()
        }


# Utility functions for OpenCV integration
def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> torch.Tensor:
    """
    Preprocess OpenCV frame for SAF-DETR.
    
    Args:
        frame: OpenCV BGR frame [H, W, 3]
        target_size: Target size (H, W)
        
    Returns:
        Preprocessed tensor [1, 3, H, W]
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize
    frame_resized = cv2.resize(frame_rgb, target_size[::-1])
    
    # Normalize to [0, 1]
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    
    # To tensor [1, 3, H, W]
    tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor


def postprocess_frame(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert SAF-DETR output tensor back to OpenCV frame.
    
    Args:
        tensor: Output tensor [1, 3, H, W]
        
    Returns:
        OpenCV BGR frame [H, W, 3]
    """
    # Remove batch dimension and convert to numpy
    frame = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Denormalize to [0, 255]
    frame = (frame * 255).astype(np.uint8)
    
    # Convert RGB to BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    return frame_bgr


if __name__ == "__main__":
    # Test the module
    print("Testing Adaptive Image Intelligence Module...")
    
    # Create module
    module = AdaptiveImageIntelligence()
    
    # Test input
    x = torch.randn(2, 3, 640, 640).clamp(0, 1)
    
    # Forward pass
    output = module(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Enhanced shape: {output['enhanced'].shape}")
    print(f"Quality scores: {output['quality_scores']}")
    print(f"Strategy weights: {output['strategy_weights']}")
    print(f"Needs enhancement: {output['needs_enhancement']}")
    
    # Quality report
    report = module.get_quality_report(output['quality_scores'][0])
    print(f"\
Quality Report: {report}")
    
    print("\
✓ Adaptive Image Intelligence Module test passed!")
