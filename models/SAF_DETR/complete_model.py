"""
SAF-DETR: Complete Integrated Model
====================================

Full SAF-DETR model integrating all custom modules with RT-DETR backbone.
Optimized for RTX 4070 with mixed precision training.

Author: SAF-DETR Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import timm
from pathlib import Path

# Import custom modules
from .adaptive_intelligence import AdaptiveImageIntelligence
from .feature_enhancement import SurveillanceFeatureEnhancement, FeaturePyramidEnhancement
from .temporal_memory import TemporalDetectionMemory
from .loss_functions import BehaviourAwareLoss
from .novel_pipeline import NovelSAFDETRPipeline, AdvancedAugmentation


class RTDETRBackbone(nn.Module):
    """
    Modified ResNet-50 backbone from RT-DETR.
    Extracts multi-scale features C3, C4, C5.
    """
    
    def __init__(self, pretrained: bool = False):
        super().__init__()
        # Use timm for ResNet-50 with features (safe fallback to pretrained=False)
        try:
            self.backbone = timm.create_model(
                'resnet50',
                pretrained=pretrained,
                features_only=True,
                out_indices=[2, 3, 4]  # C3, C4, C5
            )
        except Exception:
            self.backbone = timm.create_model(
                'resnet50',
                pretrained=False,
                features_only=True,
                out_indices=[2, 3, 4]
            )
        
        # Channel dimensions for C3, C4, C5
        self.channels = [512, 1024, 2048]
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            List of features [C3, C4, C5]
        """
        features = self.backbone(x)
        return features


class HybridEncoder(nn.Module):
    """
    RT-DETR Hybrid Encoder with AIFI and CCFF.
    Modified for surveillance optimization.
    """
    
    def __init__(self, 
                 in_channels: List[int] = [512, 1024, 2048],
                 hidden_dim: int = 256,
                 num_encoder_layers: int = 1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input projection layers
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim)
            ) for ch in in_channels
        ])
        
        # AIFI (Attention-based Intra-scale Feature Interaction)
        self.aifi = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        
        # CCFF (CNN-based Cross-scale Feature Fusion)
        self.ccff = CrossScaleFeatureFusion(hidden_dim)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of features [C3, C4, C5]
            
        Returns:
            Encoded features [B, hidden_dim, H, W]
        """
        # Project to common dimension
        proj_features = []
        for feat, proj in zip(features, self.input_proj):
            proj_features.append(proj(feat))
        
        # Get C5 for AIFI
        c5 = proj_features[-1]  # [B, hidden_dim, H/32, W/32]
        b, c, h, w = c5.shape
        
        # Flatten for transformer
        c5_flat = c5.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Apply AIFI
        aifi_out = self.aifi(c5_flat)
        aifi_out = aifi_out.permute(0, 2, 1).reshape(b, c, h, w)
        
        # CCFF: Cross-scale fusion
        fused = self.ccff(proj_features[0], proj_features[1], aifi_out)
        
        return fused


class CrossScaleFeatureFusion(nn.Module):
    """CCFF module for cross-scale feature fusion."""
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        
        # Upsampling convolutions
        self.upsample_c4 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        self.upsample_c5 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> torch.Tensor:
        """Fuse multi-scale features."""
        # Upsample C4 and C5 to C3 size
        c4_up = F.interpolate(c4, size=c3.shape[2:], mode='bilinear', align_corners=False)
        c4_up = self.upsample_c4(c4_up)
        
        c5_up = F.interpolate(c5, size=c3.shape[2:], mode='bilinear', align_corners=False)
        c5_up = self.upsample_c5(c5_up)
        
        # Concatenate and fuse
        fused = torch.cat([c3, c4_up, c5_up], dim=1)
        output = self.fusion(fused)
        
        return output


class HumanCentricDecoder(nn.Module):
    """
    Modified RT-DETR decoder optimized for human detection.
    """
    
    def __init__(self,
                 hidden_dim: int = 256,
                 num_queries: int = 300,
                 num_classes: int = 1,  # Person only
                 num_decoder_layers: int = 3):
        super().__init__()
        
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Learnable queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Prediction heads
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for background
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # x, y, w, h
        )
        
        # Feature projection for temporal memory
        self.feature_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, encoder_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_features: [B, hidden_dim, H, W]
            
        Returns:
            Dictionary with class logits, boxes, and features
        """
        b, c, h, w = encoder_features.shape
        
        # Flatten encoder features
        encoder_flat = encoder_features.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        # Prepare queries
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # [B, num_queries, C]
        
        # Decode
        decoder_output = self.decoder(queries, encoder_flat)  # [B, num_queries, C]
        
        # Predictions
        class_logits = self.class_head(decoder_output)  # [B, num_queries, num_classes+1]
        bbox_pred = self.bbox_head(decoder_output).sigmoid()  # [B, num_queries, 4]
        
        # Features for tracking
        features = self.feature_proj(decoder_output)  # [B, num_queries, C]
        
        return {
            'class_logits': class_logits,
            'boxes': bbox_pred,
            'features': features,
            'decoder_output': decoder_output
        }


class CompleteSAFDETR(nn.Module):
    """
    Complete SAF-DETR model with all modules integrated.
    """
    
    def __init__(self,
                 num_classes: int = 1,
                 hidden_dim: int = 256,
                 num_queries: int = 300,
                 use_adaptive_intelligence: bool = True,
                 use_feature_enhancement: bool = True,
                 use_temporal_memory: bool = True,
                 use_novel_pipeline: bool = True,
                 pretrained_backbone: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_temporal_memory = use_temporal_memory
        
        # 1. Adaptive Image Intelligence
        self.adaptive_intelligence = AdaptiveImageIntelligence() if use_adaptive_intelligence else None
        
        # 2. Backbone
        self.backbone = RTDETRBackbone(pretrained=pretrained_backbone)
        
        # 3. Feature Enhancement
        self.feature_enhancement = SurveillanceFeatureEnhancement(hidden_dim) if use_feature_enhancement else None
        
        # 4. Hybrid Encoder
        self.encoder = HybridEncoder(
            in_channels=self.backbone.channels,
            hidden_dim=hidden_dim
        )
        
        # 5. Human-Centric Decoder
        self.decoder = HumanCentricDecoder(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_classes=num_classes
        )
        
        # 6. Temporal Memory
        self.temporal_memory = TemporalDetectionMemory(hidden_dim) if use_temporal_memory else None
        
        # 7. Novel Pipeline
        self.novel_pipeline = NovelSAFDETRPipeline(
            feature_dim=hidden_dim,
            use_uncertainty=True,
            use_progressive=True,
            use_adaptive_fusion=True,
            use_tta=True,
            use_nas=True
        ) if use_novel_pipeline else None
        
        # Loss function
        self.criterion = BehaviourAwareLoss()
        
        # Augmentation
        self.augmentation = AdvancedAugmentation()
        
    def forward(self, 
                images: torch.Tensor,
                prev_memory: Optional[Dict] = None,
                targets: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images [B, 3, H, W]
            prev_memory: Previous temporal memory state
            targets: Optional targets for training
            
        Returns:
            Dictionary with predictions and losses
        """
        outputs = {}
        
        # Step 1: Adaptive Image Intelligence
        if self.adaptive_intelligence is not None:
            ai_out = self.adaptive_intelligence(images)
            images = ai_out['enhanced']
            outputs['quality_scores'] = ai_out['quality_scores']
            outputs['needs_enhancement'] = ai_out['needs_enhancement']
        
        # Step 2: Backbone
        backbone_features = self.backbone(images)  # [C3, C4, C5]
        outputs['backbone_features'] = backbone_features
        
        # Step 3: Hybrid Encoder
        encoder_features = self.encoder(backbone_features)  # [B, hidden_dim, H, W]
        
        # Step 4: Feature Enhancement
        if self.feature_enhancement is not None:
            enhanced_out = self.feature_enhancement(encoder_features)
            encoder_features = enhanced_out['enhanced']
            outputs['human_mask'] = enhanced_out['human_mask']
            outputs['keypoints'] = enhanced_out['keypoints']
            outputs['density'] = enhanced_out['density']
            outputs['zones'] = enhanced_out['zones']
        
        # Step 5: Decoder
        decoder_out = self.decoder(encoder_features)
        
        # Step 6: Novel Pipeline
        if self.novel_pipeline is not None:
            # Prepare temporal features if available
            temporal_features = prev_memory.get('features') if prev_memory else None
            
            # Get boxes in correct format [N, 4]
            batch_size = decoder_out['boxes'].size(0)
            boxes_flat = decoder_out['boxes'].view(-1, 4)
            features_flat = decoder_out['features'].view(-1, self.hidden_dim)
            
            # Apply novel pipeline
            novel_out = self.novel_pipeline(
                spatial_features=features_flat,
                temporal_features=temporal_features,
                initial_boxes=boxes_flat
            )
            
            # Reshape back
            refined_boxes = novel_out.get('final_boxes', boxes_flat).view(batch_size, -1, 4)
            decoder_out['boxes'] = refined_boxes
            
            outputs.update(novel_out)
        
        # Step 7: Temporal Memory
        if self.temporal_memory is not None and self.training:
            # Flatten for temporal processing
            batch_size = decoder_out['boxes'].size(0)
            num_queries = decoder_out['boxes'].size(1)
            
            boxes_flat = decoder_out['boxes'].view(-1, 4)
            features_flat = decoder_out['features'].view(-1, self.hidden_dim)
            scores_flat = decoder_out['class_logits'][:, :, 0].view(-1)  # Person class score
            
            temporal_out = self.temporal_memory(boxes_flat, features_flat, scores_flat)
            
            # Reshape back
            outputs['stabilized_boxes'] = temporal_out['stabilized_detections'].view(batch_size, num_queries, 4)
            outputs['track_ids'] = temporal_out['track_ids']
            outputs['temporal_consistency'] = temporal_out['temporal_consistency']
            outputs['active_tracks'] = temporal_out['active_tracks']
        
        # Final outputs
        outputs['pred_logits'] = decoder_out['class_logits']
        outputs['pred_boxes'] = decoder_out['boxes']
        outputs['pred_features'] = decoder_out['features']
        
        # Compute loss if targets provided
        if targets is not None and self.training:
            loss_dict = self.compute_loss(outputs, targets)
            outputs.update(loss_dict)
        
        return outputs
    
    def compute_loss(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        predictions = {
            'boxes': outputs['pred_boxes'].view(-1, 4),
            'logits': outputs['pred_logits'].view(-1, self.num_classes + 1),
            'features': outputs['pred_features'].view(-1, self.hidden_dim),
            'scores': outputs['pred_logits'][:, :, 0].view(-1)
        }
        
        target_dict = {
            'boxes': targets['boxes'],
            'labels': targets['labels']
        }
        
        total_loss, loss_dict = self.criterion(predictions, target_dict)
        
        return {'loss': total_loss, **loss_dict}
    
    def get_memory_state(self) -> Optional[Dict]:
        """Get current temporal memory state."""
        if self.temporal_memory is not None:
            return {
                'features': self.temporal_memory.feature_history[-1] 
                if self.temporal_memory.feature_history else None
            }
        return None
    
    def reset_memory(self):
        """Reset temporal memory."""
        if self.temporal_memory is not None:
            self.temporal_memory.reset()
    
    @torch.jit.ignore
    def no_weight_decay(self) -> List[str]:
        """Parameters that should not have weight decay."""
        return ['query_embed.weight']


def build_saf_detr(pretrained: bool = True, **kwargs) -> CompleteSAFDETR:
    """Build SAF-DETR model."""
    model = CompleteSAFDETR(pretrained_backbone=pretrained, **kwargs)
    return model


if __name__ == "__main__":
    # Test complete model
    print("Testing Complete SAF-DETR Model...")
    
    # Create model
    model = build_saf_detr(
        num_classes=1,
        hidden_dim=256,
        num_queries=100,  # Reduced for testing
        use_adaptive_intelligence=True,
        use_feature_enhancement=True,
        use_temporal_memory=True,
        use_novel_pipeline=True
    )
    
    # Test input
    batch_size = 2
    images = torch.randn(batch_size, 3, 640, 640)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(images)
    
    print("\
Model Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            print(f"  {key}: List of {len(value)} tensors, first shape: {value[0].shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\
Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\
✓ Complete SAF-DETR Model test passed!")
