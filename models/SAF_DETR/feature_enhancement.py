"""
SAF-DETR: Surveillance Feature Enhancement Network (SFEN)
==========================================================

Multi-scale feature enhancement tailored for surveillance scenes:
- Human-priority attention
- Crowd region attention
- Pose-prior attention
- Density estimation
- Behaviour-zone attention

Author: SAF-DETR Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class ChannelSpatialAttention(nn.Module):
    """Combined channel + spatial attention block."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 1), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 1), channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = self.channel_att(x)
        x = x * ca
        sa = self.spatial_att(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1))
        return x * sa


class HumanPriorityAttention(nn.Module):
    """Attention map that prioritises human regions in the feature map."""

    def __init__(self, channels: int):
        super().__init__()
        self.AGate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        target = features_list[0]
        avg_f = torch.stack([F.adaptive_avg_pool2d(f, target.shape[2:]) for f in features_list]).mean(0)
        max_f = torch.stack([F.adaptive_max_pool2d(f, target.shape[2:]) for f in features_list]).mean(0)
        x = self.AGate(torch.cat([avg_f, max_f], dim=1))
        return torch.sigmoid(self.head(x))


class PosePriorAttention(nn.Module):
    """Lightweight pose-prior pseudo-attention map used as auxiliary supervision."""

    def __init__(self, channels: int, num_keypoints: int = 17):
        super().__init__()
        self.keypoint_head = nn.Conv2d(channels, num_keypoints, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        keypoints = torch.sigmoid(self.keypoint_head(x))
        attention = keypoints.max(dim=1, keepdim=True)[0]
        refined = x * (1.0 + self.alpha) + self.beta * attention * x
        return refined, keypoints


class CrowdDensityHead(nn.Module):
    """Predicts a coarse crowd density map used as auxiliary supervision."""

    def __init__(self, channels: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class BehaviourZoneAttention(nn.Module):
    """Highlights regions of interest in behaviour scenes."""

    def __init__(self, channels: int, num_zones: int = 4):
        super().__init__()
        self.num_zones = num_zones
        self.zone_conv = nn.Conv2d(channels, num_zones, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.zone_conv(x), dim=1)


class FeaturePyramidEnhancement(nn.Module):
    """FPN-style top-down pathway used inside the encoder."""

    def __init__(self, channels: int):
        super().__init__()
        self.lateral_c4 = nn.Conv2d(channels, channels, 1)
        self.lateral_c3 = nn.Conv2d(channels, channels, 1)
        self.smooth_c3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.smooth_c4 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(feats) < 3:
            return feats
        c3, c4, c5 = feats
        p5 = c5
        p4 = self.smooth_c4(self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest"))
        p3 = self.smooth_c3(self.lateral_c3(c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest"))
        return [p3, p4, p5]


class SurveillanceFeatureEnhancement(nn.Module):
    """Full feature enhancement block (SFEN)."""

    def __init__(self, channels: int, use_aux_heads: bool = True):
        super().__init__()
        self.human_attention = HumanPriorityAttention(channels)
        self.pose_attention = PosePriorAttention(channels)
        self.density_head = CrowdDensityHead(channels) if use_aux_heads else None
        self.zone_attention = BehaviourZoneAttention(channels)
        self.csa = ChannelSpatialAttention(channels)
        self.fpn = FeaturePyramidEnhancement(channels)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pyramid = self.fpn([x, x, x])  # treat single-scale input as C3
        refined, keypoints = self.pose_attention(x)
        human_mask = self.human_attention(pyramid)
        density = self.density_head(refined) if self.density_head is not None else None
        zones = self.zone_attention(refined)
        enhanced = self.csa(refined) * (1.0 + human_mask)
        enhanced = self.refine(enhanced + refined)
        return {
            "enhanced": enhanced,
            "human_mask": human_mask,
            "keypoints": keypoints,
            "density": density,
            "zones": zones,
        }


if __name__ == "__main__":
    print("Testing Surveillance Feature Enhancement...")
    sf = SurveillanceFeatureEnhancement(256)
    x = torch.randn(2, 256, 80, 80)
    out = sf(x)
    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k}: {v.shape}")
        else:
            print(f"  {k}: {type(v)}")
    print("done")
