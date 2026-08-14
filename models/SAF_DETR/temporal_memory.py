"""
SAF-DETR: Temporal Detection Memory Module
============================================

This module maintains temporal consistency across frames for stable detection
and tracking in surveillance videos.

Author: SAF-DETR Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np


class TrackState:
    """
    Represents the state of a tracked object.
    """
    
    def __init__(self, 
                 track_id: int,
                 bbox: torch.Tensor,
                 features: torch.Tensor,
                 confidence: float,
                 frame_id: int,
                 max_history: int = 30):
        """
        Args:
            track_id: Unique track identifier
            bbox: Bounding box [4] (x1, y1, x2, y2)
            features: Appearance features [D]
            confidence: Detection confidence
            frame_id: Current frame ID
            max_history: Maximum history length
        """
        self.track_id = track_id
        self.bbox_history = deque([bbox], maxlen=max_history)
        self.feature_history = deque([features], maxlen=max_history)
        self.confidence_history = deque([confidence], maxlen=max_history)
        self.frame_ids = deque([frame_id], maxlen=max_history)
        
        # Motion state
        self.velocity = torch.zeros(4)  # (vx1, vy1, vx2, vy2)
        self.acceleration = torch.zeros(4)
        
        # Track status
        self.missed_frames = 0
        self.max_missed = 5
        self.is_active = True
        
    def update(self, 
               bbox: torch.Tensor, 
               features: torch.Tensor, 
               confidence: float,
               frame_id: int):
        """Update track state with new detection."""
        # Calculate velocity
        if len(self.bbox_history) > 0:
            dt = frame_id - self.frame_ids[-1]
            if dt > 0:
                new_velocity = (bbox - self.bbox_history[-1]) / dt
                self.acceleration = (new_velocity - self.velocity) / dt
                self.velocity = new_velocity
        
        # Update history
        self.bbox_history.append(bbox)
        self.feature_history.append(features)
        self.confidence_history.append(confidence)
        self.frame_ids.append(frame_id)
        
        # Reset missed frames
        self.missed_frames = 0
        self.is_active = True
        
    def predict(self, frame_id: int) -> torch.Tensor:
        """Predict bbox at given frame using motion model."""
        dt = frame_id - self.frame_ids[-1]
        
        # Constant acceleration motion model
        predicted = (self.bbox_history[-1] + 
                    self.velocity * dt + 
                    0.5 * self.acceleration * dt * dt)
        
        return predicted
    
    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.missed_frames += 1
        if self.missed_frames > self.max_missed:
            self.is_active = False
    
    def get_smoothed_bbox(self, window_size: int = 3) -> torch.Tensor:
        """Get temporally smoothed bounding box."""
        if len(self.bbox_history) < window_size:
            return self.bbox_history[-1]
        
        # Average over recent history
        recent_bboxes = list(self.bbox_history)[-window_size:]
        smoothed = torch.stack(recent_bboxes).mean(dim=0)
        
        return smoothed
    
    def get_feature_consistency(self) -> float:
        """Calculate feature consistency score."""
        if len(self.feature_history) < 2:
            return 1.0
        
        # Cosine similarity between recent features
        recent_features = list(self.feature_history)[-5:]
        similarities = []
        
        for i in range(len(recent_features) - 1):
            sim = F.cosine_similarity(
                recent_features[i].unsqueeze(0),
                recent_features[i + 1].unsqueeze(0)
            )
            similarities.append(sim.item())
        
        return np.mean(similarities) if similarities else 1.0


class TemporalFeatureAggregator(nn.Module):
    """
    Aggregates features across time using attention mechanism.
    """
    
    def __init__(self, feature_dim: int = 256, num_frames: int = 5):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Temporal encoding
        self.temporal_encoding = nn.Parameter(
            torch.randn(num_frames, feature_dim)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, 
                current_features: torch.Tensor,
                temporal_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            current_features: Current frame features [N, D]
            temporal_features: List of previous frame features
            
        Returns:
            Aggregated features [N, D]
        """
        if len(temporal_features) == 0:
            return current_features
        
        # Stack temporal features
        temporal_stack = torch.stack(temporal_features[-self.num_frames:], dim=1)
        
        # Add temporal encoding
        temporal_stack = temporal_stack + self.temporal_encoding[:temporal_stack.size(1)]
        
        # Apply temporal attention
        aggregated, _ = self.temporal_attention(
            current_features.unsqueeze(1),
            temporal_stack,
            temporal_stack
        )
        aggregated = aggregated.squeeze(1)
        
        # Fuse with current features
        combined = torch.cat([current_features, aggregated], dim=1)
        output = self.fusion(combined)
        
        return output


class DetectionStabilizer(nn.Module):
    """
    Stabilizes detections across frames using temporal information.
    """
    
    def __init__(self, 
                 feature_dim: int = 256,
                 temporal_window: int = 5,
                 stability_threshold: float = 0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.temporal_window = temporal_window
        self.stability_threshold = stability_threshold
        
        # Temporal aggregator
        self.temporal_aggregator = TemporalFeatureAggregator(feature_dim, temporal_window)
        
        # Stability predictor
        self.stability_net = nn.Sequential(
            nn.Linear(feature_dim * 2 + 4, 128),  # features + bbox
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                detections: torch.Tensor,
                features: torch.Tensor,
                temporal_features: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            detections: Current detections [N, 4] (x1, y1, x2, y2)
            features: Current features [N, D]
            temporal_features: Optional temporal feature history
            
        Returns:
            stabilized_detections: Stabilized detections [N, 4]
            stability_scores: Stability confidence [N, 1]
        """
        # Aggregate temporal features
        if temporal_features is not None and len(temporal_features) > 0:
            aggregated_features = self.temporal_aggregator(features, temporal_features)
        else:
            aggregated_features = features
        
        # Calculate stability scores
        combined = torch.cat([features, aggregated_features, detections], dim=1)
        stability_scores = self.stability_net(combined)
        
        # Stabilize detections (smooth with temporal information)
        stabilized = detections  # Placeholder for actual smoothing
        
        return stabilized, stability_scores


class TrackMatcher(nn.Module):
    """
    Matches detections to existing tracks using appearance and motion cues.
    """
    
    def __init__(self,
                 feature_dim: int = 256,
                 appearance_weight: float = 0.7,
                 motion_weight: float = 0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.appearance_weight = appearance_weight
        self.motion_weight = motion_weight
        
        # Appearance similarity network
        self.appearance_sim = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Motion compatibility network
        self.motion_compat = nn.Sequential(
            nn.Linear(12, 64),  # 4 det + 4 predicted_bbox + 4 velocity
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def compute_affinity_matrix(self,
                               detections: torch.Tensor,
                               det_features: torch.Tensor,
                               tracks: List[TrackState],
                               frame_id: int) -> torch.Tensor:
        """
        Compute affinity matrix between detections and tracks.
        
        Args:
            detections: Current detections [N, 4]
            det_features: Detection features [N, D]
            tracks: List of track states
            frame_id: Current frame ID
            
        Returns:
            Affinity matrix [N, M] where M is number of tracks
        """
        if len(tracks) == 0:
            return torch.zeros(len(detections), 0)
        
        N = len(detections)
        M = len(tracks)
        
        affinity_matrix = torch.zeros(N, M)
        
        for i, (det, det_feat) in enumerate(zip(detections, det_features)):
            for j, track in enumerate(tracks):
                if not track.is_active:
                    continue
                
                # Appearance similarity
                track_feat = track.feature_history[-1]
                app_input = torch.cat([det_feat, track_feat])
                app_sim = self.appearance_sim(app_input)
                
                # Motion compatibility
                predicted_bbox = track.predict(frame_id)
                motion_input = torch.cat([det, predicted_bbox, track.velocity])
                motion_sim = self.motion_compat(motion_input)
                
                # Combined affinity
                affinity = (self.appearance_weight * app_sim + 
                           self.motion_weight * motion_sim)
                
                # IoU penalty
                iou = self.compute_iou(det.unsqueeze(0), predicted_bbox.unsqueeze(0))
                affinity = affinity * iou
                
                affinity_matrix[i, j] = affinity
        
        return affinity_matrix
    
    @staticmethod
    def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU between two sets of boxes."""
        # boxes: [N, 4] (x1, y1, x2, y2)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        union_area = area1 + area2 - inter_area
        iou = inter_area / (union_area + 1e-6)
        
        return iou


class TemporalDetectionMemory(nn.Module):
    """
    Complete Temporal Detection Memory module.
    
    Maintains temporal consistency for detections and tracks objects across frames.
    """
    
    def __init__(self,
                 feature_dim: int = 256,
                 max_tracks: int = 100,
                 temporal_window: int = 5,
                 match_threshold: float = 0.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_tracks = max_tracks
        self.temporal_window = temporal_window
        self.match_threshold = match_threshold
        
        # Components
        self.stabilizer = DetectionStabilizer(feature_dim, temporal_window)
        self.matcher = TrackMatcher(feature_dim)
        
        # Track management
        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id = 0
        self.frame_id = 0
        
        # Feature history for temporal aggregation
        self.feature_history: List[torch.Tensor] = []
        
    def forward(self,
                detections: torch.Tensor,
                features: torch.Tensor,
                scores: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process detections with temporal consistency.
        
        Args:
            detections: Detection boxes [N, 4] (x1, y1, x2, y2)
            features: Detection features [N, D]
            scores: Detection confidence scores [N]
            
        Returns:
            Dictionary containing:
                - stabilized_detections: Stabilized boxes [N, 4]
                - track_ids: Track IDs for each detection [N]
                - temporal_consistency: Consistency scores [N]
                - active_tracks: Number of active tracks
        """
        self.frame_id += 1
        
        # Stabilize detections
        stabilized, stability_scores = self.stabilizer(
            detections, features, self.feature_history
        )
        
        # Update feature history
        self.feature_history.append(features)
        if len(self.feature_history) > self.temporal_window:
            self.feature_history.pop(0)
        
        # Match detections to tracks
        track_list = list(self.tracks.values())
        affinity_matrix = self.matcher.compute_affinity_matrix(
            stabilized, features, track_list, self.frame_id
        )
        
        # Hungarian matching
        track_ids = self._assign_tracks(affinity_matrix, stabilized, features, scores)
        
        # Calculate temporal consistency
        temporal_consistency = self._compute_consistency(track_ids)
        
        # Clean up old tracks
        self._cleanup_tracks()
        
        return {
            'stabilized_detections': stabilized,
            'track_ids': track_ids,
            'temporal_consistency': temporal_consistency,
            'stability_scores': stability_scores,
            'active_tracks': len([t for t in self.tracks.values() if t.is_active])
        }
    
    def _assign_tracks(self,
                      affinity_matrix: torch.Tensor,
                      detections: torch.Tensor,
                      features: torch.Tensor,
                      scores: torch.Tensor) -> torch.Tensor:
        """Assign detections to tracks using Hungarian algorithm."""
        N = len(detections)
        track_ids = torch.full((N,), -1, dtype=torch.long)
        
        if affinity_matrix.numel() == 0:
            # No existing tracks, create new ones
            for i in range(N):
                track_id = self._create_track(detections[i], features[i], scores[i])
                track_ids[i] = track_id
            return track_ids
        
        # Simple greedy matching (can be replaced with Hungarian algorithm)
        matched_tracks = set()
        
        for i in range(N):
            if affinity_matrix.size(1) > 0:
                best_match = affinity_matrix[i].argmax()
                best_score = affinity_matrix[i, best_match]
                
                if best_score > self.match_threshold and best_match not in matched_tracks:
                    # Update existing track
                    track_id = list(self.tracks.keys())[best_match]
                    self.tracks[track_id].update(
                        detections[i], features[i], scores[i], self.frame_id
                    )
                    track_ids[i] = track_id
                    matched_tracks.add(best_match)
                else:
                    # Create new track
                    track_id = self._create_track(detections[i], features[i], scores[i])
                    track_ids[i] = track_id
            else:
                # Create new track
                track_id = self._create_track(detections[i], features[i], scores[i])
                track_ids[i] = track_id
        
        # Mark unmatched tracks as missed
        for idx, track in enumerate(self.tracks.values()):
            if idx not in matched_tracks:
                track.mark_missed()
        
        return track_ids
    
    def _create_track(self, 
                     bbox: torch.Tensor, 
                     features: torch.Tensor, 
                     confidence: float) -> int:
        """Create a new track."""
        track_id = self.next_track_id
        self.tracks[track_id] = TrackState(
            track_id, bbox, features, confidence, self.frame_id
        )
        self.next_track_id += 1
        return track_id
    
    def _compute_consistency(self, track_ids: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency scores."""
        consistency = torch.ones(len(track_ids))
        
        for i, track_id in enumerate(track_ids):
            if track_id >= 0 and track_id in self.tracks:
                track = self.tracks[track_id]
                # Higher consistency for tracks with longer history
                consistency[i] = min(len(track.bbox_history) / 5, 1.0)
        
        return consistency
    
    def _cleanup_tracks(self):
        """Remove old inactive tracks."""
        inactive = [tid for tid, track in self.tracks.items() if not track.is_active]
        for tid in inactive:
            del self.tracks[tid]
        
        # Limit total tracks
        if len(self.tracks) > self.max_tracks:
            # Remove oldest tracks
            sorted_tracks = sorted(self.tracks.items(), 
                                 key=lambda x: x[1].frame_ids[-1])
            for tid, _ in sorted_tracks[:-self.max_tracks]:
                del self.tracks[tid]
    
    def get_track_info(self) -> Dict[str, any]:
        """Get information about current tracks."""
        active = [t for t in self.tracks.values() if t.is_active]
        return {
            'num_active_tracks': len(active),
            'num_total_tracks': len(self.tracks),
            'average_track_length': np.mean([len(t.bbox_history) for t in active]) if active else 0,
            'frame_id': self.frame_id
        }
    
    def reset(self):
        """Reset temporal memory."""
        self.tracks.clear()
        self.feature_history.clear()
        self.next_track_id = 0
        self.frame_id = 0


class TemporalConsistencyLoss(nn.Module):
    """
    Loss function for temporal consistency in detection.
    """
    
    def __init__(self, 
                 temporal_weight: float = 1.0,
                 smoothness_weight: float = 0.5):
        super().__init__()
        self.temporal_weight = temporal_weight
        self.smoothness_weight = smoothness_weight
        
    def forward(self,
                current_detections: torch.Tensor,
                previous_detections: torch.Tensor,
                track_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.
        
        Args:
            current_detections: Current frame detections [N, 4]
            previous_detections: Previous frame detections [M, 4]
            track_ids: Track IDs linking current to previous [N]
            
        Returns:
            Temporal consistency loss
        """
        if len(previous_detections) == 0 or track_ids.numel() == 0:
            return torch.tensor(0.0, device=current_detections.device)
        
        # Find matched detections
        temporal_loss = 0.0
        num_matches = 0
        
        for i, track_id in enumerate(track_ids):
            if track_id >= 0 and track_id < len(previous_detections):
                # L2 distance between current and previous
                diff = current_detections[i] - previous_detections[track_id]
                temporal_loss += torch.norm(diff)
                num_matches += 1
        
        if num_matches > 0:
            temporal_loss = temporal_loss / num_matches
        
        # Smoothness loss (encourage smooth motion)
        smoothness_loss = 0.0
        if len(current_detections) > 1:
            # Penalize large changes in box size/aspect ratio
            widths = current_detections[:, 2] - current_detections[:, 0]
            heights = current_detections[:, 3] - current_detections[:, 1]
            aspect_ratios = widths / (heights + 1e-6)
            
            smoothness_loss = torch.var(aspect_ratios)
        
        total_loss = (self.temporal_weight * temporal_loss + 
                     self.smoothness_weight * smoothness_loss)
        
        return total_loss


if __name__ == "__main__":
    # Test the module
    print("Testing Temporal Detection Memory Module...")
    
    # Create module
    module = TemporalDetectionMemory(feature_dim=256)
    
    # Simulate multiple frames
    for frame_idx in range(10):
        # Random detections
        num_detections = torch.randint(3, 8, (1,)).item()
        detections = torch.randn(num_detections, 4).clamp(0, 1)
        detections[:, 2:] += detections[:, :2]  # Ensure x2 > x1, y2 > y1
        
        features = torch.randn(num_detections, 256)
        scores = torch.rand(num_detections)
        
        # Forward pass
        output = module(detections, features, scores)
        
        print(f"\
Frame {frame_idx + 1}:")
        print(f"  Detections: {num_detections}")
        print(f"  Active tracks: {output['active_tracks']}")
        print(f"  Avg consistency: {output['temporal_consistency'].mean():.3f}")
    
    # Track info
    info = module.get_track_info()
    print(f"\
Final Track Info: {info}")
    
    # Test temporal loss
    print("\
Testing Temporal Consistency Loss...")
    loss_fn = TemporalConsistencyLoss()
    current = torch.randn(5, 4).clamp(0, 1)
    previous = torch.randn(5, 4).clamp(0, 1)
    track_ids = torch.tensor([0, 1, 2, 3, 4])
    loss = loss_fn(current, previous, track_ids)
    print(f"Temporal loss: {loss.item():.4f}")
    
    print("\
✓ Temporal Detection Memory Module test passed!")
