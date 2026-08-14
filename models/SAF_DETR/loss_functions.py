"""
SAF-DETR: Behaviour-Aware Loss Functions
=========================================

Custom loss functions for SAF-DETR that incorporate detection, tracking,
interaction awareness, and temporal stability.

Author: SAF-DETR Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class DetectionLoss(nn.Module):
    """
    Standard detection loss with classification and box regression.
    """
    
    def __init__(self,
                 num_classes: int = 1,  # Person only
                 class_weight: float = 1.0,
                 box_weight: float = 5.0,
                 giou_weight: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.box_weight = box_weight
        self.giou_weight = giou_weight
        
        # Focal loss for classification
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
    def forward(self,
                pred_boxes: torch.Tensor,
                pred_logits: torch.Tensor,
                target_boxes: torch.Tensor,
                target_labels: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute detection loss.
        
        Args:
            pred_boxes: Predicted boxes [N, 4] (x1, y1, x2, y2)
            pred_logits: Predicted class logits [N, num_classes]
            target_boxes: Target boxes [M, 4]
            target_labels: Target labels [M]
            
        Returns:
            total_loss: Combined detection loss
            loss_dict: Dictionary of individual losses
        """
        # Ensure 2D format for box calculations
        if target_boxes.dim() == 3:
            target_boxes = target_boxes.view(-1, 4)
        if target_labels.dim() > 1:
            target_labels = target_labels.view(-1)

        # Filter out zero padded targets
        if len(target_boxes) > 0:
            valid_mask = (target_boxes.abs().sum(dim=-1) > 1e-4)
            target_boxes = target_boxes[valid_mask]
            target_labels = target_labels[valid_mask]

        # Match predictions to targets
        matched_pred_idx, matched_tgt_idx = self._match_predictions(pred_boxes, target_boxes)
        
        # Classification loss
        target_classes = torch.zeros(len(pred_logits), dtype=torch.long, device=pred_logits.device)
        if len(matched_pred_idx) > 0:
            target_classes[matched_pred_idx] = target_labels[matched_tgt_idx]
            
        cls_loss = self.focal_loss(pred_logits, target_classes)
        
        # Box regression loss
        if len(matched_pred_idx) > 0:
            matched_pred_boxes = pred_boxes[matched_pred_idx]
            matched_target_boxes = target_boxes[matched_tgt_idx]
            
            # L1 loss
            l1_loss = F.l1_loss(matched_pred_boxes, matched_target_boxes)
            
            # GIoU loss
            giou_loss = self._giou_loss(matched_pred_boxes, matched_target_boxes)
        else:
            l1_loss = torch.tensor(0.0, device=pred_boxes.device)
            giou_loss = torch.tensor(0.0, device=pred_boxes.device)
        
        # Combine losses
        total_loss = (self.class_weight * cls_loss + 
                     self.box_weight * l1_loss + 
                     self.giou_weight * giou_loss)
        
        loss_dict = {
            'cls_loss': cls_loss.item(),
            'l1_loss': l1_loss.item(),
            'giou_loss': giou_loss.item(),
            'total_detection_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _match_predictions(self,
                          pred_boxes: torch.Tensor,
                          target_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match predictions to targets using IoU."""
        if len(target_boxes) == 0 or len(pred_boxes) == 0:
            empty = torch.empty(0, dtype=torch.long, device=pred_boxes.device)
            return empty, empty
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(pred_boxes, target_boxes)
        
        # Greedy matching
        matched_preds = []
        matched_targets = []
        used_targets = set()
        
        for i in range(len(pred_boxes)):
            best_iou = 0.3  # Threshold
            best_target = -1
            
            for j in range(len(target_boxes)):
                if j not in used_targets and iou_matrix[i, j].item() > best_iou:
                    best_iou = iou_matrix[i, j].item()
                    best_target = j
            
            if best_target >= 0:
                matched_preds.append(i)
                matched_targets.append(best_target)
                used_targets.add(best_target)
        
        return (torch.tensor(matched_preds, dtype=torch.long, device=pred_boxes.device),
                torch.tensor(matched_targets, dtype=torch.long, device=pred_boxes.device))
    
    @staticmethod
    def _compute_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute IoU matrix between two sets of boxes."""
        # boxes: [N, 4] (x1, y1, x2, y2)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Expand dimensions for broadcasting
        boxes1 = boxes1.unsqueeze(1)  # [N, 1, 4]
        boxes2 = boxes2.unsqueeze(0)  # [1, M, 4]
        
        inter_x1 = torch.max(boxes1[:, :, 0], boxes2[:, :, 0])
        inter_y1 = torch.max(boxes1[:, :, 1], boxes2[:, :, 1])
        inter_x2 = torch.min(boxes1[:, :, 2], boxes2[:, :, 2])
        inter_y2 = torch.min(boxes1[:, :, 3], boxes2[:, :, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
        iou = inter_area / (union_area + 1e-6)
        
        return iou
    
    @staticmethod
    def _giou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute GIoU loss."""
        # Compute IoU
        area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        area2 = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        union_area = area1 + area2 - inter_area
        iou = inter_area / (union_area + 1e-6)
        
        # Compute enclosing box
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        
        # GIoU
        giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
        
        return (1 - giou).mean()


class FocalLoss(nn.Module):
    """Focal Loss for classification."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class TrackingConsistencyLoss(nn.Module):
    """
    Loss for tracking consistency across frames.
    """
    
    def __init__(self,
                 temporal_weight: float = 1.0,
                 feature_weight: float = 0.5,
                 motion_weight: float = 0.3):
        super().__init__()
        self.temporal_weight = temporal_weight
        self.feature_weight = feature_weight
        self.motion_weight = motion_weight
        
    def forward(self,
                current_boxes: torch.Tensor,
                current_features: torch.Tensor,
                track_ids: torch.Tensor,
                track_history: Dict[int, Dict]) -> torch.Tensor:
        """
        Compute tracking consistency loss.
        
        Args:
            current_boxes: Current frame boxes [N, 4]
            current_features: Current frame features [N, D]
            track_ids: Track IDs for each detection [N]
            track_history: Dictionary of track histories
            
        Returns:
            Tracking consistency loss
        """
        temporal_loss = 0.0
        feature_loss = 0.0
        motion_loss = 0.0
        num_tracks = 0
        
        for i, track_id in enumerate(track_ids):
            if track_id < 0 or track_id not in track_history:
                continue
            
            track = track_history[track_id]
            
            # Temporal consistency (box should be close to prediction)
            if 'predicted_box' in track:
                pred_box = track['predicted_box']
                temporal_loss += F.l1_loss(current_boxes[i], pred_box)
            
            # Feature consistency (appearance should be similar)
            if 'last_features' in track:
                last_feat = track['last_features']
                # Cosine similarity
                sim = F.cosine_similarity(
                    current_features[i].unsqueeze(0),
                    last_feat.unsqueeze(0)
                )
                feature_loss += 1 - sim
            
            # Motion smoothness
            if 'velocity' in track and len(track.get('box_history', [])) > 1:
                velocity = track['velocity']
                last_box = track['box_history'][-1]
                expected_box = last_box + velocity
                motion_loss += F.l1_loss(current_boxes[i], expected_box)
            
            num_tracks += 1
        
        if num_tracks > 0:
            temporal_loss = temporal_loss / num_tracks
            feature_loss = feature_loss / num_tracks
            motion_loss = motion_loss / num_tracks
        
        total_loss = (self.temporal_weight * temporal_loss +
                     self.feature_weight * feature_loss +
                     self.motion_weight * motion_loss)
        
        return total_loss


class InteractionAwarenessLoss(nn.Module):
    """
    Loss for encouraging awareness of human interactions.
    """
    
    def __init__(self,
                 proximity_weight: float = 1.0,
                 group_weight: float = 0.5,
                 conflict_weight: float = 2.0):
        super().__init__()
        self.proximity_weight = proximity_weight
        self.group_weight = group_weight
        self.conflict_weight = conflict_weight
        
    def forward(self,
                boxes: torch.Tensor,
                features: torch.Tensor,
                interaction_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute interaction awareness loss.
        
        Args:
            boxes: Detection boxes [N, 4]
            features: Detection features [N, D]
            interaction_labels: Optional interaction labels [N, N]
            
        Returns:
            Interaction awareness loss
        """
        if len(boxes) < 2:
            return torch.tensor(0.0, device=boxes.device)
        
        # Compute pairwise distances
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2  # [N, 2]
        distances = torch.cdist(centers, centers)  # [N, N]
        
        # Compute pairwise feature similarities
        feature_sims = torch.mm(features, features.t())  # [N, N]
        
        # Proximity loss: encourage attention to nearby detections
        proximity_loss = 0.0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                # Closer boxes should have more similar features
                distance = distances[i, j]
                feature_sim = feature_sims[i, j]
                
                # Loss: nearby boxes should have similar features
                expected_sim = torch.exp(-distance / 100)  # Normalize by typical distance
                proximity_loss += F.mse_loss(feature_sim, expected_sim)
        
        proximity_loss = proximity_loss / (len(boxes) * (len(boxes) - 1) / 2 + 1e-6)
        
        # Group detection loss
        group_loss = self._compute_group_loss(boxes, features)
        
        # Conflict detection loss (if labels provided)
        conflict_loss = 0.0
        if interaction_labels is not None:
            conflict_loss = self._compute_conflict_loss(features, interaction_labels)
        
        total_loss = (self.proximity_weight * proximity_loss +
                     self.group_weight * group_loss +
                     self.conflict_weight * conflict_loss)
        
        return total_loss
    
    def _compute_group_loss(self, boxes: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Compute loss for group detection."""
        # Find groups based on proximity
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        distances = torch.cdist(centers, centers)
        
        # Group threshold (e.g., 100 pixels)
        group_threshold = 100.0
        in_group = (distances < group_threshold).float()
        
        # Within-group feature similarity should be high
        feature_sims = torch.mm(features, features.t())
        
        # Loss: in-group pairs should have high similarity
        group_loss = F.mse_loss(feature_sims * in_group, in_group)
        
        return group_loss
    
    def _compute_conflict_loss(self, 
                              features: torch.Tensor,
                              interaction_labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for conflict detection."""
        # interaction_labels: [N, N] with values {0: neutral, 1: friendly, 2: conflict}
        
        # Predict interaction from features
        feature_pairs = []
        for i in range(len(features)):
            for j in range(len(features)):
                pair_feat = torch.cat([features[i], features[j]])
                feature_pairs.append(pair_feat)
        
        if len(feature_pairs) == 0:
            return torch.tensor(0.0, device=features.device)
        
        # This would require an interaction prediction head
        # For now, return 0
        return torch.tensor(0.0, device=features.device)


class TemporalStabilityLoss(nn.Module):
    """
    Loss for temporal stability in video sequences.
    """
    
    def __init__(self,
                 smoothness_weight: float = 1.0,
                 continuity_weight: float = 0.5,
                 confidence_weight: float = 0.3):
        super().__init__()
        self.smoothness_weight = smoothness_weight
        self.continuity_weight = continuity_weight
        self.confidence_weight = confidence_weight
        
    def forward(self,
                boxes_sequence: List[torch.Tensor],
                scores_sequence: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute temporal stability loss.
        
        Args:
            boxes_sequence: List of box tensors for T frames
            scores_sequence: List of score tensors for T frames
            
        Returns:
            Temporal stability loss
        """
        if len(boxes_sequence) < 2:
            return torch.tensor(0.0, device=boxes_sequence[0].device)
        
        smoothness_loss = 0.0
        continuity_loss = 0.0
        confidence_loss = 0.0
        
        for t in range(1, len(boxes_sequence)):
            prev_boxes = boxes_sequence[t - 1]
            curr_boxes = boxes_sequence[t]
            prev_scores = scores_sequence[t - 1]
            curr_scores = scores_sequence[t]
            
            # Smoothness: penalize large changes in box parameters
            if len(prev_boxes) > 0 and len(curr_boxes) > 0:
                # Match boxes between frames
                matched_prev, matched_curr = self._match_boxes(prev_boxes, curr_boxes)
                
                if len(matched_prev) > 0:
                    # L2 distance between matched boxes
                    box_diff = matched_prev - matched_curr
                    smoothness_loss += torch.mean(box_diff ** 2)
                    
                    # Continuity: velocity should be smooth
                    if t > 1:
                        prev_prev_boxes = boxes_sequence[t - 2]
                        if len(prev_prev_boxes) > 0:
                            matched_pp, matched_p = self._match_boxes(prev_prev_boxes, prev_boxes)
                            if len(matched_pp) > 0:
                                velocity1 = matched_p - matched_pp
                                velocity2 = matched_curr - matched_prev
                                continuity_loss += torch.mean((velocity1 - velocity2) ** 2)
            
            # Confidence stability
            if len(prev_scores) > 0 and len(curr_scores) > 0:
                min_len = min(len(prev_scores), len(curr_scores))
                conf_diff = prev_scores[:min_len] - curr_scores[:min_len]
                confidence_loss += torch.mean(conf_diff ** 2)
        
        num_transitions = len(boxes_sequence) - 1
        smoothness_loss = smoothness_loss / num_transitions
        continuity_loss = continuity_loss / max(num_transitions - 1, 1)
        confidence_loss = confidence_loss / num_transitions
        
        total_loss = (self.smoothness_weight * smoothness_loss +
                     self.continuity_weight * continuity_loss +
                     self.confidence_weight * confidence_loss)
        
        return total_loss
    
    def _match_boxes(self, 
                     boxes1: torch.Tensor, 
                     boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match boxes between frames using IoU."""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return torch.zeros(0, 4, device=boxes1.device), torch.zeros(0, 4, device=boxes2.device)
        
        # Compute IoU matrix
        iou_matrix = DetectionLoss._compute_iou_matrix(boxes1, boxes2)
        
        # Greedy matching
        matched1 = []
        matched2 = []
        used = set()
        
        for i in range(len(boxes1)):
            best_j = -1
            best_iou = 0.5
            
            for j in range(len(boxes2)):
                if j not in used and iou_matrix[i, j] > best_iou:
                    best_j = j
                    best_iou = iou_matrix[i, j]
            
            if best_j >= 0:
                matched1.append(boxes1[i])
                matched2.append(boxes2[best_j])
                used.add(best_j)
        
        if len(matched1) == 0:
            return torch.zeros(0, 4, device=boxes1.device), torch.zeros(0, 4, device=boxes2.device)
        
        return torch.stack(matched1), torch.stack(matched2)


class BehaviourAwareLoss(nn.Module):
    """
    Complete behaviour-aware loss function for SAF-DETR.
    
    Combines:
    - Detection loss
    - Tracking consistency loss
    - Interaction awareness loss
    - Temporal stability loss
    """
    
    def __init__(self,
                 detection_weight: float = 1.0,
                 tracking_weight: float = 0.5,
                 interaction_weight: float = 0.3,
                 temporal_weight: float = 0.2):
        super().__init__()
        
        self.detection_weight = detection_weight
        self.tracking_weight = tracking_weight
        self.interaction_weight = interaction_weight
        self.temporal_weight = temporal_weight
        
        # Individual loss components
        self.detection_loss = DetectionLoss()
        self.tracking_loss = TrackingConsistencyLoss()
        self.interaction_loss = InteractionAwarenessLoss()
        self.temporal_stability_loss = TemporalStabilityLoss()
        
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                track_history: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute complete behaviour-aware loss.
        
        Args:
            predictions: Dictionary containing:
                - boxes: Predicted boxes [N, 4]
                - logits: Predicted logits [N, num_classes]
                - features: Predicted features [N, D]
                - scores: Confidence scores [N]
            targets: Dictionary containing:
                - boxes: Target boxes [M, 4]
                - labels: Target labels [M]
            track_history: Optional tracking history
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        
        # Detection loss
        det_loss, det_dict = self.detection_loss(
            predictions['boxes'],
            predictions['logits'],
            targets['boxes'],
            targets['labels']
        )
        loss_dict.update(det_dict)
        
        # Tracking consistency loss
        if track_history is not None and 'track_ids' in predictions:
            track_loss = self.tracking_loss(
                predictions['boxes'],
                predictions['features'],
                predictions['track_ids'],
                track_history
            )
            loss_dict['tracking_loss'] = track_loss.item()
        else:
            track_loss = torch.tensor(0.0, device=det_loss.device)
            loss_dict['tracking_loss'] = 0.0
        
        # Interaction awareness loss
        if len(predictions['boxes']) > 1:
            inter_loss = self.interaction_loss(
                predictions['boxes'],
                predictions['features']
            )
            loss_dict['interaction_loss'] = inter_loss.item()
        else:
            inter_loss = torch.tensor(0.0, device=det_loss.device)
            loss_dict['interaction_loss'] = 0.0
        
        # Temporal stability loss (requires sequence)
        if 'boxes_sequence' in predictions:
            temp_loss = self.temporal_stability_loss(
                predictions['boxes_sequence'],
                predictions['scores_sequence']
            )
            loss_dict['temporal_loss'] = temp_loss.item()
        else:
            temp_loss = torch.tensor(0.0, device=det_loss.device)
            loss_dict['temporal_loss'] = 0.0
        
        # Combine losses
        total_loss = (self.detection_weight * det_loss +
                     self.tracking_weight * track_loss +
                     self.interaction_weight * inter_loss +
                     self.temporal_weight * temp_loss)
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        return {
            'detection': self.detection_weight,
            'tracking': self.tracking_weight,
            'interaction': self.interaction_weight,
            'temporal': self.temporal_weight
        }
    
    def set_loss_weights(self, weights: Dict[str, float]):
        """Set loss weights."""
        self.detection_weight = weights.get('detection', self.detection_weight)
        self.tracking_weight = weights.get('tracking', self.tracking_weight)
        self.interaction_weight = weights.get('interaction', self.interaction_weight)
        self.temporal_weight = weights.get('temporal', self.temporal_weight)


if __name__ == "__main__":
    # Test the loss functions
    print("Testing Behaviour-Aware Loss Functions...")
    
    # Create loss function
    loss_fn = BehaviourAwareLoss()
    
    # Create dummy predictions and targets
    predictions = {
        'boxes': torch.randn(5, 4).clamp(0, 1),
        'logits': torch.randn(5, 2),
        'features': torch.randn(5, 256),
        'scores': torch.rand(5),
        'track_ids': torch.tensor([0, 1, 2, 3, 4])
    }
    
    # Ensure valid boxes (x2 > x1, y2 > y1)
    predictions['boxes'][:, 2:] = predictions['boxes'][:, :2] + torch.abs(predictions['boxes'][:, 2:])
    
    targets = {
        'boxes': torch.randn(3, 4).clamp(0, 1),
        'labels': torch.tensor([1, 1, 0])
    }
    targets['boxes'][:, 2:] = targets['boxes'][:, :2] + torch.abs(targets['boxes'][:, 2:])
    
    # Compute loss
    total_loss, loss_dict = loss_fn(predictions, targets)
    
    print(f"\
Loss Components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\
Total Loss: {total_loss.item():.4f}")
    
    # Test individual losses
    print("\
Testing Individual Loss Components...")
    
    # Detection loss
    det_loss = DetectionLoss()
    loss, det_dict = det_loss(predictions['boxes'], predictions['logits'],
                             targets['boxes'], targets['labels'])
    print(f"Detection Loss: {loss.item():.4f}")
    
    # Tracking loss
    track_loss_fn = TrackingConsistencyLoss()
    track_history = {
        0: {'predicted_box': torch.randn(4), 'last_features': torch.randn(256)},
        1: {'predicted_box': torch.randn(4), 'last_features': torch.randn(256)}
    }
    track_loss = track_loss_fn(predictions['boxes'], predictions['features'],
                               predictions['track_ids'], track_history)
    print(f"Tracking Loss: {track_loss.item():.4f}")
    
    # Interaction loss
    inter_loss_fn = InteractionAwarenessLoss()
    inter_loss = inter_loss_fn(predictions['boxes'], predictions['features'])
    print(f"Interaction Loss: {inter_loss.item():.4f}")
    
    print("\
✓ Behaviour-Aware Loss Functions test passed!")
