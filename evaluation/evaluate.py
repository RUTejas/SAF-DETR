"""
SAF-DETR Evaluation Framework
==============================

Comprehensive evaluation with multiple metrics, ablation studies,
and visualization tools.

Author: SAF-DETR Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# Import model
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.SAF_DETR.complete_model import build_saf_detr


class DetectionMetrics:
    """
    Standard detection metrics: AP, mAP, Precision, Recall.
    """
    
    def __init__(self, iou_thresholds: List[float] = [0.5, 0.75]):
        self.iou_thresholds = iou_thresholds
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.ground_truths = []
        self.scores = []
    
    def update(self, pred_boxes: np.ndarray, pred_scores: np.ndarray, 
               gt_boxes: np.ndarray, gt_labels: np.ndarray):
        """Update metrics with new predictions."""
        self.predictions.append({
            'boxes': pred_boxes,
            'scores': pred_scores
        })
        self.ground_truths.append({
            'boxes': gt_boxes,
            'labels': gt_labels
        })
    
    def compute_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of boxes."""
        # boxes: [N, 4] (x1, y1, x2, y2)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
        
        inter_area = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
        union_area = area1[:, None] + area2[None, :] - inter_area
        
        iou = inter_area / (union_area + 1e-6)
        return iou
    
    def compute_ap(self, recalls: np.ndarray, precisions: np.ndarray) -> float:
        """Compute Average Precision using 11-point interpolation."""
        # Add sentinel values
        recalls = np.concatenate([[0], recalls, [1]])
        precisions = np.concatenate([[0], precisions, [0]])
        
        # Compute precision envelope
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = max(precisions[i - 1], precisions[i])
        
        # Find points where recall changes
        indices = np.where(recalls[1:] != recalls[:-1])[0]
        
        # Compute AP
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
        return ap
    
    def evaluate(self) -> Dict[str, float]:
        """Compute all metrics."""
        results = {}
        
        for iou_thresh in self.iou_thresholds:
            tp = 0
            fp = 0
            fn = 0
            
            all_scores = []
            all_matches = []
            
            for pred, gt in zip(self.predictions, self.ground_truths):
                pred_boxes = pred['boxes']
                pred_scores = pred['scores']
                gt_boxes = gt['boxes']
                
                if len(pred_boxes) == 0:
                    fn += len(gt_boxes)
                    continue
                
                if len(gt_boxes) == 0:
                    fp += len(pred_boxes)
                    continue
                
                # Compute IoU
                iou_matrix = self.compute_iou(pred_boxes, gt_boxes)
                
                # Match predictions to ground truth
                matched_gt = set()
                for i, score in enumerate(pred_scores):
                    all_scores.append(score)
                    
                    if len(iou_matrix[i]) > 0:
                        best_iou = iou_matrix[i].max()
                        best_gt = iou_matrix[i].argmax()
                        
                        if best_iou >= iou_thresh and best_gt not in matched_gt:
                            tp += 1
                            matched_gt.add(best_gt)
                            all_matches.append(1)  # True positive
                        else:
                            fp += 1
                            all_matches.append(0)  # False positive
                    else:
                        fp += 1
                        all_matches.append(0)
                
                fn += len(gt_boxes) - len(matched_gt)
            
            # Compute precision-recall curve
            if len(all_scores) > 0:
                sorted_indices = np.argsort(all_scores)[::-1]
                sorted_matches = np.array(all_matches)[sorted_indices]
                
                cumsum_tp = np.cumsum(sorted_matches)
                cumsum_fp = np.cumsum(1 - sorted_matches)
                
                recalls = cumsum_tp / (tp + fn) if (tp + fn) > 0 else np.zeros_like(cumsum_tp)
                precisions = cumsum_tp / (cumsum_tp + cumsum_fp)
                
                ap = self.compute_ap(recalls, precisions)
                
                results[f'AP@{iou_thresh}'] = ap
                results[f'Precision@{iou_thresh}'] = precisions[-1] if len(precisions) > 0 else 0
                results[f'Recall@{iou_thresh}'] = recalls[-1] if len(recalls) > 0 else 0
            else:
                results[f'AP@{iou_thresh}'] = 0
                results[f'Precision@{iou_thresh}'] = 0
                results[f'Recall@{iou_thresh}'] = 0
        
        # mAP
        results['mAP'] = np.mean([results[f'AP@{t}'] for t in self.iou_thresholds])
        
        return results


class ViolenceDetectionMetrics:
    """
    Metrics specific to violence detection task.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.labels = []
        self.scores = []
    
    def update(self, pred_label: int, true_label: int, confidence: float):
        self.predictions.append(pred_label)
        self.labels.append(true_label)
        self.scores.append(confidence)
    
    def evaluate(self) -> Dict[str, float]:
        """Compute violence detection metrics."""
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)
        scores = np.array(self.scores)
        
        # Accuracy
        accuracy = (predictions == labels).mean()
        
        # Precision, Recall, F1 for violence class
        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()
        tn = ((predictions == 0) & (labels == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # AUC-ROC (if scores available)
        try:
            from sklearn.metrics import roc_auc_score
            auc_roc = roc_auc_score(labels, scores)
        except:
            auc_roc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'auc_roc': auc_roc
        }


class PerformanceMetrics:
    """
    System performance metrics: FPS, latency, memory usage.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.inference_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []
        self.memory_usage = []
    
    def update(self, inference_time: float, preprocessing_time: float = 0,
               postprocessing_time: float = 0, memory_mb: float = 0):
        self.inference_times.append(inference_time)
        self.preprocessing_times.append(preprocessing_time)
        self.postprocessing_times.append(postprocessing_time)
        self.memory_usage.append(memory_mb)
    
    def evaluate(self) -> Dict[str, float]:
        """Compute performance metrics."""
        inference_times = np.array(self.inference_times)
        
        return {
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'std_inference_time_ms': np.std(inference_times) * 1000,
            'min_inference_time_ms': np.min(inference_times) * 1000,
            'max_inference_time_ms': np.max(inference_times) * 1000,
            'fps': 1.0 / np.mean(inference_times),
            'avg_memory_mb': np.mean(self.memory_usage) if self.memory_usage else 0,
            'throughput': len(self.inference_times) / sum(self.inference_times) if sum(self.inference_times) > 0 else 0
        }


class AblationStudy:
    """
    Framework for ablation studies.
    """
    
    def __init__(self, base_config: Dict):
        self.base_config = base_config
        self.results = {}
    
    def run_ablation(self, model_builder: Callable, test_loader: DataLoader,
                     component_name: str, enabled: bool = True) -> Dict:
        """Run ablation for a specific component."""
        config = self.base_config.copy()
        config[component_name] = enabled
        
        model = model_builder(**config)
        evaluator = ModelEvaluator(model, test_loader)
        results = evaluator.evaluate()
        
        return results
    
    def run_full_ablation(self, model_builder: Callable, test_loader: DataLoader,
                         components: List[str]) -> Dict[str, Dict]:
        """Run full ablation study."""
        print("Running ablation study...")
        
        # Baseline: all components enabled
        print("Evaluating baseline (all components)...")
        baseline_results = self.run_ablation(model_builder, test_loader, 'baseline', True)
        self.results['baseline'] = baseline_results
        
        # Ablation: disable each component
        for component in components:
            print(f"Evaluating without {component}...")
            
            # Disable this component
            config = self.base_config.copy()
            config[component] = False
            
            model = model_builder(**config)
            evaluator = ModelEvaluator(model, test_loader)
            results = evaluator.evaluate()
            
            self.results[f'without_{component}'] = results
            
            # Compute impact
            impact = baseline_results['detection_metrics']['mAP'] - results['detection_metrics']['mAP']
            print(f"  Impact of {component}: {impact:.4f} mAP")
        
        return self.results
    
    def generate_report(self, output_path: str):
        """Generate ablation study report."""
        report = []
        report.append("=" * 80)
        report.append("Ablation Study Report")
        report.append("=" * 80)
        report.append("")
        
        # Baseline
        baseline = self.results.get('baseline', {})
        report.append("Baseline (All Components):")
        report.append(f"  mAP: {baseline.get('detection_metrics', {}).get('mAP', 0):.4f}")
        report.append(f"  Violence Accuracy: {baseline.get('violence_metrics', {}).get('accuracy', 0):.4f}")
        report.append("")
        
        # Ablation results
        report.append("Component Ablation:")
        for name, results in self.results.items():
            if name != 'baseline':
                component = name.replace('without_', '')
                mAP = results.get('detection_metrics', {}).get('mAP', 0)
                baseline_mAP = baseline.get('detection_metrics', {}).get('mAP', 0)
                impact = baseline_mAP - mAP
                report.append(f"  {component}:")
                report.append(f"    mAP: {mAP:.4f} (impact: {impact:+.4f})")
        
        report.append("")
        report.append("=" * 80)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\
'.join(report))
        
        print(f"Ablation report saved to {output_path}")


class ModelEvaluator:
    """
    Complete model evaluator.
    """
    
    def __init__(self, model: nn.Module, test_loader: DataLoader, device: str = 'cuda'):
        self.model = model
        self.test_loader = test_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Metrics
        self.detection_metrics = DetectionMetrics()
        self.violence_metrics = ViolenceDetectionMetrics()
        self.performance_metrics = PerformanceMetrics()
    
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """Run complete evaluation."""
        print("Running evaluation...")
        
        for batch in tqdm(self.test_loader, desc='Evaluating'):
            images = batch['image'].to(self.device)
            gt_boxes = batch['boxes'].cpu().numpy()
            gt_labels = batch['labels'].cpu().numpy()
            
            # Measure inference time
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            outputs = self.model(images)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_time = time.time() - start_time
            
            # Get predictions
            pred_logits = outputs['pred_logits'].cpu()
            pred_boxes = outputs['pred_boxes'].cpu()
            
            # Process each sample in batch
            for i in range(len(images)):
                # Get predictions for this sample
                scores, labels = pred_logits[i].softmax(-1).max(-1)
                boxes = pred_boxes[i]
                
                # Filter by confidence
                keep = scores > 0.5
                boxes = boxes[keep].numpy()
                scores = scores[keep].numpy()
                labels = labels[keep].numpy()
                
                # Update detection metrics
                self.detection_metrics.update(boxes, scores, gt_boxes[i], gt_labels[i])
                
                # Update violence metrics (if applicable)
                if len(labels) > 0:
                    pred_violence = (labels == 1).any()
                    true_violence = (gt_labels[i] == 1).any()
                    self.violence_metrics.update(
                        int(pred_violence), int(true_violence), 
                        scores.max() if len(scores) > 0 else 0
                    )
            
            # Update performance metrics
            self.performance_metrics.update(inference_time / len(images))
        
        # Compute final metrics
        results = {
            'detection_metrics': self.detection_metrics.evaluate(),
            'violence_metrics': self.violence_metrics.evaluate(),
            'performance_metrics': self.performance_metrics.evaluate()
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print evaluation results."""
        print("\
" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        
        print("\
Detection Metrics:")
        for key, value in results['detection_metrics'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\
Violence Detection Metrics:")
        for key, value in results['violence_metrics'].items():
            print(f"  {key}: {value:.4f}")
        
        print("\
Performance Metrics:")
        for key, value in results['performance_metrics'].items():
            if 'time' in key:
                print(f"  {key}: {value:.2f} ms")
            elif 'memory' in key:
                print(f"  {key}: {value:.2f} MB")
            else:
                print(f"  {key}: {value:.2f}")
        
        print("=" * 80)


def main():
    """Main evaluation function."""
    # Load model
    print("Loading SAF-DETR model...")
    model = build_saf_detr(
        num_classes=1,
        hidden_dim=256,
        num_queries=300,
        use_adaptive_intelligence=True,
        use_feature_enhancement=True,
        use_temporal_memory=True,
        use_novel_pipeline=True
    )
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/saf_detr/best.pth'
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create test dataset (placeholder)
    from torch.utils.data import DataLoader
    from training.train import SurveillanceDataset
    
    test_dataset = SurveillanceDataset(
        data_root='data/test',
        annotation_file='data/test/annotations.json',
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    evaluator = ModelEvaluator(model, test_loader)
    results = evaluator.evaluate()
    evaluator.print_results(results)
    
    # Save results
    output_path = 'evaluation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\
Results saved to {output_path}")


if __name__ == "__main__":
    main()
