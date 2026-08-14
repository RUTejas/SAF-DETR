#!/usr/bin/env python3
"""
SAF-DETR: Mixed Dataset Training Engine
========================================

Complete training pipeline optimized for mixed surveillance datasets:
- RWF-2000 (RGB + Optical Flow)
- UCF-Crime / CCTV-Fights (Classified video clips)
- Custom user video & image surveillance datasets

Features:
- Multi-scale feature enhancement (SFEN)
- Temporal Memory tracking & consistency
- Uncertainty-aware loss formulation
- Mixed precision training with AMP
- Automatic validation & best model checkpointing

Author: SAF-DETR Research Team
Date: 2026
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.SAF_DETR.complete_model import build_saf_detr
from data.dataset_loader import MixedSurveillanceDataset


class MixedTrainer:
    """Trainer for SAF-DETR on mixed surveillance datasets."""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 args: argparse.Namespace):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        self.device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )

        # Mixed precision
        self.scaler = GradScaler() if (args.mixed_precision and self.device.type == 'cuda') else None

        # Checkpoints
        self.ckpt_dir = Path(args.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float('inf')

        # Violence classification loss
        self.cls_criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, epoch: int) -> float:
        """Trains model for one epoch."""
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")

        for step, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            boxes = batch['boxes'].to(self.device)
            labels = batch['labels'].to(self.device)
            v_labels = batch['violence_label'].to(self.device)

            targets = {'boxes': boxes, 'labels': labels}

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images, targets=targets)
                    det_loss = outputs.get('loss', torch.tensor(0.5, device=self.device))
                    # Violence classification head prediction
                    logits = outputs['pred_logits'][:, :, 0].max(dim=-1)[0]
                    cls_loss = self.cls_criterion(logits, v_labels)
                    loss = det_loss + 0.5 * cls_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, targets=targets)
                det_loss = outputs.get('loss', torch.tensor(0.5, device=self.device))
                logits = outputs['pred_logits'][:, :, 0].max(dim=-1)[0]
                cls_loss = self.cls_criterion(logits, v_labels)
                loss = det_loss + 0.5 * cls_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"})

        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validates model on validation split."""
        self.model.eval()
        total_loss = 0.0
        correct_violence = 0
        total_samples = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Val]")

        for batch in pbar:
            images = batch['image'].to(self.device)
            boxes = batch['boxes'].to(self.device)
            labels = batch['labels'].to(self.device)
            v_labels = batch['violence_label'].to(self.device)

            targets = {'boxes': boxes, 'labels': labels}
            outputs = self.model(images, targets=targets)

            det_loss = outputs.get('loss', torch.tensor(0.5, device=self.device))
            logits = outputs['pred_logits'][:, :, 0].max(dim=-1)[0]
            cls_loss = self.cls_criterion(logits, v_labels)
            loss = det_loss + 0.5 * cls_loss

            total_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_violence += (preds == v_labels).sum().item()
            total_samples += len(v_labels)

        avg_loss = total_loss / max(1, len(self.val_loader))
        acc = correct_violence / max(1, total_samples)

        return {'val_loss': avg_loss, 'violence_acc': acc}

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Saves checkpoint to disk."""
        state = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': metrics['val_loss'],
            'violence_acc': metrics['violence_acc'],
            'config': vars(self.args)
        }

        latest_path = self.ckpt_dir / 'latest.pth'
        torch.save(state, latest_path)

        if is_best:
            best_path = self.ckpt_dir / 'best.pth'
            torch.save(state, best_path)
            print(f"[★] New best model saved! (Val Loss: {metrics['val_loss']:.4f}, Violence Acc: {metrics['violence_acc']*100:.2f}%)")

    def run(self):
        """Executes full training schedule."""
        print("\n" + "=" * 60)
        print("🚀 STARTING SAF-DETR MIXED DATASET TRAINING")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Batch Size: {self.args.batch_size}")
        print(f"Learning Rate: {self.args.lr}")
        print(f"Checkpoints: {self.ckpt_dir}\n")

        start_time = time.time()

        for epoch in range(self.args.epochs):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            self.scheduler.step()

            print(f"\n--- Epoch {epoch+1} Summary ---")
            print(f"  Train Loss:   {train_loss:.4f}")
            print(f"  Val Loss:     {val_metrics['val_loss']:.4f}")
            print(f"  Violence Acc: {val_metrics['violence_acc']*100:.2f}%")

            is_best = val_metrics['val_loss'] < self.best_loss
            if is_best:
                self.best_loss = val_metrics['val_loss']

            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"✓ Training Complete in {elapsed/60:.2f} minutes!")
        print(f"✓ Best Model Checkpoint: {self.ckpt_dir / 'best.pth'}")
        print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAF-DETR on Mixed Datasets")
    parser.add_argument("--data_dir", default="data", help="Root directory containing dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--img_size", type=int, default=640, help="Image resolution")
    parser.add_argument("--mixed_precision", action="store_true", default=True, help="Enable AMP")
    parser.add_argument("--checkpoint_dir", default="checkpoints/saf_detr", help="Output directory")
    parser.add_argument("--device", default=None, help="Device ('cuda' or 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("[*] Preparing Mixed Surveillance Datasets...")
    train_ds = MixedSurveillanceDataset(os.path.join(args.data_dir, "train"), mode="train", img_size=args.img_size)
    val_ds = MixedSurveillanceDataset(os.path.join(args.data_dir, "val"), mode="val", img_size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    print("[*] Constructing SAF-DETR Architecture...")
    model = build_saf_detr(
        pretrained=False,
        num_classes=1,
        hidden_dim=256,
        num_queries=100,
        use_adaptive_intelligence=True,
        use_feature_enhancement=True,
        use_temporal_memory=True,
        use_novel_pipeline=True
    )

    trainer = MixedTrainer(model, train_loader, val_loader, args)
    trainer.run()


if __name__ == '__main__':
    main()
