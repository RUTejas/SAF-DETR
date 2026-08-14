"""
SAF-DETR Training Pipeline
===========================

Complete training pipeline with mixed precision, distributed training,
and advanced augmentation strategies.

Author: SAF-DETR Research Team
Date: 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from tqdm import tqdm
import wandb

# Import model
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.SAF_DETR.complete_model import build_saf_detr


class SurveillanceDataset(Dataset):
    """
    Dataset loader for surveillance videos.
    Supports RWF-2000, UCF-Crime, and custom datasets.
    """
    
    def __init__(self,
                 data_root: str,
                 annotation_file: str,
                 img_size: int = 640,
                 mode: str = 'train',
                 temporal_window: int = 5):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.mode = mode
        self.temporal_window = temporal_window
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.samples = self.annotations['samples']
        
        # Transforms
        self.transforms = self._build_transforms()
        
    def _build_transforms(self):
        """Build data augmentation transforms."""
        if self.mode == 'train':
            return T.Compose([
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = self.data_root / sample['image_path']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Apply transforms
        image = self.transforms(T.ToPILImage()(image))
        
        # Prepare targets
        boxes = torch.tensor(sample['boxes'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.long)
        
        # Normalize boxes to [0, 1]
        boxes[:, [0, 2]] /= self.img_size
        boxes[:, [1, 3]] /= self.img_size
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': sample['image_id']
        }


class Trainer:
    """
    SAF-DETR Trainer with mixed precision and distributed training support.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        
        # Scheduler
        self.scheduler = self._build_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if config.get('mixed_precision', True) else None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.use_wandb = config.get('use_wandb', False)
        if self.use_wandb:
            wandb.init(project='SAF-DETR', config=config)
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
    def _build_optimizer(self):
        """Build optimizer with weight decay."""
        param_dicts = [
            {'params': [p for n, p in self.model.named_parameters() 
                       if 'backbone' not in n and p.requires_grad]},
            {'params': [p for n, p in self.model.named_parameters() 
                       if 'backbone' in n and p.requires_grad], 
             'lr': self.config['lr_backbone']}
        ]
        
        optimizer = optim.AdamW(
            param_dicts,
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.config['lr_drop'],
            gamma=0.1
        )
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['image'].to(self.device)
            targets = {
                'boxes': batch['boxes'].to(self.device),
                'labels': batch['labels'].to(self.device)
            }
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(images, targets=targets)
                    loss = outputs['loss']
            else:
                outputs = self.model(images, targets=targets)
                loss = outputs['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            
            for key, value in outputs.items():
                if 'loss' in key and isinstance(value, torch.Tensor):
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Logging
            if self.global_step % self.config.get('log_interval', 100) == 0:
                self._log_training(loss.item(), loss_components, batch_idx)
            
            self.global_step += 1
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        for key in loss_components:
            loss_components[key] /= len(self.train_loader)
        
        return {'train_loss': avg_loss, **loss_components}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        loss_components = {}
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            images = batch['image'].to(self.device)
            targets = {
                'boxes': batch['boxes'].to(self.device),
                'labels': batch['labels'].to(self.device)
            }
            
            outputs = self.model(images, targets=targets)
            loss = outputs['loss']
            
            total_loss += loss.item()
            
            for key, value in outputs.items():
                if 'loss' in key and isinstance(value, torch.Tensor):
                    if key not in loss_components:
                        loss_components[key] = 0.0
                    loss_components[key] += value.item()
        
        # Average losses
        avg_loss = total_loss / len(self.val_loader)
        for key in loss_components:
            loss_components[key] /= len(self.val_loader)
        
        return {'val_loss': avg_loss, **loss_components}
    
    def _log_training(self, loss: float, loss_components: Dict, batch_idx: int):
        """Log training metrics."""
        if self.use_wandb:
            log_dict = {'train/loss': loss, 'train/step': self.global_step}
            for key, value in loss_components.items():
                log_dict[f'train/{key}'] = value / (batch_idx + 1)
            wandb.log(log_dict)
    
    def _log_validation(self, metrics: Dict):
        """Log validation metrics."""
        if self.use_wandb:
            log_dict = {f'val/{k}': v for k, v in metrics.items()}
            log_dict['val/epoch'] = self.epoch
            wandb.log(log_dict)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint with val_loss: {self.best_val_loss:.4f}")
        
        # Save periodic checkpoint
        if self.epoch % self.config.get('save_interval', 10) == 0:
            periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pth'
            torch.save(checkpoint, periodic_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Step scheduler
            self.scheduler.step()
            
            # Log
            print(f"\
Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            
            self._log_validation(val_metrics)
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            self.save_checkpoint(is_best=is_best)
        
        print("Training complete!")
        if self.use_wandb:
            wandb.finish()


def main():
    """Main training function."""
    # Configuration
    config = {
        'lr': 1e-4,
        'lr_backbone': 1e-5,
        'weight_decay': 1e-4,
        'lr_drop': 20,
        'batch_size': 4,
        'num_epochs': 50,
        'mixed_precision': True,
        'use_wandb': False,
        'log_interval': 100,
        'save_interval': 10,
        'checkpoint_dir': 'checkpoints/saf_detr',
        
        # Model config
        'num_classes': 1,
        'hidden_dim': 256,
        'num_queries': 300,
    }
    
    # Build model
    print("Building SAF-DETR model...")
    model = build_saf_detr(
        num_classes=config['num_classes'],
        hidden_dim=config['hidden_dim'],
        num_queries=config['num_queries'],
        use_adaptive_intelligence=True,
        use_feature_enhancement=True,
        use_temporal_memory=True,
        use_novel_pipeline=True
    )
    
    # Create dummy datasets (replace with actual data loading)
    print("Creating datasets...")
    train_dataset = SurveillanceDataset(
        data_root='data/train',
        annotation_file='data/train/annotations.json',
        mode='train'
    )
    
    val_dataset = SurveillanceDataset(
        data_root='data/val',
        annotation_file='data/val/annotations.json',
        mode='val'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == "__main__":
    main()
