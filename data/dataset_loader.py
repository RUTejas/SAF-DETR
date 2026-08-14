"""
SAF-DETR: Mixed Surveillance Dataset Pipeline
==============================================

Supports mixed surveillance datasets including:
- RWF-2000 (Video / .npy with RGB + Optical Flow)
- UCF-Crime / CCTV-Fights / Hockey (Video clips organized by class)
- COCO / YOLO annotated surveillance images
- Custom user mixed folder structures (Fight/NonFight, Violence/Normal)
- Synthetic Surveillance Generator for zero-data testing & validation

Author: SAF-DETR Research Team
Date: 2026
"""

import os
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class MixedSurveillanceDataset(Dataset):
    """
    Unified dataset loader handling mixed surveillance formats:
    1. Video folders (e.g. data/train/Fight/*.mp4, data/train/NonFight/*.avi)
    2. RWF-2000 .npy preprocessed files [frames, H, W, 5]
    3. Image folders with JSON / YOLO annotations
    4. Auto-generated synthetic surveillance samples if dataset is empty
    """

    def __init__(self,
                 data_root: str,
                 mode: str = 'train',
                 img_size: int = 640,
                 seq_len: int = 16,
                 generate_synthetic_if_empty: bool = True):
        self.data_root = Path(data_root)
        self.mode = mode
        self.img_size = img_size
        self.seq_len = seq_len
        self.samples = []

        self._discover_dataset(generate_synthetic_if_empty)
        self.transforms = self._build_transforms()

    def _discover_dataset(self, allow_synthetic: bool):
        """Scans directory for various dataset layouts."""
        if not self.data_root.exists():
            self.data_root.mkdir(parents=True, exist_ok=True)

        # 1. Check for video files in subfolders (Fight / NonFight or Violence / Normal)
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        for cat_dir in self.data_root.glob('*'):
            if cat_dir.is_dir():
                label_name = cat_dir.name.lower()
                is_violent = 1 if any(w in label_name for w in ['fight', 'violence', 'crime', 'assault']) else 0
                for vfile in cat_dir.glob('*.*'):
                    if vfile.suffix.lower() in video_extensions:
                        self.samples.append({
                            'type': 'video',
                            'path': str(vfile),
                            'label': is_violent,
                            'category': cat_dir.name
                        })

        # 2. Check for RWF-2000 .npy files
        for npy_file in self.data_root.rglob('*.npy'):
            label = 1 if 'fight' in str(npy_file).lower() else 0
            self.samples.append({
                'type': 'npy',
                'path': str(npy_file),
                'label': label,
                'category': 'Violence' if label == 1 else 'Non-Violence'
            })

        # 3. Check for image annotations json
        anno_files = list(self.data_root.rglob('*annotations*.json'))
        for af in anno_files:
            try:
                with open(af, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if 'samples' in data:
                    for s in data['samples']:
                        self.samples.append({
                            'type': 'annotated_image',
                            'path': str(self.data_root / s['image_path']),
                            'boxes': s.get('boxes', []),
                            'labels': s.get('labels', [1]),
                            'label': 1 if any(l == 1 for l in s.get('labels', [0])) else 0
                        })
            except Exception:
                pass

        # 4. If empty and allowed, generate structured synthetic surveillance scenes
        if len(self.samples) == 0 and allow_synthetic:
            print(f"[{self.mode.upper()}] No raw videos found in {self.data_root}. Creating synthetic surveillance dataset index...")
            for i in range(40 if self.mode == 'train' else 10):
                is_fight = (i % 2 == 1)
                self.samples.append({
                    'type': 'synthetic',
                    'id': f"synth_{self.mode}_{i:03d}",
                    'label': 1 if is_fight else 0,
                    'category': 'Fighting' if is_fight else 'Normal'
                })

        print(f"[{self.mode.upper()}] Loaded {len(self.samples)} samples from {self.data_root}")

    def _build_transforms(self):
        """Augmentation for training and normalization for validation."""
        if self.mode == 'train':
            return T.Compose([
                T.ToPILImage(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def _generate_synthetic_frame(self, is_violent: bool) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """Generates realistic synthetic surveillance frame with bounding boxes."""
        h, w = self.img_size, self.img_size
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Dark surveillance background gradient
        top_color = np.array([20, 25, 35], dtype=np.uint8)
        bot_color = np.array([10, 15, 20], dtype=np.uint8)
        for y in range(h):
            alpha = y / h
            frame[y, :] = (top_color * (1 - alpha) + bot_color * alpha).astype(np.uint8)

        # Add noise
        noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        boxes = []
        labels = []
        num_persons = np.random.randint(2, 6) if not is_violent else np.random.randint(2, 4)

        for i in range(num_persons):
            if is_violent and i < 2:
                # Two persons close together (fighting interaction)
                bx = int(w * 0.4 + (i * 0.12 + np.random.uniform(-0.02, 0.02)) * w)
                by = int(h * 0.45 + np.random.uniform(-0.05, 0.05) * h)
                bw = int(w * np.random.uniform(0.1, 0.15))
                bh = int(bw * 2.2)
                labels.append(1)  # Violence
            else:
                bx = int(w * np.random.uniform(0.1, 0.8))
                by = int(h * np.random.uniform(0.35, 0.65))
                bw = int(w * np.random.uniform(0.08, 0.12))
                bh = int(bw * 2.2)
                labels.append(0)  # Normal

            x1 = max(0, bx - bw // 2)
            y1 = max(0, by - bh // 2)
            x2 = min(w, bx + bw // 2)
            y2 = min(h, by + bh // 2)

            # Draw person silhouette
            cv2.ellipse(frame, (bx, y1 + int(bh * 0.15)), (bw // 4, bh // 8), 0, 0, 360, (180, 190, 200), -1)
            cv2.rectangle(frame, (x1, y1 + int(bh * 0.2)), (x2, y2), (100, 110, 130), -1)

            # Normalized boxes [cx, cy, w, h]
            boxes.append([(x1 + x2) / (2 * w), (y1 + y2) / (2 * h), (x2 - x1) / w, (y2 - y1) / h])

        return frame, boxes, labels

    def _load_video_sample(self, video_path: str, is_violent: int):
        """Loads and samples frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return self._generate_synthetic_frame(bool(is_violent))

        indices = np.linspace(0, max(0, total_frames - 1), self.seq_len, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
            frames.append(frame)
        cap.release()

        # Target middle frame for detection
        rep_frame = frames[len(frames) // 2]
        h, w = self.img_size, self.img_size
        boxes = [[0.5, 0.5, 0.2, 0.4]]
        labels = [is_violent]
        return rep_frame, boxes, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        stype = sample.get('type', 'synthetic')

        if stype == 'synthetic':
            frame, boxes, labels = self._generate_synthetic_frame(bool(sample['label']))
        elif stype == 'video':
            frame, boxes, labels = self._load_video_sample(sample['path'], sample['label'])
        elif stype == 'annotated_image':
            img = cv2.imread(sample['path'])
            if img is None:
                frame, boxes, labels = self._generate_synthetic_frame(bool(sample['label']))
            else:
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                boxes = sample.get('boxes', [[0.5, 0.5, 0.2, 0.4]])
                labels = sample.get('labels', [sample.get('label', 0)])
        else:
            frame, boxes, labels = self._generate_synthetic_frame(bool(sample.get('label', 0)))

        # Convert image to tensor with transforms
        img_tensor = self.transforms(frame)

        # Pad boxes to fixed shape for clean batching
        max_boxes = 20
        padded_boxes = torch.zeros((max_boxes, 4), dtype=torch.float32)
        padded_labels = torch.zeros((max_boxes,), dtype=torch.long)

        num_valid = min(len(boxes), max_boxes)
        if num_valid > 0:
            padded_boxes[:num_valid] = torch.tensor(boxes[:num_valid], dtype=torch.float32)
            padded_labels[:num_valid] = torch.tensor(labels[:num_valid], dtype=torch.long)

        return {
            'image': img_tensor,
            'boxes': padded_boxes,
            'labels': padded_labels,
            'num_boxes': torch.tensor(num_valid, dtype=torch.long),
            'violence_label': torch.tensor(sample.get('label', 0), dtype=torch.float32)
        }


def get_mixed_dataloader(data_root: str,
                         mode: str = 'train',
                         batch_size: int = 4,
                         img_size: int = 640,
                         num_workers: int = 0) -> DataLoader:
    """Builds a high-performance DataLoader for mixed datasets."""
    dataset = MixedSurveillanceDataset(data_root, mode=mode, img_size=img_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == '__main__':
    print("Testing MixedSurveillanceDataset...")
    ds = MixedSurveillanceDataset('data/train', mode='train', img_size=640)
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"Image tensor shape: {sample['image'].shape}")
    print(f"Boxes shape: {sample['boxes'].shape}")
    print(f"Violence label: {sample['violence_label']}")
    print("Dataset test passed successfully!")
