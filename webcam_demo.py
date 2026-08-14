#!/usr/bin/env python3
"""
SAF-DETR: Webcam Demo Shortcut
==============================

Quick-start script to launch live camera inference immediately.
Usage:
    python webcam_demo.py
    python webcam_demo.py --source 0
    python webcam_demo.py --source video.mp4

Author: SAF-DETR Research Team
Date: 2026
"""

from live_predict import parse_args, LiveViolenceDetector

if __name__ == '__main__':
    args = parse_args()
    detector = LiveViolenceDetector(
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_thresh=args.thresh
    )
    detector.run(source=args.source)
