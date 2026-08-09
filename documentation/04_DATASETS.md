# Datasets for SAF-DETR

## Detection & Tracking
| Dataset | Use | Why |
|---------|-----|-----|
| **CrowdHuman** (15 000 imgs, 339 k persons) | Detection training | Designed for crowded scenes, similar to CCTV |
| **MOT20** | Tracking training | Dense pedestrian scenarios |
| **VisDrone** (Det / Track) | Aerial surveillance | Small humans, far-camera regimes |

## Behaviour / Violence
| Dataset | Use |
|---------|-----|
| **RWF-2000** | Two-class violence detection (in repo: `data/RWF2000`) |
| **UCF-Crime** | 13-class anomalous behaviour |
| **Avenue Dataset** | Unusual behaviour |
| **ShanghaiTech Campus** | Anomaly detection |

## Pose (pre-trained)
- **COCO-WholeBody** for RTMPose/HRNet training data.

## Custom CCTV Augmentation Pipeline
To reduce the domain gap, on-the-fly we apply:
- Random gamma ∈ [1.5, 3.0] — simulate low light
- Motion blur with kernel 5–15 px
- JPEG compression quality ∈ [30, 70]
- Synthetic occlusions (random rectangles 5–20 % area)
- Color jitter (hue, saturation, brightness)

## Splits
- Train 70 %, Val 15 %, Test 15 % — stratified per dataset.
- Held-out CCTV subset reserved for final demo.

## Download Hints
- CrowdHuman: https://www.crowdhuman.org/
- MOT20: https://motchallenge.net/data/MOT20/
- VisDrone: http://aiskyeye.com/
- RWF-2000: already on disk at `data/RWF2000-Video-Database-for-Violence-Detection-master/`.
- UCF-Crime: https://www.crcv.ucf.edu/projects/real-world-anomaly-detection-in-surveillance-video/
- Avenue: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/normal.html
