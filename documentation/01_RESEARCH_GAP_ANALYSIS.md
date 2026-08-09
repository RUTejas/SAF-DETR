# Research Gap Analysis — Towards Surveillance-Adaptive Detection

## 1. Problem Context
Real-world surveillance footage exhibits **low illumination, motion blur, aggressive H.264/H.265 compression, dynamic crowds, occlusion, and far-camera small human targets**. Standard detectors (YOLO, Faster R-CNN, vanilla RT-DETR) are trained on curated datasets (COCO, Objects365) where persons are large, well-lit, and isolated. Deploying them directly on CCTV produces:

- Missed detections of small humans (<32 px)
- ID switches in dense crowds
- Inability to predict violent behaviour before the peak
- No notion of *context* (proximity, approach velocity, body orientation)

## 2. Critical Gaps in Literature

| ID | Gap | Why it matters |
|----|-----|----------------|
| G1 | Detectors are **appearance-only** | They ignore the *surveillance domain shift* (low light, compression artefacts). |
| G2 | **Frame-independent** inference | No temporal consistency → jittered boxes; no pre-escalation cue modelling. |
| G3 | Behaviour recognition is **black-box** end-to-end video models | They do not expose interpretable interaction graphs. |
| G4 | Violence detection starts **after** the action | Threat escalation is not anticipated. |
| G5 | Person detectors treat humans as a **generic class** | They do not score *human-priority* features (silhouette completeness, pose visibility). |
| G6 | No unified **threat index** merges detection confidence, motion, interaction, and crowd risk. |

## 3. Proposed Research Contributions

We propose **SAF-DETR (Surveillance Adaptive Feature-Enhanced Detection Transformer)** that addresses G1–G6 simultaneously.

### Contribution 1 — Adaptive Frame Intelligence (G1)
A learned controller classifies the frame degradation regime (low-light, blur, compression, clear) and routes it through specialised *enhancement* operators *before* the backbone. Replaces hand-tuned OpenCV pipelines.

### Contribution 2 — Human-Priority Feature Enhancement (G5)
A multi-branch attention block that amplifies human silhouette and crowd-region features in the backbone output. Injects class-priority weights directly into the encoder–decoder cross-attention.

### Contribution 3 — Temporal Detection Memory (G2)
A query-conditioned memory bank that aggregates detection embeddings across frames. Provides implicit tracking, stabilises boxes, and exposes *velocity/acceleration* features for the threat module.

### Contribution 4 — Behaviour-Aware Composite Loss (G2, G4)
Training objective that jointly supervises detection, tracking consistency, interaction awareness, and temporal smoothness — explicitly shaped for surveillance dynamics.

### Contribution 5 — Human Interaction Graph + Dynamic Threat Index (G3, G4, G6)
A spatio-temporal graph of persons (nodes) and interactions (edges: distance, relative velocity, mutual orientation, pose similarity). Threat score is a *calibrated* sum, not an opaque end-to-end prediction.

## 4. Hypotheses

- **H1:** Adaptive pre-processing + human-priority enhancement improves mAP on degraded CCTV frames by ≥3 % absolute over RT-DETR baseline.
- **H2:** Temporal Memory reduces ID-switch rate by ≥30 % vs. vanilla RT-DETR + SORT/ByteTrack in dense crowd benchmarks.
- **H3:** The Dynamic Threat Index predicts violent escalation ≥0.5 s earlier than a frame-classifier baseline (RWF-2000-style evaluation).
- **H4:** Composite Behaviour-Aware Loss improves detection stability (lower box jitter) without sacrificing detection mAP.

## 5. Expected Reviewer Critiques

| Reviewer concern | Mitigation |
|------------------|------------|
| "RT-DETR is heavy — how is FPS ≥25?" | TensorRT-style FP16/INT8 export path + Win/ CUDA stream plumbing; report both training-time and inference-time numbers. |
| "You compare to RT-DETR but not DETR4Var / D-FINE." | Include ablations vs YOLOv8/v11, Faster R-CNN, and RT-DETR; additional comparisons reserved for camera-ready. |
| "How is the Adaptive controller different from CLAHE?" | Controller is *learned*, differentiable through the enhancement operators, and selected by a 4-way classifier — not a fixed rule. |
| "Behaviour loss could destabilise detection." | Ablation in Appendix: turn each term on/off, measure mAP variance. |
| "Threat index weights are hand-tuned." | Weights calibrated on a held-out validation set with grid + Bayesian search; report sensitivity curves. |

## 6. Deliverables Index

1. `documentation/02_ARCHITECTURE.md` — Full architecture diagram and math.
2. `documentation/03_DEVELOPMENT_ROADMAP.md` — 30-day plan with milestones.
3. `documentation/04_DATASETS.md` — Dataset selection, augmentation policy.
4. `documentation/05_IEEE_PAPER_DRAFT.md` — Paper skeleton ready to expand.
5. Code under `models/`, `preprocessing/`, `tracking/`, `pose/`, `behaviour/`, `threat_engine/`, `dashboard/`, `experiments/`, `training/`, `evaluation/`.

---
*This document is the canonical research gap definition for SAF-DETR.*
