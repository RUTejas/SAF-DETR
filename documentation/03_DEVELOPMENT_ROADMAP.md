# 30-Day Development Roadmap — SAF-DETR

## Week 1 — Foundations
- **D1–D2** Environment setup (CUDA, PyTorch, MMCV/MMDet, ByteTrack, RTMPose).
- **D3** Dataset curation: CrowdHuman, MOT20, RWF-2000, custom CCTV clips.
- **D4** `OpenCV Video Intelligence Module` — frame sampler, quality probe.
- **D5** `AdaptiveFrameController` — regime classifier + enhancement branches.
- **D6–D7** Baseline detectors wired up: YOLOv8/v11, RT-DETR-R50, Faster R-CNN.

## Week 2 — SAF-DETR Modules
- **D8–D9** Surveillance Feature Enhancement Network.
- **D10** Human-Priority queries + Human-Centric Decoder.
- **D11** Temporal Detection Memory (differentiable memory bank).
- **D12** Behaviour-Aware Loss (4-term composite).
- **D13–D14** Training loop (FP16, EMA, gradient accumulation).

## Week 3 — Behaviour & Threat Intelligence
- **D15** Multi-person tracking (ByteTrack/OC-SORT integration).
- **D16** Pose model wrapper.
- **D17** Human Interaction Graph (PyTorch-Geometric or handcrafted NumPy).
- **D18** Behaviour Recognition — VideoSwin-T fine-tuned on RWF-2000.
- **D19** Dynamic Threat Index + calibration on val set.
- **D20–D21** Real-time demo + dashboard (Streamlit).

## Week 4 — Experiments & Paper
- **D22–D23** Run ablations (each module on/off).
- **D24–D25** Comparison tables vs baselines (mAP, FPS, ID-switch).
- **D26–D27** IEEE paper draft (Abstract → Experiments → Conclusion).
- **D28–D29** Plots, confusion matrices, qualitative figures.
- **D30** Final integration test + presentation pack.

## Milestone Targets
| Milestone | Deliverable | Verification |
|-----------|-------------|--------------|
| M1 | Baseline detection on CrowdHuman val | mAP ≥ 0.45 (reproduces RT-DETR baseline) |
| M2 | SAF-DETR on CrowdHuman val | mAP ≥ baseline + 1.5 % |
| M3 | Behaviour classifier on RWF-2000 | accuracy ≥ 0.85 |
| M4 | Real-time demo | ≥ 25 FPS on RTX 4070 |
| M5 | Threat index calibrated | AUC ≥ 0.80 for escalation prediction |
