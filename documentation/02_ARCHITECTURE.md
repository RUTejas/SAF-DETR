# SAF-DETR — Detailed Architecture

## 1. Top-Level Pipeline
```
                  ┌──────────────────────────────────────────────┐
                  │          SAF-DETR Surveillance Stack         │
                  └──────────────────────────────────────────────┘

Camera / CCTV / Recorded Video
            │
            ▼
  ┌──────────────────────┐
  │ OpenCV Video         │  — frame extraction, timestamp sync,
  │ Intelligence Module  │    ROI, frame quality stats           │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ Adaptive Frame       │  (Module 1)                          │
  │ Intelligence (AFI)   │  regime∈{low-light, blur, compress,   │
  └──────────┬───────────┘   clear} → specialised enhancer
             ▼
  ┌──────────────────────┐
  │ Surveillance         │  (Module 2)                          │
  │ Feature Enhancement  │  Human-Priority Attention             │
  │ Network (SFEN)       │  + Crowd Region Attention            │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ Transformer Backbone │  ResNet-50 / HGNet-v2 — pretrained   │
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │ Human-Centric        │  (Module 3)                          │
  │ Detection Decoder    │  Class-prior queries,                │
  └──────────┬───────────┘  small-human optimised heads
             ▼
  ┌──────────────────────┐
  │ Temporal Detection   │  (Module 4)                          │
  │ Memory (TDM)         │  Query-conditioned memory bank       │
  └──────────┬───────────┘  — velocity/acceleration features
             ▼
  Behaviour-Aware Output  ────────────────────────────────────┐
             │                                                │
             ▼                                                │
  Multi-Person Tracking (ByteTrack / OC-SORT style)           │
             │                                                │
             ▼                                                │
  Pose Estimation (RTMPose / HRNet) ─┐                       │
             │                       │                       │
             ▼                       ▼                       ▼
  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
  │ Human Interaction  │  │ Behaviour Network  │  │ Dynamic Threat     │
  │ Graph              │  │ (VideoSwin-T)      │  │ Index              │
  │ (nodes/edges)      │  │                    │  │                    │
  └─────────┬──────────┘  └─────────┬──────────┘  └─────────┬──────────┘
            └───────────────────────┴──────────────────────┘
                                     │
                                     ▼
                          Alert System / Dashboard
```

## 2. Mathematical Formulation

### 2.1 Adaptive Frame Intelligence
Given frame $I_t \in \mathbb{R}^{H \times W \times 3}$, a quality head $f_Q$ produces regime logits $\mathbf{z}_t \in \mathbb{R}^4$ with softmax $\pi_t = \mathrm{softmax}(\mathbf{z}_t)$. A mixture-of-experts is applied:

$$\tilde I_t = \sum_{k=1}^{4} \pi_t^{(k)} \cdot E_k(I_t)$$

where $E_k$ are differentiable enhancement operators:
- $E_1$: brightness + gamma correction (low-light)
- $E_2$: Wiener-deconvolution sharpening (blur)
- $E_3$: JPEG-block-artifact reduction (compression)
- $E_4$: identity (clear)

Training supervision: $\mathcal{L}_Q = \mathrm{CE}(\pi_t, y_t)$ with $y_t$ from a pseudo-labeller.

### 2.2 Surveillance Feature Enhancement
Backbone produces multi-scale feature $\mathcal{F} = \{F_3, F_4, F_5\}$. SFEN computes a human-priority map:

$$M_h = \sigma\!\left(\mathrm{Conv}_{1\times1}\!\left(\mathrm{AGate}\!\left(\mathrm{avg}(\mathcal{F}), \mathrm{max}(\mathcal{F})\right)\right)\right)$$

Enhanced feature: $F'_l = F_l \odot (1 + \alpha M_h) + \beta \cdot \mathrm{PA}(F_l)$ where $\mathrm{PA}$ is a *pose-prior* attention derived from an auxiliary head trained with keypoint pseudo-labels. $\alpha, \beta$ are learnable scalars.

### 2.3 Human-Centric Detection Decoder
We replace generic DETR queries with **human-prior anchor queries**:

$$Q = \{q^{(h)}_{\text{person}}, q^{(h)}_{\text{group}}, q^{(h)}_{\text{interaction}}, q^{(h)}_{\text{small}}\}$$

Decoding applies deformable cross-attention (DAModule) but biases the reference points toward the downweighted detection areas — empirically improving small-human recall.

### 2.4 Temporal Detection Memory
Memory $\mathcal{M} \in \mathbb{R}^{N \times d}$ stores per-track embeddings. At frame $t$:

$$\mathbf{m}_i^{t} = \phi\!\left([\,z_i^{t-1} \,\|\, z_i^{t} \,]\right)$$

Query-conditioned cross-attention reads from $\mathcal{M}$:

$$z_i^{t,\,\text{enh}} = z_i^t + \mathrm{softmax}(q_i K^\top/\sqrt{d})V$$

Velocity feature $\mathbf{v}_i^t = z_i^t - z_i^{t-1}$ is concatenated to the behaviour head input.

### 2.5 Behaviour-Aware Composite Loss

$$\mathcal{L} = \lambda_1 \mathcal{L}_\text{det} + \lambda_2 \mathcal{L}_\text{track} + \lambda_3 \mathcal{L}_\text{interaction} + \lambda_4 \mathcal{L}_\text{temp}$$

- $\mathcal{L}_\text{det}$: standard DETR loss (Hungarian-matched).
- $\mathcal{L}_\text{track}$: identity-consistency loss between matched detections across frames.
- $\mathcal{L}_\text{interaction}$: hinge loss encouraging high embedding similarity for interacting person pairs (positive) and repulsion for non-interacting.
- $\mathcal{L}_\text{temp}$: $\ell_2$ smoothness on box positions across $k$ frames.

Default weights $(\lambda_1,\lambda_2,\lambda_3,\lambda_4) = (1.0, 0.5, 0.25, 0.1)$.

### 2.6 Dynamic Threat Index

For a window $W$ of frames and a person $i$:

$$\mathrm{Threat}_i = w_1 P_i^\text{action} + w_2 N(\|\mathbf{v}_i\|) + w_3 R_i^\text{interact} + w_4 C_i^\text{crowd}$$

$w_k$ calibrated on validation. Aggregated scene-level threat is the max-pooled threat over all persons.

## 3. Per-Stage Tensor Shape Sketch

| Stage | Input | Output |
|-------|-------|--------|
| Frame | $1\times 3\times H\times W$ | $1\times 3\times H\times W$ |
| AFI | $1\times 3\times 640\times 640$ | $1\times 3\times 640\times 640$ |
| Backbone (ResNet-50) | $1\times 3\times 640\times 640$ | $\{256\times 80, 512\times 40, 1024\times 20\}$ |
| SFEN | multi-scale | same + attention maps |
| Decoder | memory queries × layers | 300 predictions |
| TDM | past + current | enhanced embeddings |
| Threat | per-frame scores | scalar threat ∈ [0, 100] |

## 4. Hardware Budget (RTX 4070 Laptop, 8 GB)

- Training input resolution: 640 × 640.
- FP16 mixed precision.
- Batch size 4 with gradient accumulation 4 (effective 16).
- Peak VRAM ≤ 6.5 GB → safely under 8 GB.
- Inference target ≥ 25 FPS at 640.
