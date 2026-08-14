"""
SAF-DETR REST API
=================

FastAPI-based REST API for violence detection.
Deployable to free platforms like Render, Railway, Heroku.

Author: SAF-DETR Research Team
Date: 2026
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.SAF_DETR.complete_model import build_saf_detr

# Initialize FastAPI app
app = FastAPI(
    title="SAF-DETR API",
    description="Surveillance Violence Detection API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DetectionResult(BaseModel):
    """Detection result schema."""
    box: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

class ViolenceDetectionResponse(BaseModel):
    """API response schema."""
    success: bool
    detections: List[DetectionResult]
    violence_detected: bool
    violence_confidence: float
    num_persons: int
    inference_time_ms: float
    image_width: int
    image_height: int

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    version: str

@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model
    
    print("Loading SAF-DETR model...")
    model = build_saf_detr(
        num_classes=1,
        hidden_dim=256,
        num_queries=100,
        use_adaptive_intelligence=True,
        use_feature_enhancement=True,
        use_temporal_memory=False,
        use_novel_pipeline=True
    )
    
    # Load weights if available
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "saf_detr" / "best.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded trained model weights")
    else:
        print("Using randomly initialized model")
    
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

def preprocess_image(image_bytes: bytes, target_size: int = 640) -> tuple:
    """Preprocess image for model input."""
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_h, original_w = image.shape[:2]
    
    # Resize
    scale = target_size / max(original_h, original_w)
    new_h, new_w = int(original_h * scale), int(original_w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Pad to square
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # Normalize
    padded = padded.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    padded = (padded - mean) / std
    
    # To tensor
    tensor = torch.from_numpy(padded).permute(2, 0, 1).unsqueeze(0)
    
    return tensor, scale, original_w, original_h

def postprocess_predictions(outputs: dict, scale: float, threshold: float = 0.5):
    """Postprocess model outputs."""
    pred_logits = outputs['pred_logits'][0]
    pred_boxes = outputs['pred_boxes'][0]
    
    scores, labels = pred_logits.softmax(-1).max(-1)
    
    keep = scores > threshold
    
    boxes = pred_boxes[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()
    
    # Scale boxes back
    boxes[:, [0, 2]] /= scale
    boxes[:, [1, 3]] /= scale
    
    return boxes, scores, labels

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(device),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return await root()

@app.post("/detect", response_model=ViolenceDetectionResponse)
async def detect_violence(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.5
):
    """
    Detect violence in uploaded image.
    
    Args:
        file: Image file to analyze
        confidence_threshold: Minimum confidence for detections (0.0-1.0)
    
    Returns:
        Detection results with bounding boxes and violence assessment
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image
        contents = await file.read()
        
        # Preprocess
        input_tensor, scale, img_w, img_h = preprocess_image(contents)
        input_tensor = input_tensor.to(device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Postprocess
        boxes, scores, labels = postprocess_predictions(
            outputs, scale, confidence_threshold
        )
        
        # Build detections list
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            detections.append(DetectionResult(
                box=[x1, y1, x2, y2],
                confidence=float(score),
                class_id=int(label),
                class_name="person"
            ))
        
        # Determine violence (simplified - would use behavior analysis in full system)
        violence_detected = len(detections) > 2  # Multiple people = potential violence
        violence_confidence = float(scores.mean()) if len(scores) > 0 else 0.0
        
        return ViolenceDetectionResponse(
            success=True,
            detections=detections,
            violence_detected=violence_detected,
            violence_confidence=violence_confidence,
            num_persons=len(detections),
            inference_time_ms=inference_time,
            image_width=img_w,
            image_height=img_h
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_batch")
async def detect_violence_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = 0.5
):
    """
    Detect violence in multiple images.
    
    Args:
        files: List of image files to analyze
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        List of detection results
    """
    results = []
    for file in files:
        result = await detect_violence(file, confidence_threshold)
        results.append(result)
    
    return {"results": results}

@app.get("/model_info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),
        "device": str(device),
        "architecture": "SAF-DETR",
        "backbone": "ResNet-50",
        "hidden_dim": 256,
        "num_queries": 100
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
