"""Pydantic schemas for FastAPI backend."""
from __future__ import annotations

from typing import Dict
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    backbone: str = Field(..., example="resnet50")
    device: str = Field(..., example="mps")


class PredictionResponse(BaseModel):
    grade: int = Field(..., ge=0, le=4, description="Predicted DR severity grade (0-4)")
    grade_name: str = Field(..., description="Name of the severity grade")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of the predicted grade")
    probabilities: Dict[str, float] = Field(..., description="Probability distribution across all 5 classes")
    risk_level: str = Field(..., description="Assessed clinical risk level (Low, Moderate, High, Critical)")
    clinical_action: str = Field(..., description="Standard clinical guidance / follow-up recommendation")
    gradcam_overlay_base64: str = Field(..., description="Base64 data URI of the Grad-CAM heatmap overlay")
    raw_processed_base64: str = Field(..., description="Base64 data URI of the preprocessed fundus image")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
