"""FastAPI application serving Diabetic Retinopathy prediction and Grad-CAM explainability."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import HealthResponse, PredictionResponse
from backend.inference import DRInferenceEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once into memory at startup."""
    print("Pre-loading DR model and initializing inference engine...")
    engine = DRInferenceEngine.get_instance()
    print(f"DR model successfully loaded on device: {engine.device} (Backbone: {engine.backbone})")
    yield
    print("Shutting down DR API service.")


app = FastAPI(
    title="Diabetic Retinopathy Severity Grader API",
    description="FastAPI service serving deep learning DR classification (0-4) with Grad-CAM visual explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for React frontend (Vite default: http://localhost:5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins during development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Info"])
def root_info():
    return {
        "service": "Diabetic Retinopathy Severity Grader API",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    try:
        engine = DRInferenceEngine.get_instance()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            backbone=engine.backbone,
            device=str(engine.device),
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model service unhealthy: {str(e)}")


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_dr_severity(file: UploadFile = File(...)):
    """
    Accepts a retinal fundus image upload, runs preprocessing, model inference,
    and Grad-CAM heatmap generation, returning grade, confidence, and visual explanation.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        # Also check file extension fallback
        allowed_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        if not any(file.filename.lower().endswith(ext) for ext in allowed_exts):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type '{file.content_type}'. Please upload an image file (PNG, JPG, JPEG).",
            )

    try:
        contents = await file.read()
        engine = DRInferenceEngine.get_instance()
        result = engine.predict_and_explain(contents)
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing retinal image: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
