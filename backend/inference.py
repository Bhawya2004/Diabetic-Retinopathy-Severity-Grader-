"""Inference engine loading model once and generating predictions + Grad-CAM."""
from __future__ import annotations

import base64
import io
from pathlib import Path
import sys
import time
from typing import Optional

from PIL import Image
import torch
import torch.nn.functional as F

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.model import build_model
from ml.src.preprocessing import CLASS_NAMES, preprocess_fundus_image
from ml.src.gradcam import generate_gradcam_explanation

CLINICAL_GUIDANCE = {
    0: {
        "risk_level": "Low",
        "action": "No diabetic retinopathy detected. Routine annual retinal screening recommended.",
    },
    1: {
        "risk_level": "Low to Moderate",
        "action": "Mild non-proliferative DR (microaneurysms only). Follow-up exam in 6–12 months; optimize glycaemic control.",
    },
    2: {
        "risk_level": "Moderate",
        "action": "Moderate NPDR detected. Comprehensive ophthalmic referral within 3–6 months recommended.",
    },
    3: {
        "risk_level": "High",
        "action": "Severe NPDR detected. Urgent referral to a retina specialist within 1 month (high risk of progression).",
    },
    4: {
        "risk_level": "Critical",
        "action": "Proliferative DR / neovascularization detected. Immediate retina specialist referral (panretinal photocoagulation or anti-VEGF evaluation needed).",
    },
}


class DRInferenceEngine:
    _instance: Optional[DRInferenceEngine] = None

    def __init__(self, model_path: Optional[str | Path] = None):
        if model_path is None:
            # Default to final_model.pt, fallback to outputs
            candidates = [
                PROJECT_ROOT / "ml" / "models" / "final_model.pt",
                PROJECT_ROOT / "outputs" / "weighted_ce" / "best_model.pt",
                PROJECT_ROOT / "outputs" / "plain_ce" / "best_model.pt",
            ]
            for candidate in candidates:
                if candidate.exists():
                    model_path = candidate
                    break
            if model_path is None:
                raise FileNotFoundError("No trained model checkpoint found in ml/models or outputs.")

        self.model_path = Path(model_path)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location="cpu")
        self.config = checkpoint.get("config", {"model": {"backbone": "resnet50", "num_classes": 5}})
        self.model_cfg = self.config["model"]
        self.backbone = self.model_cfg.get("backbone", "resnet50")
        
        # Build & load model weights
        self.model = build_model(
            backbone=self.backbone,
            num_classes=self.model_cfg.get("num_classes", 5),
            pretrained=False,
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def get_instance(cls, model_path: Optional[str | Path] = None) -> DRInferenceEngine:
        if cls._instance is None:
            cls._instance = cls(model_path)
        return cls._instance

    def predict_and_explain(self, image_bytes: bytes) -> dict:
        start_time = time.time()
        
        # 1. Load image from bytes
        raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 2. Preprocess
        tensor, processed_pil = preprocess_fundus_image(raw_image, image_size=224)
        tensor = tensor.to(self.device)

        # 3. Predict and compute Grad-CAM
        # Note: Grad-CAM computes gradients w.r.t target class score
        heatmap, overlay_pil, overlay_base64, pred_class, confidence = generate_gradcam_explanation(
            model=self.model,
            input_tensor=tensor,
            original_pil=processed_pil,
            alpha=0.45,
        )

        # 4. Compute full probability distribution
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        probabilities = {CLASS_NAMES[i]: round(float(prob), 4) for i, prob in enumerate(probs)}

        # 5. Base64 encode preprocessed input image for UI display
        buf = io.BytesIO()
        processed_pil.save(buf, format="PNG")
        processed_base64 = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        guidance = CLINICAL_GUIDANCE.get(pred_class, {"risk_level": "Unknown", "action": "Consult ophthalmologist."})
        inference_time_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "grade": pred_class,
            "grade_name": CLASS_NAMES[pred_class],
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "risk_level": guidance["risk_level"],
            "clinical_action": guidance["action"],
            "gradcam_overlay_base64": overlay_base64,
            "raw_processed_base64": processed_base64,
            "inference_time_ms": inference_time_ms,
        }
