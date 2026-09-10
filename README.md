# Diabetic Retinopathy Severity Grader (PyTorch + FastAPI + React)

A five-class deep learning application for Diabetic Retinopathy severity grading (Grades 0–4) trained on APTOS 2019 retinal fundus images with imbalance mitigation (weighted cross-entropy / focal loss) and visual explainability via Grad-CAM.

---

## 1. Project Structure

```
Diabetic-Retinopathy-Severity-Grader/
│
├── ml/                                # Machine Learning core (PyTorch)
│   ├── requirements.txt               # PyTorch training & evaluation dependencies
│   ├── config.yaml                    # Model and training hyperparameter configurations
│   ├── models/
│   │   ├── baseline_model.pt          # Baseline plain cross-entropy checkpoint
│   │   └── final_model.pt             # Imbalance-aware best model checkpoint
│   └── src/
│       ├── preprocessing.py           # Shared preprocessing (Ben Graham crop + normalization)
│       ├── dataset.py                 # Dataset loader and weighted random sampler
│       ├── model.py                   # ResNet50 / EfficientNet backbone architecture
│       ├── losses.py                  # Focal loss & class-balanced loss functions
│       ├── gradcam.py                 # Standalone PyTorch Grad-CAM explainability generator
│       ├── train.py                   # Model training and ablation runner
│       ├── evaluate.py                # Metric evaluation (QWK, Macro-F1, per-class sensitivity)
│       └── cross_dataset_eval.py      # Domain shift validation (IDRiD)
│
├── backend/                           # FastAPI REST API
│   ├── requirements.txt               # Backend dependencies
│   ├── main.py                        # FastAPI application with CORS & /predict endpoint
│   ├── inference.py                   # Singleton model loader, inference & Grad-CAM pipeline
│   └── schemas.py                     # Pydantic request/response data contracts
│
├── results/
│   ├── figures/                       # Confusion matrices and class distribution charts
│   ├── ablation_table.csv             # Comparative ablation results
│   ├── baseline_metrics.json          # Baseline model performance metrics
│   └── final_metrics.json             # Final model performance metrics
│
└── .venv/                             # Python virtual environment (gitignored)
```

---

## 2. Environment Setup

A dedicated virtual environment `.venv` is configured. To activate it:

### On macOS / Linux:
```bash
source .venv/bin/activate
```

### On Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

To install or update dependencies:
```bash
pip install -r backend/requirements.txt
```

---

## 3. Running the FastAPI Backend

Start the FastAPI server:
```bash
uvicorn backend.main:app --reload --port 8000
```
- **Interactive API Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: `GET http://localhost:8000/health`
- **Prediction & Grad-CAM**: `POST http://localhost:8000/predict` (Accepts multipart `file=@image.jpg`)

### Testing via curl:
```bash
curl -X POST -F "file=@datasets/IDRiD-dataset/Imagenes/Imagenes/IDRiD_001.jpg" http://localhost:8000/predict
```

---

## 4. Model & Explainability Highlights

- **Backbone**: Pretrained ResNet-50 fine-tuned on fundus imagery.
- **Hardware Acceleration**: Automatically utilizes Apple Silicon Metal (`mps`), CUDA (`cuda`), or CPU fallback.
- **Visual Explainability**: Grad-CAM targets `layer4[-1]` of ResNet-50, generating a normalized heatmap and blending it with the preprocessed retinal image using the `jet` colormap for clinician interpretability.
