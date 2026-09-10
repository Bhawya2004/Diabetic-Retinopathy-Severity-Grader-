# Diabetic Retinopathy Severity Grader (PyTorch + FastAPI + React)

A full-stack deep learning application for Diabetic Retinopathy severity grading (Grades 0–4) trained on APTOS 2019 retinal fundus images with imbalance mitigation (weighted cross-entropy / focal loss), visual explainability via Grad-CAM, a FastAPI backend, and a vibrant React (Vite) frontend.

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
├── frontend/                          # React + Vite Web UI
│   ├── package.json
│   ├── vite.config.js
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Navbar.jsx             # Brand header with live API status & device indicator
│   │   │   ├── ImageUpload.jsx        # Drag-and-drop file upload & fundus preview
│   │   │   ├── PredictButton.jsx      # Action trigger with animated loading state
│   │   │   ├── ResultDisplay.jsx      # Visual Grad-CAM comparison & probability bars
│   │   │   └── Disclaimer.jsx         # Clinical screening advisory
│   │   ├── api/
│   │   │   └── predict.js             # API client connecting to FastAPI /predict
│   │   └── styles/
│   │       └── App.css                # Bright, warm clinical color palette
│   └── index.html
│
├── results/
│   ├── figures/                       # Confusion matrices and class distribution charts
│   ├── ablation_table.csv             # Comparative ablation results
│   ├── baseline_metrics.json          # Baseline model performance metrics
│   └── final_metrics.json             # Final model performance metrics
│
├── archive/                           # Preserved legacy scripts & historical outputs
└── .venv/                             # Python virtual environment (gitignored)
```

---

## 2. How to Run the Full-Stack Application

Run the backend and frontend simultaneously in **two side-by-side terminal tabs**:

### Terminal 1 — Backend (FastAPI)
```bash
# 1. Activate the virtual environment
source .venv/bin/activate

# 2. Start FastAPI on port 8000
uvicorn backend.main:app --reload --port 8000
```
- API Health: [http://localhost:8000/health](http://localhost:8000/health)
- Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### Terminal 2 — Frontend (React + Vite)
```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Start the Vite development server
npm run dev
```
- Open the web application: **[http://localhost:5173](http://localhost:5173)**

---

## 3. UI Features & Design Highlights

- **Vibrant Color Palette**: Styled with a warm, modern palette (Sunset Coral, Sunny Amber, Tangerine, and Fresh Emerald on soft cream surfaces — avoiding dark mode and corporate blues).
- **Interactive Grad-CAM Heatmap Viewer**: Side-by-side or toggled view displaying where the CNN detected diabetic lesions (microaneurysms, hemorrhages, hard exudates).
- **Multi-Class Probability Distribution**: Horizontal color-coded meters for all 5 severity stages (Grades 0 to 4).
- **Clinical Risk & Protocol Guidance**: Automated clinical next steps based on predicted stage.
