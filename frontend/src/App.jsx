import React, { useEffect, useState } from 'react';
import Navbar from './components/Navbar';
import ImageUpload from './components/ImageUpload';
import PredictButton from './components/PredictButton';
import ResultDisplay from './components/ResultDisplay';
import Disclaimer from './components/Disclaimer';
import { checkBackendHealth, predictSeverity } from './api/predict';
import { AlertCircle, Sparkles } from 'lucide-react';
import './styles/App.css';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const [backendState, setBackendState] = useState({
    online: false,
    backbone: 'resnet50',
    device: 'mps',
  });

  // Check health on mount
  useEffect(() => {
    let isMounted = true;
    const pollHealth = async () => {
      try {
        const data = await checkBackendHealth();
        if (isMounted) {
          setBackendState({
            online: true,
            backbone: data.backbone || 'resnet50',
            device: data.device || 'mps',
          });
        }
      } catch {
        if (isMounted) {
          setBackendState((prev) => ({ ...prev, online: false }));
        }
      }
    };

    pollHealth();
    const interval = setInterval(pollHealth, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleSelectFile = (file) => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleClearFile = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await predictSeverity(selectedFile);
      setResult(data);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred during prediction.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Navbar
        backendOnline={backendState.online}
        backbone={backendState.backbone}
        device={backendState.device}
      />

      <main className="main-content">
        {/* Hero Header */}
        <div className="hero-header">
          <div className="hero-tag">
            <Sparkles size={14} />
            <span>AI Retinal Diagnostic Assistant</span>
          </div>
          <h1 className="hero-title">Diabetic Retinopathy Severity Grader</h1>
          <p className="hero-desc">
            Upload a retinal fundus photograph to classify diabetic retinopathy severity (Grades 0–4)
            and visualize localized vascular biomarkers using Grad-CAM.
          </p>
        </div>

        {/* Upload Container */}
        <div className="upload-card">
          <ImageUpload
            selectedFile={selectedFile}
            previewUrl={previewUrl}
            onSelectFile={handleSelectFile}
            onClearFile={handleClearFile}
            disabled={isLoading}
          />

          <div className="predict-btn-wrapper">
            <PredictButton
              onClick={handlePredict}
              isLoading={isLoading}
              disabled={!selectedFile}
            />
          </div>

          {error && (
            <div className="error-banner" role="alert">
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Results Presentation */}
        {result && <ResultDisplay result={result} />}

        {/* Clinical Disclaimer */}
        <Disclaimer />
      </main>

      <footer className="footer">
        <p>RetinaGrade • Trained on APTOS 2019 Dataset with Imbalance Optimization • PyTorch & FastAPI</p>
      </footer>
    </div>
  );
}
