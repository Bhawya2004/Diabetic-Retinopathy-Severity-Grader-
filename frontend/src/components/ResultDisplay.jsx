import React, { useState } from 'react';
import { Layers, HelpCircle, CheckCircle2, AlertOctagon, Clock, Stethoscope } from 'lucide-react';

const GRADE_CONFIG = {
  0: {
    name: 'Grade 0 — No DR',
    badgeClass: 'grade-0',
    color: '#10b981',
    description: 'No microaneurysms or diabetic retinal abnormalities detected.',
  },
  1: {
    name: 'Grade 1 — Mild NPDR',
    badgeClass: 'grade-1',
    color: '#f59e0b',
    description: 'Microaneurysms present without substantial retinal vascular compromise.',
  },
  2: {
    name: 'Grade 2 — Moderate NPDR',
    badgeClass: 'grade-2',
    color: '#f97316',
    description: 'More extensive microaneurysms, blot hemorrhages, or early hard exudates.',
  },
  3: {
    name: 'Grade 3 — Severe NPDR',
    badgeClass: 'grade-3',
    color: '#ef4444',
    description: 'Extensive intraretinal hemorrhages (4-quadrant), venous beading, or IRMA.',
  },
  4: {
    name: 'Grade 4 — Proliferative DR',
    badgeClass: 'grade-4',
    color: '#d946ef',
    description: 'Active retinal neovascularization and/or vitreous/preretinal hemorrhage.',
  },
};

export default function ResultDisplay({ result }) {
  const [viewMode, setViewMode] = useState('side_by_side'); // 'overlay', 'raw', 'side_by_side'

  if (!result) return null;

  const {
    grade,
    grade_name,
    confidence,
    probabilities,
    risk_level,
    clinical_action,
    gradcam_overlay_base64,
    raw_processed_base64,
    inference_time_ms,
  } = result;

  const gradeInfo = GRADE_CONFIG[grade] || GRADE_CONFIG[0];

  return (
    <section className="results-container" aria-label="Analysis Results">
      {/* Top Banner Header */}
      <div className={`result-header-card ${gradeInfo.badgeClass}`}>
        <div className="result-header-left">
          <div className="grade-badge">
            <span>Grade {grade}</span>
          </div>
          <div>
            <h2 className="grade-title">{grade_name}</h2>
            <p className="grade-subtitle">{gradeInfo.description}</p>
          </div>
        </div>
        <div className="confidence-pill">
          <span className="conf-label">Confidence</span>
          <span className="conf-value">{(confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      {/* Main Grid: Images on Left, Analysis Breakdown on Right */}
      <div className="results-grid">
        {/* Left Column: Visual Explainability (Grad-CAM) */}
        <div className="result-panel image-panel">
          <div className="panel-header">
            <div className="panel-title">
              <Layers size={18} />
              <span>Grad-CAM Visual Heatmap</span>
            </div>
            <div className="view-toggle-buttons">
              <button
                type="button"
                className={`toggle-btn ${viewMode === 'side_by_side' ? 'active' : ''}`}
                onClick={() => setViewMode('side_by_side')}
              >
                Side by Side
              </button>
              <button
                type="button"
                className={`toggle-btn ${viewMode === 'overlay' ? 'active' : ''}`}
                onClick={() => setViewMode('overlay')}
              >
                Heatmap
              </button>
              <button
                type="button"
                className={`toggle-btn ${viewMode === 'raw' ? 'active' : ''}`}
                onClick={() => setViewMode('raw')}
              >
                Preprocessed
              </button>
            </div>
          </div>

          <div className="image-display-area">
            {viewMode === 'side_by_side' ? (
              <div className="side-by-side-grid">
                <div className="image-figure">
                  <span className="image-label">Preprocessed Fundus</span>
                  <img src={raw_processed_base64} alt="Preprocessed fundus" className="result-img" />
                </div>
                <div className="image-figure">
                  <span className="image-label heatmap-label">Grad-CAM Overlay</span>
                  <img src={gradcam_overlay_base64} alt="Grad-CAM heatmap overlay" className="result-img" />
                </div>
              </div>
            ) : viewMode === 'overlay' ? (
              <div className="single-image-wrapper">
                <img src={gradcam_overlay_base64} alt="Grad-CAM heatmap overlay" className="result-img single" />
              </div>
            ) : (
              <div className="single-image-wrapper">
                <img src={raw_processed_base64} alt="Preprocessed fundus" className="result-img single" />
              </div>
            )}
          </div>

          <p className="gradcam-caption">
            <strong>How to read:</strong> Warmer colors (red/yellow) indicate focal regions in the fundus where the deep CNN identified features indicative of <em>{grade_name}</em>.
          </p>
        </div>

        {/* Right Column: Probability Distribution & Clinical Recommendation */}
        <div className="result-panel data-panel">
          {/* Metrics summary row */}
          <div className="summary-stats-row">
            <div className="stat-card">
              <span className="stat-label">Risk Assessment</span>
              <span className={`stat-value risk-${risk_level.toLowerCase().replace(/\s+/g, '-')}`}>
                {risk_level}
              </span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Inference Time</span>
              <span className="stat-value speed-val">
                <Clock size={14} /> {inference_time_ms} ms
              </span>
            </div>
          </div>

          {/* Probability Distribution */}
          <div className="probabilities-card">
            <h4 className="card-subheading">Class Probability Distribution</h4>
            <div className="prob-bars-list">
              {Object.entries(probabilities || {}).map(([className, prob], idx) => {
                const isWinner = idx === grade;
                const percentage = (prob * 100).toFixed(1);
                const barColor = GRADE_CONFIG[idx]?.color || '#ff6b4a';

                return (
                  <div key={className} className={`prob-row ${isWinner ? 'winner-row' : ''}`}>
                    <div className="prob-labels">
                      <span className="prob-name">
                        {isWinner && <CheckCircle2 size={13} className="winner-icon" />}
                        {className}
                      </span>
                      <span className="prob-percent">{percentage}%</span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-fill"
                        style={{
                          width: `${Math.max(Number(percentage), 1.5)}%`,
                          backgroundColor: barColor,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Clinical Action Card */}
          <div className="clinical-card">
            <div className="clinical-header">
              <Stethoscope size={18} />
              <h4>Recommended Clinical Protocol</h4>
            </div>
            <p className="clinical-text">{clinical_action}</p>
          </div>
        </div>
      </div>
    </section>
  );
}
