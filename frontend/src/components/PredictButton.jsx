import React from 'react';
import { Loader2, Zap } from 'lucide-react';

export default function PredictButton({ onClick, isLoading, disabled }) {
  return (
    <button
      type="button"
      className={`predict-btn ${isLoading ? 'loading' : ''}`}
      onClick={onClick}
      disabled={disabled || isLoading}
    >
      {isLoading ? (
        <>
          <Loader2 size={20} className="spin" />
          <span>Analyzing Retinal Scan & Grad-CAM...</span>
        </>
      ) : (
        <>
          <Zap size={20} />
          <span>Analyze Retinal Scan</span>
        </>
      )}
    </button>
  );
}
