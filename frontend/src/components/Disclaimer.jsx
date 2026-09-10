import React from 'react';
import { ShieldAlert } from 'lucide-react';

export default function Disclaimer() {
  return (
    <aside className="disclaimer-card" aria-label="Clinical disclaimer">
      <div className="disclaimer-icon">
        <ShieldAlert size={22} />
      </div>
      <div className="disclaimer-content">
        <h4>Clinical Screening Aid — Not a Final Diagnostic Tool</h4>
        <p>
          This AI system provides automated severity triage and visual explainability using deep learning.
          All predictions should be verified by a board-certified ophthalmologist or retina specialist before clinical action.
        </p>
      </div>
    </aside>
  );
}
