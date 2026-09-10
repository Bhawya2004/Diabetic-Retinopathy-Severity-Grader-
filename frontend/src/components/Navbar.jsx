import React from 'react';
import { Eye, Activity, Cpu } from 'lucide-react';

export default function Navbar({ backendOnline, device, backbone }) {
  return (
    <header className="navbar">
      <div className="navbar-container">
        <div className="brand">
          <div className="brand-icon">
            <Eye size={26} strokeWidth={2.4} />
          </div>
          <div>
            <div className="brand-title">
              Retina<span>Grade</span>
            </div>
            <div className="brand-subtitle">Diabetic Retinopathy AI Screening</div>
          </div>
        </div>

        <div className="navbar-badges">
          <div className={`status-badge ${backendOnline ? 'online' : 'offline'}`}>
            <Activity size={15} className={backendOnline ? 'pulse' : ''} />
            <span>{backendOnline ? 'API Connected' : 'Connecting API...'}</span>
          </div>
          {backendOnline && (
            <div className="tech-badge">
              <Cpu size={15} />
              <span>{backbone.toUpperCase()} • {device.toUpperCase()}</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
