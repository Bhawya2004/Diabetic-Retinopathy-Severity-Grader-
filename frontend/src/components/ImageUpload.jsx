import React, { useRef, useState } from 'react';
import { UploadCloud, Image as ImageIcon, X, Sparkles } from 'lucide-react';

export default function ImageUpload({ selectedFile, previewUrl, onSelectFile, onClearFile, disabled }) {
  const fileInputRef = useRef(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      onSelectFile(file);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onSelectFile(file);
    }
  };

  const loadDemoSample = async () => {
    // Generate a quick demo retinal image or test image via synthetic canvas
    const canvas = document.createElement('canvas');
    canvas.width = 400;
    canvas.height = 400;
    const ctx = canvas.getContext('2d');
    
    // Create retinal fundus synthetic demo
    const grad = ctx.createRadialGradient(200, 200, 20, 200, 200, 190);
    grad.addColorStop(0, '#e76f51');
    grad.addColorStop(0.7, '#c1440e');
    grad.addColorStop(1, '#110502');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(200, 200, 190, 0, Math.PI * 2);
    ctx.fill();

    // Optic disc
    ctx.fillStyle = '#ffdd99';
    ctx.beginPath();
    ctx.ellipse(130, 200, 26, 32, 0, 0, Math.PI * 2);
    ctx.fill();

    // Vessels
    ctx.strokeStyle = '#6b0000';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(130, 200);
    ctx.bezierCurveTo(170, 150, 230, 120, 310, 100);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(130, 200);
    ctx.bezierCurveTo(180, 250, 250, 280, 330, 310);
    ctx.stroke();

    // Hemorrhages / lesions
    ctx.fillStyle = '#4a0000';
    ctx.beginPath();
    ctx.arc(240, 180, 7, 0, Math.PI * 2);
    ctx.arc(270, 220, 9, 0, Math.PI * 2);
    ctx.arc(220, 240, 6, 0, Math.PI * 2);
    ctx.fill();

    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], 'retina_demo_sample.png', { type: 'image/png' });
        onSelectFile(file);
      }
    });
  };

  return (
    <div className="upload-section">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/png, image/jpeg, image/jpg"
        style={{ display: 'none' }}
        onChange={handleFileChange}
        disabled={disabled}
      />

      {!previewUrl ? (
        <div
          className={`dropzone ${isDragOver ? 'drag-over' : ''} ${disabled ? 'disabled' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !disabled && fileInputRef.current?.click()}
        >
          <div className="dropzone-icon">
            <UploadCloud size={40} />
          </div>
          <h3 className="dropzone-title">Upload Retinal Fundus Image</h3>
          <p className="dropzone-desc">Drag and drop a retinal photograph, or browse files</p>
          <div className="dropzone-tags">
            <span>PNG</span>
            <span>JPG</span>
            <span>JPEG</span>
          </div>

          <div className="demo-sample-row" onClick={(e) => e.stopPropagation()}>
            <button
              type="button"
              className="demo-sample-btn"
              onClick={loadDemoSample}
              disabled={disabled}
            >
              <Sparkles size={14} />
              <span>Load Synthetic Fundus Demo</span>
            </button>
          </div>
        </div>
      ) : (
        <div className="preview-container">
          <div className="preview-wrapper">
            <img src={previewUrl} alt="Retinal fundus preview" className="preview-image" />
            <button
              type="button"
              className="preview-clear-btn"
              onClick={onClearFile}
              disabled={disabled}
              title="Remove image"
            >
              <X size={18} />
            </button>
          </div>
          <div className="preview-meta">
            <div className="file-info">
              <ImageIcon size={16} />
              <span className="file-name">{selectedFile?.name || 'Selected Image'}</span>
              <span className="file-size">
                {selectedFile?.size ? `(${(selectedFile.size / 1024).toFixed(1)} KB)` : ''}
              </span>
            </div>
            <button
              type="button"
              className="change-file-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled}
            >
              Replace Image
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
