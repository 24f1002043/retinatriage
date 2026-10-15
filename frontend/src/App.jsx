import { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';

const STAGE_LABELS = [
  'No DR',
  'Mild',
  'Moderate',
  'Severe',
  'Proliferative DR',
];

/* Override with VITE_API_BASE in .env to point at a deployed backend. */
const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';
const API_URL = `${API_BASE}/predict`;

/* ─── Toast Component ─── */
function Toast({ message, onClose }) {
  const [exiting, setExiting] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setExiting(true);
      setTimeout(onClose, 300);
    }, 5000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className={`toast ${exiting ? 'toast-exit' : ''}`}>
      <span>⚠️</span>
      <span>{message}</span>
    </div>
  );
}

/* ─── Main App ─── */
export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showUntrainedWarning, setShowUntrainedWarning] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  /* ─── File Handling ─── */
  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file.');
      return;
    }
    setSelectedFile(file);
    setResults(null);
    setShowUntrainedWarning(false);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files?.[0];
      handleFile(file);
    },
    [handleFile],
  );

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => setDragOver(false), []);

  const onFileChange = useCallback(
    (e) => handleFile(e.target.files?.[0]),
    [handleFile],
  );

  /* ─── Analysis ─── */
  const analyze = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResults(null);
    setShowUntrainedWarning(false);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.detail || `Server error (${response.status})`,
        );
      }

      const data = await response.json();
      setResults(data);
      if (data.untrained_warning) {
        setShowUntrainedWarning(true);
      }
    } catch (err) {
      setError(
        err.message === 'Failed to fetch'
          ? `Cannot connect to the backend at ${API_BASE}. Is the server running?`
          : err.message,
      );
    } finally {
      setLoading(false);
    }
  };

  /* ─── Parse probabilities dict from backend into ordered array ─── */
  const probEntries = results
    ? STAGE_LABELS.map((label, idx) => ({
        label,
        idx,
        value: results.probabilities?.[label] ?? 0,
      }))
    : [];

  return (
    <div className="app">
      {/* ─── Header ─── */}
      <header className="header">
        <span className="header-icon" role="img" aria-label="eye">
          👁️
        </span>
        <h1 className="header-title">RetinaTriage</h1>
        <p className="header-subtitle">
          AI-Powered Diabetic Retinopathy Screening
        </p>
      </header>

      <main className="main-content">
        {/* ─── Disclaimer ─── */}
        <div className="banner banner-disclaimer">
          ⚠️ Research/demo tool only, not a medical device, not for clinical or
          diagnostic use.
        </div>

        {/* ─── Untrained Warning ─── */}
        {showUntrainedWarning && (
          <div className="banner banner-untrained">
            ⚠️ Model has no trained weights loaded; output is not meaningful.
            See README for training instructions.
          </div>
        )}

        {/* ─── Upload ─── */}
        <div className="upload-card">
          <div
            className={`dropzone ${dragOver ? 'drag-over' : ''} ${preview ? 'has-preview' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              className="file-input"
              onChange={onFileChange}
            />

            {preview ? (
              <div className="preview-container">
                <img
                  src={preview}
                  alt="Fundus preview"
                  className="preview-image"
                />
                <p className="file-name">{selectedFile?.name}</p>
              </div>
            ) : (
              <>
                <span className="dropzone-icon">🔬</span>
                <p className="dropzone-text">
                  Drag &amp; drop a fundus image here
                </p>
                <p className="dropzone-subtext">or click to browse</p>
              </>
            )}
          </div>
        </div>

        {/* ─── Analyze Button ─── */}
        <button
          className="analyze-btn"
          onClick={analyze}
          disabled={!selectedFile || loading}
        >
          {loading && <div className="spinner" />}
          <span>{loading ? 'Analyzing…' : 'Analyze Image'}</span>
        </button>

        {/* ─── Results ─── */}
        {results && (
          <div className="results-panel">
            {/* Stage Badge */}
            <div className={`stage-card stage-${results.stage}`}>
              <p className="stage-label-small">Prediction</p>
              <div className={`stage-badge stage-${results.stage}`}>
                Stage {results.stage}: {results.stage_label}
              </div>
              <p className="stage-confidence">
                Confidence:{' '}
                <strong>{(results.confidence * 100).toFixed(1)}%</strong>
                {results.inference_ms != null && (
                  <>
                    {' · '}
                    <strong>{Math.round(results.inference_ms)} ms</strong>
                    {' inference'}
                  </>
                )}
              </p>
            </div>

            {/* Probability Bars */}
            <div className="probabilities-card">
              <p className="probabilities-title">Class Probabilities</p>
              {probEntries.map(({ label, idx, value }) => (
                <div className="prob-row" key={idx}>
                  <span className="prob-label">
                    {idx}: {label}
                  </span>
                  <div className="prob-bar-track">
                    <div
                      className={`prob-bar-fill stage-${idx}`}
                      style={{ width: `${(value * 100).toFixed(1)}%` }}
                    />
                  </div>
                  <span className="prob-value">
                    {(value * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>

            {/* Image Comparison */}
            {results.heatmap && (
              <div className="comparison-card">
                <div className="comparison-grid">
                  <div className="comparison-item">
                    <p className="comparison-label">Original</p>
                    <img
                      src={preview}
                      alt="Original fundus"
                      className="comparison-img"
                    />
                  </div>
                  <div className="comparison-item">
                    <p className="comparison-label">Grad-CAM Heatmap</p>
                    <img
                      src={`data:image/png;base64,${results.heatmap}`}
                      alt="Grad-CAM heatmap overlay"
                      className="comparison-img"
                    />
                  </div>
                  <p className="comparison-caption">
                    Red regions = areas most influencing the prediction
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* ─── Error Toast ─── */}
      {error && <Toast message={error} onClose={() => setError(null)} />}
    </div>
  );
}
