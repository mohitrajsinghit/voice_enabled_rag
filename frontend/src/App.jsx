import React, { useState, useCallback } from 'react';
import Recorder from './components/Recorder';
import AnswerCard from './components/AnswerCard';
import LatencyBadge from './components/LatencyBadge';
import { queryWithText } from './api';

/**
 * Main application component.
 * Record voice or type a query → get RAG results with sources + latency.
 */
export default function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [statusType, setStatusType] = useState('idle'); // idle, recording, transcribing, searching, generating, done, error
  const [textInput, setTextInput] = useState('');

  const handleRecorderResult = useCallback((response) => {
    setResult(response);
    setLoading(false);
  }, []);

  const handleStatusChange = useCallback((type, message) => {
    setStatusType(type);
    setStatusMessage(message);
    if (type === 'transcribing' || type === 'recording') {
      setLoading(true);
    }
    if (type === 'done' || type === 'error') {
      setLoading(false);
    }
  }, []);

  const handleTextSubmit = useCallback(async (e) => {
    e.preventDefault();
    if (!textInput.trim() || loading) return;

    setLoading(true);
    setResult(null);
    setStatusType('searching');
    setStatusMessage('Processing query...');

    try {
      const response = await queryWithText(textInput.trim());
      setResult(response);
      setStatusType('done');
      setStatusMessage('Complete');
    } catch (error) {
      setStatusType('error');
      setStatusMessage(error.message);
    } finally {
      setLoading(false);
    }
  }, [textInput, loading]);

  const getStatusDotClass = () => {
    if (statusType === 'done') return 'status-dot success';
    if (statusType === 'error') return 'status-dot error';
    if (statusType !== 'idle') return 'status-dot active';
    return 'status-dot';
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <h1>Voice RAG</h1>
        <p className="subtitle">
          Speak or type your question — get grounded answers from knowledge base
        </p>
      </header>

      {/* Main content */}
      <main className="app-main">
        <div className="container">
          {/* Recorder */}
          <Recorder
            onResult={handleRecorderResult}
            onStatusChange={handleStatusChange}
            disabled={loading}
          />

          {/* Divider */}
          <div style={{ textAlign: 'center', marginBottom: 20 }}>
            <span className="or-divider">or type your question</span>
          </div>

          {/* Text input */}
          <div className="recorder-section">
            <form className="text-input-section" onSubmit={handleTextSubmit}>
              <div className="text-input-wrapper">
                <input
                  type="text"
                  className="text-input"
                  placeholder="e.g., What is the Taj Mahal?"
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  disabled={loading}
                  id="text-query-input"
                />
                <button
                  type="submit"
                  className="submit-btn"
                  disabled={loading || !textInput.trim()}
                  id="submit-button"
                >
                  {loading ? 'Processing...' : 'Search'}
                </button>
              </div>
            </form>
          </div>

          {/* Status bar */}
          {statusType !== 'idle' && (
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 24 }}>
              <div className="status-bar">
                <span className={getStatusDotClass()} />
                <span>{statusMessage}</span>
              </div>
            </div>
          )}

          {/* Loading dots */}
          {loading && (
            <div className="loader">
              <div className="loader-dot" />
              <div className="loader-dot" />
              <div className="loader-dot" />
            </div>
          )}

          {/* Results */}
          {result && (
            <>
              <AnswerCard result={result} />
              <div className="glass-card" style={{ padding: '16px 24px' }}>
                <LatencyBadge latencies={result.latencies} />
              </div>
            </>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer style={{
        textAlign: 'center',
        padding: '24px',
        color: 'var(--text-muted)',
        fontSize: '0.8rem',
      }}>
        Voice RAG System — Multi-strategy retrieval with grounded generation
      </footer>
    </div>
  );
}
