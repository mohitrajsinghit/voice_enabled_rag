import React, { useState, useCallback } from 'react';

/**
 * AnswerCard: displays query transcript, grounded answer, guardrail verdict, and source passages.
 */
export default function AnswerCard({ result }) {
  const [highlightedSource, setHighlightedSource] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);

  if (!result) return null;

  const { transcript, answer, sources, guardrail, status } = result;
  const isRefused = status === 'refused';
  const isError = status === 'error';
  const isAnswered = status === 'answered';

  // Text-to-speech handler
  const handleSpeakAnswer = useCallback(() => {
    if (!answer || !('speechSynthesis' in window)) return;

    if (isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
      return;
    }

    const cleanText = answer.replace(/\[Source\s+\d+\]/gi, '').trim();
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    setIsSpeaking(true);
    window.speechSynthesis.speak(utterance);
  }, [answer, isSpeaking]);

  // Highlight citation on click
  const handleCitationClick = (sourceNum) => {
    setHighlightedSource(sourceNum);
    const element = document.getElementById(`source-item-${sourceNum}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  // Render formatted answer with interactive citation tags
  const renderFormattedAnswer = (rawText) => {
    if (!rawText) return null;
    const parts = rawText.split(/(\[Source\s+\d+\])/gi);
    return parts.map((part, index) => {
      const match = part.match(/\[Source\s+(\d+)\]/i);
      if (match) {
        const sourceNum = parseInt(match[1], 10);
        return (
          <span
            key={index}
            className="citation-tag"
            onClick={() => handleCitationClick(sourceNum)}
            title={`View Source ${sourceNum}`}
          >
            [Source {sourceNum}]
          </span>
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  return (
    <div className="results-container">
      {/* Main Glassmorphic Response Panel */}
      <div className={`glass-panel ${isAnswered ? 'glass-panel-glow' : ''}`} style={{ padding: '28px 32px' }}>
        
        {/* Card Top Header */}
        <div className="card-header-bar">
          <div className="card-title-group">
            {isAnswered && (
              <div className="card-icon-badge success">✓</div>
            )}
            {isRefused && (
              <div className="card-icon-badge refused">⚠️</div>
            )}
            {isError && (
              <div className="card-icon-badge error">✕</div>
            )}
            <div>
              <h2 className="card-heading">
                {isAnswered && 'Grounded AI Intelligence'}
                {isRefused && 'Safety & Scope Guardrail Triggered'}
                {isError && 'System Pipeline Notice'}
              </h2>
            </div>
          </div>

          {/* Quick status indicator badge */}
          {guardrail && (
            <span
              style={{
                fontSize: '0.78rem',
                fontFamily: 'var(--font-mono)',
                fontWeight: 700,
                padding: '4px 12px',
                borderRadius: 'var(--radius-full)',
                background: guardrail.passed ? 'rgba(16, 185, 129, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                color: guardrail.passed ? '#34d399' : '#fbbf24',
                border: `1px solid ${guardrail.passed ? 'rgba(16, 185, 129, 0.3)' : 'rgba(245, 158, 11, 0.3)'}`,
              }}
            >
              {guardrail.passed ? 'GUARDRAIL: PASS' : `GUARDRAIL: ${guardrail.category.toUpperCase()}`}
            </span>
          )}
        </div>

        {/* User Query / Voice Transcript */}
        {transcript && (
          <div className="transcript-quote-box">
            <div className="transcript-label">Transcribed Query</div>
            <div>"{transcript}"</div>
          </div>
        )}

        {/* Guardrail Refusal Details (when query was intercepted) */}
        {isRefused && guardrail && (
          <div className="guardrail-verdict-box">
            <div className="verdict-header">
              <span className="verdict-badge">🛡️ {guardrail.category} Protection</span>
            </div>
            <div className="verdict-reason">
              {guardrail.reason || 'Query was outside knowledge domain or deemed unsafe.'}
            </div>
          </div>
        )}

        {/* Grounded Answer */}
        {answer && (
          <div className="answer-content">
            {renderFormattedAnswer(answer)}
          </div>
        )}

        {/* Action Toolbar */}
        {answer && (
          <div className="answer-actions-bar">
            {'speechSynthesis' in window && (
              <button
                className="action-btn"
                onClick={handleSpeakAnswer}
                aria-label="Listen to answer"
              >
                <span>{isSpeaking ? '⏹️' : '🔊'}</span>
                <span>{isSpeaking ? 'Stop Audio' : 'Listen Answer'}</span>
              </button>
            )}
            <button
              className="action-btn"
              onClick={() => navigator.clipboard.writeText(answer)}
              aria-label="Copy answer to clipboard"
            >
              <span>📋</span>
              <span>Copy Response</span>
            </button>
          </div>
        )}
      </div>

      {/* Retrieved Knowledge Sources */}
      {sources && sources.length > 0 && (
        <div className="glass-panel" style={{ padding: '24px 32px' }}>
          <div className="sources-heading">
            <span>Retrieved Knowledge Evidence ({sources.length} Passages)</span>
            <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--neon-cyan)' }}>
              FAISS HNSW Vector Search
            </span>
          </div>

          <div className="sources-container">
            {sources.map((source, i) => (
              <SourcePassageCard
                key={i}
                source={source}
                index={i + 1}
                isHighlighted={highlightedSource === i + 1}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Individual Source Passage Card
 */
function SourcePassageCard({ source, index, isHighlighted }) {
  const [expanded, setExpanded] = useState(false);
  const { chunk, score } = source;
  const matchPct = (score * 100).toFixed(1);

  return (
    <div
      className={`source-item-card ${isHighlighted ? 'highlighted' : ''}`}
      onClick={() => setExpanded(!expanded)}
      id={`source-item-${index}`}
    >
      <div className="source-meta-row">
        <span className="source-pill-id">
          [Source {index}] • {chunk.source_doc_id || `chunk_${index}`}
        </span>
        <div className="source-score-gauge">
          <div className="score-bar-bg">
            <div
              className="score-bar-fill"
              style={{ width: `${Math.min(Math.max(score * 100, 10), 100)}%` }}
            />
          </div>
          <span className="score-num">{matchPct}% match</span>
        </div>
      </div>
      <div className={`source-body-text ${expanded ? '' : 'collapsed'}`}>
        {chunk.metadata?.window_text || chunk.text}
      </div>
    </div>
  );
}
