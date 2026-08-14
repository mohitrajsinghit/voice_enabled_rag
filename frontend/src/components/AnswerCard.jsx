import React, { useState } from 'react';

/**
 * AnswerCard: displays transcript, answer, and source passages.
 * Shows "Refused" state for rejected queries.
 */
export default function AnswerCard({ result }) {
  if (!result) return null;

  const { transcript, answer, sources, guardrail, status } = result;
  const isRefused = status === 'refused';
  const isError = status === 'error';

  return (
    <div className="answer-section">
      {/* Main answer card */}
      <div className={`glass-card answer-card ${isRefused ? 'refused' : ''}`}>
        {/* Transcript */}
        {transcript && (
          <>
            <div className="label">Transcript</div>
            <div className="transcript">"{transcript}"</div>
          </>
        )}

        {/* Status badges */}
        {isRefused && (
          <div className="refused-badge">
            <span>⚠️</span>
            <span>Query Refused</span>
          </div>
        )}

        {isError && (
          <div className="refused-badge" style={{ borderColor: 'rgba(239, 68, 68, 0.5)' }}>
            <span>❌</span>
            <span>Error</span>
          </div>
        )}

        {/* Guardrail reason */}
        {guardrail && !guardrail.passed && (
          <div style={{ marginBottom: 16 }}>
            <div className="label">Reason</div>
            <div style={{ color: 'var(--warning)', fontSize: '0.9rem', lineHeight: 1.6 }}>
              {guardrail.reason}
              {guardrail.category !== 'ok' && (
                <span
                  style={{
                    marginLeft: 8,
                    padding: '2px 8px',
                    borderRadius: 'var(--radius-full)',
                    background: 'rgba(245, 158, 11, 0.15)',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                  }}
                >
                  {guardrail.category}
                </span>
              )}
            </div>
          </div>
        )}

        {/* Answer text */}
        {answer && (
          <>
            <div className="label">Answer</div>
            <div className="answer-text">{answer}</div>
          </>
        )}

        {/* Sources */}
        {sources && sources.length > 0 && (
          <div className="sources-section">
            <div className="label">
              Sources ({sources.length} passages)
            </div>
            {sources.map((source, i) => (
              <SourceCard key={i} source={source} index={i + 1} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Individual source passage card.
 */
function SourceCard({ source, index }) {
  const [expanded, setExpanded] = useState(false);
  const { chunk, score } = source;

  return (
    <div className="source-card" onClick={() => setExpanded(!expanded)} id={`source-${index}`}>
      <div className="source-header">
        <span className="source-id">
          [Source {index}] {chunk.source_doc_id}
        </span>
        <span className="score-badge">
          {(score * 100).toFixed(1)}% match
        </span>
      </div>
      <div className={`source-text ${expanded ? '' : 'collapsed'}`}>
        {chunk.metadata?.window_text || chunk.text}
      </div>
    </div>
  );
}
