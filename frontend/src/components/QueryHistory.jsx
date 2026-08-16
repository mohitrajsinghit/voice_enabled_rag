import React, { useState } from 'react';

/**
 * QueryHistory — Session-local query history drawer.
 * Stores queries from the current browser session (ephemeral, not persisted).
 * Shows query text, language detection hint, latency, and allows re-running.
 */
export default function QueryHistory({ history, onRerun }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!history || history.length === 0) return null;

  return (
    <>
      {/* Floating trigger button */}
      <button
        className="history-trigger-btn"
        onClick={() => setIsOpen(!isOpen)}
        aria-label="Toggle query history"
        title={`${history.length} queries this session`}
      >
        <span className="history-trigger-icon">🕐</span>
        <span className="history-trigger-count">{history.length}</span>
      </button>

      {/* Slide-in drawer */}
      {isOpen && (
        <div className="history-backdrop" onClick={() => setIsOpen(false)}>
          <div className="history-drawer" onClick={(e) => e.stopPropagation()}>
            <div className="history-drawer-header">
              <h3>Session Query Log</h3>
              <button
                className="history-close-btn"
                onClick={() => setIsOpen(false)}
                aria-label="Close history"
              >
                ✕
              </button>
            </div>

            <div className="history-items-list">
              {history.map((item, idx) => (
                <div
                  key={idx}
                  className={`history-item ${item.status === 'refused' ? 'refused' : ''}`}
                  onClick={() => {
                    onRerun(item.query);
                    setIsOpen(false);
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <div className="history-item-top">
                    <span className="history-item-query">
                      {item.query.length > 60 ? item.query.slice(0, 60) + '…' : item.query}
                    </span>
                    <span className="history-item-time">{item.timestamp}</span>
                  </div>
                  <div className="history-item-bottom">
                    <span className={`history-status-chip ${item.status}`}>
                      {item.status === 'answered' ? '✓ Answered' : item.status === 'refused' ? '🛡️ Refused' : '⚠ Error'}
                    </span>
                    {item.retrievalMs != null && (
                      <span className="history-latency-chip">
                        ⚡ {item.retrievalMs.toFixed(1)}ms retrieval
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
