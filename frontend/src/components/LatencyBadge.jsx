import React from 'react';

/**
 * Latency stages configuration and labels
 */
const STAGE_CONFIG = {
  embed_query_ms: { label: 'Query Embed (e5)', classKey: 'embed', icon: '⚡' },
  faiss_search_ms: { label: 'FAISS Search', classKey: 'search', icon: '🔍' },
  retrieval_total_ms: { label: 'Retrieval Total', classKey: 'retrieval', icon: '⚡' },
};

const DISPLAY_STAGES = [
  'embed_query_ms',
  'faiss_search_ms',
  'retrieval_total_ms',
];

/**
 * Precision Latency Telemetry Waterfall Component
 */
export default function LatencyBadge({ latencies }) {
  if (!latencies || Object.keys(latencies).length === 0) return null;

  const retrievalMs = latencies.retrieval_total_ms ?? (
    (latencies.embed_query_ms || 0) + (latencies.faiss_search_ms || 0)
  );
  const isSub200 = retrievalMs > 0 && retrievalMs < 200;

  return (
    <div className="telemetry-hud" id="latency-telemetry-hud">
      <div className="telemetry-header">
        <div className="telemetry-title">
          <span>⚡ High-Precision Telemetry HUD</span>
        </div>
        {isSub200 && (
          <span className="telemetry-target-badge">
            Retrieval &lt;200ms Target Met ({retrievalMs.toFixed(1)}ms) ✅
          </span>
        )}
      </div>

      <div className="waterfall-pills-row">
        {DISPLAY_STAGES.filter((k) => latencies[k] != null && latencies[k] > 0).map((stageKey) => {
          const config = STAGE_CONFIG[stageKey];
          const val = latencies[stageKey];
          return (
            <div key={stageKey} className={`telemetry-pill ${config.classKey}`}>
              <span className="pill-stage-name">
                {config.icon} {config.label}
              </span>
              <span className="pill-stage-value">
                {val < 1 ? val.toFixed(2) : val.toFixed(1)} <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-dim)' }}>ms</span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
