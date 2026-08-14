import React from 'react';

/**
 * Latency breakdown badge showing per-stage millisecond timings.
 */

const STAGE_CONFIG = {
  transcribe_ms: { label: 'STT', className: 'stt' },
  retrieval_total_ms: { label: 'Retrieval', className: 'retrieval' },
  embed_query_ms: { label: 'Embed', className: 'retrieval' },
  faiss_search_ms: { label: 'Search', className: 'retrieval' },
  generation_ms: { label: 'Generation', className: 'generation' },
  grounding_check_ms: { label: 'Grounding', className: 'grounding' },
  end_to_end_ms: { label: 'Total', className: 'total' },
};

// Display priority order
const DISPLAY_ORDER = [
  'transcribe_ms',
  'retrieval_total_ms',
  'generation_ms',
  'grounding_check_ms',
  'end_to_end_ms',
];

export default function LatencyBadge({ latencies }) {
  if (!latencies || Object.keys(latencies).length === 0) return null;

  const pills = DISPLAY_ORDER
    .filter((key) => latencies[key] != null)
    .map((key) => {
      const config = STAGE_CONFIG[key];
      const value = latencies[key];
      return (
        <span key={key} className={`latency-pill ${config.className}`}>
          <span>{config.label}</span>
          <span>{value.toFixed(0)}ms</span>
        </span>
      );
    });

  if (pills.length === 0) return null;

  return (
    <div className="latency-section" id="latency-breakdown">
      {pills}
    </div>
  );
}
