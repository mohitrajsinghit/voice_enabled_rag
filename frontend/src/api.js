/**
 * API client for the Voice RAG backend.
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

/**
 * Send an audio blob to the backend for processing.
 * @param {Blob} audioBlob - Recorded audio blob
 * @returns {Promise<object>} PipelineResponse
 */
export async function queryWithAudio(audioBlob) {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');

  const response = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Request failed' }));
    const err = new Error(errorData.detail || `HTTP ${response.status}`);
    err.status = response.status;
    err.isRateLimit = response.status === 429;
    err.retryAfter = errorData.retry_after || 60;
    throw err;
  }

  return response.json();
}

/**
 * Send a text query to the backend.
 * @param {string} text - Query text
 * @returns {Promise<object>} PipelineResponse
 */
export async function queryWithText(text) {
  const response = await fetch(`${API_BASE}/query/text`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: 'Request failed' }));
    const err = new Error(errorData.detail || `HTTP ${response.status}`);
    err.status = response.status;
    err.isRateLimit = response.status === 429;
    err.retryAfter = errorData.retry_after || 60;
    throw err;
  }

  return response.json();
}

/**
 * Check backend health.
 * @returns {Promise<object>} HealthResponse
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return response.json();
}
