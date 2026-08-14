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
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
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
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
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
