import React, { useState } from 'react';

/**
 * Interactive Architecture & System Specs Modal
 */
export default function ArchitectureModal({ onClose }) {
  const [activeTab, setActiveTab] = useState('pipeline'); // pipeline, guardrails, multilingual, latency, security

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="glass-panel modal-container" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header-bar">
          <div className="modal-title-group">
            <div className="modal-badge-icon">⚙️</div>
            <div>
              <h2 className="modal-title">System Architecture & Technical Specs</h2>
              <p className="modal-subtitle">End-to-end pipeline, 4-tier guardrails, edge security & telemetry</p>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close Architecture Modal">
            ✕
          </button>
        </div>

        {/* Tab Navigation (Smooth scrollable segment bar) */}
        <div className="modal-tabs-nav">
          <button
            className={`modal-tab-btn ${activeTab === 'pipeline' ? 'active' : ''}`}
            onClick={() => setActiveTab('pipeline')}
            type="button"
          >
            <span>⚡</span>
            <span>Pipeline</span>
          </button>
          <button
            className={`modal-tab-btn ${activeTab === 'guardrails' ? 'active' : ''}`}
            onClick={() => setActiveTab('guardrails')}
            type="button"
          >
            <span>🛡️</span>
            <span>Guardrails</span>
          </button>
          <button
            className={`modal-tab-btn ${activeTab === 'security' ? 'active' : ''}`}
            onClick={() => setActiveTab('security')}
            type="button"
          >
            <span>🔒</span>
            <span>Security</span>
          </button>
          <button
            className={`modal-tab-btn ${activeTab === 'multilingual' ? 'active' : ''}`}
            onClick={() => setActiveTab('multilingual')}
            type="button"
          >
            <span>🌐</span>
            <span>14 Languages</span>
          </button>
          <button
            className={`modal-tab-btn ${activeTab === 'latency' ? 'active' : ''}`}
            onClick={() => setActiveTab('latency')}
            type="button"
          >
            <span>📊</span>
            <span>Latency</span>
          </button>
        </div>

        {/* Modal Body */}
        <div className="modal-body-content">
          {/* TAB 1: PIPELINE FLOW */}
          {activeTab === 'pipeline' && (
            <div className="arch-section-group">
              <div className="spec-card">
                <div className="spec-card-header">
                  <span className="spec-step-num">01</span>
                  <h4>🎙️ Voice Ingestion & STT Layer</h4>
                  <span className="spec-tag">Sarvam AI + Web Audio API</span>
                </div>
                <p>
                  Real-time microphone capture via browser <code>MediaRecorder</code> (Opus WebM) with live 48-bar frequency waveform visualization. Streams directly to Sarvam AI STT for high-accuracy multilingual transcription across 14 Indic languages.
                </p>
              </div>

              <div className="spec-card">
                <div className="spec-card-header">
                  <span className="spec-step-num">02</span>
                  <h4>✂️ 4 Vast Chunking Strategies</h4>
                  <span className="spec-tag">Multi-Strategy Benchmark</span>
                </div>
                <p>
                  Evaluated across 4 distinct chunking paradigms on the MSMARCO-XI dataset:
                </p>
                <ul className="spec-bullet-list">
                  <li><strong>Semantic Chunking:</strong> Computes inter-sentence cosine similarity and splits at natural semantic topic shifts.</li>
                  <li><strong>Sentence-Window:</strong> Indexes focused central sentences while attaching surrounding sentences in metadata for rich context injection.</li>
                  <li><strong>Recursive Character:</strong> Hierarchical splitting by double newlines, single newlines, and sentence punctuation.</li>
                  <li><strong>Fixed-Size:</strong> Predictable token-bound chunks with sliding window overlap.</li>
                </ul>
              </div>

              <div className="spec-card">
                <div className="spec-card-header">
                  <span className="spec-step-num">03</span>
                  <h4>⚡ Vector Embedding Engine</h4>
                  <span className="spec-tag">paraphrase-multilingual-MiniLM-L12-v2</span>
                </div>
                <p>
                  384-dimensional dense vectors with multilingual semantic alignment. Generates high-fidelity embeddings across all 14 Indic scripts in ~20ms.
                </p>
              </div>

              <div className="spec-card">
                <div className="spec-card-header">
                  <span className="spec-step-num">04</span>
                  <h4>🔍 Fast 650k Vector Database</h4>
                  <span className="spec-tag">FAISS IndexFlatIP (37.4ms)</span>
                </div>
                <p>
                  Indexed 509,110 passages (649,545 dense vector chunks) with normalized inner-product cosine similarity. Sub-40ms top-k retrieval across the entire half-million dataset.
                </p>
              </div>

              <div className="spec-card">
                <div className="spec-card-header">
                  <span className="spec-step-num">05</span>
                  <h4>🧠 Natural Grounded Answer Generation</h4>
                  <span className="spec-tag">Groq / Gemini / LM Studio</span>
                </div>
                <p>
                  Generates clean, fluent answers in the exact language of the user's query. Output is rendered via progressive typewriter streaming, accompanied by dedicated source evidence cards with exact similarity scores.
                </p>
              </div>
            </div>
          )}

          {/* TAB 2: 4-TIER GUARDRAILS */}
          {activeTab === 'guardrails' && (
            <div className="arch-section-group">
              <div className="spec-highlight-banner">
                <strong>🛡️ Multi-Tier Defense Architecture:</strong> Ensures the system knows <em>when not to answer</em>, preventing hallucinations, blocking attacks, and saving cloud inference costs.
              </div>

              <div className="spec-card guardrail-tier">
                <div className="spec-card-header">
                  <span className="spec-step-num" style={{ background: 'rgba(239, 68, 68, 0.2)', color: '#f87171' }}>T1</span>
                  <h4>Tier 1: Input Regex & Jailbreak Shield</h4>
                  <span className="spec-tag" style={{ color: '#f87171', borderColor: 'rgba(239, 68, 68, 0.4)' }}>&lt;1ms Instant</span>
                </div>
                <p>
                  Blocks prompt injections, system prompt leak attempts, roleplay overrides, and harmful inputs before touching embeddings or LLM APIs.
                </p>
              </div>

              <div className="spec-card guardrail-tier">
                <div className="spec-card-header">
                  <span className="spec-step-num" style={{ background: 'rgba(245, 158, 11, 0.2)', color: '#fbbf24' }}>T2</span>
                  <h4>Tier 2: Corpus Centroid Distance Filter</h4>
                  <span className="spec-tag" style={{ color: '#fbbf24', borderColor: 'rgba(245, 158, 11, 0.4)' }}>Centroid Sim Gate</span>
                </div>
                <p>
                  Computes query cosine distance to the global dataset centroid vector. Queries scoring below threshold (e.g. <em>"What is your favorite movie?"</em>, <em>"How to bake cookies"</em>) are refused in ~20ms without invoking the LLM.
                </p>
              </div>

              <div className="spec-card guardrail-tier">
                <div className="spec-card-header">
                  <span className="spec-step-num" style={{ background: 'rgba(99, 102, 241, 0.2)', color: '#a5b4fc' }}>T3</span>
                  <h4>Tier 3: Retrieval Relevance Score Gate</h4>
                  <span className="spec-tag" style={{ color: '#a5b4fc', borderColor: 'rgba(99, 102, 241, 0.4)' }}>Quality Threshold</span>
                </div>
                <p>
                  Evaluates the top-1 FAISS match score. If the best retrieved passage lacks sufficient semantic confidence, the query is refused cleanly.
                </p>
              </div>

              <div className="spec-card guardrail-tier">
                <div className="spec-card-header">
                  <span className="spec-step-num" style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#34d399' }}>T4</span>
                  <h4>Tier 4: LLM Grounding & Hallucination Judge</h4>
                  <span className="spec-tag" style={{ color: '#34d399', borderColor: 'rgba(16, 185, 129, 0.4)' }}>Post-Generation Check</span>
                </div>
                <p>
                  Analyzes every factual sentence in the generated output against retrieved context chunks. Triggers an automated 1-shot strict regeneration if ungrounded claims are detected.
                </p>
              </div>
            </div>
          )}

          {/* TAB 3: EDGE SECURITY & ABUSE PREVENTION */}
          {activeTab === 'security' && (
            <div className="arch-section-group">
              <div className="spec-highlight-banner" style={{ borderColor: 'rgba(6, 182, 212, 0.4)', background: 'rgba(6, 182, 212, 0.08)' }}>
                <strong style={{ color: '#38bdf8' }}>🔒 Production Vercel & API Hardening:</strong> Multi-layered security controls designed to prevent cost spikes, DDoS, credential scraping, and abusive automated bots.
              </div>

              <div className="spec-card">
                <div className="spec-card-header">
                  <span className="spec-step-num" style={{ background: 'rgba(6, 182, 212, 0.2)', color: '#38bdf8' }}>S1</span>
                  <h4>5 Req/Min Sliding-Window Rate Limiter</h4>
                  <span className="spec-tag">Anti-DDoS & Cost Guard</span>
                </div>
                <p>
                  Enforces an exact sliding-window limit of <strong>5 queries per minute per client IP</strong> via backend middleware. If exceeded, the system returns HTTP 429 with Retry-After headers, triggering an interactive UI countdown modal.
                </p>
              </div>

              <div className="spec-card">
                <div className="spec-card-header">
                  <span className="spec-step-num" style={{ background: 'rgba(139, 92, 246, 0.2)', color: '#c084fc' }}>S2</span>
                  <h4>1-Min Voice Gauge & 500-Char Query Cap</h4>
                  <span className="spec-tag">Resource & Audio Guard</span>
                </div>
                <p>
                  Strict input limits protect server memory and STT cloud API credits:
                </p>
                <ul className="spec-bullet-list">
                  <li><strong>1-Minute Voice Cap:</strong> Voice recordings automatically stop at 60s, guided by an interactive <strong>Green (0–35s) → Yellow (35–50s) → Red (50–60s)</strong> dynamic countdown ring on the mic orb.</li>
                  <li><strong>Max 500 Characters:</strong> Text query input validation blocks oversized prompts and token exhaustion attacks.</li>
                </ul>
              </div>

              <div className="spec-card">
                <div className="spec-card-header">
                  <span className="spec-step-num" style={{ background: 'rgba(236, 72, 153, 0.2)', color: '#f472b6' }}>S3</span>
                  <h4>Zero Client Credential Exposure & CORS Locking</h4>
                  <span className="spec-tag">Zero Trust</span>
                </div>
                <p>
                  All provider credentials (<code>GROQ_API_KEY</code>, <code>SARVAM_API_KEY</code>) reside strictly in server-side environment variables. Reverse proxy routing and CORS headers lock API access to authorized deployment origins.
                </p>
              </div>

              <div className="spec-card">
                <div className="spec-card-header">
                  <span className="spec-step-num" style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#34d399' }}>S4</span>
                  <h4>UI Debounce & Client-Side Anti-Spam</h4>
                  <span className="spec-tag">Interactive Shield</span>
                </div>
                <p>
                  Disables query dispatching while requests are in-flight with a 2-second debounce on voice/text submission buttons, preventing double-click racing and repeated concurrent pipeline execution.
                </p>
              </div>
            </div>
          )}

          {/* TAB 4: 14 INDIC LANGUAGES */}
          {activeTab === 'multilingual' && (
            <div className="arch-section-group">
              <p style={{ color: 'var(--text-muted)', fontSize: '0.92rem' }}>
                Full cross-lingual retrieval coverage across all 14 Indic languages supported by <code>ai4bharat/MSMARCO-XI</code>:
              </p>

              <div className="indic-table-wrap">
                <table className="indic-table">
                  <thead>
                    <tr>
                      <th>Language</th>
                      <th>Code</th>
                      <th>Script</th>
                      <th>Retrieval Alignment</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>English</td><td><code>en</code></td><td>Latin</td><td><strong>90.5%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Hindi (हिन्दी)</td><td><code>hi</code></td><td>Devanagari</td><td><strong>85.3%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Bengali (বাংলা)</td><td><code>bn</code></td><td>Bengali</td><td><strong>85.1%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Tamil (தமிழ்)</td><td><code>ta</code></td><td>Tamil</td><td><strong>82.2%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Telugu (తెలుగు)</td><td><code>te</code></td><td>Telugu</td><td><strong>83.8%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Marathi (मराठी)</td><td><code>mr</code></td><td>Devanagari</td><td><strong>85.1%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Gujarati (ગુજરાતી)</td><td><code>gu</code></td><td>Gujarati</td><td><strong>84.8%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Kannada (ಕನ್ನಡ)</td><td><code>kn</code></td><td>Kannada</td><td><strong>84.0%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Malayalam (മലയാളം)</td><td><code>ml</code></td><td>Malayalam</td><td><strong>85.7%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Punjabi (ਪੰਜਾਬੀ)</td><td><code>pa</code></td><td>Gurmukhi</td><td><strong>83.6%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Urdu (اردو)</td><td><code>ur</code></td><td>Perso-Arabic</td><td><strong>83.1%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Odia (ଓଡ଼ିଆ)</td><td><code>or</code></td><td>Odia</td><td><strong>81.6%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Nepali (नेपाली)</td><td><code>ne</code></td><td>Devanagari</td><td><strong>85.9%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Sanskrit (संस्कृतम्)</td><td><code>sa</code></td><td>Devanagari</td><td><strong>85.1%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                    <tr><td>Assamese (অসমীয়া)</td><td><code>as</code></td><td>Bengali-Assamese</td><td><strong>79.5%</strong></td><td><span className="status-pass">PASS ✅</span></td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 5: LATENCY ANALYTICS */}
          {activeTab === 'latency' && (
            <div className="arch-section-group">
              <div className="spec-highlight-banner" style={{ borderColor: 'rgba(16, 185, 129, 0.4)', background: 'rgba(16, 185, 129, 0.08)' }}>
                <strong style={{ color: '#34d399' }}>⚡ Retrieval Subtotal Latency:</strong> <strong>7.8ms P50 / 8.6ms P70</strong> (Comfortably beating the <strong>&lt; 50ms target</strong> ✅ across 650,000 vectors).
              </div>

              <div className="indic-table-wrap">
                <table className="indic-table">
                  <thead>
                    <tr>
                      <th>Pipeline Stage</th>
                      <th>P50 Latency</th>
                      <th>P70 Latency</th>
                      <th>P100 Latency</th>
                      <th>SLA Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>ONNX Dense Query Embedding (<code>MiniLM-L12-v2</code>)</td>
                      <td>7.2 ms</td>
                      <td>8.0 ms</td>
                      <td>16.5 ms</td>
                      <td>⚡ Sub-10ms</td>
                    </tr>
                    <tr>
                      <td>FAISS IVF Vector Search (650,000 Vectors)</td>
                      <td>0.5 ms</td>
                      <td>0.6 ms</td>
                      <td>4.6 ms</td>
                      <td>🔍 Sub-Millisecond</td>
                    </tr>
                    <tr style={{ background: 'rgba(16, 185, 129, 0.15)', fontWeight: 700 }}>
                      <td><strong>⚡ Total Retrieval Pipeline Subtotal</strong></td>
                      <td><strong>7.8 ms</strong></td>
                      <td><strong>8.6 ms</strong></td>
                      <td><strong>17.9 ms</strong></td>
                      <td><span className="status-pass">&lt; 50ms Target Crushed ✅</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', lineHeight: 1.5, marginTop: 8 }}>
                *All retrieval and vector search operations complete in under <strong>8ms P50</strong> across 650,000 vectors using ONNX Runtime graph acceleration and FAISS <code>IndexIVFFlat</code>, satisfying all latency requirements with massive headroom.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
