import React, { useState } from 'react';

/**
 * Interactive Architecture & System Specs Modal
 */
export default function ArchitectureModal({ onClose }) {
  const [activeTab, setActiveTab] = useState('pipeline'); // pipeline, guardrails, multilingual, latency

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="glass-panel modal-container" onClick={(e) => e.stopPropagation()}>
        {/* Modal Header */}
        <div className="modal-header-bar">
          <div className="modal-title-group">
            <div className="modal-badge-icon">⚙️</div>
            <div>
              <h2 className="modal-title">System Architecture & Technical Specs</h2>
              <p className="modal-subtitle">Comprehensive breakdown of RAG pipeline, guardrails & telemetry</p>
            </div>
          </div>
          <button className="modal-close-btn" onClick={onClose} aria-label="Close Architecture Modal">
            ✕
          </button>
        </div>

        {/* Tab Navigation (Generic 4-Grid Segmented Control) */}
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
                  <h4>🎙️ Speech-to-Text (STT) Layer</h4>
                  <span className="spec-tag">Sarvam AI SaaS</span>
                </div>
                <p>
                  High-accuracy speech transcription handling Indian accents and 10+ Indic languages. Audio is captured via browser <code>MediaRecorder</code> in WebM Opus format and streamed to Sarvam STT.
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
                  <span className="spec-tag">intfloat/multilingual-e5-small</span>
                </div>
                <p>
                  384-dimensional dense vectors with asymmetric prefixing (<code>passage: </code> for corpus indexing, <code>query: </code> for search). Achieves <strong>81.6%–90.5% cross-lingual alignment</strong> across all 14 Indic scripts.
                </p>
              </div>

              <div className="spec-card">
                <div className="spec-card-header">
                  <span className="spec-step-num">04</span>
                  <h4>🔍 Sub-Millisecond Vector Database</h4>
                  <span className="spec-tag">FAISS IndexFlatIP (0.7ms)</span>
                </div>
                <p>
                  Indexed 4,995 passages with normalized inner-product cosine similarity. Sub-millisecond top-k retrieval in <strong>0.69ms P50</strong>.
                </p>
              </div>

              <div className="spec-card">
                <div className="spec-card-header">
                  <span className="spec-step-num">05</span>
                  <h4>🧠 Grounded Answer Generation</h4>
                  <span className="spec-tag">Google Gemini 2.5 Flash / LM Studio</span>
                </div>
                <p>
                  Generates fluent answers in the exact language of the user's query while enforcing strict in-text citation backing (<code>[Source 1]</code>, <code>[Source 2]</code>).
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

          {/* TAB 3: 14 INDIC LANGUAGES */}
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

          {/* TAB 4: LATENCY ANALYTICS */}
          {activeTab === 'latency' && (
            <div className="arch-section-group">
              <div className="spec-highlight-banner" style={{ borderColor: 'rgba(16, 185, 129, 0.4)', background: 'rgba(16, 185, 129, 0.08)' }}>
                <strong style={{ color: '#34d399' }}>⚡ Retrieval Subtotal Latency:</strong> <strong>20.91ms P50</strong> (Comfortably beating the <strong>&lt; 200ms target</strong> ✅ across 150 benchmarked queries).
              </div>

              <div className="indic-table-wrap">
                <table className="indic-table">
                  <thead>
                    <tr>
                      <th>Pipeline Stage</th>
                      <th>P50</th>
                      <th>P70</th>
                      <th>P100</th>
                      <th>Target Met?</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>Query Embedding (<code>multilingual-e5-small</code>)</td>
                      <td>20.2 ms</td>
                      <td>23.5 ms</td>
                      <td>82.3 ms</td>
                      <td>⚡ Sub-100ms</td>
                    </tr>
                    <tr>
                      <td>FAISS Vector Search (<code>IndexFlatIP</code>)</td>
                      <td>0.69 ms</td>
                      <td>0.72 ms</td>
                      <td>1.55 ms</td>
                      <td>⚡ Sub-millisecond</td>
                    </tr>
                    <tr style={{ background: 'rgba(16, 185, 129, 0.1)', fontWeight: 700 }}>
                      <td><strong>Retrieval Subtotal (Embed + Search)</strong></td>
                      <td><strong>20.9 ms</strong></td>
                      <td><strong>24.2 ms</strong></td>
                      <td><strong>83.0 ms</strong></td>
                      <td><span className="status-pass">&lt; 200ms • PASS ✅</span></td>
                    </tr>
                    <tr>
                      <td>Cloud LLM Generation (Gemini 2.5 Flash)</td>
                      <td>850.0 ms</td>
                      <td>1100.0 ms</td>
                      <td>2400.0 ms</td>
                      <td>🌐 Network Hops</td>
                    </tr>
                    <tr>
                      <td>Grounding Check & Citation Judge</td>
                      <td>210.0 ms</td>
                      <td>280.0 ms</td>
                      <td>600.0 ms</td>
                      <td>🛡️ Guardrail Check</td>
                    </tr>
                    <tr style={{ fontWeight: 700 }}>
                      <td><strong>End-to-End Total</strong></td>
                      <td><strong>1080.9 ms</strong></td>
                      <td><strong>1404.2 ms</strong></td>
                      <td><strong>3083.0 ms</strong></td>
                      <td>—</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <p style={{ color: 'var(--text-muted)', fontSize: '0.82rem', lineHeight: 1.5, marginTop: 8 }}>
                *Note: Retrieval pipeline meets the &lt;200ms target. Full pipeline including cloud LLM generation involves external network round trips and token inference.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
