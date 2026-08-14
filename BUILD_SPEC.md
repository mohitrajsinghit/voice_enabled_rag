# Voice-Enabled RAG System — Build Spec for Claude Code

Paste this whole file into  Code (in an empty project folder) and say:
"Build this project end to end, following the implementation order at the bottom.
Ask me for API keys when you need them, stub everything else, and run tests after each phase."

---

## 0. Goal (read this first)

Build a voice-enabled RAG pipeline:

**Mic audio → STT (Sarvam) → Query guardrail → Multi-strategy retrieval (FAISS) →
Grounded answer generation (LLM) → Grounding/hallucination guardrail → JSON response**

Dataset: `ai4bharat/MSMARCO-XI` (HuggingFace) — Indic MS MARCO passages/queries.

Hard requirements to satisfy in code (not just docs):
1. STT via Sarvam API (single provider, chosen over ElevenLabs for Indic ASR).
2. At least 3 chunking strategies implemented and selectable, with a comparison script.
3. Retrieval pipeline (embedding + FAISS search) P50/P70/P100 latency reported honestly.
   Full pipeline (incl. LLM generation) latency reported separately, honestly — do not
   fudge numbers to hit 200ms if generation doesn't fit; explain the split in README.
4. A latency benchmark script that runs 100+ queries and outputs percentiles + a chart.
5. A harness: typed pipeline stages, retries with backoff, structured I/O (Pydantic),
   error recovery, per-stage tracing/logging — not a single prompt-in/text-out call.
6. Guardrails: off-topic/out-of-domain query rejection, input safety filter, grounding
   check on generated answers (reject/hedge if not supported by retrieved context),
   explicit refusal path.
7. A minimal web frontend: record mic audio, hit the API, show transcript + answer +
   retrieved sources + latency breakdown.
8. Deployable: backend on Render/Railway/Fly, frontend on Vercel — produce a live link.

---

## 1. Tech stack

- **Language:** Python 3.11 (backend), plain HTML/JS or React+Vite (frontend — keep light)
- **API framework:** FastAPI + uvicorn
- **STT:** Sarvam AI API (`saarika` model) — REST call, async
- **Embeddings:** `sentence-transformers` — `all-MiniLM-L6-v2` (fast, 384-dim) for speed;
  optionally `intfloat/multilingual-e5-small` for better Indic quality — make this configurable
- **Vector index:** FAISS (`faiss-cpu`), HNSW index, in-memory, loaded from disk at startup
- **LLM (generation):** Anthropic Claude API (`claude-sonnet-4-6` or similar) via `anthropic` SDK —
  use for both generation and the grounding-check step / or give me another options to load from lmstudio local etc
- **Validation:** Pydantic v2 for all stage I/O
- **Retry/resilience:** `tenacity` for retries with exponential backoff
- **Tracing:** simple custom `Timer` context manager + structured JSON logs (no heavy APM needed)
- **Dataset handling:** `datasets` (HuggingFace) + `pandas`
- **Testing:** `pytest`
- **Frontend:** Vite + React (or vanilla JS if you want it simpler), `MediaRecorder` API for mic capture

---

## 2. Directory structure

```
voice-rag/
├── README.md
├── BUILD_SPEC.md                  # this file
├── .env.example
├── .gitignore
├── pyproject.toml                 # or requirements.txt
│
├── data/
│   ├── raw/                       # downloaded MSMARCO-XI cache (gitignored)
│   ├── processed/                 # chunked + embedded corpus artifacts (gitignored, or small sample committed)
│   └── download_dataset.py        # pulls ai4bharat/MSMARCO-XI, saves subset to raw/
│
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app, /health, /query (audio in -> answer out)
│   │   ├── config.py              # env vars, settings via pydantic-settings
│   │   ├── schemas.py             # Pydantic models for every stage I/O
│   │   │
│   │   ├── stt/
│   │   │   ├── sarvam_client.py   # wraps Sarvam STT API call, retry-wrapped
│   │   │   └── __init__.py
│   │   │
│   │   ├── chunking/
│   │   │   ├── base.py            # Chunker ABC: chunk(doc) -> list[Chunk]
│   │   │   ├── fixed_size.py      # fixed token window + overlap
│   │   │   ├── semantic.py        # embedding-similarity boundary splitting
│   │   │   ├── sentence_window.py # index sentence, retrieve window
│   │   │   ├── recursive.py       # paragraph -> sentence recursive splitter
│   │   │   ├── metadata_aware.py  # wraps any chunker, attaches metadata (source id, lang, etc.)
│   │   │   └── registry.py        # CHUNKERS = {"fixed": ..., "semantic": ..., ...}
│   │   │
│   │   ├── indexing/
│   │   │   ├── embedder.py        # sentence-transformers wrapper, batched embed()
│   │   │   ├── build_index.py     # script: chunk corpus with chosen strategy(ies) -> FAISS index + metadata store
│   │   │   └── faiss_store.py     # load/query FAISS index, returns chunk + metadata + score
│   │   │
│   │   ├── retrieval/
│   │   │   ├── retriever.py       # embed query -> FAISS search -> top-k chunks (this is the <200ms critical path)
│   │   │   └── reranker.py        # optional: cross-encoder rerank of top-k (flag to disable for latency)
│   │   │
│   │   ├── generation/
│   │   │   ├── llm_client.py      # Anthropic client wrapper, retry-wrapped, structured output
│   │   │   └── prompts.py         # system prompts for answer generation + grounding check
│   │   │
│   │   ├── guardrails/
│   │   │   ├── input_filter.py    # off-topic detection (corpus-centroid similarity threshold)
│   │   │   │                      #   + basic unsafe-content / prompt-injection pattern check
│   │   │   ├── grounding_check.py # NLI-style or LLM-judge check: is answer supported by chunks?
│   │   │   └── policy.py          # orchestrates: reject / hedge / answer decision
│   │   │
│   │   ├── harness/
│   │   │   ├── pipeline.py        # the state machine: transcribe -> guardrail -> retrieve ->
│   │   │   │                      #   generate -> ground-check -> respond, with typed stage results
│   │   │   ├── retry.py           # tenacity retry decorators/config shared across stages
│   │   │   └── tracing.py         # Timer context manager, per-stage latency capture -> structured log
│   │   │
│   │   └── utils/
│   │       └── logging_config.py
│   │
│   ├── benchmark/
│   │   ├── run_latency_bench.py   # runs N queries end-to-end, records per-stage timings
│   │   ├── queries_sample.json    # sampled queries from MSMARCO-XI for benchmarking
│   │   ├── chunking_eval.py       # recall@k comparison across chunking strategies
│   │   └── report/                # output: latency_report.md, percentiles.json, chart.png
│   │
│   ├── tests/
│   │   ├── test_chunking.py
│   │   ├── test_retrieval.py
│   │   ├── test_guardrails.py
│   │   ├── test_pipeline.py       # integration test, mocked STT/LLM
│   │   └── conftest.py
│   │
│   ├── Dockerfile
│   └── requirements.txt
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx                 # record button, waveform/status, results panel
        ├── components/
        │   ├── Recorder.jsx        # MediaRecorder wrapper -> blob -> POST /query
        │   ├── AnswerCard.jsx      # shows answer, sources, "not grounded" state
        │   └── LatencyBadge.jsx    # shows per-stage ms breakdown for this request
        └── api.js                  # fetch wrapper to backend
```

---

## 3. Key file contracts (so Claude Code implements the right interfaces)

### `backend/app/schemas.py` — define these Pydantic models
- `TranscriptResult(text: str, language: str, confidence: float, latency_ms: float)`
- `Chunk(id: str, text: str, source_doc_id: str, strategy: str, metadata: dict)`
- `RetrievedChunk(chunk: Chunk, score: float)`
- `GuardrailVerdict(passed: bool, reason: str | None, category: Literal["ok","off_topic","unsafe","ungrounded"])`
- `PipelineResponse(transcript: str, answer: str | None, sources: list[RetrievedChunk], guardrail: GuardrailVerdict, latencies: dict[str, float], status: Literal["answered","refused","error"])`

### `backend/app/chunking/base.py`
```python
class Chunker(ABC):
    name: str
    @abstractmethod
    def chunk(self, doc_id: str, text: str, metadata: dict | None = None) -> list[Chunk]: ...
```
Every strategy (fixed, semantic, sentence_window, recursive) implements this. `metadata_aware.py`
is a decorator/wrapper, not a separate splitting algorithm — it takes any Chunker and enriches
output metadata (passage_id, language, token_count, chunk_strategy).

### `backend/app/harness/pipeline.py`
This is the harness. It must NOT be "one function that calls STT then LLM." Structure it as:
```python
class PipelineStage(ABC):
    async def run(self, ctx: PipelineContext) -> PipelineContext: ...

class VoiceRAGPipeline:
    stages: list[PipelineStage]  # [Transcribe, InputGuardrail, Retrieve, Generate, GroundingGuardrail]
    async def run(self, audio_bytes: bytes) -> PipelineResponse:
        # iterate stages, catch per-stage exceptions, apply retry policy,
        # record Timer() per stage, short-circuit to refusal if a guardrail stage fails,
        # always return a PipelineResponse (never raise to the caller)
```
Each stage failure path must produce a typed error, not an unhandled exception — this satisfies
the "harness" requirement (structured orchestration, retries, error recovery).

### `backend/app/guardrails/policy.py`
Decision order:
1. `input_filter` — reject before spending retrieval/LLM cost if clearly off-topic or unsafe
2. `retrieve` — if top-k max score < threshold, treat as low-confidence retrieval → refuse
3. `generate`
4. `grounding_check` — if generated answer isn't supported by retrieved chunks → either
   regenerate once with a stricter "only use provided context" prompt, or refuse
This must be visible in the response (`status`, `guardrail.reason`) — the demo video should
show a query that gets refused.

---

## 4. Environment variables (`.env.example`)

```
SARVAM_API_KEY=
ANTHROPIC_API_KEY=
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=./data/processed/faiss.index
CHUNK_METADATA_PATH=./data/processed/chunks.jsonl
DEFAULT_CHUNK_STRATEGY=semantic
TOP_K=5
RETRIEVAL_SCORE_THRESHOLD=0.35
GROUNDING_MODE=llm_judge   # or "nli"
LOG_LEVEL=INFO
```

---

## 5. Commands Claude Code should set up and use

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

# 1. get data
python data/download_dataset.py --n-docs 5000   # keep subset small enough for fast local dev

# 2. build index (run once per chunking strategy to compare)
python backend/app/indexing/build_index.py --strategy semantic
python backend/app/indexing/build_index.py --strategy fixed
python backend/app/indexing/build_index.py --strategy sentence_window

# 3. eval chunking strategies (recall@k)
python backend/benchmark/chunking_eval.py --strategies semantic,fixed,sentence_window --k 5

# 4. run API locally
uvicorn backend.app.main:app --reload --port 8000

# 5. run latency benchmark (100+ queries, writes P50/P70/P100 report)
python backend/benchmark/run_latency_bench.py --n-queries 150

# 6. run tests
pytest backend/tests -v

# frontend
cd frontend && npm install && npm run dev
```

---

## 6. `requirements.txt` (pin roughly, let Claude Code resolve exact versions)

```
fastapi
uvicorn[standard]
pydantic
pydantic-settings
anthropic
sentence-transformers
faiss-cpu
tenacity
datasets
pandas
numpy
python-multipart
httpx
matplotlib          # for latency chart
pytest
pytest-asyncio
```

---

## 7. Latency benchmark output format (`backend/benchmark/report/percentiles.json`)

```json
{
  "n_queries": 150,
  "stages": {
    "embed_query_ms":   {"p50": 4.2,  "p70": 5.1,  "p100": 12.8},
    "faiss_search_ms":  {"p50": 1.1,  "p70": 1.6,  "p100": 6.3},
    "retrieval_total_ms": {"p50": 5.3, "p70": 6.7, "p100": 18.9},
    "generation_ms":    {"p50": 850.0,"p70": 1100.0,"p100": 2400.0},
    "grounding_check_ms": {"p50": 210.0,"p70": 280.0,"p100": 600.0},
    "end_to_end_ms":    {"p50": 1080.0,"p70": 1400.0,"p100": 3100.0}
  },
  "note": "Retrieval-only pipeline (embed+search) meets the <200ms target; full pipeline including LLM generation and grounding check does not, since network+LLM inference dominates. Reported honestly per spec section 5."
}
```
README must explain this split plainly — don't claim <200ms for the full LLM round trip.

---

## 8. README.md must include

- Architecture diagram (ASCII is fine)
- Which chunking strategies were implemented, and results of the recall@k comparison
  (this is your evidence of "real thought put into chunking")
- Latency table (P50/P70/P100) with the retrieval-vs-full-pipeline distinction explained
- Guardrail examples: one off-topic query → refused, one in-domain query → answered with sources
- Setup instructions
- Deployment links (backend + frontend)

---

## 9. Deployment targets

- Backend: Render or Railway (Docker deploy using `backend/Dockerfile`), expose `/query` and `/health`
- Frontend: Vercel, `VITE_API_BASE_URL` env var pointing at backend
- Keep the FAISS index + chunk metadata as build artifacts baked into the Docker image
  (small subset of MSMARCO-XI, e.g. 5–10k passages) so cold start doesn't require rebuilding

---

## 10. Implementation order (tell Claude Code to follow this exactly)

1. Scaffold directory structure + empty files matching section 2
2. `data/download_dataset.py` — pull a manageable subset of `ai4bharat/MSMARCO-XI`
3. `chunking/` — implement `fixed_size`, `semantic`, `sentence_window`, `recursive`,
   `metadata_aware` wrapper, and `registry.py`
4. `indexing/` — embedder + `build_index.py` + `faiss_store.py`; build indexes for all strategies
5. `benchmark/chunking_eval.py` — recall@k comparison → pick default strategy, document result
6. `retrieval/retriever.py` — query embed + FAISS search, must be fast (this is the <200ms path)
7. `stt/sarvam_client.py` — Sarvam API integration (ask user for API key)
8. `generation/` — Anthropic client + prompts (answer generation + grounding-judge prompt)
9. `guardrails/` — input filter, grounding check, policy orchestration
10. `harness/` — pipeline state machine wiring all of the above with retries + tracing
11. `main.py` — FastAPI `/query` (multipart audio upload) and `/health`
12. `tests/` — unit tests per module + one integration test with mocked STT/LLM
13. `benchmark/run_latency_bench.py` — run 150 queries, produce percentiles.json + chart + report
14. `frontend/` — Recorder → API call → AnswerCard + LatencyBadge
15. Dockerfile + deploy backend, deploy frontend, wire live URLs into README
16. Final pass: verify guardrail refusal path is demonstrable for the demo video

---

## 11. Things NOT to fake

- Don't hardcode a "200ms" number in the report — measure it for real, and if the full
  pipeline (incl. LLM call) exceeds it, say so and explain why, per section 7.
- Don't skip the grounding guardrail — the spec explicitly asks the system to show it
  knows when *not* to answer.
- Don't submit a single fixed-size chunker with cosmetic parameter changes as "multiple
  strategies" — `semantic.py` and `sentence_window.py` need genuinely different logic.
