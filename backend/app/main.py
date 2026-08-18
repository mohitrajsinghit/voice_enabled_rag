"""FastAPI application: /query and /health endpoints."""
from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from backend.app.config import get_settings
from backend.app.generation.llm_client import LLMClient
from backend.app.guardrails.input_filter import InputFilter
from backend.app.guardrails.policy import GuardrailPolicy
from backend.app.harness.pipeline import VoiceRAGPipeline
from backend.app.indexing.embedder import get_embedder
from backend.app.indexing.faiss_store import FaissStore
from backend.app.retrieval.retriever import Retriever
from backend.app.schemas import PipelineResponse
from backend.app.stt.sarvam_client import SarvamClient
from backend.app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Sliding-window rate limiter: client request timestamp history
client_request_history: dict[str, list[float]] = defaultdict(list)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforces sliding-window rate limit per IP on query endpoints (configurable via .env)."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in ("/health", "/docs", "/openapi.json"):
            return await call_next(request)

        if request.url.path.startswith("/query"):
            client_ip = request.client.host if request.client else "unknown"
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                client_ip = forwarded_for.split(",")[0].strip()

            limit = int(os.getenv("RATE_LIMIT_REQUESTS", "5"))
            window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

            now = time.time()
            window_start = now - window_seconds

            # Keep only timestamps within the current sliding window
            timestamps = [ts for ts in client_request_history[client_ip] if ts > window_start]

            if len(timestamps) >= limit:
                oldest_in_window = timestamps[0]
                retry_after = max(1, int(oldest_in_window + window_seconds - now))
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": f"Rate limit exceeded: Maximum {limit} requests per minute allowed. Please wait {retry_after} seconds.",
                        "error_type": "rate_limit_exceeded",
                        "retry_after": retry_after,
                        "limit": limit,
                        "window_seconds": window_seconds,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            timestamps.append(now)
            client_request_history[client_ip] = timestamps

        return await call_next(request)


# Global pipeline instance
_pipeline: VoiceRAGPipeline | None = None
_index_loaded: bool = False
_chunk_count: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models and index at startup."""
    global _pipeline, _index_loaded, _chunk_count

    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting Voice RAG backend server...")

    # 1. Eagerly load dense embedder
    try:
        embedder = get_embedder()
        dim = embedder.dimension
        logger.info(f"✅ Embedder preloaded: dim={dim} (provider={settings.embedding_provider.value})")

        # Warmup: run dummy embeddings to eliminate cold-start latency spikes
        if hasattr(embedder, "warmup"):
            embedder.warmup()
            logger.info("✅ Embedder warmup complete — first-query latency eliminated")
    except Exception as e:
        logger.error(f"❌ Failed to load embedder: {e}", exc_info=True)
        raise

    # 2. Load FAISS index & chunk metadata from disk
    faiss_store = None
    try:
        index_path = settings.resolve_path(settings.faiss_index_path)
        metadata_path = settings.resolve_path(settings.chunk_metadata_path)

        if index_path.exists() and metadata_path.exists():
            faiss_store = FaissStore.load(str(index_path), str(metadata_path))
            _index_loaded = True
            _chunk_count = faiss_store.size
            logger.info(f"✅ FAISS index loaded: {_chunk_count} vectors (500k+ corpus ready)")
        else:
            logger.warning(
                f"⚠️ FAISS index not found at {index_path} or {metadata_path}. "
                "Creating empty index fallback."
            )
            import faiss
            empty_index = faiss.IndexFlatIP(dim)
            faiss_store = FaissStore(index=empty_index, chunks=[])
            _index_loaded = False
            _chunk_count = 0
    except Exception as e:
        logger.error(f"❌ Failed to load FAISS index from disk: {e}", exc_info=True)
        import faiss
        empty_index = faiss.IndexFlatIP(dim)
        faiss_store = FaissStore(index=empty_index, chunks=[])
        _index_loaded = False
        _chunk_count = 0

    # 3. Initialize STT & LLM clients
    try:
        retriever = Retriever(faiss_store, embedder)

        sarvam_key = os.getenv("SARVAM_API_KEY", "") or settings.sarvam_api_key
        stt_client = SarvamClient(api_key=sarvam_key)
        logger.info(f"✅ STT client initialized (Sarvam AI API key configured: {bool(sarvam_key and sarvam_key != 'your_sarvam_ai_api_key_here')})")

        llm_client = LLMClient()
        logger.info(f"✅ LLM client initialized: provider={llm_client.provider.value}, model={llm_client.model}")

        centroid_path = settings.resolve_path(settings.corpus_centroid_path)
        input_filter = InputFilter(
            embedder=embedder,
            centroid_path=str(centroid_path) if centroid_path.exists() else None,
            off_topic_threshold=settings.off_topic_threshold,
        )

        policy = GuardrailPolicy(
            llm_client=llm_client,
            retrieval_score_threshold=settings.retrieval_score_threshold,
        )

        _pipeline = VoiceRAGPipeline(
            stt_client=stt_client,
            input_filter=input_filter,
            retriever=retriever,
            llm_client=llm_client,
            policy=policy,
        )

        # Warmup retriever end-to-end (primes ONNX inference + FAISS index)
        retriever.retrieve("startup warmup probe", top_k=settings.top_k)
        logger.info("🚀 Voice RAG pipeline fully initialized and warmed up!")

    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline components: {e}", exc_info=True)
        _pipeline = None

    yield

    logger.info("Shutting down Voice RAG backend...")


# Create FastAPI app
app = FastAPI(
    title="Voice-Enabled RAG System",
    description="Voice-enabled RAG pipeline with Sarvam STT, multi-strategy retrieval, and grounded generation",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware: Sliding-window rate limit (5 req/min/IP) + CORS
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Models ────────────────────────────────────────────

class TextQueryRequest(BaseModel):
    """Text-mode query request (bypass STT)."""
    text: str = Field(..., max_length=500, description="Query text up to 500 characters")
    language: str = "en"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    index_loaded: bool
    chunk_count: int = 0
    message: str = ""


# ─── Endpoints ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if _pipeline else "degraded",
        index_loaded=_index_loaded,
        chunk_count=_chunk_count,
        message="" if _pipeline else "Pipeline not initialized",
    )


@app.post("/query", response_model=PipelineResponse)
async def query_audio(
    audio: UploadFile | None = File(None),
    text: str | None = Form(None),
    language: str = Form("hi-IN"),
):
    """Process a voice or text query through the RAG pipeline.

    Accepts either:
    - Multipart audio file upload (for STT processing)
    - Text form field (bypasses STT for testing)

    Returns a PipelineResponse with transcript, answer, sources,
    guardrail verdict, and per-stage latency breakdown.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    audio_bytes = None
    text_input = None

    if audio:
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
    elif text:
        text_input = text
    else:
        raise HTTPException(status_code=400, detail="Provide either audio file or text")

    response = await _pipeline.run(
        audio_bytes=audio_bytes,
        text_input=text_input,
    )

    return response


@app.post("/query/text", response_model=PipelineResponse)
async def query_text(request: TextQueryRequest):
    """Process a text query through the RAG pipeline (JSON body).

    Convenience endpoint that accepts JSON instead of multipart form.
    """
    if not _pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    response = await _pipeline.run(text_input=request.text)
    return response
