"""FastAPI application: /query and /health endpoints."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Global pipeline instance
_pipeline: VoiceRAGPipeline | None = None
_index_loaded: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models and index at startup."""
    global _pipeline, _index_loaded

    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting Voice RAG backend...")

    try:
        # Eagerly load embedder at startup (not lazily on first request)
        embedder = get_embedder()
        dim = embedder.dimension
        logger.info(f"Embedder preloaded: dim={dim} (provider={settings.embedding_provider.value})")

        # Load FAISS index
        index_path = settings.resolve_path(settings.faiss_index_path)
        metadata_path = settings.resolve_path(settings.chunk_metadata_path)

        if index_path.exists() and metadata_path.exists():
            faiss_store = FaissStore.load(str(index_path), str(metadata_path))
            _index_loaded = True
            logger.info(f"FAISS index loaded: {faiss_store.size} vectors")
        else:
            logger.warning(
                f"FAISS index not found at {index_path}. "
                "Run 'python backend/app/indexing/build_index.py' first."
            )
            # Create empty store for graceful degradation
            import faiss
            empty_index = faiss.IndexFlatIP(dim)
            faiss_store = FaissStore(index=empty_index, chunks=[])
            _index_loaded = False

        # Initialize components
        retriever = Retriever(faiss_store, embedder)

        stt_client = SarvamClient(api_key=settings.sarvam_api_key) if settings.sarvam_api_key else SarvamClient(api_key="dummy")
        logger.info("STT client initialized")

        llm_client = LLMClient()
        logger.info(f"LLM client initialized: provider={llm_client.provider.value}, model={llm_client.model}")

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

        logger.info("Voice RAG pipeline initialized successfully!")

    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
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

# CORS middleware
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
    text: str
    language: str = "en"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    index_loaded: bool
    message: str = ""


# ─── Endpoints ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if _pipeline else "degraded",
        index_loaded=_index_loaded,
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
