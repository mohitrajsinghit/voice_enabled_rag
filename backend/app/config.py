"""Voice RAG backend application configuration."""
from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    LMSTUDIO = "lmstudio"
    GEMINI = "gemini"
    GOOGLE = "google"


class EmbeddingProvider(str, Enum):
    LOCAL = "local"
    LMSTUDIO = "lmstudio"


class GroundingMode(str, Enum):
    LLM_JUDGE = "llm_judge"
    NLI = "nli"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- API Keys ---
    sarvam_api_key: str = Field(default="", description="Sarvam AI API key for STT")
    anthropic_api_key: str = Field(default="", description="Anthropic API key for Claude")
    google_api_key: str = Field(default="", description="Google Gemini API key")

    # --- LLM Provider ---
    llm_provider: LLMProvider = Field(default=LLMProvider.GEMINI, description="LLM provider: gemini, anthropic, or lmstudio")
    gemini_model: str = Field(default="gemini-3.1-flash-lite", description="Gemini model name")
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1", description="LM Studio API base URL")
    lmstudio_model: str = Field(default="local-model", description="LM Studio model name")
    anthropic_model: str = Field(default="claude-sonnet-4-6", description="Anthropic model name")

    # --- Embeddings ---
    embedding_provider: EmbeddingProvider = Field(default=EmbeddingProvider.LOCAL, description="Embedding provider: local or lmstudio")
    embedding_model: str = Field(default="paraphrase-multilingual-MiniLM-L12-v2", description="Local sentence transformer model")
    lmstudio_embedding_model: str = Field(default="text-embedding-qwen3-embedding-0.6b", description="LM Studio embedding model name")

    # --- FAISS Index ---
    faiss_index_path: str = Field(default="./data/processed/semantic/faiss.index", description="Path to FAISS index file")
    chunk_metadata_path: str = Field(default="./data/processed/semantic/chunks.jsonl", description="Path to chunk metadata JSONL")
    corpus_centroid_path: str = Field(default="./data/processed/semantic/centroid.npy", description="Path to corpus centroid embedding")

    # --- Retrieval ---
    default_chunk_strategy: str = Field(default="semantic", description="Default chunking strategy")
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    retrieval_score_threshold: float = Field(default=0.35, description="Minimum retrieval score threshold")
    enable_reranker: bool = Field(default=False, description="Enable cross-encoder reranking")

    # --- Guardrails ---
    grounding_mode: GroundingMode = Field(default=GroundingMode.LLM_JUDGE, description="Grounding check mode")
    off_topic_threshold: float = Field(default=0.25, description="Cosine similarity threshold for off-topic detection")

    # --- Logging ---
    log_level: str = Field(default="INFO", description="Logging level")

    # --- Server ---
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the project root."""
        p = Path(path)
        if p.is_absolute():
            return p
        return _PROJECT_ROOT / p


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
