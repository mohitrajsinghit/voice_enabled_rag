"""Dual-provider embedding module supporting local sentence-transformers and LM Studio."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, TYPE_CHECKING
import numpy as np

from backend.app.config import EmbeddingProvider, get_settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BaseEmbedder(Protocol):
    """Protocol defining the embedder interface."""

    @property
    def dimension(self) -> int:
        ...

    def embed(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...


class LocalEmbedder:
    """Wrapper around sentence-transformers for in-process embedding."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the sentence transformer model with GPU acceleration when available."""
        if self._model is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu":
                try:
                    torch.set_num_threads(2)
                except Exception:
                    pass
            from sentence_transformers import SentenceTransformer
            dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
            logger.info(f"Loading local embedding model on {device.upper()} ({dev_name}): {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=device)
            self._model.eval()
            dim = self._model.get_sentence_embedding_dimension() if hasattr(self._model, 'get_sentence_embedding_dimension') else self._model.get_embedding_dimension()
            logger.info(f"Model loaded on {device.upper()}. Embedding dimension: {dim}")
        return self._model

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def embed(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        """Embed a list of texts in batches."""
        if not texts:
            return np.array([])

        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            formatted_texts = [prefix + t if not t.startswith(prefix) else t for t in texts]
        else:
            formatted_texts = texts

        import torch
        with torch.inference_mode():
            embeddings = self.model.encode(
                formatted_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        return np.array(embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        result = self.embed([text], show_progress=False, is_query=True)
        return result[0]

    def warmup(self) -> None:
        """Run a dummy embedding to pre-compile JIT graphs and eliminate cold-start."""
        logger.info("Warming up LocalEmbedder...")
        _ = self.embed_query("warmup probe")
        logger.info("LocalEmbedder warmup complete.")


# Backwards compatibility alias
Embedder = LocalEmbedder


class OnnxEmbedder:
    """ONNX Runtime accelerated embedder for fast CPU inference.

    Auto-exports the sentence-transformer model to ONNX format on first use
    and caches the exported model on disk for subsequent runs. Provides
    2-3x faster single-query inference compared to PyTorch on CPU.
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self._session = None
        self._tokenizer = None
        self._dim: int | None = None
        # Cache directory for exported ONNX models
        self._onnx_dir = Path.home() / ".cache" / "voice_rag_onnx" / model_name.replace("/", "_")

    def _ensure_onnx_model(self) -> Path:
        """Export the model to ONNX if not already cached, return the ONNX model path."""
        onnx_model_path = self._onnx_dir / "model.onnx"
        if onnx_model_path.exists():
            logger.info(f"ONNX model found at {onnx_model_path}")
            return onnx_model_path

        logger.info(f"Exporting {self.model_name} to ONNX (one-time, ~30-60s)...")
        self._onnx_dir.mkdir(parents=True, exist_ok=True)

        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            # Export using Hugging Face Optimum
            ort_model = ORTModelForFeatureExtraction.from_pretrained(
                self.model_name, export=True
            )
            ort_model.save_pretrained(str(self._onnx_dir))
            logger.info(f"ONNX model exported to {self._onnx_dir}")
        except Exception as e:
            logger.error(f"ONNX export failed: {e}. Attempting manual export...")
            self._manual_onnx_export(onnx_model_path)

        return onnx_model_path

    def _manual_onnx_export(self, onnx_path: Path) -> None:
        """Fallback: manually export using torch.onnx."""
        import torch
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name, device="cpu")
        model.eval()

        # Get the underlying transformer
        transformer = model[0].auto_model
        tokenizer = model.tokenizer

        dummy_input = tokenizer("warmup", return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            torch.onnx.export(
                transformer,
                (dummy_input["input_ids"], dummy_input["attention_mask"]),
                str(onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "last_hidden_state": {0: "batch", 1: "seq"},
                },
                opset_version=14,
            )

        # Save tokenizer alongside
        tokenizer.save_pretrained(str(self._onnx_dir))
        logger.info(f"ONNX model manually exported to {onnx_path}")

    def _load_session(self) -> None:
        """Load ONNX Runtime session and tokenizer."""
        if self._session is not None:
            return

        import onnxruntime as ort
        from transformers import AutoTokenizer

        onnx_model_path = self._ensure_onnx_model()

        # Find the actual .onnx file (optimum may save as model.onnx or model_optimized.onnx)
        onnx_files = list(self._onnx_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX file found in {self._onnx_dir}")
        onnx_file = onnx_files[0]

        # Configure session for optimal CPU performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        logger.info(f"Loading ONNX session from {onnx_file}")
        self._session = ort.InferenceSession(
            str(onnx_file),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._onnx_dir))
        if self._tokenizer is None:
            # Fallback: load from the original model name
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        logger.info("ONNX embedding session loaded successfully.")

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dim is None:
            sample = self.embed_query("dimension probe")
            self._dim = len(sample)
        return self._dim

    def _mean_pool(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Apply mean pooling to token embeddings, respecting attention mask."""
        # Expand attention mask to match embedding dims: (batch, seq) -> (batch, seq, dim)
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        mask_expanded = np.broadcast_to(mask_expanded, token_embeddings.shape)

        # Sum masked embeddings and divide by mask sum
        summed = np.sum(token_embeddings * mask_expanded, axis=1)
        counts = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        return summed / counts

    def embed(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = True,
        is_query: bool = False,
    ) -> np.ndarray:
        """Embed a list of texts using ONNX Runtime."""
        if not texts:
            return np.array([])

        self._load_session()

        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            texts = [prefix + t if not t.startswith(prefix) else t for t in texts]

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Tokenize
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )

            input_ids = encoded["input_ids"].astype(np.int64)
            attention_mask = encoded["attention_mask"].astype(np.int64)

            # Run ONNX inference
            feed = {"input_ids": input_ids, "attention_mask": attention_mask}

            # Handle different output names (optimum vs manual export)
            try:
                outputs = self._session.run(None, feed)
            except Exception as e:
                logger.error(f"ONNX inference error: {e}")
                raise

            # outputs[0] = last_hidden_state: (batch, seq, dim)
            token_embeddings = outputs[0]

            # Mean pooling
            sentence_embeddings = self._mean_pool(token_embeddings, encoded["attention_mask"].astype(np.float32))

            # L2 normalize
            norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            sentence_embeddings = sentence_embeddings / norms

            all_embeddings.append(sentence_embeddings)

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        result = self.embed([text], batch_size=1, show_progress=False, is_query=True)
        return result[0]

    def warmup(self) -> None:
        """Run dummy embeddings to pre-compile ONNX graphs and eliminate cold-start."""
        logger.info("Warming up OnnxEmbedder...")
        self._load_session()
        # Run a few dummy inferences to warm up ONNX Runtime's internal caches
        for _ in range(3):
            _ = self.embed_query("warmup probe for onnx runtime")
        logger.info("OnnxEmbedder warmup complete.")


class LMStudioEmbedder:
    """Wrapper for LM Studio embeddings endpoint (OpenAI-compatible)."""

    def __init__(
        self,
        model_name: str = "text-embedding-qwen3-embedding-0.6b",
        base_url: str = "http://localhost:1234/v1",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        """Get the embedding dimension from LM Studio."""
        if self._dim is None:
            sample = self.embed_query("dimension probe")
            self._dim = len(sample)
            logger.info(f"LM Studio embedding dimension: {self._dim} (model: {self.model_name})")
        return self._dim

    def embed(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts in batches using LM Studio /v1/embeddings."""
        if not texts:
            return np.array([])

        import httpx

        all_embeddings = []
        url = f"{self.base_url}/embeddings"

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                resp = httpx.post(
                    url,
                    json={"model": self.model_name, "input": batch},
                    timeout=60.0,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"LM Studio embedding error ({resp.status_code}): {resp.text}")

                data = resp.json()
                sorted_data = sorted(data["data"], key=lambda x: x.get("index", 0))
                batch_emb = [item["embedding"] for item in sorted_data]
                all_embeddings.extend(batch_emb)
            except Exception as e:
                logger.error(f"LM Studio embedding request failed: {e}")
                raise

        arr = np.array(all_embeddings, dtype=np.float32)
        # L2-normalize vectors so inner product == cosine similarity
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query text."""
        return self.embed([text], batch_size=1, show_progress=False)[0]

    def warmup(self) -> None:
        """Warmup for LMStudio embedder (no-op, network latency dominates)."""
        pass


# Module-level cached embedder factory
_embedder_cache: dict[str, BaseEmbedder] = {}


def get_embedder(
    model_name: str | None = None,
    provider: str | EmbeddingProvider | None = None,
    base_url: str | None = None,
) -> BaseEmbedder:
    """Get a cached embedder instance (ONNX, Local, or LM Studio).

    Args:
        model_name: Model identifier (optional, defaults to config).
        provider: "local" or "lmstudio" (optional, defaults to config).
        base_url: Base URL for LM Studio (optional, defaults to config).

    Returns:
        Embedder instance implementing BaseEmbedder.
    """
    settings = get_settings()
    prov = provider or settings.embedding_provider
    if isinstance(prov, str):
        prov = EmbeddingProvider(prov.lower())

    if prov == EmbeddingProvider.LMSTUDIO:
        m_name = model_name or settings.lmstudio_embedding_model
        b_url = base_url or settings.lmstudio_base_url
        cache_key = f"lmstudio:{b_url}:{m_name}"
        if cache_key not in _embedder_cache:
            _embedder_cache[cache_key] = LMStudioEmbedder(model_name=m_name, base_url=b_url)
        return _embedder_cache[cache_key]
    else:
        m_name = model_name or settings.embedding_model
        use_onnx = settings.use_onnx_embedding

        if use_onnx:
            cache_key = f"onnx:{m_name}"
            if cache_key not in _embedder_cache:
                logger.info(f"Using ONNX-accelerated embedder for {m_name}")
                _embedder_cache[cache_key] = OnnxEmbedder(model_name=m_name)
            return _embedder_cache[cache_key]
        else:
            cache_key = f"local:{m_name}"
            if cache_key not in _embedder_cache:
                _embedder_cache[cache_key] = LocalEmbedder(model_name=m_name)
            return _embedder_cache[cache_key]

