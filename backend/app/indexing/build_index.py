"""Build FAISS index from chunked corpus.

CLI script that:
1. Loads passages from data/raw/passages.jsonl
2. Chunks with the chosen strategy
3. Embeds all chunks in batches
4. Builds a FAISS index (IndexFlatIP for normalized vectors)
5. Saves index + chunk metadata to data/processed/{strategy}/
6. Computes and saves corpus centroid for off-topic detection
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import faiss
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.chunking.registry import get_chunker, list_strategies
from backend.app.indexing.embedder import get_embedder


logger = logging.getLogger(__name__)


def load_passages(passages_path: Path) -> list[dict]:
    """Load passages from JSONL file."""
    passages = []
    with open(passages_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                passages.append(json.loads(line))
    return passages


def build_index(
    strategy: str = "semantic",
    passages_path: str | None = None,
    output_dir: str | None = None,
    embedding_model: str | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    batch_size: int = 128,
    max_passages: int | None = None,
    **chunker_kwargs,
) -> None:
    """Build FAISS index for a given chunking strategy.

    Args:
        strategy: Chunking strategy name.
        passages_path: Path to passages JSONL file.
        output_dir: Output directory for index + metadata.
        embedding_model: Embedding model name.
        provider: "local" or "lmstudio".
        base_url: LM Studio URL (if provider=lmstudio).
        batch_size: Embedding batch size (larger on GPU).
        max_passages: Maximum number of passages to process.
        **chunker_kwargs: Additional args for the chunker.
    """
    # Resolve paths
    data_dir = PROJECT_ROOT / "data"
    p_path = Path(passages_path) if passages_path else data_dir / "raw" / "passages.jsonl"
    out_dir = Path(output_dir) if output_dir else data_dir / "processed" / strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load passages
    logger.info(f"Loading passages from {p_path}")
    passages = load_passages(p_path)
    if max_passages and max_passages > 0:
        passages = passages[:max_passages]
        logger.info(f"Capped passages to {len(passages)} as requested")
    else:
        logger.info(f"Loaded {len(passages)} passages")

    # Initialize chunker
    logger.info(f"Using chunking strategy: {strategy}")
    chunker = get_chunker(strategy, wrap_metadata=True, **chunker_kwargs)

    # Chunk all passages
    logger.info("Chunking passages...")
    all_chunks = []
    t0 = time.time()
    for passage in passages:
        doc_id = passage["doc_id"]
        text = passage["text"]
        meta = {
            "source_lang": passage.get("source_lang", ""),
            "target_lang": passage.get("target_lang", ""),
            "query_id": passage.get("query_id", ""),
            "is_selected": passage.get("is_selected", 0),
        }
        chunks = chunker.chunk(doc_id, text, metadata=meta)
        all_chunks.extend(chunks)

    chunk_time = time.time() - t0
    logger.info(f"Created {len(all_chunks)} chunks in {chunk_time:.1f}s")

    if not all_chunks:
        logger.error("No chunks created! Check passage data.")
        return

    # Extract texts for embedding
    chunk_texts = [c.text for c in all_chunks]

    # Initialize embedder and embed
    embedder = get_embedder(model_name=embedding_model, provider=provider, base_url=base_url)
    logger.info(f"Embedding {len(chunk_texts)} chunks with provider={provider or 'default'}")
    t0 = time.time()
    embeddings = embedder.embed(chunk_texts, batch_size=batch_size)
    embed_time = time.time() - t0
    logger.info(f"Embedded in {embed_time:.1f}s ({len(chunk_texts)/embed_time:.0f} chunks/sec)")

    # Build FAISS index (Inner Product for normalized vectors = cosine similarity)
    dim = embeddings.shape[1]
    logger.info(f"Building FAISS IndexFlatIP (dim={dim}, n={len(embeddings)})")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save index
    index_path = out_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index to {index_path} ({index.ntotal} vectors)")

    # Save chunk metadata
    metadata_path = out_dir / "chunks.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.model_dump_json() + "\n")
    logger.info(f"Saved chunk metadata to {metadata_path}")

    # Compute and save corpus centroid (for off-topic detection)
    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)  # normalize
    centroid_path = out_dir / "centroid.npy"
    np.save(str(centroid_path), centroid)
    logger.info(f"Saved corpus centroid to {centroid_path}")

    # Summary
    logger.info(
        f"\n{'='*60}\n"
        f"Index Build Summary ({strategy})\n"
        f"{'='*60}\n"
        f"  Passages:          {len(passages)}\n"
        f"  Chunks:            {len(all_chunks)}\n"
        f"  Embedding dim:     {dim}\n"
        f"  Chunking time:     {chunk_time:.1f}s\n"
        f"  Embedding time:    {embed_time:.1f}s\n"
        f"  Index path:        {index_path}\n"
        f"  Metadata path:     {metadata_path}\n"
        f"  Centroid path:     {centroid_path}\n"
        f"{'='*60}"
    )


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from chunked corpus")
    parser.add_argument(
        "--strategy",
        type=str,
        default="semantic",
        choices=list_strategies(),
        help=f"Chunking strategy. Available: {', '.join(list_strategies())}",
    )
    parser.add_argument("--passages-path", type=str, default=None, help="Path to passages JSONL")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["local", "lmstudio"],
        help="Embedding provider: 'local' (sentence-transformers) or 'lmstudio'",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Embedding model name",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="LM Studio base URL (e.g. http://192.168.68.201:1234/v1)",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    parser.add_argument("--max-passages", type=int, default=None, help="Maximum number of passages to process (e.g. 2000)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    build_index(
        strategy=args.strategy,
        passages_path=args.passages_path,
        output_dir=args.output_dir,
        embedding_model=args.embedding_model,
        provider=args.provider,
        base_url=args.base_url,
        batch_size=args.batch_size,
        max_passages=args.max_passages,
    )


if __name__ == "__main__":
    main()
