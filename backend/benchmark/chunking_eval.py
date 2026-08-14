"""Chunking strategy evaluation: recall@k comparison across strategies."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_chunking(
    strategies: list[str],
    k_values: list[int] = [5, 10, 20],
    queries_path: str | None = None,
    max_queries: int = 500,
) -> dict:
    """Evaluate chunking strategies by recall@k.

    Args:
        strategies: List of strategy names to compare.
        k_values: List of k values for recall@k.
        queries_path: Path to queries JSONL with relevant_doc_ids.
        max_queries: Maximum queries to evaluate.

    Returns:
        Dict of {strategy: {k: recall_score}}.
    """
    from backend.app.config import get_settings
    from backend.app.indexing.embedder import get_embedder
    from backend.app.indexing.faiss_store import FaissStore
    from backend.app.retrieval.retriever import Retriever

    settings = get_settings()
    data_dir = PROJECT_ROOT / "data"
    q_path = Path(queries_path) if queries_path else data_dir / "raw" / "queries.jsonl"

    # Load queries with ground truth
    queries = []
    with open(q_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("relevant_doc_ids"):
                    queries.append(record)
    queries = queries[:max_queries]
    logging.info(f"Loaded {len(queries)} queries with ground-truth relevance")

    if not queries:
        logging.error("No queries with relevant_doc_ids found")
        return {}

    embedder = get_embedder(settings.embedding_model)
    results = {}

    for strategy in strategies:
        logging.info(f"\nEvaluating strategy: {strategy}")
        index_path = data_dir / "processed" / strategy / "faiss.index"
        meta_path = data_dir / "processed" / strategy / "chunks.jsonl"

        if not index_path.exists():
            logging.warning(f"  Index not found for {strategy}, skipping")
            continue

        store = FaissStore.load(str(index_path), str(meta_path))
        retriever = Retriever(store, embedder)

        recall_at_k = {k: [] for k in k_values}

        for query_record in queries:
            query_text = query_record["query"]
            relevant_ids = set(query_record["relevant_doc_ids"])

            if not relevant_ids or not query_text.strip():
                continue

            max_k = max(k_values)
            retrieved, _ = retriever.retrieve(query_text, top_k=max_k)

            for k in k_values:
                top_k_results = retrieved[:k]
                retrieved_source_ids = {rc.chunk.source_doc_id for rc in top_k_results}
                hits = len(relevant_ids & retrieved_source_ids)
                recall = hits / len(relevant_ids)
                recall_at_k[k].append(recall)

        strategy_results = {}
        for k in k_values:
            scores = recall_at_k[k]
            avg = float(np.mean(scores)) if scores else 0.0
            strategy_results[f"recall@{k}"] = round(avg, 4)
            logging.info(f"  recall@{k} = {avg:.4f} ({len(scores)} queries)")

        results[strategy] = strategy_results

    return results


def save_comparison(results: dict, output_dir: Path) -> None:
    """Save comparison results as markdown table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Chunking Strategy Comparison (Recall@k)",
        "",
        "| Strategy | Recall@5 | Recall@10 | Recall@20 |",
        "|----------|----------|-----------|-----------|",
    ]

    best_strategy = None
    best_recall5 = -1

    for strategy, metrics in results.items():
        r5 = metrics.get("recall@5", 0)
        r10 = metrics.get("recall@10", 0)
        r20 = metrics.get("recall@20", 0)
        lines.append(f"| {strategy} | {r5:.4f} | {r10:.4f} | {r20:.4f} |")

        if r5 > best_recall5:
            best_recall5 = r5
            best_strategy = strategy

    if best_strategy:
        lines.extend([
            "",
            f"**Recommended default strategy: `{best_strategy}`** (best recall@5 = {best_recall5:.4f})",
        ])

    report_path = output_dir / "chunking_comparison.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logging.info(f"Saved comparison to {report_path}")

    # Also save as JSON
    json_path = output_dir / "chunking_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate chunking strategies")
    parser.add_argument("--strategies", type=str, default="semantic,fixed,sentence_window,recursive",
                       help="Comma-separated strategies")
    parser.add_argument("--k", type=str, default="5,10,20", help="Comma-separated k values")
    parser.add_argument("--queries-path", type=str, default=None)
    parser.add_argument("--max-queries", type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    strategies = [s.strip() for s in args.strategies.split(",")]
    k_values = [int(k.strip()) for k in args.k.split(",")]

    results = evaluate_chunking(
        strategies=strategies,
        k_values=k_values,
        queries_path=args.queries_path,
        max_queries=args.max_queries,
    )

    if results:
        output_dir = Path(__file__).parent / "report"
        save_comparison(results, output_dir)


if __name__ == "__main__":
    main()
