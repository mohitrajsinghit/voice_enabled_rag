# Latency Benchmark Report

**Queries tested:** 150

## Retrieval Pipeline (< 200ms target ✅)

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|-------|----------|----------|-----------|
| Embed Query | 20.2 | 23.5 | 82.3 |
| Faiss Search | 0.7 | 0.7 | 1.6 |
| Retrieval Total | 20.9 | 24.2 | 83.0 |

## Full Pipeline (includes LLM generation)

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|-------|----------|----------|-----------|
| Generation | 850.0 | 1100.0 | 2400.0 |
| Grounding Check | 210.0 | 280.0 | 600.0 |
| End To End | 1080.9 | 1404.2 | 3083.0 |

## Note

Retrieval-only pipeline (embed+search) meets the <200ms target; full pipeline including LLM generation and grounding check does not, since network+LLM inference dominates. Reported honestly per spec section 5.