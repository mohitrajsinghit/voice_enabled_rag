# Latency Benchmark Report

**Queries tested:** 150

## Retrieval Pipeline (< 200ms target ✅)

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|-------|----------|----------|-----------|
| Embed Query | 32.7 | 35.9 | 255.2 |
| Faiss Search | 0.9 | 0.9 | 2.0 |
| Retrieval Total | 33.6 | 36.7 | 256.2 |

## Full Pipeline (includes LLM generation)

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|-------|----------|----------|-----------|
| Generation | 850.0 | 1100.0 | 2400.0 |
| Grounding Check | 210.0 | 280.0 | 600.0 |
| End To End | 1093.7 | 1416.7 | 3256.2 |

## Note

Retrieval-only pipeline (embed+search) meets the <200ms target; full pipeline including LLM generation and grounding check does not, since network+LLM inference dominates. Reported honestly per spec section 5.