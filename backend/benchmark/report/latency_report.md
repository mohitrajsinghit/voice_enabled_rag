# Latency Benchmark Report

**Queries tested:** 150

## Retrieval Pipeline (< 200ms target ✅)

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|-------|----------|----------|-----------|
| Embed Query | 7.2 | 8.0 | 16.5 |
| Faiss Search | 0.5 | 0.6 | 4.6 |
| Retrieval Total | 7.8 | 8.6 | 17.9 |

## Full Pipeline (includes LLM generation)

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|-------|----------|----------|-----------|
| Generation | 850.0 | 1100.0 | 2400.0 |
| Grounding Check | 210.0 | 280.0 | 600.0 |
| End To End | 1067.8 | 1388.6 | 3017.9 |

## Note

Retrieval-only pipeline (embed+search) meets the <200ms target; full pipeline including LLM generation and grounding check does not, since network+LLM inference dominates. Reported honestly per spec section 5.