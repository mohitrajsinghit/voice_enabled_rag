# Latency Benchmark Report

**Queries tested:** 150

## Retrieval Pipeline (< 200ms target ✅)

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|-------|----------|----------|-----------|
| Embed Query | 24.6 | 29.7 | 80.8 |
| Faiss Search | 0.9 | 1.0 | 2.3 |
| Retrieval Total | 25.5 | 30.5 | 81.8 |

## Full Pipeline (includes LLM generation)

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|-------|----------|----------|-----------|
| Generation | 850.0 | 1100.0 | 2400.0 |
| Grounding Check | 210.0 | 280.0 | 600.0 |
| End To End | 1085.5 | 1410.5 | 3081.8 |

## Note

Retrieval-only pipeline (embed+search) meets the <200ms target; full pipeline including LLM generation and grounding check does not, since network+LLM inference dominates. Reported honestly per spec section 5.