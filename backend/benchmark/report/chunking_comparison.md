# Chunking Strategy Comparison (Recall@k)

| Strategy | Recall@5 | Recall@10 | Recall@20 |
|----------|----------|-----------|-----------|
| semantic | 0.7520 | 0.9005 | 0.9395 |
| fixed | 0.7816 | 0.9052 | 0.9556 |
| sentence_window | 0.7137 | 0.8703 | 0.9335 |
| recursive | 0.7816 | 0.9052 | 0.9556 |

**Recommended default strategy: `fixed`** (best recall@5 = 0.7816)