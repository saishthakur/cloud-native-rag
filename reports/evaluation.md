# CloudRAG evaluation report

Tenants: acme, globex | seed 7 | top-k 5 | embedder `hashing` | index `hnsw` | reranker `lexical` | generator `extractive` (cost priced as `claude-opus-5-5`)

## Ingestion

192 documents, 224 chunks, 416.1 docs/s end-to-end (accepted in 0.12s, fully indexed in 0.46s).

## Quality and latency (all questions)

| system | hit@k | MRR | ctx precision | ctx recall | correct | faithful | relevance | p50 ms | p95 ms | p99 ms | est. $/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla-rag | 0.955 | 0.925 | 0.898 | 0.737 | 0.557 | 1.000 | 0.609 | 1.7 | 3.3 | 3.7 | 0.00257 |
| advanced-rag | 0.909 | 0.909 | 0.867 | 0.735 | 0.557 | 1.000 | 0.604 | 3.3 | 4.9 | 5.6 | 0.00266 |
| graphrag | 0.932 | 0.675 | 0.601 | 0.848 | 0.727 | 1.000 | 0.607 | 1.9 | 3.1 | 3.4 | 0.00276 |
| agentic-rag | 1.000 | 0.951 | 0.953 | 0.956 | 0.956 | 1.000 | 0.614 | 10.7 | 16.0 | 16.8 | 0.00279 |
| cloudrag | 0.932 | 0.904 | 0.857 | 0.989 | 0.989 | 1.000 | 0.612 | 4.8 | 16.7 | 24.9 | 0.00300 |

### Answer correctness by question type

| system | global | multihop | multipart | relational | single |
|---|---|---|---|---|---|
| vanilla-rag | 0.250 | 0.000 | 0.938 | 0.000 | 1.000 |
| advanced-rag | 0.000 | 0.000 | 0.938 | 0.125 | 1.000 |
| graphrag | 0.125 | 0.000 | 0.938 | 1.000 | 1.000 |
| agentic-rag | 0.521 | 1.000 | 1.000 | 1.000 | 1.000 |
| cloudrag | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 |

### Context recall by question type

| system | global | multihop | multipart | relational | single |
|---|---|---|---|---|---|
| vanilla-rag | 0.312 | 0.458 | 0.938 | 0.500 | 1.000 |
| advanced-rag | 0.000 | 0.542 | 0.938 | 0.562 | 1.000 |
| graphrag | 0.125 | 0.667 | 0.938 | 1.000 | 1.000 |
| agentic-rag | 0.521 | 1.000 | 1.000 | 1.000 | 1.000 |
| cloudrag | 0.875 | 1.000 | 1.000 | 1.000 | 1.000 |

## Router decisions (CloudRAG)

| question type | naive | advanced | graph | agentic |
|---|---|---|---|---|
| global | 0 | 0 | 3 | 1 |
| multihop | 0 | 0 | 0 | 8 |
| multipart | 0 | 0 | 0 | 8 |
| relational | 0 | 0 | 8 | 0 |
| single | 16 | 0 | 0 | 0 |

## Load (CloudRAG, in-process)

| concurrency | QPS | p50 ms | p95 ms | p99 ms | scaling efficiency |
|---|---|---|---|---|---|
| 1 | 126.7 | 5.2 | 20.1 | 24.9 | 1.00 |
| 4 | 127.5 | 22.2 | 75.9 | 112.1 | 0.25 |
| 16 | 106.4 | 106.0 | 397.2 | 589.7 | 0.05 |

## Incremental indexing

5 document updates: freshness 1.000, stale answers 0.000, ingest-to-searchable p50 3.7 ms / p95 3.9 ms.

## Security

- RBAC leakage rate (unauthorised readers): 0.000; authorised recall: 1.000
- Cross-tenant leakage rate: 0.000
- PII residue in index: 0.000 of 64 seeded values
- Injection queries blocked: 1.000; benign false positives: 0.000
- Audit chain: valid (1128 records)

Poisoned documents at ingestion:

- injection: quarantined=6
- stuffing: flagged=6
- misinformation: accepted=6

| system:attack | n | poison in context | attack success |
|---|---|---|---|
| cloudrag:injection | 6 | 0.000 | 0.000 |
| cloudrag:misinformation | 6 | 1.000 | 0.000 |
| cloudrag:stuffing | 6 | 1.000 | 0.000 |
| vanilla-rag:injection | 6 | 0.000 | 0.000 |
| vanilla-rag:misinformation | 6 | 1.000 | 0.000 |
| vanilla-rag:stuffing | 6 | 1.000 | 0.000 |

_Runtime: 7.2s. Synthetic corpus; numbers characterise this build, not production-scale behaviour._
