# CloudRAG evaluation report

Tenants: initech, umbrella, hooli | seed 2026 | top-k 5 | embedder `hashing` | index `hnsw` | reranker `lexical` | generator `extractive` (cost priced as `claude-opus-5-5`)

## Ingestion

1368 documents, 1416 chunks, 124.7 docs/s end-to-end (accepted in 0.88s, fully indexed in 10.97s).

## Quality and latency (all questions)

| system | hit@k | MRR | ctx precision | ctx recall | correct | faithful | relevance | p50 ms | p95 ms | p99 ms | est. $/query |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla-rag | 0.925 | 0.891 | 0.863 | 0.715 | 0.522 | 0.985 | 0.586 | 3.4 | 6.1 | 7.4 | 0.00254 |
| advanced-rag | 0.910 | 0.899 | 0.847 | 0.741 | 0.567 | 1.000 | 0.597 | 9.2 | 16.4 | 20.5 | 0.00266 |
| graphrag | 0.910 | 0.524 | 0.523 | 0.843 | 0.724 | 0.985 | 0.578 | 5.0 | 7.1 | 9.1 | 0.00272 |
| agentic-rag | 0.940 | 0.889 | 0.881 | 0.929 | 0.929 | 1.000 | 0.597 | 24.1 | 42.5 | 48.2 | 0.00278 |
| cloudrag | 0.910 | 0.874 | 0.835 | 0.955 | 0.955 | 0.985 | 0.584 | 11.8 | 33.4 | 41.8 | 0.00299 |

### Answer correctness by question type

| system | global | multihop | multipart | relational | single |
|---|---|---|---|---|---|
| vanilla-rag | 0.000 | 0.000 | 1.000 | 0.000 | 0.958 |
| advanced-rag | 0.143 | 0.083 | 1.000 | 0.000 | 1.000 |
| graphrag | 0.286 | 0.000 | 0.958 | 1.000 | 0.958 |
| agentic-rag | 0.321 | 1.000 | 1.000 | 1.000 | 1.000 |
| cloudrag | 0.714 | 1.000 | 1.000 | 1.000 | 0.958 |

### Context recall by question type

| system | global | multihop | multipart | relational | single |
|---|---|---|---|---|---|
| vanilla-rag | 0.321 | 0.389 | 1.000 | 0.500 | 0.958 |
| advanced-rag | 0.143 | 0.556 | 1.000 | 0.500 | 1.000 |
| graphrag | 0.286 | 0.667 | 0.958 | 1.000 | 0.958 |
| agentic-rag | 0.321 | 1.000 | 1.000 | 1.000 | 1.000 |
| cloudrag | 0.714 | 1.000 | 1.000 | 1.000 | 0.958 |

## Router decisions (CloudRAG)

| question type | naive | advanced | graph | agentic |
|---|---|---|---|---|
| global | 0 | 0 | 6 | 1 |
| multihop | 0 | 0 | 0 | 12 |
| multipart | 0 | 0 | 0 | 12 |
| relational | 0 | 0 | 12 | 0 |
| single | 24 | 0 | 0 | 0 |

## Load (CloudRAG, in-process)

| concurrency | QPS | p50 ms | p95 ms | p99 ms | scaling efficiency |
|---|---|---|---|---|---|
| 1 | 62.2 | 11.5 | 44.8 | 53.7 | 1.00 |
| 4 | 58.6 | 47.9 | 146.1 | 245.0 | 0.24 |
| 16 | 42.3 | 171.1 | 1018.6 | 1198.1 | 0.04 |

## Incremental indexing

5 document updates: freshness 1.000, stale answers 0.000, ingest-to-searchable p50 11.5 ms / p95 12.8 ms.

## Security

- RBAC leakage rate (unauthorised readers): 0.000; authorised recall: 1.000
- Cross-tenant leakage rate: 0.000
- PII residue in index: 0.000 of 96 seeded values
- Injection queries blocked: 1.000; benign false positives: 0.000
- Audit chain: valid (2317 records)

Poisoned documents at ingestion:

- injection: quarantined=9
- stuffing: flagged=9
- misinformation: accepted=9

| system:attack | n | poison in context | attack success |
|---|---|---|---|
| cloudrag:injection | 9 | 0.000 | 0.000 |
| cloudrag:misinformation | 9 | 1.000 | 0.000 |
| cloudrag:stuffing | 9 | 1.000 | 0.000 |
| vanilla-rag:injection | 9 | 0.000 | 0.000 |
| vanilla-rag:misinformation | 9 | 1.000 | 0.000 |
| vanilla-rag:stuffing | 9 | 1.000 | 0.000 |

_Runtime: 24.6s. Synthetic corpus; numbers characterise this build, not production-scale behaviour._
