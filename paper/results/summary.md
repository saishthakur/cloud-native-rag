# CloudRAG paper results

5 seeds (7, 11, 23, 42, 2026), 768 documents and 69 questions per seed, top-k 5. Values are mean ± std across seeds. Generated 2026-11-03 00:17.

## Performance comparison

| Method | Accuracy % | p50 ms | p95 ms | Throughput QPS | F1 | CPU ms/q | Energy J/q |
|---|---|---|---|---|---|---|---|
| Vanilla RAG | 54.1 ± 1.1 | 2.3 ± 0.4 | 4.0 | 317.6 ± 85.2 | 0.337 ± 0.005 | 2.9 | 0.007 |
| Advanced RAG | 54.5 ± 1.5 | 5.0 ± 0.6 | 7.1 | 160.5 ± 39.7 | 0.347 ± 0.012 | 5.6 | 0.014 |
| GraphRAG | 68.8 ± 3.0 | 2.9 ± 0.5 | 4.4 | 324.2 ± 99.1 | 0.401 ± 0.009 | 3.2 | 0.008 |
| Agentic RAG | 92.7 ± 1.6 | 18.5 ± 3.9 | 32.7 | 52.3 ± 5.1 | 0.478 ± 0.007 | 19.9 | 0.050 |
| CloudRAG (proposed) | 96.7 ± 1.1 | 7.5 ± 1.7 | 28.3 | 90.8 ± 12.5 | 0.493 ± 0.012 | 11.9 | 0.030 |

## Accuracy % by question type

| Method | Single-hop | Relational | Multi-hop | Multi-part | Aggregation |
|---|---|---|---|---|---|
| Vanilla RAG | 100.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 6.6 ± 6.6 |
| Advanced RAG | 100.0 ± 0.0 | 3.3 ± 4.6 | 0.0 ± 0.0 | 100.0 ± 0.0 | 4.4 ± 2.0 |
| GraphRAG | 93.3 ± 4.8 | 98.3 ± 3.7 | 8.3 ± 10.2 | 90.0 ± 6.3 | 4.9 ± 6.3 |
| Agentic RAG | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 31.1 ± 20.5 |
| CloudRAG (proposed) | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 69.4 ± 11.6 |

## Router decisions (all seeds)

| Question type | naive | advanced | graph | agentic |
|---|---|---|---|---|
| Single-hop | 120 | 0 | 0 | 0 |
| Relational | 0 | 0 | 60 | 0 |
| Multi-hop | 0 | 0 | 0 | 60 |
| Multi-part | 0 | 0 | 0 | 60 |
| Aggregation | 0 | 0 | 34 | 3 |

## Communication overhead (2 shards, real HTTP serialisation)

Ingestion: 2.0 internal requests and 6.1 KB per document, plus 1.16 KB on the event stream.

| Method | Internal requests/query | KB/query | p50 ms over HTTP |
|---|---|---|---|
| Vanilla RAG | 4.0 | 21.6 | 24.7 |
| Advanced RAG | 6.0 | 92.9 | 42.8 |
| GraphRAG | 4.0 | 30.5 | 24.8 |
| Agentic RAG | 19.6 | 350.4 | 189.5 |
| CloudRAG (proposed) | 12.4 | 201.8 | 68.1 |

## Security control cost

| Control | µs |
|---|---|
| pii_redaction_us_per_doc | 72.8 |
| poisoning_doc_signals_us_per_doc | 264.1 |
| poisoning_scan_us_per_chunk | 119.6 |
| provenance_sign_us_per_chunk | 16.0 |
| provenance_verify_us_per_chunk | 14.8 |
| query_guard_us_per_query | 23.1 |
| chunking_us_per_doc | 21.1 |
| ingest_security_us_per_doc | 481.0 |
| ingest_end_to_end_us_per_doc | 3945.9 |
| ingest_security_share | 12.2 % |
| query_security_us | 97.1 |
| query_security_share_of_p50 | 1.3 % |

## Security results

- cross_tenant_leakage_rate: 0.0 %
- rbac_leakage_rate: 0.0 %
- rbac_authorized_recall: 100.0 %
- pii_residual_rate: 0.0 %
- injection_block_rate: 100.0 %
- benign_false_positive_rate: 0.0 %
- probes: {'rbac_n': 60, 'rbac_authorized_n': 60, 'cross_tenant_n': 60}; PII values checked: 480; audit chains valid: True (11095 records)
- poisoned documents at ingestion: {'injection': {'quarantined': 45}, 'stuffing': {'flagged': 45}, 'misinformation': {'accepted': 45}}
- attacks (system:kind -> n / poison in context / successes): cloudrag:injection 45/0/0; cloudrag:misinformation 45/45/0; cloudrag:stuffing 45/45/0; vanilla-rag:injection 45/0/0; vanilla-rag:misinformation 45/45/0; vanilla-rag:stuffing 45/45/0

## Ingestion, memory, freshness

- ingestion 270 ± 71 docs/s end-to-end (in-process)
- index footprint 5.2 KB per chunk (4.1 MB for 816 chunks)
- process RSS 112 MB
- update freshness 100 %, ingest-to-searchable p50 7.3 ms

## Testbed

```
{
  "cpu": "12th Gen Intel(R) Core(TM) i3-1215U",
  "logical_cpus": 8,
  "os": "Windows 11",
  "python": "3.14.0",
  "ram_gb": 7.7,
  "physical_cores": 6,
  "libraries": {
    "numpy": "2.4.4",
    "fastapi": "0.162.0",
    "pydantic": "2.14.2"
  }
}
```
