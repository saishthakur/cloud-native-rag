# CloudRAG

A reference implementation of **CloudRAG: a cloud-native, microservices-based retrieval-augmented
generation framework for scalable, secure, multi-tenant enterprise knowledge platforms**
(Paranjape, Chandra, Narayana, Thakur - Manipal Institute of Technology, Phase 2).

It implements the six research objectives of the proposal:

| # | Objective | Where |
|---|---|---|
| 1 | Kubernetes-orchestrated architecture with independently deployable ingestion, embedding, indexing, retrieval, reranking, generation and observability | `cloudrag/services/apps.py`, `cloudrag/runtime.py`, `deploy/k8s/` |
| 2 | Query-complexity-aware router selecting naive / advanced / graph / agentic retrieval | `cloudrag/retrieval/router.py` |
| 3 | Hybrid retrieval: dense HNSW + BM25 + knowledge-graph traversal, reciprocal-rank fusion, cross-scoring reranker | `cloudrag/indexing/`, `cloudrag/retrieval/`, `cloudrag/reranking/` |
| 4 | Tenant-scoped security: namespace isolation, document- and chunk-level RBAC, ingestion-time PII redaction, provenance, poisoning detection, audit logging | `cloudrag/security/`, `cloudrag/ingestion/` |
| 5 | Observability and evaluation harness: retrieval quality, faithfulness, relevance, latency, throughput, cost, security events | `cloudrag/observability/`, `cloudrag/evaluation/` |
| 6 | Evaluation against vanilla, advanced, GraphRAG and agentic RAG baselines | `cloudrag/evaluation/harness.py`, `reports/` |

## Architecture

```
                 ┌──────────────────────────── gateway (public API) ────────────────────────────┐
  client ──TLS──▶│ JWT auth · per-tenant rate limit · injection filter · orchestration · audit  │
                 └───────┬───────────────────────────┬──────────────────────────────┬───────────┘
                         │ POST /internal/ingest     │ POST /internal/retrieve      │ POST /internal/generate
                 ┌───────▼────────┐          ┌───────▼─────────┐            ┌───────▼─────────┐
                 │   ingestion    │          │    retrieval    │            │   generation    │
                 │ PII redaction  │          │ router          │            │ Claude (Opus    │
                 │ poisoning scan │          │ naive/advanced/ │            │ 5.5) + breaker  │
                 │ ACL chunking   │          │ graph/agentic   │            │ + extractive    │
                 │ provenance sig │          │ RRF · integrity │            │   fallback      │
                 └───────┬────────┘          └──┬─────┬─────┬──┘            └─────────────────┘
                         │ Redis Streams        │     │     │
                         │ (cloudrag.chunks)    │     │     └──────▶ reranker
                 ┌───────▼───────────────────┐  │     └────────────▶ embedding ◀─────┐
                 │ indexing-0 … indexing-N   │◀─┘                                    │
                 │ (StatefulSet, tenant-     │───────────────────────────────────────┘
                 │  sharded) per tenant:     │
                 │  HNSW · BM25 · KG · ACLs  │       every service: /metrics /healthz /readyz
                 └───────────────────────────┘
```

* **Event-driven ingestion.** Ingestion returns `202` once a versioned `chunks.upsert` event is on the
  stream; indexing shards consume it asynchronously, so write bursts never block query serving.
  Upserts are versioned and idempotent, so replays and out-of-order delivery are harmless.
* **Tenant sharding.** Each indexing pod owns the tenants that hash (SHA-256) to its ordinal and keeps
  one physically separate namespace per tenant: its own HNSW graph, BM25 postings and knowledge
  graph. Shards snapshot to their volume together with the last processed stream id and resume from
  it after a restart.
* **Zero-trust hops.** The caller's token is forwarded on every hop and each service verifies it.
  The indexing service takes the tenant from the verified token, never from the request body.
  Service identities cannot read tenant data, and a shard refuses tenants it does not own.
* **Same code, two modes.** `LocalPlatform` runs every service in one process (development, tests,
  harness). `cloudrag serve <service>` runs one service per process over HTTP and Redis. Calling
  code is identical because the HTTP clients mirror the in-process service interfaces.

### Retrieval pathways

| strategy | what it does | chosen by the router for |
|---|---|---|
| `naive` | dense HNSW top-k | short single-fact lookups |
| `advanced` | dense + BM25 in parallel → RRF → rerank | descriptive / keyword-heavy questions |
| `graph` | dense + BM25 + relation-aware KG traversal (+ community search for aggregate questions) → weighted RRF → rerank, keeping the strongest graph hits | relational and aggregation questions |
| `agentic` | plan (decompose) → per-sub-question graph retrieval → reflect on coverage → bridge through newly found entities (≤3 hops) → fuse | multi-part questions and multi-hop chains |
| `graphrag` | pure graph local + community search (baseline only) | – |

The router uses cheap features: length, linked entities, relational and aggregation cues, and
multi-part structure. It then enforces a caller-supplied `latency_budget_ms` using live per-strategy
p95 latencies, downgrading along agentic → graph → advanced → naive. Every response carries the
decision and its reasons.

### Security controls

* **Isolation:** per-tenant namespaces; ACLs are applied *during* search, so restricted chunks never
  displace or leak through ranking. Tenant, ACL and signature are re-checked before generation.
* **RBAC:** document-level `acl` plus inline section policies (`[[acl roles=finance clearance=2]] … [[/acl]]`).
  A chunk carries every policy that applies, and all of them must allow the reader.
* **PII:** email, phone, card numbers (Luhn-checked), SSN, Aadhaar, PAN, IBAN and IP addresses are
  redacted (or keyed-pseudonymised) before chunking, and answers are re-scanned.
* **Provenance:** an HMAC over tenant, doc, chunk id, content hash, ACL, source and trust. Tampered
  chunks are dropped at retrieval and raise a security event.
* **Poisoning:** prompt-injection phrases, role/markup markers, invisible Unicode, keyword stuffing
  and near-duplicate floods are combined into a risk score. High-risk chunks are quarantined,
  medium-risk ones are indexed with lower trust, and the generator strips instruction-like
  sentences. Source-based trust ceilings (`CLOUDRAG_SOURCE_TRUST`) let systems of record outrank
  ad-hoc uploads when facts conflict.
* **Input filtering:** injection and exfiltration queries are rejected (`422`) and audited.
* **Audit:** a hash-chained, tamper-evident JSONL log of token issuance, ingestion, queries (PII-redacted)
  and security events; `GET /v1/security/audit/verify` walks the chain.

## Quick start

```bash
pip install -e ".[all]"
```

Run the test suite (47 tests):

```bash
python -m pytest
```

Run the evaluation harness and write `reports/evaluation.{md,json}`:

```bash
cloudrag eval
```

Single-process server on :8000 (all services in one process):

```bash
cloudrag serve all
```

Get a development token and ask a question:

```bash
curl -s localhost:8000/v1/auth/token -H "x-admin-key: dev-admin-key" -H "content-type: application/json" -d '{"tenant_id":"acme","user_id":"alice","roles":["ingest","employee"]}'
```

```bash
curl -s localhost:8000/v1/documents -H "authorization: Bearer $TOKEN" -H "content-type: application/json" -d '{"doc_id":"zx200","title":"Product sheet","source":"plm://products","text":"The ZX-200 Sensor carries a warranty period of 18 months."}'
```

```bash
curl -s localhost:8000/v1/query -H "authorization: Bearer $TOKEN" -H "content-type: application/json" -d '{"query":"What is the warranty period of the ZX-200 Sensor?"}'
```

Distributed on one machine (Redis + 7 services, 2 index shards; needs Docker running):

```bash
docker compose up --build
```

Kubernetes (base; then optionally the Istio mesh and Prometheus Operator overlays):

```bash
kubectl apply -k deploy/k8s/base
```

```bash
kubectl apply -k deploy/k8s/mesh && kubectl apply -k deploy/k8s/monitoring
```

### Using Claude for generation

The default generator is extractive: it answers offline by chaining cited evidence sentences, so
everything runs without an API key. To generate with Claude, set `CLOUDRAG_LLM=anthropic` (model
`claude-opus-5-5` by default; `CLOUDRAG_ANTHROPIC_MODEL` and `CLOUDRAG_ANTHROPIC_EFFORT` override it)
and provide credentials via `ANTHROPIC_API_KEY` or an `ant auth login` profile. Requests enable
server-side refusal fallbacks (`fallbacks="default"`). Each answer reports token usage and USD cost,
and on rate limits or 5xx errors a circuit breaker degrades to the extractive generator. The k8s
ConfigMap sets `CLOUDRAG_LLM=anthropic`.

### Configuration

Every field in `cloudrag/config.py` maps to an environment variable `CLOUDRAG_<FIELD>`. Important ones:
`SECRET_KEY`, `PROVENANCE_KEY`, `ADMIN_BOOTSTRAP_KEY` (secrets: set them!), `DEV_AUTH` (turn off
when an IdP issues tokens), `BUS`, `REDIS_URL`, `SHARD_COUNT`, `INDEXING_URLS`, `EMBEDDER`
(`hashing` or `sentence-transformers:<model>`; `EMBED_DIM` must match), `RERANKER` (`lexical` or
`cross-encoder:<model>`), `VECTOR_INDEX` (`hnsw` / `flat`), `PII_MODE`, `SOURCE_TRUST`,
`POISON_QUARANTINE_THRESHOLD`, `QUERY_BLOCK_THRESHOLD`, `RATE_LIMIT_RPS`.

## Results

The harness builds a deterministic synthetic enterprise corpus: vendors, products, projects and
people, with facts spread across documents. It also adds finance-restricted sections, HR records
containing PII, distractor documents, poisoned documents and adversarial queries. Every system
shares the same ingestion, index and generator, so differences come from the retrieval pathway.

**Held-out run** (`reports/heldout/`: seed 2026, tenants never used during development, 1,368
documents, extractive generator, top-k 5):

| system | answer correctness | context recall | p50 ms | p95 ms |
|---|---|---|---|---|
| vanilla RAG | 0.522 | 0.715 | 3.4 | 6.1 |
| advanced RAG | 0.567 | 0.741 | 9.2 | 16.4 |
| GraphRAG | 0.724 | 0.843 | 5.0 | 7.1 |
| agentic RAG | 0.929 | 0.929 | 24.1 | 42.5 |
| **CloudRAG (router)** | **0.955** | **0.955** | 11.8 | 33.4 |

By question type, CloudRAG matches the best baseline on single-fact, multi-part, relational and
multi-hop questions. On aggregation ("which projects use products supplied by X?") it scores 0.714
versus 0.321 for the next best system, because the router widens the context budget for that type.

Security (held-out run): RBAC leakage 0, cross-tenant leakage 0, PII residue 0 of 96 seeded
values, injection queries blocked 5/5 with 0/5 benign false positives, all 9 instruction-injection
documents quarantined, keyword-stuffing documents flagged, 0 successful poisoning attacks, and the
audit chain verified over 2,317 records. Incremental updates were searchable in about 12 ms with
fresh answers 5/5.

### Distributed run (Docker Compose)

The full stack ran under Docker Compose (Redis, seven services, two index shards; 8-CPU Docker
Desktop VM on Windows), with the synthetic corpus ingested and queried over HTTP:

* **Placement and quality:** tenants landed on different shards (acme → shard 0, globex → shard 1).
  All 192 documents were searchable 3.4 s after the batch upload. Answer correctness was
  1.000 (acme) and 0.976 (globex), and RBAC held: finance budget lines hidden from staff, visible to finance.
* **Graceful restart** of a shard restored its snapshot (112 → 112 chunks).
* **SIGKILL of a shard** right after ingesting a new document lost nothing (114 → 114 chunks; the
  memory-only document was answered correctly after restart), because the shard replays the
  Redis stream from its snapshot offset.
* **Fault isolation:** while a shard is down, only its tenants are affected. Their queries return
  `503 indexing-N unavailable`: about 8 s each for the first five (Docker's DNS is slow to report
  a missing container), then about 5 ms once the circuit breaker opens. The other tenant keeps
  being served, and service resumes about 7 s after the shard is back.
* **Throughput:** 12–17 QPS with one replica per service, all requests successful; p50 59 ms at 1
  client, about 1.2 s at 32. The bottleneck is retrieval at one core (GIL-bound). Scaling it to 3
  replicas moved the bottleneck to the reranker and the hot tenant's index shard (19.5 QPS), with
  the 8-CPU machine itself near saturation. Each HTTP hop also costs far more than an in-process
  call: the agentic path makes many of them.

### Limitations: read before citing these numbers

* **The synthetic corpus was used during development.** Router thresholds, the stemmer and the
  evidence-chaining generator were tuned while looking at it. The held-out run changes seed,
  tenants and scale but keeps the same document templates and question types, so it is optimistic
  for real enterprise text. A real dataset (e.g. a TREC RAG track subset or an internal corpus with
  labelled questions) is the next step.
* **The default embedder and reranker are lexical** (feature hashing and an interaction scorer). They
  need no model downloads and are deterministic, but they are not semantic. Production
  deployments should set `sentence-transformers:` and `cross-encoder:` models; their impact has not
  been measured here.
* **Faithfulness and relevance are lexical estimators**, not LLM-judged RAGAS scores. The extractive
  generator is faithful by construction, so faithfulness ≈ 1.0 says little about an LLM generator.
  Run `cloudrag eval --llm anthropic` (this calls the Claude API and costs money) to measure real
  generation; the cost column already prices each prompt at Claude Opus 5.5 list rates.
* **The poisoning defences are platform-wide.** Quarantine, trust and sanitisation live in the shared
  ingestion and generation layers, so the vanilla baseline is protected too. The "misinformation"
  attack (a plain false claim) is not detected at all; it fails only because its untrusted source
  ranks below the system of record. A false claim from a trusted source would succeed.
* **Throughput is CPU-bound pure Python.** One process tops out around 60–100 QPS in-process and
  about 20 QPS through the distributed stack on one machine (see above). Multi-node scaling was not
  measured. The 1M–10M document, 100–1,000-concurrent-query experiments in the proposal need a
  cluster, a compiled ANN index (the HNSW here is pure Python/NumPy) and multiple uvicorn workers
  per pod.
* **A hot tenant is bounded by one index shard.** Tenant sharding gives isolation, but all of one
  tenant's searches hit a single pod. Read replicas per shard (several consumers of the same
  stream, each holding a copy) would lift this; they are not implemented.
* **Kubernetes was not deployed.** The manifests render with `kubectl kustomize` and the same image
  runs under Docker Compose as described above, but no cluster run (HPA, NetworkPolicies, Istio)
  was performed.

## Repository layout

```
cloudrag/
  core/            data contracts, tokenisation/stemming, entity & relation extraction
  security/        JWT auth + rate limiting, PII, provenance, poisoning, query guard, audit log
  ingestion/       ACL-aware chunker, ingestion pipeline
  embedding/       embedders + embedding service
  indexing/        HNSW / flat vector index, BM25, knowledge graph, tenant store, indexing service
  retrieval/       router, RRF, retrieval strategies
  reranking/       rerankers
  generation/      extractive + Claude generators, prompt construction, cost accounting
  messaging/       in-memory and Redis Streams event bus
  clients/         HTTP clients, circuit breaker, retries
  observability/   Prometheus metrics
  evaluation/      metrics, synthetic corpus, harness
  services/        FastAPI apps per microservice
  gateway.py       public API orchestration
  runtime.py       local vs distributed wiring
  cli.py           serve / token / eval / loadtest
deploy/k8s/        base (namespace, config, redis, services+HPA+PDB, indexing StatefulSet, network policies, ingress),
                   mesh (Istio mTLS, authorization, circuit breaking), monitoring (PodMonitor, alerts)
deploy/observability/  Prometheus config and alert rules for Docker Compose
reports/           dev and held-out evaluation reports
tests/             unit, security, end-to-end, HTTP topology and bus tests
```
