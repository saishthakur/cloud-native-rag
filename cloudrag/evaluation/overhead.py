"""Overhead measurements for the paper's "computational and communication overhead" analysis.

security_overhead      cost of each security control (PII redaction, poisoning scan, provenance
                       signing / verification, query guard) per document, chunk or query, and its
                       share of end-to-end ingestion and query time.
communication_overhead the microservice topology wired over real HTTP serialisation (Starlette's
                       in-process transport), counting internal service-to-service requests and
                       bytes (headers + bodies) per query for every retrieval strategy, plus the
                       event-stream bytes per ingested document.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter

from ..clients.http import (HttpEmbeddingClient, HttpGenerationClient, HttpIndexingClient, HttpIngestionClient,
                            HttpRerankClient, HttpRetrievalClient)
from ..config import Settings
from ..core.models import Principal, QueryRequest, RequestContext
from ..embedding.embedders import EmbeddingService, build_embedder
from ..gateway import Gateway
from ..generation.generators import GenerationService, build_generator
from ..indexing.service import TOPIC, IndexingService
from ..indexing.store import IndexStore
from ..ingestion.chunker import chunk_text
from ..ingestion.pipeline import IngestionService
from ..messaging.bus import InMemoryBus
from ..reranking.rerankers import RerankService, build_reranker
from ..retrieval.router import QueryRouter
from ..retrieval.service import RetrievalService
from ..security.audit import AuditLog
from ..security.auth import TokenService
from ..security.guard import QueryGuard
from ..security.pii import PIIRedactor
from ..security.poisoning import PoisoningDetector
from ..security.provenance import ProvenanceSigner
from ..services import apps


def _per_call_us(fn, items) -> float:
    t0 = time.perf_counter()
    for it in items:
        fn(it)
    return (time.perf_counter() - t0) * 1e6 / max(1, len(items))


def security_overhead(harness) -> dict:
    s = harness.settings
    docs = [d for tc in harness.corpora.values() for d in tc.docs]
    texts = [d.text for d in docs]
    chunks = [c for ns in (harness.platform.store.namespace(t) for t in harness.tenants) for c in ns.chunks.values()]
    pii = PIIRedactor(s.pii_mode, s.provenance_key)
    det = PoisoningDetector(s.poison_flag_threshold, s.poison_quarantine_threshold)
    signer = ProvenanceSigner(s.provenance_key)
    guard = QueryGuard(s.query_block_threshold, s.max_query_chars)
    questions = [q.question for tc in harness.corpora.values() for q in tc.questions]

    pii_us = _per_call_us(pii.redact, texts)
    doc_scan_us = _per_call_us(lambda t: det.document_signals("bench", f"d{hash(t)}", t), texts)
    chunk_scan_us = _per_call_us(lambda c: det.scan_chunk(c.text), chunks)
    sign_us = _per_call_us(lambda c: signer.sign(c, "bench", "bench"), chunks)
    verify_us = _per_call_us(signer.verify, chunks)
    guard_us = _per_call_us(guard.inspect, questions)
    chunk_us = _per_call_us(lambda t: chunk_text(t, s.chunk_tokens, s.chunk_overlap_sentences), texts)

    ing = harness.report["ingestion"]
    chunks_per_doc = ing["chunks"] / ing["documents"]
    ingest_security_us = pii_us + doc_scan_us + chunks_per_doc * (chunk_scan_us + sign_us)
    e2e_ingest_us = ing["index_complete_s"] * 1e6 / ing["documents"]
    k = harness.top_k
    query_security_us = guard_us + k * verify_us
    p50_us = harness.report["quality"]["cloudrag"]["overall"]["p50_ms"] * 1000
    return {
        "pii_redaction_us_per_doc": pii_us,
        "poisoning_doc_signals_us_per_doc": doc_scan_us,
        "poisoning_scan_us_per_chunk": chunk_scan_us,
        "provenance_sign_us_per_chunk": sign_us,
        "provenance_verify_us_per_chunk": verify_us,
        "query_guard_us_per_query": guard_us,
        "chunking_us_per_doc": chunk_us,
        "ingest_security_us_per_doc": ingest_security_us,
        "ingest_end_to_end_us_per_doc": e2e_ingest_us,
        "ingest_security_share": ingest_security_us / e2e_ingest_us,
        "query_security_us": query_security_us,
        "query_security_share_of_p50": query_security_us / p50_us,
    }


class _Meter:
    def __init__(self):
        self.requests = 0
        self.bytes = 0

    def attach(self, client):
        def on_request(req):
            self.requests += 1
            self.bytes += len(req.content or b"") + sum(len(k) + len(v) + 4 for k, v in req.headers.raw)

        def on_response(resp):
            resp.read()
            self.bytes += len(resp.content) + sum(len(k) + len(v) + 4 for k, v in resp.headers.raw)

        client.event_hooks = {"request": [on_request], "response": [on_response]}
        return client


def build_http_topology(settings: Settings):
    """Every service as its own FastAPI app, connected by the real HTTP clients."""
    from fastapi.testclient import TestClient

    s = settings
    meter = _Meter()
    tokens = TokenService(s.secret_key)
    signer = ProvenanceSigner(s.provenance_key)
    bus = InMemoryBus()
    stream_bytes = Counter()
    publish = bus.publish

    def metered_publish(topic, event):
        stream_bytes["bytes"] += len(json.dumps(event))
        stream_bytes["events"] += 1
        publish(topic, event)

    bus.publish = metered_publish

    def tc(app):
        return meter.attach(TestClient(app))

    embed = HttpEmbeddingClient("", client=tc(apps.create_embedding_app(
        EmbeddingService(build_embedder(s.embedder, s.embed_dim)), tokens)))
    shards = []
    for ordinal in range(s.shard_count):
        svc = IndexingService(s.with_(shard_ordinal=ordinal), IndexStore(s, s.embed_dim, shard=ordinal), embed, tokens)
        bus.subscribe(TOPIC, f"indexing-shard-{ordinal}", svc.handle_event)
        shards.append(tc(apps.create_indexing_app(svc, tokens)))
    index = HttpIndexingClient([], clients=shards)
    rerank = HttpRerankClient("", client=tc(apps.create_reranker_app(RerankService(build_reranker(s.reranker)), tokens)))
    retrieval = HttpRetrievalClient("", client=tc(apps.create_retrieval_app(
        RetrievalService(s, index, embed, rerank, QueryRouter(), signer), tokens)))
    generation = HttpGenerationClient("", client=tc(apps.create_generation_app(
        GenerationService(build_generator(s), PIIRedactor("redact")), tokens)))
    ingestion = HttpIngestionClient("", client=tc(apps.create_ingestion_app(
        IngestionService(s, bus, signer, AuditLog(None)), tokens)))
    gw = Gateway(s, tokens, ingestion, retrieval, generation, index, AuditLog(None))
    return gw, tokens, bus, meter, stream_bytes


def communication_overhead(harness, shard_count: int = 2) -> dict:
    s = harness.settings.with_(shard_count=shard_count)
    gw, tokens, bus, meter, stream = build_http_topology(s)

    def ctx(tenant, roles):
        p = Principal(user_id=f"u@{tenant}", tenant_id=tenant, roles=roles)
        return RequestContext(principal=p, token=tokens.issue(p))

    n_docs = 0
    for tc in harness.corpora.values():
        w = ctx(tc.tenant, ["ingest"])
        for d in tc.docs:
            gw.ingest(w, d)
            n_docs += 1
    bus.wait_idle(600)
    out = {"shards": shard_count, "ingest": {
        "internal_requests_per_doc": meter.requests / n_docs,
        "internal_bytes_per_doc": meter.bytes / n_docs,
        "stream_bytes_per_doc": stream["bytes"] / n_docs,
    }, "query": {}}

    from .harness import SYSTEMS

    for system, strategy in SYSTEMS.items():
        reqs, byts, lat = [], [], []
        for tc in harness.corpora.values():
            r = ctx(tc.tenant, ["employee"])
            for q in tc.questions:
                m0, b0, t0 = meter.requests, meter.bytes, time.perf_counter()
                gw.query(r, QueryRequest(query=q.question, strategy=strategy, top_k=harness.top_k))
                lat.append((time.perf_counter() - t0) * 1000)
                reqs.append(meter.requests - m0)
                byts.append(meter.bytes - b0)
        out["query"][system] = {
            "internal_requests_per_query": statistics.fmean(reqs),
            "internal_kb_per_query": statistics.fmean(byts) / 1024,
            "p50_ms_over_http": sorted(lat)[len(lat) // 2],
        }
    bus.close()
    return out
