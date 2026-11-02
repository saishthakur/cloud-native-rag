"""Wiring for the two deployment modes.

* ``LocalPlatform``: every service in one process, connected by direct calls and the in-memory
  bus. Used for development, tests and the evaluation harness (``cloudrag serve all``).
* ``build_service_app(name)``: one microservice per process, connected over HTTP and Redis
  Streams. This is what each container / Kubernetes Deployment runs.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager

from .clients.http import (HttpEmbeddingClient, HttpGenerationClient, HttpIndexingClient, HttpIngestionClient,
                           HttpRerankClient, HttpRetrievalClient)
from .config import Settings
from .core.models import Principal, RequestContext
from .embedding.embedders import EmbeddingService, build_embedder
from .gateway import Gateway
from .generation.generators import GenerationService, build_generator
from .indexing.service import TOPIC, IndexingService
from .indexing.store import IndexStore
from .ingestion.pipeline import IngestionService
from .messaging.bus import InMemoryBus, build_bus
from .reranking.rerankers import RerankService, build_reranker
from .retrieval.router import QueryRouter
from .retrieval.service import RetrievalService
from .security.audit import AuditLog
from .security.auth import TokenService
from .security.pii import PIIRedactor
from .security.provenance import ProvenanceSigner
from .services import apps

log = logging.getLogger(__name__)


class LocalPlatform:
    def __init__(self, settings: Settings | None = None, persist_audit: bool = False):
        s = settings or Settings.from_env()
        self.settings = s
        self.tokens = TokenService(s.secret_key, ttl_s=s.token_ttl_s)
        self.signer = ProvenanceSigner(s.provenance_key)
        self.bus = InMemoryBus()
        self.embedding = EmbeddingService(build_embedder(s.embedder, s.embed_dim))
        self.store = IndexStore(s, self.embedding.dim, shard=0)
        self.indexing = IndexingService(s.with_(shard_count=1), self.store, self.embedding, self.tokens)
        self.bus.subscribe(TOPIC, "indexing-shard-0", self.indexing.handle_event)
        self.audit = AuditLog(s.resolved_audit_path if persist_audit else None)
        pii = PIIRedactor(s.pii_mode, s.provenance_key)
        self.ingestion = IngestionService(s, self.bus, self.signer, self.audit, pii=pii)
        self.reranker = RerankService(build_reranker(s.reranker))
        self.router = QueryRouter()
        self.retrieval = RetrievalService(s, self.indexing, self.embedding, self.reranker, self.router, self.signer)
        self.generation = GenerationService(build_generator(s), PIIRedactor("redact"))
        self.gateway = Gateway(s, self.tokens, self.ingestion, self.retrieval, self.generation, self.indexing,
                               self.audit)

    def context(self, tenant_id: str, user_id: str = "user", roles: list[str] | None = None,
                clearance: int = 0) -> RequestContext:
        p = Principal(user_id=user_id, tenant_id=tenant_id, roles=roles or [], clearance=clearance)
        return RequestContext(principal=p, token=self.tokens.issue(p))

    def wait_indexed(self, timeout: float = 120.0) -> bool:
        return self.bus.wait_idle(timeout)

    def app(self):
        return apps.create_gateway_app(self.gateway)

    def close(self) -> None:
        self.bus.close()


SERVICES = ("gateway", "ingestion", "embedding", "indexing", "retrieval", "reranker", "generation")


def _require(url: str, name: str) -> str:
    if not url:
        raise SystemExit(f"CLOUDRAG_{name.upper()}_URL must be set in distributed mode")
    return url


def build_service_app(name: str, settings: Settings | None = None):
    s = (settings or Settings.from_env()).with_(service_name=name)
    tokens = TokenService(s.secret_key, ttl_s=s.token_ttl_s)
    signer = ProvenanceSigner(s.provenance_key)
    timeout = s.http_timeout_s

    if name == "embedding":
        return apps.create_embedding_app(EmbeddingService(build_embedder(s.embedder, s.embed_dim)), tokens)

    if name == "reranker":
        return apps.create_reranker_app(RerankService(build_reranker(s.reranker)), tokens)

    if name == "generation":
        return apps.create_generation_app(GenerationService(build_generator(s), PIIRedactor("redact")), tokens)

    if name == "retrieval":
        svc = RetrievalService(
            s,
            HttpIndexingClient(s.indexing_url_list, timeout),
            HttpEmbeddingClient(_require(s.embedding_url, "embedding"), timeout, name="embedding"),
            HttpRerankClient(_require(s.reranker_url, "reranker"), timeout, name="reranker"),
            QueryRouter(),
            signer,
        )
        return apps.create_retrieval_app(svc, tokens)

    if name == "ingestion":
        bus = build_bus(s.bus, s.redis_url)
        svc = IngestionService(s, bus, signer, AuditLog(s.resolved_audit_path),
                               pii=PIIRedactor(s.pii_mode, s.provenance_key))
        return apps.create_ingestion_app(svc, tokens)

    if name == "indexing":
        shard = s.resolved_shard_ordinal
        embed = HttpEmbeddingClient(_require(s.embedding_url, "embedding"), timeout, name="embedding")
        store = IndexStore(s, s.embed_dim, shard=shard)
        svc = IndexingService(s, store, embed, tokens)
        bus = build_bus(s.bus, s.redis_url)
        app = apps.create_indexing_app(svc, tokens)
        app.state.ready = False
        stop = threading.Event()

        def snapshot_loop():
            while not stop.wait(s.snapshot_interval_s):
                try:
                    svc.maybe_snapshot()
                except Exception:  # noqa: BLE001
                    log.exception("snapshot failed")

        @asynccontextmanager
        async def lifespan(_app):
            restored = store.load()
            log.info("shard %d restored %d tenant namespaces", shard, restored)
            bus.subscribe(TOPIC, f"indexing-shard-{shard}", svc.handle_event,
                          start_id=store.meta.get("last_event_id"), on_handled=svc.on_handled)
            threading.Thread(target=snapshot_loop, daemon=True).start()
            app.state.ready = True
            yield
            stop.set()
            bus.close()
            svc.maybe_snapshot(force=True)

        app.router.lifespan_context = lifespan
        return app

    if name == "gateway":
        if s.bus == "memory" and not s.retrieval_url:
            # single-process deployment: gateway hosts every service in-process
            return LocalPlatform(s, persist_audit=True).app()
        gw = Gateway(
            s, tokens,
            HttpIngestionClient(_require(s.ingestion_url, "ingestion"), timeout, name="ingestion"),
            HttpRetrievalClient(_require(s.retrieval_url, "retrieval"), timeout, name="retrieval"),
            HttpGenerationClient(_require(s.generation_url, "generation"), timeout, name="generation"),
            HttpIndexingClient(s.indexing_url_list, timeout),
            AuditLog(s.resolved_audit_path),
        )
        return apps.create_gateway_app(gw)

    raise SystemExit(f"unknown service {name!r}; choose one of {', '.join(SERVICES)}")


def default_port(name: str) -> int:
    return {"gateway": 8000, "ingestion": 8001, "embedding": 8002, "indexing": 8003, "retrieval": 8004,
            "reranker": 8005, "generation": 8006}.get(name, int(os.environ.get("PORT", 8000)))
