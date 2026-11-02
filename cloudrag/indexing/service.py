"""Indexing service: consumes ingestion events, embeds chunks, and serves search primitives.

Tenants are sharded across indexing replicas (StatefulSet pods) by a stable hash; every shard
sees every event on the bus and keeps only its own tenants. Search endpoints take the tenant
from the caller's verified token, never from the request payload.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time

import numpy as np

from ..config import Settings
from ..core.models import Chunk, GraphHits, Principal, RequestContext, ScoredChunk
from ..observability.metrics import CHUNKS_INDEXED, INDEX_SIZE, SECURITY_EVENTS, STAGE_LATENCY
from ..security import ForbiddenError
from ..security.auth import TokenService
from .store import IndexStore, TenantNamespace

log = logging.getLogger(__name__)
TOPIC = "cloudrag.chunks"


def shard_for(tenant_id: str, shard_count: int) -> int:
    # SHA-256 rather than CRC32: CRC's low bits are nearly constant for short ASCII ids, which
    # would pile every tenant onto one shard whenever shard_count is even.
    digest = hashlib.sha256(tenant_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % max(1, shard_count)


class IndexingService:
    def __init__(self, settings: Settings, store: IndexStore, embedder, tokens: TokenService):
        self.settings = settings
        self.store = store
        self.embedder = embedder
        self.shard = store.shard
        self.shard_count = settings.shard_count
        self._svc = RequestContext(
            principal=Principal(user_id="svc:indexing", tenant_id="*", roles=["service"]),
            token=tokens.service_token("indexing"),
        )
        self._snap_lock = threading.Lock()
        self._last_snapshot = time.time()

    # -- write path (bus) -------------------------------------------------------
    def owns(self, tenant_id: str) -> bool:
        return shard_for(tenant_id, self.shard_count) == self.shard

    def handle_event(self, event: dict) -> None:
        p = event["payload"]
        tenant = p["tenant_id"]
        if not self.owns(tenant):
            return
        ns = self.store.namespace(tenant)
        if event["type"] == "chunks.upsert":
            chunks = [Chunk.model_validate(c) for c in p["chunks"]]
            if any(c.tenant_id != tenant for c in chunks):
                SECURITY_EVENTS.inc(type="cross_tenant_write_rejected")
                raise ValueError("chunk tenant does not match event tenant")
            with STAGE_LATENCY.time(stage="index_upsert"):
                vecs = self.embedder.embed(self._svc, [c.embed_text for c in chunks]) if chunks else np.zeros((0, 1))
                ns.upsert_document(p["doc_id"], chunks, vecs, int(p["version"]))
            CHUNKS_INDEXED.inc(len(chunks))
        elif event["type"] == "doc.delete":
            ns.delete_document(p["doc_id"], int(p["version"]))
        INDEX_SIZE.set(self.store.total_chunks(), shard=str(self.shard))

    def on_handled(self, event_id: str) -> None:
        self.store.meta["last_event_id"] = event_id
        self.maybe_snapshot()

    def maybe_snapshot(self, force: bool = False) -> None:
        if not force and time.time() - self._last_snapshot < self.settings.snapshot_interval_s:
            return
        with self._snap_lock:
            self.store.save()
            self._last_snapshot = time.time()

    # -- read path ----------------------------------------------------------------
    def _ns(self, ctx: RequestContext) -> TenantNamespace:
        p = ctx.principal
        if p.is_service or p.tenant_id in ("", "*"):
            raise ForbiddenError("search requires a tenant-scoped user identity")
        if not self.owns(p.tenant_id):
            raise ForbiddenError(f"tenant {p.tenant_id} is not served by shard {self.shard}")
        return self.store.namespace(p.tenant_id)

    def _wrap(self, ctx: RequestContext, pairs, source: str) -> list[ScoredChunk]:
        out = []
        for rank, (chunk, score, *extra) in enumerate(pairs):
            if chunk.tenant_id != ctx.principal.tenant_id:  # defence in depth: never expected
                SECURITY_EVENTS.inc(type="tenant_isolation_violation")
                log.error("tenant isolation violation: %s served to %s", chunk.chunk_id, ctx.principal.tenant_id)
                continue
            signals = {f"{source}_score": float(score), f"{source}_rank": float(rank + 1)}
            if extra:
                signals["community"] = float(extra[0])
            out.append(ScoredChunk(chunk=chunk, score=float(score), signals=signals))
        return out

    def dense_search(self, ctx: RequestContext, vector, k: int) -> list[ScoredChunk]:
        with STAGE_LATENCY.time(stage="dense_search"):
            return self._wrap(ctx, self._ns(ctx).dense(ctx.principal, np.asarray(vector, dtype=np.float32), k), "dense")

    def sparse_search(self, ctx: RequestContext, query: str, k: int) -> list[ScoredChunk]:
        with STAGE_LATENCY.time(stage="sparse_search"):
            return self._wrap(ctx, self._ns(ctx).sparse(ctx.principal, query, k), "sparse")

    def graph_search(self, ctx: RequestContext, query: str, k: int, hops: int = 2) -> GraphHits:
        with STAGE_LATENCY.time(stage="graph_search"):
            hits, paths, seeds = self._ns(ctx).graph_local(ctx.principal, query, k, hops)
            return GraphHits(hits=self._wrap(ctx, hits, "graph"), paths=paths, seeds=seeds)

    def community_search(self, ctx: RequestContext, query: str, k: int) -> GraphHits:
        with STAGE_LATENCY.time(stage="community_search"):
            hits, summaries = self._ns(ctx).graph_global(ctx.principal, query, k)
            return GraphHits(hits=self._wrap(ctx, hits, "community"), summaries=summaries)

    def link_entities(self, ctx: RequestContext, query: str) -> list[str]:
        return self._ns(ctx).link(query)

    def doc_status(self, ctx: RequestContext, doc_id: str) -> dict | None:
        meta = self._ns(ctx).docs.get(doc_id)
        if meta is None:
            return None
        return {k: v for k, v in meta.items() if k != "ids"} | {"doc_id": doc_id}

    def stats(self, ctx: RequestContext) -> dict:
        return self._ns(ctx).stats() | {"shard": self.shard}
