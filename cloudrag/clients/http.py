"""HTTP clients for service-to-service calls (distributed mode).

Each client mirrors the method signatures of the in-process service it stands in for, so the
calling code is identical in both modes. The caller's bearer token and trace id are forwarded
on every hop so each service enforces tenant scope itself. Transient failures are retried
with backoff behind a per-upstream circuit breaker; HTTP errors map back to the same
exception types the in-process services raise.
"""

from __future__ import annotations

import httpx
import numpy as np

from ..core.models import (DocumentIn, GenerationResult, GraphHits, IngestReport, RequestContext,
                           RetrievalRequest, RetrievalResult, ScoredChunk)
from ..indexing.service import shard_for
from ..security import AuthError, BlockedInputError, ForbiddenError, RateLimitedError
from .resilience import CircuitBreaker, CircuitOpenError, DependencyUnavailable, retry


class UpstreamError(Exception):
    pass


class _Http:
    def __init__(self, base_url: str, timeout: float = 30.0, client: httpx.Client | None = None, name: str = ""):
        # short connect timeout: a dead peer should fail in seconds, not after the full read timeout
        self.client = client or httpx.Client(base_url=base_url, timeout=httpx.Timeout(timeout, connect=2.0))
        self.name = name or base_url
        self.breaker = CircuitBreaker(self.name)

    def request(self, ctx: RequestContext, method: str, path: str, json=None, params=None):
        headers = {"authorization": f"Bearer {ctx.token}", "x-trace-id": ctx.trace_id}

        def attempt():
            r = self.client.request(method, path, json=json, params=params, headers=headers)
            if r.status_code == 503:  # the peer's own dependency is down: it already retried
                raise DependencyUnavailable(f"{self.name}: {r.text[:200]}")
            if r.status_code >= 500:
                raise UpstreamError(f"{method} {path} -> {r.status_code}: {r.text[:200]}")
            return r

        try:
            r = self.breaker.call(lambda: retry(attempt, attempts=2, retry_on=(httpx.TransportError, UpstreamError)))
        except (httpx.TransportError, UpstreamError, CircuitOpenError) as e:
            raise DependencyUnavailable(f"{self.name} unavailable: {e}") from e
        if r.status_code >= 400:
            detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
            exc = {401: AuthError, 403: ForbiddenError, 422: BlockedInputError, 429: RateLimitedError}.get(r.status_code)
            if exc is not None:
                raise exc(str(detail))
            if r.status_code == 404:
                return None
            raise UpstreamError(f"{method} {path} -> {r.status_code}: {detail}")
        return r.json()


class HttpEmbeddingClient(_Http):
    def embed(self, ctx: RequestContext, texts: list[str]) -> np.ndarray:
        data = self.request(ctx, "POST", "/internal/embed", {"texts": texts})
        return np.asarray(data["vectors"], dtype=np.float32)


class HttpRerankClient(_Http):
    def rerank(self, ctx, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        data = self.request(ctx, "POST", "/internal/rerank",
                            {"query": query, "top_k": top_k, "candidates": [c.model_dump() for c in candidates]})
        return [ScoredChunk.model_validate(c) for c in data]


class HttpRetrievalClient(_Http):
    def retrieve(self, ctx, req: RetrievalRequest) -> RetrievalResult:
        return RetrievalResult.model_validate(self.request(ctx, "POST", "/internal/retrieve", req.model_dump()))


class HttpGenerationClient(_Http):
    def generate(self, ctx, query: str, contexts: list[ScoredChunk]) -> GenerationResult:
        data = self.request(ctx, "POST", "/internal/generate",
                            {"query": query, "contexts": [c.model_dump() for c in contexts]})
        return GenerationResult.model_validate(data)


class HttpIngestionClient(_Http):
    def ingest(self, ctx, doc: DocumentIn) -> IngestReport:
        return IngestReport.model_validate(self.request(ctx, "POST", "/internal/ingest", doc.model_dump()))

    def delete(self, ctx, doc_id: str) -> dict:
        return self.request(ctx, "DELETE", f"/internal/documents/{doc_id}")

    def quarantine(self, ctx, limit: int = 100) -> list[dict]:
        return self.request(ctx, "GET", "/internal/quarantine", params={"limit": limit})


class HttpIndexingClient:
    """Routes each call to the shard that owns the caller's tenant (same hash as the shards use)."""

    def __init__(self, urls: list[str], timeout: float = 30.0, clients: list[httpx.Client] | None = None):
        if not urls and not clients:
            raise ValueError("at least one indexing shard URL is required")
        clients = clients or [None] * len(urls)
        urls = urls or [""] * len(clients)
        self.shards = [_Http(u, timeout, c, name=f"indexing-{i}") for i, (u, c) in enumerate(zip(urls, clients))]

    def _shard(self, ctx: RequestContext) -> _Http:
        return self.shards[shard_for(ctx.principal.tenant_id, len(self.shards))]

    def dense_search(self, ctx, vector, k: int) -> list[ScoredChunk]:
        vec = np.asarray(vector, dtype=np.float32).round(6).tolist()
        data = self._shard(ctx).request(ctx, "POST", "/internal/search/dense", {"vector": vec, "k": k})
        return [ScoredChunk.model_validate(c) for c in data]

    def sparse_search(self, ctx, query: str, k: int) -> list[ScoredChunk]:
        data = self._shard(ctx).request(ctx, "POST", "/internal/search/sparse", {"query": query, "k": k})
        return [ScoredChunk.model_validate(c) for c in data]

    def graph_search(self, ctx, query: str, k: int, hops: int = 2) -> GraphHits:
        data = self._shard(ctx).request(ctx, "POST", "/internal/search/graph", {"query": query, "k": k, "hops": hops})
        return GraphHits.model_validate(data)

    def community_search(self, ctx, query: str, k: int) -> GraphHits:
        data = self._shard(ctx).request(ctx, "POST", "/internal/search/community", {"query": query, "k": k})
        return GraphHits.model_validate(data)

    def link_entities(self, ctx, query: str) -> list[str]:
        return self._shard(ctx).request(ctx, "POST", "/internal/link", {"query": query})

    def doc_status(self, ctx, doc_id: str) -> dict | None:
        return self._shard(ctx).request(ctx, "GET", f"/internal/documents/{doc_id}")

    def stats(self, ctx) -> dict:
        return self._shard(ctx).request(ctx, "GET", "/internal/stats")
