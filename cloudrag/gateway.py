"""API gateway: authentication, rate limiting, input filtering, orchestration, auditing.

query: guard -> retrieval service (router + strategy) -> generation service -> online quality
       estimates -> audit record. Downstream services are reached through client objects that
       are in-process or HTTP depending on the deployment mode.
"""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid

from .config import Settings
from .core.models import (DocumentIn, IngestReport, Principal, QueryRequest, QueryResponse, RequestContext,
                          RetrievalRequest, RetrievalResult)
from .embedding.embedders import HashingEmbedder
from .evaluation.metrics import answer_relevance, faithfulness
from .observability.metrics import QUALITY, SECURITY_EVENTS
from .security import AuthError, BlockedInputError, ForbiddenError
from .security.audit import AuditLog
from .security.auth import TenantRateLimiter, TokenService, require_roles
from .security.guard import QueryGuard
from .security.pii import PIIRedactor


class Gateway:
    def __init__(self, settings: Settings, tokens: TokenService, ingestion, retrieval, generation, indexing,
                 audit: AuditLog):
        self.settings = settings
        self.tokens = tokens
        self.ingestion = ingestion
        self.retrieval = retrieval
        self.generation = generation
        self.indexing = indexing
        self.audit = audit
        self.guard = QueryGuard(settings.query_block_threshold, settings.max_query_chars)
        self.limiter = TenantRateLimiter(settings.rate_limit_rps, settings.rate_limit_burst)
        self.pii = PIIRedactor("redact")
        self._quality_embedder = HashingEmbedder(256)

    # -- identity -----------------------------------------------------------------------
    def authenticate(self, authorization: str | None, trace_id: str | None = None) -> RequestContext:
        try:
            principal, token = self.tokens.from_header(authorization)
        except AuthError:
            SECURITY_EVENTS.inc(type="auth_failure")
            raise
        if principal.is_service:
            raise ForbiddenError("service identities cannot call the public API")
        return RequestContext(principal=principal, token=token, trace_id=trace_id or uuid.uuid4().hex)

    def issue_token(self, admin_key: str | None, principal: Principal, ttl_s: int | None = None) -> str:
        if not self.settings.dev_auth:
            raise ForbiddenError("token issuance is disabled; use the configured identity provider")
        if not admin_key or not hmac.compare_digest(admin_key, self.settings.admin_bootstrap_key):
            SECURITY_EVENTS.inc(type="auth_failure")
            raise AuthError("invalid admin key")
        if "service" in principal.roles or principal.tenant_id in ("", "*"):
            raise ForbiddenError("cannot mint service or cross-tenant identities")
        self.audit.record("auth.token_issued", principal, roles=principal.roles, clearance=principal.clearance)
        return self.tokens.issue(principal, ttl_s)

    # -- query path -----------------------------------------------------------------------
    def _guard(self, ctx: RequestContext, query: str) -> str:
        self.limiter.check(ctx.principal.tenant_id)
        try:
            cleaned, _ = self.guard.inspect(query)
        except BlockedInputError as e:
            SECURITY_EVENTS.inc(type="query_blocked")
            self.audit.record("security.query_blocked", ctx.principal, ctx.trace_id,
                              query=self.pii.redact(query[:500])[0], verdict=e.verdict)
            raise
        return cleaned

    def retrieve(self, ctx: RequestContext, req: RetrievalRequest) -> RetrievalResult:
        cleaned = self._guard(ctx, req.query)
        res = self.retrieval.retrieve(ctx, RetrievalRequest(**req.model_dump(include=set(RetrievalRequest.model_fields)) | {"query": cleaned}))
        self.audit.record("retrieve", ctx.principal, ctx.trace_id, query=self.pii.redact(cleaned)[0],
                          strategy=res.strategy, chunks=[c.chunk.chunk_id for c in res.contexts])
        return res

    def query(self, ctx: RequestContext, req: QueryRequest) -> QueryResponse:
        t0 = time.perf_counter()
        cleaned = self._guard(ctx, req.query)
        rreq = RetrievalRequest(**req.model_dump(include=set(RetrievalRequest.model_fields)) | {"query": cleaned})
        res = self.retrieval.retrieve(ctx, rreq)
        timings = dict(res.timings_ms)
        answer, citations, cost, est, abstained, quality = None, [], 0.0, 0.0, False, {}
        if req.generate:
            gen = self.generation.generate(ctx, cleaned, res.contexts)
            answer, citations, cost, est, abstained = gen.answer, gen.citations, gen.cost_usd, gen.estimated_cost_usd, gen.abstained
            timings.update({k: v for k, v in gen.timings_ms.items() if k.endswith("_ms")})
            if not abstained:
                quality = {
                    "faithfulness": faithfulness(answer, [c.chunk.text for c in res.contexts]),
                    "answer_relevance": answer_relevance(cleaned, answer, self._quality_embedder),
                }
                for metric, value in quality.items():
                    QUALITY.observe(value, metric=metric)
        timings["gateway_total_ms"] = (time.perf_counter() - t0) * 1000
        self.audit.record(
            "query", ctx.principal, ctx.trace_id,
            query=self.pii.redact(cleaned)[0], strategy=res.strategy,
            chunks=[c.chunk.chunk_id for c in res.contexts],
            answer_sha256=hashlib.sha256((answer or "").encode()).hexdigest(),
            abstained=abstained, cost_usd=cost, security_events=res.security_events,
        )
        return QueryResponse(
            trace_id=ctx.trace_id, query=cleaned, answer=answer, abstained=abstained, citations=citations,
            strategy=res.strategy, route=res.route, contexts=res.contexts, graph_paths=res.graph_paths,
            steps=res.steps, timings_ms=timings, cost_usd=cost, estimated_cost_usd=est, quality=quality,
            security_events=res.security_events,
        )

    # -- documents --------------------------------------------------------------------------
    def ingest(self, ctx: RequestContext, doc: DocumentIn) -> IngestReport:
        self.limiter.check(ctx.principal.tenant_id)
        return self.ingestion.ingest(ctx, doc)

    def delete(self, ctx: RequestContext, doc_id: str) -> dict:
        return self.ingestion.delete(ctx, doc_id)

    def doc_status(self, ctx: RequestContext, doc_id: str) -> dict | None:
        return self.indexing.doc_status(ctx, doc_id)

    def stats(self, ctx: RequestContext) -> dict:
        return self.indexing.stats(ctx)

    def quarantine(self, ctx: RequestContext, limit: int = 100) -> list[dict]:
        return self.ingestion.quarantine(ctx, limit)

    def verify_audit(self, ctx: RequestContext) -> dict:
        require_roles(ctx.principal, "security")
        return self.audit.verify()
