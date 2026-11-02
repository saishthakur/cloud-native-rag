"""FastAPI applications: one per microservice plus the public gateway.

Every app shares the same chassis: request metrics, trace-id propagation, error mapping,
``/healthz`` (liveness), ``/readyz`` (readiness) and ``/metrics`` (Prometheus).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ..clients.resilience import CircuitOpenError, DependencyUnavailable
from ..core.models import (DocumentIn, IngestReport, Principal, QueryRequest, QueryResponse, RequestContext,
                           RetrievalRequest, RetrievalResult, ScoredChunk)
from ..observability.metrics import REGISTRY, REQUEST_LATENCY, REQUESTS
from ..security import AuthError, BlockedInputError, ForbiddenError, RateLimitedError
from ..security.auth import TokenService, require_service


def create_base_app(name: str, tokens: TokenService) -> FastAPI:
    app = FastAPI(title=f"CloudRAG {name}", version="0.1.0")
    app.state.ready = True
    app.state.tokens = tokens

    @app.middleware("http")
    async def _observe(request: Request, call_next):
        t0 = time.perf_counter()
        trace = request.headers.get("x-trace-id") or uuid.uuid4().hex
        request.state.trace_id = trace
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        REQUESTS.inc(service=name, route=path, status=str(response.status_code))
        REQUEST_LATENCY.observe(time.perf_counter() - t0, service=name, route=path)
        response.headers["x-trace-id"] = trace
        return response

    def _err(status: int):
        async def handler(request: Request, exc: Exception):
            body: dict[str, Any] = {"detail": str(exc)}
            if isinstance(exc, BlockedInputError) and exc.verdict:
                body["verdict"] = exc.verdict
            return JSONResponse(body, status_code=status)
        return handler

    app.add_exception_handler(AuthError, _err(401))
    app.add_exception_handler(ForbiddenError, _err(403))
    app.add_exception_handler(BlockedInputError, _err(422))
    app.add_exception_handler(RateLimitedError, _err(429))
    app.add_exception_handler(CircuitOpenError, _err(503))
    app.add_exception_handler(DependencyUnavailable, _err(503))

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "service": name}

    @app.get("/readyz")
    def readyz(response: Response):
        if not app.state.ready:
            response.status_code = 503
            return {"status": "starting", "service": name}
        return {"status": "ready", "service": name}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return REGISTRY.render()

    return app


def context_dependency(tokens: TokenService):
    def dep(request: Request, authorization: str | None = Header(default=None)) -> RequestContext:
        principal, token = tokens.from_header(authorization)
        return RequestContext(principal=principal, token=token, trace_id=request.state.trace_id)
    return dep


# -- request bodies -------------------------------------------------------------------------
class EmbedBody(BaseModel):
    texts: list[str] = Field(max_length=2048)


class DenseBody(BaseModel):
    vector: list[float]
    k: int = Field(10, ge=1, le=500)


class TextBody(BaseModel):
    query: str
    k: int = Field(10, ge=1, le=500)
    hops: int = Field(2, ge=1, le=4)


class RerankBody(BaseModel):
    query: str
    candidates: list[ScoredChunk]
    top_k: int = Field(10, ge=1, le=500)


class GenerateBody(BaseModel):
    query: str
    contexts: list[ScoredChunk]


class TokenBody(BaseModel):
    tenant_id: str
    user_id: str
    roles: list[str] = Field(default_factory=list)
    clearance: int = 0
    ttl_s: int | None = None


class BatchBody(BaseModel):
    documents: list[DocumentIn] = Field(max_length=500)


# -- internal services ---------------------------------------------------------------------
def create_embedding_app(svc, tokens: TokenService) -> FastAPI:
    app = create_base_app("embedding", tokens)
    ctx_dep = context_dependency(tokens)

    @app.post("/internal/embed")
    def embed(body: EmbedBody, ctx: RequestContext = Depends(ctx_dep)):
        return {"vectors": svc.embed(ctx, body.texts).round(6).tolist(), "dim": svc.dim}

    return app


def create_indexing_app(svc, tokens: TokenService) -> FastAPI:
    app = create_base_app("indexing", tokens)
    ctx_dep = context_dependency(tokens)

    @app.post("/internal/search/dense")
    def dense(body: DenseBody, ctx: RequestContext = Depends(ctx_dep)) -> list[ScoredChunk]:
        return svc.dense_search(ctx, body.vector, body.k)

    @app.post("/internal/search/sparse")
    def sparse(body: TextBody, ctx: RequestContext = Depends(ctx_dep)) -> list[ScoredChunk]:
        return svc.sparse_search(ctx, body.query, body.k)

    @app.post("/internal/search/graph")
    def graph(body: TextBody, ctx: RequestContext = Depends(ctx_dep)):
        return svc.graph_search(ctx, body.query, body.k, body.hops)

    @app.post("/internal/search/community")
    def community(body: TextBody, ctx: RequestContext = Depends(ctx_dep)):
        return svc.community_search(ctx, body.query, body.k)

    @app.post("/internal/link")
    def link(body: TextBody, ctx: RequestContext = Depends(ctx_dep)) -> list[str]:
        return svc.link_entities(ctx, body.query)

    @app.get("/internal/documents/{doc_id:path}")
    def doc_status(doc_id: str, ctx: RequestContext = Depends(ctx_dep)):
        st = svc.doc_status(ctx, doc_id)
        if st is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        return st

    @app.get("/internal/stats")
    def stats(ctx: RequestContext = Depends(ctx_dep)):
        return svc.stats(ctx)

    @app.post("/internal/snapshot")
    def snapshot(ctx: RequestContext = Depends(ctx_dep)):
        require_service(ctx.principal)
        svc.maybe_snapshot(force=True)
        return {"status": "saved", "tenants": svc.store.tenants()}

    return app


def create_reranker_app(svc, tokens: TokenService) -> FastAPI:
    app = create_base_app("reranker", tokens)
    ctx_dep = context_dependency(tokens)

    @app.post("/internal/rerank")
    def rerank(body: RerankBody, ctx: RequestContext = Depends(ctx_dep)) -> list[ScoredChunk]:
        return svc.rerank(ctx, body.query, body.candidates, body.top_k)

    return app


def create_retrieval_app(svc, tokens: TokenService) -> FastAPI:
    app = create_base_app("retrieval", tokens)
    ctx_dep = context_dependency(tokens)

    @app.post("/internal/retrieve")
    def retrieve(body: RetrievalRequest, ctx: RequestContext = Depends(ctx_dep)) -> RetrievalResult:
        return svc.retrieve(ctx, body)

    return app


def create_generation_app(svc, tokens: TokenService) -> FastAPI:
    app = create_base_app("generation", tokens)
    ctx_dep = context_dependency(tokens)

    @app.post("/internal/generate")
    def generate(body: GenerateBody, ctx: RequestContext = Depends(ctx_dep)):
        return svc.generate(ctx, body.query, body.contexts)

    return app


def create_ingestion_app(svc, tokens: TokenService) -> FastAPI:
    app = create_base_app("ingestion", tokens)
    ctx_dep = context_dependency(tokens)

    @app.post("/internal/ingest", status_code=202)
    def ingest(body: DocumentIn, ctx: RequestContext = Depends(ctx_dep)) -> IngestReport:
        return svc.ingest(ctx, body)

    @app.delete("/internal/documents/{doc_id:path}", status_code=202)
    def delete(doc_id: str, ctx: RequestContext = Depends(ctx_dep)):
        return svc.delete(ctx, doc_id)

    @app.get("/internal/quarantine")
    def quarantine(limit: int = 100, ctx: RequestContext = Depends(ctx_dep)):
        return svc.quarantine(ctx, limit)

    return app


# -- public gateway ----------------------------------------------------------------------------
def create_gateway_app(gw) -> FastAPI:
    app = create_base_app("gateway", gw.tokens)

    def ctx_dep(request: Request, authorization: str | None = Header(default=None)) -> RequestContext:
        return gw.authenticate(authorization, request.state.trace_id)

    @app.post("/v1/auth/token")
    def token(body: TokenBody, x_admin_key: str | None = Header(default=None)):
        p = Principal(user_id=body.user_id, tenant_id=body.tenant_id, roles=body.roles, clearance=body.clearance)
        return {"access_token": gw.issue_token(x_admin_key, p, body.ttl_s), "token_type": "bearer"}

    @app.get("/v1/me")
    def me(ctx: RequestContext = Depends(ctx_dep)) -> Principal:
        return ctx.principal

    @app.post("/v1/query")
    def query(body: QueryRequest, ctx: RequestContext = Depends(ctx_dep)) -> QueryResponse:
        return gw.query(ctx, body)

    @app.post("/v1/retrieve")
    def retrieve(body: RetrievalRequest, ctx: RequestContext = Depends(ctx_dep)) -> RetrievalResult:
        return gw.retrieve(ctx, body)

    @app.post("/v1/documents", status_code=202)
    def ingest(body: DocumentIn, ctx: RequestContext = Depends(ctx_dep)) -> IngestReport:
        return gw.ingest(ctx, body)

    @app.post("/v1/documents:batch", status_code=202)
    def ingest_batch(body: BatchBody, ctx: RequestContext = Depends(ctx_dep)) -> list[IngestReport]:
        return [gw.ingest(ctx, d) for d in body.documents]

    @app.get("/v1/documents/{doc_id:path}")
    def doc_status(doc_id: str, ctx: RequestContext = Depends(ctx_dep)):
        st = gw.doc_status(ctx, doc_id)
        if st is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        return st

    @app.delete("/v1/documents/{doc_id:path}", status_code=202)
    def delete(doc_id: str, ctx: RequestContext = Depends(ctx_dep)):
        return gw.delete(ctx, doc_id)

    @app.get("/v1/stats")
    def stats(ctx: RequestContext = Depends(ctx_dep)):
        return gw.stats(ctx)

    @app.get("/v1/security/quarantine")
    def quarantine(limit: int = 100, ctx: RequestContext = Depends(ctx_dep)):
        return gw.quarantine(ctx, limit)

    @app.get("/v1/security/audit/verify")
    def audit_verify(ctx: RequestContext = Depends(ctx_dep)):
        return gw.verify_audit(ctx)

    return app
