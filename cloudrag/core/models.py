"""Data contracts shared by every CloudRAG service (also the wire format between them)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

STRATEGIES = ("naive", "advanced", "graph", "agentic")
StrategyName = Literal["auto", "naive", "advanced", "graph", "agentic", "graphrag"]


class Principal(BaseModel):
    """An authenticated caller. ``tenant_id`` bounds everything the caller can touch."""

    user_id: str
    tenant_id: str
    roles: list[str] = Field(default_factory=list)
    clearance: int = 0

    def has_role(self, *roles: str) -> bool:
        return any(r in self.roles for r in roles)

    @property
    def is_service(self) -> bool:
        return "service" in self.roles


class AccessPolicy(BaseModel):
    """Who inside a tenant may read a document or chunk.

    ``roles`` / ``users`` are any-of lists; both empty means every member of the tenant
    (subject to ``min_clearance``). A chunk carries a *list* of policies that must all
    allow the reader, so section-level restrictions narrow the document-level policy.
    """

    roles: list[str] = Field(default_factory=list)
    users: list[str] = Field(default_factory=list)
    min_clearance: int = 0

    def allows(self, p: Principal) -> bool:
        if p.has_role("admin"):
            return True
        if p.clearance < self.min_clearance:
            return False
        if not self.roles and not self.users:
            return True
        return p.user_id in self.users or any(r in p.roles for r in self.roles)


def can_read(p: Principal, acl: list[AccessPolicy]) -> bool:
    return all(policy.allows(p) for policy in acl)


class Provenance(BaseModel):
    source: str
    content_hash: str
    signature: str
    ingested_at: float
    ingested_by: str
    pipeline_version: str


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    tenant_id: str
    title: str = ""
    text: str
    position: int = 0
    acl: list[AccessPolicy] = Field(default_factory=list)
    provenance: Provenance | None = None
    trust: float = 1.0
    flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        return f"{self.title}\n{self.text}" if self.title else self.text


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    signals: dict[str, float] = Field(default_factory=dict)


class GraphHits(BaseModel):
    hits: list[ScoredChunk] = Field(default_factory=list)
    paths: list[list[str]] = Field(default_factory=list)
    seeds: list[str] = Field(default_factory=list)
    summaries: list[str] = Field(default_factory=list)


class DocumentIn(BaseModel):
    doc_id: str | None = None
    title: str = ""
    text: str
    source: str = "api"
    metadata: dict[str, Any] = Field(default_factory=dict)
    acl: AccessPolicy = Field(default_factory=AccessPolicy)


class IngestReport(BaseModel):
    job_id: str
    doc_id: str
    tenant_id: str
    status: Literal["accepted", "partially_quarantined", "quarantined", "rejected"]
    chunks_indexed: int
    chunks_quarantined: int
    pii_redactions: dict[str, int] = Field(default_factory=dict)
    poisoning: dict[str, Any] = Field(default_factory=dict)
    version: int = 0


class RetrievalRequest(BaseModel):
    query: str
    strategy: StrategyName = "auto"
    top_k: int = Field(5, ge=1, le=50)
    latency_budget_ms: float | None = None
    rerank: bool = True


class QueryRequest(RetrievalRequest):
    generate: bool = True


class RouteDecision(BaseModel):
    strategy: str
    complexity: float = 0.0
    features: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    query: str
    strategy: str
    route: RouteDecision | None = None
    contexts: list[ScoredChunk] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    graph_paths: list[list[str]] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    security_events: list[dict[str, Any]] = Field(default_factory=list)


class Citation(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    title: str = ""


class GenerationResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    model: str
    abstained: bool = False
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0  # what this prompt would cost on the configured LLM
    sanitized_sentences: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    trace_id: str
    query: str
    answer: str | None = None
    abstained: bool = False
    citations: list[Citation] = Field(default_factory=list)
    strategy: str
    route: RouteDecision | None = None
    contexts: list[ScoredChunk] = Field(default_factory=list)
    graph_paths: list[list[str]] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    quality: dict[str, float] = Field(default_factory=dict)
    security_events: list[dict[str, Any]] = Field(default_factory=list)


@dataclass
class RequestContext:
    """Per-request identity and tracing data propagated across service hops."""

    principal: Principal
    token: str = ""
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
