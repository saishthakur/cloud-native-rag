"""Retrieval service: pluggable strategies behind one interface, selected by the router.

Strategies call the embedding, indexing and reranking services through client objects that
are either in-process (local mode) or HTTP (distributed mode); independent retriever calls run
concurrently. Before anything leaves this service every chunk is re-checked for provenance
integrity, ACL and tenant ownership.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..config import Settings
from ..core.entities import entity_key, extract_mentions
from ..core.models import (RequestContext, RetrievalRequest, RetrievalResult, RouteDecision, ScoredChunk,
                           can_read)
from ..core.text import STOPWORDS, content_terms, split_sentences, tokenize
from ..observability.metrics import SECURITY_EVENTS, STAGE_LATENCY
from ..security.provenance import ProvenanceSigner
from .fusion import rrf
from .router import _GLOBAL_CUES, QueryRouter, decompose

# Leaf retriever calls run on _POOL; sub-question pipelines (which themselves wait on _POOL)
# run on _PLANS, so a saturated pool can never deadlock on its own tasks.
_POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="retrieval")
_PLANS = ThreadPoolExecutor(max_workers=8, thread_name_prefix="retrieval-plan")
_QUESTION_WORDS = {"which", "what", "who", "whom", "where", "when", "how", "many", "much", "name"}


@dataclass
class _Run:
    contexts: list[ScoredChunk]
    steps: list[dict] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)


class RetrievalService:
    def __init__(self, settings: Settings, index, embedder, reranker, router: QueryRouter, signer: ProvenanceSigner):
        self.settings = settings
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.router = router
        self.signer = signer

    # -- entry point -------------------------------------------------------------
    def retrieve(self, ctx: RequestContext, req: RetrievalRequest) -> RetrievalResult:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}
        if req.strategy == "auto":
            linked = self.index.link_entities(ctx, req.query)
            route = self.router.decide(req.query, linked, req.latency_budget_ms)
            timings["route_ms"] = (time.perf_counter() - t0) * 1000
        else:
            route = RouteDecision(strategy=req.strategy, reasons=["strategy requested explicitly"])
        strategy = route.strategy
        k = req.top_k
        if req.strategy == "auto" and route.features.get("global"):
            k = min(50, 2 * req.top_k)  # aggregation needs every instance in context
            route.reasons.append(f"aggregation query: context budget raised to {k} chunks")
        t1 = time.perf_counter()
        run: _Run = getattr(self, f"_{strategy}")(ctx, req.query, k, req.rerank)
        timings["retrieve_ms"] = (time.perf_counter() - t1) * 1000
        contexts, events = self._verify(ctx, run.contexts)
        elapsed = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = elapsed
        self.router.observe(strategy, elapsed)
        STAGE_LATENCY.observe(elapsed / 1000, stage=f"retrieve_{strategy}")
        return RetrievalResult(query=req.query, strategy=strategy, route=route, contexts=contexts,
                               steps=run.steps, graph_paths=run.paths, timings_ms=timings,
                               security_events=events)

    # -- helpers ---------------------------------------------------------------------
    def _embed(self, ctx, text: str):
        return self.embedder.embed(ctx, [text])[0]

    def _rerank_blend(self, ctx, query: str, fused: list[ScoredChunk], k: int, alpha: float,
                      rerank: bool = True) -> list[ScoredChunk]:
        """alpha * normalised rerank score + (1 - alpha) * normalised fusion score."""
        if not fused:
            return []
        if not rerank:
            return fused[:k]
        reranked = self.reranker.rerank(ctx, query, fused, len(fused))
        top_rr = max((c.score for c in reranked), default=1.0) or 1.0
        top_f = max((c.score for c in fused), default=1.0) or 1.0
        fused_score = {c.chunk.chunk_id: c.score / top_f for c in fused}
        for c in reranked:
            c.score = alpha * (c.score / top_rr) + (1 - alpha) * fused_score.get(c.chunk.chunk_id, 0.0)
            c.signals["final"] = c.score
        reranked.sort(key=lambda c: -c.score)
        return reranked[:k]

    def _verify(self, ctx: RequestContext, contexts: list[ScoredChunk]):
        ok, events = [], []
        p = ctx.principal
        for sc in contexts:
            ch = sc.chunk
            problem = None
            if ch.tenant_id != p.tenant_id:
                problem = "tenant_isolation_violation"
            elif not can_read(p, ch.acl):
                problem = "acl_violation"
            elif not self.signer.verify(ch):
                problem = "integrity_violation"
            if problem:
                SECURITY_EVENTS.inc(type=problem)
                events.append({"type": problem, "chunk_id": ch.chunk_id, "doc_id": ch.doc_id})
                continue
            ok.append(sc)
        return ok, events

    # -- strategies --------------------------------------------------------------------
    def _naive(self, ctx, q: str, k: int, rerank: bool = True) -> _Run:
        hits = self.index.dense_search(ctx, self._embed(ctx, q), k)
        return _Run(hits, [{"step": "dense", "query": q, "hits": len(hits)}])

    def _advanced(self, ctx, q: str, k: int, rerank: bool = True) -> _Run:
        pool = max(4 * k, 20)
        f_dense = _POOL.submit(lambda: self.index.dense_search(ctx, self._embed(ctx, q), pool))
        f_sparse = _POOL.submit(self.index.sparse_search, ctx, q, pool)
        lists = {"dense": f_dense.result(), "sparse": f_sparse.result()}
        fused = rrf(lists, k=self.settings.rrf_k, limit=pool)
        out = self._rerank_blend(ctx, q, fused, k, alpha=0.8, rerank=rerank)
        return _Run(out, [{"step": "hybrid", "query": q, **{s: len(v) for s, v in lists.items()}}])

    def _graph(self, ctx, q: str, k: int, rerank: bool = True) -> _Run:
        pool = max(4 * k, 20)
        f_dense = _POOL.submit(lambda: self.index.dense_search(ctx, self._embed(ctx, q), pool))
        f_sparse = _POOL.submit(self.index.sparse_search, ctx, q, pool)
        f_graph = _POOL.submit(self.index.graph_search, ctx, q, pool, 2)
        want_global = bool(_GLOBAL_CUES.search(q))
        f_comm = _POOL.submit(self.index.community_search, ctx, q, pool) if want_global else None
        g = f_graph.result()
        lists = {"dense": f_dense.result(), "sparse": f_sparse.result(), "graph": g.hits}
        summaries: list[str] = []
        if f_comm is not None:
            c = f_comm.result()
            lists["community"] = c.hits
            summaries = c.summaries
        fused = rrf(lists, k=self.settings.rrf_k, weights={"graph": 1.3, "community": 0.8}, limit=pool)
        out = self._rerank_blend(ctx, q, fused, k, alpha=0.6, rerank=rerank)
        # relational evidence often shares few words with the question, so the reranker alone would
        # demote it: keep the strongest graph-traversal hits in the final context
        have = {c.chunk.chunk_id for c in out}
        missing = [gh for gh in g.hits[: min(2, k - 1)] if gh.chunk.chunk_id not in have]
        if missing:
            out = out[: max(0, k - len(missing))] + missing
        step = {"step": "graph_hybrid", "query": q, "seeds": g.seeds, **{s: len(v) for s, v in lists.items()}}
        if summaries:
            step["communities"] = summaries[:3]
        return _Run(out, [step], g.paths)

    def _graphrag(self, ctx, q: str, k: int, rerank: bool = True) -> _Run:
        """Pure GraphRAG baseline: entity-graph local search plus community-level global search."""
        f_graph = _POOL.submit(self.index.graph_search, ctx, q, 2 * k, 2)
        f_comm = _POOL.submit(self.index.community_search, ctx, q, k)
        g, c = f_graph.result(), f_comm.result()
        fused = rrf({"graph": g.hits, "community": c.hits}, k=self.settings.rrf_k, limit=k)
        return _Run(fused, [{"step": "graphrag", "query": q, "seeds": g.seeds, "communities": c.summaries[:3]}],
                    g.paths)

    def _agentic(self, ctx, q: str, k: int, rerank: bool = True) -> _Run:
        subqs = decompose(q)
        steps: list[dict] = [{"step": "plan", "sub_questions": subqs}]
        lists: dict[str, list[ScoredChunk]] = {}
        paths: list[list[str]] = []
        futures = [_PLANS.submit(self._graph, ctx, sq, k, rerank) for sq in subqs]
        for i, (sq, fut) in enumerate(zip(subqs, futures)):
            r = fut.result()
            lists[f"sub{i}"] = r.contexts
            paths.extend(r.paths)
            steps.append({"step": "retrieve", "query": sq, "top": [c.chunk.chunk_id for c in r.contexts[:3]]})

        seen = {m.key for m in extract_mentions(q)} | {entity_key(n) for n in self.index.link_entities(ctx, q)}
        latest = [c for v in lists.values() for c in v[:3]]
        for hop in range(self.settings.agentic_max_steps):
            gathered = _unique(c for v in lists.values() for c in v)
            coverage, missing = _coverage(q, gathered)
            steps.append({"step": "reflect", "coverage": round(coverage, 3), "missing": missing})
            if coverage >= 0.85 and hop > 0:
                break
            bridge = _bridge_entity(q, latest, seen)
            if bridge is None:
                break
            seen.add(entity_key(bridge))
            follow = " ".join([bridge] + missing)
            r = self._graph(ctx, follow, k, rerank)
            lists[f"hop{hop}"] = r.contexts
            latest = r.contexts[:3]
            paths.extend(r.paths)
            steps.append({"step": "bridge", "entity": bridge, "query": follow,
                          "top": [c.chunk.chunk_id for c in r.contexts[:3]]})

        weights = {name: (1.0 if name.startswith("sub") else 0.9) for name in lists}
        fused = rrf(lists, k=self.settings.rrf_k, weights=weights)
        ranked = self._rerank_blend(ctx, q, fused, len(fused), alpha=0.5, rerank=rerank)
        must = _unique(v[0] for v in lists.values() if v)
        final = must[:k]
        ids = {c.chunk.chunk_id for c in final}
        for c in ranked:
            if len(final) >= k:
                break
            if c.chunk.chunk_id not in ids:
                final.append(c)
                ids.add(c.chunk.chunk_id)
        score = {c.chunk.chunk_id: c.score for c in ranked}
        final = [ScoredChunk(chunk=c.chunk, score=score.get(c.chunk.chunk_id, 0.0), signals=c.signals) for c in final]
        final.sort(key=lambda c: -c.score)
        return _Run(final, steps, paths[:10])


def _unique(items) -> list[ScoredChunk]:
    out, seen = [], set()
    for c in items:
        if c.chunk.chunk_id not in seen:
            seen.add(c.chunk.chunk_id)
            out.append(c)
    return out


def _coverage(q: str, contexts: list[ScoredChunk]) -> tuple[float, list[str]]:
    q_terms = [t for t in dict.fromkeys(content_terms(q)) if t not in _QUESTION_WORDS]
    if not q_terms:
        return 1.0, []
    have: set[str] = set()
    for c in contexts:
        have.update(content_terms(c.chunk.text))
    missing = [t for t in q_terms if t not in have]
    return 1.0 - len(missing) / len(q_terms), missing


def _bridge_entity(q: str, contexts: list[ScoredChunk], seen: set[str]) -> str | None:
    """Pick the next entity to explore: one stated in fresh evidence *together with* an entity we
    already explored (so the chain stays connected), in a sentence that matches the question's
    relation words, preferring higher-ranked evidence."""
    q_norm = " ".join(tokenize(q))
    seen_terms = {t for key in seen for t in content_terms(key)}
    q_rel = set(content_terms(q)) - seen_terms
    scores: dict[str, float] = {}
    names: dict[str, str] = {}
    for rank, c in enumerate(contexts):
        for sent in split_sentences(c.chunk.text):
            mentions = extract_mentions(sent)
            connected = any(m.key in seen for m in mentions)
            match = len(q_rel & set(content_terms(sent)))
            if not connected and not match:
                continue
            score = (2.0 if connected else 1.0) * (1 + match) / (1 + rank)
            for m in mentions:
                if m.key in seen or m.key in q_norm or m.key in STOPWORDS:
                    continue
                if score > scores.get(m.key, 0.0):
                    scores[m.key] = score
                    names[m.key] = m.name
    if not scores:
        return None
    return names[max(scores, key=scores.get)]
