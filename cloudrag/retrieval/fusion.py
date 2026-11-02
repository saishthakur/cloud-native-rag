"""Reciprocal-rank fusion (Cormack et al., 2009) across heterogeneous retrievers."""

from __future__ import annotations

from ..core.models import ScoredChunk


def rrf(ranked: dict[str, list[ScoredChunk]], k: int = 60,
        weights: dict[str, float] | None = None, limit: int | None = None) -> list[ScoredChunk]:
    weights = weights or {}
    fused: dict[str, ScoredChunk] = {}
    scores: dict[str, float] = {}
    for source, items in ranked.items():
        w = weights.get(source, 1.0)
        for rank, sc in enumerate(items, start=1):
            cid = sc.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + w / (k + rank)
            if cid not in fused:
                fused[cid] = ScoredChunk(chunk=sc.chunk, score=0.0, signals={})
            fused[cid].signals.update(sc.signals)
            fused[cid].signals[f"{source}_rank"] = float(rank)
    out = sorted(fused.values(), key=lambda s: -scores[s.chunk.chunk_id])
    for sc in out:
        sc.score = scores[sc.chunk.chunk_id]
        sc.signals["rrf"] = sc.score
    return out[:limit] if limit else out
