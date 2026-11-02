"""Second-stage rerankers that score each (query, chunk) pair jointly.

* ``lexical`` (default): an interaction-based scorer - IDF-weighted query coverage over the
  whole chunk, best single-sentence coverage (answer-bearing sentences contain most query
  terms), ordered bigram matches, title match, and a small first-stage prior; the result is
  scaled by the chunk's ingestion trust score so flagged content is demoted.
* ``cross-encoder:<model>``: a Sentence-Transformers cross-encoder (e.g.
  ``cross-encoder/ms-marco-MiniLM-L-6-v2``) for production deployments.
"""

from __future__ import annotations

import math
from collections import Counter

from ..core.models import ScoredChunk
from ..core.text import bigrams, content_terms, split_sentences
from ..observability.metrics import STAGE_LATENCY


class LexicalCrossReranker:
    name = "lexical-cross"

    def score_pairs(self, query: str, candidates: list[ScoredChunk]) -> list[float]:
        q_terms = list(dict.fromkeys(content_terms(query)))
        if not q_terms or not candidates:
            return [0.0] * len(candidates)
        q_bi = bigrams(content_terms(query))
        n = len(candidates)
        doc_terms = [set(content_terms(c.chunk.embed_text)) for c in candidates]
        df = Counter(t for terms in doc_terms for t in set(q_terms) & terms)
        idf = {t: math.log(1.0 + (n + 1) / (df.get(t, 0) + 0.5)) for t in q_terms}
        total = sum(idf.values())
        title_terms = [set(content_terms(c.chunk.title)) for c in candidates]
        scores = []
        for rank, (c, terms) in enumerate(zip(candidates, doc_terms)):
            coverage = sum(idf[t] for t in q_terms if t in terms) / total
            best_sent = 0.0
            for s in split_sentences(c.chunk.text):
                st = set(content_terms(s))
                best_sent = max(best_sent, sum(idf[t] for t in q_terms if t in st) / total)
            bi = len(q_bi & bigrams(content_terms(c.chunk.text))) / len(q_bi) if q_bi else 0.0
            title = sum(idf[t] for t in q_terms if t in title_terms[rank]) / total
            prior = 1.0 / (1.0 + rank)
            s = 0.35 * coverage + 0.35 * best_sent + 0.12 * bi + 0.08 * title + 0.10 * prior
            scores.append(s * (0.5 + 0.5 * c.chunk.trust))
        return scores

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        scores = self.score_pairs(query, candidates)
        out = []
        for c, s in zip(candidates, scores):
            out.append(ScoredChunk(chunk=c.chunk, score=s, signals=dict(c.signals, rerank=s)))
        out.sort(key=lambda x: -x.score)
        return out[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder  # optional dependency

        self._model = CrossEncoder(model_name)
        self.name = model_name

    def rerank(self, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        raw = self._model.predict([(query, c.chunk.embed_text) for c in candidates])
        out = []
        for c, s in zip(candidates, raw):
            s = float(s) * (0.5 + 0.5 * c.chunk.trust)
            out.append(ScoredChunk(chunk=c.chunk, score=s, signals=dict(c.signals, rerank=s)))
        out.sort(key=lambda x: -x.score)
        return out[:top_k]


def build_reranker(spec: str):
    if spec == "lexical":
        return LexicalCrossReranker()
    if spec.startswith("cross-encoder:"):
        return CrossEncoderReranker(spec.split(":", 1)[1])
    raise ValueError(f"unknown reranker {spec!r}")


class RerankService:
    def __init__(self, reranker):
        self.reranker = reranker

    def rerank(self, ctx, query: str, candidates: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        with STAGE_LATENCY.time(stage="rerank"):
            return self.reranker.rerank(query, candidates, top_k)
