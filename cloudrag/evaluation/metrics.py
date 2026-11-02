"""Retrieval, generation and system metrics.

Retrieval (needs gold relevance): hit rate@k, MRR, context precision (rank-weighted average
precision, as in RAGAS), context recall (share of gold evidence documents retrieved).

Generation: answer correctness (gold answer containment / token F1), and two reference-free
estimates that the gateway also computes online for every response:
  faithfulness      share of answer sentences supported by some context sentence
  answer_relevance  similarity between the question and the answer
These lexical estimators are cheap stand-ins for RAGAS's LLM-judged versions; the harness
reports both where an LLM judge is configured.
"""

from __future__ import annotations

import math
import re

import numpy as np

from ..core.text import content_terms, normalize_answer, split_sentences

_CITE = re.compile(r"\s*\[\d+\]")


def hit_rate(retrieved_docs: list[str], gold_docs: set[str], k: int) -> float:
    return float(any(d in gold_docs for d in retrieved_docs[:k]))


def mrr(retrieved_docs: list[str], gold_docs: set[str]) -> float:
    for i, d in enumerate(retrieved_docs, start=1):
        if d in gold_docs:
            return 1.0 / i
    return 0.0


def context_precision(retrieved_docs: list[str], gold_docs: set[str]) -> float:
    hits, total = 0, 0.0
    for i, d in enumerate(retrieved_docs, start=1):
        if d in gold_docs:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def context_recall(retrieved_docs: list[str], gold_docs: set[str]) -> float:
    return len(set(retrieved_docs) & gold_docs) / len(gold_docs) if gold_docs else 1.0


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for t in p:
        if t in g:
            common[t] = min(p.count(t), g.count(t))
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def answer_correct(pred: str, golds: list[str]) -> float:
    """Share of gold answer strings contained in the prediction (partial credit for multi-part answers)."""
    if not golds:
        return 1.0
    norm = f" {normalize_answer(pred)} "
    return sum(f" {normalize_answer(g)} " in norm for g in golds) / len(golds)


def faithfulness(answer: str, contexts: list[str], threshold: float = 0.6) -> float:
    claims = [c for c in split_sentences(_CITE.sub("", answer)) if content_terms(c)]
    if not claims:
        return 1.0
    ctx_sents = [set(content_terms(s)) for t in contexts for s in split_sentences(t)]
    supported = 0
    for c in claims:
        terms = set(content_terms(c))
        best = max((len(terms & s) / len(terms) for s in ctx_sents), default=0.0)
        supported += best >= threshold
    return supported / len(claims)


def answer_relevance(question: str, answer: str, embedder) -> float:
    if not answer.strip():
        return 0.0
    v = embedder.embed([question, _CITE.sub("", answer)])
    return float(max(0.0, np.dot(v[0], v[1])))


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))
    return s[idx]
