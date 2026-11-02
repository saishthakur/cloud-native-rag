"""Incremental Okapi BM25 inverted index (supports add / remove without rebuilds)."""

from __future__ import annotations

import heapq
import math
import threading
from collections import Counter, defaultdict
from typing import Callable

from ..core.locks import PicklableLocked
from ..core.text import content_terms


class BM25Index(PicklableLocked):
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_len: dict[int, int] = {}
        self.doc_terms: dict[int, tuple[str, ...]] = {}
        self.total_len = 0
        self.lock = threading.RLock()

    def __len__(self) -> int:
        return len(self.doc_len)

    def add(self, doc: int, text: str) -> None:
        terms = content_terms(text)
        tf = Counter(terms)
        with self.lock:
            if doc in self.doc_len:
                self.remove(doc)
            for t, c in tf.items():
                self.postings[t][doc] = c
            self.doc_len[doc] = len(terms)
            self.doc_terms[doc] = tuple(tf)
            self.total_len += len(terms)

    def remove(self, doc: int) -> None:
        with self.lock:
            if doc not in self.doc_len:
                return
            for t in self.doc_terms.pop(doc):
                post = self.postings.get(t)
                if post is not None:
                    post.pop(doc, None)
                    if not post:
                        del self.postings[t]
            self.total_len -= self.doc_len.pop(doc)

    def idf(self, term: str) -> float:
        n = len(self.doc_len)
        df = len(self.postings.get(term, ()))
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int, allow: Callable[[int], bool] | None = None) -> list[tuple[int, float]]:
        q_terms = set(content_terms(query))
        with self.lock:
            n = len(self.doc_len)
            if n == 0 or not q_terms:
                return []
            avgdl = self.total_len / n
            scores: dict[int, float] = defaultdict(float)
            for t in q_terms:
                post = self.postings.get(t)
                if not post:
                    continue
                idf = self.idf(t)
                for doc, tf in post.items():
                    dl = self.doc_len[doc]
                    scores[doc] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl))
        items = ((d, s) for d, s in scores.items() if allow is None or allow(d))
        return heapq.nlargest(k, items, key=lambda x: x[1])
