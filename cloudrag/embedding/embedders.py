"""Embedding models behind one interface: ``embed(list[str]) -> np.ndarray[n, dim]``.

* ``hashing`` (default): signed feature hashing of stemmed unigrams, bigrams and character
  4-grams with sublinear TF, L2-normalised. Deterministic, stateless (no corpus fitting, so
  incremental indexing never forces re-embedding), and runs anywhere without model downloads.
* ``sentence-transformers:<model>``: any Sentence-Transformers bi-encoder, for semantic
  similarity in production deployments (``pip install cloudrag[models]``).
"""

from __future__ import annotations

import math
import zlib
from collections import Counter

import numpy as np

from ..core.text import content_terms
from ..observability.metrics import STAGE_LATENCY


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.name = f"hashing-{dim}"

    @staticmethod
    def features(text: str) -> Counter:
        terms = content_terms(text)
        f: Counter = Counter()
        for t in terms:
            f["w:" + t] += 1.0
        for a, b in zip(terms, terms[1:]):
            f[f"b:{a}_{b}"] += 0.5
        for t in set(terms):
            if len(t) >= 5:
                for i in range(len(t) - 3):
                    f["c:" + t[i : i + 4]] += 0.25
        return f

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, tf in self.features(text).items():
                h = zlib.crc32(feat.encode())
                sign = 1.0 if (h >> 16) & 1 else -1.0
                out[row, h % self.dim] += sign * (1.0 + math.log(tf) if tf >= 1 else tf)
            n = np.linalg.norm(out[row])
            if n > 0:
                out[row] /= n
        return out


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # optional dependency

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def build_embedder(spec: str, dim: int = 512):
    if spec == "hashing":
        return HashingEmbedder(dim)
    if spec.startswith("sentence-transformers:"):
        return SentenceTransformerEmbedder(spec.split(":", 1)[1])
    raise ValueError(f"unknown embedder {spec!r}")


class EmbeddingService:
    """Stateless; scale horizontally behind the Kubernetes Service (HPA on CPU)."""

    def __init__(self, embedder):
        self.embedder = embedder

    @property
    def dim(self) -> int:
        return self.embedder.dim

    def embed(self, ctx, texts: list[str]) -> np.ndarray:
        with STAGE_LATENCY.time(stage="embed"):
            if not texts:
                return np.zeros((0, self.embedder.dim), dtype=np.float32)
            return self.embedder.embed(texts)
