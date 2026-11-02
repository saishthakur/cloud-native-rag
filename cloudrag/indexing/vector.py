"""Dense vector indexes over L2-normalised embeddings (cosine similarity).

``HNSWIndex`` implements Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2018):
multi-layer greedy routing, ``ef``-bounded beam search on layer 0, and the neighbour-selection
heuristic that keeps the graph navigable. Deletes are tombstones; the owning namespace
rebuilds the index once tombstones dominate. ``FlatIndex`` is exact brute force, used as the
recall reference and for very selective ACL filters.
"""

from __future__ import annotations

import heapq
import math
import random
import threading
from typing import Callable

import numpy as np

from ..core.locks import PicklableLocked

Allow = Callable[[int], bool] | None


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class _Storage(PicklableLocked):
    def __init__(self, dim: int):
        self.dim = dim
        self._vecs = np.zeros((256, dim), dtype=np.float32)
        self.n = 0
        self.deleted: set[int] = set()
        self.lock = threading.RLock()

    def _append(self, vec: np.ndarray) -> int:
        if self.n == len(self._vecs):
            grown = np.zeros((len(self._vecs) * 2, self.dim), dtype=np.float32)
            grown[: self.n] = self._vecs[: self.n]
            self._vecs = grown
        self._vecs[self.n] = vec
        self.n += 1
        return self.n - 1

    def vector(self, i: int) -> np.ndarray:
        return self._vecs[i]

    def delete(self, i: int) -> None:
        with self.lock:
            self.deleted.add(i)

    @property
    def live(self) -> int:
        return self.n - len(self.deleted)

    def brute_force(self, q: np.ndarray, k: int, allow: Allow = None,
                    candidates: list[int] | None = None) -> list[tuple[int, float]]:
        q = _normalize(q)
        with self.lock:
            ids = candidates if candidates is not None else range(self.n)
            ids = [i for i in ids if i not in self.deleted and (allow is None or allow(i))]
            if not ids:
                return []
            sims = self._vecs[ids] @ q
        top = np.argsort(-sims)[:k]
        return [(ids[j], float(sims[j])) for j in top]


class FlatIndex(_Storage):
    kind = "flat"

    def add(self, vec) -> int:
        with self.lock:
            return self._append(_normalize(vec))

    def search(self, q, k: int, allow: Allow = None, ef: int | None = None) -> list[tuple[int, float]]:
        return self.brute_force(q, k, allow)


class HNSWIndex(_Storage):
    kind = "hnsw"

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 80, ef_search: int = 64, seed: int = 13):
        super().__init__(dim)
        self.M, self.M0 = M, 2 * M
        self.ef_construction, self.ef_search = ef_construction, ef_search
        self._mL = 1.0 / math.log(M)
        self._rng = random.Random(seed)
        self.links: list[list[list[int]]] = []  # links[node][layer] -> neighbour ids
        self.entry = -1
        self.max_level = -1

    # -- internals ------------------------------------------------------------
    def _dists(self, q: np.ndarray, ids: list[int]) -> list[float]:
        return (1.0 - self._vecs[ids] @ q).tolist()

    def _greedy(self, q, ep: int, ep_d: float, layer: int) -> tuple[int, float]:
        improved = True
        while improved:
            improved = False
            neigh = self.links[ep][layer]
            if not neigh:
                break
            for n, d in zip(neigh, self._dists(q, neigh)):
                if d < ep_d:
                    ep, ep_d, improved = n, d, True
        return ep, ep_d

    def _search_layer(self, q, entry: list[tuple[float, int]], ef: int, layer: int) -> list[tuple[float, int]]:
        visited = {i for _, i in entry}
        cand = list(entry)
        heapq.heapify(cand)
        best = [(-d, i) for d, i in entry]
        heapq.heapify(best)
        while cand:
            d, c = heapq.heappop(cand)
            if len(best) >= ef and d > -best[0][0]:
                break
            neigh = [n for n in self.links[c][layer] if n not in visited]
            if not neigh:
                continue
            visited.update(neigh)
            for n, dn in zip(neigh, self._dists(q, neigh)):
                if len(best) < ef or dn < -best[0][0]:
                    heapq.heappush(cand, (dn, n))
                    heapq.heappush(best, (-dn, n))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-nd, i) for nd, i in best)

    def _select(self, candidates: list[tuple[float, int]], m: int) -> list[tuple[float, int]]:
        """Neighbour-selection heuristic (Malkov & Yashunin, Alg. 4) with pruned-connection backfill."""
        kept: list[tuple[float, int]] = []
        pruned: list[tuple[float, int]] = []
        for d, e in sorted(candidates):
            if len(kept) >= m:
                break
            if kept:
                sims_to_kept = self._vecs[[k for _, k in kept]] @ self._vecs[e]
                if float(np.max(sims_to_kept)) > 1.0 - d:  # closer to a kept node than to the query
                    pruned.append((d, e))
                    continue
            kept.append((d, e))
        for p in pruned:
            if len(kept) >= m:
                break
            kept.append(p)
        return kept

    # -- public API -----------------------------------------------------------
    def add(self, vec) -> int:
        v = _normalize(vec)
        with self.lock:
            idx = self._append(v)
            level = int(-math.log(1.0 - self._rng.random()) * self._mL)
            self.links.append([[] for _ in range(level + 1)])
            if self.entry < 0:
                self.entry, self.max_level = idx, level
                return idx
            ep = self.entry
            ep_d = 1.0 - float(self._vecs[ep] @ v)
            for layer in range(self.max_level, level, -1):
                ep, ep_d = self._greedy(v, ep, ep_d, layer)
            entry = [(ep_d, ep)]
            for layer in range(min(level, self.max_level), -1, -1):
                found = self._search_layer(v, entry, self.ef_construction, layer)
                m_max = self.M0 if layer == 0 else self.M
                chosen = self._select(found, self.M)
                self.links[idx][layer] = [n for _, n in chosen]
                for _, n in chosen:
                    nl = self.links[n][layer]
                    nl.append(idx)
                    if len(nl) > m_max:
                        cand = list(zip(self._dists(self._vecs[n], nl), nl))
                        self.links[n][layer] = [x for _, x in self._select(cand, m_max)]
                entry = found
            if level > self.max_level:
                self.entry, self.max_level = idx, level
            return idx

    def search(self, q, k: int, allow: Allow = None, ef: int | None = None) -> list[tuple[int, float]]:
        q = _normalize(q)
        with self.lock:
            if self.entry < 0:
                return []
            ef = max(ef or self.ef_search, k)
            if allow is not None or self.deleted:
                ef = max(ef, 4 * k)
            ep = self.entry
            ep_d = 1.0 - float(self._vecs[ep] @ q)
            for layer in range(self.max_level, 0, -1):
                ep, ep_d = self._greedy(q, ep, ep_d, layer)
            found = self._search_layer(q, [(ep_d, ep)], ef, 0)
        out = []
        for d, i in found:
            if i in self.deleted or (allow is not None and not allow(i)):
                continue
            out.append((i, 1.0 - d))
            if len(out) == k:
                break
        return out


def build_vector_index(kind: str, dim: int, M: int = 16, ef_construction: int = 80, ef_search: int = 64):
    if kind == "flat":
        return FlatIndex(dim)
    if kind == "hnsw":
        return HNSWIndex(dim, M=M, ef_construction=ef_construction, ef_search=ef_search)
    raise ValueError(f"unknown vector index {kind!r}")
