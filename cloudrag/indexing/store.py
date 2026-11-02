"""Tenant-namespaced index storage.

Each tenant gets a physically separate ``TenantNamespace`` (its own HNSW graph, BM25 postings
and knowledge graph), so a query can only ever traverse the caller's own data. Inside a
namespace, chunk-level ACLs are applied *during* search (as a filter on candidates), not
after, so restricted chunks never displace or leak through ranking.
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
import time
from typing import Callable

import numpy as np

from ..config import Settings
from ..core.locks import PicklableLocked
from ..core.models import Chunk, Principal, can_read
from ..core.text import content_terms
from .bm25 import BM25Index
from .graph import KnowledgeGraph
from .vector import build_vector_index

log = logging.getLogger(__name__)


class TenantNamespace(PicklableLocked):
    def __init__(self, tenant_id: str, settings: Settings, dim: int):
        self.tenant_id = tenant_id
        self._settings = settings
        self.dim = dim
        self.lock = threading.RLock()
        self._reset()
        self.docs: dict[str, dict] = {}

    def _reset(self) -> None:
        s = self._settings
        self.vectors = build_vector_index(s.vector_index, self.dim, s.hnsw_m, s.hnsw_ef_construction, s.hnsw_ef_search)
        self.bm25 = BM25Index()
        self.graph = KnowledgeGraph()
        self.chunks: dict[int, Chunk] = {}
        self.by_chunk_id: dict[str, int] = {}

    # -- writes ---------------------------------------------------------------
    def upsert_document(self, doc_id: str, chunks: list[Chunk], vectors: np.ndarray, version: int) -> bool:
        """Replace a document's chunks. Returns False for stale (out-of-order) versions."""
        with self.lock:
            cur = self.docs.get(doc_id)
            if cur and cur["version"] > version:
                return False
            self._remove_chunks(doc_id)
            ids = [self._add(ch, vec) for ch, vec in zip(chunks, vectors)]
            self.docs[doc_id] = {"ids": ids, "version": version, "indexed_at": time.time(),
                                 "deleted": False, "chunks": len(ids)}
            self._maybe_compact()
            return True

    def delete_document(self, doc_id: str, version: int) -> bool:
        with self.lock:
            cur = self.docs.get(doc_id)
            if cur and cur["version"] > version:
                return False
            self._remove_chunks(doc_id)
            self.docs[doc_id] = {"ids": [], "version": version, "indexed_at": time.time(),
                                 "deleted": True, "chunks": 0}
            return True

    def _add(self, ch: Chunk, vec) -> int:
        i = self.vectors.add(vec)
        self.chunks[i] = ch
        self.by_chunk_id[ch.chunk_id] = i
        self.bm25.add(i, ch.embed_text)
        self.graph.add_chunk(i, ch.text)
        return i

    def _remove_chunks(self, doc_id: str) -> None:
        cur = self.docs.get(doc_id)
        if not cur:
            return
        for i in cur["ids"]:
            ch = self.chunks.pop(i, None)
            if ch is not None:
                self.by_chunk_id.pop(ch.chunk_id, None)
            self.vectors.delete(i)
            self.bm25.remove(i)
            self.graph.remove_chunk(i)
        cur["ids"] = []

    def _maybe_compact(self) -> None:
        v = self.vectors
        if v.n >= 512 and len(v.deleted) > 0.3 * v.n:
            self.rebuild()

    def rebuild(self) -> None:
        """Drop tombstones by re-inserting live chunks into fresh indexes."""
        with self.lock:
            live = [(i, self.chunks[i], self.vectors.vector(i).copy()) for i in sorted(self.chunks)]
            remap = {}
            self._reset()
            for old, ch, vec in live:
                remap[old] = self._add(ch, vec)
            for meta in self.docs.values():
                meta["ids"] = [remap[i] for i in meta["ids"] if i in remap]
            log.info("rebuilt namespace %s with %d live chunks", self.tenant_id, len(live))

    # -- reads ----------------------------------------------------------------
    def allow_fn(self, p: Principal) -> Callable[[int], bool]:
        chunks = self.chunks

        def allow(i: int) -> bool:
            c = chunks.get(i)
            return c is not None and can_read(p, c.acl)

        return allow

    def dense(self, p: Principal, vec, k: int) -> list[tuple[Chunk, float]]:
        with self.lock:
            allow = self.allow_fn(p)
            res = self.vectors.search(vec, k, allow=allow)
            if len(res) < k and self.vectors.live > len(res):
                res = self.vectors.brute_force(vec, k, allow=allow)  # very selective ACL filter
            return [(self.chunks[i], s) for i, s in res]

    def sparse(self, p: Principal, query: str, k: int) -> list[tuple[Chunk, float]]:
        with self.lock:
            return [(self.chunks[i], s) for i, s in self.bm25.search(query, k, allow=self.allow_fn(p))]

    def graph_local(self, p: Principal, query: str, k: int, hops: int = 2):
        with self.lock:
            seeds = self.graph.link(query)
            if not seeds:
                return [], [], []
            node_scores, parents = self.graph.traverse(seeds, hops=hops, query_terms=set(content_terms(query)))
            hits = self.graph.chunk_scores(node_scores, k, allow=self.allow_fn(p))
            paths = self.graph.paths(node_scores, parents)
            seed_names = [self.graph.nodes[s].name for s in seeds if s in self.graph.nodes]
            return [(self.chunks[i], s) for i, s in hits], paths, seed_names

    def graph_global(self, p: Principal, query: str, k: int):
        with self.lock:
            allow = self.allow_fn(p)
            out, seen, summaries = [], set(), []
            for comm, score in self.graph.community_search(query, k=3):
                summaries.append(comm.summary)
                for rank, i in enumerate(comm.chunks):
                    if i in seen or not allow(i):
                        continue
                    seen.add(i)
                    out.append((self.chunks[i], score / (1.0 + 0.3 * rank), comm.cid))
            out.sort(key=lambda x: -x[1])
            return out[:k], summaries

    def link(self, query: str) -> list[str]:
        with self.lock:
            return [self.graph.nodes[k].name for k in self.graph.link(query) if k in self.graph.nodes]

    def stats(self) -> dict:
        with self.lock:
            return {
                "tenant": self.tenant_id,
                "documents": sum(1 for d in self.docs.values() if not d["deleted"]),
                "chunks": len(self.chunks),
                "vector_index": self.vectors.kind,
                "tombstones": len(self.vectors.deleted),
                **self.graph.stats(),
            }


class IndexStore:
    """All namespaces owned by one indexing shard, with snapshot/restore."""

    def __init__(self, settings: Settings, dim: int, shard: int = 0):
        self.settings = settings
        self.dim = dim
        self.shard = shard
        self._ns: dict[str, TenantNamespace] = {}
        self._lock = threading.Lock()
        self.meta: dict = {"last_event_id": None}
        self.dir = os.path.join(settings.data_dir, "index", f"shard-{shard}")

    def namespace(self, tenant_id: str, create: bool = True) -> TenantNamespace | None:
        with self._lock:
            ns = self._ns.get(tenant_id)
            if ns is None and create:
                ns = self._ns[tenant_id] = TenantNamespace(tenant_id, self.settings, self.dim)
            return ns

    def tenants(self) -> list[str]:
        return sorted(self._ns)

    def total_chunks(self) -> int:
        return sum(len(ns.chunks) for ns in list(self._ns.values()))

    def save(self) -> None:
        os.makedirs(self.dir, exist_ok=True)
        for tenant, ns in list(self._ns.items()):
            with ns.lock:
                data = pickle.dumps(ns, protocol=pickle.HIGHEST_PROTOCOL)
            path = os.path.join(self.dir, f"{tenant}.pkl")
            with open(path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
        with open(os.path.join(self.dir, "meta.pkl.tmp"), "wb") as f:
            pickle.dump(self.meta, f)
        os.replace(os.path.join(self.dir, "meta.pkl.tmp"), os.path.join(self.dir, "meta.pkl"))

    def load(self) -> int:
        """Restore snapshots written by this shard's own ``save`` (trusted local files)."""
        if not os.path.isdir(self.dir):
            return 0
        n = 0
        for name in os.listdir(self.dir):
            path = os.path.join(self.dir, name)
            if name == "meta.pkl":
                with open(path, "rb") as f:
                    self.meta = pickle.load(f)
            elif name.endswith(".pkl"):
                with open(path, "rb") as f:
                    ns = pickle.load(f)
                ns._settings = self.settings
                self._ns[ns.tenant_id] = ns
                n += 1
        return n
