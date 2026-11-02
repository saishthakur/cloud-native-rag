"""Incrementally maintained knowledge graph for graph-augmented retrieval.

* Nodes are entities, edges are co-occurrence relations labelled with predicates and linked
  to the chunks that evidence them.
* ``link`` performs entity linking from free text (longest alias match, case-insensitive).
* ``traverse`` expands query entities over k hops with score decay (local search).
* ``communities`` runs weighted label propagation and builds community reports
  (top entities, key relations, representative chunks) used for global queries, in the
  spirit of GraphRAG's community summaries.
"""

from __future__ import annotations

import heapq
import math
import random
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

from ..core.entities import entity_aliases, extract_mentions, extract_relations
from ..core.locks import PicklableLocked
from ..core.text import STOPWORDS, content_terms, tokenize


@dataclass
class Node:
    key: str
    name: str
    chunks: set[int] = field(default_factory=set)
    aliases: set[tuple[str, ...]] = field(default_factory=set)


@dataclass
class Edge:
    weight: int = 0
    triples: Counter = field(default_factory=Counter)  # (head name, predicate, tail name) -> count
    chunks: set[int] = field(default_factory=set)

    def best_triple(self) -> tuple[str, str, str]:
        return self.triples.most_common(1)[0][0]


@dataclass
class Community:
    cid: int
    members: list[str]
    triples: list[str]
    chunks: list[int]
    summary: str


class KnowledgeGraph(PicklableLocked):
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.adj: dict[str, dict[str, Edge]] = defaultdict(dict)
        self.alias: dict[tuple[str, ...], str] = {}
        self._chunk_nodes: dict[int, set[str]] = {}
        self._chunk_triples: dict[int, list[tuple[tuple[str, str], tuple[str, str, str]]]] = {}
        self._communities: list[Community] | None = None
        self.lock = threading.RLock()

    # -- maintenance ------------------------------------------------------------
    def add_chunk(self, cid: int, text: str) -> None:
        mentions = extract_mentions(text)
        triples = extract_relations(text)
        with self.lock:
            keys = set()

            def ensure(m) -> None:
                node = self.nodes.get(m.key)
                if node is None:
                    node = self.nodes[m.key] = Node(key=m.key, name=m.name)
                    for a in entity_aliases(m.name):
                        if a not in self.alias:
                            self.alias[a] = m.key
                            node.aliases.add(a)
                node.chunks.add(cid)
                keys.add(m.key)

            for m in mentions:
                ensure(m)
            for a, _, b in triples:
                ensure(a)
                ensure(b)
            self._chunk_nodes[cid] = keys
            recorded = []
            for a, pred, b in triples:
                pair = (a.key, b.key) if a.key < b.key else (b.key, a.key)
                edge = self.adj[pair[0]].get(pair[1])
                if edge is None:
                    edge = Edge()
                    self.adj[pair[0]][pair[1]] = edge
                    self.adj[pair[1]][pair[0]] = edge
                triple = (self.nodes[a.key].name, pred, self.nodes[b.key].name)
                edge.weight += 1
                edge.triples[triple] += 1
                edge.chunks.add(cid)
                recorded.append((pair, triple))
            self._chunk_triples[cid] = recorded
            self._communities = None

    def remove_chunk(self, cid: int) -> None:
        with self.lock:
            for pair, triple in self._chunk_triples.pop(cid, []):
                edge = self.adj.get(pair[0], {}).get(pair[1])
                if edge is None:
                    continue
                edge.weight -= 1
                edge.triples[triple] -= 1
                if edge.triples[triple] <= 0:
                    del edge.triples[triple]
                edge.chunks.discard(cid)
                if edge.weight <= 0:
                    self.adj[pair[0]].pop(pair[1], None)
                    self.adj[pair[1]].pop(pair[0], None)
            for key in self._chunk_nodes.pop(cid, set()):
                node = self.nodes.get(key)
                if node is None:
                    continue
                node.chunks.discard(cid)
                if not node.chunks:
                    for a in node.aliases:
                        if self.alias.get(a) == key:
                            del self.alias[a]
                    for nb in list(self.adj.get(key, {})):
                        self.adj[nb].pop(key, None)
                    self.adj.pop(key, None)
                    del self.nodes[key]
            self._communities = None

    # -- linking & traversal ------------------------------------------------------
    def link(self, text: str) -> dict[str, float]:
        found: dict[str, float] = {}
        with self.lock:
            for m in extract_mentions(text):
                if m.key in self.nodes:
                    found[m.key] = 1.0
            toks = tokenize(text)
            covered = [False] * len(toks)
            for n in range(min(6, len(toks)), 0, -1):
                for i in range(len(toks) - n + 1):
                    if any(covered[i : i + n]):
                        continue
                    tup = tuple(toks[i : i + n])
                    key = self.alias.get(tup)
                    if key is None:
                        continue
                    if n == 1:
                        t = tup[0]
                        if t in STOPWORDS or (len(t) < 4 and not any(c.isdigit() for c in t)):
                            continue
                    found[key] = max(found.get(key, 0.0), 1.0 if n > 1 else 0.8)
                    for j in range(i, i + n):
                        covered[j] = True
        return found

    def traverse(self, seeds: dict[str, float], hops: int = 2, decay: float = 0.5, fanout: int = 15,
                 frontier_cap: int = 30, query_terms: set[str] | None = None) -> tuple[dict[str, float], dict[str, str]]:
        """Score-decaying BFS. With ``query_terms``, edges whose predicate matches the question's
        relation words ("used by", "supplied by") keep full weight and others are halved, so the
        walk follows the relation being asked about."""
        scores = dict(seeds)
        parents: dict[str, str] = {}
        frontier = dict(seeds)
        with self.lock:
            for _ in range(hops):
                nxt: dict[str, float] = {}
                for key, s in frontier.items():
                    neigh = sorted(self.adj.get(key, {}).items(), key=lambda kv: -kv[1].weight)[:fanout]
                    for nb, edge in neigh:
                        ns = s * decay * min(1.0, 0.6 + 0.2 * math.log1p(edge.weight))
                        if query_terms is not None:
                            preds = {t for (_, p, _) in edge.triples for t in content_terms(p)}
                            ns *= 1.0 if preds & query_terms else 0.5
                        if ns > scores.get(nb, 0.0):
                            scores[nb] = ns
                            parents[nb] = key
                            nxt[nb] = ns
                frontier = dict(heapq.nlargest(frontier_cap, nxt.items(), key=lambda kv: kv[1]))
                if not frontier:
                    break
        return scores, parents

    def chunk_scores(self, node_scores: dict[str, float], k: int,
                     allow: Callable[[int], bool] | None = None) -> list[tuple[int, float]]:
        cs: dict[int, float] = defaultdict(float)
        with self.lock:
            for key, s in node_scores.items():
                node = self.nodes.get(key)
                if node is None:
                    continue
                specificity = 1.0 / (1.0 + math.log1p(len(node.chunks)))
                for cid in node.chunks:
                    cs[cid] += s * specificity
            for key, s in node_scores.items():
                for nb, edge in self.adj.get(key, {}).items():
                    if key < nb and nb in node_scores:
                        bonus = 0.5 * (s + node_scores[nb])
                        for cid in edge.chunks:
                            cs[cid] += bonus
        items = ((c, s) for c, s in cs.items() if allow is None or allow(c))
        return heapq.nlargest(k, items, key=lambda x: x[1])

    def paths(self, node_scores: dict[str, float], parents: dict[str, str], limit: int = 8) -> list[list[str]]:
        out = []
        with self.lock:
            for key, _ in heapq.nlargest(limit, ((k, s) for k, s in node_scores.items() if k in parents),
                                         key=lambda kv: kv[1]):
                path, cur = [], key
                while cur in parents and len(path) < 6:
                    par = parents[cur]
                    edge = self.adj.get(par, {}).get(cur)
                    if edge is None or not edge.triples:
                        break
                    h, p, t = edge.best_triple()
                    path.append(f"{h} --{p}--> {t}")
                    cur = par
                if path:
                    out.append(list(reversed(path)))
        return out

    # -- communities ----------------------------------------------------------------
    def communities(self) -> list[Community]:
        with self.lock:
            if self._communities is not None:
                return self._communities
            labels = {k: k for k in self.nodes}
            order = sorted(self.nodes)
            rng = random.Random(0)
            for _ in range(12):
                rng.shuffle(order)
                changed = 0
                for k in order:
                    votes: Counter = Counter()
                    for nb, edge in self.adj.get(k, {}).items():
                        votes[labels[nb]] += edge.weight
                    if not votes:
                        continue
                    top = max(votes.values())
                    best = min(lbl for lbl, w in votes.items() if w == top)
                    if best != labels[k]:
                        labels[k] = best
                        changed += 1
                if not changed:
                    break
            groups: dict[str, list[str]] = defaultdict(list)
            for k, lbl in labels.items():
                groups[lbl].append(k)
            comms = []
            for i, members in enumerate(sorted(groups.values(), key=lambda m: (-len(m), m[0]))):
                if len(members) < 2:
                    continue
                mset = set(members)
                members.sort(key=lambda k: -len(self.adj.get(k, {})))
                edges = []
                for a in members:
                    for b, e in self.adj.get(a, {}).items():
                        if a < b and b in mset:
                            edges.append(e)
                edges.sort(key=lambda e: -e.weight)
                triples = [" ".join((h, p, t)) for h, p, t in (e.best_triple() for e in edges[:10])]
                cover: Counter = Counter()
                for k in members:
                    for c in self.nodes[k].chunks:
                        cover[c] += 1
                chunks = [c for c, _ in cover.most_common(6)]
                names = [self.nodes[k].name for k in members[:12]]
                summary = f"Entities: {', '.join(names)}. Relations: {'; '.join(triples)}."
                comms.append(Community(cid=i, members=members, triples=triples, chunks=chunks, summary=summary))
            self._communities = comms
            return comms

    def community_search(self, query: str, k: int = 3) -> list[tuple[Community, float]]:
        q_terms = set(content_terms(query))
        linked = self.link(query)
        scored = []
        for c in self.communities():
            c_terms = set(content_terms(c.summary))
            overlap = len(q_terms & c_terms) / (len(q_terms) or 1)
            membership = sum(1 for m in c.members if m in linked)
            score = overlap + 0.5 * membership + 0.05 * math.log1p(len(c.members))
            if score > 0.05:
                scored.append((c, score))
        return heapq.nlargest(k, scored, key=lambda x: x[1])

    def stats(self) -> dict:
        with self.lock:
            n_edges = sum(len(v) for v in self.adj.values()) // 2
            return {"entities": len(self.nodes), "relations": n_edges}
