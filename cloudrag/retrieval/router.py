"""Query-complexity-aware retrieval router.

The router extracts cheap features from the query (length, linked entities, relational /
multi-hop cues, aggregation cues, multi-part structure), maps them to the least expensive
pathway expected to answer it:

    naive     short single-fact lookups              (dense top-k)
    advanced  descriptive / keyword-heavy questions  (dense + BM25, RRF, rerank)
    graph     relational or aggregate questions      (+ knowledge-graph traversal, communities)
    agentic   multi-part or multi-hop chains         (decompose, iterate, reflect)

and then enforces the caller's latency budget using live per-strategy p95 latencies observed
by this replica, downgrading along agentic -> graph -> advanced -> naive when needed.
"""

from __future__ import annotations

import re
import threading
from collections import deque

from ..core.entities import extract_mentions
from ..core.models import RouteDecision
from ..core.text import content_terms
from ..observability.metrics import STRATEGY_SELECTED

ORDER = ("naive", "advanced", "graph", "agentic")
PRIOR_P95_MS = {"naive": 15.0, "advanced": 40.0, "graph": 80.0, "agentic": 250.0}

_RELATIONAL_VERBS = re.compile(
    r"\b(uses?|used|using|supplie[sd]|supply|supplier|leads?|led|manage[sd]?|manager|owns?|owned|"
    r"works?|depends?|located|headquartered|based|reports?|built|makes?|made|produces?|sponsor(s|ed)?|"
    r"member|belongs?|runs?|operates?)\b", re.I)
_RELATIONAL_CUES = re.compile(
    r"\b(relationship|related|connection|connected|linked|between|who\s+\w+s\b.*\bof\b|"
    r"(that|which|who|whose)\b.{0,40}\b(uses?|used|supplie[sd]|leads?|led|manage[sd]?|owns?|"
    r"located|headquartered|based|built|makes?|made|sponsor\w*)|"
    r"(used|supplied|led|managed|owned|made|built|sponsored|run|operated) by)\b", re.I)
_GLOBAL_CUES = re.compile(
    r"\b(overall|main themes?|summar(y|ize|ise)|across|trends?|how many|list (all|every)|"
    r"all (the )?(projects|products|vendors|suppliers|teams|people|employees|documents)|"
    r"which (projects|products|vendors|suppliers|teams|people|employees))\b", re.I)
_COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|difference between)\b", re.I)
_SPLIT = re.compile(
    r"\?\s*|;\s*|\s+(?:and|also|as well as|plus)\s+(?=(?:who|what|which|when|where|why|how|is|are|does|did|"
    r"list|name|give)\b)", re.I)


def decompose(query: str) -> list[str]:
    """Split a multi-part question into self-contained sub-questions (heuristic planner)."""
    m = re.search(r"\bcompare\s+(.+?)\s+(?:and|with|vs\.?|versus)\s+(.+?)(?:\s+(?:in terms of|on|by)\s+(.+?))?\??$",
                  query, re.I)
    if m:
        aspect = m.group(3) or ""
        return [f"{m.group(1)} {aspect}".strip(), f"{m.group(2)} {aspect}".strip()]
    parts = [p.strip(" ,.") for p in _SPLIT.split(query) if p and p.strip(" ,.")]
    parts = [p for p in parts if len(content_terms(p)) >= 2]
    return parts if len(parts) > 1 else [query.strip()]


class QueryRouter:
    def __init__(self, window: int = 200):
        self._lat: dict[str, deque] = {s: deque(maxlen=window) for s in ORDER}
        self._lock = threading.Lock()

    def observe(self, strategy: str, latency_ms: float) -> None:
        if strategy in self._lat:
            with self._lock:
                self._lat[strategy].append(latency_ms)

    def p95(self, strategy: str) -> float:
        with self._lock:
            xs = sorted(self._lat.get(strategy, ()))
        if len(xs) < 10:
            return PRIOR_P95_MS[strategy]
        return xs[min(len(xs) - 1, int(0.95 * len(xs)))]

    def features(self, query: str, linked_entities: list[str]) -> dict[str, float]:
        terms = content_terms(query)
        mentions = {m.key for m in extract_mentions(query)}
        n_entities = max(len(linked_entities), len(mentions))
        hops = len(_RELATIONAL_VERBS.findall(query))
        return {
            "n_terms": float(len(terms)),
            "n_entities": float(n_entities),
            "relational": float(bool(_RELATIONAL_CUES.search(query))),
            "relation_verbs": float(hops),
            "global": float(bool(_GLOBAL_CUES.search(query))),
            "multipart": float(len(decompose(query)) - 1),
            "compare": float(bool(_COMPARE.search(query))),
        }

    def decide(self, query: str, linked_entities: list[str], latency_budget_ms: float | None = None) -> RouteDecision:
        f = self.features(query, linked_entities)
        complexity = min(1.0, 0.04 * f["n_terms"] + 0.1 * min(f["n_entities"], 4) + 0.25 * f["relational"]
                         + 0.1 * min(f["relation_verbs"], 3) + 0.2 * f["global"] + 0.3 * min(f["multipart"], 2)
                         + 0.2 * f["compare"])
        reasons: list[str] = []
        if f["multipart"] >= 1 or f["compare"]:
            strategy = "agentic"
            reasons.append("multi-part question: decompose and retrieve per sub-question")
        elif f["relational"] and f["relation_verbs"] >= 3:
            strategy = "agentic"
            reasons.append("multi-hop relational chain: iterative bridging retrieval")
        elif f["relational"] or f["global"]:
            strategy = "graph"
            reasons.append("relational cue" if f["relational"] else "aggregation / global cue")
        elif f["n_terms"] <= 6 and f["n_entities"] <= 1:
            strategy = "naive"
            reasons.append("short single-fact lookup")
        else:
            strategy = "advanced"
            reasons.append("descriptive question: hybrid retrieval with reranking")

        if latency_budget_ms is not None:
            idx = ORDER.index(strategy)
            while idx > 0 and self.p95(ORDER[idx]) > latency_budget_ms:
                reasons.append(f"{ORDER[idx]} p95 {self.p95(ORDER[idx]):.0f}ms exceeds budget {latency_budget_ms:.0f}ms")
                idx -= 1
            strategy = ORDER[idx]

        STRATEGY_SELECTED.inc(strategy=strategy)
        return RouteDecision(strategy=strategy, complexity=round(complexity, 3), features=f, reasons=reasons)
