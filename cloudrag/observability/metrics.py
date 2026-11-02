"""Minimal Prometheus-compatible metrics (counters, gauges, histograms with labels).

Every service exposes ``GET /metrics`` in the Prometheus text format; the ServiceMonitor in
``deploy/k8s`` scrapes it. Kept dependency-free so the core library runs anywhere.
"""

from __future__ import annotations

import bisect
import threading
import time
from contextlib import contextmanager

DEFAULT_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
SCORE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _fmt_labels(names: tuple[str, ...], values: tuple[str, ...], extra: str = "") -> str:
    parts = [f'{n}="{v}"' for n, v in zip(names, values)]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


class _Metric:
    kind = "untyped"

    def __init__(self, name: str, help_: str, labels: tuple[str, ...] = ()):
        self.name, self.help, self.labels = name, help_, labels
        self._lock = threading.Lock()

    def _key(self, labels: dict) -> tuple[str, ...]:
        return tuple(str(labels.get(n, "")) for n in self.labels)


class Counter(_Metric):
    kind = "counter"

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._values: dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, **labels) -> None:
        key = self._key(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels) -> float:
        return self._values.get(self._key(labels), 0.0)

    def render(self) -> list[str]:
        return [f"{self.name}{_fmt_labels(self.labels, k)} {v}" for k, v in sorted(self._values.items())]


class Gauge(Counter):
    kind = "gauge"

    def set(self, value: float, **labels) -> None:
        with self._lock:
            self._values[self._key(labels)] = value


class Histogram(_Metric):
    kind = "histogram"

    def __init__(self, name, help_, labels=(), buckets=DEFAULT_BUCKETS):
        super().__init__(name, help_, labels)
        self.buckets = tuple(buckets)
        self._data: dict[tuple, list] = {}  # key -> [bucket_counts, sum, count]

    def observe(self, value: float, **labels) -> None:
        key = self._key(labels)
        with self._lock:
            d = self._data.setdefault(key, [[0] * len(self.buckets), 0.0, 0])
            i = bisect.bisect_left(self.buckets, value)
            if i < len(self.buckets):
                d[0][i] += 1
            d[1] += value
            d[2] += 1

    @contextmanager
    def time(self, **labels):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.observe(time.perf_counter() - t0, **labels)

    def render(self) -> list[str]:
        out = []
        for key, (counts, total, n) in sorted(self._data.items()):
            cum = 0
            for b, c in zip(self.buckets, counts):
                cum += c
                le = 'le="%s"' % b
                out.append(f"{self.name}_bucket{_fmt_labels(self.labels, key, le)} {cum}")
            inf = 'le="+Inf"'
            out.append(f"{self.name}_bucket{_fmt_labels(self.labels, key, inf)} {n}")
            out.append(f"{self.name}_sum{_fmt_labels(self.labels, key)} {total}")
            out.append(f"{self.name}_count{_fmt_labels(self.labels, key)} {n}")
        return out


class Registry:
    def __init__(self):
        self._metrics: dict[str, _Metric] = {}
        self._lock = threading.Lock()

    def _get(self, cls, name, help_, labels=(), **kw):
        with self._lock:
            m = self._metrics.get(name)
            if m is None:
                m = cls(name, help_, tuple(labels), **kw)
                self._metrics[name] = m
            return m

    def counter(self, name, help_, labels=()) -> Counter:
        return self._get(Counter, name, help_, labels)

    def gauge(self, name, help_, labels=()) -> Gauge:
        return self._get(Gauge, name, help_, labels)

    def histogram(self, name, help_, labels=(), buckets=DEFAULT_BUCKETS) -> Histogram:
        return self._get(Histogram, name, help_, labels, buckets=buckets)

    def render(self) -> str:
        lines = []
        for m in self._metrics.values():
            lines.append(f"# HELP {m.name} {m.help}")
            lines.append(f"# TYPE {m.name} {m.kind}")
            lines.extend(m.render())
        return "\n".join(lines) + "\n"


REGISTRY = Registry()

# Shared metric families (each service only populates the ones it owns).
REQUESTS = REGISTRY.counter("cloudrag_requests_total", "Requests handled", ("service", "route", "status"))
REQUEST_LATENCY = REGISTRY.histogram("cloudrag_request_seconds", "Request latency", ("service", "route"))
STAGE_LATENCY = REGISTRY.histogram("cloudrag_stage_seconds", "Pipeline stage latency", ("stage",))
STRATEGY_SELECTED = REGISTRY.counter("cloudrag_route_strategy_total", "Router decisions", ("strategy",))
SECURITY_EVENTS = REGISTRY.counter("cloudrag_security_events_total", "Security events", ("type",))
DOCS_INGESTED = REGISTRY.counter("cloudrag_documents_ingested_total", "Documents ingested", ("status",))
CHUNKS_INDEXED = REGISTRY.counter("cloudrag_chunks_indexed_total", "Chunks indexed")
PII_REDACTIONS = REGISTRY.counter("cloudrag_pii_redactions_total", "PII spans redacted", ("type",))
LLM_TOKENS = REGISTRY.counter("cloudrag_llm_tokens_total", "LLM tokens", ("model", "direction"))
LLM_COST = REGISTRY.counter("cloudrag_llm_cost_usd_total", "LLM spend in USD", ("model",))
QUALITY = REGISTRY.histogram("cloudrag_online_quality", "Online quality estimates", ("metric",), SCORE_BUCKETS)
INDEX_SIZE = REGISTRY.gauge("cloudrag_index_chunks", "Live chunks per shard", ("shard",))
BUS_EVENTS = REGISTRY.counter("cloudrag_bus_events_total", "Bus events", ("topic", "outcome"))
