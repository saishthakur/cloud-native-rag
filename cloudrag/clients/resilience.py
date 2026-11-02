"""Client-side fault isolation: circuit breaker and retry with jittered backoff.

The service mesh (Istio DestinationRule outlier detection) provides the same protection at
the network layer in Kubernetes; these keep the behaviour when running without a mesh.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class CircuitOpenError(Exception):
    pass


class DependencyUnavailable(Exception):
    """A downstream service is unreachable (HTTP 503). Callers must not retry it: the hop
    nearest the failure already did, and retrying at every layer multiplies the load."""


class CircuitBreaker:
    """closed -> (N consecutive failures) -> open -> (cooldown) -> half-open -> closed/open."""

    def __init__(self, name: str, failure_threshold: int = 5, reset_timeout_s: float = 15.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if time.monotonic() - self._opened_at >= self.reset_timeout_s:
                return "half_open"
            return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()

    def call(self, fn: Callable[[], T]) -> T:
        if not self.allow():
            raise CircuitOpenError(f"circuit {self.name} is open")
        try:
            result = fn()
        except Exception:
            self.failure()
            raise
        self.success()
        return result


def retry(fn: Callable[[], T], attempts: int = 3, base_delay: float = 0.05,
          retry_on: tuple[type[BaseException], ...] = (Exception,)) -> T:
    for i in range(attempts):
        try:
            return fn()
        except retry_on:
            if i == attempts - 1:
                raise
            time.sleep(base_delay * (2 ** i) * (0.5 + random.random()))
    raise RuntimeError("unreachable")
