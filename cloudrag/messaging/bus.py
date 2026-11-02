"""Event bus decoupling ingestion from indexing.

``InMemoryBus`` runs consumers on background threads in-process (local mode and tests).
``RedisStreamBus`` uses Redis Streams consumer groups: at-least-once delivery, acknowledgement
after successful handling, bounded retries, a dead-letter stream, and reclaiming of messages
left pending by a crashed consumer. Handlers must be idempotent (indexing upserts are
versioned, so replays are harmless).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from typing import Callable

from ..observability.metrics import BUS_EVENTS

log = logging.getLogger(__name__)
Handler = Callable[[dict], None]
MAX_ATTEMPTS = 3


def envelope(event_type: str, payload: dict, trace_id: str = "") -> dict:
    return {"id": uuid.uuid4().hex, "type": event_type, "ts": time.time(), "trace_id": trace_id, "payload": payload}


class InMemoryBus:
    def __init__(self):
        self._subs: dict[str, list[tuple[str, queue.Queue]]] = {}
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self.dead_letters: list[dict] = []

    def publish(self, topic: str, event: dict) -> None:
        for _, q in self._subs.get(topic, []):
            q.put(event)
        BUS_EVENTS.inc(topic=topic, outcome="published")

    def subscribe(self, topic: str, group: str, handler: Handler, start_id: str | None = None,
                  on_handled: Callable[[str], None] | None = None) -> None:
        q: queue.Queue = queue.Queue()
        self._subs.setdefault(topic, []).append((group, q))
        t = threading.Thread(target=self._consume, args=(topic, q, handler), daemon=True, name=f"bus-{topic}-{group}")
        t.start()
        self._threads.append(t)

    def _consume(self, topic: str, q: queue.Queue, handler: Handler) -> None:
        while not self._stop.is_set():
            try:
                event = q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        handler(event)
                        BUS_EVENTS.inc(topic=topic, outcome="handled")
                        break
                    except Exception:  # noqa: BLE001
                        log.exception("handler failed for %s (attempt %d)", topic, attempt)
                        if attempt == MAX_ATTEMPTS:
                            self.dead_letters.append(event)
                            BUS_EVENTS.inc(topic=topic, outcome="dead_letter")
                        else:
                            time.sleep(0.05 * attempt)
            finally:
                q.task_done()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        for subs in self._subs.values():
            for _, q in subs:
                while q.unfinished_tasks:
                    if time.time() > deadline:
                        return False
                    time.sleep(0.005)
        return True

    def close(self) -> None:
        self._stop.set()


class RedisStreamBus:
    def __init__(self, url: str, maxlen: int = 1_000_000, block_ms: int = 1000, claim_idle_ms: int = 60_000):
        import redis  # optional dependency

        self._r = redis.Redis.from_url(url, decode_responses=True)
        self.maxlen = maxlen
        self.block_ms = block_ms
        self.claim_idle_ms = claim_idle_ms
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.consumer = f"c-{uuid.uuid4().hex[:8]}"

    def publish(self, topic: str, event: dict) -> None:
        self._r.xadd(topic, {"data": json.dumps(event)}, maxlen=self.maxlen, approximate=True)
        BUS_EVENTS.inc(topic=topic, outcome="published")

    def subscribe(self, topic: str, group: str, handler: Handler, start_id: str | None = None,
                  on_handled: Callable[[str], None] | None = None) -> None:
        try:
            self._r.xgroup_create(topic, group, id=start_id or "0", mkstream=True)
        except Exception as e:  # noqa: BLE001
            if "BUSYGROUP" not in str(e):
                raise
            # The group remembers what was *acknowledged*, but a restarted in-memory consumer only
            # holds what its snapshot captured: rewind to the snapshot offset, or to the beginning
            # when there is no snapshot, and replay (handlers are idempotent).
            self._r.xgroup_setid(topic, group, start_id or "0")
        t = threading.Thread(target=self._consume, args=(topic, group, handler, on_handled), daemon=True)
        t.start()
        self._threads.append(t)

    def _handle(self, topic, group, msg_id, fields, handler, on_handled) -> None:
        event = json.loads(fields["data"])
        try:
            handler(event)
            self._r.xack(topic, group, msg_id)
            self._r.hdel(f"{topic}:{group}:attempts", msg_id)
            BUS_EVENTS.inc(topic=topic, outcome="handled")
            if on_handled:
                on_handled(msg_id)
        except Exception:  # noqa: BLE001
            attempts = self._r.hincrby(f"{topic}:{group}:attempts", msg_id, 1)
            log.exception("handler failed for %s %s (attempt %d)", topic, msg_id, attempts)
            if attempts >= MAX_ATTEMPTS:
                self._r.xadd(f"{topic}.dlq", {"data": fields["data"], "group": group})
                self._r.xack(topic, group, msg_id)
                BUS_EVENTS.inc(topic=topic, outcome="dead_letter")

    def _consume(self, topic, group, handler, on_handled) -> None:
        last_claim = 0.0
        while not self._stop.is_set():
            try:
                if time.time() - last_claim > self.claim_idle_ms / 1000:
                    last_claim = time.time()
                    _, claimed, *_ = self._r.xautoclaim(topic, group, self.consumer, self.claim_idle_ms, "0-0", count=50)
                    for msg_id, fields in claimed:
                        if fields:
                            self._handle(topic, group, msg_id, fields, handler, on_handled)
                resp = self._r.xreadgroup(group, self.consumer, {topic: ">"}, count=16, block=self.block_ms)
                for _, messages in resp or []:
                    for msg_id, fields in messages:
                        self._handle(topic, group, msg_id, fields, handler, on_handled)
            except Exception:  # noqa: BLE001 - keep consuming through transient Redis errors
                log.exception("redis consumer error on %s", topic)
                time.sleep(1.0)

    def wait_idle(self, timeout: float = 30.0) -> bool:
        return True

    def close(self) -> None:
        self._stop.set()


def build_bus(kind: str, redis_url: str = ""):
    if kind == "memory":
        return InMemoryBus()
    if kind == "redis":
        return RedisStreamBus(redis_url)
    raise ValueError(f"unknown bus {kind!r}")
