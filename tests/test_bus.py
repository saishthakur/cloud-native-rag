"""Event bus semantics: delivery to every consumer group, retries, dead letters, resume from offset."""

import time

import pytest

from cloudrag.messaging.bus import InMemoryBus, RedisStreamBus, envelope


def _wait(cond, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_memory_bus_fanout_and_dead_letter():
    bus = InMemoryBus()
    a, b = [], []
    bus.subscribe("t", "g1", lambda e: a.append(e["payload"]["n"]))
    bus.subscribe("t", "g2", lambda e: b.append(e["payload"]["n"]))
    bus.subscribe("t", "bad", lambda e: 1 / 0)
    for n in range(3):
        bus.publish("t", envelope("x", {"n": n}))
    assert bus.wait_idle(5)
    assert a == b == [0, 1, 2]
    assert len(bus.dead_letters) == 3
    bus.close()


@pytest.fixture
def redis_bus(monkeypatch):
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    import redis

    monkeypatch.setattr(redis.Redis, "from_url",
                        classmethod(lambda cls, url, **kw: fakeredis.FakeRedis(server=server, **kw)))
    buses = []

    def make():
        b = RedisStreamBus("redis://fake", block_ms=50)
        buses.append(b)
        return b

    yield make
    for b in buses:
        b.close()


def test_redis_bus_groups_ack_and_resume(redis_bus):
    producer = redis_bus()
    for n in range(3):
        producer.publish("chunks", envelope("x", {"n": n}))

    seen0, seen1, handled_ids = [], [], []
    shard0 = redis_bus()
    shard0.subscribe("chunks", "shard-0", lambda e: seen0.append(e["payload"]["n"]), on_handled=handled_ids.append)
    shard1 = redis_bus()
    shard1.subscribe("chunks", "shard-1", lambda e: seen1.append(e["payload"]["n"]))
    assert _wait(lambda: len(seen0) == 3 and len(seen1) == 3)
    assert seen0 == seen1 == [0, 1, 2]  # every consumer group sees every event, in order
    assert producer._r.xpending("chunks", "shard-0")["pending"] == 0

    # shard-0 crashes after snapshotting at event #1; the restarted consumer resumes from that
    # offset and replays only what came after it
    shard0.close()
    for t in shard0._threads:
        t.join(timeout=2)
    producer.publish("chunks", envelope("x", {"n": 3}))
    replay = []
    redis_bus().subscribe("chunks", "shard-0", lambda e: replay.append(e["payload"]["n"]), start_id=handled_ids[1])
    assert _wait(lambda: 2 in replay and 3 in replay) and 0 not in replay and 1 not in replay

    # shard-1 crashes before it ever snapshotted: it must replay the whole stream, not resume
    # after its last acknowledgement (that would silently lose everything it held in memory)
    assert _wait(lambda: 3 in seen1)
    shard1.close()
    for t in shard1._threads:
        t.join(timeout=2)
    replay1 = []
    redis_bus().subscribe("chunks", "shard-1", lambda e: replay1.append(e["payload"]["n"]), start_id=None)
    assert _wait(lambda: replay1 == [0, 1, 2, 3])


def test_redis_bus_dead_letters_after_retries(redis_bus):
    bus = redis_bus()
    bus.claim_idle_ms = 0  # reclaim failed (pending) messages immediately
    bus.publish("chunks", envelope("x", {"n": 1}))
    calls = []

    def boom(e):
        calls.append(1)
        raise RuntimeError("index unavailable")

    bus.subscribe("chunks", "g", boom)
    assert _wait(lambda: bus._r.xlen("chunks.dlq") == 1)
    assert len(calls) == 3
