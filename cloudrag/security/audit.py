"""Tamper-evident audit log.

Each record is one JSON line containing the SHA-256 hash of the previous record, so editing,
deleting or reordering any line breaks the chain from that point on. ``verify()`` walks the
chain. In Kubernetes each replica writes its own file (name includes the pod hostname) and the
files are shipped to write-once storage by the log collector.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any

from ..core.models import Principal

GENESIS = "0" * 64


def _digest(prev: str, body: dict) -> str:
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + canonical).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | None):
        self.path = path
        self._lock = threading.Lock()
        self._prev = GENESIS
        self._memory: list[dict] = []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    last = None
                    for line in f:
                        if line.strip():
                            last = line
                if last:
                    self._prev = json.loads(last)["hash"]

    def record(self, event: str, principal: Principal | None = None, trace_id: str = "", **details: Any) -> dict:
        with self._lock:
            body = {
                "ts": time.time(),
                "event": event,
                "tenant": principal.tenant_id if principal else None,
                "user": principal.user_id if principal else None,
                "trace_id": trace_id,
                "details": details,
                "prev": self._prev,
            }
            rec = dict(body, hash=_digest(self._prev, body))
            self._prev = rec["hash"]
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
            else:
                self._memory.append(rec)
            return rec

    def records(self) -> list[dict]:
        if not self.path:
            return list(self._memory)
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def verify(self) -> dict:
        prev = GENESIS
        recs = self.records()
        for i, rec in enumerate(recs):
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec.get("prev") != prev or _digest(prev, body) != rec.get("hash"):
                return {"ok": False, "records": len(recs), "first_invalid": i}
            prev = rec["hash"]
        return {"ok": True, "records": len(recs), "first_invalid": None}
