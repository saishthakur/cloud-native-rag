"""Ingestion service: validate -> poisoning signals -> normalise -> PII redaction -> chunk
-> per-chunk poisoning verdict -> provenance signing -> publish to the bus.

Ingestion returns as soon as the event is published (HTTP 202 semantics); indexing happens
asynchronously in the indexing shards, which decouples write bursts from query serving.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import unicodedata
import uuid
from collections import Counter, deque

from ..config import Settings
from ..core.models import Chunk, DocumentIn, IngestReport, RequestContext
from ..indexing.service import TOPIC
from ..messaging.bus import envelope
from ..observability.metrics import DOCS_INGESTED, PII_REDACTIONS, SECURITY_EVENTS, STAGE_LATENCY
from ..security import BlockedInputError
from ..security.audit import AuditLog
from ..security.auth import require_roles
from ..security.pii import PIIRedactor
from ..security.poisoning import _INVISIBLE, PoisoningDetector
from ..security.provenance import ProvenanceSigner
from .chunker import chunk_text

_DOC_ID = re.compile(r"^[A-Za-z0-9._:\-/]{1,200}$")


class IngestionService:
    def __init__(self, settings: Settings, bus, signer: ProvenanceSigner, audit: AuditLog,
                 pii: PIIRedactor | None = None, detector: PoisoningDetector | None = None):
        self.settings = settings
        self.bus = bus
        self.signer = signer
        self.audit = audit
        self.pii = pii or PIIRedactor(settings.pii_mode, settings.provenance_key)
        self.detector = detector or PoisoningDetector(settings.poison_flag_threshold,
                                                      settings.poison_quarantine_threshold)
        self._quarantine: deque = deque(maxlen=5000)
        self._qlock = threading.Lock()
        self._qdir = os.path.join(settings.data_dir, "quarantine")

    def ingest(self, ctx: RequestContext, doc: DocumentIn) -> IngestReport:
        p = ctx.principal
        require_roles(p, "ingest", "editor")
        t0 = time.perf_counter()
        doc_id = doc.doc_id or uuid.uuid4().hex
        if not _DOC_ID.match(doc_id):
            raise BlockedInputError("invalid doc_id", {"doc_id": doc_id[:50]})
        if len(doc.text) > self.settings.max_doc_chars:
            raise BlockedInputError("document too large", {"chars": len(doc.text)})
        version = time.time_ns()

        doc_signals = self.detector.document_signals(p.tenant_id, doc_id, doc.text)
        text = unicodedata.normalize("NFKC", _INVISIBLE.sub("", doc.text))
        text, pii_counts = self.pii.redact(text)
        title, title_pii = self.pii.redact(unicodedata.normalize("NFKC", doc.title))
        pii_counts.update(title_pii)

        chunks: list[Chunk] = []
        quarantined: list[dict] = []
        flagged = 0
        max_score = 0.0
        all_signals: Counter = Counter()
        source_trust = self.settings.trust_for_source(doc.source)
        for i, rc in enumerate(chunk_text(text, self.settings.chunk_tokens, self.settings.chunk_overlap_sentences)):
            verdict = self.detector.scan_chunk(rc.text, doc_signals)
            max_score = max(max_score, verdict.score)
            all_signals.update(s["signal"] for s in verdict.signals)
            acl = [doc.acl] + ([rc.section_policy] if rc.section_policy else [])
            ch = Chunk(chunk_id=f"{doc_id}::{i:04d}", doc_id=doc_id, tenant_id=p.tenant_id, title=title,
                       text=rc.text, position=i, acl=acl, metadata=dict(doc.metadata, source=doc.source))
            if verdict.action == "quarantine":
                quarantined.append({"chunk_id": ch.chunk_id, "text": rc.text[:500], **verdict.as_dict()})
                continue
            ch.trust = source_trust
            if verdict.action == "flag":
                flagged += 1
                ch.trust = round(min(source_trust, 1.0 - verdict.score), 4)
                ch.flags = sorted({s["signal"] for s in verdict.signals})
            ch.provenance = self.signer.sign(ch, source=doc.source, ingested_by=p.user_id)
            chunks.append(ch)

        if quarantined and not chunks:
            status = "quarantined"
        elif quarantined:
            status = "partially_quarantined"
        else:
            status = "accepted"

        self.bus.publish(TOPIC, envelope("chunks.upsert", {
            "tenant_id": p.tenant_id, "doc_id": doc_id, "version": version,
            "chunks": [c.model_dump() for c in chunks],
        }, ctx.trace_id))

        if quarantined:
            SECURITY_EVENTS.inc(len(quarantined), type="chunk_quarantined")
            self._store_quarantine(p.tenant_id, doc_id, quarantined)
        if flagged:
            SECURITY_EVENTS.inc(flagged, type="chunk_flagged")
        for label, n in pii_counts.items():
            PII_REDACTIONS.inc(n, type=label)
        DOCS_INGESTED.inc(status=status)
        STAGE_LATENCY.observe(time.perf_counter() - t0, stage="ingest")

        report = IngestReport(
            job_id=uuid.uuid4().hex, doc_id=doc_id, tenant_id=p.tenant_id, status=status,
            chunks_indexed=len(chunks), chunks_quarantined=len(quarantined),
            pii_redactions=dict(pii_counts), version=version,
            poisoning={"max_score": round(max_score, 4), "flagged_chunks": flagged, "signals": dict(all_signals)},
        )
        self.audit.record("document.ingest", p, ctx.trace_id, doc_id=doc_id, status=status,
                          chunks=len(chunks), quarantined=len(quarantined), pii=dict(pii_counts),
                          source=doc.source)
        return report

    def delete(self, ctx: RequestContext, doc_id: str) -> dict:
        p = ctx.principal
        require_roles(p, "ingest", "editor")
        version = time.time_ns()
        self.bus.publish(TOPIC, envelope("doc.delete", {"tenant_id": p.tenant_id, "doc_id": doc_id,
                                                        "version": version}, ctx.trace_id))
        self.audit.record("document.delete", p, ctx.trace_id, doc_id=doc_id)
        return {"doc_id": doc_id, "status": "delete_requested", "version": version}

    def _store_quarantine(self, tenant: str, doc_id: str, items: list[dict]) -> None:
        rec = {"ts": time.time(), "tenant": tenant, "doc_id": doc_id, "chunks": items}
        with self._qlock:
            self._quarantine.append(rec)
            os.makedirs(self._qdir, exist_ok=True)
            with open(os.path.join(self._qdir, f"{tenant}.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def quarantine(self, ctx: RequestContext, limit: int = 100) -> list[dict]:
        require_roles(ctx.principal, "security")
        with self._qlock:
            items = [r for r in self._quarantine if r["tenant"] == ctx.principal.tenant_id]
        return items[-limit:]
