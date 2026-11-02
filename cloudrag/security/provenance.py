"""Chunk provenance and integrity.

At ingestion every chunk gets a SHA-256 content hash and an HMAC signature binding
(tenant, doc, chunk id, content hash, ACL, source). The retrieval service re-verifies each
chunk before it can reach the generator, so content or ACL tampering inside the index
(or in transit between services) is detected and the chunk is dropped.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time

from .. import PIPELINE_VERSION
from ..core.models import Chunk, Provenance


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class ProvenanceSigner:
    def __init__(self, key: str):
        self._key = key.encode()

    def _payload(self, chunk: Chunk, digest: str, source: str) -> bytes:
        acl = json.dumps([p.model_dump() for p in chunk.acl], sort_keys=True, separators=(",", ":"))
        return "|".join(
            [chunk.tenant_id, chunk.doc_id, chunk.chunk_id, digest, acl, source, f"{chunk.trust:.4f}"]
        ).encode()

    def sign(self, chunk: Chunk, source: str, ingested_by: str) -> Provenance:
        digest = content_hash(chunk.text)
        sig = hmac.new(self._key, self._payload(chunk, digest, source), hashlib.sha256).hexdigest()
        return Provenance(
            source=source,
            content_hash=digest,
            signature=sig,
            ingested_at=time.time(),
            ingested_by=ingested_by,
            pipeline_version=PIPELINE_VERSION,
        )

    def verify(self, chunk: Chunk) -> bool:
        p = chunk.provenance
        if p is None:
            return False
        digest = content_hash(chunk.text)
        if not hmac.compare_digest(digest, p.content_hash):
            return False
        expected = hmac.new(self._key, self._payload(chunk, digest, p.source), hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, p.signature)
