"""Sentence-aware chunking with section-level access control.

Documents may mark restricted sections inline::

    Public overview text...
    [[acl roles=finance,exec clearance=2]]
    Budget figures only finance and executives may read.
    [[/acl]]

Chunks never straddle a section boundary, and a chunk inside a section carries both the
document policy and the section policy (both must allow the reader).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..core.models import AccessPolicy
from ..core.text import split_sentences

_SECTION = re.compile(r"\[\[acl(?P<attrs>[^\]]*)\]\](?P<body>.*?)\[\[/acl\]\]", re.S | re.I)
_ATTR = re.compile(r"(\w+)\s*=\s*([^\s\]]+)")


@dataclass
class RawChunk:
    text: str
    section_policy: AccessPolicy | None


def parse_policy(attrs: str) -> AccessPolicy:
    kv = {k.lower(): v for k, v in _ATTR.findall(attrs)}
    split = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    return AccessPolicy(
        roles=split(kv.get("roles", "")),
        users=split(kv.get("users", "")),
        min_clearance=int(kv.get("clearance", 0)),
    )


def split_sections(text: str) -> list[tuple[str, AccessPolicy | None]]:
    out, cursor = [], 0
    for m in _SECTION.finditer(text):
        if m.start() > cursor:
            out.append((text[cursor : m.start()], None))
        out.append((m.group("body"), parse_policy(m.group("attrs"))))
        cursor = m.end()
    if cursor < len(text):
        out.append((text[cursor:], None))
    return [(t, p) for t, p in out if t.strip()]


def chunk_text(text: str, max_words: int = 120, overlap_sentences: int = 1) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    for body, policy in split_sections(text):
        sentences = []
        for para in re.split(r"\n\s*\n", body):
            sentences.extend(split_sentences(" ".join(para.split())))
        cur: list[str] = []
        words = 0
        for sent in sentences:
            n = len(sent.split())
            if cur and words + n > max_words:
                chunks.append(RawChunk(" ".join(cur), policy))
                cur = cur[-overlap_sentences:] if overlap_sentences else []
                words = sum(len(s.split()) for s in cur)
            cur.append(sent)
            words += n
        if cur:
            chunks.append(RawChunk(" ".join(cur), policy))
    return chunks
