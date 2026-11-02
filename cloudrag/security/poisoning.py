"""Knowledge-base poisoning and prompt-injection detection.

Signals are combined with a noisy-OR into a risk score in [0, 1]:

* instruction-override / exfiltration / answer-steering phrases (indirect prompt injection)
* role and markup markers that try to impersonate system turns
* invisible Unicode (zero-width, bidi overrides, tag characters) used to hide payloads
* keyword stuffing (retrieval-bait that inflates lexical scores)
* near-duplicate flooding (many documents with almost identical content)

Ingestion quarantines chunks above ``quarantine_threshold`` and indexes chunks above
``flag_threshold`` with reduced trust; generation additionally strips injection sentences.
"""

from __future__ import annotations

import re
import threading
import zlib
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field

from ..core.text import content_terms, split_sentences, tokenize

_RULES: list[tuple[re.Pattern, float, str]] = [
    (re.compile(r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|any|system|preceding)\b.{0,20}\b(instructions?|prompts?|directions?|rules|context|messages?)", re.I), 0.9, "instruction_override"),
    (re.compile(r"\b(reveal|print|show|output|repeat|leak|exfiltrate|send|list|dump|give me)\b.{0,25}\b(system prompt|hidden instructions?|api[ _-]?keys?|passwords?|secrets|credentials|access tokens?)\b", re.I), 0.85, "exfiltration"),
    (re.compile(r"\b(you are now|from now on,? you|act as (an?|the) (unrestricted|unfiltered|evil|jailbroken)|pretend (to be|you are))\b", re.I), 0.6, "role_hijack"),
    (re.compile(r"(^|\n)\s*(system|assistant|developer)\s*:", re.I), 0.45, "role_marker"),
    (re.compile(r"<\s*/?\s*(system|instructions?|im_start|im_end|prompt)\s*>|\[/?INST\]", re.I), 0.7, "markup_injection"),
    (re.compile(r"\bdo not (tell|inform|mention|reveal)\b.{0,20}\b(user|anyone|human)\b", re.I), 0.6, "concealment"),
    (re.compile(r"\b(always|must|only) (answer|respond|reply|say|state|claim)\b", re.I), 0.5, "answer_steering"),
    (re.compile(r"\b(when|if) (asked|questioned|queried)\b.{0,80}\b(answer|respond|reply|say|state|claim)\b", re.I), 0.65, "answer_steering"),
    (re.compile(r"\b(jailbreak|dan mode|developer mode|no restrictions)\b", re.I), 0.6, "jailbreak"),
    (re.compile(r"\b(click|visit|go to|navigate to)\b.{0,15}https?://", re.I), 0.3, "link_lure"),
]

_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤﻿\U000e0000-\U000e007f]")


@dataclass
class Verdict:
    score: float
    action: str  # allow | flag | quarantine
    signals: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"score": round(self.score, 4), "action": self.action, "signals": self.signals}


def _noisy_or(weights: list[float]) -> float:
    p = 1.0
    for w in weights:
        p *= 1.0 - min(max(w, 0.0), 1.0)
    return 1.0 - p


def injection_signals(text: str) -> list[dict]:
    sig = []
    for pat, w, name in _RULES:
        m = pat.search(text)
        if m:
            sig.append({"signal": name, "weight": w, "evidence": m.group(0)[:80]})
    return sig


def injection_score(text: str) -> float:
    return _noisy_or([s["weight"] for s in injection_signals(text)])


def stuffing_signal(text: str) -> dict | None:
    terms = content_terms(text)
    if len(terms) < 30:
        return None
    counts = Counter(terms)
    top_term, top = counts.most_common(1)[0]
    ratio = top / len(terms)
    ttr = len(counts) / len(terms)
    if ratio > 0.15 or ttr < 0.25:
        return {"signal": "keyword_stuffing", "weight": 0.5,
                "evidence": f"'{top_term}' x{top} ({ratio:.0%}), type/token={ttr:.2f}"}
    return None


def simhash(text: str, bits: int = 64) -> int:
    toks = tokenize(text)
    shingles = [" ".join(toks[i : i + 3]) for i in range(max(1, len(toks) - 2))]
    v = [0] * bits
    for sh in shingles:
        h = zlib.crc32(sh.encode()) | (zlib.crc32(sh.encode()[::-1]) << 32)
        for b in range(bits):
            v[b] += 1 if (h >> b) & 1 else -1
    return sum(1 << b for b in range(bits) if v[b] > 0)


class PoisoningDetector:
    def __init__(self, flag_threshold: float = 0.35, quarantine_threshold: float = 0.7,
                 flood_window: int = 2000, flood_min_docs: int = 3, flood_hamming: int = 3):
        self.flag_threshold = flag_threshold
        self.quarantine_threshold = quarantine_threshold
        self.flood_min_docs = flood_min_docs
        self.flood_hamming = flood_hamming
        self._recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=flood_window))
        self._lock = threading.Lock()

    def document_signals(self, tenant_id: str, doc_id: str, raw_text: str) -> list[dict]:
        sig = []
        hidden = _INVISIBLE.findall(raw_text)
        if hidden:
            sig.append({"signal": "invisible_unicode", "weight": 0.6,
                        "evidence": f"{len(hidden)} hidden characters"})
        if len(tokenize(raw_text)) >= 20:
            h = simhash(raw_text)
            with self._lock:
                recent = self._recent[tenant_id]
                near = {d for hh, d in recent if d != doc_id and bin(hh ^ h).count("1") <= self.flood_hamming}
                recent.append((h, doc_id))
            if len(near) >= self.flood_min_docs:
                sig.append({"signal": "near_duplicate_flood", "weight": 0.4,
                            "evidence": f"{len(near)} near-identical documents"})
        return sig

    def scan_chunk(self, text: str, doc_signals: list[dict] | None = None) -> Verdict:
        signals = list(doc_signals or []) + injection_signals(text)
        stuffing = stuffing_signal(text)
        if stuffing:
            signals.append(stuffing)
        score = _noisy_or([s["weight"] for s in signals])
        if score >= self.quarantine_threshold:
            action = "quarantine"
        elif score >= self.flag_threshold:
            action = "flag"
        else:
            action = "allow"
        return Verdict(score=score, action=action, signals=signals)


def sanitize_context(text: str, threshold: float = 0.5) -> tuple[str, int]:
    """Drop sentences that read as instructions to the model; returns (text, n_removed)."""
    kept, removed = [], 0
    for sent in split_sentences(_INVISIBLE.sub("", text)):
        if injection_score(sent) >= threshold:
            removed += 1
        else:
            kept.append(sent)
    return " ".join(kept), removed
