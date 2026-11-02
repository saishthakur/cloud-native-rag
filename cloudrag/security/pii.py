"""Ingestion-time PII detection and redaction.

Modes:
  redact        "alice@corp.com"  -> "[EMAIL]"
  pseudonymize  "alice@corp.com"  -> "[EMAIL:3f9a1c2b]"  (keyed hash: same value -> same token,
                                    so records stay linkable without exposing the value)
  off           no change
"""

from __future__ import annotations

import hashlib
import hmac
import re
from collections import Counter


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _card_ok(s: str) -> bool:
    digits = re.sub(r"\D", "", s)
    return 13 <= len(digits) <= 19 and _luhn_ok(digits)


# (label, pattern, validator, priority) - lower priority number wins overlaps.
_PATTERNS = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None, 0),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _card_ok, 1),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), None, 2),
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}[ -]\d{4}[ -]\d{4}\b"), None, 3),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None, 4),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){3,7}(?: ?[A-Z0-9]{1,3})?\b"), None, 5),
    ("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), None, 6),
    (
        "PHONE",
        re.compile(r"(?<![\w-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,5}\)[\s.-]?)?\d{2,5}(?:[\s.-]?\d{2,6}){1,3}(?![\w-])"),
        lambda s: 10 <= len(re.sub(r"\D", "", s)) <= 13,
        7,
    ),
]


class PIIRedactor:
    def __init__(self, mode: str = "redact", key: str = ""):
        if mode not in {"redact", "pseudonymize", "off"}:
            raise ValueError(f"unknown PII mode {mode!r}")
        self.mode = mode
        self._key = key.encode()

    def find(self, text: str) -> list[tuple[int, int, str]]:
        found: list[tuple[int, int, int, str]] = []
        for label, pat, validator, prio in _PATTERNS:
            for m in pat.finditer(text):
                if validator is None or validator(m.group(0)):
                    found.append((m.start(), m.end(), prio, label))
        # keep non-overlapping spans, preferring higher-priority then longer matches
        found.sort(key=lambda x: (x[2], -(x[1] - x[0])))
        taken: list[tuple[int, int, str]] = []
        for s, e, _, label in found:
            if all(e <= ts or s >= te for ts, te, _ in taken):
                taken.append((s, e, label))
        return sorted(taken)

    def _replacement(self, label: str, value: str) -> str:
        if self.mode == "pseudonymize":
            digest = hmac.new(self._key, value.encode(), hashlib.sha256).hexdigest()[:8]
            return f"[{label}:{digest}]"
        return f"[{label}]"

    def redact(self, text: str) -> tuple[str, Counter]:
        counts: Counter = Counter()
        if self.mode == "off" or not text:
            return text, counts
        spans = self.find(text)
        if not spans:
            return text, counts
        out, cursor = [], 0
        for s, e, label in spans:
            out.append(text[cursor:s])
            out.append(self._replacement(label, text[s:e]))
            counts[label] += 1
            cursor = e
        out.append(text[cursor:])
        return "".join(out), counts
