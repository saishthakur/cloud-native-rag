"""Adversarial input filtering for user queries."""

from __future__ import annotations

import re
import unicodedata

from . import BlockedInputError
from .poisoning import _INVISIBLE, _noisy_or, injection_signals

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class QueryGuard:
    def __init__(self, block_threshold: float = 0.8, max_chars: int = 2000):
        self.block_threshold = block_threshold
        self.max_chars = max_chars

    def inspect(self, query: str) -> tuple[str, dict]:
        """Returns (cleaned query, verdict). Raises BlockedInputError when the query is rejected."""
        if len(query) > self.max_chars:
            raise BlockedInputError("query too long", {"signal": "length", "chars": len(query)})
        signals = injection_signals(query)
        if _INVISIBLE.search(query):
            signals.append({"signal": "invisible_unicode", "weight": 0.5, "evidence": "hidden characters"})
        cleaned = unicodedata.normalize("NFKC", _CONTROL.sub(" ", _INVISIBLE.sub("", query))).strip()
        if not cleaned:
            raise BlockedInputError("empty query", {"signal": "empty"})
        score = _noisy_or([s["weight"] for s in signals])
        verdict = {"score": round(score, 4), "signals": signals}
        if score >= self.block_threshold:
            raise BlockedInputError("query rejected by adversarial-input filter", verdict)
        return cleaned, verdict
