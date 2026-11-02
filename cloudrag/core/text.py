"""Tokenisation, light stemming and sentence splitting shared by every lexical component."""

from __future__ import annotations

import re
from functools import lru_cache

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers herself him himself his how i if in into is it its itself
    just me more most my myself no nor not now of off on once only or other our ours ourselves out
    over own same she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when where which
    while who whom whose why will with would you your yours yourself yourselves also tell please
    give name list show find describe explain does did done get got may might must shall s t
    """.split()
)

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n{2,}")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


_SUFFIXES = (
    ("ies", "y", 5),
    ("ied", "y", 5),
    ("ing", "", 6),
    ("ers", "", 6),
    ("er", "", 5),
    ("ed", "", 4),
    ("es", "", 4),
    ("s", "", 4),
)


@lru_cache(maxsize=200_000)
def stem(tok: str) -> str:
    """Tiny two-pass suffix stripper that conflates the inflections retrieval cares about:
    use/uses/used, supply/supplies/supplied/supplier, headquarters/headquartered, locate/located."""
    if len(tok) <= 2 or not tok.isalpha():
        return tok
    for _ in range(2):
        for suffix, repl, min_len in _SUFFIXES:
            if (tok.endswith(suffix) and len(tok) >= min_len and not tok.endswith("ss")
                    and not (suffix == "ed" and tok.endswith("eed"))):
                tok = tok[: len(tok) - len(suffix)] + repl
                break
        else:
            break
    if len(tok) >= 3 and tok.endswith("e"):
        tok = tok[:-1]
    if tok.endswith("i"):
        tok = tok[:-1] + "y"
    return tok


def content_terms(text: str) -> list[str]:
    """Stemmed non-stopword tokens; hyphenated codes also contribute their parts."""
    out: list[str] = []
    for t in tokenize(text):
        if t in STOPWORDS:
            continue
        out.append(stem(t))
        if "-" in t:
            out.extend(p for p in t.split("-") if p and p not in STOPWORDS)
    return out


def bigrams(terms: list[str]) -> set[tuple[str, str]]:
    return set(zip(terms, terms[1:]))


def split_sentences(text: str) -> list[str]:
    parts = (s.strip() for s in _SENT.split(text))
    return [p for p in parts if p]


def normalize_answer(s: str) -> str:
    return " ".join(tokenize(s))
