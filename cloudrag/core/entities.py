"""Lightweight, dependency-free entity and relation extraction for the knowledge graph.

Entities are capitalised phrases ("Project Aurora", "Northwind Components") and product /
part codes ("ZX-200"). Relations are the predicate words between two entities in the same
sentence ("Project Aurora" --uses--> "ZX-200 Sensor"). An NER model can replace this by
implementing the same two functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .text import STOPWORDS, split_sentences, tokenize

_CAP = r"[A-Z][A-Za-z0-9&]*(?:[-'][A-Za-z0-9]+)*"
_PHRASE = re.compile(rf"\b{_CAP}(?:\s+(?:of\s+|for\s+|de\s+)?{_CAP})*")

_LEADING_NOISE = frozenset(
    """
    The A An This That These Those In On At It Its Our We He She They Who What Which When Where
    Why How Is Are Was Were Does Did Do Can Could Would Should And But For If As After Before
    During Since Each All Also According However Both Please Ignore List Summarize Summarise
    Compare Tell Give Name Find Show Describe Explain Every Any Some There Then Their His Her
    Under Over From By With Without Within Into Note Today Yesterday Additionally Meanwhile
    """.split()
)
_DROP = frozenset(
    """
    January February March April May June July August September October November December
    Monday Tuesday Wednesday Thursday Friday Saturday Sunday I Q1 Q2 Q3 Q4
    """.split()
)
GENERIC_HEADS = frozenset({"project", "team", "department", "dept", "the", "programme", "program"})


@dataclass(frozen=True)
class Mention:
    name: str
    key: str
    start: int
    end: int


def entity_key(name: str) -> str:
    toks = [t for t in tokenize(name) if t != "the"]
    return " ".join(toks)


def entity_aliases(name: str) -> list[tuple[str, ...]]:
    """Token sequences that should link to this entity (full name, name minus generic head)."""
    toks = tuple(t for t in tokenize(name) if t != "the")
    out = [toks]
    if len(toks) >= 2 and toks[0] in GENERIC_HEADS:
        out.append(toks[1:])
    return [a for a in out if a]


def extract_mentions(text: str) -> list[Mention]:
    out: list[Mention] = []
    for m in _PHRASE.finditer(text):
        words = m.group(0).split()
        start = m.start()
        while words and (words[0] in _LEADING_NOISE or words[0] in _DROP):
            start += len(words[0]) + 1
            words = words[1:]
        while words and words[-1].lower() in {"of", "for", "de"}:
            words = words[:-1]
        if not words:
            continue
        name = " ".join(words)
        if len(words) == 1:
            w = words[0]
            is_code = any(c.isdigit() for c in w) or (w.isupper() and len(w) >= 2)
            if (not is_code and len(w) < 3) or w in _DROP or w.lower() in STOPWORDS:
                continue
        key = entity_key(name)
        if key:
            out.append(Mention(name=name, key=key, start=start, end=start + len(name)))
    return out


def extract_relations(text: str, max_gap_words: int = 8) -> list[tuple[Mention, str, Mention]]:
    """(head, predicate, tail) triples between neighbouring mentions within each sentence."""
    triples: list[tuple[Mention, str, Mention]] = []
    for sent in split_sentences(text):
        ms = extract_mentions(sent)
        for i, a in enumerate(ms):
            for b in ms[i + 1 : i + 3]:
                if a.key == b.key:
                    continue
                between = [w for w in tokenize(sent[a.end : b.start])]
                if len(between) > max_gap_words:
                    continue
                pred_words = [w for w in between if w not in {"the", "a", "an"}]
                pred = " ".join(pred_words) if pred_words else "related to"
                triples.append((a, pred, b))
    return triples
