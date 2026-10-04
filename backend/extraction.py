"""Deterministic, rule-based statement-to-triple extractor.

Natural-language observations are mapped onto ontology predicates with ordered
regular-expression rules. Determinism matters here: the same statement must
always yield the same triples so the ledger is reproducible. Statements that
match no rule can still be ingested with the explicit pipe syntax
``Subject | predicate | Object``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from models import TripleObservation, new_uuid, utc_now

_NAME = r"(?P<{g}>[A-Z0-9][\w'.&-]*(?:\s+(?:of\s+|de\s+|the\s+)?[A-Z0-9][\w'.&-]*)*)"
_PHRASE = r"(?P<{g}>[\w'.&-]+(?:\s+[\w'.&-]+){{0,5}}?)"


def _n(group: str) -> str:
    return _NAME.format(g=group)


def _p(group: str) -> str:
    return _PHRASE.format(g=group)


_ADV = r"(?:(?:now|currently|recently|just|has|have|had|officially)\s+)*"
_ART = r"(?:(?:a|an|the)\s+)?"


@dataclass(frozen=True)
class _Rule:
    name: str
    pattern: re.Pattern[str]
    build: Callable[[re.Match[str]], list[tuple[str, str, str, Optional[str]]]]


def _rule(name: str, regex: str, build: Callable[[re.Match[str]], list[tuple[str, str, str, Optional[str]]]]) -> _Rule:
    return _Rule(name, re.compile(r"^\s*" + regex + r"\s*[.!]?\s*$", re.IGNORECASE), build)


def _g(m: re.Match[str], key: str) -> str:
    return m.group(key).strip()


_RULES: tuple[_Rule, ...] = (
    _rule(
        "relocation_from_to",
        _n("s") + r"\s+" + _ADV + r"(?:moved|relocated|migrated)\s+(?:back\s+)?from\s+" + _n("old")
        + r"\s+to\s+" + _n("new"),
        lambda m: [(_g(m, "s"), "lives_in", _g(m, "new"), _g(m, "old"))],
    ),
    _rule(
        "relocation_to",
        _n("s") + r"\s+" + _ADV + r"(?:moved|relocated|migrated)\s+(?:back\s+)?to\s+" + _n("new"),
        lambda m: [(_g(m, "s"), "lives_in", _g(m, "new"), None)],
    ),
    _rule(
        "residence",
        _n("s") + r"\s+" + _ADV + r"(?:lives|resides|is\s+" + _ADV + r"(?:living|based|residing))\s+in\s+" + _n("o"),
        lambda m: [(_g(m, "s"), "lives_in", _g(m, "o"), None)],
    ),
    _rule(
        "job_change",
        _n("s") + r"\s+" + _ADV + r"left\s+" + _n("old") + r"\s+(?:for|to\s+join)\s+" + _n("new"),
        lambda m: [(_g(m, "s"), "works_at", _g(m, "new"), _g(m, "old"))],
    ),
    _rule(
        "employment",
        _n("s") + r"\s+" + _ADV + r"(?:works\s+(?:at|for)|joined|is\s+" + _ADV + r"(?:working\s+(?:at|for)|employed\s+(?:at|by)))\s+"
        + _n("o"),
        lambda m: [(_g(m, "s"), "works_at", _g(m, "o"), None)],
    ),
    _rule(
        "role_at_org",
        _n("s") + r"\s+(?:is|became|was\s+promoted\s+to)\s+" + _ADV + _ART + _p("role") + r"\s+(?:at|of)\s+" + _n("org"),
        lambda m: [
            (_g(m, "s"), "has_role", _g(m, "role").title(), None),
            (_g(m, "s"), "works_at", _g(m, "org"), None),
        ],
    ),
    _rule(
        "reports_to",
        _n("s") + r"\s+" + _ADV + r"(?:reports|is\s+reporting)\s+to\s+" + _n("o"),
        lambda m: [(_g(m, "s"), "reports_to", _g(m, "o"), None)],
    ),
    _rule(
        "marriage",
        _n("s") + r"\s+" + _ADV + r"(?:is\s+married\s+to|married|got\s+married\s+to)\s+" + _n("o"),
        lambda m: [(_g(m, "s"), "married_to", _g(m, "o"), None)],
    ),
    _rule(
        "headquarters",
        _n("s") + r"\s+" + _ADV + r"(?:is\s+headquartered\s+in|moved\s+(?:its\s+)?(?:headquarters|hq)\s+to|has\s+(?:its\s+)?(?:headquarters|hq)\s+in)\s+"
        + _n("o"),
        lambda m: [(_g(m, "s"), "headquartered_in", _g(m, "o"), None)],
    ),
    _rule(
        "location",
        _n("s") + r"\s+(?:is|was)\s+" + _ADV + r"(?:located|stored|situated|parked)\s+(?:in|at)\s+" + _n("o"),
        lambda m: [(_g(m, "s"), "located_in", _g(m, "o"), None)],
    ),
    _rule(
        "preference",
        _n("s") + r"\s+" + _ADV + r"(?:prefers|likes\s+best|switched\s+to\s+preferring)\s+" + _p("o"),
        lambda m: [(_g(m, "s"), "prefers", _g(m, "o").title(), None)],
    ),
    _rule(
        "ownership",
        _n("s") + r"\s+" + _ADV + r"(?:owns|bought|acquired)\s+" + _ART + _n("o"),
        lambda m: [(_g(m, "s"), "owns", _g(m, "o"), None)],
    ),
    _rule(
        "acquaintance",
        _n("s") + r"\s+" + _ADV + r"(?:knows|met|is\s+friends\s+with)\s+" + _n("o"),
        lambda m: [(_g(m, "s"), "knows", _g(m, "o"), None)],
    ),
    _rule(
        "membership",
        _n("s") + r"\s+" + _ADV + r"(?:is\s+a\s+member\s+of|joined\s+the\s+team)\s+" + _n("o"),
        lambda m: [(_g(m, "s"), "member_of", _g(m, "o"), None)],
    ),
    _rule(
        "status",
        _n("s") + r"\s+(?:status\s+is|is\s+now|has\s+status)\s+" + _p("o"),
        lambda m: [(_g(m, "s"), "has_status", _g(m, "o").title(), None)],
    ),
    _rule(
        "possessive_attribute",
        _n("s") + r"'s\s+" + _p("attr") + r"\s+is\s+" + _ADV + _p("o"),
        lambda m: [(_g(m, "s"), _g(m, "attr"), _g(m, "o").title(), None)],
    ),
)

_PIPE = re.compile(r"^\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!;])\s+|\s*\n+\s*")


class ExtractionError(ValueError):
    """Raised when a statement cannot be mapped onto any triple."""


def _strip_article(value: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", value.strip(), flags=re.IGNORECASE).strip(" .,'\"")


def extract_triples(
    text: str,
    *,
    confidence: float = 0.9,
    source: str = "user_statement",
    timestamp: Optional[datetime] = None,
    provenance_id: Optional[str] = None,
) -> list[TripleObservation]:
    """Split ``text`` into sentences and map each onto one or more triples."""
    ts = timestamp or utc_now()
    prov = provenance_id or new_uuid()
    triples: list[TripleObservation] = []
    unmatched: list[str] = []

    for sentence in (s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()):
        pipe = _PIPE.match(sentence)
        if pipe:
            raw = [(pipe.group(1), pipe.group(2), pipe.group(3), None)]
        else:
            raw = []
            for rule in _RULES:
                match = rule.pattern.match(sentence)
                if match:
                    raw = rule.build(match)
                    break
        if not raw:
            unmatched.append(sentence)
            continue
        for subject, predicate, obj, prior in raw:
            triples.append(
                TripleObservation(
                    subject=_strip_article(subject),
                    predicate=predicate,
                    object=_strip_article(obj),
                    confidence=confidence,
                    source=source,
                    provenance_id=prov,
                    timestamp=ts,
                    expected_prior_object=_strip_article(prior) if prior else None,
                    raw_text=sentence,
                )
            )

    if unmatched:
        raise ExtractionError(
            "Could not parse: "
            + "; ".join(repr(s) for s in unmatched)
            + ". Try e.g. 'Alice moved from Mumbai to London', 'Alice works at Acme', "
            "or the explicit form 'Alice | lives_in | London'."
        )
    return triples
