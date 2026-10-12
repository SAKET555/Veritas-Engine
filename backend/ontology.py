"""Predicate ontology for the temporal knowledge graph.

The ontology is the single source of truth for *cardinality* (whether a
predicate admits one or many simultaneously-valid objects per subject).
Cardinality is what makes a new observation a *conflict* rather than an
additional fact, so the reconciliation engine depends on it directly.

Aliases are the lexical surface forms a natural-language question may use to
refer to a predicate; the pruning engine uses them to compute relational
relevance without needing an embedding model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class PredicateSpec:
    """Static schema for a single relation type."""

    name: str
    functional: bool
    subject_type: str
    object_type: str
    description: str
    aliases: frozenset[str] = field(default_factory=frozenset)


_SPECS: tuple[PredicateSpec, ...] = (
    PredicateSpec(
        name="lives_in",
        functional=True,
        subject_type="Person",
        object_type="Location",
        description="Current residence of a person.",
        aliases=frozenset({"live", "lives", "living", "reside", "resides", "residence",
                           "home", "moved", "move", "relocated", "based"}),
    ),
    PredicateSpec(
        name="works_at",
        functional=True,
        subject_type="Person",
        object_type="Organization",
        description="Current employer of a person.",
        aliases=frozenset({"work", "works", "working", "employer", "employed", "job",
                           "company", "joined", "employ"}),
    ),
    PredicateSpec(
        name="has_role",
        functional=True,
        subject_type="Person",
        object_type="Role",
        description="Current job title of a person.",
        aliases=frozenset({"role", "title", "position", "job", "promoted"}),
    ),
    PredicateSpec(
        name="reports_to",
        functional=True,
        subject_type="Person",
        object_type="Person",
        description="Direct manager of a person.",
        aliases=frozenset({"report", "reports", "manager", "boss", "manages", "supervisor"}),
    ),
    PredicateSpec(
        name="married_to",
        functional=True,
        subject_type="Person",
        object_type="Person",
        description="Current spouse of a person.",
        aliases=frozenset({"married", "spouse", "wife", "husband", "partner"}),
    ),
    PredicateSpec(
        name="prefers",
        functional=True,
        subject_type="Person",
        object_type="Preference",
        description="Current stated preference of a person.",
        aliases=frozenset({"prefer", "prefers", "preference", "favorite", "favourite", "likes"}),
    ),
    PredicateSpec(
        name="headquartered_in",
        functional=True,
        subject_type="Organization",
        object_type="Location",
        description="Headquarters location of an organization.",
        aliases=frozenset({"headquartered", "headquarters", "hq", "based", "office", "city"}),
    ),
    PredicateSpec(
        name="located_in",
        functional=True,
        subject_type="Entity",
        object_type="Location",
        description="Physical location of an object, place or asset.",
        aliases=frozenset({"located", "location", "situated", "sits", "stored"}),
    ),
    PredicateSpec(
        name="has_status",
        functional=True,
        subject_type="Entity",
        object_type="Status",
        description="Current lifecycle status of an entity.",
        aliases=frozenset({"status", "state", "currently", "condition"}),
    ),
    PredicateSpec(
        name="owns",
        functional=False,
        subject_type="Entity",
        object_type="Entity",
        description="Ownership; an entity may own many things at once.",
        aliases=frozenset({"own", "owns", "owned", "possess", "has"}),
    ),
    PredicateSpec(
        name="knows",
        functional=False,
        subject_type="Person",
        object_type="Person",
        description="Acquaintance relation; many-to-many.",
        aliases=frozenset({"know", "knows", "friend", "friends", "acquainted"}),
    ),
    PredicateSpec(
        name="member_of",
        functional=False,
        subject_type="Person",
        object_type="Organization",
        description="Group membership; a person may belong to many groups.",
        aliases=frozenset({"member", "belongs", "group", "team", "club"}),
    ),
)

PREDICATES: dict[str, PredicateSpec] = {spec.name: spec for spec in _SPECS}

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_predicate(raw: str) -> str:
    """Canonicalize a free-form predicate into snake_case."""
    return _NORMALIZE_RE.sub("_", raw.strip().lower()).strip("_")


@lru_cache(maxsize=4096)
def get_spec(predicate: str) -> PredicateSpec:
    """Return the spec for a predicate; unknown predicates default to functional.

    Defaulting to functional is the *conservative* choice for a state-editing
    system: an unknown attribute-like relation (``"favorite_color"``) is far
    more likely to be single-valued than set-valued, and treating it as such
    guarantees we never surface two contradictory active values.
    """
    name = normalize_predicate(predicate)
    spec = PREDICATES.get(name)
    if spec is not None:
        return spec
    tokens = frozenset(t for t in name.split("_") if t)
    return PredicateSpec(
        name=name,
        functional=True,
        subject_type="Entity",
        object_type="Entity",
        description="Ad-hoc predicate (not in ontology); treated as functional.",
        aliases=tokens,
    )


@lru_cache(maxsize=4096)
def predicate_tokens(predicate: str) -> frozenset[str]:
    """All lexical tokens that refer to the predicate (name parts + aliases)."""
    spec = get_spec(predicate)
    name_tokens = {t for t in spec.name.split("_") if len(t) > 2}
    return frozenset(name_tokens | set(spec.aliases))
