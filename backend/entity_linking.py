"""Auditable entity linking: map surface names onto canonical entities before reconciliation.

Without linking, "Alice", "Dr. Alice" and "alice" are already one entity (ids are
slugs of the name), but "Acme" and "Acme Inc." or "Robert" and "Bob" are not.
The linker resolves a surface name to a canonical name in two ways:

explicit  aliases declared by an operator (``POST /api/entities/alias``), e.g.
          ``Bob -> Robert Smith``. Always applied.
auto      (opt-in) honorifics and legal-form suffixes are stripped ("Dr.",
          "Prof.", "Inc.", "Ltd", "GmbH", ...) and the remainder is linked to an
          entity that already exists under that normalized key. A new surface
          form is never linked to an entity that does not exist yet, and an
          ambiguous key (two existing entities normalize the same way) is left
          alone.

Every link is a ledger record (``mutation_type = "ALIAS"``) with a deterministic
rationale, chained like any other mutation, so a merge is as auditable as a
fact change and is restored by replay.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Callable, Literal, Optional

from models import entity_id_for

LinkingMode = Literal["explicit", "auto"]

_HONORIFICS = ("dr", "mr", "mrs", "ms", "mx", "prof", "sir", "dame", "rev", "hon")
_LEGAL_FORMS = ("inc", "incorporated", "ltd", "limited", "llc", "llp", "plc", "corp", "corporation", "co",
                "gmbh", "ag", "sa", "sas", "bv", "nv", "pvt", "pty", "oy", "ab", "srl", "spa", "kk")
_HONORIFIC_RE = re.compile(r"^(?:(?:" + "|".join(_HONORIFICS) + r")\.?\s+)+", re.IGNORECASE)
_LEGAL_RE = re.compile(r"(?:,?\s+(?:" + "|".join(_LEGAL_FORMS) + r")\.?)+$", re.IGNORECASE)


def normalize_surface(name: str) -> str:
    """Strip honorifics and legal-form suffixes: "Dr. Alice Rao" -> "Alice Rao", "Acme, Inc." -> "Acme"."""
    stripped = _LEGAL_RE.sub("", _HONORIFIC_RE.sub("", name.strip())).strip(" ,.")
    return stripped or name.strip()


@dataclass(frozen=True)
class AliasLink:
    alias: str
    alias_id: str
    canonical: str
    canonical_id: str
    mode: LinkingMode
    rationale: str

    def snapshot(self) -> dict:
        return {"alias": {"alias": self.alias, "alias_id": self.alias_id, "canonical": self.canonical,
                          "canonical_id": self.canonical_id, "mode": self.mode}}


class EntityLinker:
    """Alias table plus the optional normalization rule. Thread-safe."""

    def __init__(self, mode: LinkingMode = "explicit") -> None:
        self.mode = mode
        self._aliases: dict[str, AliasLink] = {}  # alias entity id -> link
        self._by_key: dict[str, set[str]] = {}  # normalized key -> entity ids seen under it
        self._lock = threading.RLock()

    def note_entity(self, entity_id: str, name: str) -> None:
        """Index an entity that exists in the graph under its normalized key (auto mode lookups)."""
        with self._lock:
            self._by_key.setdefault(normalize_key(name), set()).add(entity_id)

    # -- table ------------------------------------------------------------- #

    def aliases(self) -> list[AliasLink]:
        with self._lock:
            return sorted(self._aliases.values(), key=lambda a: (a.canonical_id, a.alias_id))

    def restore(self, snapshot: dict) -> None:
        """Re-apply an ALIAS ledger snapshot (replay)."""
        a = snapshot["alias"]
        with self._lock:
            self._aliases[a["alias_id"]] = AliasLink(a["alias"], a["alias_id"], a["canonical"], a["canonical_id"],
                                                     a["mode"], "")

    def remove(self, alias_id: str) -> None:
        with self._lock:
            self._aliases.pop(alias_id, None)

    def clear(self) -> None:
        with self._lock:
            self._aliases.clear()
            self._by_key.clear()

    def declare(self, alias: str, canonical: str) -> AliasLink:
        """Validate an operator-declared alias (the caller commits it to the ledger)."""
        alias_id, canonical_id = entity_id_for(alias), entity_id_for(canonical)
        if alias_id == canonical_id:
            raise ValueError(f"{alias!r} and {canonical!r} already resolve to the same entity id {alias_id!r}")
        with self._lock:
            target = self._aliases.get(canonical_id)
            if target is not None:  # never chain aliases: point at the final canonical entity
                canonical, canonical_id = target.canonical, target.canonical_id
                if canonical_id == alias_id:
                    raise ValueError(f"Alias cycle: {canonical!r} is already an alias of {alias!r}")
            existing = self._aliases.get(alias_id)
            if existing is not None and existing.canonical_id != canonical_id:
                raise ValueError(f"{alias!r} is already an alias of {existing.canonical!r}")
            if any(link.canonical_id == alias_id for link in self._aliases.values()):
                raise ValueError(f"{alias!r} is itself a canonical entity with aliases; it cannot become an alias")
        return AliasLink(alias.strip(), alias_id, canonical.strip(), canonical_id, "explicit",
                         f"Operator declared '{alias.strip()}' an alias of '{canonical.strip()}': future observations "
                         f"naming '{alias.strip()}' are recorded against entity '{canonical_id}'.")

    # -- resolution -------------------------------------------------------- #

    def resolve(self, name: str, lookup: Callable[[str], Optional[str]]) -> tuple[str, Optional[AliasLink]]:
        """Return ``(canonical surface name, new link to record or None)``.

        ``lookup(entity_id)`` returns the name of an existing entity, or None.
        A link is returned only when auto mode creates one not yet in the table;
        the caller must commit it to the ledger or call :meth:`remove` on failure.
        """
        name = name.strip()
        eid = entity_id_for(name)
        with self._lock:
            link = self._aliases.get(eid)
            if link is not None:
                return link.canonical, None
            if self.mode != "auto" or lookup(eid) is not None:
                return name, None
            normalized = normalize_surface(name)
            key = entity_id_for(normalized)
            matches = {i for i in self._by_key.get(key, set()) if i != eid and lookup(i) is not None}
            if len(matches) != 1:
                return name, None
            target_id = matches.pop()
            target = self._aliases.get(target_id)
            canonical, canonical_id = (target.canonical, target.canonical_id) if target else                 (lookup(target_id) or target_id, target_id)
            new = AliasLink(name, eid, canonical, canonical_id, "auto",
                            f"Automatic entity linking: '{name}' normalizes to '{normalized}' (honorifics and "
                            f"legal-form suffixes removed), which matches exactly one existing entity "
                            f"'{canonical}' [{canonical_id}].")
            self._aliases[eid] = new
            return canonical, new


def normalize_key(name: str) -> str:
    return entity_id_for(normalize_surface(name))
