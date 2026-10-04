"""Deterministic non-stationary trajectory used by the demo seed and the benchmark.

T0 – initial world state (plus a populated background of distractor entities)
T1 – contradictory observations: relocation, job change, a low-authority rumor,
     a corroboration and a late-arriving historical fact
T2 – further drift (preference change, HQ relocation) so multi-hop answers change
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

T0 = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 4, 10, 9, 0, tzinfo=timezone.utc)
PUNE_VALID_FROM = datetime(2025, 6, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ScriptedObservation:
    phase: str
    timestamp: datetime
    text: str
    confidence: float = 0.9
    source: str = "user_statement"


@dataclass(frozen=True)
class BenchmarkQuery:
    question: str
    subject: str
    predicate: str
    expected: str
    stale: tuple[str, ...]
    historical: bool = False
    expected_also: tuple[str, ...] = field(default_factory=tuple)


_CITIES = ["Paris", "Tokyo", "Toronto", "Sydney", "Madrid", "Seoul", "Chicago", "Dublin", "Lisbon", "Austin"]
_COMPANIES = ["Globex", "Initech", "Umbrella", "Hooli", "Stark Industries", "Wayne Enterprises", "Soylent", "Vandelay"]
_PEOPLE = [
    "Bianca", "Carlos", "Dana", "Elif", "Farah", "Goran", "Hiro", "Ines", "Jonas", "Kemal", "Lena", "Marek",
    "Nadia", "Omar", "Priya", "Quentin", "Rosa", "Sven", "Tariq", "Uma", "Viktor", "Wen", "Ximena", "Yusuf",
]


def _distractors() -> list[ScriptedObservation]:
    rng = random.Random(7)
    out: list[ScriptedObservation] = []
    for i, company in enumerate(_COMPANIES):
        out.append(ScriptedObservation("T0", T0 + timedelta(minutes=i), f"{company} is headquartered in {rng.choice(_CITIES)}"))
    for i, person in enumerate(_PEOPLE):
        ts = T0 + timedelta(hours=1, minutes=i)
        out.append(ScriptedObservation("T0", ts, f"{person} lives in {rng.choice(_CITIES + ['London'])}"))
        out.append(ScriptedObservation("T0", ts, f"{person} works at {rng.choice(_COMPANIES + ['Acme'])}"))
        if i % 3 == 0:
            out.append(ScriptedObservation("T0", ts, f"{person} knows {rng.choice(_PEOPLE[:i] or ['Bob'])}"))
        if i % 4 == 1:
            move_ts = T1 - timedelta(days=rng.randint(3, 20))
            out.append(ScriptedObservation("T1", move_ts, f"{person} moved to {rng.choice(_CITIES)}"))
    return out


def trajectory() -> list[ScriptedObservation]:
    core = [
        # ---- T0: initial facts ------------------------------------------------
        ScriptedObservation("T0", T0, "Alice lives in Mumbai", 0.95),
        ScriptedObservation("T0", T0, "Alice works at Infosys", 0.95),
        ScriptedObservation("T0", T0, "Alice | has_role | Software Engineer", 0.9),
        ScriptedObservation("T0", T0, "Infosys is headquartered in Bengaluru", 0.99, "system_of_record"),
        ScriptedObservation("T0", T0, "Alice prefers Tea", 0.8),
        ScriptedObservation("T0", T0, "Bob lives in London", 0.9),
        ScriptedObservation("T0", T0, "Bob works at Acme", 0.9),
        ScriptedObservation("T0", T0, "Acme is headquartered in Berlin", 0.99, "system_of_record"),
        ScriptedObservation("T0", T0, "Alice knows Bob", 0.9),
        # ---- T1: non-stationary mutations -----------------------------------
        ScriptedObservation("T1", T1, "Alice moved from Mumbai to London", 0.9),
        ScriptedObservation("T1", T1, "Alice left Infosys for Acme", 0.9),
        ScriptedObservation("T1", T1, "Alice | has_role | Staff Engineer", 0.9),
        ScriptedObservation("T1", T1, "Alice reports to Bob", 0.85),
        ScriptedObservation("T1", T1 + timedelta(days=1), "Alice lives in Paris", 0.5, "hearsay"),
        ScriptedObservation("T1", T1 + timedelta(days=2), "Alice lives in London", 0.95, "sensor"),
        # Late-arriving historical record: reported at T1 but describing 2025,
        # so its valid time precedes everything on record for (Alice, lives_in).
        ScriptedObservation("T1", PUNE_VALID_FROM, "Alice | lives_in | Pune", 0.85, "system_of_record"),
        # ---- T2: further drift -----------------------------------------------
        ScriptedObservation("T2", T2, "Alice prefers Coffee", 0.85),
        ScriptedObservation("T2", T2, "Acme moved its headquarters to Munich", 0.99, "system_of_record"),
    ]
    # Ingest in arrival order (by phase), not valid-time order, so the Pune
    # record genuinely arrives late and exercises retroactive reconciliation.
    order = {"T0": 0, "T1": 1, "T2": 2}
    return sorted(core + _distractors(), key=lambda o: (order[o.phase], o.timestamp))


QUERIES: tuple[BenchmarkQuery, ...] = (
    BenchmarkQuery("Where does Alice live?", "Alice", "lives_in", "London", ("Mumbai", "Paris", "Pune")),
    BenchmarkQuery("Where does Alice work?", "Alice", "works_at", "Acme", ("Infosys",)),
    BenchmarkQuery("What is Alice's role?", "Alice", "has_role", "Staff Engineer", ("Software Engineer",)),
    BenchmarkQuery("What does Alice prefer?", "Alice", "prefers", "Coffee", ("Tea",)),
    BenchmarkQuery(
        "Which city is Alice's employer headquartered in?", "Acme", "headquartered_in", "Munich",
        ("Berlin", "Bengaluru"),
    ),
    BenchmarkQuery(
        "Where did Alice live before London and why did it change?", "Alice", "lives_in", "London",
        ("Paris",), historical=True, expected_also=("Mumbai",),
    ),
)


def phase_boundaries() -> dict[str, datetime]:
    return {"T0": T0, "T1": T1, "T2": T2}


def latest_timestamp(observations: Optional[list[ScriptedObservation]] = None) -> datetime:
    return max(o.timestamp for o in (observations or trajectory()))
