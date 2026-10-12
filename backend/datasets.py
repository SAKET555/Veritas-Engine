"""Synthetic multi-session temporal benchmark with *independent* ground truth.

The generator simulates a world whose state drifts across sessions and records
the true validity interval of every value. Observations about that world are
then emitted the way a long-horizon agent receives them: mostly accurate
updates, plus the two things that make memory hard in practice:

* **Rumors**: low-authority (``hearsay``) claims of a false value, arriving
  *after* the true value. They never change the true state.
* **Late arrivals**: records about a period *before* anything else on file
  (e.g. a residence from last year), delivered sessions later. They change the
  past, not the present.

Ground truth comes from the simulated world, never from VERITAS's
reconciliation rules, so the benchmark is not circular. (The A/B Lab's
auto-generated questions are circular, which TECHNICAL.md §12.2 flags; this
generator removes that limitation.)

Question categories
-------------------
changed        current value of a slot that changed at least once
contested      current value of a slot that also received a rumor
stable         current value of a slot that never changed (control)
point-in-time  value of a slot on a given past date ("on March 3, 2025")
historical     the value immediately before the current one
multi-hop      the current HQ city of a person's current employer

Everything is deterministic for a given ``seed``.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from scenarios import ScriptedObservation

_FIRST = [
    "Asha", "Bruno", "Chen", "Divya", "Emeka", "Freya", "Gabriel", "Hana", "Ivan", "Jia", "Kofi", "Leila",
    "Mateo", "Nia", "Oskar", "Pooja", "Rafael", "Sana", "Tomas", "Uzma", "Vera", "Wanjiru", "Xavier", "Yara",
    "Zane", "Aiko", "Bilal", "Carmen", "Dmitri", "Esra",
]
_LAST = [
    "Rao", "Silva", "Okafor", "Novak", "Haddad", "Lindqvist", "Moreau", "Tanaka", "Kowalski", "Mensah",
    "Ferreira", "Iyer", "Brennan", "Castillo", "Volkov", "Nakamura", "Adeyemi", "Larsen", "Quispe", "Weber",
]
_CITIES = [
    "Lisbon", "Nairobi", "Osaka", "Bogota", "Krakow", "Toronto", "Accra", "Hanoi", "Porto", "Lagos",
    "Montreal", "Seville", "Tallinn", "Kyoto", "Lima", "Dakar", "Valencia", "Bergen", "Cusco", "Gdansk",
    "Hamburg", "Perth", "Quito", "Riga",
]
_ORG_STEMS = [
    "Northwind", "Contoso", "Fabrikam", "Tailspin", "Litware", "Proseware", "Adatum", "Woodgrove", "Lucerne",
    "Trey", "Wingtip", "Fourth Coffee", "Margie", "Coho", "Alpine Ski", "Blue Yonder", "Datum", "Humongous",
]
_ROLES = [
    "Analyst", "Engineer", "Senior Engineer", "Staff Engineer", "Team Lead", "Engineering Manager",
    "Product Manager", "Director", "Architect", "Researcher",
]
_PREFS = ["Tea", "Coffee", "Green Tea", "Espresso", "Cocoa", "Matcha", "Chai", "Lemonade"]

# (predicate, current, previous, point-in-time) question templates.
_TEMPLATES: dict[str, tuple[str, str, str]] = {
    "lives_in": ("Where does {s} live now?", "Where did {s} live before {cur}?", "Where did {s} live on {date}?"),
    "works_at": ("Where does {s} work now?", "Where did {s} work before {cur}?", "Where did {s} work on {date}?"),
    "has_role": ("What is {s}'s current role?", "What was {s}'s role before {cur}?", "What was {s}'s role on {date}?"),
    "prefers": ("What does {s} prefer now?", "What did {s} prefer before {cur}?", "What did {s} prefer on {date}?"),
    "headquartered_in": (
        "Where is {s} headquartered now?", "Where was {s} headquartered before {cur}?",
        "Where was {s} headquartered on {date}?",
    ),
}
_MULTI_HOP = "Which city is {s}'s employer headquartered in?"

CATEGORIES = ("changed", "contested", "stable", "point-in-time", "historical", "multi-hop")


@dataclass(frozen=True)
class TemporalQuestion:
    question: str
    answer: str
    wrong: tuple[str, ...]
    category: str
    subject: str
    path: tuple[str, ...]  # one predicate, or two for multi-hop
    kind: str  # current | as_of | previous
    as_of: Optional[datetime] = None
    before: Optional[str] = None


@dataclass
class Interval:
    value: str
    start: datetime
    end: Optional[datetime] = None


@dataclass
class Dataset:
    observations: list[ScriptedObservation]
    questions: list[TemporalQuestion]
    truth: dict[tuple[str, str], list[Interval]]
    rumors: dict[tuple[str, str], list[str]]
    config: dict = field(default_factory=dict)

    def stats(self) -> dict:
        by_source: dict[str, int] = {}
        for o in self.observations:
            by_source[o.source] = by_source.get(o.source, 0) + 1
        by_cat: dict[str, int] = {}
        for q in self.questions:
            by_cat[q.category] = by_cat.get(q.category, 0) + 1
        return {"observations": len(self.observations), "questions": len(self.questions),
                "slots": len(self.truth), "by_source": by_source, "by_category": by_cat}

    def to_json(self) -> dict:
        return {
            "config": self.config,
            "observations": [
                {"text": o.text, "timestamp": o.timestamp.isoformat(), "source": o.source,
                 "confidence": o.confidence, "session": o.phase}
                for o in self.observations
            ],
            "questions": [
                {**asdict(q), "as_of": q.as_of.isoformat() if q.as_of else None, "wrong": list(q.wrong),
                 "path": list(q.path)}
                for q in self.questions
            ],
        }


def human_date(ts: datetime) -> str:
    return f"{ts:%B} {ts.day}, {ts.year}"


def _names(n: int, rng: random.Random) -> list[str]:
    pool = [f"{f} {l}" for f in _FIRST for l in _LAST]
    rng.shuffle(pool)
    if n > len(pool):  # larger worlds: add double-barrelled surnames ("Asha Rao-Silva")
        extra = [f"{f} {a}-{b}" for f in _FIRST for a in _LAST for b in _LAST if a != b]
        rng.shuffle(extra)
        pool += extra
    if n > len(pool):
        raise ValueError(f"at most {len(pool)} people are supported")
    return pool[:n]


def _orgs(n: int) -> list[str]:
    out = list(_ORG_STEMS[:n])
    suffixes = ["Labs", "Systems", "Group", "Works", "Partners"]
    i = 0
    while len(out) < n:
        out.append(f"{_ORG_STEMS[i % len(_ORG_STEMS)]} {suffixes[(i // len(_ORG_STEMS)) % len(suffixes)]}")
        i += 1
    return out


class _World:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.truth: dict[tuple[str, str], list[Interval]] = {}
        self.rumors: dict[tuple[str, str], list[str]] = {}
        self.pre_history: set[tuple[str, str]] = set()

    def current(self, slot: tuple[str, str]) -> str:
        return self.truth[slot][-1].value

    def used(self, slot: tuple[str, str]) -> set[str]:
        return {iv.value for iv in self.truth.get(slot, [])} | set(self.rumors.get(slot, []))

    def set(self, slot: tuple[str, str], value: str, at: datetime) -> None:
        history = self.truth.setdefault(slot, [])
        if history:
            history[-1].end = at
        history.append(Interval(value, at))

    def fresh(self, slot: tuple[str, str], pool: list[str]) -> Optional[str]:
        options = [v for v in pool if v not in self.used(slot)]
        return self.rng.choice(options) if options else None


def _text(predicate: str, subject: str, value: str, old: Optional[str], rng: random.Random) -> str:
    if predicate == "lives_in":
        if old and rng.random() < 0.4:
            return f"{subject} moved from {old} to {value}"
        return rng.choice([f"{subject} moved to {value}", f"{subject} lives in {value}"])
    if predicate == "works_at":
        if old and rng.random() < 0.4:
            return f"{subject} left {old} for {value}"
        return rng.choice([f"{subject} joined {value}", f"{subject} works at {value}"])
    if predicate == "headquartered_in":
        if old:
            return f"{subject} moved its headquarters to {value}"
        return f"{subject} is headquartered in {value}"
    if predicate == "prefers":
        return f"{subject} prefers {value}"
    return f"{subject} | {predicate} | {value}"


def generate(
    *,
    people: int = 40,
    orgs: Optional[int] = None,
    sessions: int = 6,
    session_days: int = 21,
    drift: float = 0.35,
    rumor_rate: float = 0.15,
    late_rate: float = 0.15,
    max_questions: int = 60,
    seed: int = 7,
    start: datetime = datetime(2025, 1, 6, 9, 0, tzinfo=timezone.utc),
) -> Dataset:
    """Generate a deterministic multi-session trajectory and its question set."""
    if sessions < 2:
        raise ValueError("sessions must be >= 2 so that facts can change")
    rng = random.Random(seed)
    world = _World(rng)
    person_names = _names(people, rng)
    org_names = _orgs(orgs or max(4, people // 5))
    pools = {"lives_in": _CITIES, "works_at": org_names, "has_role": _ROLES, "prefers": _PREFS,
             "headquartered_in": _CITIES}
    person_slots = ("lives_in", "works_at", "has_role", "prefers")
    slots = [(p, pred) for p in person_names for pred in person_slots] + [(o, "headquartered_in") for o in org_names]

    sessions_obs: list[list[ScriptedObservation]] = [[] for _ in range(sessions)]

    def emit(k: int, ts: datetime, text: str, source: str, conf: float) -> None:
        sessions_obs[k].append(ScriptedObservation(f"S{k}", ts, text, conf, source))

    # Session 0: the initial state of every slot.
    for i, slot in enumerate(slots):
        subject, pred = slot
        ts = start + timedelta(minutes=i)
        value = rng.choice(pools[pred])
        world.set(slot, value, ts)
        source, conf = ("system_of_record", 0.99) if pred == "headquartered_in" else ("user_statement", 0.9)
        emit(0, ts, _text(pred, subject, value, None, rng), source, conf)

    # Late arrivals: pre-history for some residences / employers, delivered in a later session.
    late: list[tuple[int, ScriptedObservation]] = []
    for subject in person_names:
        for pred in ("lives_in", "works_at"):
            if rng.random() >= late_rate:
                continue
            slot = (subject, pred)
            value = world.fresh(slot, pools[pred])
            if value is None:
                continue
            first = world.truth[slot][0]
            pre_start = first.start - timedelta(days=rng.randint(90, 300))
            world.truth[slot].insert(0, Interval(value, pre_start, first.start))
            world.pre_history.add(slot)
            arrival = rng.randint(1, sessions - 1)
            late.append((arrival, ScriptedObservation(
                f"S{arrival}", pre_start, _text(pred, subject, value, None, rng), 0.9, "system_of_record")))

    # Sessions 1..n-1: drift and rumors.
    for k in range(1, sessions):
        base = start + timedelta(days=k * session_days)
        for i, slot in enumerate(slots):
            subject, pred = slot
            slot_drift = drift * (0.5 if pred == "headquartered_in" else 1.0)
            changed_at: Optional[datetime] = None
            if rng.random() < slot_drift:
                value = world.fresh(slot, pools[pred])
                if value is not None:
                    changed_at = base + timedelta(hours=rng.randint(0, 48), minutes=i % 60)
                    old = world.current(slot)
                    world.set(slot, value, changed_at)
                    source, conf = rng.choice([("user_statement", 0.9), ("sensor", 0.95), ("agent_observation", 0.88)])
                    if pred == "headquartered_in":
                        source, conf = "system_of_record", 0.99
                    emit(k, changed_at, _text(pred, subject, value, old, rng), source, conf)
            if pred != "headquartered_in" and rng.random() < rumor_rate:
                false_value = world.fresh(slot, pools[pred])
                if false_value is not None:
                    after = changed_at or base
                    ts = after + timedelta(hours=rng.randint(60, 200))
                    world.rumors.setdefault(slot, []).append(false_value)
                    text = (f"{subject} lives in {false_value}" if pred == "lives_in"
                            else _text(pred, subject, false_value, None, rng))
                    emit(k, ts, text, "hearsay", round(rng.uniform(0.55, 0.7), 2))

    observations: list[ScriptedObservation] = []
    for k in range(sessions):
        batch = sorted(sessions_obs[k], key=lambda o: o.timestamp)
        batch += [o for arrival, o in late if arrival == k]
        observations.extend(batch)

    end_of_world = max(o.timestamp for o in observations)
    questions = _questions(world, slots, person_names, end_of_world, max_questions, rng)
    config = {"people": people, "orgs": len(org_names), "sessions": sessions, "session_days": session_days,
              "drift": drift, "rumor_rate": rumor_rate, "late_rate": late_rate, "seed": seed,
              "max_questions": max_questions}
    return Dataset(observations, questions, world.truth, world.rumors, config)


def _questions(
    world: _World,
    slots: list[tuple[str, str]],
    people: list[str],
    now: datetime,
    max_questions: int,
    rng: random.Random,
) -> list[TemporalQuestion]:
    pools: dict[str, list[TemporalQuestion]] = {c: [] for c in CATEGORIES}

    for slot in slots:
        subject, pred = slot
        history = world.truth[slot]
        rumors = tuple(world.rumors.get(slot, []))
        values = [iv.value for iv in history]
        cur = history[-1].value
        now_t, before_t, at_t = _TEMPLATES[pred]
        past = tuple(dict.fromkeys(v for v in values if v != cur))

        if rumors:
            pools["contested"].append(TemporalQuestion(
                now_t.format(s=subject), cur, tuple(dict.fromkeys(rumors + past)), "contested",
                subject, (pred,), "current"))
        elif len(history) > 1:
            pools["changed"].append(TemporalQuestion(
                now_t.format(s=subject), cur, past, "changed", subject, (pred,), "current"))
        elif len(history) == 1:
            pools["stable"].append(TemporalQuestion(
                now_t.format(s=subject), cur, (), "stable", subject, (pred,), "current"))

        if len(history) > 1:
            prev = history[-2].value
            pools["historical"].append(TemporalQuestion(
                before_t.format(s=subject, cur=cur), prev,
                tuple(dict.fromkeys((cur,) + tuple(v for v in values if v not in (cur, prev)) + rumors)),
                "historical", subject, (pred,), "previous", before=cur))
            closed = [iv for iv in history[:-1] if iv.end and (iv.end - iv.start).days >= 4]
            if closed:
                iv = rng.choice(closed)
                span_days = (iv.end - iv.start).days  # type: ignore[operator]
                day = (iv.start + timedelta(days=rng.randint(1, span_days - 2))).replace(hour=12, minute=0)
                pools["point-in-time"].append(TemporalQuestion(
                    at_t.format(s=subject, date=human_date(day)), iv.value,
                    tuple(dict.fromkeys(v for v in values + list(rumors) if v != iv.value)),
                    "point-in-time", subject, (pred,), "as_of", as_of=day))

    for person in people:
        employer = world.current((person, "works_at"))
        hq_history = world.truth[(employer, "headquartered_in")]
        answer = hq_history[-1].value
        wrong = [iv.value for iv in hq_history[:-1]]
        for iv in world.truth[(person, "works_at")][:-1]:
            wrong.append(world.current((iv.value, "headquartered_in")))
        wrong = [w for w in dict.fromkeys(wrong) if w != answer]
        if wrong:
            pools["multi-hop"].append(TemporalQuestion(
                _MULTI_HOP.format(s=person), answer, tuple(wrong), "multi-hop", person,
                ("works_at", "headquartered_in"), "current"))

    for pool in pools.values():
        rng.shuffle(pool)
    picked: list[TemporalQuestion] = []
    while len(picked) < max_questions and any(pools.values()):
        for cat in CATEGORIES:
            if pools[cat] and len(picked) < max_questions:
                picked.append(pools[cat].pop())
    return picked
