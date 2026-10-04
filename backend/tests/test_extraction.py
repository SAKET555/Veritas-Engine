from __future__ import annotations

import pytest

from extraction import ExtractionError, extract_triples


@pytest.mark.parametrize(
    ("text", "expected", "prior"),
    [
        ("Alice moved from Mumbai to London", [("Alice", "lives_in", "London")], "Mumbai"),
        ("User relocated to New York", [("User", "lives_in", "New York")], None),
        ("Alice lives in London.", [("Alice", "lives_in", "London")], None),
        ("Alice is now living in Paris", [("Alice", "lives_in", "Paris")], None),
        ("Alice left Infosys for Acme", [("Alice", "works_at", "Acme")], "Infosys"),
        ("Alice works at Acme", [("Alice", "works_at", "Acme")], None),
        ("Alice is currently working at Initech", [("Alice", "works_at", "Initech")], None),
        ("Alice joined Globex", [("Alice", "works_at", "Globex")], None),
        (
            "Carol is a senior data scientist at OpenMind",
            [("Carol", "has_role", "Senior Data Scientist"), ("Carol", "works_at", "OpenMind")],
            None,
        ),
        ("Alice reports to Bob", [("Alice", "reports_to", "Bob")], None),
        ("Acme is headquartered in Berlin", [("Acme", "headquartered_in", "Berlin")], None),
        ("Acme moved its headquarters to Munich", [("Acme", "headquartered_in", "Munich")], None),
        ("Server-7 is located in Rack 4", [("Server-7", "located_in", "Rack 4")], None),
        ("Alice prefers tea", [("Alice", "prefers", "Tea")], None),
        ("Alice owns a Tesla", [("Alice", "owns", "Tesla")], None),
        ("Alice knows Bob", [("Alice", "knows", "Bob")], None),
        ("Alice's favorite color is blue", [("Alice", "favorite color", "Blue")], None),
        ("Alice | lives_in | Pune", [("Alice", "lives_in", "Pune")], None),
    ],
)
def test_rules(text: str, expected: list[tuple[str, str, str]], prior: str | None) -> None:
    triples = extract_triples(text)
    assert [(t.subject, t.predicate, t.object) for t in triples] == expected
    assert triples[0].expected_prior_object == prior
    assert triples[0].raw_text


def test_multiple_sentences_share_provenance() -> None:
    triples = extract_triples("Alice lives in London. Alice works at Acme.", provenance_id="p-1", confidence=0.7)
    assert [t.predicate for t in triples] == ["lives_in", "works_at"]
    assert {t.provenance_id for t in triples} == {"p-1"}
    assert {t.confidence for t in triples} == {0.7}


def test_unparseable_text_raises_with_hint() -> None:
    with pytest.raises(ExtractionError, match="Could not parse"):
        extract_triples("the weather is nice")


def test_one_bad_sentence_rejects_whole_statement() -> None:
    with pytest.raises(ExtractionError):
        extract_triples("Alice lives in London. blah blah")
