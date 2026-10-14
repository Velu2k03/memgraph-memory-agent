import pytest

from memgraph.extract import extract_triples
from memgraph.llm import FakeLLM, LLMError

PAYLOAD = (
    '[{"subject": "Velu", "subject_kind": "person", "relation": "works_on", '
    '"object": "MemGraph", "object_kind": "thing"}, '
    '{"subject": "Velu", "relation": "lives in", "object": "Santa Rosa"}]'
)


def test_extract_parses_json():
    triples = extract_triples("Velu works on MemGraph.", FakeLLM([PAYLOAD]))
    assert len(triples) == 2
    assert triples[0].subject == "Velu"
    assert triples[0].relation == "works_on"
    assert triples[1].relation == "lives_in"  # normalized


def test_extract_strips_fences():
    triples = extract_triples("x", FakeLLM(["```json\n" + PAYLOAD + "\n```"]))
    assert len(triples) == 2


def test_extract_dict_wrapper():
    triples = extract_triples("x", FakeLLM(['{"triples": ' + PAYLOAD + "}"]))
    assert len(triples) == 2


def test_extract_empty_text_skips_llm():
    llm = FakeLLM(["[]"])
    assert extract_triples("   ", llm) == []
    assert llm.calls == []


def test_extract_empty_list():
    assert extract_triples("hello there", FakeLLM(["[]"])) == []


def test_extract_skips_bad_items():
    bad = '[{"subject": "A"}, {"subject": "", "relation": "x", "object": "y"}, "nope"]'
    assert extract_triples("x", FakeLLM([bad])) == []


def test_extract_bad_json_raises():
    with pytest.raises(LLMError):
        extract_triples("x", FakeLLM(["not json"]))
