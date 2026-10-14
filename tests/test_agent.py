from memgraph.agent import MemoryAgent
from memgraph.graph import InMemoryGraphStore
from memgraph.llm import FakeLLM

EXTRACT = (
    '[{"subject": "Velu", "subject_kind": "person", "relation": "works_on", '
    '"object": "MemGraph", "object_kind": "thing"}]'
)
PLAN = '{"lookups": [{"subject": "Velu", "relation": null, "object": null}], "entities": ["Velu"]}'


def test_tell_stores_triples_and_memory():
    agent = MemoryAgent(InMemoryGraphStore(), FakeLLM([EXTRACT]))
    triples = agent.tell("Velu works on MemGraph.")
    assert len(triples) == 1
    assert agent.stats() == {"triples": 1, "memories": 1, "entities": 2}


def test_ask_uses_graph_context():
    llm = FakeLLM([EXTRACT, PLAN, "Velu works on MemGraph."])
    agent = MemoryAgent(InMemoryGraphStore(), llm)
    agent.tell("Velu works on MemGraph.")
    answer = agent.ask("What does Velu work on?")
    assert "MemGraph" in answer
    # the answer prompt must have seen the fact
    assert "Velu works on MemGraph" in llm.calls[2]["prompt"]


def test_ask_empty_graph_says_it_does_not_know():
    llm = FakeLLM(['{"lookups": [], "entities": []}', "I do not remember that yet."])
    agent = MemoryAgent(InMemoryGraphStore(), llm)
    assert "not remember" in agent.ask("Who is Ada?")


def test_forget():
    agent = MemoryAgent(InMemoryGraphStore(), FakeLLM([EXTRACT]))
    agent.tell("Velu works on MemGraph.")
    assert agent.forget("Velu") is True
    assert agent.stats()["triples"] == 0
