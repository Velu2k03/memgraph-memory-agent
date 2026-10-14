import pytest

from memgraph.graph import (
    InMemoryGraphStore,
    Triple,
    normalize_name,
    normalize_relation,
)


@pytest.fixture
def store():
    return InMemoryGraphStore()


def test_add_and_search_triple(store):
    store.add_triple(Triple("Velu", "works_on", "MemGraph", "person", "thing"))
    found = store.search_triples(subject="Velu")
    assert len(found) == 1
    assert found[0].relation == "works_on"
    assert found[0].object == "MemGraph"


def test_search_filters(store):
    store.add_triple(Triple("Velu", "works_on", "MemGraph"))
    store.add_triple(Triple("Velu", "lives_in", "Santa Rosa"))
    assert len(store.search_triples(subject="Velu", relation="lives_in")) == 1
    assert len(store.search_triples(object="MemGraph")) == 1
    assert store.search_triples(subject="Nobody") == []


def test_relation_normalization(store):
    store.add_triple(Triple("A", "Works At", "B"))
    assert store.search_triples(relation="works_at")[0].subject == "A"


def test_name_normalization(store):
    store.add_triple(Triple("  Velu   Murugan ", "knows", "Python"))
    assert store.search_triples(subject="Velu Murugan")


def test_no_duplicate_triples(store):
    t = Triple("A", "likes", "B")
    store.add_triple(t)
    store.add_triple(t)
    assert store.stats()["triples"] == 1


def test_memories_about(store):
    store.add_memory("Velu likes Python", ["Velu", "Python"])
    store.add_memory("Unrelated", ["Nobody"])
    assert store.memories_about("velu") == ["Velu likes Python"]


def test_forget_removes_everything(store):
    store.add_triple(Triple("Velu", "likes", "Python"))
    store.add_memory("Velu likes Python", ["Velu"])
    assert store.forget("Velu") is True
    assert store.search_triples(subject="Velu") == []
    assert store.memories_about("Velu") == []
    assert store.stats()["entities"] == 1  # Python remains


def test_forget_unknown_returns_false(store):
    assert store.forget("Nobody") is False


def test_entities(store):
    store.add_triple(Triple("Velu", "likes", "Python", "person", "thing"))
    names = {e["name"]: e["kind"] for e in store.entities()}
    assert names == {"Velu": "person", "Python": "thing"}


def test_normalize_helpers():
    assert normalize_relation("Works At!!") == "works_at"
    assert normalize_relation("!!!") == "related_to"
    assert normalize_name("  a   b ") == "a b"
