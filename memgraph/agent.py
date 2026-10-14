"""MemoryAgent: the assistant loop. tell() stores, ask() recalls."""

from __future__ import annotations

from .extract import extract_triples
from .graph import GraphStore, Triple
from .llm import LLM
from .recall import answer_question, retrieve


class MemoryAgent:
    def __init__(self, store: GraphStore, llm: LLM):
        self.store = store
        self.llm = llm

    def tell(self, text: str) -> list[Triple]:
        """Learn from a statement: extract triples, store them and the memory."""
        triples = extract_triples(text, self.llm)
        for t in triples:
            self.store.add_triple(t)
        entities = []
        for t in triples:
            for name in (t.subject, t.object):
                if name not in entities:
                    entities.append(name)
        self.store.add_memory(text, entities)
        return triples

    def ask(self, question: str) -> str:
        """Answer from the knowledge graph."""
        context = retrieve(question, self.llm, self.store)
        return answer_question(question, context, self.llm)

    def forget(self, name: str) -> bool:
        """Erase an entity and everything attached to it."""
        return self.store.forget(name)

    def stats(self) -> dict:
        return self.store.stats()
