"""Scripted demo of MemGraph (offline-friendly).

Uses the in-memory store and a scripted FakeLLM so the demo runs anywhere.
Set GEMINI_API_KEY and run with --live to use a real model instead.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memgraph.agent import MemoryAgent
from memgraph.graph import InMemoryGraphStore, store_from_env
from memgraph.llm import FakeLLM, llm_from_env

SCRIPT = [
    (
        "tell",
        "Velu is an AI Automation Engineer who lives in Santa Rosa, Philippines.",
        (
            '[{"subject": "Velu", "subject_kind": "person", "relation": "is", '
            '"object": "AI Automation Engineer", "object_kind": "concept"}, '
            '{"subject": "Velu", "subject_kind": "person", "relation": "lives_in", '
            '"object": "Santa Rosa, Philippines", "object_kind": "place"}]'
        ),
    ),
    (
        "tell",
        "Velu is building MemGraph for the Graph Hacks hackathon.",
        (
            '[{"subject": "Velu", "subject_kind": "person", "relation": "building", '
            '"object": "MemGraph", "object_kind": "thing"}, '
            '{"subject": "MemGraph", "subject_kind": "thing", "relation": "built_for", '
            '"object": "Graph Hacks", "object_kind": "event"}]'
        ),
    ),
    (
        "ask",
        "Where does Velu live?",
        (
            '{"lookups": [{"subject": "Velu", "relation": "lives_in", "object": null}], '
            '"entities": ["Velu"]}'
        ),
        "Velu lives in Santa Rosa, Philippines.",
    ),
    (
        "ask",
        "What is Velu building?",
        (
            '{"lookups": [{"subject": "Velu", "relation": "building", "object": null}], '
            '"entities": ["Velu", "MemGraph"]}'
        ),
        "Velu is building MemGraph for the Graph Hacks hackathon.",
    ),
]


def main() -> int:
    live = "--live" in sys.argv
    if live:
        llm = llm_from_env()
        store = store_from_env()
        scripted = False
    else:
        responses = [step[2] for step in SCRIPT if step[0] == "tell"]
        responses += [
            r for step in SCRIPT if step[0] == "ask" for r in (step[2], step[3])
        ]
        llm = FakeLLM(responses)
        store = InMemoryGraphStore()
        scripted = True

    agent = MemoryAgent(store, llm)
    print("=== MemGraph demo ===")
    if scripted:
        print("(scripted run: in-memory graph, canned model replies)\n")

    for step in SCRIPT:
        kind = step[0]
        if kind == "tell":
            print(f"you: {step[1]}")
            triples = agent.tell(step[1])
            for t in triples:
                print(
                    f"  graph <- ({t.subject})-[:REL {{type: {t.relation}}}]->({t.object})"
                )
        else:
            print(f"you: {step[1]}")
            print(f"memgraph: {agent.ask(step[1])}")
        print()

    s = agent.stats()
    print(
        f"graph now holds {s['entities']} entities, {s['triples']} triples, {s['memories']} memories."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
