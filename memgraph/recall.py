"""Answer questions from the graph.

The LLM plans retrieval as structured lookups (never raw Cypher from the
model, so a bad model cannot inject writes), the store executes them, and
the LLM synthesizes the final answer from the gathered facts.
"""

from __future__ import annotations

import json

from .graph import GraphStore, Triple
from .llm import LLM, LLMError, strip_code_fences

PLAN_SYSTEM = "You plan knowledge-graph lookups. Reply with JSON only."

PLAN_PROMPT = """Given the user question, output a JSON object:
{{"lookups": [{{"subject": "name or null", "relation": "snake_case or null", "object": "name or null"}}],
  "entities": ["entity names mentioned in the question"]}}

Rules:
- Each lookup matches triples by any combination of subject/relation/object; null means "any".
- Prefer specific lookups first (e.g. subject "Velu" for "Where does Velu work?").
- entities lists the proper names in the question so we can pull their memories.
- At most 6 lookups.

Question: {question}
"""

ANSWER_SYSTEM = "You answer questions using only the provided facts. Reply concisely."

ANSWER_PROMPT = """Answer the question using ONLY the facts below.
If the facts do not contain the answer, say you do not remember that yet.
Never invent names, dates, or details.

Facts:
{facts}

Question: {question}
"""


def plan_retrieval(question: str, llm: LLM) -> dict:
    raw = llm.generate(
        PLAN_PROMPT.format(question=question), system=PLAN_SYSTEM, json_mode=True
    )
    try:
        plan = json.loads(strip_code_fences(raw))
    except json.JSONDecodeError as e:
        raise LLMError(f"Could not parse retrieval plan JSON: {e}") from e
    if not isinstance(plan, dict):
        raise LLMError("Retrieval plan was not a JSON object")
    lookups = plan.get("lookups", [])
    entities = plan.get("entities", [])
    if not isinstance(lookups, list):
        lookups = []
    if not isinstance(entities, list):
        entities = []
    return {"lookups": lookups[:6], "entities": [str(e) for e in entities][:8]}


def retrieve(question: str, llm: LLM, store: GraphStore) -> dict:
    """Gather triples and memories relevant to the question."""
    plan = plan_retrieval(question, llm)
    triples: list[Triple] = []
    seen = set()
    for lookup in plan["lookups"]:
        if not isinstance(lookup, dict):
            continue
        for t in store.search_triples(
            subject=lookup.get("subject"),
            relation=lookup.get("relation"),
            object=lookup.get("object"),
        ):
            key = (t.subject, t.relation, t.object)
            if key not in seen:
                seen.add(key)
                triples.append(t)
    memories: list[str] = []
    for entity in plan["entities"]:
        for m in store.memories_about(entity):
            if m not in memories:
                memories.append(m)
    return {"triples": triples, "memories": memories[:8]}


def format_facts(context: dict) -> str:
    lines = []
    for t in context["triples"]:
        lines.append(f"- {t.subject} {t.relation.replace('_', ' ')} {t.object}")
    for m in context["memories"]:
        lines.append(f"- (memory) {m}")
    return "\n".join(lines) if lines else "(no facts found)"


def answer_question(question: str, context: dict, llm: LLM) -> str:
    facts = format_facts(context)
    return llm.generate(
        ANSWER_PROMPT.format(facts=facts, question=question),
        system=ANSWER_SYSTEM,
    ).strip()
