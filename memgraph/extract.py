"""Turn raw user text into (subject, relation, object) triples via the LLM."""

from __future__ import annotations

import json

from .graph import Triple, normalize_name, normalize_relation
from .llm import LLM, LLMError, strip_code_fences

SYSTEM = (
    "You extract factual triples from text for a knowledge graph. Reply with JSON only."
)

PROMPT = """Extract every factual triple from the text below as a JSON array.
Each triple: {{"subject": "...", "subject_kind": "person|place|org|thing|event|concept",
"relation": "short_snake_case verb phrase", "object": "...", "object_kind": "person|place|org|thing|event|concept"}}

Rules:
- Use the real names from the text; do not invent facts.
- Keep at most 12 triples; skip opinions and greetings.
- If the text states no facts, return [].

Text: {text}
"""

MAX_TRIPLES = 20


def extract_triples(text: str, llm: LLM) -> list[Triple]:
    """Extract triples from text. Returns [] when the text holds no facts."""
    if not text.strip():
        return []
    raw = llm.generate(PROMPT.format(text=text), system=SYSTEM, json_mode=True)
    try:
        data = json.loads(strip_code_fences(raw))
    except json.JSONDecodeError as e:
        raise LLMError(f"Could not parse extraction JSON: {e}") from e

    items = data if isinstance(data, list) else data.get("triples", [])
    if not isinstance(items, list):
        raise LLMError("Extraction JSON was not a list")

    triples: list[Triple] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        s, r, o = item.get("subject"), item.get("relation"), item.get("object")
        if not (isinstance(s, str) and isinstance(r, str) and isinstance(o, str)):
            continue
        if not (s.strip() and r.strip() and o.strip()):
            continue
        triples.append(
            Triple(
                subject=normalize_name(s),
                relation=normalize_relation(r),
                object=normalize_name(o),
                subject_kind=str(item.get("subject_kind") or "thing"),
                object_kind=str(item.get("object_kind") or "thing"),
            )
        )
    return triples[:MAX_TRIPLES]
