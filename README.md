# MemGraph: an AI assistant that actually remembers

Most chatbots are goldfish: every session starts from zero. MemGraph fixes that
with a **knowledge graph**. Everything you tell it becomes `(subject, relation,
object)` triples in **FalkorDB**, linked to the memories they came from. Ask a
question later and it plans graph lookups, gathers the facts, and answers from
what it genuinely stored. No vector-DB black box, no hallucinated "memory".

Built for **Graph Hacks** (FalkorDB is the heart of the project, and every line
was written during the hackathon).

```
you:       Velu is an AI Automation Engineer living in Santa Rosa.
memgraph:  stored: Velu --is--> AI Automation Engineer
           stored: Velu --lives_in--> Santa Rosa

you:       Where does Velu live?
memgraph:  Velu lives in Santa Rosa.
```

## How it works

```
tell("...") --> LLM extracts triples --> FalkorDB MERGEs nodes/edges
                                              |
ask("...")  --> LLM plans lookups --> graph search --> LLM answers from facts
```

- **Graph schema** (FalkorDB / Cypher):
  `(:Entity {name, kind})`, `(:Memory {text, created})`,
  `(a:Entity)-[r:REL {type}]->(b:Entity)`, `(m:Memory)-[:MENTIONS]->(e:Entity)`
- **Safe recall**: the model plans *structured lookups*, never raw Cypher, so
  it cannot inject writes. A `cypher` CLI command exists for read-only
  exploration (`MATCH`/`CALL` only).
- **Providers**: Google Gemini (free tier) or any OpenAI-compatible API, plain
  REST via stdlib, no heavy SDKs.

## Quickstart

```bash
docker compose up -d                 # FalkorDB on localhost:6379
pip install -r requirements.txt
export GEMINI_API_KEY=your_key       # free tier is fine
# or: export OPENAI_API_KEY=... OPENAI_BASE_URL=... OPENAI_MODEL=...

python -m memgraph.cli remember "Velu is an AI Automation Engineer."
python -m memgraph.cli ask "What does Velu do?"
python -m memgraph.cli stats
python -m memgraph.cli forget "Velu"
python -m memgraph.cli cypher "MATCH (e:Entity) RETURN e.name LIMIT 10"
```

No database handy? Add `--memory` to any command (or run `python demo.py`)
to use the in-memory store instead.

## Configuration

| Env var            | Default     | Purpose                          |
|--------------------|-------------|----------------------------------|
| `GEMINI_API_KEY`   | -           | Gemini provider (free tier)      |
| `OPENAI_API_KEY`   | -           | OpenAI-compatible provider       |
| `OPENAI_BASE_URL`  | api.openai  | Override endpoint                |
| `OPENAI_MODEL`     | gpt-4o-mini | Override model                   |
| `FALKORDB_HOST`    | localhost   | FalkorDB host                    |
| `FALKORDB_PORT`    | 6379        | FalkorDB port                    |
| `FALKORDB_PASSWORD`| -           | FalkorDB password                |
| `FALKORDB_GRAPH`   | memgraph    | Graph name                       |

## Project layout

```
memgraph/
  llm.py      LLM providers (Gemini, OpenAI-compatible, FakeLLM for tests)
  graph.py    GraphStore interface, FalkorDBStore (Cypher), InMemoryGraphStore
  extract.py  text -> triples via LLM
  recall.py   question -> retrieval plan -> facts -> answer
  agent.py    MemoryAgent: tell / ask / forget / stats
  cli.py      command line interface
tests/        pytest suite (FakeLLM + in-memory store, no services needed)
demo.py       scripted end-to-end demo (add --live for a real model + FalkorDB)
```

## Tests and lint

```bash
pytest -q
ruff check memgraph tests demo.py
```

## Why FalkorDB

Memory is relationships: who works on what, who lives where, what was built
for which event. A graph stores that natively, answers multi-hop questions
("what is Velu building for Graph Hacks?") with one traversal, and stays
readable: you can open the graph and *see* what the assistant remembers.
FalkorDB's Cypher and Redis-protocol speed make it a natural fit for a
memory layer that has to answer in real time.
