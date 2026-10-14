"""Command line interface: remember, ask, forget, stats, cypher."""

from __future__ import annotations

import argparse
import os
import re
import sys

from .agent import MemoryAgent
from .graph import InMemoryGraphStore, store_from_env
from .llm import LLMError, llm_from_env


def build_agent(use_memory: bool) -> MemoryAgent:
    llm = llm_from_env()
    store = InMemoryGraphStore() if use_memory else store_from_env()
    return MemoryAgent(store, llm)


def cmd_remember(args: argparse.Namespace) -> int:
    agent = build_agent(args.memory)
    triples = agent.tell(args.text)
    if not triples:
        print("Nothing factual to store.")
        return 0
    for t in triples:
        print(f"stored: {t.subject} --{t.relation}--> {t.object}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    agent = build_agent(args.memory)
    print(agent.ask(args.question))
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    agent = build_agent(args.memory)
    print("forgotten." if agent.forget(args.name) else "nothing known by that name.")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    agent = build_agent(args.memory)
    s = agent.stats()
    print(
        f"entities: {s['entities']}, triples: {s['triples']}, memories: {s['memories']}"
    )
    return 0


_WRITE_KEYWORDS = re.compile(r"\b(create|merge|delete|detach|set|remove|drop|load)\b")


def cmd_cypher(args: argparse.Namespace) -> int:
    from .graph import FalkorDBStore

    lowered = args.query.strip().lower()
    if not lowered.startswith(("match", "call")) or _WRITE_KEYWORDS.search(lowered):
        print(
            "Only read queries are allowed here (no CREATE/MERGE/DELETE/SET).",
            file=sys.stderr,
        )
        return 2
    store = FalkorDBStore(
        host=os.environ.get("FALKORDB_HOST", "localhost"),
        port=int(os.environ.get("FALKORDB_PORT", "6379")),
        password=os.environ.get("FALKORDB_PASSWORD"),
        graph_name=os.environ.get("FALKORDB_GRAPH", "memgraph"),
    )
    for row in store.cypher(args.query):
        print(row)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="memgraph",
        description="An AI assistant that remembers via a FalkorDB knowledge graph.",
    )
    parser.add_argument(
        "--memory",
        action="store_true",
        help="use the in-memory store instead of FalkorDB (demos, no DB needed)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("remember", help="learn a fact or statement")
    p.add_argument("text", help="the statement to remember")
    p.set_defaults(func=cmd_remember)

    p = sub.add_parser("ask", help="ask a question from memory")
    p.add_argument("question", help="the question")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("forget", help="erase an entity and its facts")
    p.add_argument("name", help="entity name to forget")
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("stats", help="show graph stats")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("cypher", help="run a read-only Cypher query (FalkorDB)")
    p.add_argument("query", help='e.g. "MATCH (e:Entity) RETURN e.name LIMIT 5"')
    p.set_defaults(func=cmd_cypher)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except LLMError as e:
        print(f"LLM error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 - friendly CLI surface
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
