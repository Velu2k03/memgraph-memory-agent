"""Graph storage.

FalkorDBStore is the real backend. Schema::

    (:Entity {name, kind})
    (:Memory {text, created})
    (a:Entity)-[r:REL {type}]->(b:Entity)
    (m:Memory)-[:MENTIONS]->(e:Entity)

InMemoryGraphStore implements the same interface for tests and for
running the demo without a database.
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    subject_kind: str = "thing"
    object_kind: str = "thing"


def normalize_relation(rel: str) -> str:
    rel = re.sub(r"[^a-z0-9]+", "_", rel.strip().lower()).strip("_")
    return rel or "related_to"


def normalize_name(name: str) -> str:
    return " ".join(name.strip().split())


class GraphStore(ABC):
    @abstractmethod
    def add_triple(self, triple: Triple) -> None: ...

    @abstractmethod
    def add_memory(self, text: str, entities: list[str]) -> None: ...

    @abstractmethod
    def search_triples(
        self,
        subject: str | None = None,
        relation: str | None = None,
        object: str | None = None,
    ) -> list[Triple]: ...

    @abstractmethod
    def memories_about(self, entity: str) -> list[str]: ...

    @abstractmethod
    def entities(self) -> list[dict]: ...

    @abstractmethod
    def forget(self, name: str) -> bool: ...

    @abstractmethod
    def stats(self) -> dict: ...

    @abstractmethod
    def clear(self) -> None: ...


class InMemoryGraphStore(GraphStore):
    def __init__(self) -> None:
        self._triples: list[Triple] = []
        self._memories: list[dict] = []
        self._kinds: dict[str, str] = {}

    def add_triple(self, triple: Triple) -> None:
        t = Triple(
            subject=normalize_name(triple.subject),
            relation=normalize_relation(triple.relation),
            object=normalize_name(triple.object),
            subject_kind=triple.subject_kind or "thing",
            object_kind=triple.object_kind or "thing",
        )
        if t not in self._triples:
            self._triples.append(t)
        self._kinds.setdefault(t.subject, t.subject_kind)
        self._kinds.setdefault(t.object, t.object_kind)

    def add_memory(self, text: str, entities: list[str]) -> None:
        self._memories.append(
            {
                "text": text,
                "entities": [normalize_name(e) for e in entities],
                "created": int(time.time()),
            }
        )

    def search_triples(self, subject=None, relation=None, object=None) -> list[Triple]:
        s = subject.strip().lower() if subject else None
        r = normalize_relation(relation) if relation else None
        o = object.strip().lower() if object else None
        return [
            t
            for t in self._triples
            if (s is None or t.subject.lower() == s)
            and (r is None or t.relation == r)
            and (o is None or t.object.lower() == o)
        ]

    def memories_about(self, entity: str) -> list[str]:
        e = entity.strip().lower()
        return [
            m["text"]
            for m in self._memories
            if any(x.lower() == e for x in m["entities"])
        ]

    def entities(self) -> list[dict]:
        return [{"name": n, "kind": k} for n, k in sorted(self._kinds.items())]

    def forget(self, name: str) -> bool:
        n = normalize_name(name)
        had_triples = any(t.subject == n or t.object == n for t in self._triples)
        had_memories = any(n in m["entities"] for m in self._memories)
        had_entity = n in self._kinds
        self._triples = [t for t in self._triples if t.subject != n and t.object != n]
        self._memories = [
            m for m in self._memories if all(e != n for e in m["entities"])
        ]
        self._kinds.pop(n, None)
        return had_triples or had_memories or had_entity

    def stats(self) -> dict:
        return {
            "triples": len(self._triples),
            "memories": len(self._memories),
            "entities": len(self._kinds),
        }

    def clear(self) -> None:
        self._triples.clear()
        self._memories.clear()
        self._kinds.clear()


class FalkorDBStore(GraphStore):
    """FalkorDB-backed store. Connects lazily-safe: pass host/port/password."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        graph_name: str = "memgraph",
    ):
        from falkordb import FalkorDB

        self._db = FalkorDB(host=host, port=port, password=password)
        self._graph = self._db.select_graph(graph_name)
        self._graph_name = graph_name

    # -- low-level escape hatch (read-only use is up to the caller) --
    def cypher(self, query: str, params: dict | None = None) -> list[list]:
        return self._graph.query(query, params or {}).result_set

    # -- GraphStore interface --
    def add_triple(self, triple: Triple) -> None:
        s = normalize_name(triple.subject)
        r = normalize_relation(triple.relation)
        o = normalize_name(triple.object)
        self.cypher(
            "MERGE (a:Entity {name: $s}) ON CREATE SET a.kind = $sk "
            "MERGE (b:Entity {name: $o}) ON CREATE SET b.kind = $ok "
            "MERGE (a)-[rel:REL {type: $r}]->(b)",
            {
                "s": s,
                "sk": triple.subject_kind or "thing",
                "o": o,
                "ok": triple.object_kind or "thing",
                "r": r,
            },
        )

    def add_memory(self, text: str, entities: list[str]) -> None:
        names = [normalize_name(e) for e in entities if e.strip()]
        if not names:
            names = []
        self.cypher(
            "CREATE (m:Memory {text: $text, created: $ts}) "
            "WITH m UNWIND $names AS n "
            "MERGE (e:Entity {name: n}) MERGE (m)-[:MENTIONS]->(e)",
            {"text": text, "ts": int(time.time()), "names": names},
        )

    def search_triples(self, subject=None, relation=None, object=None) -> list[Triple]:
        rows = self.cypher(
            "MATCH (a:Entity)-[r:REL]->(b:Entity) "
            "WHERE ($s IS NULL OR a.name = $s) "
            "AND ($r IS NULL OR r.type = $r) "
            "AND ($o IS NULL OR b.name = $o) "
            "RETURN a.name, r.type, b.name, a.kind, b.kind",
            {
                "s": normalize_name(subject) if subject else None,
                "r": normalize_relation(relation) if relation else None,
                "o": normalize_name(object) if object else None,
            },
        )
        return [
            Triple(s, rel, o, sk or "thing", ok or "thing")
            for s, rel, o, sk, ok in rows
        ]

    def memories_about(self, entity: str) -> list[str]:
        rows = self.cypher(
            "MATCH (m:Memory)-[:MENTIONS]->(e:Entity {name: $n}) "
            "RETURN m.text ORDER BY m.created DESC",
            {"n": normalize_name(entity)},
        )
        return [row[0] for row in rows]

    def entities(self) -> list[dict]:
        rows = self.cypher("MATCH (e:Entity) RETURN e.name, e.kind ORDER BY e.name")
        return [{"name": n, "kind": k or "thing"} for n, k in rows]

    def forget(self, name: str) -> bool:
        n = normalize_name(name)
        rows = self.cypher(
            "MATCH (e:Entity {name: $n}) DETACH DELETE e RETURN count(e)", {"n": n}
        )
        deleted = rows[0][0] if rows else 0
        # drop memories left orphaned by the delete
        self.cypher("MATCH (m:Memory) WHERE NOT (m)-[:MENTIONS]->() DELETE m")
        return deleted > 0

    def stats(self) -> dict:
        rows = self.cypher(
            "MATCH (e:Entity) WITH count(e) AS ec "
            "MATCH ()-[r:REL]->() WITH ec, count(r) AS rc "
            "MATCH (m:Memory) RETURN ec, rc, count(m)"
        )
        ec, rc, mc = rows[0] if rows else (0, 0, 0)
        return {"entities": ec, "triples": rc, "memories": mc}

    def clear(self) -> None:
        self._graph.delete()


def store_from_env() -> GraphStore:
    """FalkorDBStore configured from FALKORDB_HOST/PORT/PASSWORD/GRAPH env vars."""
    return FalkorDBStore(
        host=os.environ.get("FALKORDB_HOST", "localhost"),
        port=int(os.environ.get("FALKORDB_PORT", "6379")),
        password=os.environ.get("FALKORDB_PASSWORD"),
        graph_name=os.environ.get("FALKORDB_GRAPH", "memgraph"),
    )
